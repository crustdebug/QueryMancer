"""Read-only database tools exposed to the agent.

The tools work against any PostgreSQL database without configuration. Nothing
about the schema is hardcoded: table and column names are discovered at runtime
(see schema.py), and names the model guesses wrong are mapped onto the real ones
(see sql_repair.py) instead of failing.

Access is read-only but otherwise unrestricted - every table, column and row in
every non-system schema is readable. Writes are refused by the Postgres session
itself, not merely by a check here.
"""

import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field as dc_field
from typing import Any, List, Optional, Sequence

from langchain_core.messages import ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import BaseTool, tool

import schema as schema_module
import session as session_module
from config import Config
from connection import sanitize
from custom_logging import log, log_panel
from schema import DatabaseSchema, quote_identifier
from sql_repair import repair_sql, suggest_for_error

# --- Tool Management ---


def get_available_tools() -> List[BaseTool]:
    """Returns a list of all available tool instances."""
    return [
        inspect_database,
        describe_table,
        sample_table,
        execute_sql,
        find_value,
        get_distinct_column_values,
        explain_query,
    ]


def call_tool(tool_call: ToolCall) -> ToolMessage:
    """Finds and invokes the correct tool based on the agent's request."""
    tools_by_name = {t.name: t for t in get_available_tools()}
    name = tool_call["name"]
    if name not in tools_by_name:
        return ToolMessage(
            content=f"Error: Tool '{name}' not found. Available: {', '.join(tools_by_name)}",
            tool_call_id=tool_call["id"],
        )
    try:
        response = tools_by_name[name].invoke(tool_call["args"])
    except Exception as error:  # noqa: BLE001 - surfaced to the model as text
        log(f"[red]Tool '{name}' raised: {error}[/red]")
        response = f"Error running tool '{name}': {error}"
    return ToolMessage(content=truncate(str(response)), tool_call_id=tool_call["id"])


# --- Output shaping -------------------------------------------------------


def truncate(text: str, limit: Optional[int] = None) -> str:
    """Cap a tool result so one large table cannot flood the context window."""
    limit = limit or Config.MAX_TOOL_RESULT_CHARS
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text) - limit} more characters]"


def format_rows(columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Render rows as a compact pipe-separated table with a header."""
    if not rows:
        return "No rows returned."
    header = " | ".join(columns)
    body = "\n".join(" | ".join("" if v is None else str(v) for v in row) for row in rows)
    return f"{header}\n{body}\n({len(rows)} row(s))"


# --- Read-only enforcement ------------------------------------------------

_READ_ONLY_START = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)
_WRITE_STATEMENT = re.compile(
    r"^\s*(insert|update|delete|drop|create|alter|truncate|grant|revoke|copy|call|do)\b",
    re.IGNORECASE | re.MULTILINE,
)


def strip_sql_literals(sql_query: str) -> str:
    """Remove string literals and comments so keyword checks ignore data."""
    without_comments = re.sub(r"--[^\n]*", " ", sql_query)
    without_comments = re.sub(r"/\*.*?\*/", " ", without_comments, flags=re.DOTALL)
    return re.sub(r"'(?:[^']|'')*'", "''", without_comments)


def check_read_only(sql_query: str) -> Optional[str]:
    """Return an error if the statement is not a single read-only query."""
    stripped = strip_sql_literals(sql_query).strip().rstrip(";")
    if not stripped:
        return "Error: Empty query."
    if ";" in stripped:
        return "Error: Only a single statement may be executed at a time."
    if not _READ_ONLY_START.match(stripped):
        return "Error: Only read-only SELECT queries are allowed."
    if _WRITE_STATEMENT.search(stripped):
        return "Error: Data modification statements are not permitted."
    return None


# --- Query trace ----------------------------------------------------------
#
# The UI shows the SQL the agent ran and renders its rows as a table, so each
# successful query is recorded here for the caller to read afterwards. The
# trace is thread-local: two requests handled concurrently must not see each
# other's queries.

_trace_local = threading.local()


@dataclass
class ExecutedQuery:
    """One query the agent ran, with its result."""

    sql: str
    columns: List[str]
    rows: List[tuple]
    truncated: bool = False
    corrections: List[str] = dc_field(default_factory=list)


@contextmanager
def capture_queries():
    """Collect the queries run inside this block."""
    previous = getattr(_trace_local, "queries", None)
    _trace_local.queries = []
    try:
        yield _trace_local.queries
    finally:
        _trace_local.queries = previous


def _record_query(entry: ExecutedQuery) -> None:
    queries = getattr(_trace_local, "queries", None)
    if queries is not None:
        queries.append(entry)


# --- Question context -----------------------------------------------------
#
# inspect_database prunes the schema down to the tables relevant to what was
# actually asked. The question is not a tool argument - the model would have to
# restate it on every call, and could restate it wrongly - so it is bound to
# the thread for the duration of the request, the same way the session is.

_question_local = threading.local()


@contextmanager
def current_question(question: str):
    """Bind the question being answered, for schema pruning."""
    previous = getattr(_question_local, "question", None)
    _question_local.question = question
    try:
        yield
    finally:
        _question_local.question = previous


def get_current_question() -> str:
    return getattr(_question_local, "question", None) or ""


# --- Database connection --------------------------------------------------


def get_connection():
    """The database the current session is connected to."""
    return session_module.require_connection()


def _clean(error: Exception) -> str:
    """An error message with any credentials stripped out.

    Driver errors often echo the connection URL, so nothing raised by the
    database is shown to the model or written to a log without passing here.
    """
    connection = session_module.current_connection()
    settings = connection.settings if connection else None
    return sanitize(error, settings)


def get_schema(refresh: bool = False) -> DatabaseSchema:
    """The live schema of the connected database, read once and cached."""
    connection = get_connection()
    schema_module.set_dialect(connection.settings.engine)
    return schema_module.load_schema(connection, refresh=refresh)


def clear_schema_cache() -> None:
    schema_module.clear_cache()


def _resolve_table_or_error(name: str):
    """Resolve a table name, returning (table, error_message)."""
    db = get_schema()
    table = db.find_table(name)
    if table:
        return table, None

    resolved, suggestions = db.resolve_table(name)
    if resolved:
        log(f"[yellow]Resolved table '{name}' to {resolved.qualified}.[/yellow]")
        return resolved, None

    hint = f" Closest matches: {', '.join(suggestions)}." if suggestions else ""
    return None, (
        f"Error: No table named '{name}' exists.{hint} "
        f"Call inspect_database to see what is available."
    )


# --- Tools ----------------------------------------------------------------


@tool(parse_docstring=True)
def inspect_database(reasoning: str, refresh: bool = False) -> str:
    """Returns a map of the database: every table, its size and its columns.

    Call this first when you do not yet know the schema. It replaces separate
    list-tables and describe-table calls for orientation, and shows the real
    naming convention in use.

    Args:
        reasoning (str): Why you need to see the database structure.
        refresh (bool): Re-read the schema from the database, bypassing the cache.

    Returns:
        str: A table-by-table overview, followed by the foreign key relationships.
    """
    log_panel(title="Inspect Database Tool", content=f"Reasoning: {reasoning}")
    try:
        db = get_schema(refresh=refresh)
    except Exception as e:  # noqa: BLE001
        log(f"[red]Error reading schema: {e}[/red]")
        return f"Error reading the database schema: {e}"

    if not db.tables:
        return "The database contains no readable tables."

    label = get_connection().settings.database or "(current)"
    parts = [f"Database '{label}' contains {len(db.tables)} table(s).", ""]

    # On a large schema, send only the tables this question plausibly needs
    # plus whatever they are foreign-keyed to. Listing all 100+ tables costs
    # tokens on every call and gives the model more opportunities to join the
    # wrong one. Small schemas are sent whole - there is nothing to save.
    focused = None
    if len(db.tables) > Config.SCHEMA_PRUNE_THRESHOLD:
        focused = db.focused_overview(
            get_current_question(), max_tables=Config.MAX_FOCUSED_TABLES
        )

    if focused:
        parts.append(focused)
        return "\n".join(parts)

    parts.append(db.overview(max_tables=Config.MAX_OVERVIEW_TABLES))

    if db.foreign_keys:
        parts.append("")
        parts.append("Foreign key relationships (use these for JOINs):")
        parts.extend(f"  {fk.render()}" for fk in db.foreign_keys[:60])
        if len(db.foreign_keys) > 60:
            parts.append(f"  ... and {len(db.foreign_keys) - 60} more")
    else:
        parts.append("")
        parts.append(
            "No foreign keys are declared. Infer joins by matching column names "
            "such as customerId to the primary key of the related table."
        )
    return "\n".join(parts)


@tool(parse_docstring=True)
def describe_table(reasoning: str, table_name: str) -> str:
    """Returns the full column list and relationships for one table.

    The name is matched loosely, so 'customer' will find a table actually named
    'Customer' or 'customer_profile'.

    Args:
        reasoning (str): Why you need this table's structure.
        table_name (str): The table to describe.

    Returns:
        str: Columns with types, primary keys, and related tables.
    """
    log_panel(title="Describe Table Tool", content=f"Table: {table_name}\nReasoning: {reasoning}")
    try:
        table, error = _resolve_table_or_error(table_name)
        if error:
            return error
        return get_schema().describe(table)
    except Exception as e:  # noqa: BLE001
        log(f"[red]Error describing table: {e}[/red]")
        return f"Error describing table: {e}"


@tool(parse_docstring=True)
def sample_table(reasoning: str, table_name: str, row_sample_size: int = 3) -> str:
    """Retrieves a few rows so you can see what the data actually looks like.

    Args:
        reasoning (str): Why you need sample data.
        table_name (str): The table to sample.
        row_sample_size (int): Number of rows (default 3, max 10).

    Returns:
        str: A compact table of sample rows.
    """
    log_panel(
        title="Sample Table Tool",
        content=f"Table: {table_name}\nRows: {row_sample_size}\nReasoning: {reasoning}",
    )
    try:
        table, error = _resolve_table_or_error(table_name)
        if error:
            return error

        limit = max(1, min(int(row_sample_size), 10))
        # The table name comes from the schema we just read, not from the model,
        # so it is already a known-good identifier.
        columns, rows, _ = get_connection().run(
            f"SELECT * FROM {table.qualified} LIMIT {limit}"
        )
        return format_rows(columns, rows)
    except Exception as e:  # noqa: BLE001
        log(f"[red]Error sampling table: {e}[/red]")
        return f"Error sampling table: {_clean(e)}"


@tool(parse_docstring=True)
def execute_sql(reasoning: str, sql_query: str) -> str:
    """Executes a single read-only SELECT query and returns the result.

    Identifiers that do not match the schema are corrected automatically where
    possible, and the correction is reported back to you.

    Args:
        reasoning (str): Why this query answers the user's request.
        sql_query (str): The complete SELECT query to execute.

    Returns:
        str: The query results as a compact table.
    """
    log_panel(title="Execute SQL Tool", content=f"Query: {sql_query}\nReasoning: {reasoning}")

    blocked = check_read_only(sql_query)
    if blocked:
        log(f"[red]{blocked}[/red]")
        return blocked

    prefix = ""
    corrections: List[str] = []
    try:
        db = get_schema()
        repair = repair_sql(sql_query, db)
        if repair.changed:
            log(f"[yellow]Rewrote query: {'; '.join(repair.corrections)}[/yellow]")
            prefix = repair.note() + "\n\n"
            corrections = list(repair.corrections)
        sql_query = repair.sql
    except Exception as e:  # noqa: BLE001 - repair is best-effort
        log(f"[yellow]Could not repair query ({e}); running it unchanged.[/yellow]")
        db = None

    try:
        columns, rows, clipped = get_connection().run(
            sql_query, limit=Config.MAX_SQL_RESULT_ROWS
        )
        # Record the query as actually executed, so the UI can show the real
        # SQL and render the rows rather than re-parsing the text answer.
        _record_query(
            ExecutedQuery(
                sql=sql_query,
                columns=list(columns),
                rows=[tuple(row) for row in rows],
                truncated=clipped,
                corrections=corrections,
            )
        )
        result = format_rows(columns, rows)
        if clipped:
            result += (
                f"\n[Showing the first {Config.MAX_SQL_RESULT_ROWS} rows. "
                f"Add LIMIT, filters, or an aggregate to narrow the result.]"
            )
        return prefix + result
    except Exception as e:  # noqa: BLE001
        detail = _clean(e)
        log(f"[red]Error running query: {detail}[/red]")
        message = f"Error running query: {detail}"
        if db is not None:
            hint = suggest_for_error(detail, db)
            if hint:
                message += f"\n{hint}"
        return message


@tool(parse_docstring=True)
def find_value(reasoning: str, search_text: str, table_name: str = "", max_hits: int = 15) -> str:
    """Finds which table and column contains a given name or value.

    Use this when the user mentions something specific - a company, a person, a
    product code - and you do not know where it is stored. Searches the
    text columns most likely to hold labels across the database.

    Args:
        reasoning (str): Why you need to locate this value.
        search_text (str): The value to look for.
        table_name (str): Restrict the search to one table. Leave empty to search all.
        max_hits (int): Maximum matches to report (default 15).

    Returns:
        str: Where the value was found, with the matching values.
    """
    log_panel(
        title="Find Value Tool",
        content=f"Text: {search_text}\nTable: {table_name or 'all'}\nReasoning: {reasoning}",
    )
    if not search_text.strip():
        return "Error: search_text must not be empty."

    try:
        db = get_schema()
        if table_name:
            table, error = _resolve_table_or_error(table_name)
            if error:
                return error
            targets = [table]
        else:
            # Skip empty tables; scanning them costs time and finds nothing.
            targets = [t for t in db.tables if t.estimated_rows > 0] or db.tables

        hits: List[str] = []
        limit = max(1, min(int(max_hits), 50))
        connection = get_connection()
        needle = f"%{search_text.lower()}%"

        for table in targets:
            if len(hits) >= limit:
                break
            for column in table.name_columns()[:6]:
                if len(hits) >= limit:
                    break
                # LOWER(...) LIKE is case-insensitive on every engine; ILIKE is
                # PostgreSQL-only. The value is bound, never interpolated.
                query = (
                    f"SELECT DISTINCT {quote_identifier(column.name)} "
                    f"FROM {table.qualified} "
                    f"WHERE LOWER(CAST({quote_identifier(column.name)} AS CHAR(255))) "
                    f"LIKE :needle LIMIT 3"
                )
                if connection.settings.engine in ("postgresql", "oracle"):
                    query = query.replace("CHAR(255)", "TEXT")
                try:
                    _, rows, _ = connection.run(query, {"needle": needle})
                except Exception:  # noqa: BLE001
                    # Some column types cannot be cast or compared; skip them.
                    continue
                for (value,) in rows:
                    if value is not None:
                        hits.append(f"{table.qualified}.{column.name} = {value!r}")

        if not hits:
            return (
                f"'{search_text}' was not found in any indexed text column. "
                f"Try a shorter fragment, or use inspect_database to find the "
                f"right table and query it directly."
            )
        return f"Found '{search_text}' in:\n" + "\n".join(hits[:limit])
    except Exception as e:  # noqa: BLE001
        log(f"[red]Error searching for value: {_clean(e)}[/red]")
        return f"Error searching for value: {_clean(e)}"


@tool(parse_docstring=True)
def get_distinct_column_values(
    reasoning: str, table_name: str, column_name: str, value_limit: int = 25
) -> str:
    """Fetches distinct values for a column, to learn the real category labels.

    Call this before filtering on a categorical column, so you filter on values
    that actually exist.

    Args:
        reasoning (str): Why you need the distinct values.
        table_name (str): The table name.
        column_name (str): The column name.
        value_limit (int): Maximum distinct values to return (default 25).

    Returns:
        str: A comma-separated list of distinct values.
    """
    log_panel(
        title="Get Distinct Column Values Tool",
        content=f"Table: {table_name}\nColumn: {column_name}\nReasoning: {reasoning}",
    )
    try:
        table, error = _resolve_table_or_error(table_name)
        if error:
            return error

        db = get_schema()
        column = table.column(column_name)
        if column is None:
            resolved, suggestions = db.resolve_column(table, column_name)
            if resolved is None:
                hint = f" Available: {', '.join(suggestions)}." if suggestions else ""
                return f"Error: '{table.qualified}' has no column '{column_name}'.{hint}"
            log(f"[yellow]Resolved column '{column_name}' to '{resolved.name}'.[/yellow]")
            column = resolved

        limit = max(1, min(int(value_limit), 100))
        _, rows, clipped = get_connection().run(
            f"SELECT DISTINCT {quote_identifier(column.name)} FROM {table.qualified}",
            limit=limit,
        )
        values = [row[0] for row in rows]
        rendered = ", ".join("NULL" if v is None else str(v) for v in values)
        if clipped:
            rendered += f" ... [more than {limit} distinct values]"
        prefix = (
            f"(column resolved to '{column.name}')\n" if column.name.lower() != column_name.lower() else ""
        )
        return prefix + (rendered or "No values found.")
    except Exception as e:  # noqa: BLE001
        log(f"[red]Error getting distinct values: {_clean(e)}[/red]")
        return f"Error getting distinct column values: {_clean(e)}"


@tool(parse_docstring=True)
def explain_query(reasoning: str, sql_query: str) -> str:
    """Validates a query and returns its execution plan without running it.

    Args:
        reasoning (str): Why you need the query plan.
        sql_query (str): The SELECT query to explain, without the EXPLAIN keyword.

    Returns:
        str: The query plan, one step per line.
    """
    log_panel(title="Explain Query Tool", content=f"Query: {sql_query}\nReasoning: {reasoning}")

    clean = sql_query.strip()
    if clean.upper().startswith("EXPLAIN"):
        clean = clean[len("EXPLAIN") :].lstrip()

    blocked = check_read_only(clean)
    if blocked:
        return blocked

    try:
        db = get_schema()
        repair = repair_sql(clean, db)
        clean = repair.sql
        prefix = repair.note() + "\n\n" if repair.changed else ""
    except Exception:  # noqa: BLE001
        prefix = ""

    try:
        _, rows, _ = get_connection().run(f"EXPLAIN {clean}")
        return prefix + "\n".join(" | ".join(str(v) for v in row) for row in rows)
    except Exception as e:  # noqa: BLE001
        log(f"[red]Error explaining query: {_clean(e)}[/red]")
        return f"Error explaining query: {_clean(e)}"
