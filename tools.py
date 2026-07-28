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
from contextlib import contextmanager
from typing import Any, List, Optional, Sequence

import psycopg2
from psycopg2 import sql
from langchain_core.messages import ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import BaseTool, tool

import schema as schema_module
from config import Config
from custom_logging import log, log_panel
from schema import DatabaseSchema, Table, quote_identifier
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


# --- Database connection --------------------------------------------------


@contextmanager
def with_sql_cursor(readonly: bool = True):
    """Establishes a read-only database connection and provides a cursor."""
    conn = None
    try:
        conn = psycopg2.connect(
            dbname=Config.Postgres.dbname,
            user=Config.Postgres.user,
            password=Config.Postgres.password,
            host=Config.Postgres.host,
            port=Config.Postgres.port,
            connect_timeout=10,
        )
        # The database refuses writes regardless of what the agent sends.
        conn.set_session(readonly=readonly, autocommit=False)
        cur = conn.cursor()
        try:
            yield cur
        finally:
            cur.close()
    finally:
        if conn:
            conn.close()


def get_schema(refresh: bool = False) -> DatabaseSchema:
    """The live schema, read once and cached for the process."""
    with with_sql_cursor() as cursor:
        return schema_module.load_schema(cursor, refresh=refresh)


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

    parts = [f"Database '{Config.Postgres.dbname}' contains {len(db.tables)} table(s).", ""]
    parts.append(db.overview())

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
        query = sql.SQL("SELECT * FROM {table} LIMIT %s;").format(
            table=sql.SQL(table.qualified)
        )
        with with_sql_cursor() as cursor:
            cursor.execute(query, (limit,))
            columns = [d[0] for d in cursor.description]
            rows = cursor.fetchall()
        return format_rows(columns, rows)
    except Exception as e:  # noqa: BLE001
        log(f"[red]Error sampling table: {e}[/red]")
        return f"Error sampling table: {e}"


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
    try:
        db = get_schema()
        repair = repair_sql(sql_query, db)
        if repair.changed:
            log(f"[yellow]Rewrote query: {'; '.join(repair.corrections)}[/yellow]")
            prefix = repair.note() + "\n\n"
        sql_query = repair.sql
    except Exception as e:  # noqa: BLE001 - repair is best-effort
        log(f"[yellow]Could not repair query ({e}); running it unchanged.[/yellow]")
        db = None

    try:
        with with_sql_cursor() as cursor:
            cursor.execute(sql_query)
            columns = [d[0] for d in cursor.description] if cursor.description else []
            rows = cursor.fetchmany(Config.MAX_SQL_RESULT_ROWS + 1)
        clipped = len(rows) > Config.MAX_SQL_RESULT_ROWS
        rows = rows[: Config.MAX_SQL_RESULT_ROWS]
        result = format_rows(columns, rows)
        if clipped:
            result += (
                f"\n[Showing the first {Config.MAX_SQL_RESULT_ROWS} rows. "
                f"Add LIMIT, filters, or an aggregate to narrow the result.]"
            )
        return prefix + result
    except Exception as e:  # noqa: BLE001
        log(f"[red]Error running query: {e}[/red]")
        message = f"Error running query: {e}"
        if db is not None:
            hint = suggest_for_error(str(e), db)
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
        with with_sql_cursor() as cursor:
            for table in targets:
                candidates = table.name_columns()[:6]
                if not candidates:
                    continue
                for column in candidates:
                    if len(hits) >= limit:
                        break
                    try:
                        query = sql.SQL(
                            "SELECT DISTINCT {col} FROM {table} "
                            "WHERE {col} ILIKE %s AND {col} IS NOT NULL LIMIT 3;"
                        ).format(
                            col=sql.Identifier(column.name),
                            table=sql.SQL(table.qualified),
                        )
                        cursor.execute(query, (f"%{search_text}%",))
                        for (value,) in cursor.fetchall():
                            hits.append(f"{table.qualified}.{column.name} = {value!r}")
                    except Exception:  # noqa: BLE001
                        # A column may not be comparable with ILIKE; skip it and
                        # roll back so the transaction stays usable.
                        cursor.execute("ROLLBACK;")
                        continue
                if len(hits) >= limit:
                    break

        if not hits:
            return (
                f"'{search_text}' was not found in any indexed text column. "
                f"Try a shorter fragment, or use inspect_database to find the "
                f"right table and query it directly."
            )
        return f"Found '{search_text}' in:\n" + "\n".join(hits[:limit])
    except Exception as e:  # noqa: BLE001
        log(f"[red]Error searching for value: {e}[/red]")
        return f"Error searching for value: {e}"


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
        query = sql.SQL("SELECT DISTINCT {column} FROM {table} LIMIT %s;").format(
            column=sql.Identifier(column.name), table=sql.SQL(table.qualified)
        )
        with with_sql_cursor() as cursor:
            cursor.execute(query, (limit + 1,))
            values = [row[0] for row in cursor.fetchall()]

        clipped = len(values) > limit
        values = values[:limit]
        rendered = ", ".join("NULL" if v is None else str(v) for v in values)
        if clipped:
            rendered += f" ... [more than {limit} distinct values]"
        prefix = (
            f"(column resolved to '{column.name}')\n" if column.name.lower() != column_name.lower() else ""
        )
        return prefix + (rendered or "No values found.")
    except Exception as e:  # noqa: BLE001
        log(f"[red]Error getting distinct values: {e}[/red]")
        return f"Error getting distinct column values: {e}"


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
        with with_sql_cursor() as cursor:
            cursor.execute(f"EXPLAIN {clean}")
            rows = cursor.fetchall()
        return prefix + "\n".join(str(row[0]) for row in rows)
    except Exception as e:  # noqa: BLE001
        log(f"[red]Error explaining query: {e}[/red]")
        return f"Error explaining query: {e}"
