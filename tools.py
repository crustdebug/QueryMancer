"""Read-only database tools exposed to the agent.

Three concerns run through this module:

  * Safety - every query runs in a Postgres read-only session, identifiers are
    quoted through psycopg2's `sql` builder, and access is restricted to the
    tables named in Config.ALLOWED_TABLES.
  * Token economy - schema lookups are cached, and every tool result is capped
    before it re-enters the prompt. Tool output dominates an agent's token bill,
    so this is where most of the savings are.
  * Legibility - results are returned as compact tables the model can read
    without burning tokens on repeated column names.
"""

import re
import threading
from contextlib import contextmanager
from typing import Any, Iterable, List, Optional, Sequence, Set

import psycopg2
from psycopg2 import sql
from langchain_core.messages import ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import BaseTool, tool

from config import Config
from custom_logging import log, log_panel

# --- Tool Management ---


def get_available_tools() -> List[BaseTool]:
    """Returns a list of all available tool instances."""
    return [
        list_tables,
        sample_table,
        describe_table,
        execute_sql,
        fuzzy_full_text_search,
        get_foreign_key_relationships,
        get_distinct_column_values,
        explain_query,
        search_entity_by_name,
    ]


def call_tool(tool_call: ToolCall) -> ToolMessage:
    """Finds and invokes the correct tool based on the agent's request."""
    tools_by_name = {t.name: t for t in get_available_tools()}
    name = tool_call["name"]
    if name not in tools_by_name:
        return ToolMessage(
            content=f"Error: Tool '{name}' not found.",
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


# --- Access control -------------------------------------------------------

# Matches identifiers following FROM / JOIN / UPDATE / INTO, optionally
# schema-qualified and optionally double-quoted.
_TABLE_REF = re.compile(
    r"""(?:\bfrom\b|\bjoin\b|\binto\b|\bupdate\b)\s+
        (?:"(?P<quoted>[^"]+)"|(?P<plain>[A-Za-z_][A-Za-z0-9_$]*))
        (?:\s*\.\s*(?:"(?P<quoted2>[^"]+)"|(?P<plain2>[A-Za-z_][A-Za-z0-9_$]*)))?
    """,
    re.IGNORECASE | re.VERBOSE,
)


def allowed_tables() -> Optional[Set[str]]:
    """The permitted table names, or None when no restriction is configured."""
    if not Config.ALLOWED_TABLES:
        return None
    return {name.lower() for name in Config.ALLOWED_TABLES}


def check_table_allowed(table_name: str) -> Optional[str]:
    """Return an error message if `table_name` is off limits, else None."""
    allowed = allowed_tables()
    if allowed is None or table_name.lower() in allowed:
        return None
    return (
        f"Error: Access to table '{table_name}' is not permitted. "
        f"Use list_tables to see the tables you may query."
    )


def referenced_tables(sql_query: str) -> Set[str]:
    """Best-effort extraction of table names referenced by a query."""
    found: Set[str] = set()
    for match in _TABLE_REF.finditer(sql_query):
        # A schema-qualified name puts the table in the second group.
        table = match.group("quoted2") or match.group("plain2")
        if not table:
            table = match.group("quoted") or match.group("plain")
        if table:
            found.add(table.lower())
    # Common table expressions are not real tables; drop names defined by WITH.
    for cte in re.finditer(r"\bwith\s+([A-Za-z_][A-Za-z0-9_$]*)\s+as\s*\(", sql_query, re.I):
        found.discard(cte.group(1).lower())
    for cte in re.finditer(r",\s*([A-Za-z_][A-Za-z0-9_$]*)\s+as\s*\(", sql_query, re.I):
        found.discard(cte.group(1).lower())
    return found


def check_query_allowed(sql_query: str) -> Optional[str]:
    """Reject a query that touches tables outside the allowlist."""
    allowed = allowed_tables()
    if allowed is None:
        return None
    forbidden = sorted(referenced_tables(sql_query) - allowed)
    if forbidden:
        return (
            f"Error: This query references table(s) you may not access: "
            f"{', '.join(forbidden)}. Use list_tables to see permitted tables."
        )
    return None


# A statement must start with SELECT or WITH; anything else is rejected before
# it reaches the database. The read-only session is the real guarantee, but
# failing early gives the model a clearer error to correct against.
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


# --- SECURE DATABASE CONNECTION ---


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
        # Enforce a read-only session. The DB will reject INSERT, UPDATE, etc.
        conn.set_session(readonly=readonly, autocommit=False)
        cur = conn.cursor()
        try:
            yield cur
        finally:
            cur.close()
    finally:
        if conn:
            conn.close()


# --- Schema cache ---------------------------------------------------------
#
# The schema does not change during a conversation, so repeated list_tables and
# describe_table calls are served from memory. This removes whole LLM round
# trips worth of latency and, more importantly, avoids re-billing the same
# schema text on every question.

_cache: dict = {}
_cache_lock = threading.Lock()


def cached(key: str, producer):
    with _cache_lock:
        if key in _cache:
            return _cache[key]
    value = producer()
    with _cache_lock:
        _cache[key] = value
    return value


def clear_schema_cache() -> None:
    with _cache_lock:
        _cache.clear()


# --- SECURE TOOLS ---


@tool(parse_docstring=True)
def list_tables(reasoning: str) -> str:
    """Lists the tables you are permitted to query.

    Args:
        reasoning (str): A detailed explanation of why you need to see all tables,
            relating the need directly to the user's query.

    Returns:
        str: A comma-separated list of table names.
    """
    log_panel(title="List Tables Tool", content=f"Reasoning: {reasoning}")

    def load() -> str:
        allowed = allowed_tables()
        with with_sql_cursor() as cursor:
            if allowed:
                # Compare lowercased, so a quoted PascalCase table such as
                # "Employee" still matches an allowlist entry of any casing.
                cursor.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'public' AND lower(table_name) IN %s "
                    "ORDER BY table_name;",
                    (tuple(sorted(allowed)),),
                )
            else:
                cursor.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'public' ORDER BY table_name;"
                )
            tables = [row[0] for row in cursor.fetchall()]
        if not tables:
            return "No accessible tables found."
        return ", ".join(tables)

    try:
        return cached("list_tables", load)
    except Exception as e:  # noqa: BLE001
        log(f"[red]Error listing tables: {e}[/red]")
        return f"Error listing tables: {e}"


@tool(parse_docstring=True)
def describe_table(reasoning: str, table_name: str) -> str:
    """Returns the schema of a table: its columns, types and nullability.

    Args:
        reasoning (str): A detailed explanation of why you need this table's structure.
        table_name (str): The exact name of the table to describe.

    Returns:
        str: One line per column.
    """
    log_panel(title="Describe Table Tool", content=f"Table: {table_name}\nReasoning: {reasoning}")
    denied = check_table_allowed(table_name)
    if denied:
        return denied

    def load() -> str:
        with with_sql_cursor() as cursor:
            cursor.execute(
                """
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = %s
                ORDER BY ordinal_position;
                """,
                (table_name,),
            )
            rows = cursor.fetchall()
        if not rows:
            return f"Table '{table_name}' was not found in the 'public' schema."
        lines = [
            f"{name} {dtype}{'' if nullable == 'YES' else ' NOT NULL'}"
            for name, dtype, nullable in rows
        ]
        return f"{table_name}({len(rows)} columns)\n" + "\n".join(lines)

    try:
        return cached(f"describe:{table_name.lower()}", load)
    except Exception as e:  # noqa: BLE001
        log(f"[red]Error describing table: {e}[/red]")
        return f"Error describing table: {e}"


@tool(parse_docstring=True)
def sample_table(reasoning: str, table_name: str, row_sample_size: int = 3) -> str:
    """Retrieves a few sample rows to clarify what a table's columns contain.

    Args:
        reasoning (str): A detailed explanation of why you need sample data.
        table_name (str): The exact name of the table to sample.
        row_sample_size (int): Number of rows to retrieve (default 3, max 10).

    Returns:
        str: A compact table of sample rows.
    """
    log_panel(
        title="Sample Table Tool",
        content=f"Table: {table_name}\nRows: {row_sample_size}\nReasoning: {reasoning}",
    )
    denied = check_table_allowed(table_name)
    if denied:
        return denied

    limit = max(1, min(int(row_sample_size), 10))
    try:
        query = sql.SQL("SELECT * FROM {table} LIMIT %s;").format(
            table=sql.Identifier(table_name)
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

    Data modification statements are rejected.

    Args:
        reasoning (str): Why this query answers the user's request.
        sql_query (str): The complete SELECT query to execute.

    Returns:
        str: The query results as a compact table.
    """
    log_panel(title="Execute SQL Tool", content=f"Query: {sql_query}\nReasoning: {reasoning}")

    for check in (check_read_only(sql_query), check_query_allowed(sql_query)):
        if check:
            log(f"[red]{check}[/red]")
            return check

    try:
        with with_sql_cursor() as cursor:
            cursor.execute(sql_query)
            columns = [d[0] for d in cursor.description] if cursor.description else []
            # Fetch one extra row to detect truncation without pulling the
            # entire result set into memory.
            rows = cursor.fetchmany(Config.MAX_SQL_RESULT_ROWS + 1)
        clipped = len(rows) > Config.MAX_SQL_RESULT_ROWS
        rows = rows[: Config.MAX_SQL_RESULT_ROWS]
        result = format_rows(columns, rows)
        if clipped:
            result += (
                f"\n[Showing the first {Config.MAX_SQL_RESULT_ROWS} rows. "
                f"Add LIMIT, filters, or an aggregate to narrow the result.]"
            )
        return result
    except Exception as e:  # noqa: BLE001
        log(f"[red]Error running query: {e}[/red]")
        return f"Error running query: {e}. Check your SQL syntax and column names."


@tool(parse_docstring=True)
def fuzzy_full_text_search(
    reasoning: str,
    table_name: str,
    column_names: List[str],
    search_phrase: str,
    similarity_threshold: float = 0.2,
    row_limit: int = 5,
) -> str:
    """Searches text columns for approximate matches (misspellings, variations).

    Args:
        reasoning (str): Why a fuzzy search is needed.
        table_name (str): The table to search.
        column_names (List[str]): Columns to search within.
        search_phrase (str): The phrase to search for approximately.
        similarity_threshold (float): Similarity threshold from 0 to 1 (default 0.2).
        row_limit (int): Maximum rows to return (default 5).

    Returns:
        str: Rows that closely match the search phrase.
    """
    log_panel(
        title="Fuzzy Full-Text Search Tool",
        content=(
            f"Table: {table_name}\nColumns: {column_names}\n"
            f"Phrase: {search_phrase}\nReasoning: {reasoning}"
        ),
    )
    denied = check_table_allowed(table_name)
    if denied:
        return denied
    if not column_names:
        return "Error: You must provide at least one column name to search."

    limit = max(1, min(int(row_limit), 25))
    try:
        safe_cols = [sql.Identifier(col) for col in column_names]
        conditions = sql.SQL(" OR ").join(
            sql.SQL("similarity(COALESCE(CAST({col} AS text), ''), %s) > %s").format(col=col)
            for col in safe_cols
        )
        ordering = (
            sql.SQL("GREATEST(")
            + sql.SQL(", ").join(
                sql.SQL("similarity(COALESCE(CAST({col} AS text), ''), %s)").format(col=col)
                for col in safe_cols
            )
            + sql.SQL(")")
        )
        query = sql.SQL(
            "SELECT * FROM {table} WHERE {conditions} ORDER BY {ordering} DESC LIMIT %s;"
        ).format(table=sql.Identifier(table_name), conditions=conditions, ordering=ordering)

        params: List[Any] = []
        for _ in column_names:
            params.extend([search_phrase, similarity_threshold])
        params.extend([search_phrase] * len(column_names))
        params.append(limit)

        with with_sql_cursor() as cursor:
            cursor.execute(query, params)
            columns = [d[0] for d in cursor.description]
            rows = cursor.fetchall()
        if not rows:
            return "No close matches found. Try a lower similarity_threshold."
        return format_rows(columns, rows)
    except Exception as e:  # noqa: BLE001
        log(f"[red]Error running fuzzy search: {e}[/red]")
        if "function similarity(" in str(e):
            return (
                "Error: Fuzzy search requires the 'pg_trgm' extension. "
                "Run: CREATE EXTENSION IF NOT EXISTS pg_trgm;"
            )
        return f"Error running fuzzy search: {e}"


@tool(parse_docstring=True)
def get_foreign_key_relationships(reasoning: str) -> str:
    """Returns the foreign key relationships, showing how tables join.

    Args:
        reasoning (str): Why you need the join relationships.

    Returns:
        str: One relationship per line, as 'table.column -> table.column'.
    """
    log_panel(title="Get Foreign Key Relationships Tool", content=f"Reasoning: {reasoning}")

    def load() -> str:
        query = """
            SELECT
                tc.table_name, kcu.column_name,
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
              ON tc.constraint_name = kcu.constraint_name
             AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage AS ccu
              ON ccu.constraint_name = tc.constraint_name
             AND ccu.table_schema = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_schema = 'public';
        """
        with with_sql_cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()

        allowed = allowed_tables()
        if allowed is not None:
            rows = [r for r in rows if r[0].lower() in allowed and r[2].lower() in allowed]
        if not rows:
            return (
                "No foreign key relationships found. Join keys must be inferred "
                "from column names; use describe_table to compare candidates."
            )
        return "\n".join(f"{r[0]}.{r[1]} -> {r[2]}.{r[3]}" for r in rows)

    try:
        return cached("foreign_keys", load)
    except Exception as e:  # noqa: BLE001
        log(f"[red]Error getting foreign key relationships: {e}[/red]")
        return f"Error getting foreign key relationships: {e}"


@tool(parse_docstring=True)
def get_distinct_column_values(
    reasoning: str, table_name: str, column_name: str, value_limit: int = 25
) -> str:
    """Fetches distinct values for a column, to learn the real category labels.

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
    denied = check_table_allowed(table_name)
    if denied:
        return denied

    limit = max(1, min(int(value_limit), 100))
    try:
        query = sql.SQL("SELECT DISTINCT {column} FROM {table} LIMIT %s;").format(
            column=sql.Identifier(column_name), table=sql.Identifier(table_name)
        )
        with with_sql_cursor() as cursor:
            cursor.execute(query, (limit + 1,))
            values = [row[0] for row in cursor.fetchall()]
        clipped = len(values) > limit
        values = values[:limit]
        rendered = ", ".join("NULL" if v is None else str(v) for v in values)
        if clipped:
            rendered += f" ... [more than {limit} distinct values]"
        return rendered or "No values found."
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

    for check in (check_read_only(clean), check_query_allowed(clean)):
        if check:
            return check

    try:
        with with_sql_cursor() as cursor:
            cursor.execute(f"EXPLAIN {clean}")
            rows = cursor.fetchall()
        return "\n".join(str(row[0]) for row in rows)
    except Exception as e:  # noqa: BLE001
        log(f"[red]Error explaining query: {e}[/red]")
        return f"Error explaining query: {e}"


@tool(parse_docstring=True)
def search_entity_by_name(
    reasoning: str, entity_name: str, similarity_threshold: float = 0.3
) -> str:
    """Searches for a company or person across both customers and vendors.

    Use this first when the user names an entity without saying whether they are
    a customer or a vendor.

    Args:
        reasoning (str): Why you need to search for this entity.
        entity_name (str): The company or individual to search for.
        similarity_threshold (float): Similarity threshold from 0 to 1 (default 0.3).

    Returns:
        str: Matches found in the customer and vendor tables.
    """
    log_panel(
        title="Search Entity By Name Tool",
        content=f"Name: {entity_name}\nReasoning: {reasoning}",
    )

    query = """
    (SELECT 'Customer' AS entity_type, customer_name AS name,
            similarity(customer_name, %s) AS score
     FROM customer_profile
     WHERE similarity(customer_name, %s) > %s)
    UNION ALL
    (SELECT 'Vendor' AS entity_type, vendor_name AS name,
            similarity(vendor_name, %s) AS score
     FROM vendor_profile
     WHERE similarity(vendor_name, %s) > %s)
    ORDER BY score DESC
    LIMIT 20;
    """
    params = [
        entity_name,
        entity_name,
        similarity_threshold,
        entity_name,
        entity_name,
        similarity_threshold,
    ]

    try:
        with with_sql_cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
        if not rows:
            return (
                f"No entity matching '{entity_name}' was found as a customer or vendor. "
                f"Try a shorter form of the name or a lower similarity_threshold."
            )
        return "\n".join(
            f"{entity_type}: {name} (score {score:.2f})" for entity_type, name, score in rows
        )
    except Exception as e:  # noqa: BLE001
        log(f"[red]Error searching for entity: {e}[/red]")
        if "function similarity(" in str(e):
            return (
                "Error: This search requires the 'pg_trgm' extension. "
                "Run: CREATE EXTENSION IF NOT EXISTS pg_trgm;"
            )
        return f"Error searching for entity: {e}"
