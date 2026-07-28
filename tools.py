import psycopg2
from psycopg2 import sql
from contextlib import contextmanager
from typing import Any, List

from langchain.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import BaseTool

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
        search_entity_by_name
    ]

def call_tool(tool_call: ToolCall) -> Any:
    """Finds and invokes the correct tool based on the agent's request."""
    tools_by_name = {tool.name: tool for tool in get_available_tools()}
    if tool_call["name"] in tools_by_name:
        tool_to_call = tools_by_name[tool_call["name"]]
        response = tool_to_call.invoke(tool_call["args"])
        return ToolMessage(content=str(response), tool_call_id=tool_call["id"])
    else:
        return ToolMessage(
            content=f"Error: Tool '{tool_call['name']}' not found.",
            tool_call_id=tool_call["id"],
        )

# --- SECURE DATABASE CONNECTION ---

@contextmanager
def with_sql_cursor():
    """Establishes a read-only database connection and provides a cursor."""
    conn = None
    try:
        conn = psycopg2.connect(
            dbname=Config.Postgres.dbname,
            user=Config.Postgres.user,
            password=Config.Postgres.password,
            host=Config.Postgres.host,
            port=Config.Postgres.port,
        )
        # Enforce a read-only session. The DB will reject INSERT, UPDATE, DELETE, etc.
        conn.set_session(readonly=True, autocommit=False)
        cur = conn.cursor()
        yield cur
    finally:
        if conn:
            conn.close()

# --- SECURE TOOLS ---

@tool(parse_docstring=True)
def list_tables(reasoning: str) -> str:
    """Lists all user-created tables in the database's 'public' schema.

    Args:
        reasoning (str): A detailed explanation of why you need to see all tables,
            relating the need directly to the user's query.

    Returns:
        str: A string representation of a Python list containing all table names.
    """
    log_panel(title="List Tables Tool", content=f"Reasoning: {reasoning}")
    try:
        with with_sql_cursor() as cursor:
            cursor.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"
            )
            tables = [row[0] for row in cursor.fetchall()]
        return str(tables)
    except Exception as e:
        log(f"[red]Error listing tables: {str(e)}[/red]")
        return f"Error listing tables: {str(e)}"


@tool(parse_docstring=True)
def sample_table(reasoning: str, table_name: str, row_sample_size: int = 5) -> str:
    """Retrieves a small sample of rows from a specific table to understand its structure.

    Args:
        reasoning (str): A detailed explanation of why you need to see sample data from this table.
        table_name (str): The exact, case-sensitive name of the table to sample.
        row_sample_size (int): The number of rows to retrieve (default is 5).

    Returns:
        str: A string with one row per line, showing all columns for each row.
    """
    log_panel(title="Sample Table Tool", content=f"Table: {table_name}\nRows: {row_sample_size}\nReasoning: {reasoning}")
    try:
        query = sql.SQL("SELECT * FROM {table} LIMIT %s;").format(
            table=sql.Identifier(table_name)
        )
        with with_sql_cursor() as cursor:
            cursor.execute(query, (row_sample_size,))
            rows = cursor.fetchall()
        return "\n".join([str(row) for row in rows])
    except Exception as e:
        log(f"[red]Error sampling table: {str(e)}[/red]")
        return f"Error sampling table: {str(e)}"


@tool(parse_docstring=True)
def describe_table(reasoning: str, table_name: str) -> str:
    """Returns detailed schema information about a table (columns, types, constraints).

    Args:
        reasoning (str): A detailed explanation of why you need to understand this table's structure.
        table_name (str): The exact, case-sensitive name of the table to describe.

    Returns:
        str: A string containing the table's schema information.
    """
    log_panel(title="Describe Table Tool", content=f"Table: {table_name}\nReasoning: {reasoning}")
    try:
        query = sql.SQL("""
            SELECT column_name, data_type, character_maximum_length
            FROM information_schema.columns
            WHERE table_name = %s;
        """)
        with with_sql_cursor() as cursor:
            cursor.execute(query, (table_name,))
            rows = cursor.fetchall()
        return "\n".join([str(row) for row in rows])
    except Exception as e:
        log(f"[red]Error describing table: {str(e)}[/red]")
        return f"Error describing table: {str(e)}"


@tool(parse_docstring=True)
def execute_sql(reasoning: str, sql_query: str) -> str:
    """Executes a single, read-only SQL query and returns the result.
    
    Data modification statements (INSERT, UPDATE, DELETE, DROP, etc.) are strictly forbidden.

    Args:
        reasoning (str): An explanation of why this specific query is necessary to answer the user's request.
        sql_query (str): The complete, properly formatted SQL SELECT query to execute.

    Returns:
        str: A string with the query results, with one row per line.
    """
    log_panel(title="Execute SQL Tool", content=f"Query: {sql_query}\nReasoning: {reasoning}")
    # Application-level check for forbidden keywords as a first line of defense
    forbidden_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE"]
    if any(keyword in sql_query.upper() for keyword in forbidden_keywords):
        return "Error: Only read-only SELECT queries are allowed."
    try:
        # The read-only session in with_sql_cursor() provides the primary, database-level security
        with with_sql_cursor() as cursor:
            cursor.execute(sql_query)
            rows = cursor.fetchall()
        return "\n".join([str(row) for row in rows])
    except Exception as e:
        log(f"[red]Error running query: {str(e)}[/red]")
        return f"Error running query: {str(e)}. Please check your SQL syntax."


@tool(parse_docstring=True)
def fuzzy_full_text_search(
    reasoning: str,
    table_name: str,
    column_names: List[str],
    search_phrase: str,
    similarity_threshold: float = 0.2,
    row_limit: int = 5,
) -> str:
    """Searches text columns for approximate matches, ideal for misspellings or variations.

    This tool is superior to using `LIKE` for fuzzy matching.

    Args:
        reasoning (str): An explanation of why a fuzzy search is needed for the user's query.
        table_name (str): The name of the table to search.
        column_names (List[str]): A list of column names to search within.
        search_phrase (str): The phrase or value to search for approximately.
        similarity_threshold (float): The threshold for similarity (0 to 1, default 0.2).
        row_limit (int): The maximum number of rows to return (default 5).

    Returns:
        str: A string listing rows that closely match the search phrase.
    """
    log_panel(title="Fuzzy Full-Text Search Tool", content=f"Table: {table_name}\nColumns: {column_names}\nPhrase: {search_phrase}\nReasoning: {reasoning}")
    if not column_names:
        return "Error: You must provide at least one column name to search."
    try:
        with with_sql_cursor() as cursor:
            safe_cols = [sql.Identifier(col) for col in column_names]
            
            similarity_conditions = sql.SQL(" OR ").join(
                sql.SQL("similarity(COALESCE(CAST({col} AS text), ''), CAST(%s AS text)) > %s").format(col=col)
                for col in safe_cols
            )
            order_by_expression = sql.SQL("GREATEST(") + sql.SQL(", ").join(
                sql.SQL("similarity(COALESCE(CAST({col} AS text), ''), CAST(%s AS text))").format(col=col)
                for col in safe_cols
            ) + sql.SQL(")")

            query = sql.SQL("""
                SELECT * FROM {table} WHERE {conditions} ORDER BY {ordering} DESC LIMIT %s;
            """).format(
                table=sql.Identifier(table_name),
                conditions=similarity_conditions,
                ordering=order_by_expression,
            )
            
            where_params = []
            for _ in column_names:
                where_params.extend([search_phrase, similarity_threshold])
            orderby_params = [search_phrase] * len(column_names)
            params = where_params + orderby_params + [row_limit]

            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            if not rows:
                return "No close matches found."
            return "\n".join([str(row) for row in rows])
    except Exception as e:
        log(f"[red]Error running fuzzy search: {str(e)}[/red]")
        if "function similarity(" in str(e):
             return f"Error: Fuzzy search is not enabled. Ensure the 'pg_trgm' extension is active in your database."
        return f"Error running fuzzy search: {str(e)}"

@tool(parse_docstring=True)
def get_foreign_key_relationships(reasoning: str) -> str:
    """Returns all foreign key relationships in the database to understand how tables are joined.

    Args:
        reasoning (str): An explanation of why you need to see the database's join relationships.

    Returns:
        str: A string describing the foreign key relationships, one per line.
    """
    log_panel(title="Get Foreign Key Relationships Tool", content=f"Reasoning: {reasoning}")
    try:
        query = """
            SELECT
                tc.table_name, kcu.column_name,
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name
            FROM
                information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                  ON tc.constraint_name = kcu.constraint_name AND tc.table_schema = kcu.table_schema
                JOIN information_schema.constraint_column_usage AS ccu
                  ON ccu.constraint_name = tc.constraint_name AND ccu.table_schema = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_schema = 'public';
        """
        with with_sql_cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
        
        if not rows:
            return "No foreign key relationships found in the 'public' schema."
        
        formatted_relationships = [
            f"Table '{row[0]}' (column '{row[1]}') references table '{row[2]}' (column '{row[3]}')."
            for row in rows
        ]
        return "\n".join(formatted_relationships)
    except Exception as e:
        log(f"[red]Error getting foreign key relationships: {str(e)}[/red]")
        return f"Error getting foreign key relationships: {str(e)}"

@tool(parse_docstring=True)
def get_distinct_column_values(reasoning: str, table_name: str, column_name: str) -> str:
    """Fetches all distinct values for a given column, useful for understanding categories or options.

    Args:
        reasoning (str): An explanation of why you need to see the distinct values for this column.
        table_name (str): The name of the table.
        column_name (str): The name of the column.

    Returns:
        str: A string representation of a Python list containing the distinct values.
    """
    log_panel(title="Get Distinct Column Values Tool", content=f"Table: {table_name}\nColumn: {column_name}\nReasoning: {reasoning}")
    try:
        query = sql.SQL("SELECT DISTINCT {column} FROM {table};").format(
            column=sql.Identifier(column_name), table=sql.Identifier(table_name)
        )
        with with_sql_cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
        return str([row[0] for row in rows])
    except Exception as e:
        log(f"[red]Error getting distinct values: {str(e)}[/red]")
        return f"Error getting distinct column values: {str(e)}"


@tool(parse_docstring=True)
def explain_query(reasoning: str, sql_query: str) -> str:
    """Provides the query execution plan for a SQL query, without running the query itself.
    
    This is useful for validating syntax and estimating the cost of a complex query.

    Args:
        reasoning (str): A detailed explanation of why you need to see the query plan.
        sql_query (str): The SQL SELECT query to explain. DO NOT include the EXPLAIN keyword here.

    Returns:
        str: The query plan as a string, with one step per line.
    """
    log_panel(title="Explain Query Tool", content=f"Query: {sql_query}\nReasoning: {reasoning}")

    # Defensively check and remove a leading "EXPLAIN" from the AI's input
    clean_query = sql_query.strip()
    if clean_query.upper().startswith("EXPLAIN"):
        # Strip "EXPLAIN" and any following whitespace
        clean_query = clean_query[7:].lstrip()

    if "SELECT" not in clean_query.upper():
        return "Error: EXPLAIN can only be used with SELECT queries."
        
    try:
        with with_sql_cursor() as cursor:
            cursor.execute(f"EXPLAIN {clean_query}")
            rows = cursor.fetchall()
        return "\n".join([str(row[0]) for row in rows])
    except Exception as e:
        log(f"[red]Error explaining query: {str(e)}[/red]")
        return f"Error explaining query: {str(e)}"

@tool(parse_docstring=True)
def search_entity_by_name(reasoning: str, entity_name: str, similarity_threshold: float = 0.3) -> str:
    """
    Searches for an entity by name across both customers and vendors to see if they exist in the system.

    Use this tool when the user asks to check a company or individual's name without specifying
    if they are a customer or a vendor.

    Args:
        reasoning (str): A detailed explanation of why you need to search for this entity.
        entity_name (str): The name of the company or individual to search for.
        similarity_threshold (float): The threshold for similarity (0 to 1, default 0.3).

    Returns:
        str: A consolidated report of findings in both the customer and vendor tables.
    """
    log_panel(title="Search Entity By Name Tool", content=f"Name: {entity_name}\nReasoning: {reasoning}")

    query = """
    (SELECT
        'Customer' AS entity_type,
        customer_name,
        similarity(customer_name, %s) AS score
    FROM
        customer_profile
    WHERE
        similarity(customer_name, %s) > %s
    )
    UNION ALL
    (SELECT
        'Vendor' AS entity_type,
        vendor_name,
        similarity(vendor_name, %s) AS score
    FROM
        vendor_profile
    WHERE
        similarity(vendor_name, %s) > %s
    )
    ORDER BY score DESC;
    """
    params = [
        entity_name, entity_name, similarity_threshold,
        entity_name, entity_name, similarity_threshold
    ]

    try:
        with with_sql_cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()

        if not rows:
            return f"No entity found matching the name '{entity_name}' as either a customer or a vendor."

        # Update the final output string to remove the email/contact part
        results = []
        for row in rows:
            entity_type, name, score = row
            results.append(
                f"Found as {entity_type}: Name='{name}' (Similarity Score: {score:.2f})"
            )
        return "\n".join(results)

    except Exception as e:
        log(f"[red]Error searching for entity: {str(e)}[/red]")
        return f"Error searching for entity: {str(e)}"