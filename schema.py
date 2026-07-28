"""Live schema introspection and identifier resolution.

The agent must work against a database it has never seen, whose naming
conventions it cannot predict: `customer_profile` in one database is `Customer`
in the next and `tbl_cust` in a third. Nothing here is hardcoded - the schema is
read from the database at runtime and cached.

Two problems are solved:

  * Discovery - what tables and columns exist, which look like names, which look
    like entities. This drives the agent's orientation without it having to
    spend LLM calls guessing.
  * Resolution - when the model writes `customer.customer_name` but the database
    has `"Customer"."name"`, map the guess onto what actually exists rather than
    failing with a syntax error the model has to burn a request recovering from.

Postgres folds unquoted identifiers to lower case but preserves the case of
quoted ones, so `Customer` and `customer` are different tables. Every lookup
here is case-insensitive, and rendering always quotes, which is correct for
either convention.
"""

import re
import threading
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Sequence, Tuple

# Minimum similarity before a mistyped name is silently corrected. Chosen so
# genuine convention differences (customers/Customer, first_name/firstName)
# pass, while coincidental letter overlap (payment/Department) does not.
DEFAULT_CUTOFF = 0.75

# Identifiers that are safe to leave unquoted; anything else gets quoted.
_SAFE_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")

# Columns whose names suggest they hold a human-readable label. Used to guess
# which column to search when the user names an entity.
_NAME_HINTS = (
    "name",
    "title",
    "label",
    "description",
    "code",
    "email",
    "company",
    "customer",
    "vendor",
    "supplier",
    "client",
    "employee",
    "person",
    "contact",
    "username",
)

# Text-ish types worth running a similarity search against.
_TEXT_TYPES = {
    "text",
    "character varying",
    "character",
    "citext",
    "name",
    "uuid",
}


# The quoting character differs by engine: MySQL uses backticks, everything
# else the SQL-standard double quote. Set once when a connection is made so
# rendered identifiers match the dialect actually in use.
_quote_char = '"'

# Schemas that are implicit for their engine and should not be written into a
# qualified name: 'public' on PostgreSQL, 'dbo' on SQL Server, 'main' on SQLite.
# Qualifying with them is valid but adds noise to every identifier the model
# reads and writes.
_IMPLICIT_SCHEMAS = {"public", "dbo", "main", ""}


def set_dialect(engine: str) -> None:
    """Configure identifier quoting for the connected engine."""
    global _quote_char
    _quote_char = "`" if engine == "mysql" else '"'


def quote_identifier(name: str) -> str:
    """Render an identifier for SQL, quoting when the engine would need it."""
    if _SAFE_IDENTIFIER.match(name):
        return name
    if _quote_char == "`":
        return "`" + name.replace("`", "``") + "`"
    return '"' + name.replace('"', '""') + '"'


def qualify(schema: str, table: str) -> str:
    """Render a schema-qualified table reference."""
    if schema and schema.lower() not in _IMPLICIT_SCHEMAS:
        return f"{quote_identifier(schema)}.{quote_identifier(table)}"
    return quote_identifier(table)


@dataclass(frozen=True)
class Column:
    name: str
    data_type: str
    nullable: bool

    @property
    def is_texty(self) -> bool:
        return self.data_type in _TEXT_TYPES

    @property
    def looks_like_a_name(self) -> bool:
        lowered = self.name.lower()
        return self.is_texty and any(hint in lowered for hint in _NAME_HINTS)


@dataclass
class Table:
    schema: str
    name: str
    columns: List[Column] = field(default_factory=list)
    primary_key: List[str] = field(default_factory=list)
    estimated_rows: int = 0

    @property
    def qualified(self) -> str:
        return qualify(self.schema, self.name)

    @property
    def key(self) -> str:
        return f"{self.schema.lower()}.{self.name.lower()}"

    def column(self, name: str) -> Optional[Column]:
        """Case-insensitive column lookup."""
        target = name.lower()
        for column in self.columns:
            if column.name.lower() == target:
                return column
        return None

    def name_columns(self) -> List[Column]:
        """Text columns that plausibly hold a human-readable label."""
        hinted = [c for c in self.columns if c.looks_like_a_name]
        return hinted or [c for c in self.columns if c.is_texty]


@dataclass
class ForeignKey:
    schema: str
    table: str
    column: str
    ref_schema: str
    ref_table: str
    ref_column: str

    def render(self) -> str:
        return (
            f"{qualify(self.schema, self.table)}.{quote_identifier(self.column)} -> "
            f"{qualify(self.ref_schema, self.ref_table)}.{quote_identifier(self.ref_column)}"
        )


@dataclass
class DatabaseSchema:
    """Everything known about the connected database."""

    tables: List[Table] = field(default_factory=list)
    foreign_keys: List[ForeignKey] = field(default_factory=list)

    # -- lookup ----------------------------------------------------------

    def find_table(self, name: str, schema: Optional[str] = None) -> Optional[Table]:
        """Exact (case-insensitive) table lookup, optionally schema-qualified.

        Accepts 'orders' or 'sales.orders'. When several schemas hold a table of
        the same name, 'public' wins.
        """
        if "." in name and schema is None:
            schema, _, name = name.partition(".")

        target = name.strip('"').lower()
        matches = [t for t in self.tables if t.name.lower() == target]
        if schema:
            wanted = schema.strip('"').lower()
            matches = [t for t in matches if t.schema.lower() == wanted]
        if not matches:
            return None
        for table in matches:
            if table.schema == "public":
                return table
        return matches[0]

    def table_names(self) -> List[str]:
        return [t.qualified for t in self.tables]

    def foreign_keys_for(self, table_names: Sequence[str]) -> List[ForeignKey]:
        """Foreign keys touching any of the given tables."""
        wanted = {n.lower() for n in table_names}
        return [
            fk
            for fk in self.foreign_keys
            if fk.table.lower() in wanted or fk.ref_table.lower() in wanted
        ]

    # -- fuzzy resolution ------------------------------------------------

    def resolve_table(self, name: str, cutoff: float = DEFAULT_CUTOFF) -> Tuple[Optional[Table], List[str]]:
        """Map a possibly-wrong table name onto a real one.

        Returns the best match and a ranked list of alternative suggestions.
        An exact match short-circuits, so the common case costs nothing.
        """
        exact = self.find_table(name)
        if exact:
            return exact, []

        bare = name.split(".")[-1].strip('"').lower()
        scored = sorted(
            ((_similarity(bare, t.name.lower()), t) for t in self.tables),
            key=lambda pair: pair[0],
            reverse=True,
        )
        if not scored:
            return None, []

        best_score, best_table = scored[0]
        suggestions = [t.qualified for score, t in scored[:5] if score > 0.35]
        if best_score >= cutoff:
            return best_table, suggestions
        return None, suggestions

    def resolve_column(
        self, table: Table, name: str, cutoff: float = DEFAULT_CUTOFF
    ) -> Tuple[Optional[Column], List[str]]:
        """Map a possibly-wrong column name onto a real column of `table`."""
        exact = table.column(name)
        if exact:
            return exact, []

        bare = name.strip('"').lower()
        scored = sorted(
            ((_similarity(bare, c.name.lower()), c) for c in table.columns),
            key=lambda pair: pair[0],
            reverse=True,
        )
        if not scored:
            return None, []

        best_score, best_column = scored[0]
        suggestions = [c.name for score, c in scored[:5] if score > 0.35]
        if best_score >= cutoff:
            return best_column, suggestions
        return None, suggestions

    # -- rendering -------------------------------------------------------

    def overview(self, max_tables: int = 60) -> str:
        """A compact map of the database for the agent's first orientation.

        Deliberately terse: this goes into the prompt, so it lists names and
        sizes rather than full column definitions.
        """
        if not self.tables:
            return "The database contains no readable tables."

        ordered = sorted(self.tables, key=lambda t: (-t.estimated_rows, t.name.lower()))
        lines = []
        for table in ordered[:max_tables]:
            columns = ", ".join(c.name for c in table.columns[:8])
            if len(table.columns) > 8:
                columns += f", +{len(table.columns) - 8} more"
            size = f"~{table.estimated_rows:,} rows" if table.estimated_rows else "empty"
            lines.append(f"{table.qualified} ({size}): {columns}")

        rendered = "\n".join(lines)
        if len(ordered) > max_tables:
            rendered += f"\n... and {len(ordered) - max_tables} more tables"
        return rendered

    def describe(self, table: Table) -> str:
        """Full column detail for one table, plus its relationships."""
        lines = [f"{table.qualified} (~{table.estimated_rows:,} rows)"]
        for column in table.columns:
            marker = " PK" if column.name in table.primary_key else ""
            null = "" if column.nullable else " NOT NULL"
            lines.append(f"  {column.name} {column.data_type}{null}{marker}")

        related = [
            fk
            for fk in self.foreign_keys
            if fk.table.lower() == table.name.lower()
            or fk.ref_table.lower() == table.name.lower()
        ]
        if related:
            lines.append("  relationships:")
            lines.extend(f"    {fk.render()}" for fk in related)
        return "\n".join(lines)


def _similarity(a: str, b: str) -> float:
    """Similarity between two identifiers, tolerant of naming conventions.

    `customer_name`, `customerName` and `CustomerName` should all look alike, so
    separators are stripped before comparison. A containment bonus lets a short
    guess such as `customer` score well against `customer_profile`.
    """
    a_norm = re.sub(r"[^a-z0-9]", "", a.lower())
    b_norm = re.sub(r"[^a-z0-9]", "", b.lower())
    if not a_norm or not b_norm:
        return 0.0
    if a_norm == b_norm:
        return 1.0

    # Singular/plural is the most common mismatch and is always a real match.
    if a_norm.rstrip("s") == b_norm.rstrip("s"):
        return 0.97

    ratio = SequenceMatcher(None, a_norm, b_norm).ratio()

    # A containment bonus lets a short guess such as `customer` match
    # `customer_profile`. It only applies when the contained string is a
    # substantial part of the other and long enough to be meaningful -
    # otherwise `payment` would "match" `department` on a few shared letters.
    shorter, longer = sorted((len(a_norm), len(b_norm)))
    if (a_norm in b_norm or b_norm in a_norm) and shorter >= 4 and shorter / longer >= 0.5:
        ratio = max(ratio, 0.6 + 0.4 * (shorter / longer))
    return ratio



# --- Loading --------------------------------------------------------------

# Schemas that hold engine internals rather than user data. Listing them would
# bury the real tables under hundreds of catalog entries.
_EXCLUDED_SCHEMAS = {
    # PostgreSQL
    "pg_catalog", "information_schema", "pg_toast",
    # MySQL / MariaDB
    "mysql", "performance_schema", "sys",
    # SQL Server
    "sys", "guest", "db_owner", "db_accessadmin", "db_securityadmin",
    "db_ddladmin", "db_backupoperator", "db_datareader", "db_datawriter",
    "db_denydatareader", "db_denydatawriter", "information_schema",
    # Oracle
    "sysaux", "system", "outln", "xdb",
}

_cache: Dict[str, DatabaseSchema] = {}
_lock = threading.Lock()


def load_schema(connection, refresh: bool = False) -> DatabaseSchema:
    """Read the full schema through SQLAlchemy, caching the result.

    `connection` is a DatabaseConnection. SQLAlchemy's inspector is used rather
    than hand-written catalog queries so the same code serves PostgreSQL, MySQL,
    SQLite, SQL Server and Oracle.

    The cache key includes the target database, so switching connections in the
    UI does not serve a previous database's schema.
    """
    key = connection.settings.safe_url

    with _lock:
        if not refresh and key in _cache:
            return _cache[key]

    from sqlalchemy import inspect as sa_inspect

    database = DatabaseSchema()
    engine_name = connection.settings.engine

    with connection.connect() as conn:
        inspector = sa_inspect(conn)

        # SQLite has no schemas; every other engine may have several. Only the
        # default schema is read unless the engine reports more, which keeps
        # the overview focused on the user's own tables.
        default_schema = inspector.default_schema_name
        schemas = [default_schema] if default_schema else [None]
        if engine_name in ("postgresql", "mssql"):
            try:
                schemas = [
                    s for s in inspector.get_schema_names()
                    if s.lower() not in _EXCLUDED_SCHEMAS
                ] or schemas
            except Exception:  # noqa: BLE001
                pass

        by_key: Dict[str, Table] = {}
        for schema_name in schemas:
            try:
                names = inspector.get_table_names(schema=schema_name)
                names += inspector.get_view_names(schema=schema_name)
            except Exception:  # noqa: BLE001
                continue

            for table_name in names:
                try:
                    columns = inspector.get_columns(table_name, schema=schema_name)
                except Exception:  # noqa: BLE001
                    continue

                table = Table(schema=schema_name or "", name=table_name)
                for column in columns:
                    table.columns.append(
                        Column(
                            name=column["name"],
                            data_type=_type_name(column.get("type")),
                            nullable=bool(column.get("nullable", True)),
                        )
                    )

                try:
                    pk = inspector.get_pk_constraint(table_name, schema=schema_name)
                    table.primary_key = list(pk.get("constrained_columns") or [])
                except Exception:  # noqa: BLE001
                    pass

                database.tables.append(table)
                by_key[table.key] = table

                try:
                    for fk in inspector.get_foreign_keys(table_name, schema=schema_name):
                        referred_table = fk.get("referred_table")
                        constrained = fk.get("constrained_columns") or []
                        referred = fk.get("referred_columns") or []
                        if not referred_table or not constrained or not referred:
                            continue
                        database.foreign_keys.append(
                            ForeignKey(
                                schema=schema_name or "",
                                table=table_name,
                                column=constrained[0],
                                ref_schema=fk.get("referred_schema") or (schema_name or ""),
                                ref_table=referred_table,
                                ref_column=referred[0],
                            )
                        )
                except Exception:  # noqa: BLE001
                    pass

        _load_row_estimates(conn, engine_name, by_key)

    with _lock:
        _cache[key] = database
    return database


def _type_name(sa_type) -> str:
    """A readable type name, lower-cased to match the text-type checks."""
    if sa_type is None:
        return "unknown"
    try:
        return str(sa_type).lower()
    except Exception:  # noqa: BLE001
        return type(sa_type).__name__.lower()


def _load_row_estimates(conn, engine_name: str, by_key: Dict[str, "Table"]) -> None:
    """Fill in approximate row counts where the engine can supply them cheaply.

    These come from planner statistics, so they cost one query for the whole
    database. COUNT(*) per table would scan everything, which on a large
    database is far too slow to run on every connect.
    """
    from sqlalchemy import text as sa_text

    queries = {
        "postgresql": (
            "SELECT schemaname, relname, GREATEST(n_live_tup, 0) FROM pg_stat_user_tables"
        ),
        "mysql": (
            "SELECT table_schema, table_name, COALESCE(table_rows, 0) "
            "FROM information_schema.tables WHERE table_schema = DATABASE()"
        ),
    }
    query = queries.get(engine_name)
    if not query:
        return

    try:
        for schema_name, table_name, estimate in conn.execute(sa_text(query)):
            table = by_key.get(f"{(schema_name or '').lower()}.{table_name.lower()}")
            if table:
                table.estimated_rows = int(estimate or 0)
    except Exception:  # noqa: BLE001
        # Statistics are a nicety; the app works fine without them.
        pass


def clear_cache() -> None:
    with _lock:
        _cache.clear()
