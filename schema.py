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


def quote_identifier(name: str) -> str:
    """Render an identifier for SQL, quoting when Postgres would need it."""
    if _SAFE_IDENTIFIER.match(name):
        return name
    return '"' + name.replace('"', '""') + '"'


def qualify(schema: str, table: str) -> str:
    """Render a schema-qualified table reference."""
    if schema and schema != "public":
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

# System schemas are never useful to the agent and would bury the real tables.
_EXCLUDED_SCHEMAS = ("pg_catalog", "information_schema", "pg_toast")

_cache: Dict[str, DatabaseSchema] = {}
_lock = threading.Lock()


def load_schema(cursor, refresh: bool = False) -> DatabaseSchema:
    """Read the full schema from the database, caching the result.

    One pass over the catalog replaces what would otherwise be many round trips
    of separate list-tables and describe-table calls, and the result is reused for the life of
    the process.
    """
    with _lock:
        if not refresh and "schema" in _cache:
            return _cache["schema"]

    schema = DatabaseSchema()

    # Columns for every user table and view, in one query.
    cursor.execute(
        """
        SELECT c.table_schema, c.table_name, c.column_name, c.data_type, c.is_nullable
        FROM information_schema.columns c
        JOIN information_schema.tables t
          ON t.table_schema = c.table_schema AND t.table_name = c.table_name
        WHERE c.table_schema NOT IN %s
          AND t.table_type IN ('BASE TABLE', 'VIEW')
        ORDER BY c.table_schema, c.table_name, c.ordinal_position;
        """,
        (_EXCLUDED_SCHEMAS,),
    )
    by_key: Dict[str, Table] = {}
    for schema_name, table_name, column_name, data_type, nullable in cursor.fetchall():
        key = f"{schema_name.lower()}.{table_name.lower()}"
        table = by_key.get(key)
        if table is None:
            table = Table(schema=schema_name, name=table_name)
            by_key[key] = table
            schema.tables.append(table)
        table.columns.append(
            Column(name=column_name, data_type=data_type, nullable=(nullable == "YES"))
        )

    # Primary keys.
    cursor.execute(
        """
        SELECT tc.table_schema, tc.table_name, kcu.column_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON kcu.constraint_name = tc.constraint_name
         AND kcu.table_schema = tc.table_schema
        WHERE tc.constraint_type = 'PRIMARY KEY'
          AND tc.table_schema NOT IN %s;
        """,
        (_EXCLUDED_SCHEMAS,),
    )
    for schema_name, table_name, column_name in cursor.fetchall():
        table = by_key.get(f"{schema_name.lower()}.{table_name.lower()}")
        if table:
            table.primary_key.append(column_name)

    # Foreign keys.
    cursor.execute(
        """
        SELECT tc.table_schema, tc.table_name, kcu.column_name,
               ccu.table_schema, ccu.table_name, ccu.column_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON kcu.constraint_name = tc.constraint_name
         AND kcu.table_schema = tc.table_schema
        JOIN information_schema.constraint_column_usage ccu
          ON ccu.constraint_name = tc.constraint_name
         AND ccu.table_schema = tc.table_schema
        WHERE tc.constraint_type = 'FOREIGN KEY'
          AND tc.table_schema NOT IN %s;
        """,
        (_EXCLUDED_SCHEMAS,),
    )
    for row in cursor.fetchall():
        schema.foreign_keys.append(ForeignKey(*row))

    # Row estimates from the planner's statistics - effectively free, whereas
    # COUNT(*) per table would scan the entire database.
    cursor.execute(
        """
        SELECT schemaname, relname, GREATEST(n_live_tup, 0)
        FROM pg_stat_user_tables;
        """
    )
    for schema_name, table_name, estimate in cursor.fetchall():
        table = by_key.get(f"{schema_name.lower()}.{table_name.lower()}")
        if table:
            table.estimated_rows = int(estimate or 0)

    with _lock:
        _cache["schema"] = schema
    return schema


def clear_cache() -> None:
    with _lock:
        _cache.clear()
