"""Correct identifiers in generated SQL against the live schema.

An LLM writing SQL for an unfamiliar database guesses names, and its guesses
follow the conventions it saw most often in training: snake_case tables, a
`customer_name` column on a `customer` table. Real databases disagree - Prisma
produces `"Customer"."name"`, other tools produce `tbl_customer.cust_nm`.

Rather than let the query fail and spend another LLM call on the retry, this
module maps the guessed names onto the real ones and reports what it changed,
so the model learns the true names for the rest of the conversation.

It is deliberately conservative: only identifiers in table position (after
FROM/JOIN/UPDATE/INTO) and qualified column references (`alias.column`) are
touched. Bare column names are left alone, because resolving them needs scope
analysis that a regex cannot do safely.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from schema import DatabaseSchema, Table, quote_identifier

# A table reference following FROM / JOIN / INTO / UPDATE, optionally
# schema-qualified, optionally quoted, optionally followed by an alias.
_TABLE_REF = re.compile(
    r"""(?P<keyword>\b(?:from|join|into|update)\s+)
        (?P<ref>
            (?:"[^"]+"|[A-Za-z_][A-Za-z0-9_$]*)
            (?:\s*\.\s*(?:"[^"]+"|[A-Za-z_][A-Za-z0-9_$]*))?
        )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# A qualified column reference: alias.column or "Alias"."Column".
_QUALIFIED_COLUMN = re.compile(
    r"""(?<![\w."])
        (?P<qualifier>"[^"]+"|[A-Za-z_][A-Za-z0-9_$]*)
        \s*\.\s*
        (?P<column>"[^"]+"|[A-Za-z_][A-Za-z0-9_$]*)
        (?![\w."(])
    """,
    re.VERBOSE,
)

# Alias declarations, so `FROM "Customer" c` lets us resolve `c.name`.
_ALIAS = re.compile(
    r"""\b(?:from|join)\s+
        (?P<table>(?:"[^"]+"|[A-Za-z_][A-Za-z0-9_$]*)
                  (?:\s*\.\s*(?:"[^"]+"|[A-Za-z_][A-Za-z0-9_$]*))?)
        (?:\s+as)?\s+
        (?P<alias>(?!where\b|on\b|using\b|join\b|inner\b|left\b|right\b|full\b|
                     cross\b|group\b|order\b|limit\b|having\b|set\b|values\b|
                     union\b|except\b|intersect\b|offset\b|window\b|returning\b)
                  [A-Za-z_][A-Za-z0-9_$]*)
    """,
    re.IGNORECASE | re.VERBOSE,
)

_CTE_NAME = re.compile(
    r"(?:\bwith\s+(?:recursive\s+)?|,\s*)([A-Za-z_][A-Za-z0-9_$]*)\s+as\s*\(",
    re.IGNORECASE,
)


@dataclass
class RepairResult:
    sql: str
    corrections: List[str] = field(default_factory=list)
    unresolved: List[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.corrections)

    def add_correction(self, message: str) -> None:
        # The same identifier often appears several times in one query; report
        # each distinct correction once so the note stays short.
        if message not in self.corrections:
            self.corrections.append(message)

    def add_unresolved(self, message: str) -> None:
        if message not in self.unresolved:
            self.unresolved.append(message)

    def note(self) -> str:
        """A short message telling the model what was corrected."""
        parts = []
        if self.corrections:
            parts.append(
                "Note: your query used names that do not exist; they were "
                "corrected to " + "; ".join(self.corrections) + ". "
                "Use the corrected names from now on."
            )
        if self.unresolved:
            parts.append(
                "Could not resolve: " + "; ".join(self.unresolved) + ". "
                "Call describe_table to see the real column names."
            )
        return " ".join(parts)


def _strip_quotes(identifier: str) -> str:
    identifier = identifier.strip()
    if identifier.startswith('"') and identifier.endswith('"'):
        return identifier[1:-1].replace('""', '"')
    return identifier


def _split_ref(ref: str) -> tuple:
    """Split a possibly schema-qualified reference into (schema, table)."""
    parts = re.split(r"\s*\.\s*", ref.strip())
    if len(parts) == 2:
        return _strip_quotes(parts[0]), _strip_quotes(parts[1])
    return None, _strip_quotes(parts[0])


def find_cte_names(sql: str) -> set:
    """Names defined by WITH clauses, which are not real tables."""
    return {m.group(1).lower() for m in _CTE_NAME.finditer(sql)}


def repair_sql(sql: str, schema: DatabaseSchema) -> RepairResult:
    """Rewrite `sql` so its identifiers match the real schema."""
    result = RepairResult(sql=sql)
    cte_names = find_cte_names(sql)
    # alias -> resolved Table, built as table references are corrected.
    alias_map: Dict[str, Table] = {}

    # --- pass 1: table references ---------------------------------------

    def fix_table(match: re.Match) -> str:
        keyword, ref = match.group("keyword"), match.group("ref")
        schema_name, table_name = _split_ref(ref)

        if table_name.lower() in cte_names:
            return match.group(0)

        exact = schema.find_table(table_name, schema_name)
        if exact:
            alias_map[table_name.lower()] = exact
            # Re-render so a quoted PascalCase table is quoted correctly even
            # when the model wrote it bare.
            return keyword + exact.qualified

        resolved, suggestions = schema.resolve_table(table_name)
        if resolved:
            alias_map[table_name.lower()] = resolved
            result.add_correction(f"table '{table_name}' -> {resolved.qualified}")
            return keyword + resolved.qualified

        hint = f" (did you mean: {', '.join(suggestions)}?)" if suggestions else ""
        result.add_unresolved(f"table '{table_name}'{hint}")
        return match.group(0)

    sql = _TABLE_REF.sub(fix_table, sql)

    # --- pass 2: record aliases -----------------------------------------

    for match in _ALIAS.finditer(sql):
        _, table_name = _split_ref(match.group("table"))
        table = schema.find_table(table_name)
        if table:
            alias_map[match.group("alias").lower()] = table

    # --- pass 3: qualified column references ----------------------------

    def fix_column(match: re.Match) -> str:
        qualifier_raw, column_raw = match.group("qualifier"), match.group("column")
        qualifier, column = _strip_quotes(qualifier_raw), _strip_quotes(column_raw)

        if column == "*" or qualifier.lower() in cte_names:
            return match.group(0)

        table = alias_map.get(qualifier.lower())
        if table is None:
            return match.group(0)

        exact = table.column(column)
        if exact:
            return f"{quote_identifier(qualifier)}.{quote_identifier(exact.name)}"

        resolved, suggestions = schema.resolve_column(table, column)
        if resolved:
            result.add_correction(
                f"column '{qualifier}.{column}' -> {resolved.name} on {table.qualified}"
            )
            return f"{quote_identifier(qualifier)}.{quote_identifier(resolved.name)}"

        hint = f" (available: {', '.join(suggestions)})" if suggestions else ""
        result.add_unresolved(f"column '{qualifier}.{column}' on {table.qualified}{hint}")
        return match.group(0)

    result.sql = _QUALIFIED_COLUMN.sub(fix_column, sql)
    return result


def suggest_for_error(error_message: str, schema: DatabaseSchema) -> Optional[str]:
    """Turn a Postgres 'does not exist' error into an actionable hint."""
    relation = re.search(r'relation "([^"]+)" does not exist', error_message)
    if relation:
        _, name = _split_ref(relation.group(1))
        resolved, suggestions = schema.resolve_table(name)
        options = [resolved.qualified] if resolved else suggestions
        if options:
            return f"Table '{name}' does not exist. Closest matches: {', '.join(options[:5])}."

    column = re.search(r'column "?([\w.]+)"? does not exist', error_message)
    if column:
        raw = column.group(1)
        qualifier, _, bare = raw.rpartition(".")
        candidates: List[str] = []
        if qualifier:
            table = schema.find_table(qualifier)
            if table:
                _, candidates = schema.resolve_column(table, bare)
        if not candidates:
            # Search every table for a column of that name.
            for table in schema.tables:
                if table.column(bare):
                    candidates.append(f"{table.qualified}.{bare}")
                if len(candidates) >= 5:
                    break
        if candidates:
            return f"Column '{raw}' does not exist. Closest matches: {', '.join(candidates[:5])}."
    return None
