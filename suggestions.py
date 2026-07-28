"""Starter questions generated from the connected schema.

The mockup showed hardcoded prompts ("Total sales last quarter"), which would be
wrong for most databases. These are derived from the tables actually present, so
they name real tables and only appear when the data supports them.

No LLM call is involved: suggestions must be free, since they are shown before
the user has asked anything.
"""

import re
from typing import List

from schema import DatabaseSchema, Table

# Column-name fragments that hint at what a table records.
_DATE_HINTS = ("date", "_at", "time", "created", "updated", "issued", "period")
_MONEY_HINTS = ("amount", "total", "price", "cost", "value", "revenue", "salary",
                "balance", "paid", "due", "gross", "net")
_STATUS_HINTS = ("status", "state", "stage", "type", "category", "kind")


def _humanise(name: str) -> str:
    """Turn 'SalesOrder' or 'sales_order' into 'sales order' for prose."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    spaced = spaced.replace("_", " ").strip()
    return spaced.lower()


def _has(table: Table, hints) -> bool:
    return any(
        any(h in column.name.lower() for h in hints) for column in table.columns
    )


def _column_matching(table: Table, hints):
    for column in table.columns:
        if any(h in column.name.lower() for h in hints):
            return column
    return None


def suggest(db: DatabaseSchema, limit: int = 4) -> List[str]:
    """Up to `limit` starter questions grounded in this database's tables."""
    if not db or not db.tables:
        return []

    # Prefer tables with data; fall back to the largest by column count so an
    # un-analyzed database still produces something useful.
    populated = [t for t in db.tables if t.estimated_rows > 0]
    candidates = populated or db.tables
    ranked = sorted(
        candidates,
        key=lambda t: (t.estimated_rows, len(t.columns)),
        reverse=True,
    )

    questions: List[str] = []
    seen_tables = set()

    for table in ranked:
        if len(questions) >= limit:
            break
        label = _humanise(table.name)
        if label in seen_tables:
            continue
        seen_tables.add(label)

        status = _column_matching(table, _STATUS_HINTS)
        money = _column_matching(table, _MONEY_HINTS)
        date = _column_matching(table, _DATE_HINTS)

        # "total total amount" reads badly, so drop a leading 'total' the
        # column name already supplies.
        money_label = _humanise(money.name) if money else ""
        if money_label.startswith("total "):
            money_label = money_label[len("total "):]

        if money and date:
            questions.append(f"What is the total {money_label} of {label} by month?")
        elif money:
            questions.append(f"What is the total {money_label} across all {label}?")
        elif status:
            questions.append(f"How many {label} are there in each {_humanise(status.name)}?")
        elif date:
            questions.append(f"Show me the 10 most recent {label} records")
        else:
            questions.append(f"How many {label} records are there?")

    # A relationship question is usually the most revealing, so add one when
    # the schema actually declares a foreign key.
    if db.foreign_keys and len(questions) < limit:
        fk = db.foreign_keys[0]
        questions.append(
            f"Show {_humanise(fk.table)} records together with their "
            f"related {_humanise(fk.ref_table)}"
        )

    if len(questions) < limit:
        questions.append("What tables are in this database and what do they contain?")

    return questions[:limit]
