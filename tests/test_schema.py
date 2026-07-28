"""Tests for schema discovery, name resolution and SQL repair.

The fixture models a Prisma-style database with PascalCase tables and camelCase
columns, which is exactly the convention an LLM is least likely to guess.
"""

import pytest

from schema import Column, DatabaseSchema, ForeignKey, Table, quote_identifier
from sql_repair import repair_sql, suggest_for_error


def _table(name, columns, schema="public", rows=100, pk=("id",)):
    return Table(
        schema=schema,
        name=name,
        columns=[Column(n, t, True) for n, t in columns],
        primary_key=list(pk),
        estimated_rows=rows,
    )


@pytest.fixture
def db():
    schema = DatabaseSchema(
        tables=[
            _table("Customer", [("id", "integer"), ("name", "text"), ("gstNumber", "text")]),
            _table(
                "Employee",
                [
                    ("id", "integer"),
                    ("firstName", "text"),
                    ("lastName", "text"),
                    ("employeeCode", "text"),
                    ("departmentId", "integer"),
                ],
            ),
            _table(
                "Invoice",
                [
                    ("id", "integer"),
                    ("customerId", "integer"),
                    ("totalAmount", "numeric"),
                    ("status", "text"),
                ],
            ),
            _table("Department", [("id", "integer"), ("name", "text")]),
            _table("audit_log", [("id", "integer"), ("message", "text")], schema="ops"),
        ],
        foreign_keys=[
            ForeignKey("public", "Invoice", "customerId", "public", "Customer", "id"),
            ForeignKey("public", "Employee", "departmentId", "public", "Department", "id"),
        ],
    )
    return schema


# --- identifier quoting ---------------------------------------------------


def test_quoting_only_when_needed():
    assert quote_identifier("customer") == "customer"
    assert quote_identifier("Customer") == '"Customer"'
    assert quote_identifier("firstName") == '"firstName"'
    assert quote_identifier("order by") == '"order by"'


def test_embedded_quote_is_escaped():
    assert quote_identifier('we"ird') == '"we""ird"'


# --- table lookup ---------------------------------------------------------


def test_find_table_is_case_insensitive(db):
    assert db.find_table("customer").name == "Customer"
    assert db.find_table("CUSTOMER").name == "Customer"
    assert db.find_table('"Customer"').name == "Customer"


def test_find_table_accepts_schema_qualification(db):
    assert db.find_table("ops.audit_log").name == "audit_log"
    assert db.find_table("audit_log", schema="ops").schema == "ops"


def test_unknown_table_returns_none(db):
    assert db.find_table("nonexistent") is None


def test_non_public_schema_is_rendered_qualified(db):
    assert db.find_table("ops.audit_log").qualified == "ops.audit_log"
    assert db.find_table("Customer").qualified == '"Customer"'


# --- fuzzy resolution -----------------------------------------------------


@pytest.mark.parametrize(
    "guess,expected",
    [
        ("customers", "Customer"),
        ("customer_profile", "Customer"),
        ("employees", "Employee"),
        ("invoices", "Invoice"),
        ("Employees", "Employee"),
    ],
)
def test_plural_and_suffixed_guesses_resolve(db, guess, expected):
    resolved, _ = db.resolve_table(guess)
    assert resolved is not None and resolved.name == expected


def test_wildly_wrong_name_does_not_resolve(db):
    resolved, suggestions = db.resolve_table("zzzz_quarterly_forecast")
    assert resolved is None


@pytest.mark.parametrize("guess", ["course", "payment", "student", "shipment"])
def test_unrelated_names_are_not_forced_onto_a_table(db, guess):
    """Coincidental letter overlap must not silently rewrite a query.

    Regression: 'payment' once matched 'Department' and 'course' matched 'User',
    which would have sent the query to entirely the wrong table.
    """
    resolved, _ = db.resolve_table(guess)
    assert resolved is None


def test_similarity_separates_real_matches_from_coincidence():
    from schema import DEFAULT_CUTOFF, _similarity

    assert _similarity("customers", "customer") >= DEFAULT_CUTOFF
    assert _similarity("first_name", "firstName") >= DEFAULT_CUTOFF
    assert _similarity("purchase_order", "PurchaseOrder") >= DEFAULT_CUTOFF
    assert _similarity("payment", "Department") < DEFAULT_CUTOFF
    assert _similarity("course", "User") < DEFAULT_CUTOFF


@pytest.mark.parametrize(
    "guess,expected",
    [
        ("first_name", "firstName"),
        ("firstname", "firstName"),
        ("employee_code", "employeeCode"),
        ("FirstName", "firstName"),
    ],
)
def test_column_naming_convention_is_bridged(db, guess, expected):
    table = db.find_table("Employee")
    resolved, _ = db.resolve_column(table, guess)
    assert resolved is not None and resolved.name == expected


def test_unknown_column_offers_suggestions(db):
    table = db.find_table("Employee")
    resolved, suggestions = db.resolve_column(table, "salary_amount")
    assert resolved is None or suggestions


# --- SQL repair -----------------------------------------------------------


def test_lowercase_table_is_quoted_to_match_real_case(db):
    result = repair_sql("SELECT id FROM customer", db)
    assert '"Customer"' in result.sql


def test_wrong_table_name_is_corrected(db):
    result = repair_sql("SELECT id FROM customers", db)
    assert '"Customer"' in result.sql
    assert result.changed
    assert "customers" in result.note()


def test_qualified_column_is_corrected_through_alias(db):
    result = repair_sql('SELECT e.first_name FROM "Employee" e', db)
    assert '"firstName"' in result.sql
    assert result.changed


def test_join_repairs_both_sides(db):
    query = "SELECT i.id, c.name FROM invoices i JOIN customers c ON i.customer_id = c.id"
    result = repair_sql(query, db)
    assert '"Invoice"' in result.sql
    assert '"Customer"' in result.sql
    assert '"customerId"' in result.sql


def test_repeated_correction_is_reported_once(db):
    """A column used in SELECT and GROUP BY should not be listed twice."""
    query = (
        "SELECT c.company_name, count(*) FROM customers c "
        "GROUP BY c.company_name ORDER BY c.company_name"
    )
    result = repair_sql(query, db)
    company_notes = [c for c in result.corrections if "company_name" in c]
    assert len(company_notes) <= 1


def test_correct_sql_is_left_alone(db):
    query = 'SELECT "firstName" FROM "Employee" e WHERE e."employeeCode" = \'E1\''
    result = repair_sql(query, db)
    assert not result.changed


def test_cte_name_is_not_treated_as_a_table(db):
    query = (
        "WITH totals AS (SELECT id FROM customers) "
        "SELECT * FROM totals"
    )
    result = repair_sql(query, db)
    assert '"Customer"' in result.sql
    # The CTE itself must not be rewritten into a real table.
    assert "FROM totals" in result.sql


def test_unresolvable_table_is_reported_not_silently_changed(db):
    result = repair_sql("SELECT * FROM quarterly_forecast_zzz", db)
    assert result.unresolved
    assert "quarterly_forecast_zzz" in result.sql


def test_star_qualifier_is_untouched(db):
    result = repair_sql('SELECT c.* FROM "Customer" c', db)
    assert "c.*" in result.sql


def test_schema_qualified_reference_survives_repair(db):
    result = repair_sql("SELECT id FROM ops.audit_log", db)
    assert "ops.audit_log" in result.sql


# --- error hints ----------------------------------------------------------


def test_missing_relation_error_suggests_real_table(db):
    hint = suggest_for_error('relation "customers" does not exist', db)
    assert hint and "Customer" in hint


def test_missing_column_error_suggests_real_column(db):
    hint = suggest_for_error('column "first_name" does not exist', db)
    assert hint is None or "firstName" in hint


# --- overview rendering ---------------------------------------------------


def test_overview_lists_tables_with_sizes(db):
    text = db.overview()
    assert '"Customer"' in text
    assert "rows" in text


def test_describe_includes_columns_and_relationships(db):
    text = db.describe(db.find_table("Invoice"))
    assert "customerId" in text
    assert "relationships" in text


def test_name_columns_prefers_label_like_columns(db):
    names = [c.name for c in db.find_table("Employee").name_columns()]
    assert "firstName" in names
    assert "departmentId" not in names
