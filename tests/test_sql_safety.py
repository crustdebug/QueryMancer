"""Tests for read-only enforcement and the table allowlist."""

import pytest

import tools
from config import Config


@pytest.fixture
def allowlist(monkeypatch):
    monkeypatch.setattr(Config, "ALLOWED_TABLES", ["customer_invoice", "customer_profile"])
    return Config.ALLOWED_TABLES


# --- read-only enforcement ------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "SELECT 1",
        "select id from customer_invoice limit 5",
        "WITH recent AS (SELECT 1 AS n) SELECT n FROM recent",
        "  SELECT count(*) FROM customer_invoice;  ",
    ],
)
def test_read_only_accepts_select_and_cte(query):
    assert tools.check_read_only(query) is None


@pytest.mark.parametrize(
    "query",
    [
        "DELETE FROM customer_invoice",
        "UPDATE customer_invoice SET amount = 0",
        "DROP TABLE customer_invoice",
        "INSERT INTO customer_invoice VALUES (1)",
        "TRUNCATE customer_invoice",
        "SELECT 1; DROP TABLE customer_invoice",
    ],
)
def test_read_only_rejects_writes_and_stacked_statements(query):
    assert tools.check_read_only(query) is not None


def test_column_named_like_a_keyword_is_not_rejected():
    """The old substring blocklist rejected 'updated_by' because it contains UPDATE."""
    assert tools.check_read_only("SELECT updated_by, created_at FROM customer_invoice") is None


def test_write_keyword_inside_a_string_literal_is_ignored():
    query = "SELECT id FROM customer_invoice WHERE note = 'please DELETE this row'"
    assert tools.check_read_only(query) is None


def test_comment_cannot_smuggle_a_write():
    assert tools.check_read_only("SELECT 1 -- DROP TABLE customer_invoice") is None


# --- allowlist enforcement ------------------------------------------------


def test_allowed_table_passes(allowlist):
    assert tools.check_table_allowed("customer_invoice") is None


def test_disallowed_table_is_rejected(allowlist):
    assert tools.check_table_allowed("secret_payroll") is not None


def test_allowlist_is_case_insensitive(allowlist):
    assert tools.check_table_allowed("Customer_Invoice") is None


def test_query_referencing_forbidden_table_is_rejected(allowlist):
    error = tools.check_query_allowed("SELECT * FROM secret_payroll")
    assert error is not None and "secret_payroll" in error


def test_join_to_forbidden_table_is_rejected(allowlist):
    query = (
        "SELECT a.id FROM customer_invoice a "
        "JOIN secret_payroll b ON a.emp_id = b.id"
    )
    error = tools.check_query_allowed(query)
    assert error is not None and "secret_payroll" in error


def test_join_between_allowed_tables_passes(allowlist):
    query = (
        "SELECT a.id FROM customer_invoice a "
        "JOIN customer_profile b ON a.customer_id = b.id"
    )
    assert tools.check_query_allowed(query) is None


def test_cte_name_is_not_mistaken_for_a_forbidden_table(allowlist):
    query = (
        "WITH totals AS (SELECT customer_id, sum(amount) s FROM customer_invoice GROUP BY 1) "
        "SELECT * FROM totals"
    )
    assert tools.check_query_allowed(query) is None


def test_schema_qualified_table_is_resolved(allowlist):
    assert tools.check_query_allowed("SELECT * FROM public.customer_invoice") is None
    assert tools.check_query_allowed("SELECT * FROM public.secret_payroll") is not None


def test_quoted_pascal_case_table_is_matched(monkeypatch):
    """Postgres preserves case for quoted identifiers; matching must be case-folded."""
    monkeypatch.setattr(Config, "ALLOWED_TABLES", ["Employee", "Invoice"])
    assert tools.check_table_allowed("Employee") is None
    assert tools.check_query_allowed('SELECT id FROM "Employee"') is None
    assert tools.check_query_allowed('SELECT id FROM "PayrollRun"') is not None


def test_empty_allowlist_permits_everything(monkeypatch):
    monkeypatch.setattr(Config, "ALLOWED_TABLES", [])
    assert tools.check_table_allowed("anything") is None
    assert tools.check_query_allowed("SELECT * FROM anything") is None


# --- output shaping -------------------------------------------------------


def test_truncate_caps_long_output():
    text = "x" * 10_000
    result = tools.truncate(text, limit=100)
    assert len(result) < 200
    assert "truncated" in result


def test_format_rows_renders_a_header_and_count():
    out = tools.format_rows(["id", "name"], [(1, "Acme"), (2, None)])
    assert "id | name" in out
    assert "(2 row(s))" in out
