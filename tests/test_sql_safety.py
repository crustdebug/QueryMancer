"""Tests for read-only enforcement and output shaping."""

import pytest

import tools


@pytest.mark.parametrize(
    "query",
    [
        "SELECT 1",
        "select id from customer_invoice limit 5",
        "WITH recent AS (SELECT 1 AS n) SELECT n FROM recent",
        "  SELECT count(*) FROM customer_invoice;  ",
        'SELECT "firstName" FROM "Employee"',
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
        "GRANT ALL ON customer_invoice TO PUBLIC",
    ],
)
def test_read_only_rejects_writes_and_stacked_statements(query):
    assert tools.check_read_only(query) is not None


def test_column_named_like_a_keyword_is_not_rejected():
    """A substring blocklist would reject 'updated_by' for containing UPDATE."""
    assert tools.check_read_only("SELECT updated_by, created_at FROM invoice") is None


def test_write_keyword_inside_a_string_literal_is_ignored():
    query = "SELECT id FROM invoice WHERE note = 'please DELETE this row'"
    assert tools.check_read_only(query) is None


def test_comment_cannot_smuggle_a_write():
    assert tools.check_read_only("SELECT 1 -- DROP TABLE invoice") is None


# --- output shaping -------------------------------------------------------


def test_truncate_caps_long_output():
    result = tools.truncate("x" * 10_000, limit=100)
    assert len(result) < 200
    assert "truncated" in result


def test_format_rows_renders_a_header_and_count():
    out = tools.format_rows(["id", "name"], [(1, "Acme"), (2, None)])
    assert "id | name" in out
    assert "(2 row(s))" in out


def test_format_rows_handles_empty_result():
    assert tools.format_rows(["id"], []) == "No rows returned."
