"""Tests for FK-subgraph schema pruning.

A large schema does not fit in a prompt, and padding one with irrelevant tables
costs tokens and invites wrong joins. These check that pruning keeps what a
question needs and drops what it does not.
"""

import pytest

from schema import Column, DatabaseSchema, ForeignKey, Table, _keywords, _split_identifier


def _table(name, columns=(), rows=100, schema="public"):
    return Table(
        schema=schema,
        name=name,
        columns=[Column(name=c, data_type="text", nullable=True) for c in columns],
        estimated_rows=rows,
    )


@pytest.fixture
def shop():
    """A small star schema plus unrelated tables that must be pruned away."""
    db = DatabaseSchema(
        tables=[
            _table("customers", ["id", "company", "country"]),
            _table("orders", ["id", "customer_id", "total", "placed_at"]),
            _table("order_items", ["id", "order_id", "product_id", "qty"]),
            _table("products", ["id", "sku", "name", "price"]),
            # Unrelated to the order/customer subgraph.
            _table("audit_log", ["id", "actor", "action"]),
            _table("feature_flags", ["id", "flag", "enabled"]),
            _table("email_templates", ["id", "subject", "body"]),
        ],
        foreign_keys=[
            ForeignKey("public", "orders", "customer_id", "public", "customers", "id"),
            ForeignKey("public", "order_items", "order_id", "public", "orders", "id"),
            ForeignKey("public", "order_items", "product_id", "public", "products", "id"),
        ],
    )
    return db


def _names(tables):
    return {t.name for t in tables}


def test_a_named_table_is_selected(shop):
    assert "customers" in _names(shop.relevant_tables("How many customers are there?"))


def test_plural_and_singular_both_match(shop):
    """People ask about "customers" whether the table is customer or customers."""
    assert "customers" in _names(shop.relevant_tables("list every customer"))
    assert "products" in _names(shop.relevant_tables("show me each product"))


def test_foreign_key_neighbours_are_pulled_in(shop):
    """A question naming only orders still needs customers to group by them."""
    chosen = _names(shop.relevant_tables("what is the total of all orders?"))
    assert "orders" in chosen
    assert "customers" in chosen      # orders -> customers
    assert "order_items" in chosen    # order_items -> orders


def test_unrelated_tables_are_pruned_away(shop):
    chosen = _names(shop.relevant_tables("revenue per customer"))
    assert "feature_flags" not in chosen
    assert "email_templates" not in chosen


def test_a_column_name_can_select_a_table(shop):
    """The question names a column, not a table."""
    assert "products" in _names(shop.relevant_tables("group everything by sku"))


def test_a_question_matching_nothing_returns_empty(shop):
    """So the caller can fall back to the full overview rather than prompt
    the model with an empty schema."""
    assert shop.relevant_tables("what is the weather like today") == []
    assert shop.focused_overview("what is the weather like today") is None


def test_the_pruned_set_is_capped(shop):
    chosen = shop.relevant_tables("customers orders products items", max_tables=3)
    assert len(chosen) <= 3


def test_focused_overview_lists_only_chosen_tables_and_their_keys(shop):
    rendered = shop.focused_overview("revenue per customer")
    assert "orders" in rendered and "customers" in rendered
    assert "feature_flags" not in rendered
    # Foreign keys between the chosen tables are included, for JOINs.
    assert "->" in rendered


def test_focused_overview_says_that_tables_were_hidden(shop):
    """Without this the model can conclude a table it needs does not exist."""
    rendered = shop.focused_overview("revenue per customer")
    assert "other table(s) exist" in rendered


def test_pruning_actually_shrinks_a_large_schema():
    """The point of the feature: a 100-table database should not send 100
    tables for a question about two of them."""
    tables = [_table(f"unrelated_{i}", ["id", "value"]) for i in range(100)]
    tables += [
        _table("invoices", ["id", "customer_id", "amount"]),
        _table("customers", ["id", "name"]),
    ]
    db = DatabaseSchema(
        tables=tables,
        foreign_keys=[
            ForeignKey("public", "invoices", "customer_id", "public", "customers", "id")
        ],
    )
    chosen = db.relevant_tables("total invoice amount per customer")
    assert {"invoices", "customers"} <= _names(chosen)
    assert len(chosen) < 10
    assert len(db.focused_overview("total invoice amount per customer")) < len(db.overview())


@pytest.mark.parametrize(
    "identifier,expected_subset",
    [
        ("customer_orders", {"customer", "orders"}),
        ("customerOrders", {"customer", "orders"}),
        ("CustomerOrders", {"customer", "orders"}),
    ],
)
def test_identifier_splitting_handles_every_naming_convention(identifier, expected_subset):
    assert expected_subset <= _split_identifier(identifier)


def test_stopwords_are_dropped_from_the_question():
    words = _keywords("Show me all of the customers in the USA")
    assert "customers" in words
    for noise in ("show", "the", "all", "of"):
        assert noise not in words
