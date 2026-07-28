"""Tests for the question/answer cache."""

import time

import pytest

from answer_cache import AnswerCache, CachedAnswer, database_identity, make_key
from connection import ConnectionSettings

SECRET = "TopSecret-123!"

DB_A = "postgresql://user@host:5432/sales"
DB_B = "postgresql://user@host:5432/hr"


def _answer(text="42 orders"):
    return CachedAnswer(text=text, sql="SELECT count(*) FROM orders")


def test_a_repeated_question_is_served_from_the_cache():
    cache = AnswerCache()
    cache.put("How many orders?", DB_A, _answer())
    hit = cache.get("How many orders?", DB_A)
    assert hit is not None and hit.text == "42 orders"
    assert cache.hits == 1


def test_an_unseen_question_is_a_miss():
    cache = AnswerCache()
    assert cache.get("Never asked", DB_A) is None
    assert cache.misses == 1


@pytest.mark.parametrize(
    "variant",
    ["how many orders?", "  How   many orders?  ", "HOW MANY ORDERS?"],
)
def test_case_and_whitespace_variants_share_an_entry(variant):
    """Re-typing a question slightly differently should still hit."""
    cache = AnswerCache()
    cache.put("How many orders?", DB_A, _answer())
    assert cache.get(variant, DB_A) is not None


def test_the_same_question_against_a_different_database_is_a_miss():
    """The whole point of keying on the database: staging must not answer
    for production."""
    cache = AnswerCache()
    cache.put("How many orders?", DB_A, _answer("42 orders"))
    assert cache.get("How many orders?", DB_B) is None


def test_entries_expire_after_the_ttl():
    cache = AnswerCache(ttl_seconds=0.05)
    cache.put("q", DB_A, _answer())
    assert cache.get("q", DB_A) is not None
    time.sleep(0.08)
    assert cache.get("q", DB_A) is None


def test_the_cache_is_bounded_and_evicts_the_least_recently_used():
    cache = AnswerCache(max_entries=2)
    cache.put("first", DB_A, _answer("1"))
    cache.put("second", DB_A, _answer("2"))
    # Touch 'first' so 'second' becomes the least recently used.
    assert cache.get("first", DB_A) is not None
    cache.put("third", DB_A, _answer("3"))

    assert cache.get("first", DB_A) is not None
    assert cache.get("third", DB_A) is not None
    assert cache.get("second", DB_A) is None


def test_a_zero_ttl_disables_the_cache_entirely():
    cache = AnswerCache(ttl_seconds=0)
    cache.put("q", DB_A, _answer())
    assert cache.get("q", DB_A) is None
    assert not cache.enabled


def test_keys_are_distinct_when_the_boundary_between_parts_moves():
    """Concatenating question and database without a separator would make
    ('ab', 'c') and ('a', 'bc') collide."""
    assert make_key("ab", "c") != make_key("a", "bc")


def test_the_key_does_not_contain_the_question_or_the_database():
    """Keys are digests, so the cache cannot be read back to recover what
    anyone asked or where."""
    key = make_key("What is the revenue for Acme?", DB_A)
    assert "Acme" not in key
    assert "revenue" not in key
    assert "sales" not in key


def test_database_identity_carries_no_password():
    settings = ConnectionSettings(
        engine="postgresql", host="db.internal", port=5432,
        database="erp", username="reporting", password=SECRET,
    )
    identity = database_identity(settings)
    assert SECRET not in identity
    # It must still distinguish databases on the same server.
    other = database_identity(settings.with_database("staging"))
    assert identity != other


def test_invalidate_clears_entries():
    cache = AnswerCache()
    cache.put("q", DB_A, _answer())
    assert cache.invalidate() == 1
    assert cache.get("q", DB_A) is None


def test_stats_report_hits_and_misses():
    cache = AnswerCache()
    cache.put("q", DB_A, _answer())
    cache.get("q", DB_A)
    cache.get("other", DB_A)
    stats = cache.stats()
    assert stats["hits"] == 1 and stats["misses"] == 1 and stats["entries"] == 1
