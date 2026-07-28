"""Tests for API key rotation and failover."""

import pytest

from key_pool import KeyPool, PoolExhausted, _classify


class RateLimited(Exception):
    pass


def test_round_robin_spreads_calls_across_keys():
    pool = KeyPool(provider="test", keys=["a", "b", "c"])
    used = [pool.run(lambda key: key) for _ in range(6)]
    assert used == ["a", "b", "c", "a", "b", "c"]


def test_rate_limited_key_is_benched_and_next_key_used():
    pool = KeyPool(provider="test", keys=["a", "b"], cooldown_seconds=60)

    def call(key: str) -> str:
        if key == "a":
            raise RateLimited("429 quota exceeded for this key")
        return key

    assert pool.run(call) == "b"
    # The failing key is benched, so the healthy one serves the next call too.
    assert pool.run(call) == "b"
    assert pool.seconds_until_free() is not None


def test_pool_exhausted_when_every_key_is_rate_limited():
    pool = KeyPool(provider="test", keys=["a", "b"], cooldown_seconds=60)

    def always_limited(key: str):
        raise RateLimited("rate limit reached")

    with pytest.raises(PoolExhausted) as excinfo:
        pool.run(always_limited)
    assert excinfo.value.provider == "test"
    assert excinfo.value.retry_after is not None


def test_auth_failure_retires_key_permanently():
    pool = KeyPool(provider="test", keys=["bad", "good"])
    attempts = []

    def call(key: str) -> str:
        attempts.append(key)
        if key == "bad":
            raise Exception("API key not valid. Please pass a valid API key.")
        return key

    assert pool.run(call) == "good"
    # The retired key must not be handed out again.
    assert pool.run(call) == "good"
    assert attempts.count("bad") == 1


def test_non_quota_errors_propagate_instead_of_burning_keys():
    pool = KeyPool(provider="test", keys=["a", "b", "c"])
    attempts = []

    def call(key: str):
        attempts.append(key)
        raise ValueError("malformed request body")

    with pytest.raises(ValueError):
        pool.run(call)
    # Rotating would not help for a bad request, so only one key is spent.
    assert len(attempts) == 1


def test_cooldown_expiry_returns_key_to_rotation(monkeypatch):
    pool = KeyPool(provider="test", keys=["a"], cooldown_seconds=30)

    def limited(key: str):
        raise RateLimited("429")

    with pytest.raises(PoolExhausted):
        pool.run(limited)

    # Advance the clock past the cooldown.
    import key_pool

    real_monotonic = key_pool.time.monotonic
    monkeypatch.setattr(key_pool.time, "monotonic", lambda: real_monotonic() + 31)
    assert pool.run(lambda key: key) == "a"


@pytest.mark.parametrize(
    "message,expected",
    [
        ("429 Too Many Requests", "rate_limit"),
        ("RESOURCE_EXHAUSTED: quota", "rate_limit"),
        ("API key not valid", "auth"),
        ("401 Unauthorized", "auth"),
        ("connection reset by peer", "other"),
    ],
)
def test_error_classification(message, expected):
    assert _classify(Exception(message)) == expected


def test_keys_are_redacted_in_stats():
    secret = "fake-key-" + "0123456789abcdef"
    pool = KeyPool(provider="test", keys=[secret])
    rendered = str(pool.stats())
    assert secret not in rendered
