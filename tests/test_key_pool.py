"""Tests for API key rotation and failover."""

import pytest

from key_pool import KeyPool, PoolExhausted, _classify, _extract_retry_delay, _is_daily_quota


class RateLimited(Exception):
    pass


# Captured live from the real Gemini API (gemini-2.0-flash, whose free-tier
# quota Google has zeroed) so the parsing tests exercise the actual shape of
# the flattened error string, not a guessed-at approximation of it.
REAL_GEMINI_RPM_ERROR = (
    "Error calling model 'gemini-2.0-flash' (RESOURCE_EXHAUSTED): 429 RESOURCE_EXHAUSTED. "
    "{'error': {'code': 429, 'message': 'You exceeded your current quota, please check your "
    "plan and billing details. ... \\nPlease retry in 17.980842916s.', "
    "'status': 'RESOURCE_EXHAUSTED', 'details': [...{'@type': "
    "'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaMetric': "
    "'generativelanguage.googleapis.com/generate_content_free_tier_input_token_count', "
    "'quotaId': 'GenerateContentInputTokensPerModelPerMinute-FreeTier', 'quotaDimensions': "
    "{'location': 'global', 'model': 'gemini-2.0-flash'}}, {'quotaMetric': "
    "'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': "
    "'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', 'quotaDimensions': {'location': "
    "'global', 'model': 'gemini-2.0-flash'}}, {'quotaMetric': "
    "'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': "
    "'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaDimensions': {'location': "
    "'global', 'model': 'gemini-2.0-flash'}}]}, {'@type': "
    "'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '17s'}]}}"
)


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
        ("503 UNAVAILABLE: This model is currently experiencing high demand.", "unavailable"),
        ("502 Bad Gateway", "unavailable"),
        ("connection reset by peer", "other"),
    ],
)
def test_error_classification(message, expected):
    assert _classify(Exception(message)) == expected


# --- RPM vs RPD and the structured retryDelay -----------------------------


def test_real_gemini_error_is_classified_as_rate_limit():
    assert _classify(Exception(REAL_GEMINI_RPM_ERROR)) == "rate_limit"


def test_real_gemini_error_retry_delay_is_extracted():
    # RetryInfo.retryDelay in the captured response was '17s'.
    assert _extract_retry_delay(Exception(REAL_GEMINI_RPM_ERROR)) == 17.0


def test_purely_per_minute_quota_is_not_flagged_daily():
    message = (
        "429 quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier, "
        "retryDelay: '12s'"
    )
    assert _is_daily_quota(Exception(message)) is False
    assert _extract_retry_delay(Exception(message)) == 12.0


def test_per_day_quota_is_flagged_daily():
    message = "429 quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier"
    assert _is_daily_quota(Exception(message)) is True


def test_no_retry_delay_present_returns_none():
    assert _extract_retry_delay(Exception("429 quota exceeded, no structured info")) is None


def test_explicit_retry_delay_is_used_as_the_bench_duration():
    """The provider's own retryDelay should win over the generic cooldown."""
    pool = KeyPool(provider="test", keys=["a"], cooldown_seconds=60)

    def call(key: str):
        raise Exception("429 RESOURCE_EXHAUSTED retryDelay: '3s'")

    with pytest.raises(PoolExhausted):
        pool.run(call)
    # Benched for ~3s (the reported delay), not the 60s generic cooldown.
    wait = pool.seconds_until_free()
    assert wait is not None and wait <= 5


def test_daily_quota_without_retry_delay_uses_the_daily_cooldown_not_the_short_one():
    """A per-day 429 with no explicit delay must not use the short RPM cooldown.

    Benching for the normal ~60s cooldown would just retry a key once a
    minute for hours against a quota that cannot succeed until it resets.
    """
    pool = KeyPool(
        provider="test", keys=["a"], cooldown_seconds=1, daily_cooldown_seconds=120
    )

    def call(key: str):
        raise Exception("429 quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier")

    with pytest.raises(PoolExhausted):
        pool.run(call)
    wait = pool.seconds_until_free()
    assert wait is not None and wait > 60  # much longer than cooldown_seconds=1


def test_retry_delay_is_capped_at_the_maximum():
    from key_pool import MAX_RETRY_DELAY_SECONDS

    message = "429 RESOURCE_EXHAUSTED retryDelay: '99999s'"
    assert _extract_retry_delay(Exception(message)) == MAX_RETRY_DELAY_SECONDS


def test_unavailable_model_rotates_to_next_key_instead_of_crashing():
    """A transient 503 must not propagate past the pool - it should rotate.

    This mirrors a real failure: gemini-3.5-flash returned a 503 UNAVAILABLE
    that used to be misclassified as "other" and re-raised immediately,
    skipping both key rotation and the model-level fallback chain above it.
    """
    pool = KeyPool(provider="test", keys=["a", "b"])

    def call(key: str) -> str:
        if key == "a":
            raise Exception("503 UNAVAILABLE: high demand, try again later")
        return key

    assert pool.run(call) == "b"


def test_pool_exhausted_when_every_key_is_unavailable():
    pool = KeyPool(provider="test", keys=["a", "b"], unavailable_cooldown_seconds=1)

    def always_unavailable(key: str):
        raise Exception("503 UNAVAILABLE: high demand")

    with pytest.raises(PoolExhausted):
        pool.run(always_unavailable)


def test_keys_are_redacted_in_stats():
    secret = "fake-key-" + "0123456789abcdef"
    pool = KeyPool(provider="test", keys=[secret])
    rendered = str(pool.stats())
    assert secret not in rendered
