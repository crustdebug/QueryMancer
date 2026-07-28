"""Round-robin API key rotation with rate-limit awareness.

Free API tiers are capped per key, not per user, so several keys from different
projects multiply the usable quota. This module spreads calls across a pool and
takes a key out of service when the provider says it is over quota.

Three failure modes are distinguished, because they deserve different handling:

  * A rate limit (HTTP 429, "quota exceeded") is temporary. The key is benched
    for a cooldown and returns to the rotation afterwards.
  * An auth failure (HTTP 401/403, "API key not valid") is permanent for this
    process. The key is retired immediately so it is never tried again.
  * A transient server error (HTTP 503, "overloaded", "unavailable") is not
    the key's fault at all - retrying the same key would likely fail the same
    way. It is benched briefly anyway so the pool moves on to another key (or,
    once the pool is exhausted, RotatingChatModel falls back to the next
    model) rather than crashing the request outright.

Anything else is a genuine error and is raised to the caller rather than being
silently retried against every key in the pool.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, TypeVar

from custom_logging import log

T = TypeVar("T")

RATE_LIMIT_MARKERS = (
    "rate limit",
    "ratelimit",
    "429",
    "quota",
    "resource_exhausted",
    "resource exhausted",
    "too many requests",
)

AUTH_FAILURE_MARKERS = (
    "api key not valid",
    "invalid api key",
    "invalid_api_key",
    "api_key_invalid",
    "unauthorized",
    "permission denied",
    "permission_denied",
    "401",
    "403",
)

# A transiently unavailable model/service - not a quota or auth problem, but
# still worth benching briefly so the pool (and, above it, the model-fallback
# chain) moves on instead of the whole request failing outright.
UNAVAILABLE_MARKERS = (
    "503",
    "unavailable",
    "overloaded",
    "high demand",
    "server_error",
    "internal error",
    "try again later",
    "bad gateway",
    "502",
    "504",
    "gateway timeout",
    "service unavailable",
)


def _classify(error: BaseException) -> str:
    """Bucket a provider exception into 'rate_limit', 'auth', 'unavailable', or 'other'."""
    message = f"{type(error).__name__}: {error}".lower()
    # Auth is checked first: an invalid key sometimes also mentions "quota".
    if any(marker in message for marker in AUTH_FAILURE_MARKERS):
        return "auth"
    if any(marker in message for marker in RATE_LIMIT_MARKERS):
        return "rate_limit"
    if any(marker in message for marker in UNAVAILABLE_MARKERS):
        return "unavailable"
    return "other"


def _redact(key: str) -> str:
    """Render a key for logs without exposing it."""
    if len(key) <= 8:
        return "****"
    return f"{key[:4]}...{key[-4:]}"


@dataclass
class _KeyState:
    key: str
    index: int
    blocked_until: float = 0.0
    retired: bool = False
    successes: int = 0
    rate_limit_hits: int = 0
    unavailable_hits: int = 0

    def usable(self, now: float) -> bool:
        return not self.retired and now >= self.blocked_until


@dataclass
class KeyPool:
    """A rotating pool of interchangeable API keys for one provider.

    Thread-safe: Streamlit reruns can touch the same cached pool concurrently.
    """

    provider: str
    keys: List[str]
    cooldown_seconds: float = 60.0
    unavailable_cooldown_seconds: float = 5.0
    _states: List[_KeyState] = field(default_factory=list, init=False)
    _cursor: int = field(default=0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self) -> None:
        self._states = [_KeyState(key=key, index=i) for i, key in enumerate(self.keys)]

    def __len__(self) -> int:
        return len(self._states)

    @property
    def has_capacity(self) -> bool:
        """True if any key could serve a request now or after a cooldown."""
        return any(not state.retired for state in self._states)

    def _next_usable(self) -> Optional[_KeyState]:
        """Take the next usable key, advancing the round-robin cursor."""
        now = time.monotonic()
        total = len(self._states)
        for offset in range(total):
            state = self._states[(self._cursor + offset) % total]
            if state.usable(now):
                # Start the next search after the key we just handed out, so
                # consecutive calls spread across the pool instead of hammering
                # the first healthy key.
                self._cursor = (self._cursor + offset + 1) % total
                return state
        return None

    def seconds_until_free(self) -> Optional[float]:
        """How long until a benched key becomes usable, if any will."""
        now = time.monotonic()
        waits = [
            state.blocked_until - now
            for state in self._states
            if not state.retired and state.blocked_until > now
        ]
        return min(waits) if waits else None

    def run(self, operation: Callable[[str], T]) -> T:
        """Call `operation` with a key, rotating past rate-limited keys.

        Tries each usable key at most once. Raises the last provider error if
        the whole pool is exhausted.
        """
        if not self._states:
            raise RuntimeError(f"No API keys configured for provider '{self.provider}'.")

        last_error: Optional[BaseException] = None
        attempts = 0

        while attempts < len(self._states):
            with self._lock:
                state = self._next_usable()
            if state is None:
                break

            attempts += 1
            try:
                result = operation(state.key)
            except BaseException as error:  # noqa: BLE001 - re-raised below
                kind = _classify(error)
                last_error = error
                with self._lock:
                    if kind == "auth":
                        state.retired = True
                        log(
                            f"[red]{self.provider}: key {_redact(state.key)} rejected "
                            f"(auth). Retiring it for this session.[/red]"
                        )
                    elif kind == "rate_limit":
                        state.rate_limit_hits += 1
                        state.blocked_until = time.monotonic() + self.cooldown_seconds
                        log(
                            f"[yellow]{self.provider}: key {_redact(state.key)} rate-limited. "
                            f"Benched {self.cooldown_seconds:.0f}s, rotating to next key.[/yellow]"
                        )
                    elif kind == "unavailable":
                        state.unavailable_hits += 1
                        # Much shorter than the rate-limit cooldown: an
                        # overloaded model typically clears in seconds, and
                        # the point here is mainly to let the pool - and above
                        # it, the model fallback chain - move on immediately
                        # rather than hammering the same unavailable model.
                        state.blocked_until = time.monotonic() + self.unavailable_cooldown_seconds
                        log(
                            f"[yellow]{self.provider}: key {_redact(state.key)} hit a "
                            f"temporarily unavailable model. Rotating to next key.[/yellow]"
                        )
                    else:
                        # Not a quota problem - rotating keys would not help.
                        raise
                continue

            with self._lock:
                state.successes += 1
            return result

        raise PoolExhausted(
            provider=self.provider,
            pool_size=len(self._states),
            retry_after=self.seconds_until_free(),
        ) from last_error

    def stats(self) -> List[dict]:
        """Per-key counters, for display in the UI."""
        now = time.monotonic()
        rows = []
        for state in self._states:
            if state.retired:
                status = "retired"
            elif state.blocked_until > now:
                status = f"cooling {state.blocked_until - now:.0f}s"
            else:
                status = "ready"
            rows.append(
                {
                    "key": _redact(state.key),
                    "status": status,
                    "calls": state.successes,
                    "rate_limits": state.rate_limit_hits,
                    "unavailable": state.unavailable_hits,
                }
            )
        return rows


class PoolExhausted(RuntimeError):
    """Every key for a provider is rate-limited or retired."""

    def __init__(self, provider: str, pool_size: int, retry_after: Optional[float] = None):
        self.provider = provider
        self.pool_size = pool_size
        self.retry_after = retry_after
        detail = (
            f"retry in about {retry_after:.0f}s"
            if retry_after is not None
            else "no keys left in rotation"
        )
        super().__init__(
            f"All {pool_size} key(s) for '{provider}' are unavailable ({detail})."
        )
