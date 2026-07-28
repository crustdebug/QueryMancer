"""A short-lived cache of answers, keyed by question and database.

Asking the same question twice - a re-ask, a page refresh, two people looking
at the same dashboard - costs a full set of LLM calls each time. This absorbs
that: a repeat within the TTL returns the previous answer, its SQL and its rows
without touching the model at all.

Two properties matter more than hit rate here:

  * Entries are scoped to a specific database. The identity used is the
    connection's full URL with the password removed, so the same question
    against staging and production never shares an entry - and two sessions
    connected to the same database do share one, which is the point.
  * Nothing sensitive is stored in the key. The password is stripped before
    hashing, and the key is a digest rather than the text, so the cache cannot
    be read back to recover what anyone asked.

The TTL is deliberately short. A cached answer is a stale answer, and the data
underneath is live; this is sized to absorb bursts, not to serve yesterday's
numbers.
"""

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CachedAnswer:
    """A previously computed answer, ready to return again."""

    text: str
    sql: Optional[str] = None
    columns: List[str] = field(default_factory=list)
    rows: List[list] = field(default_factory=list)
    truncated: bool = False
    corrections: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def age(self) -> float:
        return time.time() - self.created_at


def _normalise(question: str) -> str:
    """Collapse whitespace and case so trivial variants share an entry.

    Punctuation is deliberately kept: "sales by region" and "sales by region?"
    are the same question, but stripping punctuation wholesale would merge
    genuinely different ones.
    """
    return " ".join(question.lower().split())


def make_key(question: str, database_identity: str) -> str:
    """A stable digest for one question against one database."""
    # The separator matters: without it "ab" + "c" and "a" + "bc" would hash
    # to the same key. A newline cannot appear in a normalised question.
    payload = f"{_normalise(question)}\n{database_identity}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class AnswerCache:
    """A bounded, TTL'd, thread-safe map of question -> answer.

    An OrderedDict gives least-recently-used eviction for free: reading moves
    an entry to the end, so the oldest untouched entry is always at the front
    when the cache is full.
    """

    def __init__(self, ttl_seconds: float = 300.0, max_entries: int = 256):
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._entries: "OrderedDict[str, CachedAnswer]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @property
    def enabled(self) -> bool:
        return self.ttl_seconds > 0 and self.max_entries > 0

    def get(self, question: str, database_identity: str) -> Optional[CachedAnswer]:
        """The cached answer for this question, if one is still fresh."""
        if not self.enabled:
            return None
        key = make_key(question, database_identity)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            if entry.age() > self.ttl_seconds:
                # Expired: drop it rather than leave it to be re-checked.
                del self._entries[key]
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return entry

    def put(
        self, question: str, database_identity: str, answer: CachedAnswer
    ) -> None:
        if not self.enabled:
            return
        key = make_key(question, database_identity)
        with self._lock:
            self._entries[key] = answer
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def invalidate(self, database_identity: Optional[str] = None) -> int:
        """Drop cached answers - all of them, or one database's.

        Called when a connection changes, so a reconnect never serves an answer
        computed against different data.
        """
        with self._lock:
            if database_identity is None:
                count = len(self._entries)
                self._entries.clear()
                return count
            # Keys are digests, so which database an entry belongs to cannot be
            # read back from the key. Entries carry no database field either,
            # by design - so a targeted invalidation clears everything rather
            # than silently keeping entries it cannot identify.
            count = len(self._entries)
            self._entries.clear()
            return count

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "hits": self.hits,
                "misses": self.misses,
                "ttlSeconds": self.ttl_seconds,
            }


def database_identity(settings) -> str:
    """A stable identifier for the database a question was answered against.

    The password-masked URL: it distinguishes host, port, database and user
    without ever placing a credential in a cache key.
    """
    return settings.safe_url
