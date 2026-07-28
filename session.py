"""Per-session database connections and conversation history.

Credentials belong to one browser session and must not leak into another, so
they are kept in a server-side store keyed by an opaque session id, never
written to disk and never placed in an environment variable. The id travels in
an HttpOnly cookie, so page scripts cannot read it either.

The tools call `require_connection()` rather than taking a connection argument,
because LangChain tool signatures are part of the model-visible schema and a
connection handle has no business appearing there. The active session is bound
to the current thread for the duration of a request.
"""

import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from connection import ConnectionSettings, DatabaseConnection

# How long a session may sit idle before it is discarded along with its
# credentials. Short enough that an abandoned browser tab does not keep a
# database connection alive indefinitely.
SESSION_IDLE_SECONDS = 8 * 60 * 60


@dataclass
class Message:
    role: str  # "user" or "assistant"
    text: str
    sql: Optional[str] = None
    columns: List[str] = field(default_factory=list)
    rows: List[list] = field(default_factory=list)
    truncated: bool = False
    corrections: List[str] = field(default_factory=list)
    error: bool = False

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "text": self.text,
            "sql": self.sql,
            "columns": self.columns,
            "rows": self.rows,
            "truncated": self.truncated,
            "corrections": self.corrections,
            "error": self.error,
        }


@dataclass
class Conversation:
    id: str
    title: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    messages: List[Message] = field(default_factory=list)
    # The LangChain message history backing this conversation.
    history: list = field(default_factory=list)
    # Which database answered this conversation, captured when it was created.
    # A session can reconnect to a different database while older conversations
    # remain in the sidebar, so this is stored per conversation rather than read
    # from the live connection - otherwise revisiting an old thread would label
    # it with whatever database happens to be attached now.
    #
    # Display strings only (name and engine label). No host, user, or password:
    # these travel to the browser, and connection details are not the client's
    # business. See _connection_state in server.py.
    database: str = ""
    engine: str = ""
    engine_label: str = ""

    def summary(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "updated_at": self.updated_at,
            "message_count": len(self.messages),
            "database": self.database,
            "engine": self.engine,
            "engineLabel": self.engine_label,
        }

    def to_dict(self) -> dict:
        return {
            **self.summary(),
            "messages": [m.to_dict() for m in self.messages],
        }


@dataclass
class Session:
    id: str
    connection: Optional[DatabaseConnection] = None
    conversations: Dict[str, Conversation] = field(default_factory=dict)
    order: List[str] = field(default_factory=list)
    last_seen: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_seen = time.time()

    def new_conversation(self, title: str = "New question") -> Conversation:
        conversation = Conversation(id=secrets.token_urlsafe(9), title=title)
        # Stamp the database now, while we know which one is attached. Doing it
        # here rather than at the call sites means every way of starting a
        # conversation records it, including future ones.
        settings = self.connection.settings if self.connection is not None else None
        if settings is not None:
            conversation.database = settings.display_name
            conversation.engine = settings.engine
            conversation.engine_label = settings.label
        self.conversations[conversation.id] = conversation
        self.order.insert(0, conversation.id)
        return conversation

    def ordered_conversations(self) -> List[Conversation]:
        return [self.conversations[i] for i in self.order if i in self.conversations]

    def close(self) -> None:
        if self.connection is not None:
            self.connection.dispose()
            self.connection = None


class SessionStore:
    """In-memory session registry. Nothing here is persisted."""

    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self) -> Session:
        # 32 bytes of entropy: the id is the only thing guarding one user's
        # connection from another, so it must not be guessable.
        session = Session(id=secrets.token_urlsafe(32))
        with self._lock:
            self._sessions[session.id] = session
        return session

    def get(self, session_id: Optional[str]) -> Optional[Session]:
        if not session_id:
            return None
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if time.time() - session.last_seen > SESSION_IDLE_SECONDS:
                # Expired: drop it and its credentials.
                del self._sessions[session_id]
                session.close()
                return None
            session.touch()
            return session

    def get_or_create(self, session_id: Optional[str]) -> Session:
        return self.get(session_id) or self.create()

    def drop(self, session_id: Optional[str]) -> None:
        if not session_id:
            return
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is not None:
            session.close()

    def sweep(self) -> int:
        """Discard idle sessions. Returns how many were removed."""
        cutoff = time.time() - SESSION_IDLE_SECONDS
        with self._lock:
            stale = [i for i, s in self._sessions.items() if s.last_seen < cutoff]
            removed = [self._sessions.pop(i) for i in stale]
        for session in removed:
            session.close()
        return len(removed)


store = SessionStore()

# The session serving the current request, bound per thread so concurrent
# requests never observe each other's connection.
_active = threading.local()


class _ActiveSession:
    """Binds a session to this thread for the duration of a block."""

    def __init__(self, session: Optional[Session]):
        self.session = session
        self.previous = None

    def __enter__(self):
        self.previous = getattr(_active, "session", None)
        _active.session = self.session
        return self.session

    def __exit__(self, *exc):
        _active.session = self.previous
        return False


def use_session(session: Optional[Session]) -> _ActiveSession:
    return _ActiveSession(session)


def current_session() -> Optional[Session]:
    return getattr(_active, "session", None)


def current_connection() -> Optional[DatabaseConnection]:
    """The connection for the active session, or None if not connected."""
    session = current_session()
    return session.connection if session else None


def require_connection() -> DatabaseConnection:
    """The current connection, or a clear error telling the user to connect."""
    connection = current_connection()
    if connection is None:
        raise NotConnected(
            "No database is connected. Choose your database type and enter "
            "your connection details to get started."
        )
    return connection


def set_connection(connection: Optional[DatabaseConnection]) -> None:
    """Install a connection on the active session, disposing any previous one."""
    session = current_session()
    if session is None:
        raise NotConnected("No active session.")
    if session.connection is not None and session.connection is not connection:
        session.connection.dispose()
    session.connection = connection


def connect(settings: ConnectionSettings) -> tuple:
    """Validate and install a connection. Returns (ok, message).

    The connection is installed only if it actually works, so a failed attempt
    never replaces a working one.
    """
    complete, why = settings.is_complete()
    if not complete:
        return False, why

    candidate = DatabaseConnection(settings)
    ok, message = candidate.test()
    if not ok:
        candidate.dispose()
        return False, message

    set_connection(candidate)
    return True, message


def disconnect() -> None:
    """Drop the connection and forget the credentials."""
    session = current_session()
    if session is not None and session.connection is not None:
        session.connection.dispose()
        session.connection = None


class NotConnected(RuntimeError):
    """Raised when a tool runs before a database has been connected."""
