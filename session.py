"""Per-session database connection state.

Credentials belong to one browser session and must not leak into another, so
they are never stored in a module-level global, never written to disk, and never
placed in an environment variable. Streamlit's `session_state` is the store; this
module is the only thing that touches it, so the rules live in one place.

The tools call `current_connection()` rather than receiving a connection
argument, because LangChain tool signatures are part of the model-visible schema
and a connection handle has no business appearing there.
"""

import threading
from typing import Optional

from connection import ConnectionSettings, DatabaseConnection

_SESSION_KEY = "_db_connection"

# Used only when running outside Streamlit - tests and check_setup.py. A real
# web session never reads this.
_fallback: Optional[DatabaseConnection] = None
_fallback_lock = threading.Lock()


def _session_state():
    """Streamlit's session state, or None when not running in Streamlit."""
    try:
        import streamlit as st

        # Touching session_state outside a script run raises; treat that as
        # "not in a session" rather than letting it propagate.
        _ = st.session_state
        return st.session_state
    except Exception:  # noqa: BLE001
        return None


def set_connection(connection: Optional[DatabaseConnection]) -> None:
    """Install the connection for this session, disposing any previous one."""
    global _fallback

    previous = current_connection()
    if previous is not None and previous is not connection:
        previous.dispose()

    state = _session_state()
    if state is not None:
        state[_SESSION_KEY] = connection
    else:
        with _fallback_lock:
            _fallback = connection


def current_connection() -> Optional[DatabaseConnection]:
    """The connection for this session, or None if not connected yet."""
    state = _session_state()
    if state is not None:
        return state.get(_SESSION_KEY)
    with _fallback_lock:
        return _fallback


def require_connection() -> DatabaseConnection:
    """The current connection, or a clear error telling the user to connect."""
    connection = current_connection()
    if connection is None:
        raise NotConnected(
            "No database is connected. Open the sidebar, choose your database "
            "type, enter the connection details and select Connect."
        )
    return connection


def connect(settings: ConnectionSettings) -> tuple:
    """Validate and install a connection. Returns (ok, message).

    The connection is only installed if it actually works, so a failed attempt
    never replaces a working session.
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
    set_connection(None)


class NotConnected(RuntimeError):
    """Raised when a tool runs before a database has been connected."""
