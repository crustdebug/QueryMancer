"""The sidebar connection panel.

Credentials typed here live in Streamlit's per-session state and nowhere else:
not in a file, not in an environment variable, not in a module global. Closing
the browser session discards them. The password field is never rendered back
into the page, and everything shown or logged goes through the masking in
connection.py first.
"""

from typing import Optional

import streamlit as st

import schema as schema_module
import session as session_module
from config import Config
from connection import DEFAULT_ENGINE, ENGINES, ConnectionSettings

_FORM_ENGINE = "_form_engine"
_STATUS = "_connection_status"


def _remember_status(ok: bool, message: str) -> None:
    st.session_state[_STATUS] = (ok, message)


def _show_status() -> None:
    status = st.session_state.get(_STATUS)
    if not status:
        return
    ok, message = status
    if ok:
        st.success(message, icon="✅")
    else:
        st.error(message, icon="⚠️")


def _prefill() -> Optional[ConnectionSettings]:
    """Pre-fill the form from DATABASE_URL if one was provided, else nothing.

    This is a convenience for local development only. The value is used to fill
    the form, not stored as a credential.
    """
    url = Config.DEFAULT_DATABASE_URL
    if not url:
        return None
    try:
        return ConnectionSettings.from_url(url)
    except ValueError:
        return None


def _connect(settings: ConnectionSettings) -> None:
    """Attempt the connection and record the outcome for display."""
    schema_module.clear_cache()
    ok, message = session_module.connect(settings)
    _remember_status(ok, message)
    if ok:
        # Drop cached UI data belonging to the previous database.
        st.cache_data.clear()


def render_connection_panel() -> bool:
    """Draw the connection controls. Returns True when a database is connected."""
    connection = session_module.current_connection()

    if connection is not None:
        _render_connected(connection)
        return True

    _render_form()
    return False


def _render_connected(connection) -> None:
    settings = connection.settings
    st.success(f"Connected · {settings.label}", icon="🔌")
    # summary and safe_url both mask the password.
    st.caption(settings.summary)

    databases = connection.list_databases()
    if databases and settings.database in databases and len(databases) > 1:
        choice = st.selectbox(
            "Database",
            databases,
            index=databases.index(settings.database),
            help="Switch to another database on the same server.",
        )
        if choice != settings.database:
            _connect(settings.with_database(choice))
            st.rerun()

    left, right = st.columns(2)
    with left:
        if st.button("Reload schema", use_container_width=True):
            schema_module.clear_cache()
            st.cache_data.clear()
            st.rerun()
    with right:
        if st.button("Disconnect", use_container_width=True):
            session_module.disconnect()
            schema_module.clear_cache()
            st.cache_data.clear()
            st.session_state.pop(_STATUS, None)
            st.rerun()

    _show_status()


def _render_form() -> None:
    st.info("Connect a database to begin.", icon="🔌")
    _show_status()

    prefill = _prefill()
    engine_keys = list(ENGINES)
    default_engine = prefill.engine if prefill else DEFAULT_ENGINE

    engine = st.selectbox(
        "Database type",
        engine_keys,
        index=engine_keys.index(st.session_state.get(_FORM_ENGINE, default_engine)),
        format_func=lambda key: ENGINES[key]["label"],
        key=_FORM_ENGINE,
    )

    tab_fields, tab_url = st.tabs(["Enter details", "Connection string"])

    with tab_fields:
        _render_field_form(engine, prefill)

    with tab_url:
        _render_url_form()

    st.caption(
        "Credentials stay in this browser session only. They are never written "
        "to disk and never appear in logs."
    )


def _render_field_form(engine: str, prefill: Optional[ConnectionSettings]) -> None:
    spec = ENGINES[engine]
    use_prefill = prefill is not None and prefill.engine == engine

    with st.form(f"connect_fields_{engine}"):
        if engine == "sqlite":
            database = st.text_input(
                "Database file",
                value=prefill.database if use_prefill else "",
                placeholder="C:\\path\\to\\database.db",
            )
            settings = ConnectionSettings(engine=engine, database=database.strip())
        else:
            host = st.text_input(
                "Host", value=prefill.host if use_prefill else "localhost"
            )
            port = st.number_input(
                "Port",
                min_value=1,
                max_value=65535,
                value=int(
                    (prefill.port if use_prefill and prefill.port else spec["default_port"])
                    or 5432
                ),
                step=1,
            )
            database = st.text_input(
                "Database", value=prefill.database if use_prefill else ""
            )
            user = st.text_input(
                "User", value=prefill.username if use_prefill else ""
            )
            # type="password" keeps the value out of the rendered DOM as text.
            password = st.text_input("Password", type="password")
            settings = ConnectionSettings(
                engine=engine,
                host=host.strip(),
                port=int(port),
                database=database.strip(),
                username=user.strip(),
                password=password,
            )

        if st.form_submit_button("Connect", use_container_width=True):
            _connect(settings)
            st.rerun()

    driver = spec.get("package")
    if driver:
        st.caption(f"Requires the `{driver}` package.")


def _render_url_form() -> None:
    with st.form("connect_url"):
        url = st.text_input(
            "Connection string",
            type="password",  # it embeds the password
            placeholder="postgresql://user:password@host:5432/database",
            help=(
                "Supported: postgresql://, mysql://, sqlite:///path/to.db, "
                "mssql://, oracle://"
            ),
        )
        if st.form_submit_button("Connect", use_container_width=True):
            try:
                _connect(ConnectionSettings.from_url(url))
            except ValueError as error:
                _remember_status(False, str(error))
            st.rerun()
