"""Tests for connection settings, credential safety and multi-engine support.

The privacy tests are the important ones here: a password must never reach a
log, an error message, a rendered URL or a repr.
"""

import os
import sqlite3
import tempfile
import time

import pytest

import schema as schema_module
import session
from connection import ENGINES, ConnectionSettings, DatabaseConnection, sanitize

SECRET = "hunter2-VerySecret!"


# --- URL parsing ----------------------------------------------------------


@pytest.mark.parametrize(
    "url,engine",
    [
        ("postgresql://u:p@h:5432/db", "postgresql"),
        ("postgres://u:p@h/db", "postgresql"),
        ("postgresql+psycopg2://u:p@h:5432/db", "postgresql"),
        ("mysql://u:p@h:3306/db", "mysql"),
        ("mysql+pymysql://u:p@h/db", "mysql"),
        ("mariadb://u:p@h/db", "mysql"),
        ("mssql://u:p@h/db", "mssql"),
        ("oracle://u:p@h/db", "oracle"),
    ],
)
def test_common_url_schemes_are_recognised(url, engine):
    assert ConnectionSettings.from_url(url).engine == engine


def test_url_fields_are_parsed():
    settings = ConnectionSettings.from_url("postgresql://alice:pw@db.example.com:6543/sales")
    assert settings.host == "db.example.com"
    assert settings.port == 6543
    assert settings.database == "sales"
    assert settings.username == "alice"
    assert settings.password == "pw"


def test_missing_port_falls_back_to_the_engine_default():
    assert ConnectionSettings.from_url("mysql://u:p@h/db").port == 3306


def test_sqlite_url_is_treated_as_a_path():
    settings = ConnectionSettings.from_url("sqlite:///C:/data/app.db")
    assert settings.engine == "sqlite"
    assert settings.database == "C:/data/app.db"


def test_unsupported_engine_is_rejected_clearly():
    with pytest.raises(ValueError, match="Unsupported database type"):
        ConnectionSettings.from_url("mongodb://h/db")


def test_empty_url_is_rejected():
    with pytest.raises(ValueError):
        ConnectionSettings.from_url("   ")


def test_url_without_scheme_is_rejected():
    with pytest.raises(ValueError):
        ConnectionSettings.from_url("localhost:5432/db")


# --- credential safety ----------------------------------------------------


@pytest.fixture
def secret_settings():
    return ConnectionSettings(
        engine="postgresql",
        host="db.internal",
        port=5432,
        database="prod",
        username="admin",
        password=SECRET,
    )


def test_password_is_masked_in_safe_url(secret_settings):
    assert SECRET not in secret_settings.safe_url
    assert "***" in secret_settings.safe_url


def test_password_is_absent_from_repr(secret_settings):
    assert SECRET not in repr(secret_settings)


def test_password_is_absent_from_summary(secret_settings):
    assert SECRET not in secret_settings.summary


def test_sanitize_removes_password_from_driver_errors(secret_settings):
    leaked = f"FATAL: auth failed for postgresql://admin:{SECRET}@db.internal:5432/prod"
    assert SECRET not in sanitize(leaked, secret_settings)


def test_sanitize_masks_credentials_it_was_not_told_about():
    """Errors may carry a URL we did not construct; mask those too."""
    leaked = "connection refused: mysql://root:someOtherPassword@10.0.0.5:3306/app"
    assert "someOtherPassword" not in sanitize(leaked)


def test_url_includes_the_password_only_when_actually_connecting(secret_settings):
    from urllib.parse import quote_plus

    # The real URL must carry it (the driver needs it), percent-encoded.
    assert quote_plus(SECRET) in secret_settings.url()
    # The display form must not, in either form.
    display = secret_settings.url(hide_password=True)
    assert SECRET not in display and quote_plus(SECRET) not in display


def test_special_characters_in_password_are_encoded():
    settings = ConnectionSettings(
        engine="postgresql", host="h", port=5432, database="d",
        username="u", password="p@ss/word:1",
    )
    # The raw form would break URL parsing if it were not encoded.
    assert "p@ss/word:1" not in settings.url()


# --- validation -----------------------------------------------------------


def test_incomplete_settings_are_reported():
    ok, why = ConnectionSettings(engine="postgresql", host="h").is_complete()
    assert not ok
    assert "database" in why and "user" in why


def test_sqlite_requires_only_a_path():
    assert ConnectionSettings(engine="sqlite", database="x.db").is_complete()[0]
    assert not ConnectionSettings(engine="sqlite").is_complete()[0]


def test_with_database_switches_target_without_touching_credentials(secret_settings):
    switched = secret_settings.with_database("other")
    assert switched.database == "other"
    assert switched.password == secret_settings.password
    assert switched.host == secret_settings.host


def test_every_engine_has_a_label_and_driver():
    for key, spec in ENGINES.items():
        assert spec["label"]
        assert spec["driver"]


# --- live SQLite round trip ----------------------------------------------


@pytest.fixture(autouse=True)
def active_session():
    """Connections belong to a session, so give each test its own."""
    with session.use_session(session.store.create()) as current:
        yield current


@pytest.fixture
def sqlite_db():
    path = os.path.join(tempfile.mkdtemp(), "test.db")
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE Customer (id INTEGER PRIMARY KEY, companyName TEXT);
        CREATE TABLE SalesOrder (
            id INTEGER PRIMARY KEY,
            customerId INTEGER REFERENCES Customer(id),
            totalAmount NUMERIC
        );
        INSERT INTO Customer (companyName) VALUES ('Acme'), ('Globex');
        INSERT INTO SalesOrder (customerId, totalAmount) VALUES (1, 100), (2, 250);
        """
    )
    conn.commit()
    conn.close()
    yield path
    schema_module.clear_cache()
    session.disconnect()


def test_sqlite_connects_and_reports_success(sqlite_db):
    ok, message = session.connect(ConnectionSettings(engine="sqlite", database=sqlite_db))
    assert ok, message


def test_schema_is_discovered_from_sqlite(sqlite_db):
    session.connect(ConnectionSettings(engine="sqlite", database=sqlite_db))
    import tools

    db = tools.get_schema(refresh=True)
    names = {t.name for t in db.tables}
    assert {"Customer", "SalesOrder"} <= names
    assert db.foreign_keys, "foreign keys should be discovered"


def test_sqlite_rejects_writes(sqlite_db):
    session.connect(ConnectionSettings(engine="sqlite", database=sqlite_db))
    import tools

    result = tools.execute_sql.invoke(
        {"reasoning": "t", "sql_query": "DELETE FROM Customer"}
    )
    assert result.lower().startswith("error")
    # The row must still be there.
    _, rows, _ = session.require_connection().run("SELECT count(*) FROM Customer")
    assert rows[0][0] == 2


def test_bad_connection_does_not_replace_a_working_one(sqlite_db):
    session.connect(ConnectionSettings(engine="sqlite", database=sqlite_db))
    working = session.current_connection()

    ok, _ = session.connect(
        ConnectionSettings(
            engine="postgresql", host="127.0.0.1", port=1,
            database="nope", username="nobody", password="x",
        )
    )
    assert not ok
    assert session.current_connection() is working


def test_tools_fail_clearly_when_not_connected():
    session.disconnect()
    with pytest.raises(session.NotConnected):
        session.require_connection()


def test_failed_connection_message_carries_no_password():
    ok, message = session.connect(
        ConnectionSettings(
            engine="postgresql", host="127.0.0.1", port=1,
            database="nope", username="nobody", password=SECRET,
        )
    )
    assert not ok
    assert SECRET not in message


# --- short display names --------------------------------------------------


@pytest.mark.parametrize(
    "database,expected",
    [
        (r"C:\Users\me\data\shop.db", "shop.db"),
        ("/var/lib/app/production.sqlite", "production.sqlite"),
        ("shop.db", "shop.db"),
        ("", "(no file)"),
    ],
)
def test_sqlite_display_name_is_just_the_file(database, expected):
    """A SQLite path is far too long for a sidebar chip; the file name is what
    identifies it. Both separators are handled regardless of the host OS."""
    settings = ConnectionSettings(engine="sqlite", database=database)
    assert settings.display_name == expected


def test_server_engine_display_name_is_the_database_name():
    settings = ConnectionSettings(
        engine="postgresql", host="db.internal", port=5432,
        database="erp", username="reporting", password=SECRET,
    )
    assert settings.display_name == "erp"


def test_display_name_falls_back_to_host_when_no_database_is_named():
    settings = ConnectionSettings(engine="postgresql", host="db.internal", database="")
    assert settings.display_name == "db.internal"


def test_display_name_never_contains_the_password():
    settings = ConnectionSettings(
        engine="mysql", host="h", database="sales", username="u", password=SECRET,
    )
    assert SECRET not in settings.display_name


# --- statement timeouts ---------------------------------------------------


def test_postgres_gets_a_server_side_statement_timeout():
    """A runaway generated query must be bounded by the server, so it still
    applies if the client goes away mid-query."""
    from connection import DatabaseConnection

    settings = ConnectionSettings(
        engine="postgresql", host="h", port=5432, database="d",
        username="u", password=SECRET,
    )
    conn = DatabaseConnection(settings, statement_timeout=7)
    assert "statement_timeout=7000" in conn.connect_args()["options"]


def test_a_zero_timeout_installs_no_limit():
    from connection import DatabaseConnection

    settings = ConnectionSettings(
        engine="postgresql", host="h", port=5432, database="d", username="u",
    )
    assert "options" not in DatabaseConnection(settings, statement_timeout=0).connect_args()


def test_sqlite_queries_are_interrupted_past_the_deadline(tmp_path):
    """SQLite has no statement_timeout, so a progress handler bounds it.
    Verified against a real query that would otherwise run far longer."""
    import sqlite3
    from connection import DatabaseConnection

    path = str(tmp_path / "slow.db")
    sqlite3.connect(path).close()

    conn = DatabaseConnection(
        ConnectionSettings(engine="sqlite", database=path), statement_timeout=1
    )
    # A recursive CTE that would count to a billion if left alone.
    runaway = (
        "WITH RECURSIVE forever(n) AS ("
        "  SELECT 1 UNION ALL SELECT n + 1 FROM forever WHERE n < 1000000000"
        ") SELECT count(*) FROM forever"
    )
    started = time.monotonic()
    with pytest.raises(Exception):
        conn.run(runaway)
    elapsed = time.monotonic() - started
    # Interrupted near the deadline rather than running to completion.
    assert elapsed < 15, f"query ran {elapsed:.1f}s despite a 1s timeout"
    conn.dispose()


def test_a_normal_query_is_unaffected_by_the_timeout(tmp_path):
    """The timeout must not interfere with queries that finish quickly."""
    import sqlite3
    from connection import DatabaseConnection

    path = str(tmp_path / "fine.db")
    raw = sqlite3.connect(path)
    raw.executescript("CREATE TABLE t(id INTEGER); INSERT INTO t VALUES (1),(2),(3);")
    raw.commit()
    raw.close()

    conn = DatabaseConnection(
        ConnectionSettings(engine="sqlite", database=path), statement_timeout=5
    )
    columns, rows, _ = conn.run("SELECT count(*) FROM t")
    assert rows[0][0] == 3
    conn.dispose()


# --- connection-string query parameters -----------------------------------


def test_sslmode_from_the_url_is_preserved():
    """Hosted Postgres (Neon, Supabase, Heroku) hands out URLs ending in
    ?sslmode=require and refuses the connection without it. Dropping the
    query string made every such database fail to connect."""
    settings = ConnectionSettings.from_url(
        "postgresql://u:p@ep-x.neon.tech/neondb?sslmode=require"
    )
    assert settings.options.get("sslmode") == "require"
    assert "sslmode=require" in settings.url()


def test_channel_binding_is_dropped():
    """libpq understands it; psycopg2's connect() rejects it as an unknown
    keyword. sslmode already enforces what it is there for."""
    settings = ConnectionSettings.from_url(
        "postgresql://u:p@ep-x.neon.tech/neondb?sslmode=require&channel_binding=require"
    )
    assert "channel_binding" not in settings.options
    assert settings.options.get("sslmode") == "require"


def test_multiple_parameters_are_kept():
    settings = ConnectionSettings.from_url(
        "postgresql://u:p@h/db?sslmode=verify-full&application_name=querymancer"
    )
    assert settings.options["sslmode"] == "verify-full"
    assert settings.options["application_name"] == "querymancer"


def test_a_url_without_parameters_still_parses():
    settings = ConnectionSettings.from_url("postgresql://u:p@h:5432/db")
    assert settings.options == {}


def test_empty_parameter_values_are_ignored():
    settings = ConnectionSettings.from_url("postgresql://u:p@h/db?sslmode=")
    assert "sslmode" not in settings.options


def test_parameters_do_not_leak_the_password():
    settings = ConnectionSettings.from_url(
        f"postgresql://u:{SECRET}@h/db?sslmode=require"
    )
    assert SECRET not in settings.safe_url
    assert "sslmode=require" in settings.safe_url
