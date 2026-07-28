"""Database connections for any supported engine.

The app connects to whatever database the user points it at - PostgreSQL,
MySQL/MariaDB, SQLite, SQL Server or Oracle - so nothing here may assume
PostgreSQL. SQLAlchemy supplies the dialect handling: URL construction, quoting
rules, and a uniform cursor.

Credential handling is deliberately strict:

  * Credentials live only where the caller puts them. Nothing in this module
    writes them to disk, and the Streamlit UI keeps them in per-session memory
    that is never persisted.
  * A password never appears in a rendered URL, a log line, an exception, or
    the UI. `ConnectionSettings.safe_url` masks it, and `sanitize` scrubs it
    out of driver errors before they are shown or logged.
"""

import re
import time
from dataclasses import dataclass, field, replace
from pathlib import PurePath
from typing import Dict, List, Optional
from urllib.parse import parse_qs, quote_plus, urlparse

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from config import Config

# Engines the app understands, with the driver each one needs. `default_port`
# is used to fill the form; `list_databases_sql` powers the "switch database"
# picker where the engine supports it.
ENGINES: Dict[str, dict] = {
    "postgresql": {
        "label": "PostgreSQL",
        "driver": "postgresql+psycopg2",
        "default_port": 5432,
        "package": "psycopg2-binary",
        "list_databases_sql": (
            "SELECT datname FROM pg_database "
            "WHERE NOT datistemplate AND datallowconn ORDER BY datname"
        ),
        "readonly_sql": "SET TRANSACTION READ ONLY",
    },
    "mysql": {
        "label": "MySQL / MariaDB",
        "driver": "mysql+pymysql",
        "default_port": 3306,
        "package": "pymysql",
        "list_databases_sql": (
            "SELECT schema_name FROM information_schema.schemata "
            "WHERE schema_name NOT IN "
            "('mysql','information_schema','performance_schema','sys') "
            "ORDER BY schema_name"
        ),
        "readonly_sql": "SET SESSION TRANSACTION READ ONLY",
    },
    "sqlite": {
        "label": "SQLite (file)",
        "driver": "sqlite",
        "default_port": None,
        "package": None,
        "list_databases_sql": None,
        "readonly_sql": None,  # Enforced via the file URI instead.
    },
    "mssql": {
        "label": "SQL Server",
        "driver": "mssql+pyodbc",
        "default_port": 1433,
        "package": "pyodbc",
        "list_databases_sql": "SELECT name FROM sys.databases ORDER BY name",
        "readonly_sql": None,
    },
    "oracle": {
        "label": "Oracle",
        "driver": "oracle+oracledb",
        "default_port": 1521,
        "package": "oracledb",
        "list_databases_sql": None,
        "readonly_sql": "SET TRANSACTION READ ONLY",
    },
}

DEFAULT_ENGINE = "postgresql"


@dataclass(frozen=True)
class ConnectionSettings:
    """Everything needed to reach one database.

    Frozen so a stored connection cannot be mutated by accident from the UI.
    """

    engine: str = DEFAULT_ENGINE
    host: str = "localhost"
    port: Optional[int] = None
    database: str = ""
    username: str = ""
    password: str = field(default="", repr=False)  # never shown by repr()
    # Extra driver arguments, e.g. {"sslmode": "require"}.
    options: Dict[str, str] = field(default_factory=dict)

    # -- construction ----------------------------------------------------

    @classmethod
    def from_url(cls, url: str) -> "ConnectionSettings":
        """Parse a connection string such as postgresql://user:pw@host:5432/db.

        Accepts the common scheme spellings (postgres, postgresql+psycopg2,
        mysql+pymysql, ...) and maps them onto a supported engine.
        """
        raw = url.strip()
        if not raw:
            raise ValueError("Connection string is empty.")

        # SQLite is a file path, not a host, so handle it before urlparse.
        if raw.startswith("sqlite"):
            path = raw.split("///", 1)[-1] if "///" in raw else ""
            return cls(engine="sqlite", database=path, host="", port=None)

        parsed = urlparse(raw)
        scheme = (parsed.scheme or "").lower()
        if not scheme:
            raise ValueError(
                "Connection string must start with an engine, "
                "for example postgresql://user:password@host:5432/database"
            )

        engine = _engine_for_scheme(scheme)
        if engine is None:
            supported = ", ".join(ENGINES)
            raise ValueError(f"Unsupported database type '{scheme}'. Supported: {supported}.")

        # Query parameters carry connection requirements, not decoration:
        # hosted Postgres (Neon, Supabase, Heroku) hands out URLs ending in
        # ?sslmode=require and refuses the connection without it. Dropping
        # them made every such URL fail with an error that named none of this.
        options = {
            key: values[0]
            for key, values in parse_qs(parsed.query or "").items()
            if values and values[0]
        }
        # channel_binding is understood by libpq but not by psycopg2's
        # connect(), which rejects it as an unknown keyword. sslmode already
        # covers what it is there to enforce.
        options.pop("channel_binding", None)

        return cls(
            engine=engine,
            host=parsed.hostname or "localhost",
            port=parsed.port or ENGINES[engine]["default_port"],
            database=(parsed.path or "").lstrip("/"),
            username=parsed.username or "",
            password=parsed.password or "",
            options=options,
        )

    def with_database(self, database: str) -> "ConnectionSettings":
        """A copy pointing at a different database on the same server."""
        return replace(self, database=database)

    # -- rendering -------------------------------------------------------

    @property
    def spec(self) -> dict:
        return ENGINES.get(self.engine, ENGINES[DEFAULT_ENGINE])

    @property
    def label(self) -> str:
        return self.spec["label"]

    @property
    def display_name(self) -> str:
        """A short name for this database, for chips and headings.

        SQLite databases are file paths, which are far too long to sit in a
        sidebar row or a header badge - the file name is the part that
        identifies it. Server engines already use a short database name, and
        fall back to the host for the rare connection that omits one.
        """
        if self.engine == "sqlite":
            if not self.database:
                return "(no file)"
            # PurePath handles both separators regardless of the host OS, so a
            # POSIX path still shortens correctly on Windows and vice versa.
            return PurePath(self.database.replace("\\", "/")).name or self.database
        return self.database or self.host

    def url(self, hide_password: bool = False) -> str:
        """The SQLAlchemy URL. Set hide_password for anything user-visible."""
        if self.engine == "sqlite":
            return f"sqlite:///{self.database}"

        driver = self.spec["driver"]
        auth = ""
        if self.username:
            auth = quote_plus(self.username)
            if self.password:
                auth += ":" + ("***" if hide_password else quote_plus(self.password))
            auth += "@"

        port = f":{self.port}" if self.port else ""
        database = f"/{self.database}" if self.database else ""
        query = ""
        if self.options:
            query = "?" + "&".join(
                f"{quote_plus(k)}={quote_plus(str(v))}" for k, v in self.options.items()
            )
        return f"{driver}://{auth}{self.host}{port}{database}{query}"

    @property
    def safe_url(self) -> str:
        """A URL safe to display or log: the password is masked."""
        return self.url(hide_password=True)

    @property
    def summary(self) -> str:
        if self.engine == "sqlite":
            return f"SQLite · {self.database or '(no file)'}"
        location = f"{self.host}:{self.port}" if self.port else self.host
        return f"{self.label} · {self.username}@{location}/{self.database}"

    def is_complete(self) -> tuple:
        """Whether these settings can be used, and why not if they cannot."""
        if self.engine == "sqlite":
            if not self.database:
                return False, "Enter the path to the SQLite file."
            return True, ""
        missing = []
        if not self.host:
            missing.append("host")
        if not self.database:
            missing.append("database")
        if not self.username:
            missing.append("user")
        if missing:
            return False, "Missing: " + ", ".join(missing) + "."
        return True, ""


def _engine_for_scheme(scheme: str) -> Optional[str]:
    """Map a URL scheme such as 'postgres' or 'mysql+pymysql' to an engine."""
    base = scheme.split("+", 1)[0]
    aliases = {
        "postgres": "postgresql",
        "postgresql": "postgresql",
        "psql": "postgresql",
        "mysql": "mysql",
        "mariadb": "mysql",
        "sqlite": "sqlite",
        "mssql": "mssql",
        "sqlserver": "mssql",
        "oracle": "oracle",
    }
    return aliases.get(base)


def sanitize(message: str, settings: Optional[ConnectionSettings] = None) -> str:
    """Strip credentials out of a driver error before it is shown or logged.

    Driver errors frequently echo the connection URL, and some include the
    password in plain text. Nothing user-visible should carry it.
    """
    text_out = str(message)
    if settings and settings.password:
        text_out = text_out.replace(settings.password, "***")
        text_out = text_out.replace(quote_plus(settings.password), "***")
    # Catch any user:password@host pattern, including ones we did not build.
    text_out = re.sub(r"(://[^:/@\s]+):([^@/\s]+)@", r"\1:***@", text_out)
    return text_out


def _sqlite_deadline(timeout: float):
    """A SQLite progress handler that aborts the query past a deadline.

    SQLite has no statement_timeout; returning non-zero from a progress
    handler interrupts the running statement, which is the supported way to
    bound one.
    """
    deadline = time.monotonic() + timeout

    def handler() -> int:
        return 1 if time.monotonic() > deadline else 0

    return handler


class DatabaseConnection:
    """A live connection to one database, with read-only enforcement.

    Engines differ in how read-only is imposed, so the strongest mechanism each
    one offers is used: a read-only transaction where supported, and for SQLite
    an immutable file URI. The SQL-level checks in tools.py apply regardless, so
    an engine without a session-level guarantee is still protected.
    """

    def __init__(
        self,
        settings: ConnectionSettings,
        connect_timeout: int = 10,
        statement_timeout: Optional[int] = None,
    ):
        self.settings = settings
        self._engine: Optional[Engine] = None
        self._connect_timeout = connect_timeout
        # Ceiling on a single statement. A model can write a query that
        # accidentally cross-joins two large tables; without this the database
        # works on it until it finishes or the pool is starved, and the user
        # just sees the app hang.
        self.statement_timeout = (
            Config.STATEMENT_TIMEOUT_SECONDS if statement_timeout is None else statement_timeout
        )

    # -- engine ----------------------------------------------------------

    def connect_args(self) -> dict:
        """Driver arguments for this connection, including the timeout.

        Built separately from _build_engine so the arguments can be inspected
        and tested without opening a connection: SQLAlchemy merges connect_args
        into the driver call only at connect time, so they are not readable
        back off a constructed Engine.
        """
        settings = self.settings
        timeout = self.statement_timeout

        if settings.engine == "sqlite":
            return {"uri": True}

        args: dict = {}
        if settings.engine == "postgresql":
            args["connect_timeout"] = self._connect_timeout
            # The statement timeout is deliberately NOT passed here as
            # "-c statement_timeout=...". A connection pooler in transaction
            # mode - Neon's -pooler endpoint, PgBouncer, Supabase's pooler -
            # rejects the whole connection with "unsupported startup parameter
            # in options: statement_timeout", because it cannot honour a
            # setting baked into the startup packet across pooled backends.
            # _apply_statement_timeout issues it as a SET on each connection
            # instead, which poolers do support and which is equally
            # server-enforced once established.
        elif settings.engine == "mysql":
            args["connect_timeout"] = self._connect_timeout
            if timeout:
                args["read_timeout"] = int(timeout)
        elif settings.engine == "mssql":
            if timeout:
                args["timeout"] = int(timeout)
        return args

    def _build_engine(self) -> Engine:
        settings = self.settings
        kwargs: dict = {"pool_pre_ping": True}

        if settings.engine == "sqlite":
            # 'mode=ro' makes the driver itself refuse writes.
            url = f"sqlite:///file:{settings.database}?mode=ro&uri=true"
            return create_engine(url, connect_args=self.connect_args(), **kwargs)

        return create_engine(settings.url(), connect_args=self.connect_args(), **kwargs)

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            self._engine = self._build_engine()
        return self._engine

    def dispose(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None

    # -- queries ---------------------------------------------------------

    def connect(self):
        """A connection with read-only applied where the engine supports it."""
        conn = self.engine.connect()
        readonly_sql = self.settings.spec.get("readonly_sql")
        if readonly_sql:
            try:
                conn.execute(text(readonly_sql))
            except Exception:  # noqa: BLE001
                # Some managed services disallow the statement. The SQL-level
                # checks still stand, so continue rather than refusing to run.
                pass

        # SQLite takes its limit from the driver rather than a connect arg, and
        # setting statement_timeout per session covers Postgres connections
        # made through a pooler that drops the connect-time options string.
        self._apply_statement_timeout(conn)
        return conn

    def _apply_statement_timeout(self, conn) -> None:
        """Best-effort per-session statement timeout."""
        timeout = self.statement_timeout
        if not timeout:
            return
        engine = self.settings.engine
        try:
            if engine == "postgresql":
                conn.execute(text(f"SET statement_timeout = {int(timeout * 1000)}"))
            elif engine == "sqlite":
                # SQLAlchemy exposes the DBAPI connection; SQLite interrupts a
                # long query through a progress handler rather than a setting.
                raw = conn.connection.dbapi_connection
                raw.set_progress_handler(_sqlite_deadline(timeout), 10_000)
            elif engine == "mysql":
                # MySQL 5.7.8+ / MariaDB use different names; try both.
                try:
                    conn.execute(
                        text(f"SET SESSION max_execution_time = {int(timeout * 1000)}")
                    )
                except Exception:  # noqa: BLE001
                    conn.execute(text(f"SET SESSION max_statement_time = {int(timeout)}"))
        except Exception:  # noqa: BLE001
            # A timeout we could not install is not a reason to refuse the
            # query - the request-level ceiling still applies above this.
            pass

    def run(self, sql: str, params: Optional[dict] = None, limit: Optional[int] = None):
        """Execute a read query, returning (columns, rows, truncated)."""
        with self.connect() as conn:
            result = conn.execute(text(sql), params or {})
            if result.returns_rows:
                columns = list(result.keys())
                if limit is not None:
                    rows = result.fetchmany(limit + 1)
                    truncated = len(rows) > limit
                    return columns, rows[:limit], truncated
                return columns, result.fetchall(), False
            return [], [], False

    def test(self) -> tuple:
        """Try the connection. Returns (ok, message) with no credentials in it."""
        try:
            with self.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True, f"Connected to {self.settings.summary}"
        except Exception as error:  # noqa: BLE001
            return False, sanitize(error, self.settings)

    def list_databases(self) -> List[str]:
        """Other databases on this server, for the switcher. Empty if N/A."""
        query = self.settings.spec.get("list_databases_sql")
        if not query:
            return []
        try:
            with self.connect() as conn:
                return [row[0] for row in conn.execute(text(query))]
        except Exception:  # noqa: BLE001
            return []
