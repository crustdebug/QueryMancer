"""Tests for the HTTP API.

The isolation tests matter most: two browser sessions must never see each
other's database connection or conversations.
"""

import os
import sqlite3
import tempfile

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

import schema as schema_module
import server
import session as session_module

SECRET = "TopSecret-123!"


@pytest.fixture
def db_path(tmp_path):
    path = str(tmp_path / "shop.db")
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE Customer(id INTEGER PRIMARY KEY, companyName TEXT);
        CREATE TABLE SalesOrder(
            id INTEGER PRIMARY KEY,
            customerId INTEGER REFERENCES Customer(id),
            totalAmount NUMERIC
        );
        INSERT INTO Customer(companyName) VALUES ('Acme'),('Globex'),('Initech');
        INSERT INTO SalesOrder(customerId,totalAmount) VALUES (1,100),(2,250),(1,75);
        """
    )
    conn.commit()
    conn.close()
    return path


class ScriptedModel:
    """Runs one real query, then answers. No API quota needed."""

    def __init__(self, sql="SELECT companyName FROM Customer ORDER BY id"):
        self.sql = sql
        self.calls = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[{
                    "name": "execute_sql",
                    "args": {"reasoning": "answer", "sql_query": self.sql},
                    "id": "c1",
                }],
            )
        return AIMessage(content="There are three customers: Acme, Globex and Initech.")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(server, "_model", ScriptedModel())
    schema_module.clear_cache()
    with TestClient(server.app) as c:
        yield c
    schema_module.clear_cache()


def connect(client, path):
    return client.post("/api/connect", json={"engine": "sqlite", "database": path}).json()


# --- basic surface --------------------------------------------------------


def test_index_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "QueryMancer" in response.text


def test_state_lists_engines_before_connecting(client):
    state = client.get("/api/state").json()
    assert state["connected"] is False
    ids = {e["id"] for e in state["engines"]}
    assert {"postgresql", "mysql", "sqlite"} <= ids


def test_session_cookie_is_httponly(client):
    response = client.get("/api/state")
    header = response.headers.get("set-cookie", "")
    assert "httponly" in header.lower()
    assert "samesite=lax" in header.lower()


# --- connecting -----------------------------------------------------------


def test_connect_reports_success_and_suggestions(client, db_path):
    body = connect(client, db_path)
    assert body["ok"] is True
    assert body["connected"] is True
    assert body["suggestions"], "should suggest questions from the schema"


def test_suggestions_name_real_tables(client, db_path):
    body = connect(client, db_path)
    joined = " ".join(body["suggestions"]).lower()
    assert "customer" in joined or "sales order" in joined


def test_schema_endpoint_lists_tables(client, db_path):
    connect(client, db_path)
    data = client.get("/api/schema").json()
    names = {t["name"] for t in data["tables"]}
    assert any("Customer" in n for n in names)
    assert data["foreignKeys"] >= 1


def test_bad_connection_reports_without_leaking_password(client):
    body = client.post(
        "/api/connect",
        json={
            "engine": "postgresql", "host": "127.0.0.1", "port": 1,
            "database": "nope", "username": "u", "password": SECRET,
        },
    ).json()
    assert body["ok"] is False
    assert SECRET not in str(body)


def test_unknown_engine_is_rejected(client):
    body = client.post("/api/connect", json={"engine": "mongodb"}).json()
    assert body["ok"] is False


def test_disconnect_clears_the_connection(client, db_path):
    connect(client, db_path)
    assert client.get("/api/state").json()["connected"] is True
    client.post("/api/disconnect")
    assert client.get("/api/state").json()["connected"] is False


# --- asking ---------------------------------------------------------------


def test_ask_requires_a_connection(client):
    body = client.post("/api/ask", json={"question": "how many?"}).json()
    assert body["ok"] is False
    assert "connect" in body["message"].lower()


def test_ask_rejects_an_empty_question(client, db_path):
    connect(client, db_path)
    body = client.post("/api/ask", json={"question": "   "}).json()
    assert body["ok"] is False


def test_ask_returns_answer_sql_and_rows(client, db_path):
    connect(client, db_path)
    body = client.post("/api/ask", json={"question": "List the customers"}).json()

    assert body["ok"] is True
    message = body["message"]
    assert "customers" in message["text"].lower()
    # The SQL the UI shows must be the query that actually ran.
    assert "Customer" in message["sql"]
    assert message["columns"] == ["companyName"]
    assert [r[0] for r in message["rows"]] == ["Acme", "Globex", "Initech"]


def test_ask_creates_a_conversation_titled_from_the_question(client, db_path):
    connect(client, db_path)
    body = client.post("/api/ask", json={"question": "List the customers"}).json()
    assert body["title"] == "List the customers"
    assert body["conversationId"]


def test_conversation_can_be_reloaded(client, db_path):
    connect(client, db_path)
    body = client.post("/api/ask", json={"question": "List the customers"}).json()
    convo = client.get(f"/api/conversations/{body['conversationId']}").json()
    assert [m["role"] for m in convo["messages"]] == ["user", "assistant"]


def test_second_question_starts_a_second_conversation(client, db_path):
    connect(client, db_path)
    client.post("/api/ask", json={"question": "First question"})
    client.post("/api/ask", json={"question": "Second question"})
    titles = [c["title"] for c in client.get("/api/state").json()["conversations"]]
    assert "First question" in titles and "Second question" in titles


def test_follow_up_stays_in_the_same_conversation(client, db_path):
    connect(client, db_path)
    first = client.post("/api/ask", json={"question": "First"}).json()
    second = client.post(
        "/api/ask",
        json={"question": "And again", "conversationId": first["conversationId"]},
    ).json()
    assert second["conversationId"] == first["conversationId"]
    convo = client.get(f"/api/conversations/{first['conversationId']}").json()
    assert len(convo["messages"]) == 4


def test_unknown_conversation_returns_an_error(client):
    assert "error" in client.get("/api/conversations/nope").json()


# --- which database answered a conversation -------------------------------


def test_conversation_records_the_database_it_was_asked_against(client, db_path):
    connect(client, db_path)
    body = client.post("/api/ask", json={"question": "List the customers"}).json()

    summary = client.get("/api/state").json()["conversations"][0]
    # The short display name, not the full path: this sits in a sidebar chip.
    assert summary["database"] == "shop.db"
    assert summary["engine"] == "sqlite"
    assert summary["engineLabel"]

    # The same stamp is available when the thread is reopened.
    convo = client.get(f"/api/conversations/{body['conversationId']}").json()
    assert convo["database"] == "shop.db"


def test_reconnecting_elsewhere_does_not_relabel_older_conversations(
    client, db_path, tmp_path
):
    """The reason the database is stored per conversation rather than read from
    the live connection: a session can move to another database while earlier
    threads stay in the sidebar, and those threads describe the old data."""
    connect(client, db_path)
    first = client.post("/api/ask", json={"question": "Question on the first db"}).json()

    second_path = str(tmp_path / "other.db")
    conn = sqlite3.connect(second_path)
    conn.executescript("CREATE TABLE Customer(id INTEGER PRIMARY KEY, companyName TEXT);")
    conn.commit()
    conn.close()

    client.post("/api/disconnect")
    connect(client, second_path)
    client.post("/api/ask", json={"question": "Question on the second db"})

    by_title = {c["title"]: c for c in client.get("/api/state").json()["conversations"]}
    assert by_title["Question on the first db"]["database"] == "shop.db"
    assert by_title["Question on the second db"]["database"] == "other.db"

    # And reopening the older thread still reports the database it actually used.
    convo = client.get(f"/api/conversations/{first['conversationId']}").json()
    assert convo["database"] == "shop.db"


def test_conversation_stamp_exposes_only_display_fields(client, db_path):
    """The stamp reaches the browser, so it must carry nothing but display text.

    Asserted as a whitelist rather than by grepping for a password: SQLite
    needs no credentials, so a "password not present" check would pass here
    even if the stamp did carry one. Pinning the exact key set is what
    actually stops a host, user, or URL being added to it later.
    """
    connect(client, db_path)
    client.post("/api/ask", json={"question": "List the customers"})

    summary = client.get("/api/state").json()["conversations"][0]
    assert set(summary) == {
        "id",
        "title",
        "updated_at",
        "message_count",
        "database",
        "engine",
        "engineLabel",
    }


def test_connection_state_never_returns_the_password(client, tmp_path):
    """A connected session's state must not echo credentials back to the client."""
    settings_path = str(tmp_path / "creds.db")
    sqlite3.connect(settings_path).close()

    # Drive the session store directly: this asserts the shape of the response
    # for a credentialed engine without needing a live Postgres to accept it.
    session = session_module.store.create()
    with session_module.use_session(session):
        from connection import ConnectionSettings, DatabaseConnection

        settings = ConnectionSettings(
            engine="postgresql",
            host="db.internal",
            port=5432,
            database="erp",
            username="reporting",
            password=SECRET,
        )
        session.connection = DatabaseConnection(settings)
        state = server._connection_state(session)
        conversation = session.new_conversation("Revenue last quarter")

    # The password never leaves the server, in either payload.
    assert SECRET not in f"{state} {conversation.summary()}"

    # The connection card deliberately shows where the session is attached
    # (host and user), but a conversation stamp is only ever a display name -
    # it is repeated on every history row and has no reason to carry more.
    stamp = str(conversation.summary())
    assert "db.internal" not in stamp
    assert "reporting" not in stamp
    assert conversation.database == "erp"
    assert conversation.engine_label == "PostgreSQL"


# --- session isolation ----------------------------------------------------


def test_two_sessions_do_not_share_a_connection(db_path, monkeypatch):
    monkeypatch.setattr(server, "_model", ScriptedModel())
    schema_module.clear_cache()

    alice = TestClient(server.app)
    bob = TestClient(server.app)

    assert connect(alice, db_path)["ok"] is True

    # Bob has his own cookie jar and must not inherit Alice's connection.
    assert bob.get("/api/state").json()["connected"] is False
    assert bob.post("/api/ask", json={"question": "hi"}).json()["ok"] is False


def test_two_sessions_do_not_share_conversations(db_path, monkeypatch):
    monkeypatch.setattr(server, "_model", ScriptedModel())
    schema_module.clear_cache()

    alice = TestClient(server.app)
    bob = TestClient(server.app)

    connect(alice, db_path)
    alice.post("/api/ask", json={"question": "Alice's private question"})

    connect(bob, db_path)
    bob_titles = [c["title"] for c in bob.get("/api/state").json()["conversations"]]
    assert "Alice's private question" not in bob_titles


def test_expired_session_is_discarded(monkeypatch):
    store = session_module.SessionStore()
    created = store.create()
    monkeypatch.setattr(session_module, "SESSION_IDLE_SECONDS", -1)
    assert store.get(created.id) is None


# --- value serialisation --------------------------------------------------


def test_database_values_survive_json(tmp_path, monkeypatch):
    """Decimals, dates and bytes all appear in real result sets."""
    import datetime
    import decimal

    assert server._jsonable(decimal.Decimal("10.50")) == 10.5
    assert server._jsonable(decimal.Decimal("42")) == 42
    assert server._jsonable(datetime.date(2026, 7, 28)) == "2026-07-28"
    assert "bytes" in server._jsonable(b"\x00\x01\x02")
    assert server._jsonable(None) is None


# --- answer caching -------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_answer_cache():
    """The cache is process-wide, so tests must not leak into each other."""
    server._answer_cache.invalidate()
    yield
    server._answer_cache.invalidate()


def test_repeating_a_question_skips_the_model(client, db_path):
    """The point of the cache: a re-ask costs no LLM calls."""
    connect(client, db_path)
    first = client.post("/api/ask", json={"question": "List the customers"}).json()
    calls_after_first = server._model.calls

    second = client.post("/api/ask", json={"question": "List the customers"}).json()
    assert server._model.calls == calls_after_first, "model was called again"
    assert second["message"]["text"] == first["message"]["text"]
    assert second["message"]["cached"] is True
    assert first["message"]["cached"] is False


def test_a_cached_answer_keeps_its_sql_and_rows(client, db_path):
    connect(client, db_path)
    first = client.post("/api/ask", json={"question": "List the customers"}).json()
    second = client.post("/api/ask", json={"question": "List the customers"}).json()
    assert second["message"]["sql"] == first["message"]["sql"]
    assert second["message"]["rows"] == first["message"]["rows"]
    assert second["message"]["columns"] == first["message"]["columns"]


def test_a_different_question_is_not_served_from_the_cache(client, db_path):
    connect(client, db_path)
    client.post("/api/ask", json={"question": "List the customers"})
    calls = server._model.calls
    body = client.post("/api/ask", json={"question": "Count the orders"}).json()
    assert server._model.calls > calls
    assert body["message"]["cached"] is False


def test_a_follow_up_is_never_served_from_the_cache(client, db_path):
    """A follow-up depends on what came before it, so the question text alone
    does not identify the answer."""
    connect(client, db_path)
    first = client.post("/api/ask", json={"question": "List the customers"}).json()
    calls = server._model.calls

    # The same text again, but inside the existing conversation.
    follow_up = client.post(
        "/api/ask",
        json={"question": "List the customers", "conversationId": first["conversationId"]},
    ).json()
    assert server._model.calls > calls, "follow-up wrongly served from cache"
    assert follow_up["message"]["cached"] is False


def test_the_same_question_against_another_database_is_recomputed(client, db_path, tmp_path):
    """Two databases must never share a cached answer."""
    connect(client, db_path)
    client.post("/api/ask", json={"question": "List the customers"})
    calls = server._model.calls

    other = str(tmp_path / "other.db")
    conn = sqlite3.connect(other)
    conn.executescript(
        "CREATE TABLE Customer(id INTEGER PRIMARY KEY, companyName TEXT);"
        "INSERT INTO Customer(companyName) VALUES ('Different');"
    )
    conn.commit()
    conn.close()

    client.post("/api/disconnect")
    connect(client, other)
    body = client.post("/api/ask", json={"question": "List the customers"}).json()
    assert server._model.calls > calls, "answer leaked across databases"
    assert body["message"]["cached"] is False


def test_cache_statistics_track_hits_and_misses(client, db_path):
    """Asserted on the cache directly: /api/usage also reports these, but it
    needs a real model object for the rest of its payload."""
    connect(client, db_path)
    client.post("/api/ask", json={"question": "List the customers"})
    client.post("/api/ask", json={"question": "List the customers"})
    stats = server._answer_cache.stats()
    assert stats["hits"] >= 1
    assert stats["entries"] >= 1


# --- demo database --------------------------------------------------------


DEMO_SECRET = "dem0-Pa55word!"


def test_no_demo_offered_when_none_is_configured(client, monkeypatch):
    monkeypatch.setattr(server, "DEMO_DATABASE_URL", "")
    assert client.get("/api/state").json()["demo"]["available"] is False


def test_demo_is_offered_when_configured(client, monkeypatch, db_path):
    monkeypatch.setattr(server, "DEMO_DATABASE_URL", f"sqlite:///{db_path}")
    assert client.get("/api/state").json()["demo"]["available"] is True


def test_state_never_reveals_the_demo_connection_string(client, monkeypatch):
    """The whole point of the button: the browser learns a demo exists, not
    how to connect to it."""
    monkeypatch.setattr(
        server,
        "DEMO_DATABASE_URL",
        f"postgresql://demo:{DEMO_SECRET}@db.internal:5432/shop",
    )
    body = client.get("/api/state").text
    assert DEMO_SECRET not in body
    assert "db.internal" not in body


def test_connecting_to_the_demo_works(client, monkeypatch, db_path):
    monkeypatch.setattr(server, "DEMO_DATABASE_URL", f"sqlite:///{db_path}")
    body = client.post("/api/connect-demo").json()
    assert body["ok"] is True
    assert body["connected"] is True
    assert client.get("/api/state").json()["connected"] is True


def test_the_demo_response_reveals_neither_credentials_nor_location(
    client, monkeypatch, db_path
):
    """Not just the password: the host, user and database name of a demo the
    visitor did not supply are not theirs to see either. A hosted demo would
    otherwise hand out demo_user@ep-xxx.neon.tech:5432/neondb - everything
    needed to attack it except the password."""
    monkeypatch.setattr(server, "DEMO_DATABASE_URL", f"sqlite:///{db_path}")
    connect_body = client.post("/api/connect-demo").text
    state_body = client.get("/api/state").text

    for body in (connect_body, state_body):
        assert "shop.db" not in body, "demo location leaked"
        assert db_path not in body
        assert "sqlite:///" not in body


def test_the_conversation_stamp_does_not_leak_the_demo_location(
    client, monkeypatch, db_path
):
    """Every history row carries a database stamp. It records display_name,
    which for a demo would be the real filename - making the sidebar the one
    place a visitor could read where the demo actually lives."""
    monkeypatch.setattr(server, "DEMO_DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setattr(server, "DEMO_LABEL", "Sample data")
    client.post("/api/connect-demo")
    client.post("/api/ask", json={"question": "List the customers"})

    body = client.get("/api/state").text
    assert "shop.db" not in body
    summary = client.get("/api/state").json()["conversations"][0]
    assert summary["database"] == "Sample data"


def test_a_users_own_connection_still_shows_where_they_are_attached(client, db_path):
    """The withholding applies only to the demo: a database the visitor typed
    in themselves must still be identified in the sidebar."""
    connect(client, db_path)
    assert "shop.db" in client.get("/api/state").text


def test_connecting_your_own_database_after_the_demo_clears_the_demo_flag(
    client, monkeypatch, db_path
):
    monkeypatch.setattr(server, "DEMO_DATABASE_URL", f"sqlite:///{db_path}")
    client.post("/api/connect-demo")
    assert client.get("/api/state").json()["isDemo"] is True
    connect(client, db_path)
    assert client.get("/api/state").json()["isDemo"] is False


def test_a_failing_demo_does_not_leak_the_url_in_the_error(client, monkeypatch):
    """Driver errors quote the connection URL, and this endpoint is public."""
    monkeypatch.setattr(
        server,
        "DEMO_DATABASE_URL",
        f"postgresql://demo:{DEMO_SECRET}@127.0.0.1:1/nope",
    )
    response = client.post("/api/connect-demo")
    assert response.json()["ok"] is False
    assert DEMO_SECRET not in response.text
    assert "127.0.0.1" not in response.text


def test_connecting_to_an_unconfigured_demo_is_refused(client, monkeypatch):
    monkeypatch.setattr(server, "DEMO_DATABASE_URL", "")
    body = client.post("/api/connect-demo").json()
    assert body["ok"] is False


def test_the_demo_connection_is_usable_for_questions(client, monkeypatch, db_path):
    """Connecting is not enough - the session must actually be able to answer."""
    monkeypatch.setattr(server, "DEMO_DATABASE_URL", f"sqlite:///{db_path}")
    client.post("/api/connect-demo")
    body = client.post("/api/ask", json={"question": "List the customers"}).json()
    assert body["ok"] is True
    assert body["message"]["rows"]
