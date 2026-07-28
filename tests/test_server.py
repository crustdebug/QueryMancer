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
    assert "Querio" in response.text


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
