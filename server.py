"""HTTP API and static host for the QueryMancer interface.

Run with:  python -m uvicorn server:app --reload

The browser holds only an opaque session id in an HttpOnly cookie. Database
credentials live server-side in memory for the life of that session and are
never sent back to the client - not in an API response, not in an error
message. See session.py and connection.py for the guarantees.
"""

import asyncio
import datetime as dt
import decimal
import logging
import time
import uuid
from pathlib import Path
from typing import Any, List, Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import Body, Cookie, FastAPI, Response  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

import schema as schema_module  # noqa: E402
import session as session_module  # noqa: E402
import suggestions as suggestions_module  # noqa: E402
import tools  # noqa: E402
from agent import ask, create_history  # noqa: E402
from connection import ENGINES, ConnectionSettings, sanitize  # noqa: E402
from key_pool import PoolExhausted  # noqa: E402
from models import RotatingChatModel  # noqa: E402
from session import Message, NotConnected  # noqa: E402

log = logging.getLogger("querymancer")

STATIC_DIR = Path(__file__).parent / "web"
COOKIE_NAME = "querymancer_session"

app = FastAPI(title="QueryMancer", docs_url=None, redoc_url=None)

# One model per process. The key pool's cooldown state lives on it, so
# rebuilding per request would forget which keys are rate-limited.
_model: Optional[RotatingChatModel] = None
_model_error: Optional[str] = None


def get_model() -> RotatingChatModel:
    global _model, _model_error
    if _model is None:
        _model = RotatingChatModel().bind_tools(tools.get_available_tools())
        _model_error = None
    return _model


# --- helpers --------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    """Convert database values into something JSON can carry.

    Decimals, dates and bytes all appear in real result sets and none of them
    survive json.dumps untouched.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, decimal.Decimal):
        # float() would silently lose precision on large monetary values.
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<{len(bytes(value))} bytes>"
    if isinstance(value, uuid.UUID):
        return str(value)
    return str(value)


def _session_response(payload: dict, session, response: Response) -> JSONResponse:
    """Return JSON and refresh the session cookie."""
    result = JSONResponse(payload)
    result.set_cookie(
        COOKIE_NAME,
        session.id,
        httponly=True,      # page scripts cannot read it
        samesite="lax",     # not sent on cross-site requests
        max_age=session_module.SESSION_IDLE_SECONDS,
        path="/",
    )
    return result


def _connection_state(session) -> dict:
    connection = session.connection
    if connection is None:
        return {"connected": False}
    settings = connection.settings
    return {
        "connected": True,
        # Never includes the password.
        "name": settings.display_name,
        # The full identifier, for a tooltip. A SQLite path is long enough that
        # the short name alone can be ambiguous between two files.
        "fullName": settings.database or settings.host,
        "engine": settings.engine,
        "engineLabel": settings.label,
        "summary": settings.summary,
    }


def _title_from(question: str) -> str:
    text = " ".join(question.split())
    return text[:48] + ("…" if len(text) > 48 else "")


# --- routes ---------------------------------------------------------------


@app.get("/api/state")
def get_state(querymancer_session: Optional[str] = Cookie(default=None)):
    """Everything the UI needs to render on load."""
    session = session_module.store.get_or_create(querymancer_session)

    with session_module.use_session(session):
        state = _connection_state(session)
        state["engines"] = [
            {"id": key, "label": spec["label"], "defaultPort": spec["default_port"]}
            for key, spec in ENGINES.items()
        ]
        state["conversations"] = [c.summary() for c in session.ordered_conversations()]
        state["suggestions"] = _suggestions_for(session)
        state["modelReady"] = _model_status()

    return _session_response(state, session, Response())


def _model_status() -> dict:
    try:
        model = get_model()
        return {"ok": True, "model": f"{model.active_model.provider.value}/{model.active_model.name}"}
    except Exception as error:  # noqa: BLE001
        return {"ok": False, "message": str(error)}


def _suggestions_for(session) -> List[str]:
    if session.connection is None:
        return []
    try:
        return suggestions_module.suggest(tools.get_schema())
    except Exception:  # noqa: BLE001
        return []


@app.post("/api/connect")
def connect(
    payload: dict = Body(...),
    querymancer_session: Optional[str] = Cookie(default=None),
):
    """Connect a database for this session."""
    session = session_module.store.get_or_create(querymancer_session)

    with session_module.use_session(session):
        try:
            url = (payload.get("url") or "").strip()
            if url:
                settings = ConnectionSettings.from_url(url)
            else:
                engine = payload.get("engine") or "postgresql"
                spec = ENGINES.get(engine)
                if spec is None:
                    raise ValueError(f"Unknown database type '{engine}'.")
                port = payload.get("port") or spec["default_port"]
                settings = ConnectionSettings(
                    engine=engine,
                    host=(payload.get("host") or "").strip(),
                    port=int(port) if port else None,
                    database=(payload.get("database") or "").strip(),
                    username=(payload.get("username") or "").strip(),
                    password=payload.get("password") or "",
                )
        except (ValueError, TypeError) as error:
            return _session_response(
                {"ok": False, "message": str(error)}, session, Response()
            )

        schema_module.clear_cache()
        ok, message = session_module.connect(settings)

        body: dict = {"ok": ok, "message": sanitize(message, settings)}
        if ok:
            body.update(_connection_state(session))
            body["suggestions"] = _suggestions_for(session)
            body["databases"] = session.connection.list_databases()
        return _session_response(body, session, Response())


@app.post("/api/disconnect")
def disconnect(querymancer_session: Optional[str] = Cookie(default=None)):
    session = session_module.store.get_or_create(querymancer_session)
    with session_module.use_session(session):
        session_module.disconnect()
        schema_module.clear_cache()
    return _session_response({"ok": True, "connected": False}, session, Response())


@app.get("/api/schema")
def get_schema_overview(querymancer_session: Optional[str] = Cookie(default=None)):
    """Tables in the connected database, for the schema panel."""
    session = session_module.store.get_or_create(querymancer_session)
    with session_module.use_session(session):
        if session.connection is None:
            return _session_response({"tables": []}, session, Response())
        try:
            db = tools.get_schema()
            tables = [
                {
                    "name": t.qualified,
                    "rows": t.estimated_rows,
                    # Full column detail for the inspector panel: name, type,
                    # and whether it is part of the primary key.
                    "columns": [
                        {
                            "name": c.name,
                            "type": c.data_type,
                            "nullable": c.nullable,
                            "primaryKey": c.name in t.primary_key,
                        }
                        for c in t.columns
                    ],
                }
                for t in sorted(db.tables, key=lambda t: (-t.estimated_rows, t.name.lower()))
            ]
            return _session_response(
                {
                    "tables": tables,
                    "foreignKeys": len(db.foreign_keys),
                    "totalRows": sum(t.estimated_rows for t in db.tables),
                },
                session,
                Response(),
            )
        except Exception as error:  # noqa: BLE001
            return _session_response(
                {"tables": [], "error": _safe_error(session, error)}, session, Response()
            )


@app.get("/api/conversations/{conversation_id}")
def get_conversation(
    conversation_id: str, querymancer_session: Optional[str] = Cookie(default=None)
):
    session = session_module.store.get_or_create(querymancer_session)
    conversation = session.conversations.get(conversation_id)
    if conversation is None:
        return _session_response({"error": "Not found"}, session, Response())
    return _session_response(conversation.to_dict(), session, Response())


@app.post("/api/conversations")
def new_conversation(querymancer_session: Optional[str] = Cookie(default=None)):
    session = session_module.store.get_or_create(querymancer_session)
    conversation = session.new_conversation()
    return _session_response(conversation.to_dict(), session, Response())


def _safe_error(session, error: Exception) -> str:
    settings = session.connection.settings if session.connection else None
    return sanitize(error, settings)


@app.post("/api/ask")
async def ask_question(
    payload: dict = Body(...),
    querymancer_session: Optional[str] = Cookie(default=None),
):
    """Answer one question, returning the reply plus the SQL that produced it."""
    session = session_module.store.get_or_create(querymancer_session)
    question = (payload.get("question") or "").strip()
    conversation_id = payload.get("conversationId")

    if not question:
        return _session_response(
            {"ok": False, "message": "Ask a question first."}, session, Response()
        )

    if session.connection is None:
        return _session_response(
            {"ok": False, "message": "Connect a database before asking a question."},
            session,
            Response(),
        )

    conversation = session.conversations.get(conversation_id) if conversation_id else None
    if conversation is None:
        conversation = session.new_conversation(_title_from(question))
        conversation.history = create_history()
    elif not conversation.messages:
        conversation.title = _title_from(question)

    conversation.messages.append(Message(role="user", text=question))

    # The agent is synchronous and does blocking network and database work, so
    # run it off the event loop to keep the server responsive.
    result = await asyncio.to_thread(_run_agent, session, conversation, question)

    conversation.messages.append(result)
    conversation.updated_at = time.time()

    payload_out = {
        "ok": not result.error,
        "conversationId": conversation.id,
        "title": conversation.title,
        "message": result.to_dict(),
        "conversations": [c.summary() for c in session.ordered_conversations()],
    }
    return _session_response(payload_out, session, Response())


def _run_agent(session, conversation, question: str) -> Message:
    """Run the agent loop and package the answer with its query trace."""
    with session_module.use_session(session):
        try:
            model = get_model()
        except Exception as error:  # noqa: BLE001
            return Message(role="assistant", text=str(error), error=True)

        try:
            with tools.capture_queries() as executed:
                answer = ask(question, conversation.history, model)
        except NotConnected as error:
            return Message(role="assistant", text=str(error), error=True)
        except PoolExhausted as error:
            return Message(
                role="assistant",
                text=(
                    f"Every API key is currently rate-limited. {error} "
                    "Add more keys to the pool, or wait for the cooldown."
                ),
                error=True,
            )
        except Exception as error:  # noqa: BLE001
            log.exception("Agent failed")
            return Message(
                role="assistant", text=_safe_error(session, error), error=True
            )

        message = Message(role="assistant", text=answer)

        # Attach the last successful query: that is the one that produced the
        # answer, and showing every discovery query would bury it.
        if executed:
            final = executed[-1]
            message.sql = final.sql
            message.columns = list(final.columns)
            message.rows = [[_jsonable(v) for v in row] for row in final.rows]
            message.truncated = final.truncated
            message.corrections = list(final.corrections)
        return message


@app.get("/api/usage")
def usage(querymancer_session: Optional[str] = Cookie(default=None)):
    """Token usage and key-pool status."""
    session = session_module.store.get_or_create(querymancer_session)
    try:
        model = get_model()
    except Exception:  # noqa: BLE001
        return _session_response({"available": False}, session, Response())
    return _session_response(
        {
            "available": True,
            "active": f"{model.active_model.provider.value}/{model.active_model.name}",
            "session": model.session_usage,
            "last": model.last_usage,
            "keys": model.pool_status(),
        },
        session,
        Response(),
    )


# --- static UI ------------------------------------------------------------

if STATIC_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")


@app.get("/")
def index():
    page = STATIC_DIR / "index.html"
    if not page.is_file():
        return JSONResponse(
            {"error": "UI not built. Expected web/index.html."}, status_code=500
        )
    return FileResponse(page)
