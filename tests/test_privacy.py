"""Tests for local-only mode and the pre-deployment access controls.

The local-only tests matter more than most: the whole point of the mode is a
guarantee that no database content reaches a third party, and a guarantee that
silently does not hold is worse than none at all.
"""

import pytest
from fastapi.testclient import TestClient

import server
from config import Config, ModelConfig, ModelProvider


@pytest.fixture
def local_only(monkeypatch):
    monkeypatch.setattr(Config, "LOCAL_ONLY", True)
    return Config


# --- local-only model selection -------------------------------------------


def test_local_only_chain_contains_only_local_providers(local_only, monkeypatch):
    """Every cloud key configured, yet nothing cloud may appear in the chain."""
    monkeypatch.setattr(
        Config, "API_KEYS", {p: ["a-real-key"] for p in ModelProvider}
    )
    chain = Config.model_chain()
    assert chain, "local-only mode must still produce a usable chain"
    for model in chain:
        assert model.provider in Config.LOCAL_PROVIDERS, f"{model.name} is not local"


def test_a_leftover_cloud_api_key_cannot_leak_data_in_local_only_mode(
    local_only, monkeypatch
):
    """The exact failure this mode exists to prevent: a user installs Ollama
    for privacy but still has GOOGLE_API_KEY in their .env. Before local-only
    mode, Gemini came first and Ollama was only reached once every cloud key
    was exhausted."""
    monkeypatch.setattr(Config, "API_KEYS", {ModelProvider.GEMINI: ["leftover-key"]})
    providers = {m.provider for m in Config.model_chain()}
    assert ModelProvider.GEMINI not in providers
    assert providers <= set(Config.LOCAL_PROVIDERS)


def test_normal_mode_still_prefers_the_cloud_chain(monkeypatch):
    """Local-only must be opt-in; the default behaviour is unchanged."""
    monkeypatch.setattr(Config, "LOCAL_ONLY", False)
    monkeypatch.setattr(Config, "API_KEYS", {ModelProvider.GEMINI: ["key"]})
    chain = Config.model_chain()
    assert chain[0].provider == ModelProvider.GEMINI


def test_local_only_respects_the_configured_ollama_model(local_only, monkeypatch):
    monkeypatch.setattr(
        Config, "LOCAL_MODEL", ModelConfig("llama3.1:8b", 0.0, ModelProvider.OLLAMA)
    )
    assert Config.model_chain()[0].name == "llama3.1:8b"


def test_local_only_chain_has_no_duplicates(local_only, monkeypatch):
    """LOCAL_MODEL may also appear in FALLBACK_MODELS."""
    monkeypatch.setattr(
        Config, "LOCAL_MODEL", ModelConfig("qwen2.5:latest", 0.0, ModelProvider.OLLAMA)
    )
    names = [m.name for m in Config.model_chain()]
    assert len(names) == len(set(names))


# --- what the UI is allowed to claim --------------------------------------


def test_privacy_mode_reports_local_when_the_whole_chain_is_local(
    local_only, monkeypatch
):
    monkeypatch.setattr(Config, "API_KEYS", {p: ["key"] for p in ModelProvider})
    assert Config.privacy_mode()["dataStaysLocal"] is True


def test_privacy_mode_does_not_claim_local_when_a_cloud_model_is_reachable(monkeypatch):
    """A badge claiming privacy the configuration does not provide would be
    worse than no badge."""
    monkeypatch.setattr(Config, "LOCAL_ONLY", False)
    monkeypatch.setattr(Config, "API_KEYS", {ModelProvider.GEMINI: ["key"]})
    assert Config.privacy_mode()["dataStaysLocal"] is False


# --- access code ----------------------------------------------------------


@pytest.fixture
def locked(monkeypatch):
    monkeypatch.setattr(server, "ACCESS_CODE", "letmein")
    with TestClient(server.app) as client:
        yield client


def test_api_is_refused_without_the_access_code(locked):
    assert locked.get("/api/state").status_code == 401


def test_the_page_is_refused_without_the_access_code(locked):
    response = locked.get("/")
    assert response.status_code == 401
    assert "Access code" in response.text


def test_the_correct_code_unlocks_the_app(locked):
    assert locked.post("/api/unlock", json={"code": "letmein"}).json()["ok"] is True
    # The cookie set by /api/unlock now carries the session through.
    assert locked.get("/api/state").status_code == 200


def test_a_wrong_code_is_rejected(locked):
    response = locked.post("/api/unlock", json={"code": "guess"})
    assert response.status_code == 401
    assert locked.get("/api/state").status_code == 401


def test_the_unlock_page_itself_is_reachable_while_locked(locked):
    assert locked.get("/unlock").status_code == 200


def test_the_health_check_is_reachable_while_locked(locked):
    """A platform health probe carries no cookie. If the gate blocked it,
    every deploy would look unhealthy and be rolled back."""
    response = locked.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_the_health_check_warms_the_model(monkeypatch):
    """An uptime pinger keeps the process awake, but the first question would
    still pay to build the provider clients. Warming here moves that cost off
    the first visitor."""
    monkeypatch.setattr(server, "ACCESS_CODE", "")
    monkeypatch.setattr(server, "_model", None)

    built = []

    class FakeModel:
        def bind_tools(self, tools):
            built.append(True)
            return self

    monkeypatch.setattr(server, "RotatingChatModel", lambda: FakeModel())
    with TestClient(server.app) as client:
        assert client.get("/healthz").status_code == 200
    assert built, "health check did not warm the model"


def test_the_health_check_survives_a_model_that_cannot_be_built(monkeypatch):
    """A missing API key must not fail the probe: the platform would read that
    as an unhealthy deploy and roll back a release that is otherwise fine."""
    monkeypatch.setattr(server, "ACCESS_CODE", "")
    monkeypatch.setattr(server, "_model", None)

    def explode():
        raise RuntimeError("No usable model configured.")

    monkeypatch.setattr(server, "RotatingChatModel", explode)
    with TestClient(server.app) as client:
        response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_the_health_check_discloses_nothing_about_configuration(locked):
    """It is reachable without the access code, so it must not leak which
    providers are configured or whether a database is attached."""
    body = locked.get("/healthz").json()
    assert set(body) == {"status"}


def test_no_access_code_configured_leaves_the_app_open(monkeypatch):
    """Local use must need no setup."""
    monkeypatch.setattr(server, "ACCESS_CODE", "")
    with TestClient(server.app) as client:
        assert client.get("/api/state").status_code == 200


def test_a_forged_access_cookie_does_not_unlock(locked):
    locked.cookies.set(server.ACCESS_COOKIE, "not-the-code")
    assert locked.get("/api/state").status_code == 401


# --- cookie flags ---------------------------------------------------------


def test_the_session_cookie_is_marked_secure_over_https(monkeypatch):
    """The session id is the only thing separating one user's database
    connection from another's, so it must not travel over plain HTTP."""
    monkeypatch.setattr(server, "ACCESS_CODE", "")
    with TestClient(server.app, base_url="https://testserver") as client:
        header = client.get("/api/state").headers["set-cookie"]
    assert "Secure" in header
    assert "HttpOnly" in header


def test_a_proxy_terminating_tls_still_marks_the_cookie_secure(monkeypatch):
    """Deployments put TLS at a proxy, so the app sees http:// internally and
    must trust the forwarded-proto header or the flag would never be set."""
    monkeypatch.setattr(server, "ACCESS_CODE", "")
    with TestClient(server.app) as client:
        header = client.get(
            "/api/state", headers={"x-forwarded-proto": "https"}
        ).headers["set-cookie"]
    assert "Secure" in header


def test_the_cookie_is_not_secure_over_plain_http(monkeypatch):
    """A browser silently discards a Secure cookie sent over http://, which
    would lock out local users with nothing to explain why."""
    monkeypatch.setattr(server, "ACCESS_CODE", "")
    monkeypatch.setattr(server, "FORCE_SECURE_COOKIE", False)
    with TestClient(server.app) as client:
        header = client.get("/api/state").headers["set-cookie"]
    assert "Secure" not in header
    assert "HttpOnly" in header


def test_force_secure_cookie_overrides_the_scheme(monkeypatch):
    monkeypatch.setattr(server, "ACCESS_CODE", "")
    monkeypatch.setattr(server, "FORCE_SECURE_COOKIE", True)
    with TestClient(server.app) as client:
        assert "Secure" in client.get("/api/state").headers["set-cookie"]


def test_unlocking_over_plain_http_actually_works(monkeypatch):
    """The regression this scheme detection exists to prevent: a Secure cookie
    over http:// is discarded, so the user unlocks and is asked to unlock
    again, with nothing to explain why."""
    monkeypatch.setattr(server, "ACCESS_CODE", "letmein")
    monkeypatch.setattr(server, "FORCE_SECURE_COOKIE", False)
    with TestClient(server.app) as client:
        assert client.post("/api/unlock", json={"code": "letmein"}).json()["ok"]
        assert client.get("/api/state").status_code == 200
