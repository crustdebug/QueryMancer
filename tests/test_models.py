"""Tests for RotatingChatModel's model-level fallback behaviour.

These use fake clients rather than real provider SDKs, patched in at
models.create_llm - the single seam _client_for calls through - so the tests
are fast and deterministic while still exercising the real invoke() loop.
"""

import pytest

import models as models_module
from config import ModelConfig, ModelProvider
from key_pool import PoolExhausted
from models import RotatingChatModel, _model_is_unavailable


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeClient:
    """Stands in for a provider SDK client bound to one (model, key) pair."""

    def __init__(self, behavior):
        self.behavior = behavior  # callable(messages) -> response, or raises

    def bind_tools(self, tools):
        return self

    def invoke(self, messages, **kwargs):
        return self.behavior(messages)


def _config(name="model-a", provider=ModelProvider.GEMINI):
    return ModelConfig(name=name, temperature=0.0, provider=provider)


@pytest.fixture
def fake_credentials(monkeypatch):
    """Give every provider a single fake key, so KeyPool has something to try."""
    from config import Config

    monkeypatch.setattr(
        Config,
        "API_KEYS",
        {p: ["fake-key"] for p in ModelProvider},
    )


def test_model_not_found_falls_back_to_the_next_model(monkeypatch, fake_credentials):
    """A 404 for one model must not crash invoke() when another model is configured.

    This mirrors a real failure: gemini-2.5-flash-lite returned
    '404 NOT_FOUND ... no longer available to new users' while
    gemini-3.1-flash-lite (next in the chain) worked fine.
    """
    broken = _config("gemini-broken")
    working = _config("gemini-working")

    def factory(model_config, api_key=None):
        if model_config.name == "gemini-broken":
            def fail(messages):
                raise Exception(
                    "404 NOT_FOUND. This model models/gemini-broken is no "
                    "longer available to new users."
                )
            return FakeClient(fail)
        return FakeClient(lambda messages: FakeResponse("pong"))

    monkeypatch.setattr(models_module, "create_llm", factory)

    model = RotatingChatModel(models=[broken, working])
    response = model.invoke([("human", "hi")])

    assert response.content == "pong"
    assert model.active_model is working


def test_non_model_errors_are_not_swallowed(monkeypatch, fake_credentials):
    """A genuine bug (not 'model unavailable') must still propagate, not be
    silently treated as a reason to skip to the next model."""
    broken = _config("gemini-a")

    def factory(model_config, api_key=None):
        def fail(messages):
            raise ValueError("malformed request: missing required field 'contents'")
        return FakeClient(fail)

    monkeypatch.setattr(models_module, "create_llm", factory)

    model = RotatingChatModel(models=[broken])
    with pytest.raises(ValueError, match="malformed request"):
        model.invoke([("human", "hi")])


def test_all_models_unavailable_raises_with_details_in_the_cause_chain(
    monkeypatch, fake_credentials
):
    """When every model is unavailable, invoke() raises PoolExhausted - but
    the specific 404s from each model must still be discoverable, not lost."""
    def factory(model_config, api_key=None):
        def fail(messages):
            raise Exception(f"404 NOT_FOUND: {model_config.name} does not exist")
        return FakeClient(fail)

    monkeypatch.setattr(models_module, "create_llm", factory)

    model = RotatingChatModel(models=[_config("a"), _config("b")])
    with pytest.raises(PoolExhausted) as excinfo:
        model.invoke([("human", "hi")])

    # The per-model 404 detail lives in the exception cause, not discarded.
    assert "does not exist" in str(excinfo.value.__cause__)


def test_pool_exhausted_and_model_unavailable_can_both_trigger_fallback(
    monkeypatch, fake_credentials
):
    """Rate-limited then model-not-found then success - all three fallback
    paths (PoolExhausted, unavailable-model, success) chained in one call."""
    rate_limited = _config("gemini-limited")
    not_found = _config("gemini-gone")
    working = _config("gemini-ok")

    def factory(model_config, api_key=None):
        if model_config.name == "gemini-limited":
            def fail(messages):
                raise Exception("429 RESOURCE_EXHAUSTED: quota")
            return FakeClient(fail)
        if model_config.name == "gemini-gone":
            def fail(messages):
                raise Exception("404 NOT_FOUND: model not found")
            return FakeClient(fail)
        return FakeClient(lambda messages: FakeResponse("pong"))

    monkeypatch.setattr(models_module, "create_llm", factory)

    model = RotatingChatModel(models=[rate_limited, not_found, working])
    response = model.invoke([("human", "hi")])

    assert response.content == "pong"
    assert model.active_model is working


@pytest.mark.parametrize(
    "message,expected",
    [
        ("404 NOT_FOUND", True),
        ("This model is no longer available to new users.", True),
        ("Model not found: gemini-x", True),
        ("Unknown model 'foo'", True),
        ("429 RESOURCE_EXHAUSTED: quota", False),
        ("503 UNAVAILABLE: high demand", False),
        ("connection reset by peer", False),
    ],
)
def test_model_unavailable_classification(message, expected):
    assert _model_is_unavailable(Exception(message)) is expected
