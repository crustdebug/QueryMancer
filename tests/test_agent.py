"""Tests for the agent loop, using a stub model so no API key is needed."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

import agent
from agent import ask, create_history, trim_history


class StubModel:
    """Replays a scripted list of responses and records what it was sent."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def invoke(self, messages, **kwargs):
        self.calls.append(list(messages))
        return self.responses.pop(0)


def test_direct_answer_needs_one_call():
    model = StubModel([AIMessage(content="42 invoices.")])
    history = create_history()
    answer = ask("How many invoices?", history, model)

    assert answer == "42 invoices."
    assert len(model.calls) == 1
    # History keeps the question and answer, not the tool traffic.
    assert isinstance(history[-1], AIMessage)
    assert isinstance(history[-2], HumanMessage)


def test_tool_call_is_executed_and_fed_back(monkeypatch):
    tool_response = AIMessage(
        content="",
        tool_calls=[{"name": "list_tables", "args": {"reasoning": "need names"}, "id": "call_1"}],
    )
    model = StubModel([tool_response, AIMessage(content="Found 3 tables.")])
    monkeypatch.setattr(
        agent, "call_tool", lambda tc: ToolMessage(content="a, b, c", tool_call_id=tc["id"])
    )

    answer = ask("What tables exist?", create_history(), model)

    assert answer == "Found 3 tables."
    assert len(model.calls) == 2
    # The tool result must be visible to the model on the second call.
    assert any(isinstance(m, ToolMessage) for m in model.calls[1])


def test_iteration_limit_still_produces_an_answer(monkeypatch):
    looping = AIMessage(
        content="",
        tool_calls=[{"name": "list_tables", "args": {"reasoning": "again"}, "id": "x"}],
    )
    # Always asks for a tool, then one final answer after the limit is hit.
    model = StubModel([looping] * 3 + [AIMessage(content="Partial answer.")])
    monkeypatch.setattr(
        agent, "call_tool", lambda tc: ToolMessage(content="ok", tool_call_id=tc["id"])
    )

    answer = ask("Loop forever", create_history(), model, max_iterations=3)

    assert answer == "Partial answer."
    assert len(model.calls) == 4  # 3 iterations plus the wrap-up call.


def test_history_is_trimmed_to_the_configured_window():
    history = create_history()
    for i in range(30):
        history.append(HumanMessage(content=f"q{i}"))
        history.append(AIMessage(content=f"a{i}"))

    trimmed = trim_history(history, max_messages=6)

    assert isinstance(trimmed[0], SystemMessage)
    assert len(trimmed) == 7
    assert trimmed[-1].content == "a29"


def test_trimming_drops_orphaned_tool_messages():
    """A ToolMessage without its originating AIMessage is rejected by providers."""
    history = [SystemMessage(content="sys")]
    for i in range(10):
        history.append(HumanMessage(content=f"q{i}"))
        history.append(
            AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": f"id{i}"}])
        )
        history.append(ToolMessage(content="result", tool_call_id=f"id{i}"))
        history.append(AIMessage(content=f"a{i}"))

    trimmed = trim_history(history, max_messages=6)

    assert not any(isinstance(m, ToolMessage) for m in trimmed)
    assert not any(isinstance(m, AIMessage) and m.tool_calls for m in trimmed)


def test_short_history_is_left_untouched():
    history = create_history()
    history.append(HumanMessage(content="hello"))
    assert trim_history(history, max_messages=12) == history
