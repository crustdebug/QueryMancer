"""The agent loop: turn a natural-language question into a SQL-backed answer."""

from datetime import datetime
from typing import List, Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from config import Config
from custom_logging import green_border_style, log, log_panel
from tools import call_tool

SYSTEM_PROMPT = """
You are Querymancer, an expert PostgreSQL analyst. You turn natural-language
questions into correct SQL and answer them in as few tool calls as possible.

## Method
1. Identify the tables you need. Call `list_tables` only if the names are not
   already obvious or known from earlier in this conversation.
2. Call `describe_table` before writing SQL against a table you have not yet
   inspected. Never invent column names.
3. Call `get_foreign_key_relationships` before any JOIN.
4. When the user names a company or person, start with `search_entity_by_name`.
5. When filtering on a category, use `get_distinct_column_values` first so you
   filter on values that actually exist.
6. Run the final query with `execute_sql`.

## Efficiency
Every tool call costs a request against a limited quota. Batch your discovery:
if you need three tables described, ask for all three before writing SQL rather
than alternating between describing and querying. Prefer one well-formed query
over several exploratory ones. Do not re-fetch a schema you already have in
this conversation - reread it from the messages above.

## Rules
- Include a `reasoning` argument on every tool call.
- Only read-only SELECT queries. Never attempt to modify data.
- Select the columns you need, not `SELECT *`, except in `sample_table`.
- Always put a LIMIT on queries that could return many rows.
- Use explicit JOIN syntax with the keys from `get_foreign_key_relationships`.
- If a query errors or returns nothing, read the error, fix your assumption,
  and try once more. Do not repeat an identical failing query.
- If a tool reports a table is not permitted, do not retry it. Work with the
  tables you are allowed to read, and say so if the question cannot be answered.

## Answering
Write for business analysts who do not read SQL. Lead with the answer. Use a
Markdown table for multi-row results. State the figure and its units plainly.
Briefly note which tables the answer came from. If results were truncated, say
so. Never present a guess as fact - if the data is ambiguous, say what is
ambiguous.

Today's date is {today}.
""".strip()


def build_system_prompt() -> str:
    return SYSTEM_PROMPT.format(today=datetime.now().strftime("%Y-%m-%d"))


def create_history() -> List[BaseMessage]:
    return [SystemMessage(content=build_system_prompt())]


def trim_history(
    messages: List[BaseMessage], max_messages: Optional[int] = None
) -> List[BaseMessage]:
    """Keep the system prompt plus the most recent turns.

    A long conversation would otherwise resend every prior message on each call,
    so cost grows quadratically. Trimming keeps recent context, which is what
    follow-up questions actually depend on.

    Tool messages are dropped when trimming, because an orphaned ToolMessage
    whose originating AIMessage has been cut is rejected by most providers.
    """
    limit = max_messages or Config.MAX_HISTORY_MESSAGES
    system = [m for m in messages[:1] if isinstance(m, SystemMessage)]
    rest = messages[len(system) :]
    if len(rest) <= limit:
        return list(messages)

    kept = [m for m in rest if not isinstance(m, ToolMessage)]
    kept = [m for m in kept if not (isinstance(m, AIMessage) and m.tool_calls)]
    return system + kept[-limit:]


def ask(
    query: str,
    history: List[BaseMessage],
    llm,
    max_iterations: Optional[int] = None,
) -> str:
    """Run the tool-calling loop until the model produces a final answer.

    `history` is updated in place with the user message and the final answer, so
    the caller keeps a clean transcript without the intermediate tool traffic.
    """
    max_iterations = max_iterations or Config.MAX_AGENT_ITERATIONS
    log_panel(title="User Request", content=f"Query: {query}", border_style=green_border_style)

    history.append(HumanMessage(content=query))
    # Work on a trimmed copy; tool traffic stays out of the durable history.
    messages: List[BaseMessage] = trim_history(history)

    for iteration in range(max_iterations):
        response = llm.invoke(messages)
        messages.append(response)

        tool_calls = getattr(response, "tool_calls", None)
        if not tool_calls:
            answer = response.content if isinstance(response.content, str) else str(response.content)
            history.append(AIMessage(content=answer))
            return answer

        for tool_call in tool_calls:
            messages.append(call_tool(tool_call))

        log(f"[dim]Iteration {iteration + 1}/{max_iterations} complete.[/dim]")

    # Out of iterations: ask for the best answer available from what we have,
    # rather than failing outright and wasting the work already paid for.
    log("[yellow]Iteration limit reached; requesting a final answer.[/yellow]")
    messages.append(
        HumanMessage(
            content=(
                "You have reached the tool-call limit. Answer now using only the "
                "information gathered above. State clearly what remains uncertain."
            )
        )
    )
    final = llm.invoke(messages)
    answer = final.content if isinstance(final.content, str) else str(final.content)
    history.append(AIMessage(content=answer))
    return answer
