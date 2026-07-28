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
You are Querio, an expert SQL analyst. You turn natural-language questions into
correct SQL and answer them in as few tool calls as possible.

You have no prior knowledge of this database. Its tables and columns may follow
any naming convention - snake_case, camelCase or PascalCase - and the names will
often not be what you would expect. Discover the real names; never assume them.

## Method
1. Call `inspect_database` first, unless the schema is already visible earlier in
   this conversation. One call gives you every table, its size, its columns and
   the foreign keys - enough to plan the whole query.
2. Call `describe_table` only when you need the full column list of a table the
   overview abbreviated.
3. When the user mentions a specific value - a company, a person, a code - and
   you cannot tell which table holds it, call `find_value` to locate it.
4. Before filtering on a categorical column, call `get_distinct_column_values`
   so you filter on values that exist.
5. Run the final query with `execute_sql`.

## Naming
Write identifiers exactly as `inspect_database` reports them. Most engines fold
unquoted names to one case, so a table created as "Customer" must be quoted to
be found: SELECT "firstName" FROM "Employee". Quote any name containing capitals
or unusual characters.
If you get a name slightly wrong, it will be corrected automatically and the
correction reported to you - use the corrected name from then on.

## Efficiency
Every tool call spends a request from a limited quota. `inspect_database` is
cached, so calling it once is cheap and calling it repeatedly is wasteful.
Plan the whole query from the overview rather than describing tables one at a
time. Prefer one well-formed query over several exploratory ones.

## Rules
- Include a `reasoning` argument on every tool call.
- Only read-only SELECT queries. Never attempt to modify data.
- Select the columns you need, not `SELECT *`, except in `sample_table`.
- Always put a LIMIT on queries that could return many rows.
- Use explicit JOIN syntax, with the foreign keys from the overview. If none are
  declared, join on matching column names such as customerId to id.
- If a query errors, read the error and the suggestions it carries, fix your
  assumption, and try again. Do not repeat an identical failing query.

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


def extract_text(content) -> str:
    """Pull plain text out of a message's content, whatever shape it's in.

    Some providers (newer Gemini models among them) return content as a list
    of blocks - {"type": "text", "text": ..., "extras": {...}} - rather than a
    plain string, so the response can carry a "signature" or other metadata
    block alongside the text. Concatenating just the text blocks is what a
    plain string would have been.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in (None, "text"):
                text = block.get("text")
                if text:
                    parts.append(text)
        if parts:
            return "".join(parts)
    return str(content)


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
            answer = extract_text(response.content)
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
    answer = extract_text(final.content)
    history.append(AIMessage(content=answer))
    return answer
