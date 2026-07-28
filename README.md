# Querymancer: SQL AI Agent

Ask questions about a PostgreSQL database in plain English. An LLM agent
explores the schema with read-only tools, writes the SQL itself, runs it, and
answers in Markdown.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt

cp .env.example .env          # then fill in your database and API keys
streamlit run app.py
```

Optional, for the fuzzy-search tools:

```sql
CREATE EXTENSION IF NOT EXISTS pg_trgm;
```

## Using several free API keys

Free tiers are capped per key, so the app treats each provider as a **pool** of
interchangeable keys and rotates through them round-robin. Put several in
`.env`, either comma-separated or numbered:

```dotenv
GOOGLE_API_KEY=key_one,key_two,key_three
# or
GOOGLE_API_KEY_1=key_one
GOOGLE_API_KEY_2=key_two
```

Gemini's free quota is granted per Google Cloud *project*, so create a new
project for each key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
Three keys give roughly three times the daily requests.

When a key returns a rate-limit error it is benched for
`Config.KEY_COOLDOWN_SECONDS` and the next key takes over. A key rejected for
bad credentials is retired immediately rather than retried. When every key for
a model is unavailable, the app falls back to the next entry in
`Config.FALLBACK_MODELS` — by default `gemini-2.0-flash-lite`, then Groq, then
Perplexity, then a local Ollama model. Each provider has separate quota, so
adding a Groq key roughly doubles your headroom again.

The sidebar shows each key's status, call count, and rate-limit count.

## Staying inside your limits

Cost in an agent loop is dominated by tool output being resent on every step,
so the defaults in [config.py](config.py) target that directly:

| Setting | Default | Effect |
|---|---|---|
| `MAX_AGENT_ITERATIONS` | 8 | Hard cap on LLM calls per question |
| `MAX_TOOL_RESULT_CHARS` | 4000 | Truncates any single tool result |
| `MAX_SQL_RESULT_ROWS` | 50 | Caps rows returned to the model |
| `MAX_HISTORY_MESSAGES` | 12 | Drops old turns so prompts stay flat |

Schema lookups (`list_tables`, `describe_table`,
`get_foreign_key_relationships`) are cached in memory for the process, so
repeated questions about the same tables cost no extra requests. The system
prompt also tells the model to reuse schemas already in the conversation rather
than re-fetching them.

If you hit limits often, lower `MAX_AGENT_ITERATIONS` to 5 and switch
`Config.MODEL` to `GEMINI_FLASH_LITE`, which has a larger free allowance.

## Accuracy

The agent is steered toward correctness by construction rather than by asking
nicely: it must call `describe_table` before writing SQL against a table, must
call `get_foreign_key_relationships` before a JOIN, and is told to check real
values with `get_distinct_column_values` before filtering on a category.
`temperature` is 0 for every model.

## Safety

Queries run in a PostgreSQL **read-only session**, so writes are rejected by
the database itself. On top of that, `execute_sql` accepts only a single
`SELECT`/`WITH` statement, and every tool checks the requested table against
`Config.ALLOWED_TABLES`. Set that list to `[]` to allow the whole `public`
schema.

## Configuration

Everything is in [config.py](config.py): `MODEL` and `FALLBACK_MODELS` choose
the models, `ALLOWED_TABLES` restricts access, and the limits above control
spend.

## Tests

```bash
pytest
```

The suite covers key rotation, failover, read-only enforcement, and the agent
loop. It uses a stub model and needs no database or API key.
