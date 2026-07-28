# Querymancer: SQL AI Agent

Ask questions about a PostgreSQL database in plain English. An LLM agent
explores the schema with read-only tools, writes the SQL itself, runs it, and
answers in Markdown.

Point it at **any** database and it works: the schema is discovered at runtime,
so nothing about your tables is hardcoded. Change `POSTGRES_DB` in `.env` and
restart — there is no schema list to maintain.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt

cp .env.example .env          # then fill in your database and API keys
streamlit run app.py
```

Then check everything is wired up:

```bash
python check_setup.py
```

It reports how many tables the agent can see, how many API keys are in
rotation, and makes one live model call to confirm the whole path works.

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

The whole schema is read once and cached for the process, so `inspect_database`
and `describe_table` cost no extra database round trips and no extra LLM calls
on repeat questions. The system prompt also tells the model to reuse schema
information already visible in the conversation rather than re-fetching it.

If you hit limits often, lower `MAX_AGENT_ITERATIONS` to 5 and switch
`Config.MODEL` to `GEMINI_FLASH_LITE`, which has a larger free allowance.

## Working with an unfamiliar database

The agent has no built-in knowledge of your schema. [schema.py](schema.py) reads
the catalog on first use — every table, column, type, primary key, foreign key
and row estimate, across all non-system schemas — and caches it. A single
`inspect_database` call then orients the model, replacing a chain of
list-tables/describe-table round trips.

### Wrong names are corrected automatically

An LLM writing SQL for an unseen database guesses names from convention: it
writes `customers.customer_name` where your database has `"Customer"."name"`.
[sql_repair.py](sql_repair.py) maps the guess onto what actually exists before
the query runs, and tells the model what it changed so the rest of the
conversation uses the real names. A real example against a Prisma schema:

```sql
-- the model wrote
SELECT c.company_name, sum(o.total_amount)
FROM customers c JOIN sales_orders o ON o.customer_id = c.id
GROUP BY c.company_name

-- what actually ran
SELECT c."companyName", sum(o."totalAmount")
FROM "Customer" c JOIN "SalesOrder" o ON o."customerId" = c.id
GROUP BY c."companyName"
```

This handles snake_case ↔ camelCase ↔ PascalCase, singular/plural, and
Postgres's case-folding rule (an unquoted `Customer` means `customer`, so
mixed-case names must be quoted). Matching is fuzzy above
`Config.NAME_MATCH_CUTOFF`; below it the name is left alone and reported as
unresolved with suggestions, rather than being silently changed to something
wrong. Bare unqualified columns are deliberately not rewritten — that needs
scope analysis a regex cannot do safely — but Postgres's own "did you mean"
hint is passed back to the model, which recovers on the next step.

Don't know where a value lives? `find_value` searches label-like text columns
across the database and reports which table and column contains it.

If you change the schema while the app is running, click **Reload schema** in
the sidebar.

## Accuracy

Correctness is enforced by construction rather than by asking nicely: the model
must inspect the schema before writing SQL, joins use the discovered foreign
keys, and `get_distinct_column_values` supplies real category labels before a
`WHERE` clause is written. `temperature` is 0 for every model.

## Safety

Queries run in a PostgreSQL **read-only session**, so the database itself
rejects any write — that is the real guarantee, not a keyword filter. On top of
that, `execute_sql` accepts only a single `SELECT`/`WITH` statement, with string
literals and comments stripped before the check so a column named `updated_by`
or a literal containing `DELETE` is not misread as a write.

Within read-only, access is **unrestricted**: every table, column and row the
database user can see. The agent is as trusted as the credentials in your
`.env`, so point it at a user with only the privileges you want exposed — a
read-only role scoped to the right schemas is the right control here.

## Configuration

Everything is in [config.py](config.py): `MODEL` and `FALLBACK_MODELS` choose
the models, `NAME_MATCH_CUTOFF` tunes how aggressively names are corrected, and
the limits above control spend.

## Tests

```bash
pytest
```

The suite covers key rotation and failover, read-only enforcement, schema
discovery, name resolution across naming conventions, and the agent loop. It
uses a stub model and an in-memory schema, so it needs no database or API key.
