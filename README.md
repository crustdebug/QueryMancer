# Querymancer: SQL AI Agent

Ask questions about your database in plain English. An LLM agent explores the
schema with read-only tools, writes the SQL itself, runs it, and answers in
Markdown.

Connect **any** database from the sidebar — PostgreSQL, MySQL/MariaDB, SQLite,
SQL Server or Oracle — with a form or a connection string. The schema is
discovered at runtime, so nothing about your tables is hardcoded and there is
no configuration file to maintain.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt

cp .env.example .env          # add your API key(s) - no database details needed
streamlit run app.py
```

Then open the sidebar and connect your database. Only API keys go in `.env`.

To check your keys work before starting:

```bash
python check_setup.py
python check_setup.py postgresql://user:password@host:5432/db   # also test a database
```

## Connecting your database

Open the sidebar, choose your database type, and either fill in the fields or
paste a connection string:

| Engine | Connection string | Driver needed |
|---|---|---|
| PostgreSQL | `postgresql://user:pw@host:5432/db` | `psycopg2-binary` ✅ |
| MySQL / MariaDB | `mysql://user:pw@host:3306/db` | `pymysql` ✅ |
| SQLite | `sqlite:///C:/path/to/app.db` | none ✅ |
| SQL Server | `mssql://user:pw@host:1433/db` | `pyodbc` |
| Oracle | `oracle://user:pw@host:1521/db` | `oracledb` |

The first three work out of the box; uncomment the others in
`requirements.txt` if you need them. Once connected, the **Database** dropdown
switches between databases on the same server without re-entering credentials.

### How your credentials are handled

- They are kept **in your browser session only** — never written to a file,
  an environment variable, or a module global.
- Nothing is persisted: closing the tab or clicking **Disconnect** discards them.
- Passwords are masked everywhere they could surface — the displayed URL, the
  connection summary, log lines, and driver error messages, which often echo
  the connection URL back verbatim.
- On a shared or deployed instance, one person's connection is never visible to
  another, because session state is per-session by construction.

`DATABASE_URL` in `.env` is supported for local convenience only. It pre-fills
the form; it is not required and not treated as stored state.

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

Two independent layers block writes:

1. **The database itself.** Each engine's strongest available guarantee is
   applied — a read-only transaction on PostgreSQL, MySQL and Oracle, and an
   immutable file handle (`mode=ro`) on SQLite.
2. **Statement checking.** `execute_sql` accepts only a single `SELECT`/`WITH`
   statement. String literals and comments are stripped first, so a column
   named `updated_by` or a literal containing `DELETE` is not misread as a
   write, and a stacked `SELECT 1; DROP TABLE x` is rejected outright.

Within read-only, access is **unrestricted**: every table, column and row the
connected user can see. The agent is exactly as privileged as the credentials
you give it, so connecting a read-only database role is the right control if
you want a narrower boundary.

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
