<div align="center">

# QueryMancer

**Ask any SQL database questions in plain English.**

An LLM agent explores your schema with read-only tools, writes the SQL itself,
runs it, and answers with the results — plus the exact query it ran, so you can
check its work.

[**▶ Try the live demo**](https://querymancer.onrender.com) &nbsp;·&nbsp;
[Architecture](#architecture) &nbsp;·&nbsp;
[Engineering notes](#engineering-notes)

[![Tests](https://img.shields.io/badge/tests-236%20passing-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/python-3.12+-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

</div>

![QueryMancer answering a question about a live PostgreSQL database](docs/images/querymancer.png)

> The live demo ships with a sample store database — click **Try the sample
> database** and ask away, no setup required. It is hosted on a free tier that
> sleeps when idle, so the first request may take ~40 seconds to wake.

## What it does

Point it at PostgreSQL, MySQL/MariaDB, SQLite, SQL Server or Oracle — via a
form or a connection string — and ask questions. The schema is discovered at
runtime, so nothing about your tables is hardcoded and there is no
configuration file to maintain.

|  | |
|---|---|
| **Writes its own SQL** | An 8-step tool loop: inspect the schema, sample rows, learn categorical values, then query. Every statement it runs is shown to you. |
| **Survives free-tier quotas** | Round-robin API-key rotation with per-key cooldowns, then automatic fallback across Gemini → Groq → Perplexity → local Ollama. |
| **Fixes its own mistakes** | When the model writes `customer.customer_name` and the table is really `"Customer"."name"`, the query is rewritten against the live schema instead of failing. |
| **Reads 100-table schemas** | Keyword scoring plus one-hop foreign-key traversal prunes the prompt to the tables a question actually needs. |
| **Never writes to your data** | Read-only enforced twice: at the database session and by parsing every statement before it runs. |
| **Keeps credentials off disk** | Held in server memory for one session, never persisted, never returned to the browser. |

## Try it locally

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt

cp .env.example .env          # add your API key(s) - no database details needed
python -m uvicorn server:app  # then open http://127.0.0.1:8000
```

Then connect your database in the browser. Only API keys go in `.env`.

Add `--reload` while developing, or `--host 0.0.0.0 --port 8000` to expose it
on your network.

## The interface

A single-page app served by the FastAPI backend — no build step, no npm.

- **Pick a database** from the start screen, then enter details or paste a
  connection string.
- **Ask in plain English.** Answers arrive as formatted text, with results
  rendered as a real table and a bar chart when the shape suits it.
- **View SQL** under each answer shows the exact query that ran, including any
  names QueryMancer corrected for you.
- **Recent questions** in the sidebar keeps each conversation; follow-up
  questions stay in context.
- **Starter questions** are generated from your actual tables and columns, not
  hardcoded, so they name things that really exist.

To check your keys work before starting:

```bash
python check_setup.py
python check_setup.py postgresql://user:password@host:5432/db   # also test a database
```

## Architecture

```
Browser ──── FastAPI (server.py) ──── RotatingChatModel (models.py)
  │              │                          │
  │              │                          ├── KeyPool ── round-robin, cooldowns
  │              │                          └── fallback chain across providers
  │              │
  │              ├── AnswerCache ── repeat questions skip the LLM entirely
  │              │
  │              └── Session ── credentials in memory, one per browser
  │                     │
  │                     └── DatabaseConnection ── read-only, statement timeout
  │                              │
  └── HttpOnly cookie            ├── schema.py ── runtime introspection + pruning
      (opaque session id)        ├── sql_repair.py ── fixes wrong identifiers
                                 └── tools.py ── the 7 tools the agent may call
```

**The agent loop** (`agent.py`). The model is given seven read-only tools and
iterates up to eight times: orient with `inspect_database`, drill in with
`describe_table` or `sample_table`, check real values with
`get_distinct_column_values`, then `execute_sql`. Each tool result is truncated
before it re-enters the prompt, because in an agent loop the dominant cost is
resending prior output on every step.

| Module | Responsibility |
|---|---|
| [`agent.py`](agent.py) | The tool-calling loop and system prompt |
| [`tools.py`](tools.py) | Seven read-only tools; read-only enforcement; query trace |
| [`schema.py`](schema.py) | Runtime introspection, fuzzy name resolution, FK-subgraph pruning |
| [`sql_repair.py`](sql_repair.py) | Rewrites wrong identifiers against the real schema |
| [`models.py`](models.py) | Key rotation and cross-provider model fallback |
| [`key_pool.py`](key_pool.py) | Per-key rate-limit classification and cooldowns |
| [`connection.py`](connection.py) | Multi-engine connections, read-only, statement timeouts |
| [`session.py`](session.py) | Per-browser sessions; credentials never touch disk |
| [`answer_cache.py`](answer_cache.py) | TTL'd answer cache keyed by question + database |
| [`server.py`](server.py) | HTTP API, access gate, static hosting |

Roughly 3,500 lines of application code and 1,700 lines of tests, no frontend
build step — the UI is hand-written HTML/CSS/JS served directly by FastAPI.

## Engineering notes

The interesting problems in this project were not "call an LLM" — they were
everything around it.

**Free-tier quotas are the real constraint.** A single Gemini key runs out
quickly under an agent loop that makes up to eight calls per question. So keys
rotate round-robin with per-key cooldowns, and the pool distinguishes failure
modes that deserve different handling: a per-minute 429 benches a key for
seconds, a per-day 429 benches it for far longer (Gemini reports which in the
error body, along with a `retryDelay` that is trusted over any guess), an auth
failure retires the key permanently, and a 503 rotates without penalising the
key at all. Above that, whole models fall back — including a 404 path added
after a Gemini model was silently deprecated mid-project.

**Large schemas do not fit in a prompt.** A 100-table database sent verbatim
costs tokens on every call and gives the model more chances to join the wrong
thing. Tables are scored against the words in the question — matching
`snake_case`, `camelCase` and `PascalCase`, and singular against plural — and
then one hop of foreign-key traversal pulls in the join targets a query needs
even when the question never names them. The output states how many tables were
hidden, so the model asks for the full map instead of concluding a table does
not exist.

**Models get names wrong.** Rather than returning a syntax error and paying for
another round trip, `sql_repair.py` resolves each identifier against the live
schema by similarity and rewrites the query, reporting every correction it made
so a wrong guess is visible rather than silent.

**Hosted databases have sharp edges.** Two bugs found only by deploying: query
parameters were dropped from connection strings, so every `?sslmode=require`
database refused the connection; and passing `statement_timeout` as a libpq
startup option is rejected outright by any pooler in transaction mode — it has
to be a `SET` on each connection instead.

**Privacy is a design constraint, not a feature.** Credentials live in server
memory for one session and are never written to disk or returned to the
browser. `QUERYMANCER_LOCAL_ONLY=1` restricts the model chain to locally-hosted
models so no data reaches a third party — enforced in `model_chain()` rather
than by asking users to delete their API keys, because Ollama sitting at the
end of a fallback chain would only be reached after every cloud key was
exhausted.

## Connecting your database

Choose your database type on the start screen, then either fill in the fields
or paste a connection string:

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
`Config.FALLBACK_MODELS` — by default `gemini-3.1-flash-lite`, then the other
Gemini models, then Groq, then Perplexity, then a local Ollama model. Each
provider has separate quota, so adding a Groq key roughly doubles your
headroom again.

The sidebar shows each key's status, call count, and rate-limit count.

## Private mode: keeping your data off third-party APIs

Answering a question means sending rows of real data to whichever model is
serving it. That is unavoidable for a cloud model — an LLM cannot tell you
which customer spent the most without seeing customer data.

If that is not acceptable for your data, run a local model instead:

```bash
# 1. Install Ollama and pull a model that supports tool calling
ollama pull qwen2.5

# 2. Turn on local-only mode
QUERYMANCER_LOCAL_ONLY=1 python -m uvicorn server:app
```

In this mode **no database content leaves your machine**. The model runs
locally, and cloud providers are excluded from the fallback chain entirely.

This is enforced in `Config.model_chain()` rather than left to you removing
your API keys, and the reason matters: an Ollama model sitting at the end of
the normal fallback chain is only reached once every cloud key is exhausted.
Without the flag, a `.env` with a leftover `GOOGLE_API_KEY` would quietly send
your data to Google despite you having installed Ollama specifically to avoid
that. With `QUERYMANCER_LOCAL_ONLY=1`, that key is ignored.

When the whole chain is local, the app shows a "your data never leaves this
machine" badge. It appears only when that is actually true — a chain that can
still reach a cloud provider shows nothing rather than a claim it cannot keep.

| Variable | Effect |
|---|---|
| `QUERYMANCER_LOCAL_ONLY=1` | Use only locally-hosted models |
| `OLLAMA_MODEL` | Which Ollama model to use (default `qwen2.5:latest`) |

Local models are slower and less accurate at SQL than Gemini. That is the
trade: correctness and speed against your data never leaving the building.

## Deploying it somewhere public

The [live demo](https://querymancer.onrender.com) runs on Render's free tier
from [`render.yaml`](render.yaml), with a Neon Postgres holding the sample data
seeded by [`seed_demo.py`](seed_demo.py). A [`Dockerfile`](Dockerfile) is
included for hosts that prefer a container.

To offer a one-click sample database of your own, set `QUERYMANCER_DEMO_URL` to
a **read-only** connection string. It is used server-side only: the browser
learns that a demo exists and what to call it, never where it lives or how to
reach it.

### Keeping a free instance awake

Free tiers sleep after ~15 minutes idle, and the next visitor waits for a cold
start. Point any uptime checker at `/healthz` every 10 minutes to prevent that:

| Setting | Value |
|---|---|
| URL | `https://your-app.onrender.com/healthz` |
| Type | HTTP(s) |
| Interval | 10 minutes (under the ~15 minute idle timeout) |

`/healthz` is exempt from the access code — a probe carries no cookie — and it
also builds the model chain, so the ping absorbs that cost instead of the first
visitor. It returns 200 even when no API key is configured, because a probe
that failed on configuration would make the platform roll back an otherwise
healthy deploy.

[UptimeRobot](https://uptimerobot.com) offers 50 monitors free at 5-minute
resolution and needs no card.

Three things to set before exposing this to the internet.

**1. Require an access code.** Without one, anyone who finds the URL can point
the app at a database.

```bash
QUERYMANCER_ACCESS_CODE=some-long-shared-secret
```

Every route is then gated behind an unlock page. This is applied as
middleware, so a route added later is protected by default.

**2. Serve it over HTTPS.** The session cookie is marked `Secure`
automatically when the request arrives over HTTPS, including via a proxy's
`X-Forwarded-Proto` header. Set `QUERYMANCER_FORCE_SECURE_COOKIE=1` to require
it unconditionally.

**3. Know that sessions are in-memory.** A restart or redeploy disconnects
everyone — they will need to re-enter their database credentials. That is
deliberate: credentials are never written to disk, so there is nothing to
persist and nothing to steal from the filesystem. It does mean the app does
not survive a restart gracefully, and does not run as multiple instances
behind a load balancer without sticky sessions.

### Database load

Pointing this at a database your main application also uses will affect that
application. The pool defaults to SQLAlchemy's settings (up to 15 connections
per session), sessions last 8 hours, and a generated query can scan a large
table for the full `STATEMENT_TIMEOUT_SECONDS` before being cut off.

Prefer a **read replica**. Failing that, give QueryMancer its own database
user with a hard connection limit, so it cannot starve your application:

```sql
CREATE USER querymancer WITH PASSWORD '...';
GRANT CONNECT ON DATABASE yourdb TO querymancer;
GRANT USAGE ON SCHEMA public TO querymancer;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO querymancer;
ALTER ROLE querymancer SET statement_timeout = '10s';
ALTER ROLE querymancer CONNECTION LIMIT 5;
```

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
pytest          # 236 tests, ~12s, no database or API key required
```

The suite runs against a stub model and temporary SQLite databases, so it needs
no credentials and no network.

| Area | What is covered |
|---|---|
| [`test_key_pool.py`](tests/test_key_pool.py) | Rotation, per-minute vs per-day quota handling, auth retirement, cooldown expiry |
| [`test_models.py`](tests/test_models.py) | Model-level fallback, and that genuine bugs are *not* swallowed as fallback triggers |
| [`test_schema_pruning.py`](tests/test_schema_pruning.py) | FK-subgraph selection, naming conventions, that a 100-table schema actually shrinks |
| [`test_sql_safety.py`](tests/test_sql_safety.py) | Write statements, multi-statement injection, comment evasion |
| [`test_connection.py`](tests/test_connection.py) | Multi-engine URLs, SSL parameters, statement timeouts, credential masking |
| [`test_privacy.py`](tests/test_privacy.py) | Local-only enforcement, access gate, cookie flags |
| [`test_server.py`](tests/test_server.py) | API surface, session isolation, answer caching, demo-credential withholding |

Several tests exist because they caught a real bug — the pooler rejecting a
startup parameter, a `Secure` cookie silently locking out plain-HTTP users, a
demo database's hostname leaking through a sidebar label. Each is named after
the behaviour it protects rather than the function it calls.
