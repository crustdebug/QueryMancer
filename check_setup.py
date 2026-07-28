#!/usr/bin/env python3
"""Verify the database connection and API key pools before running the app.

Usage:  python check_setup.py
"""

import sys

from dotenv import load_dotenv

load_dotenv()

from config import Config  # noqa: E402


def check_database() -> bool:
    print("Database")
    print(f"  target: {Config.Postgres.user}@{Config.Postgres.host}:"
          f"{Config.Postgres.port}/{Config.Postgres.dbname}")
    try:
        from tools import with_sql_cursor

        with with_sql_cursor() as cursor:
            cursor.execute("SELECT current_database();")
            (name,) = cursor.fetchone()
            cursor.execute(
                "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public';"
            )
            (table_count,) = cursor.fetchone()
    except Exception as error:  # noqa: BLE001
        print(f"  FAILED: {error}")
        return False

    print(f"  connected to '{name}' with {table_count} table(s) in the public schema")

    # Read the schema the same way the agent does, so this reports exactly what
    # the agent will see.
    try:
        from tools import get_schema

        db = get_schema(refresh=True)
    except Exception as error:  # noqa: BLE001
        print(f"  schema introspection FAILED: {error}")
        return False

    populated = [t for t in db.tables if t.estimated_rows > 0]
    print(f"  agent sees {len(db.tables)} readable table(s) across "
          f"{len({t.schema for t in db.tables})} schema(s)")
    print(f"  {len(db.foreign_keys)} foreign key(s) available for joins")
    if populated:
        largest = max(populated, key=lambda t: t.estimated_rows)
        print(f"  largest table: {largest.qualified} (~{largest.estimated_rows:,} rows)")
    else:
        print("  note: every table appears empty. Queries will return no rows.")
        print("        If you just loaded data, run ANALYZE; to refresh statistics.")
    return True


def check_keys() -> bool:
    print("\nAPI keys")
    total = 0
    for provider, keys in Config.API_KEYS.items():
        if keys:
            print(f"  {provider.value:12} {len(keys)} key(s) in the pool")
            total += len(keys)
    if not total:
        print("  No API keys found. Add GOOGLE_API_KEY to your .env file.")
        return False

    chain = Config.model_chain()
    print("\nModel chain (tried in order)")
    for index, model in enumerate(chain, 1):
        pool = len(Config.credentials(model.provider).keys) or 1
        print(f"  {index}. {model.provider.value}/{model.name} ({pool} key(s))")
    if len(chain) == 1:
        print("\n  Only one model is configured. Add keys for another provider")
        print("  (for example GROQ_API_KEY) so the app can fall back when")
        print("  your Gemini quota runs out.")
    return True


def check_live_call() -> bool:
    print("\nLive model call")
    try:
        from models import RotatingChatModel

        model = RotatingChatModel()
        response = model.invoke([("human", "Reply with the single word: pong")])
    except Exception as error:  # noqa: BLE001
        print(f"  FAILED: {type(error).__name__}: {str(error)[:200]}")
        return False
    served = f"{model.active_model.provider.value}/{model.active_model.name}"
    print(f"  ok - answered by {served}: {str(response.content)[:40]!r}")
    if model.last_usage:
        print(f"  tokens: {model.last_usage}")
    return True


if __name__ == "__main__":
    db_ok = check_database()
    keys_ok = check_keys()
    call_ok = check_live_call() if keys_ok else False

    print()
    if db_ok and call_ok:
        print("Ready. Run: streamlit run app.py")
        sys.exit(0)
    print("Setup is incomplete - see the messages above.")
    sys.exit(1)
