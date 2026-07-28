import random

import streamlit as st
from dotenv import load_dotenv

# Load environment variables before importing anything that reads them.
load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: E402

from agent import ask, create_history  # noqa: E402
from config import Config  # noqa: E402
from key_pool import PoolExhausted  # noqa: E402
from models import RotatingChatModel  # noqa: E402
from tools import allowed_tables, get_available_tools, with_sql_cursor  # noqa: E402

LOADING_MESSAGES = [
    "Consulting the ancient tomes of SQL wisdom...",
    "Casting query spells on your database...",
    "Summoning data from the digital realms...",
    "Deciphering your request into database runes...",
    "Brewing a potion of perfect query syntax...",
    "Channeling the power of database magic...",
    "Translating your words into the language of tables...",
    "Waving my SQL wand to fetch your results...",
    "Performing database divination...",
    "Aligning the database stars for optimal results...",
    "Consulting with the database spirits...",
    "Transforming natural language into database incantations...",
    "Peering into the crystal ball of your database...",
    "Opening a portal to your data dimension...",
    "Enchanting your request with SQL magic...",
    "Invoking the ancient art of query optimization...",
    "Reading between the tables to find your answer...",
    "Conjuring insights from your database depths...",
    "Weaving a tapestry of joins and filters...",
    "Preparing a feast of data for your consideration...",
]

st.set_page_config(
    page_title="Querymancer",
    page_icon="🧙‍♂️",
    layout="centered",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner=False)
def get_model() -> RotatingChatModel:
    """Build the rotating model once per session and reuse it.

    Caching matters here: the pool's cooldown state lives on this object, so
    rebuilding it on every rerun would forget which keys are rate-limited.
    """
    return RotatingChatModel().bind_tools(get_available_tools())


@st.cache_data(ttl=300, show_spinner=False)
def load_table_overview():
    """Table names and row counts for the sidebar, refreshed every 5 minutes."""
    allowed = allowed_tables()
    with with_sql_cursor() as cursor:
        if allowed:
            cursor.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' AND lower(table_name) IN %s "
                "ORDER BY table_name;",
                (tuple(sorted(allowed)),),
            )
        else:
            cursor.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' ORDER BY table_name;"
            )
        names = [row[0] for row in cursor.fetchall()]

        # An estimate from the planner's statistics, which is effectively free.
        # COUNT(*) on every table would scan the whole database on each load.
        cursor.execute(
            "SELECT relname, GREATEST(n_live_tup, 0) FROM pg_stat_user_tables "
            "WHERE schemaname = 'public';"
        )
        counts = dict(cursor.fetchall())
    return [(name, counts.get(name)) for name in names]


def load_css(css_file):
    try:
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except OSError:
        pass


load_css("assets/style.css")

st.header("Querymancer")
st.subheader("Talk to your database using natural language")

# --- Sidebar -------------------------------------------------------------

with st.sidebar:
    st.write("# Database")
    st.write(f"**Database:** {Config.Postgres.dbname}")
    st.write(f"**User:** {Config.Postgres.user}")
    st.write(f"**Host:** {Config.Postgres.host}:{Config.Postgres.port}")

    try:
        tables = load_table_overview()
        st.write(f"**Tables:** {len(tables)}")
        with st.expander("Show tables"):
            for name, count in tables:
                label = f"{count:,} rows" if count is not None else "unknown size"
                st.write(f"- {name} ({label})")
    except Exception as error:  # noqa: BLE001
        st.error(f"Cannot reach the database: {error}")

    st.write("# Model")
    try:
        model = get_model()
        st.write(f"**Active:** {model.active_model.provider.value}/{model.active_model.name}")
        status = model.pool_status()
        st.write(f"**Keys in rotation:** {len(status)}")
        with st.expander("Key pool status"):
            for row in status:
                st.write(
                    f"- `{row['key']}` · {row['model']} · {row['status']} · "
                    f"{row['calls']} calls, {row['rate_limits']} limits"
                )
        if model.session_usage:
            st.write("# Token usage (session)")
            st.write(f"**Input:** {model.session_usage.get('input_tokens', 0):,}")
            st.write(f"**Output:** {model.session_usage.get('output_tokens', 0):,}")
            st.write(f"**Total:** {model.session_usage.get('total_tokens', 0):,}")
    except Exception as error:  # noqa: BLE001
        st.error(str(error))

# --- Chat ----------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = create_history()

for message in st.session_state.messages:
    if isinstance(message, SystemMessage):
        continue
    is_user = isinstance(message, HumanMessage)
    with st.chat_message("user" if is_user else "ai", avatar="🧑‍💻" if is_user else "🤖"):
        st.markdown(message.content)

if prompt := st.chat_input("Ask a question about your data..."):
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()
        with st.spinner(random.choice(LOADING_MESSAGES)):
            try:
                model = get_model()
                answer = ask(prompt, st.session_state.messages, model)
            except PoolExhausted as exhausted:
                answer = (
                    f"**All API keys are currently rate-limited.**\n\n{exhausted}\n\n"
                    "Add more keys to the pool in your `.env`, or wait for the "
                    "cooldown to elapse."
                )
                st.session_state.messages.append(AIMessage(content=answer))
            except Exception as error:  # noqa: BLE001
                answer = f"**Something went wrong.**\n\n```\n{error}\n```"
                st.session_state.messages.append(AIMessage(content=answer))

        placeholder.markdown(answer)

        usage = getattr(get_model(), "last_usage", {})
        if usage:
            st.caption(
                f"{usage.get('input_tokens', 0):,} in · "
                f"{usage.get('output_tokens', 0):,} out · "
                f"{usage.get('total_tokens', 0):,} total tokens"
            )
