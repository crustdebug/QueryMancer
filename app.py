import random

import streamlit as st
from dotenv import load_dotenv

# Load environment variables before importing anything that reads them.
load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: E402

from agent import ask, create_history  # noqa: E402
from connection import sanitize  # noqa: E402
from connection_ui import render_connection_panel  # noqa: E402
from key_pool import PoolExhausted  # noqa: E402
from models import RotatingChatModel  # noqa: E402
from session import NotConnected, current_connection  # noqa: E402
from tools import get_available_tools, get_schema  # noqa: E402

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
    """Every readable table with its estimated size, for the sidebar."""
    db = get_schema()
    return [
        (table.qualified, table.estimated_rows, len(table.columns))
        for table in sorted(db.tables, key=lambda t: (-t.estimated_rows, t.name.lower()))
    ]


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
    connected = render_connection_panel()

    if connected:
        try:
            tables = load_table_overview()
            st.write(f"**Tables:** {len(tables)} (all readable)")
            with st.expander("Show tables"):
                for name, rows, columns in tables:
                    size = f"~{rows:,} rows" if rows else "empty"
                    st.write(f"- {name} — {size}, {columns} cols")
        except Exception as error:  # noqa: BLE001
            st.error(f"Could not read the schema: {error}")

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

if not connected:
    st.info(
        "**No database connected.** Open the sidebar, pick your database type, "
        "and enter your connection details or a connection string.",
        icon="🔌",
    )

placeholder_text = (
    "Ask a question about your data..." if connected else "Connect a database first"
)

if prompt := st.chat_input(placeholder_text, disabled=not connected):
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()
        with st.spinner(random.choice(LOADING_MESSAGES)):
            try:
                model = get_model()
                answer = ask(prompt, st.session_state.messages, model)
            except NotConnected as error:
                answer = f"**{error}**"
                st.session_state.messages.append(AIMessage(content=answer))
            except PoolExhausted as exhausted:
                answer = (
                    f"**All API keys are currently rate-limited.**\n\n{exhausted}\n\n"
                    "Add more keys to the pool in your `.env`, or wait for the "
                    "cooldown to elapse."
                )
                st.session_state.messages.append(AIMessage(content=answer))
            except Exception as error:  # noqa: BLE001
                # Driver errors can echo the connection URL, so mask before display.
                current = current_connection()
                detail = sanitize(error, current.settings if current else None)
                answer = f"**Something went wrong.**\n\n```\n{detail}\n```"
                st.session_state.messages.append(AIMessage(content=answer))

        placeholder.markdown(answer)

        usage = getattr(get_model(), "last_usage", {})
        if usage:
            st.caption(
                f"{usage.get('input_tokens', 0):,} in · "
                f"{usage.get('output_tokens', 0):,} out · "
                f"{usage.get('total_tokens', 0):,} total tokens"
            )
