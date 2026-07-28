import random
import streamlit as st
from dotenv import load_dotenv

# Load environment variables first, before importing config
load_dotenv()

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from agent import ask, create_history
from config import Config, ModelProvider
from models import create_llm
from tools import get_available_tools, with_sql_cursor
import json
import pandas as pd
import altair as alt
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

load_dotenv()

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

@st.cache_resource(show_spinner=False)
def get_model() -> BaseChatModel:
    llm = create_llm(Config.MODEL)
    if Config.MODEL.provider != ModelProvider.TOGETHER:
        llm = llm.bind_tools(get_available_tools())
    return llm

def load_css(css_file):
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(
    page_title="Querymancer",
    page_icon="🧙‍♂️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

load_css("assets/style.css")

st.header("Querymancer")
st.subheader("Talk to your database using natural language")

with st.sidebar:
    st.write("# Database Information")
    st.write(f"**Database:** {Config.Postgres.dbname}")
    st.write(f"**User:** {Config.Postgres.user}")
    st.write(f"**Host:** {Config.Postgres.host}")
    st.write(f"**Port:** {Config.Postgres.port}")

    with with_sql_cursor() as cursor:
        if Config.ALLOWED_TABLES:
            tables_list = tuple(Config.ALLOWED_TABLES)
            cursor.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name IN %s;",
                (tables_list,)
            )
        else:
            cursor.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"
            )
        tables = [row[0] for row in cursor.fetchall()]
        st.write("**Tables:**")
        for table in tables:
            cursor.execute(f"SELECT count(*) FROM {table};")
            count = cursor.fetchone()[0]
            st.write(f"- {table} ({count} rows)")

if "messages" not in st.session_state:
    st.session_state.messages = create_history()

for message in st.session_state.messages:
    if type(message) is SystemMessage:
        continue
    is_user = type(message) is HumanMessage
    avatar = "🧑‍💻" if is_user else "🤖"
    with st.chat_message("user" if is_user else "ai", avatar=avatar):
        st.markdown(message.content)

if prompt := st.chat_input("Type your message..."):
    with st.chat_message("user", avatar="🧑‍💻"):
        st.session_state.messages.append(HumanMessage(prompt))
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        message_placeholder.status(random.choice(LOADING_MESSAGES), state="running")

        response = ask(prompt, st.session_state.messages, get_model())
        
        input_tokens = len(tokenizer.encode(prompt))
        output_tokens = len(tokenizer.encode(response))
        
        st.write(f"Input Tokens: {input_tokens}")
        st.write(f"Output Tokens: {output_tokens}")

        message_placeholder.markdown(response)
        st.session_state.messages.append(AIMessage(response))