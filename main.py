import streamlit as st
from agent import ask, create_history
from config import MODEL_CONFIG

# Mock function to load model - replace with real LangChain/Ollama model loading
def load_model(config):
    # placeholder model loader
    return lambda history: type('Response', (), {'content': 'Mock response', 'tool_call': None})()

def main():
    st.title("Querymancer: SQL AI Agent")
    query = st.text_input("Ask a question about the database:")
    if "history" not in st.session_state:
        st.session_state.history = create_history()

    if query:
        response = ask(query, st.session_state.history, model=load_model(MODEL_CONFIG))
        st.markdown(response)

if __name__ == "__main__":
    main()
