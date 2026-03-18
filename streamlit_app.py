import streamlit as st

from config.settings import *
from ingestion.loader import load_and_split
from embeddings.embedding_model import get_embeddings
from vectorstore.astra_db import init_db
from tools.wiki_tool import get_wiki_tool
from router.router import get_router
from graph.workflow import build_workflow
from graph.state import GraphState
from graph import nodes

import streamlit as st

st.title("🚀 App Loaded Successfully")
st.write("If you see this, UI is working")


from langchain_groq import ChatGroq


# 🔥 Cache heavy setup
@st.cache_resource
def setup_app():
    docs = load_and_split()
    embeddings = get_embeddings()

    store = init_db(embeddings, docs)
    retriever = store.as_retriever()

    wiki = get_wiki_tool()

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile"
    )

    router = get_router(llm)

    app = build_workflow(router, retriever, wiki, llm, GraphState, nodes)

    return app


# 🚀 Load system once
app = setup_app()

st.title("💰 AI Financial Advisor")

# 🔥 Sidebar profile
st.sidebar.header("📊 Your Profile")

age = st.sidebar.text_input("Age")
income = st.sidebar.text_input("Monthly Income (₹)")
risk = st.sidebar.selectbox("Risk Level", ["low", "medium", "high"])

user_profile = {
    "age": age,
    "income": income,
    "risk": risk
}

# 🔥 Memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 🔥 Chat UI
user_input = st.text_input("💬 Ask your financial question:")

if user_input:
    result = app.invoke({
        "question": user_input,
        "user_profile": user_profile,
        "chat_history": st.session_state.chat_history
    })

    answer = result["generation"]

    # Store memory
    st.session_state.chat_history.append({
        "question": user_input,
        "answer": answer
    })

    st.write("📊 **Answer:**")
    st.write(answer)

# 🔥 Show history
if st.session_state.chat_history:
    st.subheader("🧠 Conversation History")

    for chat in st.session_state.chat_history[::-1]:
        st.write(f"**You:** {chat['question']}")
        st.write(f"**AI:** {chat['answer']}")
        st.write("---")