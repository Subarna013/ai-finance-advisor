from config.settings import *
from ingestion.loader import load_and_split
from embeddings.embedding_model import get_embeddings
from vectorstore.astra_db import init_db
from tools.wiki_tool import get_wiki_tool
from router.router import get_router
from graph.workflow import build_workflow
from graph.state import GraphState
from graph import nodes

from langchain_groq import ChatGroq


# 🔥 User Profile
def get_user_profile():
    print("\n📊 Setup your financial profile:")

    age = input("Age: ")
    income = input("Monthly income (₹): ")
    risk = input("Risk level (low / medium / high): ")

    return {
        "age": age,
        "income": income,
        "risk": risk
    }


def main():
    print("🚀 Starting Finance Advisor AI...\n")

    # Step 1: Load data
    docs = load_and_split()

    # Step 2: Embeddings
    embeddings = get_embeddings()

    # Step 3: Vector DB
    store = init_db(embeddings, docs)
    retriever = store.as_retriever()

    # Step 4: Tools
    wiki = get_wiki_tool()

    # Step 5: LLM
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile"
    )

    # Step 6: Router
    router = get_router(llm)

    # Step 7: Build workflow
    app = build_workflow(router, retriever, wiki, llm, GraphState, nodes)

    print("✅ Finance Advisor AI Ready!\n")

    # 🔥 Collect user profile
    user_profile = get_user_profile()

    # 🔥 NEW: Initialize memory
    chat_history = []

    # Step 8: Chat loop
    while True:
        question = input("\n💬 Ask: ")

        if question.lower() == "exit":
            print("👋 Exiting...")
            break

        # 🔥 UPDATED: pass memory + profile
        result = app.invoke({
            "question": question,
            "user_profile": user_profile,
            "chat_history": chat_history
        })

        print("\n📊 Answer:\n")
        print(result["generation"])
        print("\n" + "-"*50)

        # 🔥 NEW: store conversation
        chat_history.append({
            "question": question,
            "answer": result["generation"]
        })


if __name__ == "__main__":
    main()