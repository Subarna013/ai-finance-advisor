from tools.stock_tool import get_stock_data


def stock_analysis(state):
    question = state["question"]

    # simple ticker detection
    words = question.upper().split()
    ticker = None

    for w in words:
        if len(w) <= 5:  # basic filter
            ticker = w
            break

    if ticker:
        stock_data = get_stock_data(ticker)
    else:
        stock_data = {}

    return {
        "stock_data": stock_data,
        "question": question,
        "documents": state.get("documents", []),
        "user_profile": state.get("user_profile", {})
    }

def generate(state, llm):
    context = "\n\n".join([doc.page_content for doc in state["documents"]])
    profile = state.get("user_profile", {})
    stock = state.get("stock_data", {})
    history = state.get("chat_history", [])
    portfolio = state.get("portfolio", "")

    history_text = "\n".join([
        f"User: {h['question']}\nAI: {h['answer']}"
        for h in history[-5:]
    ])

    prompt = f"""
You are an advanced financial advisor.

User Profile:
- Age: {profile.get("age")}
- Income: {profile.get("income")}
- Risk: {profile.get("risk")}

Previous Conversation:
{history_text}

Stock Data:
{stock}

Portfolio:
{portfolio}

Rules:
- If portfolio is given → analyze diversification and risk
- Suggest improvements
- Align with user's risk profile
- Be practical and simple

Context:
{context}

Question: {state['question']}
"""

    response = llm.invoke(prompt)

    return {
        "generation": response.content,
        "documents": state["documents"],
        "question": state["question"],
        "user_profile": profile,
        "stock_data": stock,
        "chat_history": history,
        "portfolio": portfolio
    }


def extract_portfolio(state):
    question = state["question"]

    # simple detection
    if "%" in question:
        portfolio = question
    else:
        portfolio = ""

    return {
        "portfolio": portfolio,
        "question": question,
        "documents": state.get("documents", []),
        "user_profile": state.get("user_profile", {}),
        "stock_data": state.get("stock_data", {}),
        "chat_history": state.get("chat_history", [])
    }

def retrieve(state, retriever):
    question = state["question"]

    docs = retriever.invoke(question)

    return {
        "documents": docs,
        "question": question,
        "user_profile": state.get("user_profile", {}),
        "chat_history": state.get("chat_history", [])
    }
