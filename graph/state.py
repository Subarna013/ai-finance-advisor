from typing import List, Dict
from typing_extensions import TypedDict
from langchain_core.documents import Document

class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Document]
    user_profile: Dict
    stock_data: dict
    chat_history: list
    portfolio: str