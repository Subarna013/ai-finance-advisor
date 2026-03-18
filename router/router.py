from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "wiki_search"] = Field(
        ...,
        description="Route question to finance vectorstore or wikipedia"
    )

def get_router(llm):
    structured_llm = llm.with_structured_output(RouteQuery)

    system = """You are an expert router.

Rules:
- Use 'vectorstore' for finance-related questions:
  (stocks, inflation, mutual funds, portfolio, risk, RBI, SEBI, investing)

- Use 'wiki_search' for:
  (history, people, general knowledge, non-finance topics)
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{question}")
    ])

    return prompt | structured_llm