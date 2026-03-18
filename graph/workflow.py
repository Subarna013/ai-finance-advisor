from langgraph.graph import StateGraph, START, END

def build_workflow(router, retriever, wiki, llm, GraphState, nodes):

    workflow = StateGraph(GraphState)

    # Nodes
    workflow.add_node("retrieve", lambda s: nodes.retrieve(s, retriever))
    workflow.add_node("wiki", lambda s: nodes.wiki_search(s, wiki))
    workflow.add_node("generate", lambda s: nodes.generate(s, llm))
    workflow.add_node("stock", lambda s: nodes.stock_analysis(s))
    workflow.add_node("portfolio", lambda s: nodes.extract_portfolio(s))

    # Router logic
    def route(state):
        result = router.invoke({"question": state["question"]})
        return "wiki" if result.datasource == "wiki_search" else "retrieve"

    # Start → route
    workflow.add_conditional_edges(
        START,
        route,
        {
            "wiki": "wiki",
            "retrieve": "retrieve"
        }
    )

    # Flow to generation
    workflow.add_edge("retrieve", "stock")
    workflow.add_edge("wiki", "stock")
    workflow.add_edge("stock", "portfolio")
    workflow.add_edge("portfolio", "generate")
    # End
    workflow.add_edge("generate", END)

    return workflow.compile()