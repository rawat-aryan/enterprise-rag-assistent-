from langgraph.graph import StateGraph, END
from graph.state import RAGState
from graph.nodes import rewrite_node,retry_node,retrieve_node,answer_node
from graph.router import route_docs

def build_graph(rewriter, retriever, rag_chain):
    graph = StateGraph(RAGState)

    graph.add_node("rewrite",lambda s: rewrite_node(s,rewriter))
    graph.add_node("retry", lambda s: retry_node(s,rewriter))
    graph.add_node("retrieve", lambda s: retrieve_node(s,retriever))
    graph.add_node("answer", lambda s: answer_node(s,rag_chain))

    graph.set_entry_point("rewrite")
    graph.add_edge("rewrite","retrieve")

    graph.add_conditional_edges(
        "retrieve",
        route_docs,
        {
            "answer":"answer",
            "retry":"retry"
         }
    )

    graph.add_edge("retry","answer")
    graph.add_edge("answer",END)    

    return graph.compile()