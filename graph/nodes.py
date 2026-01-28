from graph.tools import keyword_fallback_tool

def rewrite_node(state, rewriter):
    question = state["question"]
    rewritten = rewriter.invoke({"question": question})
    return {"rewritten_query": rewritten}


def retrieve_node(state, retriever):
    query = state["rewritten_query"]
    docs = retriever.invoke(query)
    return {"documents": docs}


def answer_node(state, rag_chain):
    query = state["rewritten_query"]
    answer = rag_chain.invoke(query)
    return {"answer": answer}


def retry_node(state, retriever):
    # simple retry strategy: broaden query
    query = state["question"]
    docs = retriever.invoke(query)
    return {"documents": docs}


def tool_node(state):
    return keyword_fallback_tool(state)

