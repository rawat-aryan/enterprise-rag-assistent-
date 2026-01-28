def route_docs(state):
    docs = state["documents"]
    if len(docs) ==0:
        return "tool_fallback"
    return "answer"