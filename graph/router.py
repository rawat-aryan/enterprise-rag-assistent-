def route_docs(state):
    docs = state["documents"]
    if len(docs) ==0:
        return "retry"
    return "anwer"
