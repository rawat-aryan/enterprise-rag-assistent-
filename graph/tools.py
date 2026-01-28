def keyword_fallback_tool(state):
    """
    Simple fallback tool:
    If semantic retrieval fails, we try keyword-style retrieval.
    """
    question = state["question"]

    return {
        "rewritten_query": question + " (explain with keywords)"
    }
