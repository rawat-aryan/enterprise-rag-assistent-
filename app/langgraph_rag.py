from ast import main
from ingestion.loader import load_pdf
from ingestion.splitter import split_documents
from vectorstore.faiss_store import load_faiss_index, build_faiss_index

from langchain_google_genai import ChatGoogleGenerativeAI
from config.settings import GOOGLE_API_KEY

from rag.query_rewriter import build_query_rewriter
from rag.rag_chain import build_rag_chain
from graph.graph import build_graph

def main():
    # Load docs
    docs = load_pdf()
    chunks = split_documents(docs)

    # Load/build vectorstore
    vectorstore = load_faiss_index()
    if vectorstore is None:
        vectorstore = build_faiss_index(chunks)

    # LLM
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        temperature=0,
        google_api_key=GOOGLE_API_KEY
    )

    # Components
    rewriter = build_query_rewriter(llm)

    rag_chain, retriever = build_rag_chain(vectorstore, llm)

    # Graph
    graph_app = build_graph(rewriter, retriever, rag_chain)

    print("\nLangGraph Agentic RAG Ready\n")

    # Chat Loop
    while True:
        q = input("Ask: ").strip()

        # Exit condition
        if q.lower() == "exit":
            print("\n👋 Exiting chatbot...\n")
            break

        # Empty input handling
        if q == "":
            print("⚠️ Please type a valid question.\n")
            continue

        # Run LangGraph workflow
        result = graph_app.invoke({"question": q})

        # Print Answer
        print("\n====================")
        print("Answer:\n")
        print(result["answer"])
        print("====================\n")

        # Print Sources (Citations)
        if "documents" in result and len(result["documents"]) > 0:
            print("Sources Used:")
            for doc in result["documents"]:
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "N/A")
                print(f"- {source} (page {page})")
        else:
            print("No relevant sources found (fallback triggered).")

        print("\n---------------------------------\n")


if __name__ == "__main__":
    main()
