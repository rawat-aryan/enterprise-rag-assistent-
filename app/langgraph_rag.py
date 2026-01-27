from ingestion.loader import load_pdf
from ingestion.splitter import split_documents
from vectorstore.faiss_store import load_faiss_index, build_faiss_index

from langchain_google_genai import ChatGoogleGenerativeAI
from config.settings import GOOGLE_API_KEY

from rag.query_rewriter import build_query_rewriter
from rag.rag_chain import build_rag_chain
from graph.graph import build_graph

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
app = build_graph(rewriter, retriever, rag_chain)

print("\n✅ LangGraph Agentic RAG Ready\n")

while True:
    q = input("Ask: ")
    if q.lower() == "exit":
        break

    result = app.invoke({"question": q})

    print("\nAnswer:\n", result["answer"])
