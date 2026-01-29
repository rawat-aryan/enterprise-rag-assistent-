from fastapi import FastAPI
from pydantic import BaseModel

from ingestion.loader import load_pdf 
from ingestion.splitter import split_documents
from vectorstore.faiss_store import build_faiss_index, load_faiss_index

from langchain_google_genai import ChatGoogleGenerativeAI
from config.settings import GOOGLE_API_KEY

from rag.query_rewriter import build_query_rewriter
from rag.rag_chain import build_rag_chain
from graph.graph import build_graph


#endpoint
app = FastAPI(title="Enterprise Agentic RAG API")

class QueryRequest(BaseModel):
    question: str

# Load and prepare documents
print("Initializing RAG System...")

docs = load_pdf()
chunks = split_documents(docs)

vectorstore = load_faiss_index()
if not vectorstore:
    vectorstore = build_faiss_index(chunks)

llm  = ChatGoogleGenerativeAI(google_api_key=GOOGLE_API_KEY,
                                  model="gemini-2.5-flash",
                                    temperature=0,
                                    )
rewriter = build_query_rewriter(llm)
rag_chain, retriever = build_rag_chain(vectorstore, llm)
graph_app = build_graph(rewriter, retriever, rag_chain)

print("RAG System Initialized.")

@app.post("/ask")
def ask_question(request: QueryRequest):
    result = graph_app.invoke({"question": request.question})

    sources = []
    if "documents" in result:
        for doc in result["documents"]:
            sources.append({
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page")
            })
    return {
        "question": request.question,
        "answer": result["answer"],
        "sources": sources
    }   
@app.get("/")
def home():
    return {"message": "Enterprise RAG API is running. Go to /docs"}
