from ingestion.loader import  load_pdf 
from ingestion.splitter import split_documents
from vectorstore.faiss_store import build_faiss_index

from langchain_google_genai import ChatGoogleGenerativeAI
from config.settings import GOOGLE_API_KEY

from rag.rag_chain import build_rag_chain, build_rag_chain

docs = load_pdf("data/raw_docs/sample.pdf")
chunks = split_documents(docs)

vectorstore = build_faiss_index(chunks)

llm  = ChatGoogleGenerativeAI(google_api_key=GOOGLE_API_KEY,
                                  model="gemini-2.5-flash",
                                  temperature=0,
                                  )

rag_chain = build_rag_chain(vectorstore, llm)

print("\n RAG ChatBot Ready  (type exit to stop) \n")

while True:
    q = input("Ask: ")
    if q.lower() == "exit":
        break   

    answer = rag_chain.invoke(q)
    print("\nANswer:\n", answer)