from ingestion.loader import  load_pdf 
from ingestion.splitter import split_documents
from vectorstore.faiss_store import build_faiss_index, load_faiss_index

from langchain_google_genai import ChatGoogleGenerativeAI
from config.settings import GOOGLE_API_KEY

from rag.rag_chain import build_rag_chain
from rag.query_rewriter import build_query_rewriter 

docs = load_pdf()
chunks = split_documents(docs)


vectorstore = load_faiss_index()
if not vectorstore:
    vectorstore = build_faiss_index(chunks)
else:
    print("FAISS index loaded from disk.")  


llm  = ChatGoogleGenerativeAI(google_api_key=GOOGLE_API_KEY,
                                  model="gemini-2.5-flash",
                                  temperature=0,
                                  )

rewriter = build_query_rewriter(llm)


rag_chain, retriever = build_rag_chain(vectorstore, llm)

print("\n RAG ChatBot Ready  (type exit to stop) \n")

while True:
    q = input("Ask: ")
    if q.lower() == "exit":
        break 

    rewrittern_query = rewriter.invoke({"question": q})
    print(f"\n Rewrittern Query: {rewrittern_query}\n")

    docs = retriever.invoke(rewrittern_query) 


    answer = rag_chain.invoke(rewrittern_query)
    print(f"\n Answer: {answer}\n")
    print("\nSources: ")

    for doc in docs:
        print(f"- {doc.metadata.get('source')} (page {doc.metadata.get('page')})")
