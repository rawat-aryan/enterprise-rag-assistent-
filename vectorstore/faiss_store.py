from langchain_community.vectorstores import FAISS
from vectorstore.embeddings import get_embeddings

def build_faiss_index(chunks):
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings      
    )
    return vectorstore

