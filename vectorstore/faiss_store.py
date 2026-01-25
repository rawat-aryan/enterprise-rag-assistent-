import os
from langchain_community.vectorstores import FAISS
from vectorstore.embeddings import get_embeddings

INDEX_PATH = "data/faiss_index"
def build_faiss_index(chunks):
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_PATH)
    return vectorstore

def load_faiss_index():
    embeddingd = get_embeddings()
    if os.path.exists(INDEX_PATH):
        return FAISS.load_local(INDEX_PATH, embeddingd,allow_dangerous_deserialization=True)
    return None