from ingestion.loader import load_pdf
from ingestion.splitter import split_documents
from vectorstore.faiss_store import build_faiss_index

docs = load_pdf("data/raw_docs/sample.pdf")
chunks = split_documents(docs)

vectorstore = build_faiss_index(chunks)
results = vectorstore.similarity_search("What is this document about?",
                                         k=3
                                         )
for i, res in enumerate(results):
    print(f"Result {i+1}:\n")
    print(res.page_content[:300])