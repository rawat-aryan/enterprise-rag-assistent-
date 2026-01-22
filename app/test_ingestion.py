from ingestion.loader import load_pdf
from ingestion.splitter import split_documents

docs = load_pdf("data/raw_docs/sample.pdf")
chunks = split_documents(docs)

print("Total pages:", len(docs))
print("Total chunks:", len(chunks))
print("\nSample chunk:\n")
print(chunks[0].page_content)
print("\nMetadata:\n")
print(chunks[0].metadata)