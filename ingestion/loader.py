from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader

def load_pdf(folder_path = "data/raw_docs"):

    loader = DirectoryLoader(
        folder_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents