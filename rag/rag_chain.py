from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def build_rag_chain(vectorstore, llm):

    retriever = vectorstore.as_retriever(
        search_type = "mmr",
        search_kwargs={"k": 3, "fetch_k": 8}
        )

    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.
Answer ONLY using the context below.

Context:
{context}

Question:
{question}

Answer:
""")
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain =(
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain,retriever