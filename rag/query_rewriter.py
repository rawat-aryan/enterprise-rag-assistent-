from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def build_query_rewriter(llm):
    prompt= ChatPromptTemplate.from_template("""
You are a query rewriting assistant.

Rewrite the user question into a clear standalone query
that can be used for document retrieval.

User Question:
{question}

Standalone Retrieval Query:
""")
    return prompt | llm | StrOutputParser()

