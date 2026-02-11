# src/rag/chain.py

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

llm = ChatAnthropic(model="claude-sonnet-4-5-20250929")

prompt = ChatPromptTemplate.from_template("""
Answer the question using the context below.
If the context doesn't contain relevant information, respond with
"No matching information found."

Context:
{context}

Question: {input}
Answer:
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_chain(vectorstore):
    """Create a RAG chain from a vectorstore"""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    chain = (
        {
            "context": retriever | format_docs,
            "input": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain