# src/rag/chain.py

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

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

def build_rag_chain(retriever):
    """Create a RAG chain from a retriever"""
    
    # Wrap retriever in RunnableLambda to make it pipeable
    def retrieve_and_format(query):
        docs = retriever.get_relevant_documents(query)
        return format_docs(docs)
    
    chain = (
        {
            "context": RunnableLambda(retrieve_and_format),
            "input": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain