# src/rag/chain.py

from langchain_anthropic import ChatAnthropic
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

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

document_chain = create_stuff_documents_chain(llm, prompt)

def build_rag_chain(vectorstore):
    """Create a retrieval chain from a vectorstore"""
    retrieval_chain = create_retrieval_chain(
        vectorstore.as_retriever(search_kwargs={"k": 5}),
        document_chain
    )
    return retrieval_chain