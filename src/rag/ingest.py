# src/rag/ingest.py

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

def create_chunks(documents):
    """Split documents into chunks"""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,       # Max 800 characters per chunk
        chunk_overlap=200,    # 200 char overlap between chunks (prevents losing context at boundaries)
        separators=["\n\n", "\n", ".", " "]  # Try splitting in this order
    )

    chunks = splitter.split_documents(documents)
    print(f"✂️  {len(documents)} documents → {len(chunks)} chunks")
    return chunks


def build_vectorstore(chunks, persist_directory="./chroma_db"):
    """Vectorize chunks and save to Vector DB"""

    # Use OpenAI's embedding model to vectorize text
    # (text-embedding-3-small is cheap and performs well enough)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Save vectors to Chroma (Vector DB)
    # Setting persist_directory means data survives between restarts
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    print(f"💾 Saved {len(chunks)} chunks to Vector DB")
    return vectorstore