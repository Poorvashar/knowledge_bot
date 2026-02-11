# scripts/test_hybrid.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from src.rag.retriever import create_hybrid_retriever
from src.rag.chain import build_rag_chain

print("Loading vector database and documents...\n")

# Load Vector DB
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# ★ Load chunks (needed for BM25)
# Extract documents from Chroma
all_docs = vectorstore.get(include=["documents", "metadatas"])
chunks = [
    Document(page_content=doc, metadata=meta)
    for doc, meta in zip(all_docs["documents"], all_docs["metadatas"])
]

print(f"Loaded {len(chunks)} chunks from vector database")

# Build Hybrid Retriever
hybrid_retriever = create_hybrid_retriever(vectorstore, chunks, k=5)

# Build RAG chain
chain = build_rag_chain(hybrid_retriever)

# Test: keyword search case
print("\n" + "="*60)
print("Test 1: Search by command name (BM25 shines here)")
print("="*60)
answer = chain.invoke("How to use aws ec2 describe-instances")
print(f"Answer: {answer[:200]}...")

# Test: semantic search case
print("\n" + "="*60)
print("Test 2: Search by natural language (vector search shines here)")
print("="*60)
answer = chain.invoke("How to check production server status")
print(f"Answer: {answer[:200]}...")

print("\n Hybrid search test complete!")