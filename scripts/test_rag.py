# scripts/test_rag.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from src.rag.chain import build_rag_chain

print("Loading vector database...\n")

# Load the saved Vector DB
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Build the RAG chain
chain = build_rag_chain(vectorstore)

# Test questions
questions = [
    "What is this document about?",
    "What information is in the CSV file?",
    "What does the paper discuss?",
]

for q in questions:
    print(f"\n{'='*60}")
    print(f"Question: {q}")
    print(f"{'='*60}")
    
    # The chain returns just the answer string
    answer = chain.invoke(q)
    print(f"Answer: {answer}")