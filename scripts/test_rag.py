# scripts/test_rag.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from src.rag.chain import build_rag_chain

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
    "How do I launch an EC2 instance?",
    "What's the deployment process for production?",
    "What's good for lunch?",  # ← This should return "No matching information found"
]

for q in questions:
    print(f"\n{'='*60}")
    print(f"Question: {q}")
    print(f"{'='*60}")
    result = chain.invoke({"input": q})
    print(f"Answer: {result['answer']}")
    print(f"\nSources: {[doc.metadata.get('source', 'unknown') for doc in result['context']]}")
