# src/rag/retriever.py

from langchain_community.retrievers import BM25Retriever

class SimpleHybridRetriever:
    """Manual hybrid retriever combining BM25 and vector search"""
    
    def __init__(self, bm25_retriever, vector_retriever, bm25_weight=0.4):
        self.bm25_retriever = bm25_retriever
        self.vector_retriever = vector_retriever
        self.bm25_weight = bm25_weight
        self.vector_weight = 1 - bm25_weight
    
    def get_relevant_documents(self, query):
        """Get documents from both retrievers and merge"""
        # Use invoke() instead of _get_relevant_documents()
        bm25_docs = self.bm25_retriever.invoke(query)
        vector_docs = self.vector_retriever.invoke(query)
        
        # Combine and deduplicate
        all_docs = bm25_docs + vector_docs
        seen = set()
        unique_docs = []
        
        for doc in all_docs:
            content_hash = hash(doc.page_content)
            if content_hash not in seen:
                seen.add(content_hash)
                unique_docs.append(doc)
        
        return unique_docs[:5]
    
    def invoke(self, query):
        """LangChain-compatible invoke method"""
        return self.get_relevant_documents(query)

def create_hybrid_retriever(vectorstore, documents, k=5):
    """Create a hybrid retriever combining BM25 and vector search"""
    
    # BM25 Retriever (keyword search)
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k
    
    # Vector Retriever (semantic search)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    
    # Create hybrid retriever
    hybrid_retriever = SimpleHybridRetriever(
        bm25_retriever=bm25_retriever,
        vector_retriever=vector_retriever,
        bm25_weight=0.4
    )
    
    return hybrid_retriever