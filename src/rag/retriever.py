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
        # Get results from both
        bm25_docs = self.bm25_retriever.get_relevant_documents(query)
        vector_docs = self.vector_retriever.get_relevant_documents(query)
        
        # Combine and deduplicate
        all_docs = bm25_docs + vector_docs
        seen = set()
        unique_docs = []
        
        for doc in all_docs:
            # Use content hash to detect duplicates
            content_hash = hash(doc.page_content)
            if content_hash not in seen:
                seen.add(content_hash)
                unique_docs.append(doc)
        
        return unique_docs[:5]  # Return top 5
    
    def invoke(self, query):
        """LangChain-compatible invoke method"""
        return self.get_relevant_documents(query)

def create_hybrid_retriever(vectorstore, documents, k=5):
    """
    Create a hybrid retriever combining BM25 and vector search
    
    Args:
        vectorstore: Chroma vector store
        documents: List of LangChain documents (for BM25)
        k: Number of results to return
    """
    
    # BM25 Retriever (keyword search)
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k
    
    # Vector Retriever (semantic search)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    
    # Create hybrid retriever
    hybrid_retriever = SimpleHybridRetriever(
        bm25_retriever=bm25_retriever,
        vector_retriever=vector_retriever,
        bm25_weight=0.4  # 40% BM25, 60% vector
    )
    
    return hybrid_retriever