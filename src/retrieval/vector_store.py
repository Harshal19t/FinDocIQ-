import os
import json
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


class FinancialVectorStore:
    """Manages embedding generation and Qdrant vector collection indexing."""

    def __init__(
        self,
        collection_name: str = "financial_knowledge_base",
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",
        db_path: str = "data/qdrant_db"
    ):
        self.collection_name = collection_name
        print(f"Loading embedding model: {embedding_model_name}...")
        self.embedder = SentenceTransformer(embedding_model_name)
        
        # Initialize local Qdrant persistence database
        os.makedirs(db_path, exist_ok=True)
        self.client = QdrantClient(path=db_path)

    def initialize_collection(self, vector_size: int = 384):
        """Creates or recreates the Qdrant vector collection."""
        collections = [c.name for c in self.client.get_collections().collections]
        
        if self.collection_name in collections:
            print(f"Collection '{self.collection_name}' already exists. Re-using collection.")
        else:
            print(f"Creating collection '{self.collection_name}'...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )

    def index_documents(self, documents: List[Dict[str, Any]]):
        """Embeds text chunks and uploads them as vector points to Qdrant."""
        if not documents:
            print("No documents provided for indexing.")
            return

        texts = [doc["content"] for doc in documents]
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedder.encode(texts, show_progress_bar=True)

        points = []
        for idx, (doc, vector) in enumerate(zip(documents, embeddings)):
            points.append(
                PointStruct(
                    id=idx,
                    vector=vector.tolist(),
                    payload={
                        "chunk_id": doc.get("chunk_id", f"c_{idx}"),
                        "page": doc.get("page", 1),
                        "content": doc.get("content", ""),
                        "metadata": doc.get("metadata", {})
                    }
                )
            )

        print(f"Uploading {len(points)} vectors to Qdrant...")
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print("Indexing completed successfully!")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Queries Qdrant vector store and returns top matching chunks."""
        query_vector = self.embedder.encode(query).tolist()
        
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k
        )

        results = []
        for point in search_result.points:
            results.append({
                "score": point.score,
                "content": point.payload.get("content"),
                "page": point.payload.get("page"),
                "chunk_id": point.payload.get("chunk_id")
            })
        return results
    
    def close(self):
        """Explicitly closes the Qdrant client connection."""
        if hasattr(self, "client") and self.client:
            self.client.close()

if __name__ == "__main__":
    # Sample chunks from Day 1 ingestion format
    sample_chunks = [
        {
            "chunk_id": "p1_c0",
            "page": 1,
            "content": "Total consolidated revenue increased by $4.2 billion or 12% in FY2024. Net income reached $1.8 billion.",
            "metadata": {"has_numbers": True}
        },
        {
            "chunk_id": "p1_c1",
            "page": 1,
            "content": "Operating expenses grew by 5% due to research and development investments in AI infrastructure.",
            "metadata": {"has_numbers": True}
        },
        {
            "chunk_id": "p2_c0",
            "page": 2,
            "content": "The company maintains a strong liquidity position with $3.5 billion in cash and cash equivalents.",
            "metadata": {"has_numbers": True}
        }
    ]

    vector_store = FinancialVectorStore()
    vector_store.initialize_collection(vector_size=384)
    vector_store.index_documents(sample_chunks)

    print("\n--- Testing Vector Retrieval ---")
    query = "What was the total revenue growth in FY2024?"
    matches = vector_store.search(query, top_k=2)
    
    for i, match in enumerate(matches, 1):
        print(f"\nMatch {i} (Similarity Score: {match['score']:.4f}):")
        print(f"Page {match['page']} [{match['chunk_id']}]: {match['content']}")
    
    vector_store.close()