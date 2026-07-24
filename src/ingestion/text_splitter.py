import json
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

class FinancialTextSplitter:
    """Splits long SEC filings into overlapping chunks while preserving financial metrics."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", ";", " ", ""]
        )

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Splits raw page documents into vector-ready chunks with metadata."""
        chunked_docs = []
        
        for doc in documents:
            text = doc.get("text", "")
            page_num = doc.get("page_number", 0)
            
            chunks = self.splitter.split_text(text)
            
            for chunk_idx, chunk in enumerate(chunks):
                chunked_docs.append({
                    "chunk_id": f"p{page_num}_c{chunk_idx}",
                    "page": page_num,
                    "content": chunk,
                    "metadata": {
                        "length": len(chunk),
                        "has_numbers": any(char.isdigit() for char in chunk)
                    }
                })
                
        print(f"Generated {len(chunked_docs)} chunked units from input documents.")
        return chunked_docs


if __name__ == "__main__":
    from src.ingestion.pdf_parser import FinancialDataIngestor
    
    # Run Day 1 Ingestion Pipeline Test
    ingestor = FinancialDataIngestor()
    dataset_file = ingestor.load_hf_financial_dataset(num_samples=200)
    
    # Test Text Splitting
    sample_text = [
        {
            "page_number": 1,
            "text": "Total consolidated revenue increased by $4.2 billion or 12% in FY2024. Net income reached $1.8 billion, driven by strong growth in the Cloud services segment. Operating expenses were $2.1 billion."
        }
    ]
    
    splitter = FinancialTextSplitter(chunk_size=120, chunk_overlap=20)
    chunks = splitter.chunk_documents(sample_text)
    
    print("\nSample Chunk Output:")
    print(json.dumps(chunks[0], indent=2))