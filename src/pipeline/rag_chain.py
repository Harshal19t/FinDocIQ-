import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from src.retrieval.vector_store import FinancialVectorStore


class FinancialRAGPipeline:
    """Combines Qdrant vector retrieval with fine-tuned LoRA model generation."""

    def __init__(
        self,
        adapter_path: str = "models/financial_qlora_adapter",
        base_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
        collection_name: str = "financial_knowledge_base"
    ):
        print("Initializing Vector Store connection...")
        self.vector_store = FinancialVectorStore(collection_name=collection_name)

        print("Loading fine-tuned LLM and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            device_map="cpu"
        )
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()
        
    def format_rag_prompt(self, query: str, context_chunks: list) -> str:
        """Constructs a context-grounded instruction prompt."""
        formatted_context = "\n---\n".join(
            [f"[Page {chunk['page']}]: {chunk['content']}" for chunk in context_chunks]
        )
        
    def format_rag_prompt(self, query: str, context_chunks: list) -> str:
        """Constructs a context-grounded instruction prompt."""
        formatted_context = "\n---\n".join(
            [f"[Page {chunk['page']}]: {chunk['content']}" for chunk in context_chunks]
        )

        prompt = (
            f"<|im_start|>system\n"
            f"You are a financial analyst. Summarize the answer to the user question using the context below.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Context:\n{formatted_context}\n\n"
            f"Question: {query}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        return prompt
    
    def query(self, user_query: str, top_k: int = 2) -> dict:
        """Executes the full RAG workflow: Retrieve -> Augment -> Generate."""
        print(f"\nSearching vector store for top {top_k} relevant chunks...")
        retrieved_chunks = self.vector_store.search(user_query, top_k=top_k)

        if not retrieved_chunks:
            return {
                "query": user_query,
                "answer": "No relevant context found in vector store.",
                "sources": []
            }

        prompt = self.format_rag_prompt(user_query, retrieved_chunks)
        inputs = self.tokenizer(prompt, return_tensors="pt")

        print("Generating grounded response...")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()

        return {
            "query": user_query,
            "answer": response_text,
            "sources": [
                {"chunk_id": c["chunk_id"], "page": c["page"], "score": round(c["score"], 4)}
                for c in retrieved_chunks
            ]
        }

    def close(self):
        """Cleanly close database connections."""
        self.vector_store.close()


if __name__ == "__main__":
    rag_chain = FinancialRAGPipeline()

    test_query = "What was the total revenue growth in FY2024?"
    result = rag_chain.query(test_query)

    print("\n" + "=" * 60)
    print("QUERY:", result["query"])
    print("-" * 60)
    print("ANSWER:", result["answer"])
    print("-" * 60)
    print("SOURCES USED:", result["sources"])
    print("=" * 60)

    rag_chain.close()