import sys
import os
import pandas as pd

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.pipeline.rag_chain import FinancialRAGPipeline


# Benchmark dataset containing expected ground-truth responses
TEST_SUITE = [
    {
        "id": "TC01_NUMERIC",
        "question": "What was the total revenue growth in FY2024?",
        "expected": "$4.2 billion or 12%"
    },
    {
        "id": "TC02_QUALITATIVE",
        "question": "Why did operating expenses increase?",
        "expected": "Research and development investments in AI infrastructure"
    },
    {
        "id": "TC03_MULTIPAGE",
        "question": "How much liquidity does the company hold in cash?",
        "expected": "$3.5 billion"
    },
    {
        "id": "TC04_GUARDRAIL",
        "question": "What were the total stock buybacks in 2025?",
        "expected": "Insufficient information provided."
    }
]


def run_benchmark():
    print("=" * 60)
    print("STARTING RAG PIPELINE BENCHMARK")
    print("=" * 60)

    eval_records = []

    with FinancialRAGPipeline() as rag:
        for test_case in TEST_SUITE:
            tc_id = test_case["id"]
            question = test_case["question"]
            expected = test_case["expected"]

            print(f"\nRunning [{tc_id}]: '{question}'...")
            
            result = rag.query(question, top_k=2)
            generated_answer = result["answer"]
            sources = result["sources"]

            # Compute simple keyword/substring overlap accuracy
            passed = any(kw.lower() in generated_answer.lower() for kw in expected.split())

            top_source = f"Page {sources[0]['page']} ({sources[0]['chunk_id']})" if sources else "None"
            top_score = sources[0]["score"] if sources else 0.0

            eval_records.append({
                "Test ID": tc_id,
                "Question": question,
                "Generated Answer": generated_answer,
                "Expected Metric": expected,
                "Top Chunk": top_source,
                "Similarity Score": top_score,
                "Match Status": "PASSED" if passed else "FAILED"
            })

    # Display results summary
    df = pd.DataFrame(eval_records)
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(df[["Test ID", "Match Status", "Similarity Score", "Generated Answer"]].to_string(index=False))

    # Export to CSV report
    output_dir = "data/eval"
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "rag_benchmark_report.csv")
    df.to_csv(report_path, index=False)
    print(f"\nSaved evaluation report to: {report_path}")


if __name__ == "__main__":
    run_benchmark()