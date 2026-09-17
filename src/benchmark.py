import json
from pathlib import Path

from vector_store import create_vector_store
from chunker import create_chunks
from doc_loader import load_documents
from config import QA_FILE, RESULTS_DIR


def load_questions(qa_file):
    with open(qa_file, "r", encoding="utf-8") as f:
        return json.load(f)["questions"]


def evaluate_vector_store(vector_store, questions, k=5):
    hits = 0
    reciprocal_ranks = []

    for question in questions:
        results = vector_store.similarity_search(
            question["question"],
            k=k,
        )

        target_page = question["source_page"]

        rank = None

        for i, doc in enumerate(results, start=1):
            if doc.metadata.get("page_no") == target_page:
                rank = i
                break

        if rank is not None:
            hits += 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)

    recall = hits / len(questions)
    mrr = sum(reciprocal_ranks) / len(questions)

    return recall, mrr


def run_benchmark(document_path, qa_file=QA_FILE):
    document_path = Path(document_path)

    document_id = (
        document_path.stem.lower().replace(" ", "_")
    )

    print("\nLoading document...")
    documents = load_documents(document_path)

    print("Loading questions...")
    questions = load_questions(qa_file)

    print("Creating chunks...")
    chunks_by_method = create_chunks(documents)

    results = {}

    for method_name, chunks in chunks_by_method.items():

        print(f"\nEvaluating: {method_name}")
        print(f"Chunks: {len(chunks)}")

        vector_store = create_vector_store(
            chunks,
            method_name=method_name,
            document_id=document_id,
        )

        recall, mrr = evaluate_vector_store(
            vector_store,
            questions,
            k=5,
        )

        results[method_name] = {
            "chunks": len(chunks),
            "recall@5": recall,
            "mrr@5": mrr,
        }

        print(f"Recall@5: {recall:.4f}")
        print(f"MRR@5:    {mrr:.4f}")

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        RESULTS_DIR / f"{document_id}_benchmark_results.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print(f"\nBenchmark results saved to:")
    print(output_file)

    return results, output_file


if __name__ == "__main__":
    run_benchmark(
        document_path="src/data/documents/Apple 2025 annual report.pdf"
    )