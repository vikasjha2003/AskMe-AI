from pathlib import Path

from chunker import create_chunks, get_chunking_methods
from vector_store import create_vector_store
from rag import answer_question
from doc_loader import load_documents
from benchmark import run_benchmark
from result_generator import generate_results
from config import RESULTS_DIR


def qa_mode():
    file_path = input("\nEnter document path: ").strip()
    file = Path(file_path)

    if not file.exists():
        print("File not found.")
        return

    document_id = file.stem.lower().replace(" ", "_")

    methods = get_chunking_methods()
    method_names = list(methods.keys())

    print("\nAvailable chunking methods:")

    for i, method in enumerate(method_names):
        print(f"{i}. {method}")

    try:
        choice = int(input("\nChoose chunking method: ").strip())
    except ValueError:
        print("Please enter a valid number.")
        return

    if choice < 0 or choice >= len(method_names):
        print("Invalid choice.")
        return

    method = method_names[choice]

    print(f"\nSelected: {method}")

    print("\nLoading document...")
    documents = load_documents(file_path)

    print("Creating chunks...")
    chunks = create_chunks(
        documents,
        selected_methods=[method],
    )

    print("Creating vector store...")
    create_vector_store(
        chunks[method],
        method_name=method,
        document_id=document_id,
    )

    print("\nAsk questions about your document.")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Question: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        print("\nGenerating answer...")

        answer = answer_question(
            question,
            method_name=method,
            document_id=document_id,
            k=5,
        )

        print(f"\nAnswer: {answer}\n")


def benchmark_mode():
    file_path = input("\nEnter document path: ").strip()
    file = Path(file_path)

    if not file.exists():
        print("File not found.")
        return

    qa_file = input("Enter QA JSON path: ").strip()
    qa_path = Path(qa_file)

    if not qa_path.exists():
        print("QA JSON file not found.")
        return

    print("\n========== RUNNING BENCHMARK ==========")

    results, benchmark_file = run_benchmark(
        document_path=file_path,
        qa_file=qa_file,
    )

    print("\n========== BENCHMARK RESULTS ==========")

    for method, result in results.items():
        print(
            f"{method:12} | "
            f"Chunks: {result['chunks']:4} | "
            f"Recall@5: {result['recall@5']:.4f} | "
            f"MRR@5: {result['mrr@5']:.4f}"
        )

    print("\nGenerating answer results...")

    for method in results:
        answers = generate_results(
            document_path=file_path,
            qa_file=qa_file,
            method_name=method,
            k=5,
        )

        document_id = (
            file.stem.lower().replace(" ", "_")
        )

        output_file = (
            RESULTS_DIR
            / f"{document_id}_{method}_results.json"
        )

        with open(output_file, "w", encoding="utf-8") as f:
            import json
            json.dump(
                answers,
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"Saved: {output_file}")

    print(f"\nBenchmark metrics saved to:")
    print(benchmark_file)


def main():
    print("========== AskMe AI ==========")

    print("\n1. Ask Questions")
    print("2. Benchmark Document")

    choice = input("\nChoose an option: ").strip()

    if choice == "1":
        qa_mode()

    elif choice == "2":
        benchmark_mode()

    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()