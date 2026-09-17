import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata

from embedding import get_embeddings
from chunker import create_chunks
from doc_loader import load_documents


CHROMA_DIR = Path("src/data/chroma_db")
QA_FILE = Path("src/data/questions_and_answers/apple_2025_annual_report.json")


def load_questions():
    with open(QA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)["questions"]


def prepare_metadata(chunks):
    for chunk in chunks:
        doc_items = chunk.metadata.get("dl_meta", {}).get("doc_items", [])

        pages = []

        for item in doc_items:
            for prov in item.get("prov", []):
                page_no = prov.get("page_no")

                if page_no is not None:
                    pages.append(page_no)

        if pages:
            chunk.metadata["page_no"] = min(pages)

    return filter_complex_metadata(chunks)


def create_vector_store(chunks, method_name):
    embeddings = get_embeddings()

    chunks = prepare_metadata(chunks)

    collection_name = f"askme_ai_{method_name}"

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    vector_store.add_documents(chunks)

    return vector_store


def evaluate_vector_store(vector_store, questions, k=5):
    hits = 0
    reciprocal_ranks = []

    for question in questions:
        results = vector_store.similarity_search(
            question["question"],
            k=k
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


def main():
    documents = load_documents()
    questions = load_questions()

    chunks_by_method = create_chunks(documents)

    results = {}

    for method_name, chunks in chunks_by_method.items():
        print(f"\nEvaluating: {method_name}")
        print(f"Chunks: {len(chunks)}")

        vector_store = create_vector_store(
            chunks,
            method_name
        )

        recall, mrr = evaluate_vector_store(
            vector_store,
            questions,
            k=5
        )

        results[method_name] = {
            "chunks": len(chunks),
            "recall@5": recall,
            "mrr@5": mrr,
        }

        print(f"Recall@5: {recall:.4f}")
        print(f"MRR@5: {mrr:.4f}")

    print("\n========== BENCHMARK RESULTS ==========")

    for method, result in results.items():
        print(
            f"{method:12} | "
            f"Chunks: {result['chunks']:4} | "
            f"Recall@5: {result['recall@5']:.4f} | "
            f"MRR@5: {result['mrr@5']:.4f}"
        )


if __name__ == "__main__":
    main()