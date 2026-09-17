import json
from pathlib import Path

from retriever import get_vector_store


QA_FILE = Path("src/data/questions_and_answers/apple_2025_annual_report.json")


def load_questions():
    with open(QA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data["questions"]


def evaluate_retrieval(k=5):
    vector_store = get_vector_store()
    questions = load_questions()

    total = len(questions)
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
            page = doc.metadata.get("page_no")

            if page == target_page:
                rank = i
                break

        if rank is not None:
            hits += 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)

    recall_at_k = hits / total
    mrr = sum(reciprocal_ranks) / total

    print(f"Questions: {total}")
    print(f"Recall@{k}: {recall_at_k:.4f}")
    print(f"MRR@{k}: {mrr:.4f}")


if __name__ == "__main__":
    evaluate_retrieval(k=5)