import json
from pathlib import Path

from retriever import get_retriever
from local_llm import generate


def load_questions(qa_file):
    with open(qa_file, "r", encoding="utf-8") as f:
        return json.load(f)["questions"]


def generate_results(
    document_path,
    qa_file,
    method_name="recursive",
    k=5,
):
    document_path = Path(document_path)
    qa_file = Path(qa_file)

    document_id = (
        document_path.stem.lower().replace(" ", "_")
    )

    questions = load_questions(qa_file)

    retriever = get_retriever(
        method_name=method_name,
        document_id=document_id,
        k=k,
    )

    results = []

    for question in questions:
        documents = retriever.invoke(
            question["question"]
        )

        context = "\n\n".join(
            document.page_content
            for document in documents
        )

        prompt = f"""Answer the question using ONLY the provided context.

If the answer cannot be found in the context, say:
"I don't know based on the provided document."

Context:
{context}

Question:
{question["question"]}

Answer:"""

        answer = generate(prompt).strip()

        results.append({
            "id": question["id"],
            "question": question["question"],
            "ground_truth": question["ground_truth"],
            "answer": answer,
            "category": question["category"],
            "difficulty": question["difficulty"],
            "source_page": question["source_page"],
            "retrieved_pages": [
                document.metadata.get("page_no")
                for document in documents
            ],
        })

        print(f"Completed: {question['id']}")

    return results