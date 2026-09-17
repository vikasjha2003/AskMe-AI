from retriever import get_retriever
from local_llm import generate


def answer_question(question, method_name="recursive", k=5):
    retriever = get_retriever(
        method_name=method_name,
        k=k
    )

    documents = retriever.invoke(question)

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    prompt = f""" /no_think
Answer the question using ONLY the provided context.

If the answer cannot be found in the context, say:
"I don't know based on the provided document."

Context:
{context}

Question:
{question}

Answer:"""

    return generate(prompt)


if __name__ == "__main__":
    question = "What was Apple's total net sales for fiscal year 2025?"

    answer = answer_question(
        question,
        method_name="recursive",
        k=5
    )

    print("\nAnswer:")
    print(answer)