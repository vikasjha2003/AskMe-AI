from retriever import get_retriever
from local_llm import generate


def answer_question(
    question,
    method_name="recursive",
    document_id="default",
    k=5,
):
    retriever = get_retriever(
        method_name=method_name,
        document_id=document_id,
        k=k,
    )

    documents = retriever.invoke(question)

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
{question}

Answer:"""

    return generate(prompt)