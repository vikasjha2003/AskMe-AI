from pathlib import Path

from langchain_chroma import Chroma

from embedding import get_embeddings


CHROMA_DIR = Path("src/data/chroma_db")


def get_vector_store():
    embeddings = get_embeddings()

    return Chroma(
        collection_name="askme_ai",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def get_retriever(k=5):
    vector_store = get_vector_store()

    return vector_store.as_retriever(
        search_kwargs={"k": k}
    )


if __name__ == "__main__":
    retriever = get_retriever(k=5)

    query = "What was Apple's total net sales for fiscal year 2025?"

    results = retriever.invoke(query)

    print(f"Retrieved {len(results)} chunks\n")

    for i, doc in enumerate(results, 1):
        print(f"--- Chunk {i} ---")
        print(doc.page_content[:1000])
        print()