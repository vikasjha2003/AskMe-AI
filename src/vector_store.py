from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata

from embedding import get_embeddings
from chunker import create_chunks
from doc_loader import load_documents


CHROMA_DIR = Path("src/data/chroma_db")


def create_vector_store(chunks):
    embeddings = get_embeddings()

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

    chunks = filter_complex_metadata(chunks)

    vector_store = Chroma(
        collection_name="askme_ai",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    vector_store.add_documents(chunks)

    return vector_store


if __name__ == "__main__":
    documents = load_documents()

    chunked_documents = create_chunks(
        documents,
        selected_methods=["recursive"]
    )

    chunks = chunked_documents["recursive"]

    vector_store = create_vector_store(chunks)

    print(f"Added {len(chunks)} chunks to ChromaDB")