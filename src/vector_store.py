import re

from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata

from embedding import get_embeddings
from config import CHROMA_DIR


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


def sanitize_id(value):
    value = re.sub(r"[^a-zA-Z0-9._-]", "_", value)
    value = value.strip("._-")

    return value


def create_vector_store(chunks, method_name, document_id):
    embeddings = get_embeddings()

    chunks = prepare_metadata(chunks)

    document_id = sanitize_id(document_id)
    method_name = sanitize_id(method_name)

    collection_name = f"askme_{document_id}_{method_name}"

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    vector_store.add_documents(chunks)

    return vector_store