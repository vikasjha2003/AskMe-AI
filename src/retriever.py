import re

from langchain_chroma import Chroma

from embedding import get_embeddings
from config import CHROMA_DIR


def sanitize_id(value):
    value = re.sub(r"[^a-zA-Z0-9._-]", "_", value)
    value = value.strip("._-")

    return value


def get_vector_store(method_name="recursive", document_id="default"):
    embeddings = get_embeddings()

    document_id = sanitize_id(document_id)
    method_name = sanitize_id(method_name)

    return Chroma(
        collection_name=f"askme_{document_id}_{method_name}",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def get_retriever(method_name="recursive", document_id="default", k=5):
    vector_store = get_vector_store(
        method_name=method_name,
        document_id=document_id,
    )

    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": 15,
            "lambda_mult": 0.75,
        },
    )