from langchain_huggingface import HuggingFaceEmbeddings


MODEL_NAME = "BAAI/bge-small-en-v1.5"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


if __name__ == "__main__":
    embeddings = get_embeddings()

    text = "Apple's total net sales for fiscal year 2025 were $416,161 million."

    vector = embeddings.embed_query(text)

    print(f"Embedding dimensions: {len(vector)}")
    print(vector[:10])