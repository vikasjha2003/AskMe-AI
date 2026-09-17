from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)


def fixed_chunking(documents, chunk_size, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[""],
    )

    return splitter.split_documents(documents)


def recursive_chunking(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    return splitter.split_documents(documents)


def token_chunking(documents, chunk_size, chunk_overlap=50):
    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return splitter.split_documents(documents)


def get_chunking_methods():
    return {
        "fixed_256": lambda docs: fixed_chunking(docs, 256),
        "fixed_512": lambda docs: fixed_chunking(docs, 512),
        "fixed_1024": lambda docs: fixed_chunking(docs, 1024),
        "recursive": lambda docs: recursive_chunking(docs),
        "token_256": lambda docs: token_chunking(docs, 256),
        "token_512": lambda docs: token_chunking(docs, 512),
        "token_1024": lambda docs: token_chunking(docs, 1024),
    }


def create_chunks(documents, selected_methods=None):
    methods = get_chunking_methods()

    if selected_methods is None:
        selected_methods = list(methods.keys())

    invalid_methods = set(selected_methods) - set(methods.keys())

    if invalid_methods:
        raise ValueError(
            f"Invalid methods: {invalid_methods}\n"
            f"Available methods: {list(methods.keys())}"
        )

    results = {}

    for method_name in selected_methods:
        results[method_name] = methods[method_name](documents)

    return results