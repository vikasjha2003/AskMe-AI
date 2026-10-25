from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)


def fixed_chunking(documents, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[""],
    )

    return splitter.split_documents(documents)


def recursive_chunking(documents, chunk_size=800, chunk_overlap=160):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    return splitter.split_documents(documents)


def token_chunking(documents, chunk_size, chunk_overlap):
    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return splitter.split_documents(documents)


def get_chunking_methods():
    return {
        "fixed_200": lambda docs: fixed_chunking(docs,200,40),
        "fixed_400": lambda docs: fixed_chunking(docs,400,80),
        "fixed_800": lambda docs: fixed_chunking(docs,800,160),
        "recursive": lambda docs: recursive_chunking(docs),
        "token_100": lambda docs: token_chunking(docs,100,20),
        "token_200": lambda docs: token_chunking(docs,200,40),
        "token_400": lambda docs: token_chunking(docs,400,80),
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