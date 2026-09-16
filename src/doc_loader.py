from pathlib import Path
import json

from langchain_core.documents import Document
from langchain_docling import DoclingLoader


CACHE_FILE = Path("src/data/documents/processed_documents.json")

FILE_PATH = "src/data/documents/Apple 2025 annual report.pdf"


def load_documents():
    # Load cached documents if available
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        return [
            Document(
                page_content=item["page_content"],
                metadata=item["metadata"]
            )
            for item in data
        ]

    # Parse PDFs using Docling
    loader = DoclingLoader(
        file_path=[str(file) for file in FILE_PATH]
    )

    documents = loader.load()

    # Cache parsed documents
    data = [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata
        }
        for doc in documents
    ]

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return documents


if __name__ == "__main__":
    documents = load_documents()

    print(f"Loaded {len(documents)} documents")