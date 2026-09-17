from pathlib import Path
import json

from langchain_core.documents import Document
from langchain_docling import DoclingLoader


def load_documents(file_path):
    file_path = Path(file_path)

    cache_file = (
        Path("src/data/documents")
        / f"{file_path.stem}_processed.json"
    )

    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        return [
            Document(
                page_content=item["page_content"],
                metadata=item["metadata"],
            )
            for item in data
        ]

    loader = DoclingLoader(file_path=str(file_path))
    documents = loader.load()

    data = [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in documents
    ]

    cache_file.parent.mkdir(parents=True, exist_ok=True)

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return documents