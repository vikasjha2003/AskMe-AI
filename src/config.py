from pathlib import Path

CHUNKING_METHODS = [
    "fixed_256",
    "fixed_512",
    "fixed_1024",
    "recursive",
    "token_256",
    "token_512",
    "token_1024",
]

CHROMA_DIR = Path("src/data/chroma_db")

RESULTS_DIR = Path("src/data/results")

QA_FILE = Path("src/data/questions_and_answers/apple_2025_annual_report.json")

DOCUMENT_PATH = Path("src/data/documents/Apple 2025 annual report.pdf")