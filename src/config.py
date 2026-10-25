from pathlib import Path

CHUNKING_METHODS = [
    "fixed_200",
    "fixed_400",
    "fixed_800",
    "recursive",
    "token_100",
    "token_200",
    "token_400",
]

CHROMA_DIR = Path("src/data/chroma_db")

RESULTS_DIR = Path("src/data/results")

QA_FILE = Path("src/data/questions_and_answers/apple_2025_annual_report.json")

DOCUMENT_PATH = Path("src/data/documents/Apple 2025 annual report.pdf")