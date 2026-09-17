# AskMe AI

AskMe AI is a document-grounded Retrieval-Augmented Generation (RAG) project for PDF-based question answering and retrieval benchmarking. It is built around a practical workflow: extract text from a PDF, split it into chunks, embed the chunks, store them in a vector database, retrieve relevant context for a question, and answer using a local LLM grounded in that context.

The repository is especially useful for experimenting with chunking strategies and evaluating how different chunk sizes and splitting methods affect retrieval quality on a real document.

## What this project does

This project helps you:

- load and process a PDF document
- generate multiple chunking variants of the same document
- embed those chunks with SentenceTransformers
- store them in Chroma for vector retrieval
- ask questions grounded in document context
- benchmark which chunking strategy performs best on a known QA set
- generate evaluation metrics like Recall@5 and MRR@5

It is designed both as a working RAG example and as a small research sandbox for document Q&A evaluation.

## Core idea

The key research question behind the project is:

- how do different chunking strategies influence retrieval and answer quality?

In many RAG pipelines, document chunking is a major determinant of both retrieval precision and downstream answer quality. AskMe AI lets you compare strategies like fixed-size chunking, recursive chunking, and token-aware chunking against the same corpus and evaluation questions.

## Architecture overview

The project follows a standard RAG flow:

1. Load a PDF document with Docling
2. Convert the document pages into LangChain `Document` objects
3. Split the document using a selected chunking method
4. Create embeddings from chunk text
5. Store the chunks in ChromaDB
6. Retrieve the most relevant chunks for a question
7. Build a grounded prompt from the retrieved context
8. Generate an answer with a local GGUF model
9. Benchmark the retrieval pipeline using a JSON QA set

## Project structure

```text
AskMe-AI/
├── README.md
├── requirements.txt
├── .python-version
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── benchmark.py
│   ├── chunker.py
│   ├── doc_loader.py
│   ├── embedding.py
│   ├── local_llm.py
│   ├── rag.py
│   ├── retriever.py
│   ├── result_generator.py
│   ├── vector_store.py
│   └── data/
│       ├── documents/
│       │   └── Apple 2025 annual report.pdf
│       ├── questions_and_answers/
│       │   └── apple_2025_annual_report.json
│       ├── chroma_db/
│       └── results/
```

## Module breakdown

### `src/main.py`

Entry point for the interactive app. It presents a simple menu:

- Ask Questions
- Benchmark Document

This CLI lets a user select a document, choose a chunking strategy, and interactively query the document.

### `src/doc_loader.py`

Loads a PDF using `langchain_docling.DoclingLoader` and caches the extracted content as a JSON file under `src/data/documents`. This avoids reprocessing the same file repeatedly.

### `src/chunker.py`

Implements the document chunking strategies used in the project:

- fixed_256
- fixed_512
- fixed_1024
- recursive
- token_256
- token_512
- token_1024

The project intentionally compares several fixed-size and token-aware chunk sizes so you can evaluate retrieval tradeoffs.

### `src/embedding.py`

Uses `langchain_huggingface.HuggingFaceEmbeddings` with the model:

- `sentence-transformers/all-MiniLM-L6-v2`

This provides embeddings for the chunks inserted into ChromaDB.

### `src/vector_store.py`

Creates and stores a Chroma collection for each document and chunking method. It also extracts page metadata so retrieved items can be linked back to source pages.

### `src/retriever.py`

Defines the retriever used during Q&A:

- Chroma vector store
- `search_type="mmr"`
- `k=5`
- `fetch_k=15`
- `lambda_mult=0.75`

MMR helps balance relevance and diversity when several related passages are retrieved.

### `src/rag.py`

Builds the final prompt for the LLM:

- retrieves relevant chunks
- concatenates them as context
- prompts the model to answer only from the supplied evidence
- instructs the model to respond with a fallback when the answer is not present

### `src/local_llm.py`

Loads a local GGUF model using `llama-cpp-python` and generates answers with a chat-completion call.

The current configuration expects:

```text
models/Qwen3-8B-Q4_K_M.gguf
```

### `src/benchmark.py`

Runs evaluation over the built-in QA set. For each chunking method it:

- creates a vector store
- runs similarity search on each question
- compares returned chunk page numbers with the expected source page
- computes Recall@5 and MRR@5
- saves benchmark output to `src/data/results`

### `src/result_generator.py`

Generates per-question answer outputs for each chunking method using the retriever and local generation prompt. This is useful for inspecting how the same question is answered across methods.

## Data included in the repo

The repository includes a practical document QA benchmark based on Apple's 2025 annual report.

### Document

```text
src/data/documents/Apple 2025 annual report.pdf
```

### QA dataset

```text
src/data/questions_and_answers/apple_2025_annual_report.json
```

The dataset contains questions across several categories, including:

- factual_lookup
- table_query
- multi_hop
- summarization
- needle_in_haystack

Each question includes:

- the question text
- expected answer
- source page number
- difficulty label
- category

## Tech stack

This repo uses the following stack:

- Python
- LangChain
- LangChain Text Splitters
- LangChain Chroma
- Hugging Face embeddings
- Sentence Transformers
- ChromaDB
- Docling
- PyMuPDF / PDF extraction tooling
- llama.cpp / llama-cpp-python
- local GGUF model inference
- JSON-based evaluation dataset

## Requirements

The project dependencies are defined in `requirements.txt`.

Key packages include:

- `langchain-docling`
- `docling`
- `langchain`
- `langchain-text-splitters`
- `langchain-huggingface`
- `sentence-transformers`
- `langchain-chroma`
- `chromadb`
- `ragas`
- `datasets`
- `llama-cpp-python`

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/vikasjha2003/AskMe-AI.git
cd AskMe-AI
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the local model

This project expects a GGUF model file at:

```text
models/Qwen3-8B-Q4_K_M.gguf
```

Create a `models/` directory and place the model there before running the LLM workflow.

## Running the app

```bash
python src/main.py
```

You will see a menu:

```text
1. Ask Questions
2. Benchmark Document
```

### Ask Questions mode

This mode lets you:

- enter a document path
- choose a chunking strategy
- create a vector store for that method
- ask questions interactively

Example flow:

```bash
python src/main.py
```

Then select:

```text
1
```

### Benchmark Document mode

This mode lets you:

- provide a PDF document
- provide a QA JSON file
- benchmark all chunking methods
- save summary metrics and answer outputs

Example flow:

```bash
python src/main.py
```

Then select:

```text
2
```

## Evaluation metrics

The benchmark measures retrieval quality using:

- Recall@5
- MRR@5

These metrics compare retrieved document pages against the document page referenced by each QA item. This allows the project to rank different chunking strategies quantitatively.

## Output artifacts

Benchmark and answer outputs are saved in:

```text
src/data/results/
```

Typical generated outputs include:

- benchmark summaries
- per-method metrics
- generated answers for each question
- retrieved page metadata

## Why this repo is useful

This repository is a strong example of a minimal but complete RAG training/evaluation project because it demonstrates:

- PDF ingestion
- chunking experimentation
- embedding generation
- vector search
- local LLM inference
- retrieval benchmarking
- document-grounded answer generation

It is particularly valuable if you want to study how chunking choices affect retrieval performance on a real-world financial document.

## Current strengths

- straightforward RAG architecture
- multiple chunking strategies built in
- QA benchmark dataset included
- real PDF-based workflow
- modular code organization
- practical local-model inference path

## Notable limitations and opportunities

The repo is intentionally lightweight and research-oriented. Some natural next improvements include:

- environment-variable-based configuration
- better CLI argument support with `argparse`
- more complete logging and error handling
- a web UI using Streamlit or Gradio
- support for multi-document ingestion
- answer-quality metrics beyond retrieval metrics
- automated testing
- Docker support
- persistent model/cache configuration

## License

This repository does not currently include a `LICENSE` file. If you plan to distribute or publish the project, consider adding an explicit OSS license such as MIT or Apache-2.0.

## Contributing

Contributions are welcome in areas such as:

- better chunking methods
- improved benchmark reporting
- support for more PDF or document ingestion types
- answer evaluation beyond page retrieval
- cleaner configuration and deployment patterns

## Summary

AskMe AI is a compact but practical RAG project for document-based question answering and benchmarking. It combines PDF extraction, chunking, embeddings, vector search, retrieval, and local LLM generation, making it a useful project for understanding how retrieval components influence end-to-end question answering quality.

If your goal is to experiment with chunk sizes, compare retrieval methods, or build a document-grounded local QA pipeline, this repository provides a strong foundation.

## Quick start

```bash
git clone https://github.com/vikasjha2003/AskMe-AI.git
cd AskMe-AI
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/main.py
```

If you want to benchmark the included Apple annual report dataset, run the benchmark flow from the app menu and provide the included PDF and QA JSON file.
