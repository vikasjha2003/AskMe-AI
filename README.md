# AskMe AI

AskMe AI is a Retrieval-Augmented Generation (RAG) project designed to answer questions from uploaded documents and benchmark how different chunking strategies affect retrieval quality and answer generation.

The project is built around a real-world PDF document workflow and is optimized for experimentation: it supports multiple chunking methods, vector search, local LLM inference, and evaluation using a curated question-answer dataset.

## Overview

AskMe AI enables users to:
- upload or reference a PDF document
- split the document using multiple chunking strategies
- generate embeddings for each chunk
- retrieve relevant context with vector similarity search
- answer questions using a local language model grounded in retrieved context
- benchmark retrieval performance against known questions and pages

This makes the repository useful both as a document Q&A system and as a research tool for evaluating chunking effectiveness in RAG pipelines.

## Why this project exists

The core objective is to understand how document preprocessing and chunking affect downstream retrieval and answer quality.

In many RAG systems, chunking is one of the most important factors influencing:
- retrieval precision
- recall of relevant passages
- overall answer quality
- computational efficiency

This project explores several chunking methods and compares them using evaluation metrics on a known document.

## Features

- PDF document ingestion using Docling
- Multiple chunking strategies:
  - fixed-size character chunking
  - recursive chunking
  - token-based chunking
- Embeddings using Sentence Transformers
- Vector database storage with Chroma
- Semantic retrieval using MMR-based retriever
- Local LLM-driven answer generation using llama.cpp
- Benchmarking of chunking methods using Recall@5 and MRR@5
- JSON-based QA evaluation dataset
- CLI-based application flow for Q&A and benchmarking

## Architecture

The system follows a standard RAG pipeline:

1. Load document
2. Convert pages/content into LangChain documents
3. Chunk the document using a selected strategy
4. Generate embeddings
5. Store chunks in a vector store
6. Retrieve relevant chunks for a question
7. Pass retrieved context into an LLM
8. Generate an answer grounded only in the supplied context
9. Benchmark retrieval quality against known answers

## Tech Stack

- Python
- LangChain
- LangChain Text Splitters
- LangChain Chroma
- Hugging Face Embeddings
- Sentence Transformers
- ChromaDB
- Docling
- PyMuPDF / PDF processing tools
- llama.cpp
- Qwen-style local GGUF model
- RAGAS-inspired evaluation approach
- JSON-based QA dataset

## Repository Structure

```text
AskMe-AI/
├── README.md
├── requirements.txt
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

## Document and QA Data

The repository includes:
- a PDF source document: Apple 2025 annual report
- a question-answer dataset in JSON format
- benchmark output files written under `src/data/results`

This makes the repo suitable for systematic comparison of retrieval methods and chunking strategies.

## Chunking Methods

The project evaluates these chunking methods:

- `fixed_256`
- `fixed_512`
- `fixed_1024`
- `recursive`
- `token_256`
- `token_512`
- `token_1024`

These cover both fixed-size and token-aware chunking strategies, allowing comparison across different granularity levels.

## Retrieval and Generation

The retrieval layer uses Chroma with embeddings from `sentence-transformers/all-MiniLM-L6-v2`.

The retriever is configured with:
- `search_type="mmr"`
- `k` documents requested
- `fetch_k` for candidate selection
- `lambda_mult=0.75`

This hybrid strategy balances relevance and diversity, which is useful when multiple related passages may appear in a document.

The answer generation layer builds a prompt using retrieved context and instructs the model to answer only from the provided document evidence.

## Evaluation

The benchmark script evaluates retrieval effectiveness by measuring:

- Recall@5
- MRR@5

The evaluation compares retrieved chunks against the expected page of a question and computes:
- percentage of questions finding the correct page within the top 5 results
- mean reciprocal rank of the first correct result

This gives a practical signal for how well a given chunking configuration supports retrieval.

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/vikasjha2003/AskMe-AI.git
cd AskMe-AI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download a local model

The project expects a GGUF model file at:

```text
models/Qwen3-8B-Q4_K_M.gguf
```

Place the model in a `models/` directory before running the local LLM workflow.

### 5. Run the app

```bash
python src/main.py
```

You will see a menu with:
- Ask Questions
- Benchmark Document

## Usage Modes

### A. Ask Questions

This mode lets you:
- enter a document path
- choose a chunking method
- build a vector store
- ask questions interactively

Example workflow:
```bash
python src/main.py
```

Then choose:
```text
1. Ask Questions
```

### B. Benchmark Document

This mode lets you:
- provide a PDF document
- provide a QA JSON file
- run retrieval benchmarking across chunking methods
- generate result files for comparison

Example:
```bash
python src/main.py
```

Then choose:
```text
2. Benchmark Document
```

## Example Benchmark Flow

The system uses a QA file like:

```text
src/data/questions_and_answers/apple_2025_annual_report.json
```

This JSON dataset contains questions, answers, categories, source pages, and difficulty labels. The benchmark script loads these, retrieves documents for each question, and compares the retrieved page numbers to the expected answer source page.

## Output Artifacts

The project saves benchmark outputs in:

```text
src/data/results/
```

Typical output includes:
- per-document benchmark summaries
- per-method evaluation metrics
- generated answer results for each chunking strategy

## Current Project Strengths

- clear RAG pipeline structure
- multiple benchmarking modes
- strong focus on chunking strategy comparison
- practical document QA workflow
- modular design with separate concerns for loading, chunking, embedding, retrieval, and generation

## Potential Improvements

The repository is already solid for experimentation, but here are some natural next improvements:

- add a proper CLI interface with argparse
- add model configuration via environment variables
- add automated benchmark reporting
- add evaluation for answer correctness besides retrieval metrics
- support multi-document ingestion
- improve caching and persistence handling
- add Docker support
- add CI tests for unit-level validation
- add a Streamlit or Gradio interface for easier usage

## License

This project currently does not include an explicit license file. If you plan to publish or share it publicly, add a license such as MIT or Apache-2.0.

## Contributing

Contributions are welcome. Potential areas include:

- improving chunking strategies
- benchmarking additional retrieval methods
- adding support for more document types
- integrating more robust evaluation metrics
- improving model configuration and deployment

## Project Summary

AskMe AI is a practical and research-oriented document Q&A system built to evaluate how chunking and retrieval design affect a RAG pipeline. It is especially relevant for PDF-based document understanding, research experiments, and document intelligence workflows.

The project demonstrates a complete end-to-end RAG architecture using:
- document preprocessing
- vector search
- local language models
- evaluation metrics
- experimental benchmarking

## Quick Start

```bash
git clone https://github.com/vikasjha2003/AskMe-AI.git
cd AskMe-AI
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/main.py
```

If you want to take this project further, consider adding a web UI, automated benchmarking dashboard, or a more robust evaluation layer for answer quality.
