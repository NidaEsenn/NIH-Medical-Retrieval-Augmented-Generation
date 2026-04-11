# MedRAG: A Retrieval-Augmented Medical Question Answering System with Verifiable Citations

MedRAG is an end-to-end medical question answering project built around retrieval-augmented generation in a high-stakes domain. The system retrieves relevant medical evidence from NIH-aligned data, compares multiple retrieval strategies, and presents grounded outputs with traceable citations.

The current codebase includes:

- a Streamlit frontend for interactive medical QA
- a chunking pipeline that converts raw data into a stable retrieval schema
- a sparse retrieval baseline using BM25
- a dense retrieval pipeline using sentence embeddings and FAISS
- a hybrid retrieval pipeline using BM25 recall plus dense reranking

## Project Goal

The goal of this project is to build a local medical RAG system that can:

- answer medical questions using retrieved evidence rather than unsupported generation
- compare sparse, dense, and hybrid retrieval methods
- support answer verifiability through explicit source citations
- provide an end-to-end user-facing interface

This project is motivated by the fact that medical QA requires accuracy, transparency, and trust. Large language models can sound fluent while still hallucinating unsupported claims. MedRAG is designed to reduce that risk by grounding answers in retrieved source material.

## Dataset

The project is designed around MedQuAD, a collection of NIH medical question-answer pairs. Each example includes medical content together with source metadata such as document identity and source URL.

For retrieval, the system standardizes chunked data into this fixed schema:

- `chunk_id`
- `chunk_text`
- `document_id`
- `source_page_url`

This schema is shared across chunking, sparse retrieval, dense retrieval, and hybrid retrieval so that all retrieval methods operate against the same contract.

## System Overview

The intended MedRAG pipeline is:

```text
User Query
  -> Retriever (BM25 / Dense / Hybrid)
  -> Top-k Relevant Chunks
  -> Local LLM
  -> Grounded Answer + Citations
```

At the current stage, the frontend already supports real retrieval and evidence display. The next model-facing step is to pass retrieved chunks into a local LLM for answer generation.

## Retrieval Methods

### Sparse Retrieval

The sparse baseline uses BM25 over chunked medical text. It provides a keyword-based baseline that is easy to interpret and useful for comparison.

### Dense Retrieval

The dense retriever uses a pre-trained sentence embedding model to encode chunks and queries into vector representations. Chunk embeddings are precomputed and indexed with FAISS for efficient similarity search.

### Hybrid Retrieval

The hybrid retriever combines both approaches:

- BM25 first recalls a candidate set
- dense embeddings rerank the candidate chunks

This design supports stronger recall than pure dense retrieval on some queries while still benefiting from semantic reranking.

## Repository Structure

```text
app.py                         Streamlit frontend
medrag_chunking/              Chunking pipeline
medrag_retrieval/             BM25, dense, and hybrid retrieval
medrag_ui/                    Frontend-side retrieval adapter logic
scripts/                      Helper scripts for chunking and retrieval demos
examples/                     Small sample data for local testing
```

## Getting Started

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the frontend:

```bash
streamlit run app.py
```

## Chunking

Build chunks from a raw CSV:

```bash
python3 scripts/build_chunks.py \
  --input examples/raw_medquad_example.csv \
  --output artifacts/chunks/medquad_chunks.csv \
  --text-column answer \
  --document-id-column document_id \
  --source-url-column source_page_url
```

The chunking pipeline outputs a CSV with the same stable schema used by all retrievers.

## Retrieval Demos

Run BM25:

```bash
python3 scripts/run_bm25_demo.py \
  --chunks examples/generated_chunks_example.csv \
  --query "What are symptoms of iron deficiency anemia?" \
  --top-k 2
```

Run dense retrieval:

```bash
python3 scripts/run_dense_demo.py \
  --chunks examples/generated_chunks_example.csv \
  --query "What are symptoms of iron deficiency anemia?" \
  --top-k 2
```

Run hybrid retrieval:

```bash
python3 scripts/run_hybrid_demo.py \
  --chunks examples/generated_chunks_example.csv \
  --query "What are symptoms of iron deficiency anemia?" \
  --top-k 2 \
  --candidate-k 3
```

## Frontend Status

The frontend is already connected to all three retrieval modes:

- `Sparse (BM25)`
- `Dense`
- `Hybrid`

It currently shows retrieved evidence directly in the answer preview and presents citations from the retrieved chunks. This supports demos and retrieval analysis now, while keeping the interface ready for local LLM integration next.

## Evaluation Direction

The intended evaluation focus for the project includes:

- `Recall@k`
- `Precision@k`
- qualitative answer grounding and citation consistency

Because retrieval methods share the same chunk-level contract, they can be compared directly on the same data.

## Next Steps

- connect retrieved evidence to a local LLM for grounded generation
- expand evaluation using MedQuAD-based benchmarks
- improve citation rendering and answer faithfulness checks
- test larger chunk corpora and model variants
