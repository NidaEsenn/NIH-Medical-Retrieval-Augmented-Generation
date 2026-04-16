from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from medrag_chunking import ChunkingConfig, chunk_dataset
from medrag_llm import OllamaConnectionError, generate_grounded_answer
from medrag_retrieval import BM25Retriever, DenseRetriever, HybridRetriever


DEFAULT_GENERATED_CHUNKS = Path("examples/generated_chunks_example.csv")
DEFAULT_RAW_DATA = Path("examples/raw_medquad_example.csv")


def _resolve_chunks_path() -> Path:
    candidate_paths = [
        Path("artifacts/chunks/medquad_chunks.csv"),
        DEFAULT_GENERATED_CHUNKS,
    ]
    for path in candidate_paths:
        if path.exists():
            return path
    return DEFAULT_GENERATED_CHUNKS


def _ensure_demo_chunks() -> Path:
    chunks_path = _resolve_chunks_path()
    if chunks_path.exists():
        return chunks_path

    raw_df = pd.read_csv(DEFAULT_RAW_DATA)
    chunks_df = chunk_dataset(
        raw_df,
        ChunkingConfig(
            text_column="answer",
            document_id_column="document_id",
            source_url_column="source_page_url",
            chunk_size_words=25,
            chunk_overlap_words=5,
            min_chunk_words=10,
        ),
    )
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_df.to_csv(chunks_path, index=False)
    return chunks_path


@st.cache_resource(show_spinner=False)
def get_bm25_retriever(chunks_path: str) -> BM25Retriever:
    return BM25Retriever.from_csv(chunks_path)


@st.cache_resource(show_spinner=False)
def get_dense_retriever(chunks_path: str) -> DenseRetriever:
    retriever = DenseRetriever.from_csv(chunks_path)
    retriever.build_index(batch_size=16)
    return retriever


@st.cache_resource(show_spinner=False)
def get_hybrid_retriever(chunks_path: str) -> HybridRetriever:
    retriever = HybridRetriever.from_csv(chunks_path)
    retriever.build_dense_embeddings(batch_size=16)
    return retriever


def _rows_to_citations(rows: pd.DataFrame) -> List[Dict[str, str]]:
    citations: List[Dict[str, str]] = []
    for _, row in rows.iterrows():
        citations.append(
            {
                "title": str(row["document_id"]),
                "source": str(row["document_id"]),
                "focus_area": "Retrieved chunk",
                "snippet": str(row["chunk_text"]),
                "url": str(row["source_page_url"]),
            }
        )
    return citations


def _build_preview_answer(query: str, rows: pd.DataFrame, strategy: str) -> str:
    if rows.empty:
        return (
            f"No relevant chunks were retrieved for: '{query}'. "
            "Try a more specific medical query or load a richer chunk dataset."
        )

    snippets = [str(value).strip() for value in rows["chunk_text"].head(2).tolist()]
    evidence = " ".join(snippets)
    return (
        f"Using {strategy.lower()} retrieval, the system found evidence suggesting: {evidence} "
        "This is the retrieval preview used when a local LLM response is unavailable."
    )


def _build_metrics(rows: pd.DataFrame, strategy: str) -> Dict[str, str]:
    avg_score = float(rows["score"].mean()) if not rows.empty else 0.0
    max_score = float(rows["score"].max()) if not rows.empty else 0.0
    return {
        "Top-k returned": str(len(rows)),
        "Avg score": f"{avg_score:.3f}",
        "Best score": f"{max_score:.3f}",
        "Mode": strategy,
    }


def run_retrieval(
    query: str,
    strategy: str,
    top_k: int,
    candidate_k: int,
    temperature: float,
    model_name: str = "llama3",
) -> Dict[str, Any]:
    chunks_path = _ensure_demo_chunks()

    if strategy == "Sparse (BM25)":
        rows = get_bm25_retriever(str(chunks_path)).retrieve(query, top_k=top_k)
    elif strategy == "Dense":
        rows = get_dense_retriever(str(chunks_path)).retrieve(query, top_k=top_k)
    else:
        rows = get_hybrid_retriever(str(chunks_path)).retrieve(
            query,
            top_k=top_k,
            candidate_k=max(candidate_k, top_k),
            batch_size=16,
        )

    citations = _rows_to_citations(rows)
    avg_score = float(rows["score"].mean()) if not rows.empty else 0.0
    llm_error = ""

    try:
        answer = generate_grounded_answer(
            query=query,
            citations=citations,
            model_name=model_name,
            temperature=temperature,
        )
    except OllamaConnectionError as exc:
        llm_error = str(exc)
        answer = _build_preview_answer(query, rows, strategy)

    return {
        "answer": answer,
        "retrieval_strategy": strategy,
        "grounding_score": f"{avg_score:.3f}",
        "citations": citations,
        "metrics": _build_metrics(rows, strategy),
        "chunks_path": str(chunks_path),
        "llm_model": model_name,
        "llm_error": llm_error,
    }
