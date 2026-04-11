from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from medrag_retrieval.contracts import REQUIRED_COLUMNS


WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True)
class ChunkingConfig:
    """Configuration for converting raw rows into the fixed retrieval chunk schema."""

    text_column: str = "answer"
    document_id_column: str = "document_id"
    source_url_column: str = "source_page_url"
    chunk_size_words: int = 120
    chunk_overlap_words: int = 20
    min_chunk_words: int = 20


def normalize_text(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def split_text_into_chunks(
    text: str,
    *,
    chunk_size_words: int,
    chunk_overlap_words: int,
    min_chunk_words: int,
) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []

    words = normalized.split(" ")
    if len(words) <= chunk_size_words:
        return [normalized] if len(words) >= min_chunk_words else []

    step = chunk_size_words - chunk_overlap_words
    if step <= 0:
        raise ValueError("chunk_overlap_words must be smaller than chunk_size_words")

    chunks: list[str] = []
    for start in range(0, len(words), step):
        end = start + chunk_size_words
        chunk_words = words[start:end]
        if len(chunk_words) < min_chunk_words:
            continue
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break

    return chunks


def _resolve_document_id(row: pd.Series, row_idx: int, config: ChunkingConfig) -> str:
    if config.document_id_column in row and pd.notna(row[config.document_id_column]):
        value = str(row[config.document_id_column]).strip()
        if value:
            return value
    return f"doc_{row_idx}"


def _resolve_source_url(row: pd.Series, config: ChunkingConfig) -> str:
    if config.source_url_column in row and pd.notna(row[config.source_url_column]):
        return str(row[config.source_url_column]).strip()
    return ""


def _build_chunk_rows(raw_df: pd.DataFrame, config: ChunkingConfig) -> list[dict[str, str]]:
    if config.text_column not in raw_df.columns:
        raise ValueError(f"Input data must contain text column: {config.text_column}")

    chunk_rows: list[dict[str, str]] = []

    for row_idx, row in raw_df.iterrows():
        source_text = row[config.text_column]
        if pd.isna(source_text):
            continue

        document_id = _resolve_document_id(row, row_idx, config)
        source_page_url = _resolve_source_url(row, config)
        chunks = split_text_into_chunks(
            str(source_text),
            chunk_size_words=config.chunk_size_words,
            chunk_overlap_words=config.chunk_overlap_words,
            min_chunk_words=config.min_chunk_words,
        )

        for chunk_idx, chunk_text in enumerate(chunks):
            chunk_rows.append(
                {
                    "chunk_id": f"{document_id}_chunk_{chunk_idx}",
                    "chunk_text": chunk_text,
                    "document_id": document_id,
                    "source_page_url": source_page_url,
                }
            )

    return chunk_rows


def chunk_dataset(raw_df: pd.DataFrame, config: ChunkingConfig | None = None) -> pd.DataFrame:
    config = config or ChunkingConfig()
    chunk_rows = _build_chunk_rows(raw_df, config)
    chunks_df = pd.DataFrame(chunk_rows, columns=REQUIRED_COLUMNS)
    return chunks_df.reset_index(drop=True)


def chunk_csv_to_csv(
    input_path: str | Path,
    output_path: str | Path,
    *,
    config: ChunkingConfig | None = None,
) -> pd.DataFrame:
    raw_df = pd.read_csv(input_path)
    chunks_df = chunk_dataset(raw_df, config=config)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_df.to_csv(output_path, index=False)
    return chunks_df
