from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from medrag_retrieval.contracts import RetrievalConfig
from medrag_retrieval.data import load_chunks_csv, validate_chunk_schema


TOKEN_PATTERN = re.compile(r"\b\w+\b")


def default_tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text.lower())


@dataclass
class BM25Index:
    tokens_by_row: List[List[str]]
    term_frequencies: List[Counter]
    inverted_index: Dict[str, List[int]]
    idf: Dict[str, float]
    document_lengths: List[int]
    avg_document_length: float


class BM25Retriever:
    """BM25 baseline built around a stable chunk schema."""

    def __init__(
        self,
        chunks_df: pd.DataFrame,
        *,
        config: RetrievalConfig | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.config = config or RetrievalConfig()
        self.k1 = k1
        self.b = b
        self.chunks_df = validate_chunk_schema(chunks_df)
        self.index = self._build_index(self.chunks_df[self.config.text_column].tolist())

    @classmethod
    def from_csv(
        cls,
        path: str,
        *,
        config: RetrievalConfig | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> "BM25Retriever":
        return cls(load_chunks_csv(path), config=config, k1=k1, b=b)

    def retrieve(self, query: str, top_k: int = 5) -> pd.DataFrame:
        """
        Contract:
        Retrieval takes in a query string and returns top-k rows from the chunk schema,
        ranked by BM25 score, with an added score column.
        """
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        query_tokens = default_tokenize(query)
        if not query_tokens:
            results = self.chunks_df.head(0).copy()
            results[self.config.score_column] = pd.Series(dtype="float64")
            return results

        scores = self._score_query(query_tokens)
        ranked_indices = sorted(scores, key=scores.get, reverse=True)[:top_k]

        if not ranked_indices:
            results = self.chunks_df.head(0).copy()
            results[self.config.score_column] = pd.Series(dtype="float64")
            return results

        results = self.chunks_df.iloc[ranked_indices].copy()
        results[self.config.score_column] = [scores[row_idx] for row_idx in ranked_indices]
        return results.reset_index(drop=True)

    def _build_index(self, texts: List[str]) -> BM25Index:
        tokens_by_row: List[List[str]] = []
        term_frequencies: List[Counter] = []
        inverted_index: Dict[str, List[int]] = defaultdict(list)
        document_lengths: List[int] = []

        for row_idx, text in enumerate(texts):
            tokens = default_tokenize(text)
            frequencies = Counter(tokens)
            tokens_by_row.append(tokens)
            term_frequencies.append(frequencies)
            document_lengths.append(len(tokens))

            for term in frequencies:
                inverted_index[term].append(row_idx)

        document_count = len(texts)
        avg_document_length = sum(document_lengths) / document_count if document_count else 0.0
        idf = {
            term: math.log(1 + (document_count - len(postings) + 0.5) / (len(postings) + 0.5))
            for term, postings in inverted_index.items()
        }

        return BM25Index(
            tokens_by_row=tokens_by_row,
            term_frequencies=term_frequencies,
            inverted_index=dict(inverted_index),
            idf=idf,
            document_lengths=document_lengths,
            avg_document_length=avg_document_length,
        )

    def _score_query(self, query_tokens: List[str]) -> Dict[int, float]:
        scores: Dict[int, float] = defaultdict(float)
        unique_terms = set(query_tokens)

        for term in unique_terms:
            postings = self.index.inverted_index.get(term, [])
            if not postings:
                continue

            idf = self.index.idf.get(term, 0.0)
            for row_idx in postings:
                term_frequency = self.index.term_frequencies[row_idx][term]
                document_length = self.index.document_lengths[row_idx]
                norm = self.k1 * (
                    1 - self.b + self.b * document_length / max(self.index.avg_document_length, 1e-9)
                )
                contribution = idf * ((term_frequency * (self.k1 + 1)) / (term_frequency + norm))
                scores[row_idx] += contribution

        return dict(scores)
