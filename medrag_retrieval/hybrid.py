from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from medrag_retrieval.bm25 import BM25Retriever
from medrag_retrieval.contracts import RetrievalConfig
from medrag_retrieval.data import load_chunks_csv, validate_chunk_schema
from medrag_retrieval.dense import DenseRetriever


class HybridRetriever:
    """
    Hybrid retrieval baseline:
    1. Use BM25 for initial candidate recall.
    2. Use precomputed dense embeddings for reranking.
    3. Return top-k rows from the stable chunk schema plus a score column.
    """

    def __init__(
        self,
        chunks_df: pd.DataFrame,
        *,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        config: RetrievalConfig | None = None,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        normalize_embeddings: bool = True,
    ) -> None:
        self.config = config or RetrievalConfig()
        self.chunks_df = validate_chunk_schema(chunks_df)
        self.bm25 = BM25Retriever(self.chunks_df, config=self.config, k1=bm25_k1, b=bm25_b)
        self.dense = DenseRetriever(
            self.chunks_df,
            model_name=model_name,
            config=self.config,
            normalize_embeddings=normalize_embeddings,
        )

    @classmethod
    def from_csv(
        cls,
        path: str,
        *,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        config: RetrievalConfig | None = None,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        normalize_embeddings: bool = True,
    ) -> "HybridRetriever":
        return cls(
            load_chunks_csv(path),
            model_name=model_name,
            config=config,
            bm25_k1=bm25_k1,
            bm25_b=bm25_b,
            normalize_embeddings=normalize_embeddings,
        )

    def build_dense_embeddings(self, *, batch_size: int = 64) -> None:
        self.dense.ensure_embeddings(batch_size=batch_size)

    def retrieve(self, query: str, top_k: int = 5, candidate_k: int = 20, batch_size: int = 64) -> pd.DataFrame:
        """
        Contract:
        Retrieval takes in a query string and returns top-k rows from the chunk schema
        with an added score column.
        """
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if candidate_k < top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k")

        candidates = self.bm25.retrieve(query, top_k=candidate_k)
        if candidates.empty:
            empty = self.chunks_df.head(0).copy()
            empty[self.config.score_column] = pd.Series(dtype="float64")
            return empty

        all_embeddings = self.dense.ensure_embeddings(batch_size=batch_size)
        query_embedding = self.dense.encode_query(query)
        candidate_indices = self._resolve_candidate_indices(candidates[self.config.id_column].tolist())
        candidate_embeddings = all_embeddings[candidate_indices]

        rerank_scores = candidate_embeddings @ query_embedding
        ranked_order = np.argsort(-rerank_scores)[:top_k]

        results = candidates.iloc[ranked_order].copy()
        results[self.config.score_column] = [float(rerank_scores[idx]) for idx in ranked_order]
        return results.reset_index(drop=True)

    def _resolve_candidate_indices(self, chunk_ids: Iterable[str]) -> list[int]:
        id_to_index = {
            chunk_id: row_idx
            for row_idx, chunk_id in enumerate(self.chunks_df[self.config.id_column].tolist())
        }
        return [id_to_index[chunk_id] for chunk_id in chunk_ids]
