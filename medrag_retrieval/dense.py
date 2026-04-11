from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from medrag_retrieval.contracts import RetrievalConfig
from medrag_retrieval.data import load_chunks_csv, validate_chunk_schema


class DenseRetriever:
    """
    Dense retrieval baseline:
    1. Use a pre-trained sentence embedding model.
    2. Precompute embeddings for all chunks.
    3. Build a FAISS vector index for top-k retrieval.
    """

    def __init__(
        self,
        chunks_df: pd.DataFrame,
        *,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        config: RetrievalConfig | None = None,
        normalize_embeddings: bool = True,
    ) -> None:
        self.config = config or RetrievalConfig()
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.chunks_df = validate_chunk_schema(chunks_df)
        self.model = self._load_model(model_name)
        self.embeddings: np.ndarray | None = None
        self.index: Any | None = None

    @classmethod
    def from_csv(
        cls,
        path: str,
        *,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        config: RetrievalConfig | None = None,
        normalize_embeddings: bool = True,
    ) -> "DenseRetriever":
        return cls(
            load_chunks_csv(path),
            model_name=model_name,
            config=config,
            normalize_embeddings=normalize_embeddings,
        )

    def build_index(self, *, batch_size: int = 64) -> None:
        texts = self.chunks_df[self.config.text_column].tolist()
        self.embeddings = self._encode_texts(texts=texts, batch_size=batch_size)
        self.index = self._build_faiss_index(self.embeddings)

    def ensure_embeddings(self, *, batch_size: int = 64) -> np.ndarray:
        if self.embeddings is None:
            texts = self.chunks_df[self.config.text_column].tolist()
            self.embeddings = self._encode_texts(texts=texts, batch_size=batch_size)
        return self.embeddings

    def save_index(self, index_path: str | Path, embeddings_path: str | Path | None = None) -> None:
        if self.index is None or self.embeddings is None:
            raise ValueError("Dense index has not been built yet. Call build_index() first.")

        faiss = self._import_faiss()
        index_path = Path(index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))

        if embeddings_path is not None:
            embeddings_path = Path(embeddings_path)
            embeddings_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(embeddings_path, self.embeddings)

    def load_index(self, index_path: str | Path, embeddings_path: str | Path | None = None) -> None:
        faiss = self._import_faiss()
        self.index = faiss.read_index(str(index_path))

        if embeddings_path is not None:
            self.embeddings = np.load(embeddings_path)

    def retrieve(self, query: str, top_k: int = 5) -> pd.DataFrame:
        """
        Contract:
        Retrieval takes in a query string and returns top-k rows from the chunk schema
        with an added score column.
        """
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if self.index is None:
            raise ValueError("Dense index has not been built yet. Call build_index() first.")

        query_embedding = self._encode_texts(texts=[query], batch_size=1)
        scores, row_indices = self.index.search(query_embedding, top_k)

        ranked_rows = []
        ranked_scores = []
        for row_idx, score in zip(row_indices[0], scores[0]):
            if row_idx < 0:
                continue
            ranked_rows.append(int(row_idx))
            ranked_scores.append(float(score))

        if not ranked_rows:
            empty = self.chunks_df.head(0).copy()
            empty[self.config.score_column] = pd.Series(dtype="float64")
            return empty

        results = self.chunks_df.iloc[ranked_rows].copy()
        results[self.config.score_column] = ranked_scores
        return results.reset_index(drop=True)

    def encode_query(self, query: str) -> np.ndarray:
        return self._encode_texts(texts=[query], batch_size=1)[0]

    def _load_model(self, model_name: str) -> Any:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for dense retrieval. "
                "Install it with `pip install sentence-transformers`."
            ) from exc

        return SentenceTransformer(model_name)

    def _encode_texts(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )
        return np.asarray(embeddings, dtype="float32")

    def _build_faiss_index(self, embeddings: np.ndarray) -> Any:
        faiss = self._import_faiss()
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        return index

    def _import_faiss(self) -> Any:
        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is required for dense retrieval. Install it with `pip install faiss-cpu`."
            ) from exc

        return faiss
