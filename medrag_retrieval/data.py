from __future__ import annotations

from pathlib import Path

import pandas as pd

from medrag_retrieval.contracts import REQUIRED_COLUMNS


def validate_chunk_schema(df: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        missing_display = ", ".join(missing_columns)
        raise ValueError(f"Chunk data is missing required columns: {missing_display}")

    cleaned = df.copy()
    cleaned["chunk_text"] = cleaned["chunk_text"].fillna("").astype(str)
    cleaned = cleaned[cleaned["chunk_text"].str.strip().ne("")]

    if cleaned["chunk_id"].duplicated().any():
        duplicates = cleaned.loc[cleaned["chunk_id"].duplicated(), "chunk_id"].astype(str).tolist()
        duplicate_preview = ", ".join(duplicates[:5])
        raise ValueError(f"chunk_id values must be unique. Duplicate examples: {duplicate_preview}")

    return cleaned.reset_index(drop=True)


def load_chunks_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Chunk CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    return validate_chunk_schema(df)
