from __future__ import annotations

from dataclasses import dataclass


REQUIRED_COLUMNS = (
    "chunk_id",
    "chunk_text",
    "document_id",
    "source_page_url",
)


@dataclass(frozen=True)
class RetrievalConfig:
    """Stable retrieval contract for all retriever implementations."""

    text_column: str = "chunk_text"
    id_column: str = "chunk_id"
    document_column: str = "document_id"
    source_url_column: str = "source_page_url"
    score_column: str = "score"


CONTRACT_DESCRIPTION = (
    "Retrieval takes in a query string and returns top-k rows from the chunk schema "
    "with an added score column."
)
