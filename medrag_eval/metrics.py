from __future__ import annotations


def recall_at_k(retrieved_doc_ids: list[str], true_doc_id: str) -> float:
    """1.0 if true_doc_id appears anywhere in retrieved_doc_ids, else 0.0."""
    return 1.0 if true_doc_id in retrieved_doc_ids else 0.0


def precision_at_k(retrieved_doc_ids: list[str], true_doc_id: str) -> float:
    """Fraction of retrieved docs that match true_doc_id."""
    if not retrieved_doc_ids:
        return 0.0
    correct = sum(1 for d in retrieved_doc_ids if d == true_doc_id)
    return correct / len(retrieved_doc_ids)
