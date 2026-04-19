from __future__ import annotations

from typing import Any

import pandas as pd

from medrag_eval.metrics import precision_at_k, recall_at_k


def _get_query_column(questions_df: pd.DataFrame) -> str:
    """Return 'question' if present, else 'answer' as a fallback query source."""
    if "question" in questions_df.columns:
        return "question"
    if "answer" in questions_df.columns:
        return "answer"
    raise ValueError("questions_df must have a 'question' or 'answer' column")


def evaluate_retriever(
    questions_df: pd.DataFrame,
    retriever: Any,
    *,
    k: int = 5,
    candidate_k: int = 20,
) -> dict[str, float]:
    """
    Compute mean Recall@k and Precision@k for a single retriever.

    questions_df must have columns: document_id, and either question or answer.
    retriever must implement .retrieve(query, top_k) -> DataFrame with document_id column.
    HybridRetriever also accepts candidate_k; it is passed when the retriever supports it.
    """
    query_col = _get_query_column(questions_df)
    recalls: list[float] = []
    precisions: list[float] = []

    for _, row in questions_df.iterrows():
        query = str(row[query_col])
        true_doc = str(row["document_id"])

        try:
            import inspect
            sig = inspect.signature(retriever.retrieve)
            if "candidate_k" in sig.parameters:
                results = retriever.retrieve(query, top_k=k, candidate_k=max(candidate_k, k))
            else:
                results = retriever.retrieve(query, top_k=k)
        except Exception:
            recalls.append(0.0)
            precisions.append(0.0)
            continue

        retrieved_doc_ids = results["document_id"].tolist() if not results.empty else []
        recalls.append(recall_at_k(retrieved_doc_ids, true_doc))
        precisions.append(precision_at_k(retrieved_doc_ids, true_doc))

    n = len(recalls)
    return {
        f"Recall@{k}": round(sum(recalls) / n, 4) if n else 0.0,
        f"Precision@{k}": round(sum(precisions) / n, 4) if n else 0.0,
        "n_questions": n,
    }


def evaluate_all_methods(
    questions_df: pd.DataFrame,
    chunks_df: pd.DataFrame,
    *,
    k: int = 5,
    candidate_k: int = 20,
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> pd.DataFrame:
    """
    Build all three retrievers, run evaluation, and return a comparison DataFrame.
    """
    from medrag_retrieval import BM25Retriever, DenseRetriever, HybridRetriever

    print("Building BM25 index...")
    bm25 = BM25Retriever(chunks_df)

    print("Building Dense index...")
    dense = DenseRetriever(chunks_df, model_name=dense_model)
    dense.build_index(batch_size=16)

    print("Building Hybrid index...")
    hybrid = HybridRetriever(chunks_df, model_name=dense_model)
    hybrid.build_dense_embeddings(batch_size=16)

    results = []
    for name, retriever in [("BM25", bm25), ("Dense", dense), ("Hybrid", hybrid)]:
        print(f"Evaluating {name}...")
        metrics = evaluate_retriever(questions_df, retriever, k=k, candidate_k=candidate_k)
        results.append({"Method": name, **metrics})

    return pd.DataFrame(results).set_index("Method")
