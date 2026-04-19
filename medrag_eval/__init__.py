"""Evaluation module for MedRAG retrieval."""

from medrag_eval.metrics import precision_at_k, recall_at_k
from medrag_eval.evaluator import evaluate_retriever, evaluate_all_methods

__all__ = [
    "precision_at_k",
    "recall_at_k",
    "evaluate_retriever",
    "evaluate_all_methods",
]
