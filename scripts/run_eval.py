"""
Compare Recall@k and Precision@k across BM25, Dense, and Hybrid retrievers.

Usage:
    python3 scripts/run_eval.py \
        --questions examples/raw_medquad_example.csv \
        --chunks examples/generated_chunks_example.csv \
        --k 5

If --questions has a 'question' column it is used as the query.
Otherwise 'answer' is used as a proxy (useful for the example data).
"""
from __future__ import annotations

import argparse

import pandas as pd

from medrag_chunking import ChunkingConfig, chunk_dataset
from medrag_eval import evaluate_all_methods


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval methods with Recall@k and Precision@k")
    parser.add_argument("--questions", default="examples/raw_medquad_example.csv", help="CSV with question/answer + document_id")
    parser.add_argument("--chunks", default=None, help="Pre-built chunks CSV (auto-generated from questions if omitted)")
    parser.add_argument("--k", type=int, default=5, help="Top-k for evaluation")
    parser.add_argument("--candidate-k", type=int, default=20, help="BM25 candidate pool size for Hybrid")
    args = parser.parse_args()

    questions_df = pd.read_csv(args.questions)
    print(f"Loaded {len(questions_df)} questions from {args.questions}")

    if args.chunks:
        chunks_df = pd.read_csv(args.chunks)
        print(f"Loaded {len(chunks_df)} chunks from {args.chunks}")
    else:
        print("No chunks file provided — chunking from questions file...")
        chunks_df = chunk_dataset(
            questions_df,
            ChunkingConfig(
                text_column="answer",
                document_id_column="document_id",
                source_url_column="source_page_url",
            ),
        )
        print(f"Generated {len(chunks_df)} chunks")

    results = evaluate_all_methods(
        questions_df,
        chunks_df,
        k=args.k,
        candidate_k=args.candidate_k,
    )

    print(f"\n=== Retrieval Evaluation (k={args.k}) ===")
    print(results.to_string())


if __name__ == "__main__":
    main()
