"""
Analyze how chunk size and overlap affect BM25 Recall@k.

Sweeps over chunk_size_words and chunk_overlap_words, re-chunks the raw data
with each config, runs BM25 evaluation, and prints a comparison table.

Usage:
    python3 scripts/run_chunking_analysis.py \
        --questions examples/raw_medquad_example.csv \
        --k 5
"""
from __future__ import annotations

import argparse
from itertools import product

import pandas as pd

from medrag_chunking import ChunkingConfig, chunk_dataset
from medrag_eval.evaluator import evaluate_retriever
from medrag_retrieval import BM25Retriever


CHUNK_SIZES = [60, 100, 150, 200]
OVERLAPS = [0, 10, 20, 30]


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk size and overlap impact on Recall@k")
    parser.add_argument("--questions", default="examples/raw_medquad_example.csv")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    questions_df = pd.read_csv(args.questions)
    print(f"Loaded {len(questions_df)} questions\n")

    rows = []
    for chunk_size, overlap in product(CHUNK_SIZES, OVERLAPS):
        if overlap >= chunk_size:
            continue

        chunks_df = chunk_dataset(
            questions_df,
            ChunkingConfig(
                text_column="answer",
                document_id_column="document_id",
                source_url_column="source_page_url",
                chunk_size_words=chunk_size,
                chunk_overlap_words=overlap,
                min_chunk_words=max(10, overlap + 1),
            ),
        )

        if chunks_df.empty:
            continue

        retriever = BM25Retriever(chunks_df)
        metrics = evaluate_retriever(questions_df, retriever, k=args.k)

        rows.append({
            "chunk_size": chunk_size,
            "overlap": overlap,
            "n_chunks": len(chunks_df),
            f"Recall@{args.k}": metrics[f"Recall@{args.k}"],
            f"Precision@{args.k}": metrics[f"Precision@{args.k}"],
        })

    results = pd.DataFrame(rows)
    print(f"=== Chunking Analysis (BM25, k={args.k}) ===")
    print(results.to_string(index=False))

    best = results.loc[results[f"Recall@{args.k}"].idxmax()]
    print(f"\nBest config: chunk_size={int(best['chunk_size'])}, overlap={int(best['overlap'])} "
          f"→ Recall@{args.k}={best[f'Recall@{args.k}']:.4f}")


if __name__ == "__main__":
    main()
