from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medrag_retrieval import DenseRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dense retrieval on a chunk CSV.")
    parser.add_argument("--chunks", required=True, help="Path to the chunk CSV file.")
    parser.add_argument("--query", required=True, help="Query string to retrieve against.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of rows to return.")
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence embedding model name.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retriever = DenseRetriever.from_csv(args.chunks, model_name=args.model_name)
    retriever.build_index(batch_size=args.batch_size)
    results = retriever.retrieve(args.query, top_k=args.top_k)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
