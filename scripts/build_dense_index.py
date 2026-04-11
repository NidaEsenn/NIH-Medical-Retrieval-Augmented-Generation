from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medrag_retrieval import DenseRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute chunk embeddings and build a FAISS index.")
    parser.add_argument("--chunks", required=True, help="Path to the chunk CSV file.")
    parser.add_argument("--index-out", required=True, help="Where to save the FAISS index.")
    parser.add_argument("--embeddings-out", help="Optional path to save raw embedding vectors.")
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
    retriever.save_index(args.index_out, embeddings_path=args.embeddings_out)
    print(f"Saved FAISS index to {args.index_out}")
    if args.embeddings_out:
        print(f"Saved embeddings to {args.embeddings_out}")


if __name__ == "__main__":
    main()
