from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medrag_chunking import ChunkingConfig, chunk_dataset

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk raw medical documents into the MedRAG retrieval schema.")
    parser.add_argument("--input", required=True, help="Path to the raw input CSV.")
    parser.add_argument("--output", required=True, help="Path to the output chunk CSV.")
    parser.add_argument("--text-column", default="answer", help="Column containing the text to chunk.")
    parser.add_argument("--document-id-column", default="document_id", help="Column containing document ids.")
    parser.add_argument("--source-url-column", default="source_page_url", help="Column containing source URLs.")
    parser.add_argument("--chunk-size-words", type=int, default=120, help="Target chunk size in words.")
    parser.add_argument("--chunk-overlap-words", type=int, default=20, help="Word overlap between chunks.")
    parser.add_argument("--min-chunk-words", type=int, default=20, help="Minimum words required per chunk.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_df = pd.read_csv(args.input)
    config = ChunkingConfig(
        text_column=args.text_column,
        document_id_column=args.document_id_column,
        source_url_column=args.source_url_column,
        chunk_size_words=args.chunk_size_words,
        chunk_overlap_words=args.chunk_overlap_words,
        min_chunk_words=args.min_chunk_words,
    )
    chunks_df = chunk_dataset(raw_df, config=config)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_df.to_csv(output_path, index=False)
    print(f"Saved {len(chunks_df)} chunks to {output_path}")
    print(chunks_df.head(min(len(chunks_df), 5)).to_string(index=False))


if __name__ == "__main__":
    main()
