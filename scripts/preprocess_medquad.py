"""
Preprocess the raw MedQuAD CSV into a clean, evaluation-ready dataset.

Usage:
    python3 scripts/preprocess_medquad.py \
        --input medquad.csv \
        --output medquad_preprocessed.csv

Download medquad.csv from:
    https://www.kaggle.com/datasets/pythonafroz/medquad-medical-question-answer-for-ai-research
"""
from __future__ import annotations

import argparse
import html
import re
import unicodedata

import pandas as pd


def normalize_text(text: str, lowercase: bool = False) -> str:
    if pd.isna(text):
        return ""
    text = str(text)
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = re.sub(r"\s+", " ", text).strip()
    if lowercase:
        text = text.lower()
    return text


def is_low_quality_row(question: str, answer: str, min_answer_words: int = 6) -> bool:
    if not question or not answer:
        return True
    if len(question.split()) < 2:
        return True
    if len(answer.split()) < min_answer_words:
        return True
    bad_patterns = [r"^n/?a$", r"^unknown$", r"^not available$", r"^none$"]
    answer_l = answer.lower().strip()
    return any(re.match(p, answer_l) for p in bad_patterns)


def preprocess_medquad(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["source_name"] = df["source"]
    df["question_clean"] = df["question"].apply(lambda x: normalize_text(x, lowercase=False))
    df["answer_clean"] = df["answer"].apply(lambda x: normalize_text(x, lowercase=False))

    df = df[(df["question_clean"] != "") & (df["answer_clean"] != "")].copy()

    mask_low_quality = df.apply(
        lambda row: is_low_quality_row(row["question_clean"], row["answer_clean"]),
        axis=1,
    )
    df = df[~mask_low_quality].copy()

    df["_question_dedup"] = df["question"].apply(lambda x: normalize_text(x, lowercase=True))
    df["_answer_dedup"] = df["answer"].apply(lambda x: normalize_text(x, lowercase=True))
    df = df.drop_duplicates(subset=["_question_dedup", "_answer_dedup"]).copy()
    df = df.drop(columns=["_question_dedup", "_answer_dedup"])

    df = df.reset_index(drop=True)
    df["document_id"] = [f"doc_{i:06d}" for i in range(len(df))]
    df["answer"] = df["answer_clean"]
    df["source_page_url"] = ""
    df["content_question_answer"] = (
        "Question: " + df["question_clean"] + " Answer: " + df["answer_clean"]
    )
    df["question_word_count"] = df["question_clean"].str.split().str.len()
    df["answer_word_count"] = df["answer_clean"].str.split().str.len()

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess MedQuAD CSV for evaluation")
    parser.add_argument("--input", default="medquad.csv", help="Path to raw medquad.csv")
    parser.add_argument("--output", default="medquad_preprocessed.csv", help="Output path")
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input)
    print(f"Raw rows: {len(df)}")

    df = df.dropna(subset=["answer"])
    df = df[df["answer"].str.split().str.len() >= 6].copy()

    df_clean = preprocess_medquad(df)

    df_clean.to_csv(args.output, index=False)
    print(f"Saved {len(df_clean)} rows to {args.output}")
    print(f"Removed {len(df) - len(df_clean)} duplicates/low-quality rows")
    print(f"Columns: {list(df_clean.columns)}")


if __name__ == "__main__":
    main()
