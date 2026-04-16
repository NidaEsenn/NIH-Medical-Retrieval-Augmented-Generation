from __future__ import annotations

from typing import Iterable


SYSTEM_PROMPT = """You are a medical question answering assistant.

Use only the retrieved medical context provided to answer the user's question.
Do not use outside knowledge.
If the retrieved context does not contain enough information, say that the answer is not supported by the retrieved evidence.
Do not invent diagnoses, treatments, or facts.
Keep the answer concise, factual, and grounded in the provided context.
When possible, cite supporting chunks using square brackets like [1] or [2].
"""


def build_context_block(citations: Iterable[dict[str, str]]) -> str:
    blocks: list[str] = []
    for idx, citation in enumerate(citations, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[{idx}] document_id: {citation['title']}",
                    f"source_page_url: {citation['url']}",
                    f"context: {citation['snippet']}",
                ]
            )
        )
    return "\n\n".join(blocks)


def build_user_prompt(query: str, citations: list[dict[str, str]]) -> str:
    context_block = build_context_block(citations)
    return f"""Retrieved context:
{context_block}

User question:
{query}

Return:
1. A short grounded answer using only the retrieved context.
2. If the evidence is incomplete, say so clearly.
3. Include citation numbers for the chunks you used.
"""
