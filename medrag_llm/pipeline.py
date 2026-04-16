from __future__ import annotations

from typing import Any

from medrag_llm.client import OllamaClient
from medrag_llm.prompts import SYSTEM_PROMPT, build_user_prompt


def generate_grounded_answer(
    *,
    query: str,
    citations: list[dict[str, str]],
    model_name: str = "llama3",
    temperature: float = 0.2,
    client: OllamaClient | None = None,
) -> str:
    if not citations:
        return (
            "The system could not find enough retrieved evidence to support a grounded answer "
            "for this question."
        )

    client = client or OllamaClient()
    user_prompt = build_user_prompt(query, citations)
    return client.generate(
        model=model_name,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=temperature,
    )
