from __future__ import annotations

from typing import Dict, List


def _citations_for_query(query: str) -> List[Dict[str, str]]:
    topic = query.strip().rstrip("?") or "medical guidance"
    return [
        {
            "title": "Iron-Deficiency Anemia",
            "source": "NIH: National Heart, Lung, and Blood Institute",
            "focus_area": "Blood Disorders",
            "snippet": (
                "Common symptoms may include fatigue, weakness, dizziness, headaches, "
                "shortness of breath, and pale skin. Symptoms can vary depending on severity."
            ),
            "url": "https://www.nhlbi.nih.gov/health/anemia/iron-deficiency-anemia",
        },
        {
            "title": "Anemia",
            "source": "MedlinePlus",
            "focus_area": "Symptoms and Diagnosis",
            "snippet": (
                f"For questions related to {topic.lower()}, NIH-aligned sources often emphasize "
                "matching the answer to symptom severity, likely causes, and when a patient should seek care."
            ),
            "url": "https://medlineplus.gov/anemia.html",
        },
        {
            "title": "Dietary Iron",
            "source": "NIH Office of Dietary Supplements",
            "focus_area": "Nutrition",
            "snippet": (
                "Low iron intake or reduced iron absorption can contribute to iron deficiency. "
                "Evaluation typically considers both symptoms and the underlying cause."
            ),
            "url": "https://ods.od.nih.gov/factsheets/Iron-Consumer/",
        },
    ]


def generate_demo_response(query: str, strategy: str, top_k: int) -> Dict[str, object]:
    citations = _citations_for_query(query)[:top_k]

    answer = (
        "People with iron deficiency anemia commonly report fatigue, weakness, headaches, "
        "dizziness, shortness of breath, and pale skin. In a grounded medical QA setting, "
        "the answer should also note that symptom intensity varies, and persistent or worsening "
        "symptoms should prompt evaluation by a clinician because anemia can have multiple causes."
    )

    metrics_by_strategy = {
        "Hybrid": {"Recall@5": "0.92", "Precision@5": "0.78", "Answer support": "High"},
        "Sparse (BM25)": {"Recall@5": "0.81", "Precision@5": "0.69", "Answer support": "Moderate"},
        "Dense": {"Recall@5": "0.87", "Precision@5": "0.73", "Answer support": "High"},
    }

    grounding_by_strategy = {
        "Hybrid": "0.91",
        "Sparse (BM25)": "0.76",
        "Dense": "0.84",
    }

    return {
        "answer": answer,
        "retrieval_strategy": strategy,
        "grounding_score": grounding_by_strategy.get(strategy, "0.80"),
        "citations": citations,
        "metrics": metrics_by_strategy.get(strategy, metrics_by_strategy["Hybrid"]),
    }
