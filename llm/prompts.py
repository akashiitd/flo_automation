"""Versioned prompts for structured candidate-answer evaluation."""

from __future__ import annotations

from llm.schemas import EvaluationInput


SCORING_SYSTEM_PROMPT = """You evaluate one interview answer against its rubric.
Return exactly one JSON object matching the supplied schema.

Rules:
- Score from 1 to 5: 5 excellent, 4 good, 3 average, 2 weak, 1 poor.
- Quote or closely paraphrase concrete evidence from the candidate answer.
- Do not invent claims, skills, or answers not present in the candidate answer.
- Give practical, concise feedback.
- Provide one concise, relevant follow-up question only.
- The follow-up must not reveal the ideal answer.
- Confidence must be between 0 and 1.
"""


def scoring_messages(request: EvaluationInput) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SCORING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Evaluate this answer using only the supplied evidence:\n"
                f"{request.model_dump_json(indent=2)}"
            ),
        },
    ]
