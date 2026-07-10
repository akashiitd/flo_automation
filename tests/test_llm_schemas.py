from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm.schemas import QuestionEvaluation


def test_question_evaluation_accepts_the_plan_contract() -> None:
    evaluation = QuestionEvaluation.model_validate(
        {
            "question_id": 1,
            "score": 3,
            "rating_label": "Average",
            "evidence": [
                "Candidate mentioned API layer",
                "Candidate missed observability and retries",
            ],
            "follow_up": "How would you handle retries and timeout failures?",
            "feedback": (
                "Candidate showed basic understanding but lacked production depth."
            ),
            "confidence": 0.72,
        }
    )

    assert evaluation.score == 3
    assert evaluation.rating_label == "Average"
    assert evaluation.confidence == 0.72


def test_question_evaluation_rejects_a_score_outside_one_to_five() -> None:
    with pytest.raises(ValidationError):
        QuestionEvaluation.model_validate(
            {
                "question_id": 1,
                "score": 6,
                "rating_label": "Excellent",
                "evidence": ["An unsupported claim"],
                "follow_up": None,
                "feedback": "Feedback text",
                "confidence": 0.8,
            }
        )


def test_question_evaluation_requires_a_follow_up_suggestion() -> None:
    with pytest.raises(ValidationError):
        QuestionEvaluation.model_validate(
            {
                "question_id": 1,
                "score": 3,
                "rating_label": "Average",
                "evidence": ["Candidate mentioned an API layer"],
                "follow_up": None,
                "feedback": "The answer omitted retries and observability.",
                "confidence": 0.8,
            }
        )
