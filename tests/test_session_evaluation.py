from __future__ import annotations

import asyncio
from pathlib import Path

from evaluator.session_evaluation import (
    SessionEvaluationError,
    evaluate_session,
    load_session_inputs,
)
from llm.schemas import ProviderMetadata, StructuredGeneration


class RecordingGenerator:
    async def generate_structured(
        self, messages: object, *args: object, **kwargs: object
    ) -> dict[str, object]:
        question_id = 1 if "Question one?" in str(messages) else 2
        score = 4 if question_id == 1 else 2
        return StructuredGeneration(
            output={
                "question_id": question_id,
                "score": score,
                "rating_label": "Good" if score == 4 else "Weak",
                "evidence": [f"Evidence for question {question_id}"],
                "follow_up": f"Follow-up {question_id}?",
                "feedback": f"Feedback {question_id}",
                "confidence": 0.8,
            },
            metadata=ProviderMetadata(
                provider="test",
                model="test-model",
                request_purpose="feedback_draft",
                latency_ms=1,
                input_tokens=1,
                output_tokens=1,
                estimated_cost_usd=0,
            ),
        ).model_dump(mode="json")


def _write_session(session: Path) -> None:
    session.mkdir()
    (session / "questions.json").write_text(
        """[
  {"id": 1, "question_text": "Question one?", "ideal_answer": "Ideal one"},
  {"id": 2, "question_text": "Question two?", "ideal_answer": "Ideal two"}
]""",
        encoding="utf-8",
    )
    (session / "transcript.json").write_text(
        """{
  "segments": [
    {"question_id": 1, "text": "Candidate's first answer.", "source": "system"},
    {"question_id": 2, "text": "Candidate's second answer.", "source": "system"}
  ]
}""",
        encoding="utf-8",
    )


def test_evaluate_session_writes_evidence_grounded_preview(tmp_path: Path) -> None:
    session = tmp_path / "saved_session"
    _write_session(session)

    result = asyncio.run(
        evaluate_session(load_session_inputs(session), RecordingGenerator())
    )

    assert result.overall_recommendation == "Borderline"
    assert [evaluation.question_id for evaluation in result.question_scores] == [1, 2]
    assert result.evaluation_path.is_file()
    assert result.feedback_preview_path.read_text(encoding="utf-8") == (
        "# Feedback preview\n\n"
        "This file is a draft only. It is not submitted to FloCareer.\n\n"
        "## Question 1 — Good (4/5)\n\n"
        "Feedback 1\n\n"
        "Evidence: Evidence for question 1\n\n"
        "## Question 2 — Weak (2/5)\n\n"
        "Feedback 2\n\n"
        "Evidence: Evidence for question 2\n"
    )


def test_session_input_rejects_unmapped_candidate_transcript_segments(
    tmp_path: Path,
) -> None:
    session = tmp_path / "saved_session"
    _write_session(session)
    (session / "transcript.json").write_text(
        '{"segments": [{"text": "An answer without a question boundary.", "source": "system"}]}',
        encoding="utf-8",
    )

    try:
        load_session_inputs(session)
    except SessionEvaluationError as error:
        assert "question_id" in str(error)
    else:
        raise AssertionError("unmapped candidate segments must be rejected")
