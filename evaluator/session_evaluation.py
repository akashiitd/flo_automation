"""Offline, evidence-grounded evaluation of an explicitly segmented session."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from app.questions import InterviewQuestion
from evaluator.scoring import StructuredGenerator, evaluate_answer
from llm.schemas import EvaluationInput, ModelClass, QuestionEvaluation
from llm.usage_tracker import UsageTracker


class SessionEvaluationError(ValueError):
    """A saved session is incomplete or cannot be evaluated safely."""


@dataclass(frozen=True, slots=True)
class SessionInputs:
    session_dir: Path
    questions: tuple[InterviewQuestion, ...]
    answers_by_question_id: Mapping[int, str]


@dataclass(frozen=True, slots=True)
class SessionQuestionScore:
    question_id: int
    score: int
    rating_label: str
    evidence: tuple[str, ...]
    follow_up: str
    feedback: str
    confidence: float


@dataclass(frozen=True, slots=True)
class SessionEvaluationResult:
    overall_recommendation: str
    confidence: float
    question_scores: tuple[SessionQuestionScore, ...]
    evaluation_path: Path
    feedback_preview_path: Path


def _read_json_object(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise SessionEvaluationError(f"Could not read {path.name}: {error}") from error
    except json.JSONDecodeError as error:
        raise SessionEvaluationError(
            f"{path.name} is not valid JSON: {error}"
        ) from error


def _require_text(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise SessionEvaluationError(f"{field} must be a non-empty string")
    return text


def load_session_inputs(session_dir: Path) -> SessionInputs:
    """Load only candidate-only segments that have a recorded question boundary."""

    session = session_dir.resolve()
    raw_questions = _read_json_object(session / "questions.json")
    if not isinstance(raw_questions, list) or not raw_questions:
        raise SessionEvaluationError(
            "questions.json must contain at least one question"
        )
    questions: list[InterviewQuestion] = []
    for position, raw_question_value in enumerate(raw_questions, start=1):
        if not isinstance(raw_question_value, dict):
            raise SessionEvaluationError(
                f"questions.json item {position} must be an object"
            )
        raw_question = cast(dict[str, object], raw_question_value)
        raw_question_id = raw_question.get("id")
        if not isinstance(raw_question_id, int | str):
            raise SessionEvaluationError(
                f"questions.json item {position} has an invalid id"
            )
        try:
            question_id = int(raw_question_id)
        except ValueError as error:
            raise SessionEvaluationError(
                f"questions.json item {position} has an invalid id"
            ) from error
        questions.append(
            InterviewQuestion(
                id=question_id,
                question_text=_require_text(
                    raw_question.get("question_text"),
                    field=f"questions.json item {position}.question_text",
                ),
                ideal_answer=_require_text(
                    raw_question.get("ideal_answer"),
                    field=f"questions.json item {position}.ideal_answer",
                ),
            )
        )
    question_ids = {question.id for question in questions}
    if len(question_ids) != len(questions):
        raise SessionEvaluationError("questions.json contains duplicate question IDs")

    raw_transcript = _read_json_object(session / "transcript.json")
    if not isinstance(raw_transcript, dict):
        raise SessionEvaluationError("transcript.json must contain a segments array")
    raw_segments = raw_transcript.get("segments")
    if not isinstance(raw_segments, list):
        raise SessionEvaluationError("transcript.json must contain a segments array")

    answers: defaultdict[int, list[str]] = defaultdict(list)
    for position, raw_segment_value in enumerate(raw_segments, start=1):
        if not isinstance(raw_segment_value, dict):
            raise SessionEvaluationError(
                f"transcript.json segment {position} must be an object"
            )
        raw_segment = cast(dict[str, object], raw_segment_value)
        # The configured selected Loopback route emits candidate audio as system.
        # Ignore known non-candidate sources rather than accidentally scoring them.
        if str(raw_segment.get("source") or "").strip() != "system":
            continue
        text = str(raw_segment.get("text") or "").strip()
        if not text:
            continue
        raw_question_id = raw_segment.get("question_id")
        if not isinstance(raw_question_id, int | str):
            raise SessionEvaluationError(
                "Each candidate system-audio segment must include question_id; "
                "do not infer question boundaries from timestamps"
            )
        try:
            question_id = int(raw_question_id)
        except ValueError as error:
            raise SessionEvaluationError(
                "Each candidate system-audio segment must include question_id; "
                "do not infer question boundaries from timestamps"
            ) from error
        if question_id not in question_ids:
            raise SessionEvaluationError(
                f"transcript.json segment {position} references unknown question_id "
                f"{question_id}"
            )
        answers[question_id].append(text)

    if not answers:
        raise SessionEvaluationError(
            "No candidate-only system-audio answers were found"
        )
    return SessionInputs(
        session_dir=session,
        questions=tuple(questions),
        answers_by_question_id={key: " ".join(value) for key, value in answers.items()},
    )


def _recommendation(scores: Sequence[SessionQuestionScore]) -> str:
    average = sum(score.score for score in scores) / len(scores)
    if average >= 4.5:
        return "Strong Hire"
    if average >= 3.5:
        return "Hire"
    if average >= 2.5:
        return "Borderline"
    return "No Hire"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _feedback_preview(scores: Sequence[SessionQuestionScore]) -> str:
    lines = [
        "# Feedback preview",
        "",
        "This file is a draft only. It is not submitted to FloCareer.",
        "",
    ]
    for score in scores:
        lines.extend(
            (
                f"## Question {score.question_id} — {score.rating_label} ({score.score}/5)",
                "",
                score.feedback,
                "",
                f"Evidence: {'; '.join(score.evidence)}",
                "",
            )
        )
    return "\n".join(lines).rstrip() + "\n"


async def evaluate_session(
    inputs: SessionInputs,
    generator: StructuredGenerator,
    *,
    model_class: ModelClass = "deep",
) -> SessionEvaluationResult:
    """Evaluate recorded answers and write local drafts without browser side effects."""

    usage_tracker = UsageTracker(inputs.session_dir / "llm_usage.jsonl")
    scores: list[SessionQuestionScore] = []
    for question in inputs.questions:
        answer = inputs.answers_by_question_id.get(question.id)
        if answer is None:
            continue
        generation = await evaluate_answer(
            EvaluationInput(
                question_id=question.id,
                question=question.question_text,
                ideal_answer=question.ideal_answer,
                candidate_answer=answer,
            ),
            generator,
            model_class=model_class,
            usage_tracker=usage_tracker,
        )
        evaluation = QuestionEvaluation.model_validate(generation.output)
        scores.append(
            SessionQuestionScore(
                question_id=evaluation.question_id,
                score=evaluation.score,
                rating_label=evaluation.rating_label,
                evidence=tuple(evaluation.evidence),
                follow_up=evaluation.follow_up,
                feedback=evaluation.feedback,
                confidence=evaluation.confidence,
            )
        )
    if not scores:
        raise SessionEvaluationError("No question had a candidate answer to evaluate")

    confidence = sum(score.confidence for score in scores) / len(scores)
    recommendation = _recommendation(scores)
    evaluation_path = inputs.session_dir / "evaluation.json"
    preview_path = inputs.session_dir / "feedback_preview.md"
    _write_json(
        evaluation_path,
        {
            "schema_version": 1,
            "overall_recommendation": recommendation,
            "confidence": confidence,
            "question_scores": [asdict(score) for score in scores],
            "submission": "blocked_pending_human_review",
        },
    )
    preview_path.write_text(_feedback_preview(scores), encoding="utf-8")
    return SessionEvaluationResult(
        overall_recommendation=recommendation,
        confidence=confidence,
        question_scores=tuple(scores),
        evaluation_path=evaluation_path,
        feedback_preview_path=preview_path,
    )
