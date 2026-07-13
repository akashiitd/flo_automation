from __future__ import annotations

import json
from pathlib import Path

from evaluator.skill_evaluation import (
    aggregate_skill_assessments,
    skill_evidence_from_question,
    write_skill_assessment_preview,
)
from orchestrator.state import (
    CoverageStatus,
    QuestionContentType,
    QuestionMappingSource,
    QuestionPlanItem,
    SkillAssessmentStatus,
    SkillParameter,
)


def _plan_item() -> QuestionPlanItem:
    return QuestionPlanItem(
        question_id=1,
        content_type=QuestionContentType.INTERVIEW_QUESTION,
        target_skill_ids=["api"],
        mandatory_skill_coverage=["api"],
        estimated_minutes=3,
        priority=100,
        selected=True,
        mapping_source=QuestionMappingSource.DETERMINISTIC,
        mapping_confidence=0.9,
        mapping_evidence=["question asks about API retries"],
    )


def _skills() -> list[SkillParameter]:
    return [
        SkillParameter(
            id="api",
            name="API",
            requirement="Mandatory",
            level="Professional",
            rating_scale=5,
        ),
        SkillParameter(
            id="python",
            name="Python",
            requirement="Mandatory",
            level="Professional",
            rating_scale=5,
        ),
    ]


def test_skill_assessments_use_only_question_mapped_evidence() -> None:
    evidence = skill_evidence_from_question(
        plan_item=_plan_item(),
        question_id=1,
        candidate_answer="I would use an idempotency key and bounded retries.",
        question_score=4,
        confidence=0.9,
    )

    assessments, coverage = aggregate_skill_assessments(
        skill_parameters=_skills(), evidence=evidence
    )

    by_skill = {item.skill_id: item for item in assessments}
    assert by_skill["api"].status is SkillAssessmentStatus.ASSESSED
    assert by_skill["api"].proposed_score == 4
    assert by_skill["python"].status is SkillAssessmentStatus.INSUFFICIENT_EVIDENCE
    assert by_skill["python"].proposed_score is None
    assert coverage["api"].status is CoverageStatus.SUFFICIENT
    assert coverage["python"].status is CoverageStatus.INSUFFICIENT_EVIDENCE


def test_low_confidence_mapped_evidence_never_invents_a_score() -> None:
    evidence = skill_evidence_from_question(
        plan_item=_plan_item(),
        question_id=1,
        candidate_answer="I would try retries.",
        question_score=5,
        confidence=0.4,
    )

    assessments, coverage = aggregate_skill_assessments(
        skill_parameters=_skills(), evidence=evidence
    )

    assert assessments[0].status is SkillAssessmentStatus.INSUFFICIENT_EVIDENCE
    assert assessments[0].proposed_score is None
    assert coverage["api"].status is CoverageStatus.PARTIALLY_ASSESSED


def test_skill_preview_is_local_and_explicitly_blocks_submission(
    tmp_path: Path,
) -> None:
    evidence = skill_evidence_from_question(
        plan_item=_plan_item(),
        question_id=1,
        candidate_answer="I would use retries.",
        question_score=4,
        confidence=0.9,
    )
    assessments, _ = aggregate_skill_assessments(
        skill_parameters=_skills(), evidence=evidence
    )

    json_path, markdown_path = write_skill_assessment_preview(
        tmp_path / "session", assessments=assessments, evidence=evidence
    )

    assert json.loads(json_path.read_text(encoding="utf-8"))["submission"] == (
        "blocked_pending_human_review"
    )
    assert "does not select FloCareer stars" in markdown_path.read_text(
        encoding="utf-8"
    )
