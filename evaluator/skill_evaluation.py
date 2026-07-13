"""Evidence-grounded skill previews; never platform rating mutations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

from orchestrator.state import (
    CoverageState,
    CoverageStatus,
    QuestionPlanItem,
    SkillAssessment,
    SkillAssessmentStatus,
    SkillEvidence,
    SkillParameter,
)

_SUFFICIENT_CONFIDENCE = 0.70


def skill_evidence_from_question(
    *,
    plan_item: QuestionPlanItem,
    question_id: int,
    candidate_answer: str,
    question_score: int,
    confidence: float,
) -> list[SkillEvidence]:
    """Create evidence only for skills explicitly mapped to this question."""

    answer = candidate_answer.strip()
    if not answer:
        return []
    skill_ids = dict.fromkeys(
        [*plan_item.target_skill_ids, *plan_item.mandatory_skill_coverage]
    )
    return [
        SkillEvidence(
            evidence_id=_evidence_id(question_id, skill_id, answer),
            skill_id=skill_id,
            question_id=question_id,
            transcript_evidence=answer[-4_000:],
            question_score=question_score,
            relevance_weight=plan_item.mapping_confidence,
            confidence=confidence,
        )
        for skill_id in skill_ids
    ]


def aggregate_skill_assessments(
    *,
    skill_parameters: Sequence[SkillParameter],
    evidence: Sequence[SkillEvidence],
) -> tuple[list[SkillAssessment], dict[str, CoverageState]]:
    """Aggregate mapped evidence without inventing a score for missing skills."""

    by_skill: dict[str, list[SkillEvidence]] = {}
    for item in evidence:
        by_skill.setdefault(item.skill_id, []).append(item)
    assessments: list[SkillAssessment] = []
    coverage: dict[str, CoverageState] = {}
    for skill in skill_parameters:
        items = by_skill.get(skill.id, [])
        if not items:
            assessments.append(
                SkillAssessment(
                    skill_id=skill.id,
                    proposed_score=None,
                    status=SkillAssessmentStatus.INSUFFICIENT_EVIDENCE,
                    rationale="No evaluated answer was mapped to this skill.",
                    confidence=0.0,
                )
            )
            coverage[skill.id] = CoverageState(
                status=CoverageStatus.INSUFFICIENT_EVIDENCE,
                confidence=0.0,
            )
            continue
        total_weight = sum(item.relevance_weight * item.confidence for item in items)
        weighted_score = sum(
            item.question_score * item.relevance_weight * item.confidence
            for item in items
        )
        confidence = sum(item.confidence for item in items) / len(items)
        score = round(weighted_score / total_weight) if total_weight else None
        evidence_ids = [item.evidence_id for item in items]
        if score is None or confidence < _SUFFICIENT_CONFIDENCE:
            status = SkillAssessmentStatus.INSUFFICIENT_EVIDENCE
            rationale = "Mapped evidence is not confident enough for a proposed score."
            coverage_status = CoverageStatus.PARTIALLY_ASSESSED
            proposed_score = None
        else:
            status = SkillAssessmentStatus.ASSESSED
            rationale = "Proposed from evaluated answers mapped to this skill only."
            coverage_status = CoverageStatus.SUFFICIENT
            proposed_score = score
        assessments.append(
            SkillAssessment(
                skill_id=skill.id,
                proposed_score=proposed_score,
                status=status,
                evidence_ids=evidence_ids,
                rationale=rationale,
                confidence=confidence,
            )
        )
        coverage[skill.id] = CoverageState(
            status=coverage_status,
            confidence=confidence,
            evidence_ids=evidence_ids,
        )
    return assessments, coverage


def write_skill_assessment_preview(
    session_dir: Path,
    *,
    assessments: Sequence[SkillAssessment],
    evidence: Sequence[SkillEvidence],
) -> tuple[Path, Path]:
    """Write local-only preview artifacts, never FloCareer star selections."""

    session_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    json_path = session_dir / "skill_assessment_preview.json"
    markdown_path = session_dir / "skill_assessment_preview.md"
    json_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "submission": "blocked_pending_human_review",
                "assessments": [item.model_dump(mode="json") for item in assessments],
                "evidence": [item.model_dump(mode="json") for item in evidence],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Skill assessment preview",
        "",
        "Draft only. It does not select FloCareer stars or submit feedback.",
        "",
    ]
    for assessment in assessments:
        score = (
            f"{assessment.proposed_score}/5"
            if assessment.proposed_score is not None
            else "insufficient evidence"
        )
        lines.extend(
            [
                f"## {assessment.skill_id} — {score}",
                "",
                assessment.rationale,
                "",
            ]
        )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, markdown_path


def _evidence_id(question_id: int, skill_id: str, answer: str) -> str:
    digest = hashlib.sha256(f"{question_id}\0{skill_id}\0{answer}".encode()).hexdigest()
    return f"skill-evidence-{digest[:24]}"


__all__ = [
    "aggregate_skill_assessments",
    "skill_evidence_from_question",
    "write_skill_assessment_preview",
]
