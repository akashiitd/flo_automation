"""Deterministic, offline planning from read-only FloCareer scan artifacts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from orchestrator.state import (
    QuestionContentType,
    QuestionMappingSource,
    QuestionPlanArtifact,
    QuestionPlanItem,
    SkillParameter,
    SkillParametersArtifact,
)


class QuestionPlanningError(ValueError):
    """Raised when planning artifacts are incomplete or structurally unsafe."""


@dataclass(frozen=True, slots=True)
class ScannedQuestion:
    id: int
    question_text: str
    ideal_answer: str
    has_code_editor: bool


@dataclass(frozen=True, slots=True)
class QuestionPlanResult:
    artifact: QuestionPlanArtifact
    plan_path: Path
    markdown_path: Path
    selected_question_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _Candidate:
    question: ScannedQuestion
    content_type: QuestionContentType
    target_skill_ids: tuple[str, ...]
    mandatory_skill_coverage: tuple[str, ...]
    mapping_confidence: float
    mapping_evidence: tuple[str, ...]
    estimated_minutes: float
    priority: int


_WORD = re.compile(r"[a-z0-9]+")
_SKILL_STOP_WORDS = frozenset(
    {
        "and",
        "development",
        "engineering",
        "for",
        "of",
        "the",
        "with",
    }
)
_INSTRUCTION_MARKERS = (
    "before you begin",
    "interview instructions",
    "please read",
    "setup instruction",
    "welcome to the interview",
    "this interview will",
)
_RESERVED_MINUTES = 8.0
_CODING_MINUTES = 6.0
_QUESTION_MINUTES = 3.0
_NON_QUESTION_MINUTES = 1.0


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise QuestionPlanningError(f"Could not read {path.name}: {error}") from error
    except json.JSONDecodeError as error:
        raise QuestionPlanningError(
            f"{path.name} is not valid JSON: {error}"
        ) from error


def _load_questions(session_dir: Path) -> tuple[ScannedQuestion, ...]:
    raw = _read_json(session_dir / "questions.json")
    if not isinstance(raw, list) or not raw:
        raise QuestionPlanningError("questions.json must contain at least one object")
    questions: list[ScannedQuestion] = []
    for position, value in enumerate(raw, start=1):
        if not isinstance(value, dict):
            raise QuestionPlanningError(
                f"questions.json item {position} must be an object"
            )
        question = cast(dict[str, object], value)
        question_id = question.get("id")
        if not isinstance(question_id, int) or isinstance(question_id, bool):
            raise QuestionPlanningError(
                f"questions.json item {position} has an invalid numeric id"
            )
        text = question.get("question_text")
        ideal_answer = question.get("ideal_answer", "")
        has_code_editor = question.get("has_code_editor", False)
        if not isinstance(text, str) or not isinstance(ideal_answer, str):
            raise QuestionPlanningError(
                f"questions.json item {position} text fields must be strings"
            )
        if not isinstance(has_code_editor, bool):
            raise QuestionPlanningError(
                f"questions.json item {position}.has_code_editor must be boolean"
            )
        questions.append(
            ScannedQuestion(
                id=question_id,
                question_text=text.strip(),
                ideal_answer=ideal_answer.strip(),
                has_code_editor=has_code_editor,
            )
        )
    if len({question.id for question in questions}) != len(questions):
        raise QuestionPlanningError("questions.json contains duplicate question IDs")
    return tuple(questions)


def _load_skills(session_dir: Path) -> tuple[SkillParameter, ...]:
    raw = _read_json(session_dir / "skill_parameters.json")
    try:
        artifact = SkillParametersArtifact.model_validate(raw)
    except ValueError as error:
        raise QuestionPlanningError(
            f"skill_parameters.json is invalid: {error}"
        ) from error
    if not artifact.parameters:
        raise QuestionPlanningError(
            "skill_parameters.json contains no skill parameters"
        )
    return tuple(artifact.parameters)


def _tokens(value: str) -> frozenset[str]:
    return frozenset(
        word
        for word in _WORD.findall(value.casefold())
        if word not in _SKILL_STOP_WORDS
    )


def _classify(question: ScannedQuestion) -> QuestionContentType:
    if not question.question_text or len(_tokens(question.question_text)) < 2:
        return QuestionContentType.MALFORMED
    if question.has_code_editor:
        return QuestionContentType.CODING_QUESTION
    normalized = " ".join(question.question_text.casefold().split())
    if "?" not in normalized and any(
        marker in normalized for marker in _INSTRUCTION_MARKERS
    ):
        return QuestionContentType.INSTRUCTION
    return QuestionContentType.INTERVIEW_QUESTION


def _map_skills(
    question: ScannedQuestion, skills: tuple[SkillParameter, ...]
) -> tuple[tuple[str, ...], float, tuple[str, ...]]:
    corpus = f"{question.question_text} {question.ideal_answer}"
    corpus_tokens = _tokens(corpus)
    normalized_corpus = " ".join(_WORD.findall(corpus.casefold()))
    matches: list[tuple[str, float, str]] = []
    for skill in skills:
        skill_tokens = _tokens(skill.name)
        if not skill_tokens:
            continue
        normalized_skill = " ".join(_WORD.findall(skill.name.casefold()))
        if normalized_skill and normalized_skill in normalized_corpus:
            confidence = 1.0
            evidence = f"{skill.id}: exact skill-name phrase match"
        else:
            matched_terms = sorted(skill_tokens & corpus_tokens)
            overlap = len(matched_terms) / len(skill_tokens)
            if overlap < 0.5:
                continue
            confidence = round(overlap, 2)
            evidence = f"{skill.id}: matched terms {', '.join(matched_terms)}"
        matches.append((skill.id, confidence, evidence))
    matches.sort(key=lambda match: (-match[1], match[0]))
    return (
        tuple(skill_id for skill_id, _, _ in matches),
        (matches[0][1] if matches else 0.0),
        tuple(evidence for _, _, evidence in matches),
    )


def _candidate_for(
    question: ScannedQuestion, skills: tuple[SkillParameter, ...]
) -> _Candidate:
    content_type = _classify(question)
    target_skill_ids, confidence, mapping_evidence = _map_skills(question, skills)
    mandatory_ids = {
        skill.id for skill in skills if skill.requirement.casefold() == "mandatory"
    }
    mandatory_coverage = tuple(
        skill_id for skill_id in target_skill_ids if skill_id in mandatory_ids
    )
    estimated_minutes = (
        _CODING_MINUTES
        if content_type is QuestionContentType.CODING_QUESTION
        else (
            _QUESTION_MINUTES
            if content_type is QuestionContentType.INTERVIEW_QUESTION
            else _NON_QUESTION_MINUTES
        )
    )
    priority = (
        len(mandatory_coverage) * 100
        + len(target_skill_ids) * 10
        + (5 if content_type is QuestionContentType.CODING_QUESTION else 0)
    )
    return _Candidate(
        question=question,
        content_type=content_type,
        target_skill_ids=target_skill_ids,
        mandatory_skill_coverage=mandatory_coverage,
        mapping_confidence=confidence,
        mapping_evidence=mapping_evidence,
        estimated_minutes=estimated_minutes,
        priority=priority,
    )


def _skip_reason_for_content(content_type: QuestionContentType) -> str:
    if content_type is QuestionContentType.INSTRUCTION:
        return "excluded: setup/instruction card is not a candidate question"
    if content_type is QuestionContentType.MALFORMED:
        return "excluded: malformed or empty candidate question"
    raise AssertionError("selected question types do not have a content skip reason")


def _rank_candidates(
    candidates: tuple[_Candidate, ...], *, minutes: int
) -> dict[int, str | None]:
    """Return selected IDs as ``None`` and every skipped source card with a reason."""

    remaining_minutes = max(0.0, float(minutes) - _RESERVED_MINUTES)
    mandatory_ids = {
        skill_id
        for candidate in candidates
        for skill_id in candidate.mandatory_skill_coverage
    }
    covered_skill_ids: set[str] = set()
    selected_skill_sets: list[set[str]] = []
    decisions: dict[int, str | None] = {}
    eligible = [
        candidate
        for candidate in candidates
        if candidate.content_type
        in (QuestionContentType.INTERVIEW_QUESTION, QuestionContentType.CODING_QUESTION)
    ]
    for candidate in candidates:
        if candidate not in eligible:
            decisions[candidate.question.id] = _skip_reason_for_content(
                candidate.content_type
            )

    pending = list(eligible)
    while pending:
        pending.sort(
            key=lambda candidate: (
                -len(set(candidate.mandatory_skill_coverage) - covered_skill_ids),
                -int(candidate.content_type is QuestionContentType.CODING_QUESTION),
                -candidate.priority,
                candidate.question.id,
            )
        )
        candidate = pending.pop(0)
        question_id = candidate.question.id
        target_ids = set(candidate.target_skill_ids)
        if (
            not target_ids
            and candidate.content_type is not QuestionContentType.CODING_QUESTION
        ):
            decisions[question_id] = "not selected: no deterministic skill mapping"
            continue
        if (
            target_ids
            and target_ids <= covered_skill_ids
            and candidate.content_type is not QuestionContentType.CODING_QUESTION
        ):
            decisions[question_id] = "not selected: redundant skill coverage"
            continue
        if (
            target_ids
            and candidate.content_type is not QuestionContentType.CODING_QUESTION
            and not (set(candidate.mandatory_skill_coverage) - covered_skill_ids)
            and any(
                len(target_ids & selected_ids) / min(len(target_ids), len(selected_ids))
                >= 0.8
                for selected_ids in selected_skill_sets
                if selected_ids
            )
        ):
            decisions[question_id] = "not selected: near-identical skill coverage"
            continue
        if candidate.estimated_minutes > remaining_minutes:
            decisions[question_id] = (
                "not selected: time budget reserved for introduction, follow-up, "
                "candidate questions, and closing"
            )
            continue
        decisions[question_id] = None
        remaining_minutes -= candidate.estimated_minutes
        covered_skill_ids.update(target_ids)
        selected_skill_sets.append(target_ids)

    # Every mandatory skill that can be mapped is selected before any redundant card.
    assert mandatory_ids <= covered_skill_ids or not any(
        set(candidate.mandatory_skill_coverage) - covered_skill_ids
        for candidate in eligible
        if decisions.get(candidate.question.id) is None
    )
    return decisions


def _load_edits(
    path: Path,
) -> tuple[set[int], dict[int, str], tuple[int, ...], set[str]]:
    raw = _read_json(path)
    if not isinstance(raw, dict):
        raise QuestionPlanningError("plan edits must be a JSON object")
    edits = cast(dict[str, object], raw)
    if set(edits) - {"select", "skip", "order", "out_of_scope_skill_ids"}:
        raise QuestionPlanningError(
            "plan edits allow only select, skip, order, and out_of_scope_skill_ids"
        )
    raw_select = edits.get("select", [])
    raw_skip = edits.get("skip", {})
    raw_order = edits.get("order", [])
    raw_out_of_scope = edits.get("out_of_scope_skill_ids", [])
    if not isinstance(raw_select, list) or not isinstance(raw_order, list):
        raise QuestionPlanningError("plan edit select and order must be arrays")
    if not isinstance(raw_out_of_scope, list) or any(
        not isinstance(skill_id, str) or not skill_id.strip()
        for skill_id in raw_out_of_scope
    ):
        raise QuestionPlanningError("out_of_scope_skill_ids must be skill ID strings")
    if not isinstance(raw_skip, dict):
        raise QuestionPlanningError("plan edit skip must be an object")
    if any(
        not isinstance(question_id, int) or isinstance(question_id, bool)
        for question_id in [*raw_select, *raw_order]
    ):
        raise QuestionPlanningError("plan edit IDs must be integers")
    select = {cast(int, question_id) for question_id in raw_select}
    order = tuple(cast(int, question_id) for question_id in raw_order)
    if len(order) != len(set(order)):
        raise QuestionPlanningError("plan edit order contains duplicate question IDs")
    skip: dict[int, str] = {}
    for raw_id, reason in cast(dict[str, object], raw_skip).items():
        try:
            question_id = int(raw_id)
        except (TypeError, ValueError) as error:
            raise QuestionPlanningError(
                "plan edit skip IDs must be integers"
            ) from error
        if not isinstance(reason, str) or not reason.strip():
            raise QuestionPlanningError(
                "plan edit skip reasons must be non-empty strings"
            )
        skip[question_id] = reason.strip()
    if select & set(skip):
        raise QuestionPlanningError("a plan edit cannot select and skip one question")
    return select, skip, order, {str(skill_id) for skill_id in raw_out_of_scope}


def _apply_edits(
    candidates: tuple[_Candidate, ...],
    decisions: dict[int, str | None],
    edits_path: Path | None,
    valid_skill_ids: set[str],
) -> tuple[dict[int, str | None], tuple[int, ...]]:
    selected = [
        candidate.question.id
        for candidate in candidates
        if decisions[candidate.question.id] is None
    ]
    if edits_path is None:
        return decisions, tuple(selected)
    select, skip, order, out_of_scope_skill_ids = _load_edits(edits_path)
    candidate_by_id = {candidate.question.id: candidate for candidate in candidates}
    unknown_ids = (select | set(skip) | set(order)) - set(candidate_by_id)
    if unknown_ids:
        raise QuestionPlanningError(
            f"plan edits reference unknown IDs: {sorted(unknown_ids)}"
        )
    unknown_skill_ids = out_of_scope_skill_ids - valid_skill_ids
    if unknown_skill_ids:
        raise QuestionPlanningError(
            f"plan edits reference unknown skill IDs: {sorted(unknown_skill_ids)}"
        )
    for question_id, reason in skip.items():
        decisions[question_id] = reason
    scoped_out_coding_ids = {
        candidate.question.id
        for candidate in candidates
        if candidate.content_type is QuestionContentType.CODING_QUESTION
        and set(candidate.target_skill_ids) & out_of_scope_skill_ids
    }
    if select & scoped_out_coding_ids:
        raise QuestionPlanningError(
            "operator edits cannot select coding questions whose skill is out of scope"
        )
    for question_id in scoped_out_coding_ids:
        decisions[question_id] = "not selected: coding skill explicitly out of scope"
    for question_id in select:
        candidate = candidate_by_id[question_id]
        if candidate.content_type not in (
            QuestionContentType.INTERVIEW_QUESTION,
            QuestionContentType.CODING_QUESTION,
        ):
            raise QuestionPlanningError(
                "operator edits cannot select an instruction or malformed card"
            )
        decisions[question_id] = None
    selected = [
        candidate.question.id
        for candidate in candidates
        if decisions[candidate.question.id] is None
    ]
    if not set(order) <= set(selected):
        raise QuestionPlanningError(
            "plan edit order must contain selected question IDs"
        )
    ordered = list(order) + [
        question_id for question_id in selected if question_id not in order
    ]
    return decisions, tuple(ordered)


def _render_markdown(
    artifact: QuestionPlanArtifact, selected_ids: tuple[int, ...]
) -> str:
    by_id = {item.question_id: item for item in artifact.items}
    lines = ["# Offline interview plan", "", "## Selected questions", ""]
    for position, question_id in enumerate(selected_ids, start=1):
        item = by_id[question_id]
        skills = ", ".join(item.target_skill_ids) or "no deterministic mapping"
        lines.append(
            f"{position}. Question {question_id} — {item.content_type}; "
            f"skills: {skills}; estimate: {item.estimated_minutes:g} min"
        )
    if not selected_ids:
        lines.append("No questions fit the available time after reserved time.")
    lines.extend(("", "## Skipped source cards", ""))
    for item in artifact.items:
        if not item.selected:
            lines.append(f"- Question {item.question_id}: {item.skip_reason}")
    lines.extend(
        (
            "",
            "This is an offline draft only. Operator edits must be recorded before a live call.",
            "",
        )
    )
    return "\n".join(lines)


def _write_owner_only(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.touch(mode=0o600, exist_ok=True)
    temporary.chmod(0o600)
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
    path.chmod(0o600)


def build_interview_plan(
    session_dir: Path, *, minutes: int, edits_path: Path | None = None
) -> QuestionPlanResult:
    """Build a deterministic offline plan without browser, LLM, or audio effects."""

    if minutes <= 0:
        raise QuestionPlanningError("minutes must be greater than zero")
    session = session_dir.resolve()
    questions = _load_questions(session)
    skills = _load_skills(session)
    candidates = tuple(_candidate_for(question, skills) for question in questions)
    decisions = _rank_candidates(candidates, minutes=minutes)
    decisions, selected_ids = _apply_edits(
        candidates,
        decisions,
        edits_path,
        {skill.id for skill in skills},
    )
    selected_position = {
        question_id: index for index, question_id in enumerate(selected_ids)
    }
    ordered_candidates = tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                0 if candidate.question.id in selected_position else 1,
                selected_position.get(candidate.question.id, candidate.question.id),
                candidate.question.id,
            ),
        )
    )
    artifact = QuestionPlanArtifact(
        items=[
            QuestionPlanItem(
                question_id=candidate.question.id,
                content_type=candidate.content_type,
                target_skill_ids=list(candidate.target_skill_ids),
                mandatory_skill_coverage=list(candidate.mandatory_skill_coverage),
                estimated_minutes=candidate.estimated_minutes,
                priority=candidate.priority,
                selected=decisions[candidate.question.id] is None,
                skip_reason=decisions[candidate.question.id],
                mapping_source=QuestionMappingSource.DETERMINISTIC,
                mapping_confidence=candidate.mapping_confidence,
                mapping_evidence=list(candidate.mapping_evidence),
            )
            for candidate in ordered_candidates
        ]
    )
    plan_path = session / "interview_plan.json"
    markdown_path = session / "interview_plan.md"
    _write_owner_only(
        plan_path,
        artifact.model_dump_json(indent=2) + "\n",
    )
    _write_owner_only(markdown_path, _render_markdown(artifact, selected_ids))
    return QuestionPlanResult(
        artifact=artifact,
        plan_path=plan_path,
        markdown_path=markdown_path,
        selected_question_ids=selected_ids,
    )
