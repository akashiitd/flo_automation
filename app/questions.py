"""Shared immutable interview-question contract."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InterviewQuestion:
    id: int
    question_text: str
    ideal_answer: str
