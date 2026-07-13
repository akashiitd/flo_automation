"""Read-only FloCareer skill-parameter extraction contracts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExtractedSkillParameter:
    """One rating parameter rendered in FloCareer's candidate skill panel."""

    id: str
    name: str
    requirement: str
    level: str
    rating_scale: int
    source: str = "flocareer_dom"
