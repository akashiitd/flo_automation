"""Append-only per-session provider usage and cost records."""

from __future__ import annotations

import json
from pathlib import Path

from llm.schemas import ProviderMetadata


class UsageTracker:
    def __init__(self, path: Path) -> None:
        self.path = path

    def record(self, metadata: ProviderMetadata) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = metadata.model_dump(mode="json")
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            output.write("\n")
