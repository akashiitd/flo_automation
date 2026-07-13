"""Session-scoped screenshot persistence."""

from __future__ import annotations

import re
from pathlib import Path

from playwright.sync_api import Page


def save_screenshot(page: Page, screenshots_dir: Path, name: str) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    if not safe_name:
        raise ValueError("screenshot name must contain a letter or number")
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    path = screenshots_dir / f"{safe_name}.png"
    page.screenshot(path=str(path), full_page=True)
    path.chmod(0o600)
    return path
