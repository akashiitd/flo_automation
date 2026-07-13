from __future__ import annotations

import stat
from pathlib import Path

from playwright.sync_api import sync_playwright

from browser.screenshots import save_screenshot


def test_screenshot_artifacts_are_owner_only(tmp_path: Path) -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_content("<h1>Private interview artifact</h1>")

        screenshot = save_screenshot(page, tmp_path, "private")

        browser.close()

    assert stat.S_IMODE(screenshot.stat().st_mode) == 0o600
