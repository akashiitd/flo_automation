"""Local readiness checks for the supervised interview copilot."""

from __future__ import annotations

import json
import platform
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

from app.config import Settings


class Status(str, Enum):
    OK = "OK"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    status: Status
    detail: str


@dataclass(frozen=True, slots=True)
class ProbeResult:
    available: bool
    detail: str


@dataclass(frozen=True, slots=True)
class HealthReport:
    checks: tuple[CheckResult, ...]

    @property
    def overall(self) -> str:
        if any(check.status is Status.FAIL for check in self.checks):
            return "NOT_READY"
        return "READY_FOR_BROWSER_SCAN"

    def status_for(self, name: str) -> Status:
        for check in self.checks:
            if check.name == name:
                return check.status
        raise KeyError(name)

    def render(self) -> str:
        lines = ["Health check"]
        lines.extend(
            f"[{check.status.value}] {check.name}: {check.detail}"
            for check in self.checks
        )
        lines.extend(("", f"Overall: {self.overall}"))
        return "\n".join(lines)


class HealthProbes(Protocol):
    def get_json(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 3,
    ) -> dict[str, object]: ...

    def browser_launch(self) -> ProbeResult: ...

    def url_reachable(self, url: str, *, timeout: float = 2) -> ProbeResult: ...


class LocalHealthProbes:
    """Real network and browser probes used by the CLI."""

    def get_json(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 3,
    ) -> dict[str, object]:
        request = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected a JSON object from {url}")
        return payload

    def browser_launch(self) -> ProbeResult:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            return ProbeResult(
                False,
                "Playwright is not installed; run 'uv sync'",
            )

        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                browser.close()
        except Exception as error:  # Playwright exposes several launch exceptions.
            return ProbeResult(
                False,
                f"browser launch failed ({error}); run 'uv run playwright install chromium'",
            )
        return ProbeResult(True, "Chromium launched and closed successfully")

    def url_reachable(self, url: str, *, timeout: float = 2) -> ProbeResult:
        try:
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return ProbeResult(True, f"HTTP {response.status}")
        except urllib.error.HTTPError as error:
            return ProbeResult(True, f"HTTP {error.code}")
        except (OSError, urllib.error.URLError) as error:
            return ProbeResult(
                False, str(error.reason if hasattr(error, "reason") else error)
            )


def _path_check(name: str, path: Path, *, directory: bool = False) -> CheckResult:
    exists = path.is_dir() if directory else path.is_file()
    kind = "directory" if directory else "file"
    if exists:
        return CheckResult(name, Status.OK, str(path))
    return CheckResult(name, Status.FAIL, f"{kind} not found: {path}")


def _runs_directory_check(path: Path) -> CheckResult:
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path, prefix=".health-", delete=True):
            pass
    except OSError as error:
        return CheckResult("Runs directory writable", Status.FAIL, str(error))
    return CheckResult("Runs directory writable", Status.OK, str(path))


def _model_ids(payload: dict[str, object]) -> list[str]:
    raw_models = payload.get("data", [])
    if not isinstance(raw_models, list):
        return []
    model_ids: list[str] = []
    for raw_model in raw_models:
        if isinstance(raw_model, dict):
            model_id = raw_model.get("id")
            if isinstance(model_id, str):
                model_ids.append(model_id)
    return model_ids


def run_health_checks(
    settings: Settings,
    *,
    probes: HealthProbes | None = None,
) -> HealthReport:
    """Run mandatory readiness checks and optional integration probes."""

    active_probes = probes or LocalHealthProbes()
    checks: list[CheckResult] = [
        CheckResult("Python", Status.OK, platform.python_version()),
        _runs_directory_check(settings.runs_dir),
        _path_check(
            "Meeting transcriber path",
            settings.meeting_transcriber_path,
            directory=True,
        ),
        _path_check(
            "Apple Speech adapter",
            settings.meeting_transcriber_path / "src" / "apple_speech_transcriber.py",
        ),
    ]

    try:
        models_payload = active_probes.get_json(
            f"{settings.lmstudio_base_url}/models",
            timeout=settings.llm_fast_timeout_seconds,
        )
    except Exception as error:
        checks.extend(
            (
                CheckResult("LM Studio reachable", Status.FAIL, str(error)),
                CheckResult("Local models", Status.FAIL, "LM Studio unavailable"),
            )
        )
    else:
        checks.append(
            CheckResult(
                "LM Studio reachable",
                Status.OK,
                settings.lmstudio_base_url,
            )
        )
        model_ids = _model_ids(models_payload)
        checks.append(
            CheckResult(
                "Local models",
                Status.OK if model_ids else Status.FAIL,
                ", ".join(model_ids) if model_ids else "no models reported",
            )
        )

    if settings.llm_fallback_provider.lower() != "openrouter":
        checks.append(
            CheckResult(
                "OpenRouter fallback",
                Status.WARN,
                "OpenRouter is not the selected fallback provider",
            )
        )
    elif not settings.llm_allow_cloud_candidate_data:
        checks.append(
            CheckResult(
                "OpenRouter fallback",
                Status.WARN,
                "disabled because cloud candidate data is not allowed",
            )
        )
    elif not settings.openrouter_api_key:
        checks.append(
            CheckResult(
                "OpenRouter fallback",
                Status.FAIL,
                "cloud fallback is enabled but OPENROUTER_API_KEY is missing",
            )
        )
    else:
        try:
            active_probes.get_json(
                f"{settings.openrouter_base_url}/models",
                headers={
                    "Authorization": f"Bearer {settings.openrouter_api_key}",
                    "HTTP-Referer": settings.openrouter_site_url,
                    "X-Title": settings.openrouter_app_name,
                },
                timeout=settings.llm_fast_timeout_seconds,
            )
        except Exception as error:
            checks.append(CheckResult("OpenRouter fallback", Status.FAIL, str(error)))
        else:
            checks.append(
                CheckResult("OpenRouter fallback", Status.OK, "API key accepted")
            )

    browser = active_probes.browser_launch()
    checks.append(
        CheckResult(
            "Playwright browser launch",
            Status.OK if browser.available else Status.FAIL,
            browser.detail,
        )
    )

    supertonic = active_probes.url_reachable(settings.supertonic_base_url)
    checks.append(
        CheckResult(
            "Supertonic",
            Status.OK if supertonic.available else Status.WARN,
            (
                f"available at {settings.supertonic_base_url} ({supertonic.detail})"
                if supertonic.available
                else f"not running ({supertonic.detail})"
            ),
        )
    )

    return HealthReport(tuple(checks))
