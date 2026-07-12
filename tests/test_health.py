from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.health import CheckResult, HealthReport, ProbeResult, Status, run_health_checks


class ReadyProbes:
    def get_json(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 3,
    ) -> dict[str, object]:
        assert url == "http://lmstudio.test/v1/models"
        return {"data": [{"id": "google/gemma-4-12b"}]}

    def browser_launch(self) -> ProbeResult:
        return ProbeResult(True, "Chromium launched")

    def url_reachable(self, url: str, *, timeout: float = 2) -> ProbeResult:
        assert url == "http://supertonic.test"
        return ProbeResult(False, "connection refused")


def test_health_is_ready_when_required_services_pass(tmp_path: Path) -> None:
    transcriber = tmp_path / "Meeting_transcriber_with_LLM"
    source = transcriber / "src"
    source.mkdir(parents=True)
    (source / "apple_speech_transcriber.py").touch()

    settings = Settings.load(
        project_root=tmp_path,
        environ={
            "MEETING_TRANSCRIBER_PATH": str(transcriber),
            "RUNS_DIR": str(tmp_path / "runs"),
            "LMSTUDIO_BASE_URL": "http://lmstudio.test/v1",
            "SUPERTONIC_BASE_URL": "http://supertonic.test",
            "LLM_ALLOW_CLOUD_CANDIDATE_DATA": "false",
        },
    )

    report = run_health_checks(settings, probes=ReadyProbes())

    assert report.overall == "READY_FOR_BROWSER_SCAN"
    assert report.status_for("LM Studio reachable") is Status.OK
    assert report.status_for("Local models") is Status.OK
    assert report.status_for("OpenRouter fallback") is Status.WARN
    assert report.status_for("Supertonic") is Status.WARN
    assert (tmp_path / "runs").is_dir()


def test_health_is_not_ready_when_a_required_service_fails(tmp_path: Path) -> None:
    settings = Settings.load(
        project_root=tmp_path,
        environ={
            "MEETING_TRANSCRIBER_PATH": str(tmp_path / "missing"),
            "RUNS_DIR": str(tmp_path / "runs"),
            "LMSTUDIO_BASE_URL": "http://lmstudio.test/v1",
            "SUPERTONIC_BASE_URL": "http://supertonic.test",
        },
    )

    report = run_health_checks(settings, probes=ReadyProbes())

    assert report.overall == "NOT_READY"
    assert report.status_for("Meeting transcriber path") is Status.FAIL
    assert report.status_for("Apple Speech adapter") is Status.FAIL


def test_browser_readiness_does_not_require_local_model_availability() -> None:
    report = HealthReport(
        (
            CheckResult("Runs directory writable", Status.OK, "runs"),
            CheckResult("LM Studio reachable", Status.FAIL, "offline"),
            CheckResult("Local models", Status.FAIL, "offline"),
            CheckResult("Playwright browser launch", Status.OK, "ready"),
        )
    )

    assert report.overall == "NOT_READY"
    assert report.browser_ready is True
