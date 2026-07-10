"""Typed configuration loaded from defaults, ``.env``, and the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


DEFAULT_TRANSCRIBER_PATH = Path("../Meeting_transcriber_with_LLM")


DEFAULTS: dict[str, str] = {
    "LLM_PROVIDER_MODE": "auto",
    "LLM_PRIMARY_PROVIDER": "lmstudio",
    "LLM_FALLBACK_PROVIDER": "openrouter",
    "LLM_FAST_TIMEOUT_SECONDS": "8",
    "LLM_DEEP_TIMEOUT_SECONDS": "20",
    "LLM_FALLBACK_CONFIDENCE_THRESHOLD": "0.65",
    "LLM_ALLOW_CLOUD_CANDIDATE_DATA": "false",
    "LMSTUDIO_BASE_URL": "http://127.0.0.1:1234/v1",
    "LMSTUDIO_API_KEY": "lm-studio",
    "LMSTUDIO_FAST_MODEL": "ornith-1.0-35b",
    "LMSTUDIO_DEEP_MODEL": "ornith-1.0-35b",
    "OPENROUTER_API_KEY": "",
    "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
    "OPENROUTER_FAST_MODEL": "google/gemini-2.5-flash-lite",
    "OPENROUTER_DEEP_MODEL": "google/gemini-2.5-flash",
    "OPENROUTER_SITE_URL": "http://localhost",
    "OPENROUTER_APP_NAME": "FloCareer Interview Copilot",
    "MEETING_TRANSCRIBER_PATH": str(DEFAULT_TRANSCRIBER_PATH),
    "TRANSCRIPTION_BACKEND": "apple-speech",
    "TRANSCRIBE_SYSTEM_AUDIO": "true",
    "TRANSCRIBE_MICROPHONE": "false",
    "BROWSER_HEADLESS": "false",
    "BROWSER_USER_DATA_DIR": ".browser-profile",
    "FLOCAREER_URL": "https://app.flocareer.com/",
    "SUPERTONIC_BASE_URL": "http://127.0.0.1:7788",
    "SUPERTONIC_VOICE": "interviewer_voice",
    "RUNS_DIR": "runs",
    "DEFAULT_INTERVIEW_MINUTES": "25",
    "REQUIRE_APPROVAL_BEFORE_FINISH": "true",
}


def _parse_bool(name: str, value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false, got {value!r}")


def _parse_float(name: str, value: str) -> float:
    try:
        return float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be numeric, got {value!r}") from error


def _parse_int(name: str, value: str) -> int:
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {value!r}") from error


def _resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (project_root / path).resolve()


def _read_dotenv(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}

    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"Invalid .env line {line_number}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value
    return values


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings with paths resolved relative to the project root."""

    project_root: Path
    llm_provider_mode: str
    llm_primary_provider: str
    llm_fallback_provider: str
    llm_fast_timeout_seconds: float
    llm_deep_timeout_seconds: float
    llm_fallback_confidence_threshold: float
    llm_allow_cloud_candidate_data: bool
    lmstudio_base_url: str
    lmstudio_api_key: str
    lmstudio_fast_model: str
    lmstudio_deep_model: str
    openrouter_api_key: str
    openrouter_base_url: str
    openrouter_fast_model: str
    openrouter_deep_model: str
    openrouter_site_url: str
    openrouter_app_name: str
    meeting_transcriber_path: Path
    transcription_backend: str
    transcribe_system_audio: bool
    transcribe_microphone: bool
    browser_headless: bool
    browser_user_data_dir: Path
    flocareer_url: str
    supertonic_base_url: str
    supertonic_voice: str
    runs_dir: Path
    default_interview_minutes: int
    require_approval_before_finish: bool

    @classmethod
    def load(
        cls,
        *,
        project_root: Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> Settings:
        root = (project_root or Path.cwd()).resolve()
        values = DEFAULTS.copy()
        values.update(_read_dotenv(root / ".env"))
        values.update(dict(os.environ if environ is None else environ))

        return cls(
            project_root=root,
            llm_provider_mode=values["LLM_PROVIDER_MODE"],
            llm_primary_provider=values["LLM_PRIMARY_PROVIDER"],
            llm_fallback_provider=values["LLM_FALLBACK_PROVIDER"],
            llm_fast_timeout_seconds=_parse_float(
                "LLM_FAST_TIMEOUT_SECONDS", values["LLM_FAST_TIMEOUT_SECONDS"]
            ),
            llm_deep_timeout_seconds=_parse_float(
                "LLM_DEEP_TIMEOUT_SECONDS", values["LLM_DEEP_TIMEOUT_SECONDS"]
            ),
            llm_fallback_confidence_threshold=_parse_float(
                "LLM_FALLBACK_CONFIDENCE_THRESHOLD",
                values["LLM_FALLBACK_CONFIDENCE_THRESHOLD"],
            ),
            llm_allow_cloud_candidate_data=_parse_bool(
                "LLM_ALLOW_CLOUD_CANDIDATE_DATA",
                values["LLM_ALLOW_CLOUD_CANDIDATE_DATA"],
            ),
            lmstudio_base_url=values["LMSTUDIO_BASE_URL"].rstrip("/"),
            lmstudio_api_key=values["LMSTUDIO_API_KEY"],
            lmstudio_fast_model=values["LMSTUDIO_FAST_MODEL"],
            lmstudio_deep_model=values["LMSTUDIO_DEEP_MODEL"],
            openrouter_api_key=values["OPENROUTER_API_KEY"],
            openrouter_base_url=values["OPENROUTER_BASE_URL"].rstrip("/"),
            openrouter_fast_model=values["OPENROUTER_FAST_MODEL"],
            openrouter_deep_model=values["OPENROUTER_DEEP_MODEL"],
            openrouter_site_url=values["OPENROUTER_SITE_URL"],
            openrouter_app_name=values["OPENROUTER_APP_NAME"],
            meeting_transcriber_path=_resolve_path(
                root, values["MEETING_TRANSCRIBER_PATH"]
            ),
            transcription_backend=values["TRANSCRIPTION_BACKEND"],
            transcribe_system_audio=_parse_bool(
                "TRANSCRIBE_SYSTEM_AUDIO", values["TRANSCRIBE_SYSTEM_AUDIO"]
            ),
            transcribe_microphone=_parse_bool(
                "TRANSCRIBE_MICROPHONE", values["TRANSCRIBE_MICROPHONE"]
            ),
            browser_headless=_parse_bool(
                "BROWSER_HEADLESS", values["BROWSER_HEADLESS"]
            ),
            browser_user_data_dir=_resolve_path(root, values["BROWSER_USER_DATA_DIR"]),
            flocareer_url=values["FLOCAREER_URL"],
            supertonic_base_url=values["SUPERTONIC_BASE_URL"].rstrip("/"),
            supertonic_voice=values["SUPERTONIC_VOICE"],
            runs_dir=_resolve_path(root, values["RUNS_DIR"]),
            default_interview_minutes=_parse_int(
                "DEFAULT_INTERVIEW_MINUTES", values["DEFAULT_INTERVIEW_MINUTES"]
            ),
            require_approval_before_finish=_parse_bool(
                "REQUIRE_APPROVAL_BEFORE_FINISH",
                values["REQUIRE_APPROVAL_BEFORE_FINISH"],
            ),
        )

    def missing_required(self) -> list[str]:
        required = {
            "LLM_PRIMARY_PROVIDER": self.llm_primary_provider,
            "LMSTUDIO_BASE_URL": self.lmstudio_base_url,
            "LMSTUDIO_FAST_MODEL": self.lmstudio_fast_model,
            "MEETING_TRANSCRIBER_PATH": str(self.meeting_transcriber_path),
            "FLOCAREER_URL": self.flocareer_url,
        }
        return [name for name, value in required.items() if not value.strip()]

    def safe_dump(self) -> str:
        """Return an operator-readable summary without rendering secret values."""

        openrouter_key = "configured" if self.openrouter_api_key else "not configured"
        lmstudio_key = "configured" if self.lmstudio_api_key else "not configured"
        return "\n".join(
            (
                f"LLM provider mode: {self.llm_provider_mode}",
                f"LLM primary provider: {self.llm_primary_provider}",
                f"LLM fallback provider: {self.llm_fallback_provider}",
                f"LM Studio URL: {self.lmstudio_base_url}",
                f"LM Studio API key: {lmstudio_key}",
                f"LM Studio fast model: {self.lmstudio_fast_model}",
                f"LM Studio deep model: {self.lmstudio_deep_model}",
                f"OpenRouter API key: {openrouter_key}",
                f"Cloud candidate data allowed: {self.llm_allow_cloud_candidate_data}",
                f"Meeting transcriber path: {self.meeting_transcriber_path}",
                f"Transcription backend: {self.transcription_backend}",
                f"System audio enabled: {self.transcribe_system_audio}",
                f"Microphone enabled: {self.transcribe_microphone}",
                f"FloCareer URL: {self.flocareer_url}",
                f"Runs directory: {self.runs_dir}",
                f"Approval required before finish: {self.require_approval_before_finish}",
            )
        )
