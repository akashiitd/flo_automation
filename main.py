"""Command-line entry point for the FloCareer interview copilot."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import cast

from app.config import Settings
from app.health import run_health_checks
from browser.action_guard import BrowserAction, approval_token_for
from browser.join_workflow import JoinWorkflowError
from browser.playwright_controller import (
    BrowserScanError,
    join_candidate_dry_run,
    join_candidate_live,
    scan_candidate_questions,
    scan_dashboard,
)
from evaluator.scoring import evaluate_answer
from llm.lmstudio_provider import LMStudioProvider
from llm.openrouter_provider import OpenRouterProvider
from llm.provider_router import HumanReviewRequired, ProviderRouter
from llm.schemas import (
    EvaluationInput,
    ModelClass,
    ProviderMetadata,
    QuestionEvaluation,
    StructuredGeneration,
)
from llm.usage_tracker import UsageTracker
from transcriber.apple_speech_adapter import AppleSpeechAdapter


PROJECT_ROOT = Path(__file__).resolve().parent


def _config_dump(settings: Settings) -> int:
    print("Configuration")
    print(settings.safe_dump())
    print()

    failed = False
    missing = settings.missing_required()
    if missing:
        failed = True
        print(f"[FAIL] Missing required config: {', '.join(missing)}")
    else:
        print("[OK] No missing required config")

    if settings.meeting_transcriber_path.is_dir():
        print(f"[OK] Transcriber path exists: {settings.meeting_transcriber_path}")
    else:
        failed = True
        print(f"[FAIL] Transcriber path missing: {settings.meeting_transcriber_path}")

    if settings.lmstudio_base_url:
        print(f"[OK] LM Studio URL configured: {settings.lmstudio_base_url}")

    if settings.openrouter_api_key:
        print("[OK] OpenRouter key configured")
    elif settings.llm_allow_cloud_candidate_data:
        failed = True
        print("[FAIL] OpenRouter cloud use is enabled but its key is unavailable")
    else:
        print(
            "[WARN] OpenRouter optional/unavailable; cloud candidate data is disabled"
        )

    try:
        settings.runs_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=settings.runs_dir, delete=True):
            pass
    except OSError as error:
        failed = True
        print(f"[FAIL] Runs directory is not writable: {error}")
    else:
        print(f"[OK] Runs directory writable: {settings.runs_dir}")

    print()
    print(f"Overall: {'INVALID' if failed else 'VALID'}")
    return 1 if failed else 0


def _sample_evaluation_input() -> EvaluationInput:
    return EvaluationInput(
        question_id=1,
        question="How would you build a LangChain microservice?",
        ideal_answer=(
            "Mentions an API layer, model and tool orchestration, tracing, "
            "timeouts, retries, validation, and deployment concerns."
        ),
        candidate_answer=(
            "I would create an API, call the model through LangChain, validate "
            "the request, and return the response."
        ),
    )


async def _llm_test(
    settings: Settings,
    *,
    provider_name: str,
    model_class: ModelClass,
) -> int:
    provider = (
        LMStudioProvider(settings)
        if provider_name == "lmstudio"
        else OpenRouterProvider(settings)
    )
    usage_tracker = UsageTracker(settings.runs_dir / "llm_tests" / "llm_usage.jsonl")
    try:
        generation = await evaluate_answer(
            _sample_evaluation_input(),
            provider,
            model_class=model_class,
            usage_tracker=usage_tracker,
        )
    except Exception as error:
        detail = str(error) or type(error).__name__
        print(f"LLM test failed for {provider_name}: {detail}", file=sys.stderr)
        return 1
    finally:
        await provider.aclose()

    print(json.dumps(generation.model_dump(mode="json"), indent=2))
    return 0


class _TimeoutProvider:
    name = "lmstudio"

    async def generate_structured(
        self, *args: object, **kwargs: object
    ) -> dict[str, object]:
        raise TimeoutError("simulated local timeout")


class _FallbackProbeProvider:
    name = "openrouter"

    def __init__(self) -> None:
        self.calls = 0

    async def generate_structured(
        self, *args: object, **kwargs: object
    ) -> dict[str, object]:
        self.calls += 1
        return StructuredGeneration(
            output={
                "question_id": 1,
                "score": 3,
                "rating_label": "Average",
                "evidence": ["Candidate mentioned an API layer"],
                "follow_up": "How would you handle retries?",
                "feedback": "Basic understanding with production gaps.",
                "confidence": 0.8,
            },
            metadata=ProviderMetadata(
                provider="openrouter",
                model="simulated-openrouter-model",
                request_purpose="feedback_draft",
                latency_ms=1,
                input_tokens=1,
                output_tokens=1,
                estimated_cost_usd=0,
                pii_redaction_ran=True,
            ),
        ).model_dump(mode="json")


async def _llm_failover_test(settings: Settings) -> int:
    request = _sample_evaluation_input()

    blocked_fallback = _FallbackProbeProvider()
    blocked_router = ProviderRouter(
        replace(
            settings,
            llm_allow_cloud_candidate_data=False,
            openrouter_api_key="",
        ),
        primary=_TimeoutProvider(),
        fallback=blocked_fallback,
    )
    try:
        await evaluate_answer(request, blocked_router)
    except HumanReviewRequired:
        if blocked_fallback.calls != 0:
            print("[FAIL] Cloud provider was called while cloud use was disabled")
            return 1
        print("[OK] Cloud-disabled timeout stopped for human review")
    else:
        print("[FAIL] Cloud-disabled timeout did not stop for human review")
        return 1

    allowed_fallback = _FallbackProbeProvider()
    allowed_router = ProviderRouter(
        replace(
            settings,
            llm_allow_cloud_candidate_data=True,
            openrouter_api_key="simulation-only",
        ),
        primary=_TimeoutProvider(),
        fallback=allowed_fallback,
    )
    generation = await evaluate_answer(request, allowed_router)
    QuestionEvaluation.model_validate(generation.output)
    if allowed_fallback.calls != 1 or not generation.metadata.fallback_used:
        print("[FAIL] Allowed timeout did not route exactly once to the fallback")
        return 1
    print("[OK] Cloud-enabled timeout routed to the fallback")
    print("[OK] Fallback preserved the evaluator schema and metadata")
    print("Overall: PASS")
    return 0


def _positive_seconds(value: str) -> float:
    seconds = float(value)
    if seconds <= 0:
        raise argparse.ArgumentTypeError("seconds must be greater than zero")
    return seconds


def _listen_test(settings: Settings, *, seconds: float) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    session_id = f"listen_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    adapter: AppleSpeechAdapter | None = None

    def show_segment(segment: object) -> None:
        source = getattr(segment, "source", "unknown")
        speaker = getattr(segment, "speaker", "Other") or "Other"
        text = getattr(segment, "text", "")
        print(f"[{source}] [{speaker}] {text}", flush=True)

    try:
        adapter = AppleSpeechAdapter(
            settings,
            session_id=session_id,
            on_segment=show_segment,
        )
        print("Starting Apple Speech listener")
        print("Mode: system audio only (microphone disabled)")
        print(f"Duration: {seconds:g} seconds")
        if not adapter.start():
            adapter.stop()
            print(
                "Apple Speech failed to start. Check Speech Recognition and "
                "Screen & System Audio Recording permissions.",
                file=sys.stderr,
            )
            return 1
    except Exception as error:
        if adapter is not None:
            adapter.stop()
        print(f"Apple Speech failed to start: {error}", file=sys.stderr)
        return 1

    interrupted = False
    try:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
    except KeyboardInterrupt:
        interrupted = True
        print("Stopping early at user request")
    finally:
        assert adapter is not None
        summary = adapter.stop()

    print(f"Captured segments: {summary.segment_count}")
    print(f"Saved transcript JSON: {summary.json_path}")
    print(f"Saved transcript text: {summary.text_path}")

    payload = json.loads(summary.json_path.read_text(encoding="utf-8"))
    segments = payload.get("segments", [])
    system_segments = [
        segment
        for segment in segments
        if isinstance(segment, dict) and segment.get("source") == "system"
    ]
    microphone_segments = [
        segment
        for segment in segments
        if isinstance(segment, dict) and segment.get("source") == "microphone"
    ]
    if microphone_segments:
        print("Validation failed: microphone segments were captured", file=sys.stderr)
        return 1
    if not system_segments:
        suffix = " after interruption" if interrupted else ""
        print(
            f"Validation failed: no system-audio transcript was captured{suffix}",
            file=sys.stderr,
        )
        return 1
    print("Validation passed: system audio captured and microphone remained off")
    return 0


def _browser_scan(settings: Settings, *, login_timeout_seconds: float) -> int:
    print("FloCareer browser scan")
    print("Safety mode: read-only; interview launch actions are disabled")
    try:
        result = scan_dashboard(
            settings,
            login_timeout_seconds=login_timeout_seconds,
            progress=lambda message: print(message, flush=True),
        )
    except BrowserScanError as error:
        print(f"Browser scan failed: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        detail = str(error) or type(error).__name__
        print(f"Browser scan failed: {detail}", file=sys.stderr)
        return 1

    print("FloCareer dashboard loaded")
    print(f"Found scheduled interviews: {len(result.interviews)}")
    for index, interview in enumerate(result.interviews, start=1):
        print(f"{index}. {interview.summary}")
    if not result.interviews:
        print("No scheduled interview rows were visible on the dashboard")
        print(f"Screenshot saved: {result.screenshot_path}")
        print(
            "Validation incomplete: browser access is working, but candidate-row "
            "extraction requires at least one visible scheduled interview."
        )
        return 2
    print(f"Screenshot saved: {result.screenshot_path}")
    print("Validation passed: dashboard scanned without launching an interview")
    return 0


def _join_dry_run(
    settings: Settings,
    *,
    candidate_name: str,
    login_timeout_seconds: float,
) -> int:
    print("FloCareer guarded join discovery")
    print("Safety mode: dry run; launch and Join actions are blocked")
    health = run_health_checks(settings)
    if health.overall != "READY_FOR_BROWSER_SCAN":
        print(health.render(), file=sys.stderr)
        print(
            "Join dry run failed: health prerequisites are not ready", file=sys.stderr
        )
        return 1
    try:
        result = join_candidate_dry_run(
            settings,
            candidate_name=candidate_name,
            login_timeout_seconds=login_timeout_seconds,
            progress=lambda message: print(message, flush=True),
        )
    except (BrowserScanError, JoinWorkflowError) as error:
        print(f"Join dry run failed: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        detail = str(error) or type(error).__name__
        print(f"Join dry run failed: {detail}", file=sys.stderr)
        return 1

    print(f"Candidate identifier: {result.candidate_identifier}")
    print(f"Candidate screenshot: {result.candidate_found_screenshot}")
    print(f"Dry-run screenshot: {result.join_dry_run_screenshot}")
    print(f"Action log: {result.action_log_path}")
    print("Validation passed: launch control found and blocked by dry run")
    return 0


def _join_live(
    settings: Settings,
    *,
    candidate_name: str,
    login_timeout_seconds: float,
) -> int:
    print("FloCareer approved live join")
    print("Safety mode: Launch and Join require separate approvals")
    print("Consent OK requires another approval when the form is shown")
    print("Hang-up and FINISH are always blocked")
    health = run_health_checks(settings)
    if health.overall != "READY_FOR_BROWSER_SCAN":
        print(health.render(), file=sys.stderr)
        print("Live join failed: health prerequisites are not ready", file=sys.stderr)
        return 1

    def request_approval(
        action: BrowserAction, candidate_identifier: str
    ) -> str | None:
        expected = approval_token_for(action, candidate_identifier)
        print()
        print(f"Approval required for {action.value}")
        print(f"Type exactly: {expected}")
        try:
            return input("Approval token: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nApproval cancelled", file=sys.stderr)
            return None

    def wait_for_manual_end(candidate_identifier: str) -> None:
        expected = f"CONFIRM-INTERVIEW-ENDED {candidate_identifier}"
        print()
        print("Interview joined. The automation will not click hang-up.")
        print("End the interview manually in FloCareer when appropriate.")
        print("The browser will remain open until you confirm it has ended.")
        while True:
            try:
                entered = input(
                    f"After it ends, type exactly: {expected}\nConfirmation: "
                )
            except (EOFError, KeyboardInterrupt):
                print(
                    "\nConfirmation still required; browser remains under the "
                    "live command's control.",
                    file=sys.stderr,
                )
                time.sleep(1)
                continue
            if entered.strip() == expected:
                return
            print("Confirmation did not match; browser remains open.")

    try:
        result = join_candidate_live(
            settings,
            candidate_name=candidate_name,
            login_timeout_seconds=login_timeout_seconds,
            progress=lambda message: print(message, flush=True),
            request_approval=request_approval,
            wait_for_manual_end=wait_for_manual_end,
        )
    except (BrowserScanError, JoinWorkflowError) as error:
        print(f"Live join failed: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        detail = str(error) or type(error).__name__
        print(f"Live join failed: {detail}", file=sys.stderr)
        return 1

    print(f"Candidate identifier: {result.candidate_identifier}")
    if result.consent_screenshot is not None:
        print(f"Consent screenshot: {result.consent_screenshot}")
    else:
        print("Consent form: not shown; FloCareer opened verified pre-call directly")
    print(f"Pre-call screenshot: {result.pre_call_screenshot}")
    print(f"Joined screenshot: {result.joined_screenshot}")
    print(f"Action log: {result.action_log_path}")
    print("Validation passed: interview joined after all required approvals")
    return 0


def _questions_scan(
    settings: Settings,
    *,
    candidate_name: str,
    login_timeout_seconds: float,
) -> int:
    print("FloCareer approved question scan")
    print("Safety mode: Launch requires approval; Join is never clicked")
    print("Question cards may be expanded; evaluation controls are untouched")
    print("Code editor DOM is inspected only and is never enabled for the candidate")
    health = run_health_checks(settings)
    if health.overall != "READY_FOR_BROWSER_SCAN":
        print(health.render(), file=sys.stderr)
        print(
            "Question scan failed: health prerequisites are not ready", file=sys.stderr
        )
        return 1

    def request_approval(
        action: BrowserAction, candidate_identifier: str
    ) -> str | None:
        expected = approval_token_for(action, candidate_identifier)
        print()
        print(f"Approval required for {action.value}")
        print(f"Type exactly: {expected}")
        try:
            return input("Approval token: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nApproval cancelled", file=sys.stderr)
            return None

    try:
        result = scan_candidate_questions(
            settings,
            candidate_name=candidate_name,
            request_approval=request_approval,
            login_timeout_seconds=login_timeout_seconds,
            progress=lambda message: print(message, flush=True),
        )
    except (BrowserScanError, JoinWorkflowError) as error:
        print(f"Question scan failed: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        detail = str(error) or type(error).__name__
        print(f"Question scan failed: {detail}", file=sys.stderr)
        return 1

    coding = [
        str(question.id) for question in result.questions if question.has_code_editor
    ]
    print(f"Extracted questions: {len(result.questions)}")
    print(f"Coding question IDs: {', '.join(coding) if coding else 'none detected'}")
    associations = ", ".join(
        (
            f"{observation.question_id}={observation.association_status}"
            if observation.question_id is not None
            else "unresolved=ambiguous"
        )
        for observation in result.code_editor_dom_observations
    )
    observation_ids = [
        observation.question_id
        for observation in result.code_editor_dom_observations
        if observation.question_id is not None
    ]
    capture_complete = len(observation_ids) == len(set(observation_ids)) and set(
        observation_ids
    ) == {int(question_id) for question_id in coding}
    print(f"Code editor DOM associations: {associations or 'none detected'}")
    print(
        f"Code editor DOM capture: {'complete' if capture_complete else 'incomplete'}"
    )
    print(f"Questions JSON: {result.questions_path}")
    print(f"Code editor DOM: {result.code_editor_dom_path}")
    print(f"Expanded screenshot: {result.screenshot_path}")
    print(f"Action log: {result.action_log_path}")
    print("Validation passed: questions read without clicking Join")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Supervised FloCareer interview automation copilot"
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser(
        "config-dump", help="validate and safely print configuration"
    )
    subcommands.add_parser("health", help="check local runtime readiness")
    llm_test = subcommands.add_parser(
        "llm-test", help="run the structured evaluator through one provider"
    )
    llm_test.add_argument(
        "--provider",
        choices=("lmstudio", "openrouter"),
        default="lmstudio",
    )
    llm_test.add_argument(
        "--model-class",
        choices=("fast", "deep"),
        default="fast",
    )
    subcommands.add_parser(
        "llm-failover-test",
        help="simulate guarded local-to-cloud failover without sending data",
    )
    listen_test = subcommands.add_parser(
        "listen-test",
        help="capture Apple Speech system audio and save a session transcript",
    )
    listen_test.add_argument("--seconds", type=_positive_seconds, default=60.0)
    browser_scan = subcommands.add_parser(
        "browser-scan",
        help="open and read the FloCareer dashboard without launching interviews",
    )
    browser_scan.add_argument(
        "--login-timeout",
        type=_positive_seconds,
        default=180.0,
        help="seconds to wait for manual login in the opened browser",
    )
    join = subcommands.add_parser(
        "join",
        help="find one candidate's launch control without launching an interview",
    )
    join.add_argument("--candidate", required=True, help="exact visible candidate name")
    join_mode = join.add_mutually_exclusive_group(required=True)
    join_mode.add_argument(
        "--dry-run",
        action="store_true",
        help="discover the launch control while blocking Launch and Join",
    )
    join_mode.add_argument(
        "--live",
        action="store_true",
        help="request approvals for Launch, optional Consent OK, and Join",
    )
    join.add_argument(
        "--login-timeout",
        type=_positive_seconds,
        default=180.0,
        help="seconds to wait for manual login in the opened browser",
    )
    questions_scan = subcommands.add_parser(
        "questions-scan",
        help="launch with approval and read all questions without clicking Join",
    )
    questions_scan.add_argument(
        "--candidate", required=True, help="exact visible candidate name"
    )
    questions_scan.add_argument(
        "--login-timeout",
        type=_positive_seconds,
        default=180.0,
        help="seconds to wait for manual login in the opened browser",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    project_root: Path = PROJECT_ROOT,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = Settings.load(
            project_root=project_root,
            environ=os.environ if environ is None else environ,
        )
    except ValueError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    if args.command == "config-dump":
        return _config_dump(settings)
    if args.command == "health":
        report = run_health_checks(settings)
        print(report.render())
        return 0 if report.overall == "READY_FOR_BROWSER_SCAN" else 1
    if args.command == "llm-test":
        return asyncio.run(
            _llm_test(
                settings,
                provider_name=args.provider,
                model_class=cast(ModelClass, args.model_class),
            )
        )
    if args.command == "llm-failover-test":
        return asyncio.run(_llm_failover_test(settings))
    if args.command == "listen-test":
        return _listen_test(settings, seconds=args.seconds)
    if args.command == "browser-scan":
        return _browser_scan(
            settings,
            login_timeout_seconds=args.login_timeout,
        )
    if args.command == "join":
        if args.dry_run:
            return _join_dry_run(
                settings,
                candidate_name=args.candidate,
                login_timeout_seconds=args.login_timeout,
            )
        return _join_live(
            settings,
            candidate_name=args.candidate,
            login_timeout_seconds=args.login_timeout,
        )
    if args.command == "questions-scan":
        return _questions_scan(
            settings,
            candidate_name=args.candidate,
            login_timeout_seconds=args.login_timeout,
        )
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
