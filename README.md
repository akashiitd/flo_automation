# FloCareer Interview Copilot

A local-first, supervised copilot for FloCareer interviewers on macOS. The
project combines read-only browser scanning, Apple Speech system-audio
transcription, structured LLM evaluation, guarded cloud fallback, and
per-session audit files.

The copilot is intentionally human-controlled. The implemented version does
not launch interviews, click hang-up, fill feedback, or submit `FINISH`.

The complete roadmap and safety constraints are documented in
[`FLOCAREER_AUTOMATION_PLAN.md`](FLOCAREER_AUTOMATION_PLAN.md).

## Current status

| Capability | Status | Command |
| --- | --- | --- |
| Configuration validation | Implemented | `uv run python main.py config-dump` |
| Local environment health check | Implemented | `uv run python main.py health` |
| LM Studio structured evaluation | Implemented | `uv run python main.py llm-test --provider lmstudio` |
| OpenRouter structured evaluation | Implemented, explicit opt-in | `uv run python main.py llm-test --provider openrouter` |
| Guarded provider failover | Implemented | `uv run python main.py llm-failover-test` |
| Apple Speech system-audio capture | Implemented | `uv run python main.py listen-test --seconds 60` |
| Read-only FloCareer dashboard scan | Implemented | `uv run python main.py browser-scan` |
| Join, question extraction, feedback fill, final submit | Not implemented | — |

## Safety model

- FloCareer authentication is completed manually in a persistent local browser.
- The scanner reads visible dashboard content and saves a screenshot; it does
  not open candidate menus or launch interviews.
- Candidate audio is captured from macOS system audio. Microphone capture is
  forced off in `listen-test`.
- LM Studio is the local primary provider.
- OpenRouter is blocked unless both an API key and explicit cloud-data consent
  are configured.
- Candidate PII is redacted before every permitted OpenRouter generation.
- Runtime transcripts, screenshots, browser profiles, and `.env` are ignored by
  Git.
- A human remains responsible for interview questions, feedback, hiring
  decisions, and final submission.

## Requirements

- macOS on Apple Silicon
- Python 3.12 or newer
- [`uv`](https://docs.astral.sh/uv/)
- LM Studio with its local API server available at `http://127.0.0.1:1234/v1`
- The existing `Meeting_transcriber_with_LLM` project with its Apple Speech
  helper
- macOS permissions for Speech Recognition and Screen & System Audio Recording
- Optional: an OpenRouter API key for guarded cloud fallback

The current tested local model is `ornith-1.0-35b`.

## Installation

```bash
git clone https://github.com/akashiitd/flo_automation.git
cd flo_automation

uv sync
uv run playwright install chromium
cp .env.example .env
```

Edit `.env` locally. At minimum, verify the path to the existing transcription
project:

```env
MEETING_TRANSCRIBER_PATH=../Meeting_transcriber_with_LLM
```

If it is not a sibling directory, use an absolute path. Never commit `.env`.

## Configuration

Important settings in `.env`:

```env
# Local provider
LMSTUDIO_BASE_URL=http://127.0.0.1:1234/v1
LMSTUDIO_FAST_MODEL=ornith-1.0-35b
LMSTUDIO_DEEP_MODEL=ornith-1.0-35b

# Cloud fallback: disabled until explicitly permitted
OPENROUTER_API_KEY=
LLM_ALLOW_CLOUD_CANDIDATE_DATA=false

# Existing Apple Speech project
MEETING_TRANSCRIBER_PATH=../Meeting_transcriber_with_LLM
TRANSCRIPTION_BACKEND=apple-speech
TRANSCRIBE_SYSTEM_AUDIO=true
TRANSCRIBE_MICROPHONE=false

# FloCareer and safety
FLOCAREER_URL=https://app.flocareer.com/
REQUIRE_APPROVAL_BEFORE_FINISH=true
```

To allow OpenRouter, set both values locally:

```env
OPENROUTER_API_KEY=your-key
LLM_ALLOW_CLOUD_CANDIDATE_DATA=true
```

The application never prints the key. Redaction is deterministic but should
not be treated as a substitute for organizational approval to use cloud
processing.

## Usage

Run commands from the repository root.

### 1. Validate configuration

```bash
uv run python main.py config-dump
```

The output reports whether secrets are configured without displaying them.

### 2. Check local readiness

Start the LM Studio API server, load the configured model, then run:

```bash
uv run python main.py health
```

Expected final status:

```text
Overall: READY_FOR_BROWSER_SCAN
```

Supertonic is currently optional and may appear as a warning.

### 3. Test local evaluation

```bash
uv run python main.py llm-test --provider lmstudio
```

The command returns a validated score, rating, transcript evidence, follow-up,
feedback, confidence, latency, tokens, and estimated cost.

### 4. Test OpenRouter

Only after cloud processing has been approved and enabled:

```bash
uv run python main.py llm-test --provider openrouter
uv run python main.py llm-failover-test
```

Provider usage is appended to:

```text
runs/llm_tests/llm_usage.jsonl
```

### 5. Test Apple Speech system audio

```bash
uv run python main.py listen-test --seconds 60
```

While it runs, play English speech from Chrome or another Mac application. You
do not need to speak. The command passes only when at least one `system` segment
is captured and no `microphone` segment is present. Press `Control+C` to stop
early.

Output is saved under:

```text
runs/listen_<timestamp>/transcript.json
runs/listen_<timestamp>/transcript.txt
```

### 6. Scan the FloCareer dashboard

```bash
uv run python main.py browser-scan --login-timeout 180
```

The scanner initially opens `https://app.flocareer.com/` in a persistent local
Chromium profile. Complete authentication manually in that browser. FloCareer
may redirect to `/interviewer/` after login.

The command extracts scheduled interview cards and saves:

```text
runs/browser_scan_<timestamp>/screenshots/dashboard.png
```

It does not click the three-dot menu or launch an interview.

## Validation and development

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uvx ty check app browser evaluator llm transcriber main.py
```

The project currently has unit/integration coverage for configuration, health
checks, structured schemas, JSON repair, provider behavior, PII redaction,
failover policy, transcript persistence, Apple Speech integration, delayed
authentication states, and FloCareer table/card extraction.

## Session data and privacy

Runtime files live under `runs/<session_id>/`. Depending on the command, a
session may contain transcripts, LLM usage records, and dashboard screenshots.
These files can contain candidate or interview information and are excluded
from Git by default.

Before sharing logs, redact candidate names, contact details, resume fields,
account identifiers, and unnecessary interview content.

## Troubleshooting

### Apple Speech captures nothing

In **System Settings → Privacy & Security**, enable:

- Speech Recognition
- Screen & System Audio Recording
- Microphone only if a future workflow explicitly requires it

Then play audible speech from another Mac application during `listen-test`.

### LM Studio is unreachable or slow

- Start the LM Studio Developer/API server.
- Confirm `LMSTUDIO_BASE_URL=http://127.0.0.1:1234/v1`.
- Load the configured model before a live session.
- Reduce context length if latency is too high.

### OpenRouter is blocked

Both `OPENROUTER_API_KEY` and
`LLM_ALLOW_CLOUD_CANDIDATE_DATA=true` are required. Leaving cloud processing
disabled is a supported local-only configuration.

### FloCareer reports that you are logged out

Run `browser-scan` in headed mode and complete login manually in the opened
Chromium window. The session is stored in `.browser-profile/` for later scans.
Do not send credentials or OTPs through logs or chat.

### Dashboard loads but no candidates are extracted

Confirm that scheduled interview cards are visibly present. The command reports
validation as incomplete when no cards are visible, rather than treating an
empty or loading shell as a pass.

## Roadmap

The next guarded milestones are:

1. Candidate lookup and join workflow with a strict `--dry-run` mode.
2. Question and rubric extraction.
3. Code-editor visibility control.
4. Offline full-session evaluation.
5. Stateful interview orchestration and timer events.
6. Human-approved feedback autofill.
7. Optional Supertonic voice integration after text-only stability.

Final submission remains outside unattended automation.
