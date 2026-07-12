# FloCareer Interview Copilot

A local-first, supervised copilot for FloCareer interviewers on macOS. The
project combines read-only browser scanning, Apple Speech system-audio
transcription, structured LLM evaluation, guarded cloud fallback, and
per-session audit files.

The copilot is intentionally human-controlled. Launch is available only after
an exact candidate-bound approval. The implementation never clicks hang-up,
fills feedback, changes ratings, or submits `FINISH`.

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
| Guarded candidate join discovery | Implemented and live-validated | `uv run python main.py join --candidate "Exact Name" --dry-run` |
| Approved real Launch, Join, and candidate wait | Implemented; live validation pending | `uv run python main.py join --candidate "Exact Name" --live` |
| Approved normal-question extraction | Implemented; 17-card live scan completed, final multiline fix pending revalidation | `uv run python main.py questions-scan --candidate "Exact Name"` |
| Coding-question detection and DOM capture | Implemented against semantic fixtures; watched revalidation pending | `uv run python main.py questions-scan --candidate "Exact Name"` |
| Guarded code-editor visibility | Integrated with the persistent live Join session; watched validation pending | `uv run python main.py join --candidate "Exact Name" --live --enable-code-editor-question 9` |
| Feedback fill and final submit | Not implemented | — |

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

### 7. Validate a candidate's launch control safely

After confirming the exact visible candidate name with `browser-scan`, run:

```bash
uv run python main.py join --candidate "Exact Candidate Name" --dry-run
```

The command requires `--dry-run`. It selects exactly one case-insensitive exact
name match, opens only that candidate card's menu, confirms one visible
`Launch Video Interview` control, and proves the launch action is blocked. It
never clicks Launch or Join. Missing names do not fuzzy-match and duplicate
names stop as ambiguous.

Artifacts are saved under `runs/join_<timestamp>/`:

```text
screenshots/candidate_found.png
screenshots/join_dry_run.png
action_log.jsonl
```

### 8. Launch, accept consent, and Join with separate approvals

Use only for a future scheduled interview while watching the browser:

```bash
uv run python main.py join --candidate "Exact Candidate Name" --live
```

The command pauses before Launch and prints a candidate-bound approval phrase.
After Launch it safely accepts either FloCareer path: when the `Interviewer
Consent Form` appears, it verifies and screenshots the form and requires a
separate approval before clicking its scoped `OK`; when consent was previously
acknowledged, FloCareer may open the verified `Joining as ...` pre-call page
directly and no consent click is attempted. Join always requires its own
approval. Each phrase is single-use and is never written to the action log.
After Join, the browser remains open until you manually end the interview and
type the displayed end confirmation. Automation never clicks hang-up or
`FINISH`.

Live-session artifacts are saved under `runs/join_live_<timestamp>/`:

```text
screenshots/candidate_found.png
screenshots/launch_approval.png
screenshots/consent.png
screenshots/pre_call.png
screenshots/joined.png
action_log.jsonl
```

### 9. Read and expand questions without joining

Run this while watching the browser:

```bash
uv run python main.py questions-scan --candidate "Exact Candidate Name"
```

The command requires the candidate-bound Launch approval and, only when shown,
a separate Consent OK approval. It then expands visible question cards and
saves their full text, ideal answers, rating guidelines, and locator hints.
A watched run reached all 17 normal cards. Full multiline extraction now binds
to FloCareer's supplied `.clFESingleSugDet` structure and has automated
coverage; that final correction awaits the next watched revalidation.
Semantic coding-question detection and automatic read-only DOM capture have
fixture coverage and await watched revalidation. The command never selects a
language, opens or changes an editor, or enables `SHOW CODE EDITOR TO
CANDIDATE`. It never clicks Join, feedback, ratings, `Mark as`, hang-up, or
`FINISH`.

Artifacts are saved under `runs/questions_scan_<timestamp>/`:

```text
questions.json
code_editor_dom.json
screenshots/questions_expanded.png
action_log.jsonl
```

`code_editor_dom.json` is generated automatically and read-only. For every
question exposing a semantic `Code Editor` tab, it records the question-number
evidence, exact tab count, SHOW/HIDE labels, switch-like control candidates,
their rendered state, the nearest control wrapper, and the encompassing card
structure. Each observation is classified as `unique`, `none`, or `ambiguous`.
Hidden-but-mounted controls are captured without opening the tab. Neither the
tab nor any candidate control is clicked during discovery.

To capture the post-tab visibility state without changing candidate visibility,
opt in to reversible scoped navigation:

```bash
uv run python main.py questions-scan --candidate "Exact Candidate Name" \
  --inspect-code-editor-tabs
```

Only exact coding-question `Code Editor` tabs are opened; the `Question` tab is
restored after capture. Browser-only guarded commands do not require LM Studio
to be running.

Structural HTML is allowlist-redacted and capped at 50,000 characters per
snapshot, with truncation and SHA-256 metadata. Unknown number layouts remain
`unresolved` rather than being guessed. The file is written with owner-only
permissions.

This private diagnostic can contain interview content and DOM attributes. Keep
it under ignored `runs/`; do not publish it or paste it into issues or chat.

### 10. Guarded code-editor visibility module

The isolated `browser/code_editor_workflow.py` guard/state module is implemented
for later stitching into the persistent live-session controller. It requires
the candidate binding established only after the exact-match joined-room
transition succeeds, scopes navigation to one exact question-number element,
opens one exact `Code Editor` tab, and interprets the visible state label
conservatively:

```text
SHOW CODE EDITOR TO CANDIDATE -> currently hidden
HIDE CODE EDITOR TO CANDIDATE -> currently visible; do not click
```

Showing a hidden editor requires a single-use approval bound to the candidate
identifier and question ID. The state is revalidated after the operator pause,
and the action passes only after the same card remains stably in the `HIDE...`
state. Ambiguous cards, labels, tabs, or semantic switch controls fail closed.
The module does not guess an unlabelled visual control, use coordinates, select
a language, type code, click hang-up, or click `FINISH`.

The verified card-scoped switch contract is
`input[type="checkbox"][name^="codeSwitch-"]`; it is always scoped to the
already-verified question card. Candidate-visible actions are available only
inside the persistent `join --live` session, after the candidate is connected:

```bash
uv run python main.py join --candidate "Exact Candidate Name" --live \
  --enable-code-editor-question 9
```

Launch, optional consent, Join, and showing the editor each require their own
fresh approval. The live session records `LAUNCHED`, `INTERVIEWER_IN_ROOM`,
`WAITING_FOR_CANDIDATE`, and `CANDIDATE_CONNECTED` transitions in
`room_state_log.jsonl`. The workflow remains watched-validation only until it
has passed against a future scheduled interview.

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
