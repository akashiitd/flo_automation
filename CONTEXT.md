# FloCareer Interview Copilot Context

## Glossary

- **Interview Session** — One isolated, supervised interview run. It owns a
  random session identifier, its audit artifacts, and its durable graph thread.
- **Interview Event** — An immutable, deduplicated observation or operator
  command entering one Interview Session. An event describes what happened; it
  does not itself authorize an action.
- **Intent Decision** — A bounded interpretation of candidate speech, with a
  confidence value and evidence from that speech. It recommends a route but
  cannot override safety policy.
- **Effect Request** — A typed request for an external operation such as
  speech playback, guarded browser work, transcript persistence, or evaluation.
  It is not proof that the operation occurred.
- **Effect Result** — The durable outcome recorded for an Effect Request. An
  uncertain result requires recovery review rather than an automatic replay.
- **Question Plan Item** — One candidate question or explicitly skipped source
  card, including its skill mapping, priority, time estimate, and skip reason.
- **Skill Parameter** — A read-only FloCareer assessment dimension, including
  its requirement, level, and rating scale.
- **Skill Evidence** — A citation from one evaluated question answer that
  supports assessment of one Skill Parameter.
- **Skill Assessment** — An evidence-grounded proposed score for one Skill
  Parameter, or an explicit insufficient-evidence outcome.
- **Human Interrupt** — A graph pause that needs an operator decision because
  authority, safety, or recovery cannot be resolved automatically.
