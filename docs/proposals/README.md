# Proposals

Proposals explain why a BitNet-rs effort exists, what user pain or repo risk it
addresses, and what success means before a lane becomes a spec, ADR, plan, or
active work item.

Proposals do not own implementation status, active work, generated status, or
public support claims. Those belong to plans, active goals, status documents,
policy ledgers, and proof receipts.

## Source-of-truth role

| Question | Source of truth |
| --- | --- |
| Why does this lane exist? | `docs/proposals/` |
| What must be true? | `docs/specs/` |
| What decision did we make? | `docs/adr/` |
| What PRs execute it? | `plans/` plus the active goal or campaign manifest |
| What is actively executing now? | `.bitnet/goals/active.toml` or `docs/tracking/campaigns/<campaign>/active.toml` |
| What is currently supported? | `docs/status/` plus proof artifacts |
| What does policy enforce or except? | `policy/*.toml` and workflow gates |
| What happened? | Receipts, artifacts, campaign events, closeouts |

## BitNet claim rules

Every proposal that affects user-visible capability claims should link to the
status, model-artifact, hardware, CI, and campaign surfaces that will carry the
actual proof. A proposal may describe intended outcomes, but it must not claim
support without support-tier proof or an equivalent receipt pointer.

## Proposal shape

New proposals should include:

```text
Status:
Owner:
Created:
Target milestone:
Linked specs:
Linked ADRs:
Linked plan:
Support-tier impact:
Policy impact:
```

Recommended sections:

- Problem
- Users and surfaces
- Success criteria
- Proposed shape
- Alternatives considered
- Specs to create or update
- ADRs needed
- Implementation campaign shape
- Evidence plan
- Risks
- Non-goals
- Exit criteria
- Claim boundary
