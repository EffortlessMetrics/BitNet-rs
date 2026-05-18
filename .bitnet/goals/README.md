# BitNet-rs active goals

This directory holds repo-level active goal manifests for the BitNet-rs
source-of-truth stack.

The active manifest is optional until a lane is explicitly promoted to a
repo-level goal. Campaign-local work may continue to use
`docs/tracking/campaigns/<campaign>/active.toml`, but each work item must have
one clear active execution authority.

## Files

```text
.bitnet/goals/active.toml
.bitnet/goals/archive/YYYY-MM-DD-<lane>.toml
```

## Manifest role

An active goal owns only current machine-readable execution state:

- lane id and title;
- linked proposal, specs, ADRs, and plan;
- current objective;
- checkable end state;
- claim boundaries;
- work items and proof commands;
- status and support-tier pointers.

It does not own product rationale, durable decisions, generated metrics, or
public support claims. Link to the proposal, spec, ADR, plan, status docs, and
policy ledgers for those truths.

## Template

```toml
id = "bitnet-lane-id"
title = "Human readable lane title"
status = "active"
owner = "codex-claude"
created = "YYYY-MM-DD"

proposal = "docs/proposals/BITNET-PROP-0001-lane.md"
plan = "plans/lane/implementation-plan.md"

specs = [
  "docs/specs/BITNET-SPEC-0001-contract.md",
]

adrs = [
  "docs/adr/BITNET-ADR-0001-decision.md",
]

objective = """
State the current lane objective in one paragraph.
"""

end_state = [
  "Checkable end-state outcome.",
]

claim_boundaries = [
  "Do not claim answer readiness until support-tier proof exists.",
]

status_docs = [
  "docs/status/SUPPORT_TIERS.md",
]

[[work_item]]
id = "work-item-id"
status = "ready"
spec = "docs/specs/BITNET-SPEC-0001-contract.md"
adr = "docs/adr/BITNET-ADR-0001-decision.md"
plan = "plans/lane/implementation-plan.md#work-item-work-item-id"
current_pointer = "docs/status/SUPPORT_TIERS.md"
claim_boundary = "What this work item may and may not claim."
commands = [
  "git diff --check",
]
```

## Lifecycle

- Create `active.toml` only when a lane has a selected plan and proof contract.
- Archive a replaced active goal under `archive/` before activating a new one.
- Use `status = "paused"` with a `reason` when there is no selected lane.
- Do not leave multiple active repo-level goals.
