# BitNet-rs active goals

This directory holds the machine-readable active goal manifest for the BitNet-rs
source-of-truth stack.

## Role

`.bitnet-rs/goals/active.toml` answers what an agent is actively executing now.
It does not replace proposals, specs, ADRs, implementation plans, status
receipts, or policy ledgers.

Use the stack in this order:

```text
Roadmap
  -> Proposal
    -> Spec
      -> ADR
        -> Implementation plan
          -> Active goal
            -> PR
              -> Proof
```

## Files

- `active.toml` — the current active or paused lane manifest.
- `archive/` — immutable archived manifests named `YYYY-MM-DD-<lane>.toml`.

`active.toml` may be absent during the scaffold rollout. When it exists, agents
must read it before selecting work. Until all existing BitNet-rs campaigns are
migrated, a plan may explicitly point to a campaign-local manifest under
`docs/tracking/campaigns/<campaign>/active.toml`; that pointer is the temporary
execution authority for that lane.

## Active manifest shape

```toml
id = "bitnet-lane-id"
title = "Human readable lane title"
status = "active"
owner = "codex-claude"
created = "2026-05-17"

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
  "Do not claim a public capability without support-tier proof.",
]

status_docs = [
  "docs/status/README.md",
]

[[work_item]]
id = "work-item-id"
status = "ready"
spec = "docs/specs/BITNET-SPEC-0001-contract.md"
adr = "docs/adr/BITNET-ADR-0001-decision.md"
plan = "plans/lane/implementation-plan.md#work-item-work-item-id"
current_pointer = "docs/status/README.md"
claim_boundary = "What this work item may and may not claim."
commands = [
  "git diff --check",
]
```

## Rules

1. Keep active goals machine-readable TOML.
2. Work on exactly one ready work item per PR.
3. Do not broaden claims beyond the linked spec, status row, and proof commands.
4. Archive an old manifest before replacing it with a new lane.
5. Do not hand-edit generated status; run the named generator/checker.
