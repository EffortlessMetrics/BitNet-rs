# BitNet-rs active goals

This directory is the repo-level entrypoint for machine-readable active goals.
It exists to point agents at the current lane without making proposals, specs,
ADRs, plans, status documents, or generated campaign dashboards do each
other's jobs.

## Role

Active goal manifests own:

- the current lane identifier and status;
- links to the proposal, specs, ADRs, and implementation plan;
- the ready work items an agent may select;
- proof commands for each work item;
- claim boundaries and status pointers.

They do not own product rationale, behavior contracts, durable decisions,
generated metrics, or public support claims.

## Files

```text
.bitnet-rs/goals/active.toml
.bitnet-rs/goals/archive/YYYY-MM-DD-<lane>.toml
```

If `active.toml` is absent, agents must use an explicitly named campaign item or
stop and report that no repo-level active goal is selected.

## Relationship to campaign tracking

BitNet-rs also has campaign-local manifests under
`docs/tracking/campaigns/<campaign>/active.toml`. Those campaign manifests
remain authoritative for campaign event history, generated dashboards, branch
metadata, and campaign-specific merge policy. A repo-level active goal should
link to the relevant campaign manifest when campaign tracking is the executable
work authority.

## Manifest shape

```toml
id = "bitnet-lane-id"
title = "Human-readable lane title"
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

campaigns = [
  "docs/tracking/campaigns/example/active.toml",
]

objective = """
State the current lane objective in one paragraph.
"""

end_state = [
  "Checkable end-state outcome.",
]

claim_boundaries = [
  "Do not broaden support claims without proof.",
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
campaign = "docs/tracking/campaigns/example/active.toml"
claim_boundary = "What this work item may and may not claim."
commands = [
  "git diff --check",
]
```

## Agent rule

Read `docs/reference/SPEC_SYSTEM.md` before interpreting a manifest. Select one
ready work item, run the listed proof commands, and stop instead of inventing
work when the manifest or linked artifacts are missing.
