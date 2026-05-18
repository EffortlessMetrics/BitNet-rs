# BitNet-rs active goals

This directory holds the cross-lane, machine-readable active goal manifest for
BitNet-rs when a lane is selected for agent execution.

The active goal is not a replacement for proposals, specs, ADRs, implementation
plans, support tiers, policy ledgers, or proof receipts. It points to those
artifacts and tells agents what to execute now.

## Files

```text
.bitnet-rs/goals/active.toml
.bitnet-rs/goals/archive/YYYY-MM-DD-<lane>.toml
```

`active.toml` may be absent while no cross-lane goal has been activated. If a
paused manifest is used, prefer this shape:

```toml
id = "bitnet-rs-paused"
title = "No selected implementation lane"
status = "paused"
owner = "codex-claude"
created = "2026-05-17"
reason = "No selected implementation lane."
```

## Required behavior

- Keep one active cross-lane goal at a time.
- Archive replaced manifests under `archive/`.
- Link to the proposal, plan, specs, ADRs, status docs, and policy ledgers that
  define the lane.
- Keep work items small and machine-readable.
- Include proof commands for every ready work item.
- Do not put generated dashboards or long rationale in TOML.

## Agent boot order

1. Read `AGENTS.md` or `CLAUDE.md`.
2. Read `docs/reference/SPEC_SYSTEM.md`.
3. Read `.bitnet-rs/goals/active.toml` if it exists.
4. Read the linked implementation plan.
5. Read the linked spec for the selected work item.
6. Read linked ADRs for constraints.
7. Inspect `git status`.
8. Implement exactly one ready work item.
9. Run the listed proof commands and `git diff --check`.
