# Tracker Model

The alignment tracker is moving from one high-churn global ledger toward campaign-local control cards plus generated dashboards.

The existing files under `docs/tracking/bitnet-alignment/` remain the transition source of truth until generator/checker tooling lands. New work should prefer campaign-local tracker files for planning and should avoid hand-editing global dashboard rows except for transition compatibility.

## Model

Each campaign has:

- `CAMPAIGN.md` for narrative context, constraints, sequencing, non-goals, and review policy.
- `active.toml` for compact machine-readable work items.
- `events/` for append-only lifecycle records in later PRs.
- `generated/` for derived dashboards in later PRs.

Global dashboards should be generated from campaign manifests and events. Agents should not solve global dashboard conflicts by deleting hardware lanes.

## Work Item Contract

Each `[[work_item]]` in `active.toml` should include:

- `id`
- `status`
- `branch`
- `stackable`
- `requires_human_merge`
- `blocked_by`
- `acceptance`
- `commands`

Items may also include `allowed_paths`, `forbidden_paths`, `may_claim`, and `must_not_claim` when the scope needs hard boundaries.

## State Rules

Use boring states:

- `proposed`
- `ready`
- `in_progress`
- `pr_open`
- `blocked`
- `merged`
- `superseded`

Do not mark a PR as `merged` until the merge SHA exists. Use GitHub PRs as live locks and append-only events as audit records in later PRs.

## Stackability

Most runtime and hardware implementation work should set:

```toml
stackable = false
requires_human_merge = true
```

Docs-only scaffolding can be stackable when it does not change a lane contract or shared runtime surface. Backend identity, probe, smoke, parity, receipt, and benchmark work should generally be non-stackable.

## Transition Rules

- Keep hardware lane visibility intact.
- Do not remove A770, NPU, 258V, 8250U, AMD, NVIDIA, or M4 coordination rows.
- Do not edit generated dashboards by hand once generator tooling exists.
- If generated files conflict during rebase, regenerate them instead of resolving tables manually.
- If two branches touch the same item manifest, stop and resolve ownership before continuing.
