# Tracker Model

The alignment tracker is moving from one high-churn global ledger toward campaign-local control cards plus generated dashboards.

Campaign-local TOML manifests and append-only events are the intended source of truth for active campaign work. The existing files under `docs/tracking/bitnet-alignment/` remain transition surfaces for compatibility and historical context; normal item PRs should stop editing those global files once generated dashboards are available.

## Model

Each campaign has:

- `CAMPAIGN.md` for narrative context, constraints, sequencing, non-goals, and review policy.
- `active.toml` for compact machine-readable work items.
- `events/` for append-only lifecycle records in later PRs.
- `generated/` for derived dashboards.

Global dashboards are generated from campaign manifests and events. Agents should not solve global dashboard conflicts by deleting hardware lanes.

Lane ownership and shared-surface rules are defined in
[`LANE_OWNERSHIP.md`](LANE_OWNERSHIP.md). PRs must declare their lane,
campaign, work item, orchestrator, branch, base main SHA, allowed paths, shared
surfaces touched, and closeout requirement so parallel orchestrators can avoid
collisions.

Use:

```bash
cargo run -p xtask --no-default-features -- campaign list
cargo run -p xtask --no-default-features -- campaign status apple-m4
cargo run -p xtask --no-default-features -- campaign next apple-m4
cargo run -p xtask --no-default-features -- campaign check apple-m4
cargo run -p xtask --no-default-features -- campaign generate
cargo run -p xtask --no-default-features -- campaign doctor
```

CI runs the strict forms with `--locked`:

```bash
cargo run --locked -p xtask --no-default-features -- campaign doctor
cargo run --locked -p xtask --no-default-features -- campaign generate --check
```

`campaign doctor` treats stale generated dashboards and normal item PR edits to
legacy global tracker files as failures. Branches whose names contain
`tracker-infra`, `legacy-migration`, or `generated-dashboard` are the intended
maintenance exceptions for legacy tracker edits.

When `GITHUB_REPOSITORY` and `GITHUB_TOKEN` are available outside push-to-main
checks, `campaign doctor` also reconciles campaign state against live open
GitHub PRs. In pull-request CI, strict reconciliation is scoped to the current
PR so parallel branches do not need to carry one another's campaign TOML state.
The current open PR must have its claimed item marked `pr_open` with a matching
`pr_open` event, and an item marked `pr_open` must still have a live open PR
that claims the item.

## Work Item Contract

Each `[[work_item]]` in `active.toml` should include:

- `id`
- `status`
- `branch`
- `stackable`
- `review_mode`
- `merge_policy`
- `human_gate`
- `blocked_by`
- `acceptance`
- `commands`

Items should also include `allowed_paths`, `forbidden_paths`, `may_claim`, and `must_not_claim` for implementation or runtime work. Documentation-only items may keep these short, but they should still make their boundaries explicit.

Normal item PRs should edit only their campaign files and their scoped implementation paths. They should not hand-edit:

- `docs/tracking/bitnet-alignment/status.md`
- `docs/tracking/bitnet-alignment/workstream-ledger.yaml`
- `docs/tracking/generated/*.md`
- `docs/tracking/campaigns/*/generated/*.md`

Tracker infrastructure PRs may touch transition docs and generated dashboards.

## State Rules

Use boring states:

- `proposed`
- `ready`
- `in_progress`
- `pr_open`
- `blocked`
- `merged`
- `superseded`

Do not mark a PR as `merged` until the merge SHA exists. Use GitHub PRs as live locks and append-only events as audit records.

Lifecycle events are TOML files under `events/` with these event types:

- `in_progress`
- `pr_open`
- `blocked`
- `superseded`
- `merged`
- `closeout`

Merged events must include `merge_sha`. `pr_open` events must include the PR
number; head SHA remains recommended so humans can audit which branch state
opened the PR.

## Stackability

Most runtime and hardware implementation work should set:

```toml
stackable = false
review_mode = "codex_premerge"
merge_policy = "automerge_when_green"
human_gate = "on_blocker_only"
```

`stackable = false` means dependent work waits for this item to land. It does
not mean Codex should stop before review, CI repair, auto-merge, or closeout.

For work items with `review_mode = "codex_premerge"`,
`merge_policy = "automerge_when_green"`, and
`human_gate = "on_blocker_only"`, the agent is authorized and expected to:

1. edit files within the item scope,
2. run scoped validation,
3. commit,
4. push,
5. open or update the PR,
6. refresh the agent-owned PR branch when needed, including merge-from-main,
   rebase, `gh pr update-branch`, or `--force-with-lease` after branch, status,
   and diff inspection,
7. address CI, bot, and reviewer feedback,
8. merge the PR when required checks are green and GitHub reports it mergeable,
9. create and merge closeout tracker PRs when required.

The commit, push, PR creation, agent-owned PR branch refresh, CI/bot/reviewer
repair, merge, and tracker closeout boundaries are not human gates for those
items. Human involvement is required only for true blockers: GitHub permissions
or branch protection preventing merge, direct mutation of `origin/main`,
destructive cleanup, secret/model-binary exposure risk, unresolved
kernel/math/tokenizer/loader semantic conflict, acceptance criteria that
conflict with repo policy, or a cost/exposure/release decision genuinely outside
the ticket scope.

This campaign work item policy is the authority for campaign branches. If an
older agent runbook describes default human approvals, maintainer-ready handoff,
or manual intervention for ordinary commit, push, PR, PR branch refresh, CI
repair, merge, or closeout work, follow the campaign work item policy instead.
The durable operations reference is
[`docs/development/AGENTIC_PR_OPERATIONS.md`](../development/AGENTIC_PR_OPERATIONS.md).

Allowed `review_mode` values are:

- `codex_premerge`
- `human_required`
- `external_required`
- `none`

Allowed `merge_policy` values are:

- `automerge_when_green`
- `codex_merge_when_green`
- `manual_only`
- `no_merge`

Allowed `human_gate` values are:

- `never`
- `on_blocker_only`
- `before_merge`
- `always`

Docs-only scaffolding can be stackable when it does not change a lane contract
or shared runtime surface. Backend identity, probe, smoke, parity, receipt, and
benchmark work should generally be non-stackable.

## Transition Rules

- Keep hardware lane visibility intact.
- Do not remove A770, NPU, 258V, 8250U, AMD, NVIDIA, or M4 coordination rows.
- Do not edit generated dashboards by hand once generator tooling exists.
- If generated files conflict during rebase, regenerate them instead of resolving tables manually.
- If two branches touch the same item manifest, stop and resolve ownership before continuing.
- Treat `docs/tracking/bitnet-alignment/status.md` and `docs/tracking/bitnet-alignment/workstream-ledger.yaml` as transition surfaces. Freeze or generate them in a later PR instead of keeping them as another hand-edited source of truth.
