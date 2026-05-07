<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Tracker infrastructure Campaign Status

- Campaign: `tracker-infra`
- State: `active`
- Objective: Finish the move from global hand-edited alignment trackers to campaign-local TOML manifests, append-only events, generated dashboards, and enforced xtask gates.

## Work Items

| Item | State | PR | Branch | Review | Merge | Human gate | Acceptance |
|---|---|---:|---|---|---|---|---|
| TRACKER-001 | merged | #3660 | `codex/tracker-infra/TRACKER-001-campaign-local-gates` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add campaign-local tracker model docs, missing campaign manifests, append-only event rules, advisory xtask campaign check/generate/doctor commands, and generated global dashboards. |
| TRACKER-002 | merged | #3681 | `codex/tracker-infra/TRACKER-002-ci-enforcement` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add CI enforcement for campaign doctor and generated-dashboard freshness after the advisory tracker gate has landed, with stale generated dashboards and normal legacy tracker edits treated as hard failures. |
| TRACKER-003 | pr_open | #3724 | `codex/tracker-infra/TRACKER-003-current-pr-reconciliation` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Scope GitHub PR reconciliation to the current pull request in PR CI so parallel campaign branches do not need to carry each other's item TOML changes. |

## Hard Constraints

- Do not touch runtime code, kernels, or dependencies for tracker infrastructure.
- Do not remove hardware lane visibility.
- Do not mark work merged without a merge SHA.
- Do not name the pattern after the other repository it was borrowed from.
