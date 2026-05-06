<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Tracker infrastructure Campaign Status

- Campaign: `tracker-infra`
- State: `active`
- Objective: Finish the move from global hand-edited alignment trackers to campaign-local TOML manifests, append-only events, generated dashboards, and advisory xtask gates.

## Work Items

| Item | State | PR | Branch | Acceptance |
|---|---|---:|---|---|
| TRACKER-001 | pr_open | #3660 | `codex/tracker-infra/TRACKER-001-campaign-local-gates` | Add campaign-local tracker model docs, missing campaign manifests, append-only event rules, advisory xtask campaign check/generate/doctor commands, and generated global dashboards. |

## Hard Constraints

- Do not touch runtime code, kernels, or dependencies for tracker infrastructure.
- Do not remove hardware lane visibility.
- Do not mark work merged without a merge SHA.
- Do not name the pattern after the other repository it was borrowed from.
