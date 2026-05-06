# Tracker Infrastructure Campaign

Campaign ID: `tracker-infra`

Status: active

## Objective

Finish the move from global hand-edited alignment trackers to campaign-local TOML manifests, append-only events, generated dashboards, and advisory xtask gates.

## End State

- Campaign manifests and events are the source of truth for active work.
- Generated dashboards preserve global coordination visibility.
- Legacy global status and ledger files are transition surfaces, not normal item PR targets.
- xtask can check, generate, and doctor campaign state.

## Hard Constraints

- Do not touch runtime code, kernels, or dependencies for tracker infrastructure.
- Do not remove hardware lane visibility.
- Do not mark work merged without a merge SHA.
- Do not name the pattern after the other repository it was borrowed from.

## Work Items

| Work item | Status | Notes |
|---|---|---|
| TRACKER-001 | in_progress | Add advisory campaign check/generate/doctor commands and generated dashboards. |

## Review Policy

Tracker infrastructure PRs may touch global tracker transition docs and generated dashboards. Normal item PRs should not hand-edit legacy global tracker files once generation lands.
