# Crate Collapse Campaign

Campaign ID: `crate-collapse`

Status: active

## Objective

Collapse low-risk public microcrates into SRP modules while preserving behavior, feature gates, and public API intent.

## End State

- Low-risk leaf crates are moved into destination modules.
- Imports and workspace membership are updated in small PRs.
- Behavior is unchanged unless an item explicitly permits a public API change.
- Verification gates remain green after each collapse.

## Hard Constraints

- Do not combine crate movement with runtime proof.
- Do not change hardware lane semantics.
- Do not move domain crates before their leaf dependencies are settled.
- Do not mass-format unrelated files.

## Work Items

| Work item | Status | Notes |
|---|---|---|
| INV-001 | merged | Crate consolidation inventory merged in #3632. |
| LEAF-001 | proposed | Collapse warn-once into common. |

## Review Policy

Crate-collapse PRs can be stackable only when they touch disjoint crates and manifests. Shared workspace manifest changes should be reviewed carefully because they create rebase pressure.
