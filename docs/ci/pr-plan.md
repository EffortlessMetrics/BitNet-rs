# PR Plan Routing Schema

`xtask ci plan` is the routing authority for BitNet-rs PR CI. Workflows should
consume its JSON instead of reimplementing path classifiers in shell.

The planner has two audiences:

1. humans, through the GitHub step summary; and
2. workflows, through a stable `ci-plan.json` schema.

This document specifies schema version `1`.

## JSON contract

A valid `ci-plan.json` for schema version `1` has this top-level shape:

```json
{
  "schema_version": 1,
  "budget": {
    "preferred_default_lem": 25,
    "default_limit_lem": 35,
    "estimated_lem": 0,
    "posture": "pennies|default|elevated|high|hard"
  },
  "classification": {
    "docs_only": false,
    "tracker_only": false,
    "rust_inputs_changed": false,
    "manifest_or_toolchain_changed": false,
    "public_api_changed": false,
    "gpu_changed": false,
    "macos_changed": false,
    "model_validation_changed": false,
    "coverage_requested": false,
    "full_ci_requested": false
  },
  "selected_lanes": [],
  "skipped_lanes": [],
  "packages": {
    "changed": [],
    "direct_dependents": [],
    "canaries": []
  },
  "risk_packs": [],
  "labels": []
}
```

Unknown future fields are allowed. Consumers must ignore unknown fields and
must fail closed when `schema_version` is absent or greater than the highest
version they understand.

## Budget fields

| Field | Meaning |
| --- | --- |
| `preferred_default_lem` | Preferred ordinary-PR budget target from `policy/ci-budget.toml`. |
| `default_limit_lem` | Normal default upper bound from `policy/ci-budget.toml`. |
| `estimated_lem` | Sum of selected static or learned lane estimates. |
| `posture` | Machine-readable posture: `pennies`, `default`, `elevated`, `high`, or `hard`. |

Budget guard labels are `full-ci`, `ci-budget-override`, and `ci-budget-ack`.
The initial guard is advisory unless the caller passes an explicit enforcement
mode.

## Classification fields

| Field | True when |
| --- | --- |
| `docs_only` | All changed files are documentation or docs-owned metadata. |
| `tracker_only` | The diff is limited to tracking/campaign files. |
| `rust_inputs_changed` | Rust source, Cargo manifests, lockfile, toolchain, build scripts, or Rust-affecting config changed. |
| `manifest_or_toolchain_changed` | `Cargo.toml`, `Cargo.lock`, `rust-toolchain.toml`, or `.cargo/**` changed. |
| `public_api_changed` | Public crate API, package, or release-facing surfaces changed. |
| `gpu_changed` | GPU HAL/backend/shader/device-selection paths changed. |
| `macos_changed` | Apple Silicon, Metal, macOS workflow, or macOS-specific code paths changed. |
| `model_validation_changed` | Model validation, crossval, fixture, receipt, or model-gate surfaces changed. |
| `coverage_requested` | Labels include `coverage` or `full-ci`. |
| `full_ci_requested` | Labels include `full-ci`. |

Phase 3 may add more precise no-Rust fields such as `no_rust_inputs`,
`tracker_or_campaign_only`, and `hardware_receipt_only`. Until then,
consumers should derive no-Rust behavior from `rust_inputs_changed == false`
only when they also understand the changed file set.

## Lane selection fields

`selected_lanes` contains lane IDs that are expected to run for the PR head.
Blocking lanes in this list are enforced by PR Gate. `skipped_lanes` contains
lane IDs intentionally not selected, with the reason reported in the human
summary.

Lane IDs must match `policy/ci-lanes.toml` and the authoritative lane whitelist
in `policy/ci-lane-whitelist.toml`.

PR Gate behavior:

| Lane state | Selected blocking lane | Unselected lane |
| --- | --- | --- |
| `success` | pass | pass |
| `failure`, `cancelled`, `timed_out` | fail | fail only if workflow policy says it is independently blocking |
| `skipped` | fail | pass |
| missing after timeout | fail | pass |

## Package selection fields

The `packages` object is advisory in schema version `1` and becomes actionable
when CI Core switches to package-selected proof.

| Field | Meaning |
| --- | --- |
| `changed` | Workspace packages directly touched by the diff. |
| `direct_dependents` | Workspace packages that directly depend on changed packages. |
| `canaries` | Additional packages or test surfaces selected because a risk pack requires them. |

Manifest/toolchain/foundational changes should set a broad-sweep reason in the
human summary even if `changed` is small.

## Required fixture cases

Planner tests should include fixtures for:

- docs-only changes,
- tracker-only changes,
- ordinary Rust changes,
- manifest/toolchain changes,
- GPU changes,
- macOS/Metal changes,
- coverage label requests,
- full-ci label requests.

Each fixture should assert the schema version, classification booleans, selected
lanes, skipped lanes, budget estimate, posture, labels, risk packs, and package
selection where applicable.

## Workflow consumption rules

Workflows should prefer `ci-plan.json` over duplicated path filters. If a
workflow still needs native GitHub `paths` filters for cost control, the filter
must be a coarse launcher only; the job body should still validate whether the
lane is selected before spending significant work.

No workflow should make macOS, Windows, Docker, model download, or live hardware
work part of ordinary unlabeled PR execution.
