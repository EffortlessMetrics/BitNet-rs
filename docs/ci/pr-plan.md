# PR Plan

`xtask ci plan` emits the `ci-plan.json` artifact used by PR Plan and PR Gate.
PR Plan publishes the advisory artifact for visibility; PR Gate recomputes the
same plan on the PR head and uses selected blocking lanes as its routing input.

## Stable Schema

The JSON artifact uses `schema_version = 1` and keeps these top-level fields:

```json
{
  "schema_version": 1,
  "budget": {
    "preferred_default_lem": 25,
    "default_limit_lem": 35,
    "estimated_lem": 0,
    "posture": "pennies"
  },
  "classification": {
    "no_rust_inputs": false,
    "docs_only": false,
    "tracker_only": false,
    "tracker_or_campaign_only": false,
    "hardware_receipt_only": false,
    "policy_docs_only": false,
    "rust_inputs_changed": true,
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
    "canaries": [],
    "selected": [],
    "broad_sweep_required": false,
    "selection_reason": "changed packages plus direct dependents and canaries"
  },
  "risk_packs": [],
  "labels": []
}
```

`selected_lanes` entries include `id`, `name`, `estimated_lem`, `reason`, and
`blocking`. `skipped_lanes` entries include `id`, `name`, `reason`, and
`blocking`.

`packages` is the CI Core package-selection contract. `changed` comes from
changed workspace package paths, `direct_dependents` comes from `cargo metadata`,
`canaries` comes from risk packs, and `selected` is the union used by CI Core
when a broad sweep is not required. `broad_sweep_required=true` preserves the
full core sweep for manifest, toolchain, and shared-foundation changes.

## Boundaries

PR Gate consumes `selected_lanes[]` with `blocking = true`. Selected blocking
lanes must produce successful check runs; selected blocking lanes that are
skipped fail the gate. Unselected lanes may be skipped, and selected advisory
lanes are reported without blocking the gate.

The planner may estimate route jobs and policy lanes, but those estimates do
not promote expensive macOS, Windows, Docker, GPU, coverage, model-validation,
or performance lanes onto ordinary PRs unless the plan selects them. Branch
protection still remains a separate repository setting.

## Fixture Coverage

Schema fixture tests cover:

- docs-only changes,
- tracker-only changes,
- hardware receipt-only changes,
- policy docs-only changes,
- ordinary Rust changes,
- manifest/toolchain and public API changes,
- GPU and macOS changes,
- model-validation paths,
- `coverage` and `full-ci` label classification.
