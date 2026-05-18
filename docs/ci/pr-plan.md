# PR Plan

`xtask ci plan` emits the `ci-plan.json` artifact used by PR Plan and future
routing consumers. The artifact is advisory until PR Gate is explicitly moved
to consume it.

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
    "docs_only": false,
    "tracker_only": false,
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
    "canaries": []
  },
  "risk_packs": [],
  "labels": []
}
```

`selected_lanes` entries include `id`, `name`, `estimated_lem`, `reason`, and
`blocking`. `skipped_lanes` entries include `id`, `name`, `reason`, and
`blocking`.

## Boundaries

This schema does not change workflow routing by itself. Workflows must not
depend on new fields until the dedicated PR Gate consumption item lands.

The planner may estimate route jobs and policy lanes, but those estimates are
visibility only. They are not branch-protection rules and they do not promote
expensive macOS, Windows, Docker, GPU, coverage, model-validation, or
performance lanes onto ordinary PRs.

## Fixture Coverage

Schema fixture tests cover:

- docs-only changes,
- tracker-only changes,
- ordinary Rust changes,
- manifest/toolchain and public API changes,
- GPU and macOS changes,
- model-validation paths,
- `coverage` and `full-ci` label classification.
