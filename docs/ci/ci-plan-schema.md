# CI Plan JSON Schema

`xtask ci plan` emits `ci-plan.json`, the stable routing contract consumed by
GitHub Actions workflows and agents. The schema is introduced by the routed
verification rollout before workflows begin relying on it.

## Compatibility rules

- `schema_version` must be incremented for breaking changes.
- New optional fields may be added without incrementing the version.
- Existing fields must not change meaning within a schema version.
- Workflows must tolerate unknown fields.
- Required arrays should be emitted as empty arrays instead of being omitted.
- Required booleans should be emitted as `false` instead of being omitted.

## Schema version 1

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

## Field semantics

### `budget`

| Field | Meaning |
| --- | --- |
| `preferred_default_lem` | Preferred default PR target from `policy/ci-budget.toml`. |
| `default_limit_lem` | Normal default PR limit from `policy/ci-budget.toml`. |
| `estimated_lem` | Sum of selected lane estimates in Linux-equivalent minutes. |
| `posture` | Stable enum: `pennies`, `default`, `elevated`, `high`, or `hard`. |

### `classification`

| Field | True when |
| --- | --- |
| `docs_only` | All changed files are documentation-only and cannot affect Rust build inputs. |
| `tracker_only` | All changed files are campaign/tracker metadata. |
| `rust_inputs_changed` | A changed file can affect Rust compile, lint, test, or feature behavior. |
| `manifest_or_toolchain_changed` | Cargo manifests, lockfiles, toolchain files, or `.cargo/**` changed. |
| `public_api_changed` | Public API surfaces changed and compatibility/MSRV risk may be elevated. |
| `gpu_changed` | GPU, accelerator, kernel, or GPU workflow paths changed. |
| `macos_changed` | Apple Silicon, Metal, macOS runner, or Apple hardware docs/workflow paths changed. |
| `model_validation_changed` | Model validation, receipts, cross-validation, or golden output surfaces changed. |
| `coverage_requested` | `coverage` or equivalent coverage label is present. |
| `full_ci_requested` | `full-ci` label is present. |

### `selected_lanes` and `skipped_lanes`

Each lane entry should use stable lane IDs from `policy/ci-lanes.toml` and
`policy/ci-lane-whitelist.toml`. A future schema may promote entries from
strings to objects with reasons; version 1 consumers must treat strings as the
required interoperable format.

### `packages`

| Field | Meaning |
| --- | --- |
| `changed` | Workspace packages directly touched by changed Rust inputs. |
| `direct_dependents` | Workspace packages that directly depend on changed packages. |
| `canaries` | Additional packages or smoke checks selected for risk-specific evidence. |

### `risk_packs`

Stable risk-pack IDs selected from `policy/ci-risk-packs.toml`, such as
`manifest_release`.

### `labels`

Normalized PR labels that influenced routing.

## Required fixture coverage

Schema version 1 must have fixture tests for at least:

- docs-only changes,
- tracker-only changes,
- ordinary Rust changes,
- manifest/toolchain changes,
- GPU changes,
- macOS changes,
- `full-ci` label,
- coverage label.
