# Configuration layering and clamps

This document describes how BitNet builds test configurations and why the final "clamp"
happens in the test wrapper rather than the manager.

## Layers

1. **Scenario defaults** – `ScenarioConfigManager::resolve(scenario, environment)` yields
   the base structure for the requested scenario and host environment.
2. **Platform caps** – manager applies stable platform limits (e.g., Windows ≤ 8, macOS ≤ 6).
3. **Context clamps** – the *test wrapper* (see `tests/test_configuration_scenarios.rs`)
   applies runtime, context‑driven overrides:
   - **Fast‑feedback** (`target_feedback_time`): JSON‑only, no coverage/perf; artifacts off if ≤ 30s
   - **Resources** (parallelism/network/disk cache)
   - **Quality** (coverage thresholds, cross‑validation, tolerances)

```mermaid
flowchart TD
    A[Scenario Defaults] --> B[Environment Overlay]
    B --> C[Platform Caps<br/>(Manager)]
    C --> D[Context Clamps<br/>(Test Wrapper)]
    D --> E[Effective Test Config]
```

## Why clamps live in the test wrapper

- Avoids **double application** when other consumers also clamp.
- Keeps `ScenarioConfigManager` reusable as a pure base config builder.
- Makes fast/slow test modes explicit and easy to reason about in the harness.

If an external consumer needs clamped configs from the manager directly, see ADR‑0001 ("Unify clamps") for the alternative.

## Fast‑feedback specifics

| Target feedback time (tft) | Coverage | Performance | Formats | Artifacts | Parallel cap |
|---|---|---|---|---|---|
| `tft > 120s` | unchanged | unchanged | unchanged | unchanged | unchanged |
| `30s < tft ≤ 120s` | **off** | **off** | **JSON‑only** | unchanged | `≤ 4` |
| `tft ≤ 30s` | **off** | **off** | **JSON‑only** | **off** | `≤ 4` |

## Cross‑validation tolerances

Ensure non‑negative values; when set, apply to both token and numerical metrics:

```rust
let tol = qr.accuracy_tolerance.max(0.0);
if tol > 0.0 {
    cfg.crossval.tolerance.min_token_accuracy = tol;
    cfg.crossval.tolerance.numerical_tolerance = tol;
}
```

## Env safety

All env mutations in tests must be guarded by `env_guard()` (shared `OnceLock<Mutex<()>>`).
