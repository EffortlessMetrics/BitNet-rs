# ADR-0001: Configuration layering and clamp location

- **Status:** Accepted
- **Context:** We refactored config building into a base manager + final context clamps.
- **Decision:** `ScenarioConfigManager` returns a base config (scenario + env + platform caps).
  The test harness wrapper applies context clamps (fast‑feedback, resources, quality).
- **Consequences:**
  - Manager stays reusable and deterministic.
  - Tests retain full control over runtime behavior without risking double application.
  - If external users require clamped configs from the manager, we may later offer a
    `get_effective_config(ctx)` method and move clamps centrally (see "Option A" below).

## Alternatives considered
- **Centralize clamps inside the manager**: simpler API for external callers but risks
  double application for internal harnesses; harder to reason about test‑only behavior.
- **Feature‑gated clamps**: introduce a `full-framework` feature to toggle clamps; adds
  surface area and risks drift.

## How to revert
If we decide to centralize:
1. Move the wrapper's clamp function into `ScenarioConfigManager::get_effective_config(ctx)`.
2. Delete the wrapper clamps from `tests/test_configuration_scenarios.rs`.
3. Keep the "double‑clamp detector" tests and point them at the new entry point.

## Option A (future)
Introduce a public `get_effective_config(ctx)` that internally calls `resolve()` and then
applies the clamps in the library. Tests drop their local clamps.