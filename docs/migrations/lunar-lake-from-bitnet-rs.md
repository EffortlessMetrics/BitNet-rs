# Lunar Lake BitNet-rs to bitnet-rs-swarm Migration Inventory

Migration item: `SWARM-LNL258V-MIGRATE-001`

Source repo: `EffortlessMetrics/BitNet-rs`

Target repo: `EffortlessMetrics/bitnet-rs-swarm`

Created: `2026-05-21T15:23:58Z`

## Purpose

This inventory records the Lunar Lake operating state carried into
`bitnet-rs-swarm`. It is a handoff ledger, not new validation evidence.

The migrated lane should be read from the current swarm artifacts under
`ci/hardware/intel-258v/2026-05-08/`, the corpus fixture under
`ci/quality/lunar-lake-answer-corpus-v2.yaml`, and the Intel 258V campaign
tracker.

## Source Cutoff

The old-repo operating state is anchored by the latest imported Lunar Lake
route, regression, telemetry, and audit receipts:

- `LNL258V-BENCH-004`: telemetry-context capture and refreshed benchmark,
  durability, regression-v2, and comparison artifacts.
- `LNL258V-ROUTE-017`: profile-scoped route status with GPU promoted only for
  scoped profiles at that point.
- `LNL258V-ROUTE-018`: explicit route-promotion scope and claim-boundary
  regression/comparison fields.
- `LNL258V-BITNET-INTAKE-001`: shared BitNet semantic-intake receipt.
- `LNL258V-REG-007`: strict regression gates on BitNet semantic-intake
  freshness.
- `LNL258V-GOAL-AUDIT-013`: post-A770-006 audit refresh, recording A770-006 as
  adjacent selected-device OpenCL parity evidence, not a Lunar Lake route or
  BitNet semantic trigger.

Swarm then continued the lane with later no-inference and evidence-indexing
work. The current committed Lunar Lake audit cutoff is
`LNL258V-GOAL-AUDIT-023` from PR #253, with tracker closeout PR #254 and
`source_revision = 116392b13009ef0ddae8223b345e64592d79e504`.

That cutoff records adjacent CUDA/Qwen3 phase-trace PRs #247 and #250 plus
`LNL258V-GOAL-AUDIT-022` closeout PR #248 as non-Lunar-Lake-route evidence.
They do not change route policy, promotion status, inference evidence, power
claims, or BitNet behavior.

## Carried State

The current swarm lane carries:

- operator readiness: `lunar-lake-operator-readiness.json`
- route policy: `lunar-lake-route-promotion.json`
- route profile comparison: `lunar-lake-route-profile-comparison.json`
- cold/warm benchmark: `lunar-lake-cold-warm-profile-benchmark.json`
- telemetry context: `lunar-lake-power-thermal-context.json`
- durability evidence: `lunar-lake-durability-bundle.json`
- strict regression v2: `lunar-lake-regression-bundle-v2.json`
- operator comparison: `lunar-lake-operator-comparison.json`
- BitNet semantic source/intake receipts:
  `lunar-lake-bitnet-semantic-source-changes.json` and
  `lunar-lake-bitnet-semantic-intake.json`
- dense Qwen CPU/OpenVINO corpus v2 evidence and diagnoses
- NPU cold-start, cache, and resident-session receipts
- low_power battery-plan and AC-blocked telemetry receipts
- no-inference excellence audit and goal artifact checklist

The machine-readable item list is
`ci/hardware/intel-258v/MIGRATION_MANIFEST.json`.

## Current Operating State

The latest committed audit marks the lane as operator-ready and profile-aware,
but not complete:

- Qwen CPU remains the default and correctness/control route where no promoted
  profile route exists.
- OpenVINO GPU is profile-promoted for interactive profiles recorded in the
  current route ledger.
- OpenVINO NPU is promoted only for `warm_resident`.
- BitNet CPU remains the protected reference path.
- `low_power` remains blocked because battery-mode telemetry, energy-proxy
  evidence, measured thermal readings, and benchmark-qualified power advantage
  are missing.

## Not Moved

This migration inventory intentionally does not carry:

- stale historical receipt churn that is not referenced by the latest route,
  regression, comparison, audit, or manifest surfaces;
- local model binaries, model caches, OpenVINO IR directories, or machine-local
  virtual environments;
- old branch history as tracker truth;
- A770 diagnostic runtime machinery inside Lunar Lake command surfaces;
- dense SLM evidence as BitNet QK256/I2_S proof.

## Claim Boundary

This inventory makes no new claim:

- no new inference was run;
- no route is promoted or demoted;
- no speedup or power advantage is claimed;
- no native OpenCL, native NPU, or broad accelerator claim is made;
- no dense SLM receipt is treated as BitNet proof;
- no BitNet QK256/I2_S behavior is changed.

## Next Swarm Work

The active blocker remains `LNL258V-POWER-006`: collect real battery-mode
`low_power` telemetry and energy-proxy evidence on the Core Ultra 7 258V laptop.

That work must use the committed runbook and plan receipts before changing route
policy:

- `docs/hardware/intel-258v-low-power-battery-runbook.md`
- `lunar-lake-low-power-battery-plan.json`
- `lunar-lake-low-power-battery-telemetry-blocked.json`

`low_power` may remain blocked, or it may be promoted only if answer gates,
fallback-free route identity, stable timing, and benchmark-qualified power
advantage all pass.
