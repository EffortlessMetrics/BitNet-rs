# Lunar Lake Migration From BitNet-rs

This document is the handoff ledger for moving the Lunar Lake operating lane from
`EffortlessMetrics/BitNet-rs` into `EffortlessMetrics/bitnet-rs-swarm`.

The migration moves operating state, not old repository churn. This inventory PR
does not copy evidence receipts, run inference, change route policy, or alter any
BitNet QK256/I2_S behavior. It records the source cutoff, the accepted state to
carry, the current swarm gaps, and the next migration work.

## Source Cutoff

Source repository: `EffortlessMetrics/BitNet-rs`

Target repository: `EffortlessMetrics/bitnet-rs-swarm`

Primary Lunar Lake cutoff:

| Item | Source PR | Source merge | Notes |
| --- | ---: | --- | --- |
| `LNL258V-GOAL-AUDIT-013` | #6116 | `32f82e16889f20f264e69cfb90866bbb98c0ea74` | Latest no-inference Lunar Lake audit refresh after A770-006 closeout. |
| `LNL258V-GOAL-AUDIT-013` closeout | #6117 | `c2b0c6a55e3e643234d5b5b87132bd9558a1dcd6` | Tracker closeout for the audit refresh. |

Adjacent source anchors:

| Item | Source PR | Source merge | Migration status |
| --- | ---: | --- | --- |
| `LNL258V-BENCH-004` | #5289 | old-repo tracker anchor | Telemetry-context layer must be preserved. |
| `LNL258V-ROUTE-017` | #5777 | `320d794048d9f428bc7cdaf93a1085ef9bf457f6` | Profile-scoped route status. |
| `LNL258V-ROUTE-018` | #5780 | `c3261fa814d20408b8027ed86db7f9252b7f6325` | Route-promotion claim-boundary clarity. |
| `LNL258V-BITNET-INTAKE-001` | #5785 | `57263796fa4d382e004e13adb32d037192911b4a` | BitNet semantic-intake receipt surface. |
| `LNL258V-REG-007` | #5788 | old-repo tracker anchor | Strict regression v2 indexes BitNet semantic intake. |
| `A770-006` | #6110 | `e98d4fb6574087a83ed47e6dae7316d9dc12f40d` | Adjacent selected-device OpenCL `matmul_i2s` CPU parity evidence; not a Lunar Lake route or semantic trigger. |
| `A770-006` closeout | #6112 | `c5560df6f01a3dffb34249244731427446b3a547` | The audit artifact records this as adjacent evidence. |
| `A770-007` | #6113 | `d248ffd3077da72eb1be665f281aa1cd2a4a4f14` | Adjacent selected-device receipt identity evidence accepted after the audit refresh. |
| `A770-007` closeout | #6118 | `7f5cda90d318e77e3c13d7bbd6597f637271e175` | Carry as adjacent status only unless a later Lunar Lake audit refresh indexes it. |

The old audit artifact reports `source_revision =
c5560df6f01a3dffb34249244731427446b3a547`, because the audit was generated
after the A770-006 closeout and before the later A770-007 closeout.

## Accepted Operating State

The state to carry forward is:

- Dense Qwen CPU remains the default route id, correctness plate, and fallback
  route.
- OpenVINO GPU route promotion is profile-scoped. In the old-repo cutoff it is
  not a broad GPU or acceleration claim.
- OpenVINO NPU route state is profile-specific. Later old-repo receipts promote
  NPU only for `warm_resident`; cold one-off and `low_power` remain blocked.
- BitNet CPU is a separate protected reference path and is not proven by dense
  SLM evidence.
- Strict regression v2 indexes corpus v2, route-profile comparison,
  cold/warm benchmark, durability, telemetry context, and BitNet semantic intake.
- Shared BitNet semantic fixes from A770/CUDA/Mac/CPU lanes stale Lunar Lake
  BitNet evidence only after they merge to main and require a 258V rerun.
- `LNL258V-POWER-006` remains the active blocker for excellent all-profile
  support: no real battery-mode `low_power` telemetry, no usable energy-proxy
  power advantage, and no measured thermal readings.

## Current Swarm Gap

The swarm checkout already contains a broad Lunar Lake command and artifact
surface, but it is not at the old-repo cutoff. The following source artifacts
exist in the old repo and are missing from swarm at this inventory point:

- `lunar-lake-bitnet-semantic-source-changes.json`
- `lunar-lake-bitnet-semantic-intake.json`
- `lunar-lake-openvino-npu-cache-experiment.json`
- `lunar-lake-openvino-npu-resident-session.json`
- `lunar-lake-excellence-audit.json`
- `lunar-lake-goal-artifact-checklist.json`
- `lunar-lake-low-power-battery-telemetry-blocked.json`
- `lunar-lake-low-power-energy-proxy.json`
- `lunar-lake-power-profile-evidence.json`
- `lunar-lake-thermal-temperature-availability.json`

Several core artifacts already exist in swarm but differ from the old-repo
cutoff and must be refreshed in a controlled carry PR instead of treated as
current:

- `lunar-lake-operator-readiness.json`
- `lunar-lake-route-promotion.json`
- `lunar-lake-route-profile-comparison.json`
- `lunar-lake-cold-warm-profile-benchmark.json`
- `lunar-lake-power-thermal-context.json`
- `lunar-lake-durability-bundle.json`
- `lunar-lake-regression-bundle-v2.json`
- `lunar-lake-operator-comparison.json`
- `slm-answer-corpus-qwen25-cpu-corpus-v2.json`
- `slm-openvino-cpu-gpu-npu-corpus-v2.json`
- `lunar-lake-openvino-gpu-corpus-v2-diagnosis.json`
- `lunar-lake-openvino-npu-corpus-v2-diagnosis.json`
- `lunar-lake-openvino-npu-cold-start-diagnosis.json`

`cpu-reference-bundle-after-semantic-fix.json` is already byte-identical between
the old repo and swarm at this inventory point.

## Moved In This PR

This PR moves only the migration ledger:

- `docs/migrations/lunar-lake-from-bitnet-rs.md`
- `ci/hardware/intel-258v/MIGRATION_MANIFEST.json`

No Lunar Lake runtime receipt is copied in this inventory PR.

## Not Moved

The migration must not carry:

- stale historical receipts that are superseded by the latest old-repo operating
  artifacts;
- A770 diagnostic runtime machinery into Lunar Lake CLI paths;
- model binaries, local caches, generated build output, or host-local telemetry
  that is not committed evidence;
- dense SLM success as BitNet proof;
- selected-device A770 OpenCL proof as Lunar Lake route promotion or
  acceleration evidence;
- false swarm merge history for old BitNet-rs PRs.

## Claim Boundary

This migration inventory makes no new technical claim:

- no new inference;
- no new route promotion;
- no speedup, sustained-throughput, power-advantage, acceleration, native NPU,
  native OpenCL, or broad quality claim;
- no GPU/NPU promotion beyond what committed receipts already support;
- no BitNet QK256/I2_S behavior change;
- no A770 selected-device proof treated as a Lunar Lake semantic rerun trigger.

## Next Swarm Work

Recommended next PRs:

1. `SWARM-LNL258V-MIGRATE-002`: copy the latest accepted Lunar Lake operating
   artifacts and corpus fixtures from the old repo, then validate every moved
   JSON.
2. `SWARM-LNL258V-MIGRATE-003`: port only missing CLI surfaces needed to read
   those receipts, especially BitNet semantic intake and strict regression v2
   fail-closed behavior.
3. `SWARM-LNL258V-MIGRATE-004`: reconcile the swarm tracker with
   `migrated_from_bitnet_rs` events instead of synthetic swarm merge history.
4. `SWARM-LNL258V-REG-001`: prove the migrated state by running strict
   `lunar-lake regress` and `lunar-lake compare` in swarm.
5. Resume evidence work from `SWARM-LNL258V-POWER-006` / low-power battery-mode
   telemetry, then recompute route policy only from migrated and newly generated
   swarm evidence.
