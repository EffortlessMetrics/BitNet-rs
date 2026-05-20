# Intel Arc A770 Migration From BitNet-rs

This document is the handoff ledger for moving the Intel Arc A770 operating
lane from `EffortlessMetrics/BitNet-rs` into
`EffortlessMetrics/bitnet-rs-swarm`.

The migration moves operating state, not old repository churn. This inventory PR
does not copy A770 receipts, run inference, change OpenCL kernels, change BitNet
QK256/I2_S behavior, or promote any A770 claim. It records the source cutoff,
the accepted source state to carry, current swarm gaps, and the next migration
work.

## Source Cutoff

Source repository: `EffortlessMetrics/BitNet-rs`

Target repository: `EffortlessMetrics/bitnet-rs-swarm`

Primary A770 cutoff:

| Item | Source PR | Source merge | Notes |
| --- | ---: | --- | --- |
| `A770-005` | #6072 | `b995c5110866d1920538fff2d9019fcd3803e3c5` | Selected-device tiny OpenCL smoke on real A770 with `fallback_used=false`; no BitNet inference, QK256, speed, residency, or quality claim. |
| `A770-005` closeout | #6079 | `70b79c78228a4da323aa216f74562e0c54ce005f` | Tracker closeout for the selected OpenCL smoke. |
| `A770-006` | #6110 | `e98d4fb6574087a83ed47e6dae7316d9dc12f40d` | Selected-device OpenCL `matmul_i2s` CPU parity receipt; not official BitNet QK256 production semantics. |
| `A770-006` closeout | #6112 | `c5560df6f01a3dffb34249244731427446b3a547` | Tracker closeout for A770-006. |
| `A770-007` | #6113 | `d248ffd3077da72eb1be665f281aa1cd2a4a4f14` | Selected-device receipt identity for the validated smoke/parity path. |
| `A770-007` closeout | #6118 | `7f5cda90d318e77e3c13d7bbd6597f637271e175` | Tracker closeout for A770-007. |

Adjacent diagnostic source anchor:

| Item | Source PR | Source merge | Migration status |
| --- | ---: | --- | --- |
| Durable A770 layer-trace diagnostics | #5946 | `c37a9ecc3ca588b34498ac4209c84fdfa731ba3b` | Diagnostic tooling only. Carry by content audit when swarm resumes CPU/reference numerical attribution. |

## Accepted Operating State

The state to carry forward is:

- A770 is an OpenCL-first selected-device lane, not generic OpenCL proof and not
  Intel NPU, Arc 140V, CUDA, OpenVINO GPU, Metal, or CPU proof.
- Tiny selected-device OpenCL execution exists in the source repo for the A770
  route with `fallback_used=false`.
- Minimal `matmul_i2s` CPU/OpenCL parity exists in the source repo for the
  selected A770 route.
- Selected-device receipt identity exists in the source repo for the validated
  A770 smoke/parity path.
- These receipts remain diagnostic and do not prove BitNet inference, official
  QK256 production semantics, semantic quality, performance, residency, or
  completion.
- Older A770 diagnostic/runtime PRs remain content-bearing until exact
  successor, duplicate, clean port, historical capture, or explicit content
  rejection is proven.

## Current Swarm Gap

The swarm checkout already contains an A770 campaign skeleton, route docs, and a
kernel capability matrix, but it is not at the old-repo A770 cutoff. At this
inventory point:

- `docs/tracking/campaigns/intel-a770/active.toml` still starts at
  `A770-003` as `ready`.
- `docs/tracking/campaigns/intel-a770/generated/status.md` still shows only
  `A770-003`.
- `ci/hardware/amd-5700x-intel-a770/a770-kernel-capability-matrix.json` exists
  but differs from the source cutoff.
- the source A770-005, A770-006, and A770-007 receipts are missing from swarm.
- the source A770 selected-device receipt identity report is missing from swarm.
- the source A770 campaign events and closeout state are missing from swarm.

These gaps should be closed by controlled carry PRs. Do not synthesize swarm
merge history for old BitNet-rs PRs.

## Moved In This PR

This PR moves only the migration ledger:

- `docs/migrations/a770-from-bitnet-rs.md`
- `ci/hardware/amd-5700x-intel-a770/MIGRATION_MANIFEST.json`

No A770 runtime receipt is copied in this inventory PR.

## Not Moved

The migration must not carry:

- stale historical receipts without source-cutoff mapping;
- A770 diagnostic runtime machinery without a content audit;
- broad old PR closures or stale-stack disposition;
- model binaries, local caches, generated build output, or host-local telemetry
  that is not committed evidence;
- selected-device OpenCL smoke as BitNet inference proof;
- `matmul_i2s` parity as official BitNet QK256 production proof;
- selected-device identity as a performance, residency, or semantic-quality
  claim.

## Claim Boundary

This migration inventory makes no new technical claim:

- no new inference;
- no new OpenCL execution;
- no new route promotion;
- no speedup, sustained-throughput, acceleration, native OpenCL product, or
  broad quality claim;
- no BitNet QK256/I2_S behavior change;
- no A770 semantic quality, selected attention, resident KV, attention score
  residency, softmax residency, value-mix residency, full residency,
  performance, or completion claim.

The accepted source evidence is diagnostic selected-device evidence only until
swarm carries it, verifies it, and advances the later gates.

## Next Swarm Work

Recommended next PRs:

1. `SWARM-A770-MIGRATE-002`: copy the latest accepted A770 receipts, capability
   matrix, selected-device identity report, and campaign events from the source
   cutoff; validate every moved JSON.
2. `SWARM-A770-MIGRATE-003`: reconcile the swarm A770 tracker with
   `migrated_from_bitnet_rs` events instead of synthetic swarm merge history.
3. `SWARM-A770-DIAG-001`: content-audit #5946 and the older A770 diagnostic
   stack for exact missing numerical-attribution tooling before porting any
   diagnostic code.
4. `SWARM-A770-QK256-001`: continue real A770 QK256 OpenCL implementation only
   after the migrated selected-device state is available in swarm.
5. `SWARM-A770-BEHAVIOR-001`: prove CPU/reference behavior and broad
   multi-token prompts before A770 performance or residency claims.
