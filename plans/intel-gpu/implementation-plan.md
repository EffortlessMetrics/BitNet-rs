# Intel GPU implementation plan

## Purpose

Lay down the Intel GPU source-of-truth, specs, proof rails, and user-visible
status surfaces so BitNet-rs can productize Intel GPU support without conflating
A770 native OpenCL, Arc 140V native OpenCL, OpenVINO GPU, Intel NPU, CPU, CUDA,
BitNet QK256, and dense SLM proof families.

## Scope controls

- Documentation/spec PRs do not promote runtime claims.
- Documentation/spec PRs do not change QK256 kernels, OpenCL kernels, model
  coverage, route matrix support levels, or receipt validation behavior unless a
  specific later work item says so.
- Runtime PRs must link to the relevant route spec and preserve selected backend
  identity, runtime API, fallback truth, proof family, quality state, timing
  profile, residency boundary, and not-claims.
- A770 OpenCL, Arc 140V OpenCL, OpenVINO GPU, OpenVINO NPU, CPU, and CUDA remain
  separate proof families.

## Phase 0: source-of-truth alignment

### PR 0: `docs(intel-gpu): add Intel GPU source-of-truth map`

Add:

- `docs/intel-gpu/README.md`
- `plans/intel-gpu/README.md`
- `plans/intel-gpu/implementation-plan.md`

Update:

- `docs/specs/INDEX.md`
- `docs/tracking/campaigns/intel-a770/active.toml`
- `docs/tracking/campaigns/intel-258v-platform/active.toml`
- generated campaign docs if the campaign generator requires it

Acceptance:

- Docs only.
- No route promotion.
- No receipt schema or receipt artifact changes.
- A770 native OpenCL is documented as the discrete BitNet path.
- A770 OpenVINO GPU is documented as reference runtime evidence only.
- Arc 140V OpenVINO GPU is documented as the dense SLM candidate route.
- Arc 140V native OpenCL is documented as smoke/parity first.
- NPU remains a separate OpenVINO NPU lane.
- CPU remains a reference plate, not GPU proof.

Proof commands:

```bash
cargo run --locked -p xtask --no-default-features -- campaign check intel-a770
cargo run --locked -p xtask --no-default-features -- campaign check intel-258v-platform
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

## Phase 1: Intel GPU specs

Add proposal/spec artifacts without runtime promotion:

1. `docs/proposals/BITNET-PROP-0006-intel-gpu-productization.md`
2. `docs/specs/BITNET-SPEC-INTEL-GPU-ROUTE-CONTRACT.md`
3. `docs/specs/BITNET-SPEC-INTEL-GPU-DEVICE-IDENTITY.md`
4. `docs/specs/BITNET-SPEC-INTEL-GPU-BITNET-QK256.md`
5. `docs/specs/BITNET-SPEC-INTEL-GPU-DENSE-SLM.md`
6. `docs/specs/BITNET-SPEC-INTEL-GPU-QUALITY.md`
7. `docs/specs/BITNET-SPEC-INTEL-GPU-PERFORMANCE.md`
8. `docs/specs/BITNET-SPEC-INTEL-GPU-RESIDENCY.md`
9. `docs/specs/BITNET-SPEC-INTEL-GPU-STATUS-SURFACE.md`

Acceptance:

- Backend labels and route IDs are concrete.
- Receipt identity fields are normalized across A770, Arc 140V, OpenCL, Level
  Zero, OpenVINO GPU, and telemetry contexts.
- The BitNet QK256 spec rejects toy I2_S kernels as official QK256 proof.
- Dense SLM OpenVINO GPU promotion is exact-profile only.
- Quality, performance, and residency are separate gates.
- Status surfaces expose not-claims and next required proof.

## Phase 2: A770 route truth and proof ledger

Reconcile `ci/hardware/amd-5700x-intel-a770/**`,
`ci/hardware/amd-5700x-intel-a770/verify-receipts.py`,
`ci/hardware/device-kernel-routing.toml`, and the A770 campaign.

Acceptance:

- If claim-grade receipts are committed, route rows list them and use the
  correct claim level.
- If claim-grade receipts are absent, route rows remain diagnostic and document
  missing artifacts.
- Strict A770 claims require `selected_backend=intel-arc-a770-opencl`,
  `fallback_used=false`, proof receipts, and preserved not-claims.
- Speedup or full residency claims fail validation unless accepted gates exist.

## Phase 3: A770 native OpenCL productization

Sequence selected-device identity refresh, QK256 scaled I2_S/I8_S parity
fixtures, claim-grade QK256 OpenCL receipts, deterministic BitNet answer
behavior, quality-gated profile timings, and trusted partial-acceleration
promotion.

Allowed first product claim, only after gates pass:

```text
Official BitNet b1.58 I2_S/QK256 has trusted partial A770 OpenCL acceleration
for named operations.
```

Not allowed:

- all Intel GPUs
- all OpenCL devices
- dense SLM/Gemma support
- selected-attention residency
- KV residency
- full support-op residency
- full device residency
- speedup without quality-gated same-device profile history

## Phase 4: Lunar Lake Arc 140V OpenVINO GPU route

Classify OpenVINO GPU corpus-v2 failures, add profile-specific timing, then
promote only exact profiles where quality passes, fallback is false, profile
timing is applicable, and a benchmark-qualified CPU/UX/power advantage is
accepted.

Not allowed:

- native OpenCL proof from OpenVINO GPU
- NPU proof from OpenVINO GPU
- BitNet QK256 proof from dense SLM OpenVINO GPU
- broad Arc 140V GPU support claims

## Phase 5: Arc 140V native OpenCL BitNet-adjacent lane

Refresh selected-device native OpenCL parity, add BitNet QK256 candidate
fixtures, and decide whether Arc 140V should pursue a native BitNet route before
A770 reaches trusted partial acceleration.

Default posture:

```text
A770 owns native BitNet OpenCL first; Arc 140V remains smoke/parity and dense
OpenVINO GPU candidate unless evidence says otherwise.
```

## Phase 6: shared Intel GPU UX

Add user-visible status and explanation surfaces only after the route contracts
exist:

- `bitnet receipts explain <receipt>` route-family explanation.
- `docs/status/INTEL_GPU_CAPABILITY_MATRIX.md`.
- `bitnet gpu doctor --vendor intel --format json`.

Each surface must show route ID, proof family, claim level, selected backend,
runtime API, quality status, performance status, residency status, server status,
not-claims, and next required proof.
