# AMD ROCm Implementation Plan

Status: active
Owner: rocm/productization
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0008](../../docs/proposals/BITNET-PROP-0008-amd-rocm-productization.md)
Linked specs: [ROCm route contract](../../docs/specs/BITNET-SPEC-ROCM-ROUTE-CONTRACT.md), [ROCm device identity](../../docs/specs/BITNET-SPEC-ROCM-DEVICE-IDENTITY.md), [ROCm kernel compile](../../docs/specs/BITNET-SPEC-ROCM-KERNEL-COMPILE.md), [ROCm BitNet QK256](../../docs/specs/BITNET-SPEC-ROCM-BITNET-QK256.md), [ROCm dense SLM](../../docs/specs/BITNET-SPEC-ROCM-DENSE-SLM.md), [ROCm quality](../../docs/specs/BITNET-SPEC-ROCM-QUALITY.md), [ROCm performance](../../docs/specs/BITNET-SPEC-ROCM-PERFORMANCE.md), [ROCm residency](../../docs/specs/BITNET-SPEC-ROCM-RESIDENCY.md), [ROCm status surface](../../docs/specs/BITNET-SPEC-ROCM-STATUS-SURFACE.md)
Linked ADRs: [BITNET-ADR-0005](../../docs/adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md)
Linked plan: n/a
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: registers and sequences AMD ROCm without runtime promotion
Policy impact: ordinary PRs remain source-only; live ROCm proof is opt-in

## Work Item: ROCM-DOCS-000

Status: ready
Campaign: `docs/tracking/campaigns/amd-rocm/active.toml`
Blocked by: n/a
Blocks: ROCM-PROBE-001 through ROCM-SERVER-025

### Goal

Add the AMD ROCm source-of-truth map, campaign tracker, proposal, specs, and
claim rails so follow-on PRs can productize ROCm without conflating HIP/ROCm,
CUDA, OpenCL, WGPU, CPU, OpenVINO, BitNet QK256, dense SLM, speed, residency,
or server proof families.

### Production Delta

Docs and planning only. This item registers the lane at `registered` claim
level 0 and does not change runtime code, kernels, model coverage, or CI live
hardware execution.

### Non-Goals

- Do not claim generic AMD GPU support.
- Do not claim ROCm from source text tests.
- Do not claim HIP compile from source embedding.
- Do not claim execution from compile smoke.
- Do not claim BitNet QK256 from dense SLM ROCm proof.
- Do not claim dense SLM from BitNet QK256 proof.
- Do not claim speedup without exact-profile review.
- Do not claim full residency without per-phase residency evidence.
- Do not add live ROCm hardware execution to ordinary PR CI.
- Do not touch CUDA, A770 OpenCL, Apple Metal, Intel NPU, or CPU kernels.

### Acceptance

- `docs/rocm/README.md` registers the ROCm lane and current claim boundary.
- `plans/rocm/README.md` and this plan define the PR sequence and validation.
- `docs/tracking/campaigns/amd-rocm/active.toml` and `CAMPAIGN.md` define the
  ready documentation item and follow-on sequence.
- `docs/proposals/BITNET-PROP-0008-amd-rocm-productization.md` explains why the
  lane exists.
- ROCm route, device, compile, BitNet QK256, dense SLM, quality, performance,
  residency, and status specs exist.
- `docs/specs/INDEX.md` and `docs/hardware/HARDWARE_MATRIX.md` list ROCm as a
  registered/scaffold lane without runtime, model, speed, residency, or server
  promotion.

### Proof Commands

```bash
cargo run --locked -p xtask --no-default-features -- campaign check amd-rocm
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

### Rollback

Revert the ROCm docs, plan, proposal, specs, campaign tracker, specs index row,
and hardware matrix row. No runtime migration is needed.

## Follow-On PR Queue

| Order | ID | Title | Scope | Acceptance summary |
| --- | --- | --- | --- | --- |
| 1 | ROCM-PROBE-001 | `probe(rocm): harden ROCm detection receipt` | CLI/receipt doctor surface | `bitnet rocm doctor --format json` works without ROCm, reports missing prerequisites, and claims no execution. |
| 2 | ROCM-PROBE-002 | `probe(rocm): selected-device identity` | strict selected device probe | `--device 0 --strict` resolves selected backend only with GPU identity and fails closed otherwise. |
| 3 | ROCM-KERNEL-003 | `rocm(kernel): add HIP compile-smoke harness` | hipcc/HIPRTC compile receipt | compile logs and GFX target recorded; no runtime launch or model claim. |
| 4 | ROCM-KERNEL-004 | `rocm(kernel): add tiny HIP runtime smoke` | vector-add HIP launch | selected AMD GPU, HIP launch success, CPU parity, fallback false; no BitNet claim. |
| 5 | ROCM-RECEIPT-005 | `rocm(receipts): validate ROCm receipt claim booleans` | receipt validation | rejects generic selected backend, strict fallback, ungated speed, ungated full residency, and proof-family leakage. |
| 6 | ROCM-QK256-006 | `test(rocm): QK256 scaled I2S-I8S fixtures` | CPU scalar oracle fixtures | locks rows, columns, tails, patterns, activations, scales; no ROCm execution required. |
| 7 | ROCM-QK256-007 | `feat(rocm): QK256 scaled I2S-I8S HIP GEMV` | HIP GEMV kernel | HIP launch, scalar parity, tails pass, fallback false; no model-level claim. |
| 8 | ROCM-QK256-008 | `feat(rocm): persistent ROCm BitNet context` | context/streams/modules/workspaces | weights uploaded once, no per-token upload, workspace reused. |
| 9 | ROCM-QK256-009 | `dispatch(rocm): route strict QK256 through ROCm` | strict dispatch wiring | strict `amd-rocm` fails closed when unavailable; QK256 ROCm invocations > 0; CPU fallback 0. |
| 10 | ROCM-QUALITY-010 | `test(rocm): one-token BitNet proof` | official BitNet one-token | generated token or divergence classified; fallback false; speedup false. |
| 11 | ROCM-QUALITY-011 | `test(rocm): deterministic BitNet answer corpus` | answer corpus | tiny corpus passes or exact blocker recorded; prompt/generated IDs and parity/divergence recorded. |
| 12 | ROCM-QUALITY-012 | `test(rocm): long-decode and warm-session proof` | decode/warm session | upload once, context reused, quality passes, fallback false, speedup false. |
| 13 | ROCM-DENSE-013 | `model(rocm): dense SLM artifact contract` | Qwen2.5 0.5B Q8_0 candidate | CPU sanity exists or is recorded; no BitNet QK256 claim. |
| 14 | ROCM-DENSE-014 | `test(rocm): dense single-linear parity` | dense linear fixture | CPU/ROCm parity, shape/tensor role recorded, fallback false, no full model claim. |
| 15 | ROCM-DENSE-015 | `test(rocm): dense one-token / short-decode` | dense decode proof | valid text, generated IDs where available, dense route, speed false, BitNet proof false. |
| 16 | ROCM-DENSE-016 | `test(rocm): dense warm-session` | dense session proof | model loaded once, weights uploaded once, multi-turn quality gate. |
| 17 | ROCM-BENCH-017 | `bench(rocm): ROCm phase timing profiles` | benchmark receipts | same-model CPU comparator and relevant CUDA/A770 comparator; speedup false. |
| 18 | ROCM-BENCH-018 | `bench(rocm): exact-profile speed review` | profile review | accept/reject each profile; no global speedup, unsupported model, or full residency claim. |
| 19 | ROCM-CLAIMS-019 | `claims(rocm): promote product CLI route` | model status/receipts UX | coverage/status/quickstart promotion only after quality, warm session, and benchmark review. |
| 20 | ROCM-SERVER-020 | `server(rocm): dense SLM exact-profile server smoke` | non-streaming server | selected concrete AMD ROCm backend, fallback false, quality pass, exact-profile server scope. |
| 21 | ROCM-SERVER-021 | `server(rocm): BitNet exact-profile server smoke` | BitNet server | official route, QK256 invocation count, fallback false, answer pass, exact-profile server scope. |

## Default Validation

Docs/spec PRs:

```bash
cargo run --locked -p xtask --no-default-features -- campaign check amd-rocm
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

Source-only ROCm PRs:

```bash
cargo test --locked -p bitnet-rocm --no-default-features
cargo check --locked -p bitnet-rocm --no-default-features
git diff --check
```

Live ROCm proof is opt-in only under workflow dispatch, main, scheduled,
release, explicit `rocm-ci`/`gpu-ci`/`model-validation` labels, or a hardware
campaign receipt. Ordinary PRs must not require ROCm hardware, HIP compile,
model downloads, or live device launches.
