# AMD ROCm Lane

Status: registered
Owner: rocm/productization
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0008](../proposals/BITNET-PROP-0008-amd-rocm-productization.md)
Linked specs: [ROCm route contract](../specs/BITNET-SPEC-ROCM-ROUTE-CONTRACT.md), [ROCm device identity](../specs/BITNET-SPEC-ROCM-DEVICE-IDENTITY.md), [ROCm kernel compile](../specs/BITNET-SPEC-ROCM-KERNEL-COMPILE.md), [ROCm BitNet QK256](../specs/BITNET-SPEC-ROCM-BITNET-QK256.md), [ROCm dense SLM](../specs/BITNET-SPEC-ROCM-DENSE-SLM.md), [ROCm quality](../specs/BITNET-SPEC-ROCM-QUALITY.md), [ROCm performance](../specs/BITNET-SPEC-ROCM-PERFORMANCE.md), [ROCm residency](../specs/BITNET-SPEC-ROCM-RESIDENCY.md), [ROCm status surface](../specs/BITNET-SPEC-ROCM-STATUS-SURFACE.md)
Linked ADRs: [BITNET-ADR-0005](../adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md)
Linked plan: [ROCm implementation plan](../../plans/rocm/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: registers AMD ROCm as a docs-only hardware lane at claim level 0
Policy impact: no CI or runtime policy exception

## Purpose

This directory is the source-of-truth map for AMD ROCm productization. It
registers AMD ROCm as BitNet-rs's selected-device AMD GPU lane without claiming
that any AMD GPU, HIP compile, HIP runtime launch, BitNet QK256 model route,
dense SLM route, performance profile, residency class, or server profile is
ready.

The lane starts from the existing `bitnet-rocm` crate and its source-text HIP
kernel checks, then sequences proof toward a product route only after concrete
HIP/ROCm runtime identity and selected AMD GPU identity are recorded.

## First User-Facing End State

The desired user path is intentionally receipt-backed:

```bash
bitnet rocm doctor --format json
bitnet model status --device amd-rocm
bitnet ask --device amd-rocm --model microsoft-bitnet-b1.58-2B-4T-i2s "..."
bitnet chat --device amd-rocm --model qwen2.5-0.5b-instruct-q8_0
bitnet bench --device amd-rocm --model <model-id> --profile warm_session
bitnet receipts explain --latest
```

None of those commands are promoted by this registration document. They are the
end state that later PRs must earn through the implementation plan.

## Required Receipt Questions

Every ROCm receipt must eventually answer:

- Which AMD GPU?
- Which ROCm and HIP runtime?
- Which GFX target?
- Which model family?
- Which kernel or graph route?
- Was fallback used?
- Did answer quality pass?
- Did CPU/ROCm parity pass?
- Was speedup accepted for this exact profile?
- Which not-claims still apply?

## Candidate Selected Backends

The first real product target must be one exact route, chosen from available
hardware and AMD support status, for example:

```text
amd-radeon-rx-7900-xtx-rocm
amd-radeon-rx-7900-xt-rocm
amd-radeon-rx-7800-xt-rocm
amd-radeon-pro-w7900-rocm
amd-instinct-mi300x-rocm
```

`amd`, `gpu`, `rocm`, `hip`, and `radeon` are forbidden as standalone proof
labels because they do not identify a selected backend.

## Current Registered State

| Area | Current status | Claim boundary |
| --- | --- | --- |
| ROCm source and detection scaffolding | exists in `bitnet-rocm` | source/detection only |
| HIP kernel source text checks | exists | not compile proof |
| Selected AMD GPU runtime proof | missing | no selected backend claim |
| HIP compile smoke | missing | no compile claim |
| HIP execution smoke | missing | no execution claim |
| BitNet QK256 ROCm parity | missing | no BitNet route claim |
| Dense SLM ROCm proof | missing | no dense route claim |
| Answer-ready ROCm route | missing | no user-facing support claim |
| Speed, residency, and server readiness | missing | no speed, full-residency, or server claim |

## Permanent Not-Claims

Early ROCm docs, receipts, and status surfaces must preserve these not-claims
until exact-profile proof promotes a narrower claim:

```text
not_cuda
not_opencl
not_wgpu
not_openvino_gpu
not_cpu_acceleration
not_full_rocm_residency
not_generic_amd_gpu_support
not_all_rocm_versions
not_all_radeon_support
not_bitnet_qk256_proof_from_dense_slm
not_dense_slm_proof_from_bitnet
not_speedup_without_profile_review
not_server_ready_without_server_receipt
```

## External Authority Boundary

AMD's current ROCm Linux documentation says the supported GPU table is the
official support boundary and that unlisted GPUs are not officially supported.
It also publishes LLVM targets such as `gfx1100`, `gfx1101`, `gfx1200`,
`gfx1201`, and `gfx942` for selected Radeon, Radeon PRO, and Instinct devices.
The ROCm status surface must record the exact ROCm/HIP version and support
status seen at proof time instead of treating a visible HIP device as generic
AMD GPU support.

Linux ROCm proof and Windows HIP SDK proof are separate evidence families.
