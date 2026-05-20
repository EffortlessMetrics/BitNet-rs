# BITNET-SPEC-ROCM-KERNEL-COMPILE: HIP Kernel Compile Proof

Status: proposed
Owner: rocm/productization
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0008](../proposals/BITNET-PROP-0008-amd-rocm-productization.md)
Linked specs: [ROCm route contract](BITNET-SPEC-ROCM-ROUTE-CONTRACT.md), [ROCm device identity](BITNET-SPEC-ROCM-DEVICE-IDENTITY.md)
Linked ADRs: [BITNET-ADR-0005](../adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md)
Linked plan: [AMD ROCm implementation plan](../../plans/rocm/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: defines compile ladder; no compile proof yet
Policy impact: live compile remains opt-in unless feature-gated and available

## Purpose

Move ROCm proof beyond embedded source-text checks. Current tests can prove that
HIP source text exists and contains expected markers; they do not prove that the
source compiles, targets a specific GFX architecture, loads into a runtime
module, launches, or participates in model inference.

## Compile Levels

```text
source_embedded
hip_syntax_static
hipcc_compile_hostless
hipcc_compile_for_gfx_target
hiprtc_compile_runtime
tiny_kernel_launch
fixture_kernel_launch
model_route_launch
```

## Required Receipt Fields

```json
{
  "kernel_source_id": "qk256_i8s_scaled_gemv_hip",
  "compile_mode": "hipcc|hiprtc",
  "gfx_target": "gfx1100",
  "compile_status": "passed",
  "build_log": "",
  "runtime_launch": false,
  "fallback_used": false
}
```

Receipts must also include the ROCm route contract identity fields when compile
or launch proof is tied to a selected backend.

## Hard Rules

```text
source text contains "__global__" is not compile proof.
compile proof is not execution proof.
execution smoke is not model inference proof.
```

## Ordinary CI Boundary

Ordinary PRs may run source-text tests without ROCm hardware. HIP compile and
runtime smoke must be feature-gated or explicit opt-in when the HIP SDK and
selected AMD GPU are present. Missing ROCm prerequisites should produce blocked
or unavailable receipts, not panics or false success.
