# BITNET-SPEC-ROCM-ROUTE-CONTRACT: AMD ROCm Route Contract

Status: proposed
Owner: rocm/productization
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0008](../proposals/BITNET-PROP-0008-amd-rocm-productization.md)
Linked specs: [ROCm device identity](BITNET-SPEC-ROCM-DEVICE-IDENTITY.md), [ROCm kernel compile](BITNET-SPEC-ROCM-KERNEL-COMPILE.md)
Linked ADRs: [BITNET-ADR-0005](../adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md)
Linked plan: [AMD ROCm implementation plan](../../plans/rocm/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: defines labels; no runtime promotion
Policy impact: no CI policy exception

## Purpose

Define the ROCm backend labels, route IDs, proof families, and receipt identity
fields that keep AMD ROCm proof separate from CUDA, OpenCL, WGPU, CPU,
OpenVINO, and generic GPU support.

## Required Route IDs

```text
amd_rocm_runtime_probe
amd_rocm_hip_compile_smoke
amd_rocm_tiny_kernel_smoke
amd_rocm_bitnet_qk256_gemv
amd_rocm_bitnet_qk256_gemm
amd_rocm_dense_slm_gemv
amd_rocm_dense_slm_warm_session
amd_rocm_server_exact_profile
```

## Backend Labels

Receipts must preserve the requested backend, concrete selected backend,
runtime API, runtime stack, and target ISA:

```text
requested_backend = "amd-rocm"
selected_backend = "amd-radeon-rx-7900-xtx-rocm"
runtime_api = "hip"
runtime_stack = "rocm"
gfx_target = "gfx1100"
```

`selected_backend` must be a concrete route for the selected device. The values
below are examples only and must not be claimed until the device is actually
resolved and proven:

```text
amd-radeon-rx-7900-xtx-rocm
amd-radeon-rx-7900-xt-rocm
amd-radeon-rx-7800-xt-rocm
amd-radeon-pro-w7900-rocm
amd-instinct-mi300x-rocm
```

## Forbidden Standalone Proof Labels

Do not use these labels by themselves as proof labels:

```text
amd
gpu
rocm
hip
radeon
```

They are too generic to identify a selected backend, route, runtime, device,
model family, or proof stage.

## Proof Families

ROCm proof must use ROCm-specific families such as:

```text
rocm_bitnet_qk256_hip
rocm_dense_slm_hip
rocm_dense_llm_hip
rocm_source_compile_smoke
rocm_runtime_smoke
rocm_external_reference
```

CUDA, A770 OpenCL, OpenVINO GPU, Apple Metal, CPU AVX2/AVX-512, NPU, and
generic GPU proof families are not interchangeable with these families.

## Required Receipt Fields

```json
{
  "requested_backend": "amd-rocm",
  "selected_backend": "amd-radeon-rx-7900-xtx-rocm",
  "runtime_api": "hip",
  "runtime_stack": "rocm",
  "rocm_version": "7.2.3",
  "hip_version": "7.2.53211",
  "gfx_target": "gfx1100",
  "device": {
    "name": "AMD Radeon RX 7900 XTX",
    "arch": "RDNA3",
    "pci_bus_id": "...",
    "vram_bytes": 0,
    "compute_units": 0,
    "wavefront_size": 64
  },
  "fallback_used": false,
  "fallback_reason": null,
  "proof_family": "rocm_bitnet_qk256_hip"
}
```

If any field is not available, the receipt must say so explicitly and must not
promote a selected-backend claim that depends on it.

## Claim Levels

| Level | Meaning | Public claim |
| ---: | --- | --- |
| 0 | `registered` | ROCm lane exists as docs/status only. |
| 1 | `runtime_detected` | ROCm/HIP and selected AMD GPU visible. |
| 2 | `source_compile_smoke` | HIP source compiles for target GFX. |
| 3 | `runtime_smoke` | Tiny HIP kernel executes on selected GPU. |
| 4 | `parity_tested` | CPU/ROCm fixture parity for selected op. |
| 5 | `answer_candidate` | Model answers bounded prompts on ROCm. |
| 6 | `answer_ready` | Corpus/profile quality passes. |
| 7 | `benchmark_candidate` | Timings recorded, speed not accepted. |
| 8 | `performance_proven` | Exact-profile speed/power accepted. |
| 9 | `resident_proven` | Named phases resident on ROCm. |
| 10 | `product_cli_ready` | ask/chat/status/receipts are user-ready. |
| 11 | `server_exact_profile_ready` | Bounded server route proven. |
