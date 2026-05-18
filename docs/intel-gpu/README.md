# Intel GPU source-of-truth map

Status: proposed
Owner: intel-gpu/product
Created: 2026-05-18
Linked proposal: `docs/proposals/BITNET-PROP-0006-intel-gpu-productization.md`
Linked specs: `docs/specs/BITNET-SPEC-INTEL-GPU-ROUTE-CONTRACT.md`, `docs/specs/BITNET-SPEC-INTEL-GPU-DEVICE-IDENTITY.md`, `docs/specs/BITNET-SPEC-INTEL-GPU-BITNET-QK256.md`, `docs/specs/BITNET-SPEC-INTEL-GPU-DENSE-SLM.md`, `docs/specs/BITNET-SPEC-INTEL-GPU-QUALITY.md`, `docs/specs/BITNET-SPEC-INTEL-GPU-PERFORMANCE.md`, `docs/specs/BITNET-SPEC-INTEL-GPU-RESIDENCY.md`, `docs/specs/BITNET-SPEC-INTEL-GPU-STATUS-SURFACE.md`
Linked ADRs: n/a
Linked plan: `plans/intel-gpu/implementation-plan.md`
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: Documentation only; no route promotion.
Policy impact: Preserves existing proof-family and no-fallback boundaries.

## Purpose

Intel GPU support is a vendor-specific accelerator family, not a generic
"GPU works" bucket. This map exists so users, maintainers, and agents can tell
which Intel GPU lane is being discussed before any receipt, route matrix, status
page, or benchmark claim is interpreted.

This file does not promote a runtime route. It records the source-of-truth map
that future receipts and status surfaces must follow.

## Lanes

| Lane | Hardware | Runtime | First serious target | Claim family |
| --- | --- | --- | --- | --- |
| A770 native GPU | Arc A770 16GB discrete GPU | OpenCL first; Level Zero candidate later | BitNet I2_S/QK256 trusted partial acceleration | `intel-arc-a770-opencl` |
| Lunar Lake integrated GPU | Arc 140V / Core Ultra 7 258V | OpenVINO GPU and native OpenCL | Dense Qwen SLM candidate routing first; BitNet-adjacent parity later | `openvino-gpu` / `arc140v-opencl` |

## Route-family map

| Route family | Meaning | Current posture |
| --- | --- | --- |
| A770 native OpenCL | Discrete Arc A770 selected-device BitNet path. | OpenCL-first path for native BitNet QK256 proof; no generic Intel GPU claim. |
| A770 OpenVINO GPU | A770 through OpenVINO GPU runtime or graph APIs. | Reference runtime path only; not native OpenCL proof. |
| Arc 140V native OpenCL | Lunar Lake integrated GPU selected-device OpenCL lane. | Smoke/parity lane; not A770 proof and not BitNet QK256 support until fixtures and answer proof exist. |
| Arc 140V OpenVINO GPU | Lunar Lake OpenVINO GPU `GPU.X` dense SLM route. | Dense SLM candidate route; promotion remains exact-profile and quality-gated. |
| Intel NPU | OpenVINO NPU / Intel AI Boost lane. | Separate from every GPU lane. |
| CPU | Reference and comparator plate. | CPU proof cannot count as Intel GPU execution. |

## Non-interchangeable proof families

These boundaries must appear in specs, receipts, model coverage, status pages,
route matrices, and `receipts explain` output:

```text
A770 OpenCL proof is not Arc 140V proof.
Arc 140V OpenCL proof is not A770 proof.
OpenVINO GPU proof is not native OpenCL proof.
OpenVINO GPU proof is not NPU proof.
Intel GPU proof is not CUDA proof.
Dense SLM OpenVINO proof is not BitNet QK256/I2_S proof.
BitNet QK256 proof is not dense SLM proof.
CPU fallback cannot count as Intel GPU execution.
Generic OpenCL is not selected Intel GPU proof.
Generic GPU is not selected Intel GPU proof.
```

## Current truth anchors

- `docs/specs/intel-arc-a770-gpu-roadmap.md` defines A770 as the discrete
  OpenCL-first Intel GPU lane and keeps OpenVINO GPU reference evidence
  separate.
- `docs/specs/a770-bitnet-claim-boundary.md` defines the first A770 product
  claim as trusted partial BitNet I2_S acceleration, not full residency.
- `ci/hardware/device-kernel-routing.toml` remains the route-matrix truth for
  committed device/kernel proof; diagnostic rows must not be promoted without
  claim-grade receipts.
- `docs/tracking/campaigns/intel-258v-platform/CAMPAIGN.md` keeps Core Ultra 7
  258V CPU, Arc 140V GPU, and NPU proof labels separate.

## Documentation-only boundary

This source map adds no kernels, receipts, route promotions, model coverage
claims, speed claims, or residency claims. Future implementation PRs must still
produce selected-device receipts with `fallback_used=false` before claiming an
Intel GPU route.
