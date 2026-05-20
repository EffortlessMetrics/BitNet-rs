# BITNET-PROP-0006: Intel GPU productization

Status: proposed
Owner: intel-gpu/product
Created: 2026-05-18
Linked proposal: n/a
Linked specs: `docs/specs/BITNET-SPEC-INTEL-GPU-ROUTE-CONTRACT.md`, `docs/specs/BITNET-SPEC-INTEL-GPU-DEVICE-IDENTITY.md`, `docs/specs/BITNET-SPEC-INTEL-GPU-BITNET-QK256.md`, `docs/specs/BITNET-SPEC-INTEL-GPU-DENSE-SLM.md`, `docs/specs/BITNET-SPEC-INTEL-GPU-QUALITY.md`, `docs/specs/BITNET-SPEC-INTEL-GPU-PERFORMANCE.md`, `docs/specs/BITNET-SPEC-INTEL-GPU-RESIDENCY.md`, `docs/specs/BITNET-SPEC-INTEL-GPU-STATUS-SURFACE.md`
Linked ADRs: n/a
Linked plan: `plans/intel-gpu/implementation-plan.md`
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: Defines rationale only; no support claim promotion.
Policy impact: No exception.

## Thesis

Intel GPU support gives BitNet-rs a vendor-diverse accelerator path: native
OpenCL/Level-Zero-style kernels for packed BitNet on A770 and graph/runtime
acceleration for dense SLMs on Lunar Lake Arc 140V through OpenVINO GPU. The
value is not generic GPU detection; it is selected-device, selected-model,
receipt-backed local inference.

## Product lanes

| Lane | Product purpose | First serious target |
| --- | --- | --- |
| A770 native OpenCL | Discrete Intel GPU path for packed BitNet kernels. | BitNet I2_S/QK256 trusted partial acceleration. |
| Arc 140V OpenVINO GPU | Integrated Lunar Lake GPU graph/runtime path. | Dense Qwen SLM candidate routing and exact-profile promotion. |
| Arc 140V native OpenCL | Integrated native-kernel smoke/parity path. | BitNet-adjacent parity evidence after A770 path is grounded. |
| A770 OpenVINO GPU | Reference runtime comparison path. | Runtime/device comparison only unless separately specified. |

OpenVINO GPU is a runtime/graph lane, not native OpenCL proof. Dense SLM proof
and BitNet proof are separate. Performance is profile-specific. Full residency
is named-phase only until every required phase is proven.

## User value

Users should eventually be able to ask for an Intel GPU route and receive an
answer plus a receipt that explains:

- exact selected backend and runtime API;
- exact device identity;
- model family and proof family;
- whether fallback was used;
- quality result and failure taxonomy when relevant;
- timing profile and comparator boundary;
- transfer and residency status;
- not-claims and next proof needed.

## Non-interchangeable proof families

Intel GPU productization must preserve these boundaries:

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

## Product posture

A usable Intel GPU product is not a one-off fast run. It is a route-specific,
quality-gated, receipt-backed claim where maintainers can explain exactly what
ran and what did not run. A770 should mature toward named BitNet partial
acceleration. Arc 140V should mature first through dense OpenVINO GPU exact
profiles and separately through native OpenCL smoke/parity evidence.

## Non-goals

This proposal does not:

- promote any current route;
- claim generic Intel GPU support;
- claim speedup;
- claim full device residency;
- claim OpenVINO GPU as native OpenCL;
- transfer A770 proof to Arc 140V, or Arc 140V proof to A770;
- transfer dense SLM proof to BitNet QK256, or BitNet proof to dense SLMs.
