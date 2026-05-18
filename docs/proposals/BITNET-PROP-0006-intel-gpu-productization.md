# BITNET-PROP-0006: Intel GPU productization

## Status

Proposed.

## Thesis

Intel GPU support gives BitNet-rs a vendor-diverse accelerator path: native
OpenCL/Level-Zero-style kernels for packed BitNet on A770 and graph/runtime
acceleration for dense SLMs on Lunar Lake Arc 140V through OpenVINO GPU. The
value is not generic GPU detection; it is selected-device, selected-model,
receipt-backed local inference.

## Product lanes

- A770 is the discrete native OpenCL BitNet lane. Its first product claim is
  trusted partial BitNet I2_S/QK256 acceleration for named operations, not all
  Intel GPUs and not full device residency.
- Arc 140V is the integrated Lunar Lake GPU lane. Its near-term native OpenCL
  role is smoke/parity; its first serious model route is dense SLM candidate
  routing through OpenVINO GPU.
- OpenVINO GPU is a runtime/graph lane, not native OpenCL proof.
- Dense SLM proof and BitNet QK256/I2_S proof are separate model families.
- Intel NPU proof is separate from OpenVINO GPU proof even when both use
  OpenVINO tooling.
- CPU evidence is the reference plate and fallback detector, not GPU execution.

## User value

Users should be able to ask which exact Intel route is available and receive an
answer that names the hardware, runtime, model family, fallback state, quality
result, timing profile, transfer/residency state, and not-claims. Maintainers
should be able to reject overbroad GPU claims from receipts and route matrices.

## Claim policy

Performance is profile-specific. A speed claim requires quality passing,
`fallback_used=false`, an applicable profile timing bundle, same-model and
same-profile comparator evidence, same-device history, and an accepted claim.
Full residency is named-phase only until every required phase is proven.

## Non-goals

This proposal does not promote any route, change kernels, change model coverage,
or treat Microsoft BitNet.cpp GPU evidence as proof of native BitNet-rs Intel GPU
execution. External BitNet.cpp material is context only unless BitNet-rs receipts
prove the selected route.
