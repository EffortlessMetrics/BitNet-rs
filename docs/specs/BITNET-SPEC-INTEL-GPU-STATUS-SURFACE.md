# BITNET-SPEC-INTEL-GPU-STATUS-SURFACE

Status: proposed
Owner: intel-gpu/product
Created: 2026-05-18
Linked proposal: `docs/proposals/BITNET-PROP-0006-intel-gpu-productization.md`
Linked specs: `docs/specs/BITNET-SPEC-INTEL-GPU-ROUTE-CONTRACT.md`, `docs/specs/BITNET-SPEC-INTEL-GPU-DEVICE-IDENTITY.md`, `docs/specs/BITNET-SPEC-INTEL-GPU-QUALITY.md`, `docs/specs/BITNET-SPEC-INTEL-GPU-PERFORMANCE.md`, `docs/specs/BITNET-SPEC-INTEL-GPU-RESIDENCY.md`
Linked ADRs: n/a
Linked plan: `plans/intel-gpu/implementation-plan.md`
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: Defines future UX; no route promotion.
Policy impact: No exception.

## Purpose

Intel GPU status surfaces must make proof-family truth visible to users and
agents without requiring them to inspect raw receipts.

## Future commands

Status surfaces should eventually include:

```bash
bitnet model status --device intel-arc-a770-opencl
bitnet model status --device intel-arc-140v-openvino-gpu
bitnet receipts explain <receipt>
bitnet lunar-lake routes --format json
bitnet gpu doctor --vendor intel
```

## Required output fields

Human and JSON output should include:

```text
route id
proof family
claim level
selected backend
runtime API
quality status
performance status
residency status
server status
not-claims
next required proof
```

## Required explanations

`receipts explain` and status pages must distinguish at least:

- A770 native OpenCL BitNet proof;
- A770 OpenVINO GPU reference proof;
- Arc 140V OpenVINO GPU dense SLM proof;
- Arc 140V native OpenCL smoke/parity proof;
- Intel NPU proof;
- CPU reference proof;
- CUDA proof;
- unsupported, candidate, exact-profile, and full-route states.

## Not-claims

Every Intel GPU status surface must say when a route is not generic Intel GPU
support, not native OpenCL proof, not OpenVINO GPU proof, not NPU proof, not
BitNet QK256 proof, not dense SLM proof, not a speedup claim, or not full
residency.
