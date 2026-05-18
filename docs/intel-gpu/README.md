# Intel GPU source-of-truth map

## Purpose

Intel GPU support in BitNet-rs is a vendor-specific accelerator family, not a
single generic `gpu` bucket. This map keeps the discrete A770 lane, Lunar Lake
Arc 140V lane, OpenVINO GPU graph/runtime lane, Intel NPU lane, CPU reference
lane, CUDA lane, BitNet QK256 proof, and dense SLM proof separate until a
route-specific receipt proves otherwise.

This document is intentionally a source-of-truth alignment map only. It does
not promote a runtime route, add receipt requirements beyond the linked specs,
or change model coverage.

## Route-family boundaries

| Route family | Hardware | Runtime/API | First target | Current claim posture |
| --- | --- | --- | --- | --- |
| A770 native OpenCL | Arc A770 discrete GPU | OpenCL first; Level Zero may follow later | BitNet I2_S/QK256 named-op acceleration | Discrete BitNet lane governed by the A770 roadmap and claim-boundary spec. |
| A770 OpenVINO GPU reference | Arc A770 discrete GPU | OpenVINO GPU | Reference runtime comparison only | Reference evidence only; not native OpenCL proof. |
| Arc 140V native OpenCL | Lunar Lake Arc 140V integrated GPU | OpenCL | Native smoke/parity lane before BitNet-adjacent work | Integrated-GPU proof only; not A770 proof and not OpenVINO proof. |
| Arc 140V OpenVINO GPU | Lunar Lake Arc 140V integrated GPU | OpenVINO GPU / OpenVINO GenAI | Dense Qwen SLM candidate routing | Dense SLM candidate route; not BitNet QK256 proof and not NPU proof. |
| Intel NPU | Lunar Lake Intel AI Boost NPU | OpenVINO NPU | Separate NPU candidate evidence | NPU proof only; not GPU proof. |
| CPU reference plate | Host CPU | Scalar/AVX2/AVX-512/CPU OpenVINO as applicable | Reference correctness and comparison | Comparator evidence only; CPU fallback cannot satisfy GPU proof. |

## Non-conflation rules

These rules apply to specifications, receipts, status pages, route matrices,
model coverage, and future `receipts explain` output:

- A770 OpenCL proof is not Arc 140V proof.
- Arc 140V OpenCL proof is not A770 proof.
- OpenVINO GPU proof is not native OpenCL proof.
- OpenVINO GPU proof is not NPU proof.
- Intel GPU proof is not CUDA proof.
- Dense SLM OpenVINO proof is not BitNet QK256/I2_S proof.
- BitNet QK256 proof is not dense SLM proof.
- CPU fallback cannot count as Intel GPU execution.
- Generic OpenCL is not selected Intel GPU proof.
- Generic GPU is not selected Intel GPU proof.

## Shared claim-level ladder

Intel GPU routes use the following ladder unless a stricter route-specific spec
supersedes it:

| Level | Meaning | Public claim |
| --- | --- | --- |
| `unsupported` | No valid route or proof. | None. |
| `runtime_detected` | Device is visible. | Detection only. |
| `compile_smoke` | Kernel or graph compiles. | Compile only. |
| `kernel_smoke` | Tiny kernel or graph executes. | Smoke only. |
| `parity_tested` | CPU/GPU fixture parity exists. | Fixture parity. |
| `answer_ready` | Strict answer corpus or bounded useful answers pass. | Answer route. |
| `behavior_proven` | Prompt conditioning, stop/repetition, and long decode pass. | Behavior route. |
| `benchmark_candidate` | Timing fields are recorded. | Diagnostic performance. |
| `performance_proven` | Quality-gated profile beats a baseline with history. | Exact-profile performance. |
| `resident_proven` | A named operation or phase is resident. | Named residency only. |
| `complete` | All required ops, residency, and server gates pass. | Full route. |

`performance_proven`, `resident_proven`, and `complete` must remain separate.
A route may be performance-proven for one profile without being resident or
complete, and a named residency proof does not imply profile speedup.

## Receipt identity baseline

Every Intel GPU receipt should make the selected lane explicit enough for a
reader to reject generic GPU claims. At minimum, route-specific receipts should
record:

```json
{
  "requested_backend": "intel-arc-a770 | intel-arc-140v | openvino-gpu",
  "selected_backend": "intel-arc-a770-opencl | intel-arc-140v-opencl | openvino-gpu",
  "runtime_api": "opencl | openvino_genai | openvino_runtime | level_zero",
  "runtime_device": "GPU.0 | GPU.1 | OpenCL platform/device index",
  "fallback_used": false,
  "fallback_reason": null,
  "model_family": "bitnet | dense_slm | small_llm",
  "proof_family": "bitnet_qk256_opencl | dense_slm_openvino_gpu | arc140v_opencl_smoke",
  "device_identity": {
    "name": "...",
    "vendor": "Intel",
    "pci_device_id": "0x56A0 | 0x64A0 | ...",
    "driver_version": "...",
    "vram_or_shared_memory_bytes": 0
  },
  "claim_boundary": {
    "native_opencl_proof": true,
    "openvino_gpu_proof": false,
    "bitnet_qk256_proof": true,
    "dense_slm_proof": false,
    "full_residency_claim": false,
    "speedup_claim": false
  }
}
```

A route-specific spec may require more fields, stricter enums, or a different
claim-boundary shape, but it must preserve the same non-conflation semantics.

## Current source-of-truth stack

- A770 route identity and proof rails are defined by
  `docs/specs/intel-arc-a770-gpu-roadmap.md`.
- The first A770 BitNet product claim is bounded by
  `docs/specs/a770-bitnet-claim-boundary.md`.
- Lunar Lake CPU, Arc 140V GPU, and NPU proof labels are kept separate by
  `docs/tracking/campaigns/intel-258v-platform/CAMPAIGN.md` and its active
  goal.
- The shared Intel GPU rollout sequence starts in
  `plans/intel-gpu/implementation-plan.md`.

## Current route posture

- A770 native OpenCL is the discrete BitNet path. It must not be promoted from
  diagnostic or smoke evidence to answer/performance/residency claims without
  committed selected-device receipts and the A770 claim-boundary gates.
- A770 OpenVINO GPU is a reference runtime path. It does not prove native
  OpenCL QK256 execution.
- Arc 140V OpenCL is the integrated-GPU native smoke/parity path. It does not
  prove A770 behavior, OpenVINO GPU behavior, or BitNet inference.
- Arc 140V OpenVINO GPU is a dense SLM candidate route. It can become promoted
  only per exact profile after fallback-free quality, timing applicability,
  comparator, telemetry, and benchmark-history gates pass.
- Intel NPU evidence remains a separate OpenVINO NPU lane.
- CPU evidence remains the reference plate and comparator, not GPU proof.
