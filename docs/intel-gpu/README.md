# Intel GPU source-of-truth map

Intel GPU support in BitNet-rs is a vendor-specific accelerator family, not a
single generic "GPU works" bucket. Every claim must name the selected device,
runtime API, model family, proof family, fallback state, quality status,
performance profile, and residency boundary.

## Route families

| Family | Hardware | Runtime | First target | Claim boundary |
| --- | --- | --- | --- | --- |
| A770 native OpenCL | Arc A770 16GB discrete GPU | OpenCL first, Level Zero only as a later candidate | BitNet I2_S/QK256 named-operation acceleration | `intel-arc-a770-opencl` |
| A770 OpenVINO reference | Arc A770 16GB discrete GPU | OpenVINO GPU | Runtime/reference comparison only | not native OpenCL proof |
| Arc 140V native OpenCL | Lunar Lake Arc 140V / Core Ultra 7 258V | OpenCL | Selected-device smoke/parity first | `arc140v-opencl` |
| Arc 140V OpenVINO GPU | Lunar Lake Arc 140V / Core Ultra 7 258V | OpenVINO GPU / GenAI | Dense Qwen SLM candidate routing first | `openvino-gpu` |
| Intel NPU | Lunar Lake Intel AI Boost NPU | OpenVINO NPU | Separate NPU proof lane | not GPU proof |
| CPU reference plate | Host CPU | scalar/AVX2/AVX-512/native runtime | Comparator and fallback detector | not GPU proof |

## Non-negotiable proof boundaries

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

## Claim ladder

Intel GPU route documents, receipts, route matrices, status surfaces, and
`receipts explain` output should use this ladder without collapsing adjacent
levels:

| Level | Meaning | Public claim |
| --- | --- | --- |
| `unsupported` | No valid route or proof. | none |
| `runtime_detected` | Device is visible. | detection only |
| `compile_smoke` | Kernel or graph compiles. | compile only |
| `kernel_smoke` | Tiny kernel or graph executes. | smoke only |
| `parity_tested` | CPU/GPU fixture parity exists. | fixture parity |
| `answer_ready` | Strict answer corpus or bounded useful answers pass. | answer route |
| `behavior_proven` | Prompt conditioning, stop/repetition, or long decode gates pass. | behavior route |
| `benchmark_candidate` | Timing fields are recorded. | diagnostic performance |
| `performance_proven` | Quality-gated profile beats a baseline with history. | exact-profile performance |
| `resident_proven` | A named op or phase is resident. | named residency only |
| `complete` | Required ops, residency, and server gates pass. | full route |

`performance_proven`, `resident_proven`, and `complete` are separate outcomes.
A route can have one without the others only when the matching receipts say so.

## Required receipt shape

Intel GPU receipts must make the proof family explicit. Fields may be nested to
match a command-specific schema, but the same facts must be present:

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

## Current route posture

- A770 native OpenCL is the discrete BitNet lane, but QK256, embedding, and
  LM-head rows remain diagnostic until claim-grade committed receipts, quality
  proof, benchmark/history proof, and residency boundaries are synchronized.
- A770 OpenVINO GPU is a reference runtime lane and does not prove native
  OpenCL kernels or BitNet QK256 execution.
- Arc 140V OpenVINO GPU dense SLM evidence is promising but remains candidate
  until corpus quality, profile timing applicability, direct-token limitations,
  benchmark-qualified advantage, and telemetry context gates are satisfied.
- Arc 140V native OpenCL currently belongs to selected-device smoke/parity work;
  it is not A770 proof and not BitNet QK256 proof.
- Intel NPU and CPU evidence remain separate proof families.

## Source-of-truth stack

- Proposal: `docs/proposals/BITNET-PROP-0006-intel-gpu-productization.md`.
- Shared plan: `plans/intel-gpu/implementation-plan.md`.
- Existing A770 rails: `docs/specs/intel-arc-a770-gpu-roadmap.md` and
  `docs/specs/a770-bitnet-claim-boundary.md`.
- Existing Lunar Lake rails: `docs/tracking/campaigns/intel-258v-platform/` and
  the Lunar Lake route/profile comparison receipts under `ci/hardware/intel-258v/`.
- Forthcoming Intel GPU route contracts: `BITNET-SPEC-INTEL-GPU-*` specs listed
  in `plans/intel-gpu/implementation-plan.md`.
