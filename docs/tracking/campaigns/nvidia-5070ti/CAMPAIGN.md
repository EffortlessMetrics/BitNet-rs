# NVIDIA 5070 Ti Campaign

Campaign ID: `nvidia-5070ti`

Status: complete

## Objective

Maintain the completed RTX 5070 Ti CUDA BitNet proof lane with selected-device receipts and no CPU, OpenCL, WGPU, dense CUDA, or generic GPU conflation.

## End State

- RTX 5070 Ti CUDA backend identity is distinct from generic CUDA and WGPU.
- CUDA and NVML probe facts are recorded before kernel execution claims.
- CUDA smoke, parity, receipts, and benchmarks are sequenced after identity.
- Strict BitNet CUDA proof receipts name the official GGUF, explicit tokenizer,
  W1.58A8 layout, selected RTX 5070 Ti CUDA backend, 9950X3D CPU reference,
  fallback status, QK256 CUDA invocation counts, upload-once weight residency,
  and measured timing.
- CUDA-DENSE-001 remains an optional dense regular-LLM reference lane and is not
  part of completed BitNet packed-kernel proof.

## Hard Constraints

- CUDA visibility is not kernel execution.
- WGPU smoke is not CUDA proof.
- CPU fallback cannot count as CUDA execution.
- Performance claims require driver, CUDA, VRAM, power, and thermal context.
- `speedup_claim` must remain false unless a same-model fallback-free benchmark
  receipt explicitly upgrades it.
- Dense regular-LLM CUDA receipts cannot satisfy BitNet packed I2S or QK256
  proof acceptance.

## Proof Ledger

RTX 5070 Ti CUDA BitNet proof state:

- strict selected backend: `nvidia-rtx-5070-ti-cuda`
- runtime API: `cuda`
- model: `microsoft/bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf`
- tokenizer: explicit external Llama BPE with BitNet.cpp answer template
- quantization/layout: `W1.58A8`, `gguf_packed_i2_s`
- CUDA kernel family: `qk256`
- weights uploaded once: `true`
- per-token weight upload: `false`
- QK256 CUDA kernel: `qk256_gemv_cuda`
- CUDA kernel invocations: `210` for one-token proof, `1680` for short decode
- BitNet linear CPU fallback: `0`
- fallback used: `false`
- speedup claim: `false`

Committed proof receipts:

- `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-06/cuda-smoke.json`
- `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-06/cuda-parity.json`
- `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-06/cuda-benchmark.json`
- `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-06/strict-bitnet-cuda-proof.json`
- `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-06/strict-bitnet-cuda-short-decode.json`
- `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-06/strict-bitnet-cuda-benchmark.json`
- `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/strict-cuda-ask-math.json`
- `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cuda-answer-corpus.json`
- `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cpu-avx512-answer-corpus.json`
- `ci/hardware/windows-9950x3d-rtx5070ti/2026-05-08/cpu-avx512-vs-cuda-answer-parity.json`

Allowed claims:

- The RTX 5070 Ti CUDA selected backend executes receipt-backed CUDA kernels.
- Strict BitNet CUDA one-token and short-decode proofs run through QK256 CUDA
  kernels with zero BitNet linear CPU fallback.
- The current benchmark is a baseline with `speedup_claim=false`.

Not allowed:

- Do not claim dense regular-LLM CUDA proves BitNet packed inference.
- Do not claim WGPU, Vulkan, D3D12, or generic `cuda` as RTX 5070 Ti CUDA proof.
- Do not claim speedup unless a later same-model fallback-free receipt upgrades
  `speedup_claim` under the benchmark policy.

## Productization Follow-Up

The completed proof lane is not the same as a polished user-answer product path.
The next productization track is defined in
`docs/specs/rtx5070ti-cuda-answer-readiness.md`. `MODEL-ARTIFACT-007`,
`CPU-ANSWER-007`, and `CUDA-ANSWER-010` now provide the artifact, CPU, and strict
RTX 5070 Ti CUDA answer-corpus gates needed for the next step: CPU/CUDA answer
parity against the same artifact, tokenizer authority, prompt template, and
deterministic corpus.

Coherent CUDA answer claims still remain scoped. The current proof covers the
committed deterministic answer corpus, not broad chat quality, production server
readiness, or speedup.

`CUDA-ANSWER-011` records the first same-box CPU AVX-512 versus RTX 5070 Ti CUDA
answer-corpus comparison. Both backends pass the corpus, but the original
receipt preserved top-k logit divergence for every case and a `yes_no_water`
generated-answer divergence.

`CUDA-ANSWER-012` closes the generated-answer divergence by aligning the CUDA
QK256 inline-scale path with BitNet.cpp I2_S x I8_S activation semantics. CPU
AVX-512 and RTX 5070 Ti CUDA now match generated token IDs and decoded text for
all five committed deterministic corpus cases. Exact top-k logit parity remains
open for four cases.

`CUDA-PROD-001` starts the user-path work after the strict corpus and parity
proofs. Strict `bitnet ask` must validate backend, fallback, and answer quality
even when the user omits `--receipt-out`; a default receipt path is acceptable
for strict ask, but the receipt still must preserve `speedup_claim=false`.

Answer receipts must keep the completed proof invariants intact:

- selected backend: `nvidia-rtx-5070-ti-cuda`
- runtime API: `cuda`
- `qk256_gemv_cuda` invocation count greater than zero
- weights uploaded once: `true`
- per-token weight upload: `false`
- BitNet linear CPU fallback: `0`
- prompt prefill exercised: `true`
- answer quality gate passed: `true`
- speedup claim: `false` unless later benchmark-qualified

## Work Items

| Work item | Status | Notes |
|---|---|---|
| RTX5070TI-003 | merged | Preserved selected-device CUDA identity in #3679. |
| RTX5070TI-004 | merged | Added CUDA and NVML runtime probe in #3691. |
| RTX5070TI-005 | merged | Tiny CUDA kernel smoke receipt merged in #3723. |
| RTX5070TI-006 | merged | CPU/CUDA tiny fixture parity receipt merged in #3749. |
| RTX5070TI-007 | merged | CUDA receipt validation and kernel counters merged in #3756. |
| RTX5070TI-008 | merged | CUDA benchmark baseline merged in #3770. |
| CUDA-BITNET-001 | merged | Persistent CUDA BitNet context and weight handles merged in #3776. |
| CUDA-BITNET-002 | merged | Reusable CUDA I2S primitive merged in #3782. |
| CUDA-BITNET-003 | merged | QK256 fused dequant GEMV CUDA path merged in #3786. |
| CUDA-BITNET-004 | merged | CUDA BitNet upload-once weight handling merged in #3790. |
| CUDA-BITNET-005 | merged | BitNetLinear CUDA routing merged in #3792. |
| CUDA-BITNET-006 | merged | Strict one-token BitNet CUDA proof merged in #3801. |
| CUDA-BITNET-007 | merged | Strict short-decode BitNet CUDA proof merged in #3806. |
| CUDA-BITNET-008 | merged | Strict BitNet CUDA benchmark baseline merged in #3823. |
| CUDA-BITNET-009 | merged | Routed upload-once strict proof receipts merged in #3837. |
| CUDA-ANSWER-010 | merged | Strict RTX 5070 Ti CUDA answer corpus passes after the QK256 I2_S layout alignment in #4024. |
| CUDA-ANSWER-011 | merged | Same-box CPU AVX-512 and RTX 5070 Ti CUDA both pass the corpus, but the original receipt preserved top-k logit divergence and one generated-answer divergence. |
| CUDA-ANSWER-012 | merged | CUDA QK256 I8_S activation semantics close generated-token parity for all five committed corpus cases; exact top-k parity remains open. |
| CUDA-PROD-001 | in_progress | Strict `bitnet ask` validates through a default answer receipt when `--receipt-out` is omitted. |
| CUDA-DENSE-001 | proposed | Optional dense regular-LLM CUDA reference lane; not part of BitNet packed proof completion. |

## Review Policy

CUDA PRs are non-stackable when they touch backend identity, kernels, receipts, or benchmark interpretation.
