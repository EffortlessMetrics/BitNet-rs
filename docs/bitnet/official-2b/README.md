# Official BitNet 2B Source Map

## Purpose

`microsoft/BitNet-b1.58-2B-4T` is the official BitNet-rs reference model
family. This source map records the current source-of-truth stack for making
that model boringly correct, route-explicit, fallback-safe, benchmarkable, and
impossible to overclaim.

This is a governance source map only. It does not add model binaries, runtime
code, new receipts, or claim promotion.

## Current Answer Authority

The current answer-authority artifact is the official Microsoft I2_S GGUF route
recorded as:

```text
coverage_row = bitnet_official_2b_i2s_qk256
model_family = bitnet_b1_58
artifact_kind = gguf_i2_s
route_family = gguf_i2_s_qk256
tokenizer_authority = external_llama_bpe
prompt_authority = bitnetcpp-answer
current_tier = product_cli_ready
```

That row is the shared answer-ready plate for CPU and CUDA I2_S/QK256 work only
when the exact official artifact, external Microsoft tokenizer authority,
`llama-bpe` pre-tokenizer compatibility decision, and BitNet.cpp answer prompt
authority are preserved.

## Current Claim Boundary

The official I2_S/QK256 row may remain `product_cli_ready` for bounded CPU/CUDA
CLI use because the coverage matrix records:

- `reference_good=true`
- `cpu_answer_ready=true`
- `accelerator_answer_ready=true`
- `bitnet_packed_i2s_qk256_proof=true`

The same row must continue to reject broader claims until route-specific proof
exists:

- `benchmark_qualified=false`
- `server_ready=false`
- `speedup_claim=false`
- `full_residency_claim=false`
- dense regular-LLM CUDA proof does not satisfy BitNet packed proof
- exact-profile server smoke does not imply broad production server readiness

## Route Split

The official model family is not one vague `BitNet works` claim. Each route has
its own row, specs, receipts, and promotion gate.

| Route | Coverage row | Current posture | Claim boundary |
|---|---|---|---|
| I2_S/QK256 GGUF | `bitnet_official_2b_i2s_qk256` | Product CLI-ready for bounded CPU/CUDA answer lanes | Does not prove speedup, full residency, broad server readiness, Apple, A770, TL1, TL2, or BF16/GPU-int2. |
| TL1 ARM | `bitnet_official_2b_tl1_arm_candidate` | Registered candidate | Needs TL1 layout, scalar oracle, and ARM/NEON or Apple proof; cannot inherit I2_S proof. |
| TL2 x86 | `bitnet_official_2b_tl2_x86_candidate` | Registered candidate | Needs TL2 layout, scalar oracle, runner verification, and x86 proof; cannot inherit I2_S proof. |
| BF16 to GPU int2/W2A8 | `bitnet_official_2b_bf16_gpu_int2_candidate` | Registered candidate | Needs BF16 artifact contract, packer parity, and GPU int2 runtime proof; cannot inherit GGUF I2_S proof. |
| Dense fallback | none | Diagnostic only | Cannot promote BitNet answer, packed-kernel, speed, residency, or server claims. |

## Product End State

The official model is fully governed when normal user-facing commands can verify
and explain all of these fields for every claimed route:

```text
exact official artifact
exact tokenizer and pre-tokenizer authority
exact prompt template and stop policy
explicit route family: I2_S/QK256, TL1, TL2, or BF16/GPU-int2
requested and selected backend
selected kernel and invocation counters
fallback_used=false where strict
answer quality result
profile-scoped speed decision
per-phase residency class
server readiness scope or explicit false
forbidden claims and next proof required
```

Representative user surfaces are:

```bash
bitnet model verify microsoft-bitnet-b1.58-2B-4T-i2s
bitnet model status --model microsoft-bitnet-b1.58-2B-4T-i2s
bitnet ask --model microsoft-bitnet-b1.58-2B-4T-i2s --device cpu
bitnet ask --model microsoft-bitnet-b1.58-2B-4T-i2s --device cuda
bitnet chat --model microsoft-bitnet-b1.58-2B-4T-i2s --device cuda
bitnet bench --model microsoft-bitnet-b1.58-2B-4T-i2s --device cuda
bitnet receipts explain --latest
```

## Source-of-Truth Stack

| Layer | Artifact |
|---|---|
| Proposal | `docs/proposals/BITNET-PROP-0014-official-bitnet-2b-productization.md` (planned) |
| Artifact contract specs | `docs/specs/BITNET-SPEC-OFFICIAL-2B-ARTIFACT-CONTRACT.md` and `docs/specs/BITNET-SPEC-OFFICIAL-2B-TOKENIZER-PROMPT.md` (planned) |
| Route specs | `docs/specs/BITNET-SPEC-OFFICIAL-2B-I2S-QK256.md` and `docs/specs/BITNET-SPEC-OFFICIAL-2B-TL1-TL2.md` (planned) |
| Backend specs | CPU, CUDA, Apple, and A770/OpenCL official 2B specs (planned) |
| Quality/perf/product specs | Quality, performance, residency, server, and status-surface specs (planned) |
| Plan | `plans/official-bitnet-2b/implementation-plan.md` |
| Active campaign | `docs/tracking/campaigns/official-bitnet-2b/active.toml` |
| Current claim ledgers | `ci/model-artifacts/model-coverage-matrix.toml` and `ci/model-artifacts/model-kernel-compatibility.toml` |
| Answer-artifact gate | `docs/model-artifacts/ANSWER_ARTIFACT_GATE.md` |
| CPU path constraints | `docs/bitnet/BITNET_CPU_PATH_PLAN.md` |
| CUDA proof lane | `docs/tracking/campaigns/nvidia-5070ti/active.toml` and `plans/cuda-5070ti-productization/bitnet-official-i2s.md` |

## Non-goals

This source map does not:

- add or commit model binaries;
- change loader, tokenizer, kernel, server, or CLI runtime behavior;
- promote TL1, TL2, BF16/GPU-int2, Apple, A770, speed, full-residency, or broad
  server readiness claims;
- allow no-scale F32 diagnostic QK256 to count as production I2_S proof;
- allow dense regular-LLM CUDA proof to satisfy BitNet packed I2_S/QK256 proof;
- allow CUDA receipts to satisfy CPU, Apple, A770, TL1, TL2, or BF16/GPU-int2
  proof.
