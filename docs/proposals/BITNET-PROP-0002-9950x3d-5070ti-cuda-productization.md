# BITNET-PROP-0002: 9950X3D + RTX 5070 Ti CUDA Productization

Status: proposed
Owner: cuda/product
Type: proposal

## Problem

BitNet-rs has advanced CUDA proof on the 9950X3D + RTX 5070 Ti machine, but the
proof surfaces are still distributed across campaign notes, model coverage
rows, hardware receipts, CUDA specs, benchmark reviews, and CLI receipt
explainers. The next CUDA lane should turn that proof into a boring,
receipt-backed product bench without blurring model families or overstating
speed and answer claims.

The platform already has strict RTX 5070 Ti CUDA BitNet receipts with selected
backend `nvidia-rtx-5070-ti-cuda`, runtime API `cuda`, official Microsoft I2_S
GGUF identity, external tokenizer authority, QK256 CUDA invocation counts,
upload-once weights, no per-token weight upload, zero BitNet linear CPU
fallback, deterministic answer-corpus proof, CPU/CUDA generated-token parity,
and `speedup_claim=false`.

The same platform also has a dense regular-LLM CUDA lane for Qwen2.5 0.5B Q8_0.
That lane is useful product evidence for dense SLMs, but it is not BitNet,
I2_S, QK256, or 1-bit proof. Additional dense SLM and small dense LLM rows such
as Qwen3, SmolLM2, Llama 3.2, Gemma, and Phi are candidates only until their own
artifact, tokenizer, prompt, CPU, CUDA, benchmark, and receipt gates pass.

## Proposal

Make the 9950X3D + RTX 5070 Ti machine the canonical CUDA product bench for
BitNet-rs x86 CPU reference plus NVIDIA CUDA target work.

The CUDA productization lane should produce normal user paths that are strict,
receipt-backed, and honest:

```bash
bitnet model verify <artifact>
bitnet ask --device nvidia-rtx-5070-ti-cuda --model <artifact> "..."
bitnet chat --device nvidia-rtx-5070-ti-cuda --model <artifact>
bitnet bench --device cuda --model <artifact>
bitnet receipts explain --latest
```

Those commands should tell the user the exact model family, artifact identity,
tokenizer authority, prompt authority, selected backend, fallback status,
quality result, benchmark status, and receipt path. They must also say what the
receipt does not prove.

## Source-Of-Truth Links

This lane uses BitNet-rs source-of-truth surfaces. It does not introduce a
parallel tracker or hidden goal file.

- [Source-of-truth and claim boundaries](../specs/BITNET-SPEC-0001-source-of-truth-and-claim-boundaries.md)
  defines how product claims map to proof artifacts.
- [RTX 5070 Ti roadmap](../specs/nvidia-rtx-5070-ti-roadmap.md) defines the
  CUDA hardware lane, receipt fields, strict backend identity, proof ledger, and
  claim boundary.
- [RTX 5070 Ti CUDA answer readiness](../specs/rtx5070ti-cuda-answer-readiness.md)
  defines strict CUDA answer receipts, fallback rejection, answer quality gates,
  prompt authority, and CPU/CUDA answer parity.
- [Answer Artifact Gate](../model-artifacts/ANSWER_ARTIFACT_GATE.md) remains
  the model answer-readiness authority.
- `ci/model-artifacts/model-coverage-matrix.toml` remains the model-family,
  tier, proof, and claim-control ledger.
- [Hardware Matrix](../hardware/HARDWARE_MATRIX.md) remains the hardware lane
  identity and proof-stage authority.
- [NVIDIA 5070 Ti Campaign](../tracking/campaigns/nvidia-5070ti/CAMPAIGN.md)
  and `docs/tracking/campaigns/nvidia-5070ti/active.toml` remain the live
  campaign execution authorities.
- [CI Cost and Verification Policy](../ci/cost-and-verification-policy.md)
  remains the CI economics authority for default versus risk-routed proof.

## Goals

- Make 9950X3D + RTX 5070 Ti the primary CUDA product bench for local inference
  validation.
- Keep the CPU reference identity `amd-9950x3d-cpu-avx512` separate from the
  CUDA target identity `nvidia-rtx-5070-ti-cuda`.
- Keep official BitNet 2B I2_S/QK256 proof separate from dense SLM and small
  dense LLM proof.
- Preserve the current BitNet CUDA claim boundary: strict answer-corpus and
  CLI-ready evidence exists, but speedup remains false until governed
  benchmark qualification accepts exact profiles.
- Treat Qwen2.5 0.5B Q8_0 as the first dense CUDA SLM product lane while
  preserving `bitnet_packed_i2s_qk256_proof=false`.
- Promote Qwen3, SmolLM2, Llama 3.2, Gemma, and Phi candidates one model at a
  time through their own proof ladder.
- Make `bitnet receipts explain` the common user-facing proof summary for ask,
  chat or warm-session, and benchmark receipts.
- Keep server claims separate until a strict CUDA server smoke path emits its
  own fallback-free receipt.

## Model-Family Boundaries

| Family | First product target | May claim when gated | Must not claim |
| --- | --- | --- | --- |
| Official BitNet 2B I2_S/QK256 | Microsoft I2_S GGUF on RTX 5070 Ti CUDA | BitNet QK256 CUDA answer readiness for scoped receipts | dense SLM proof, global speedup, server readiness |
| Dense SLM | Qwen2.5 0.5B Q8_0 | dense regular-LLM CUDA answer readiness for scoped receipts | BitNet, I2_S, QK256, or 1-bit proof |
| Dense SLM candidates | Qwen3 0.6B, SmolLM2 360M, SmolLM2 1.7B | candidate status until their own ladder passes | inherited Qwen2.5 CUDA proof |
| Small dense LLM candidates | Llama 3.2 1B/3B, Gemma/Phi small | candidate or diagnostic status until their own ladder passes | supported CUDA answers, speedup, or server readiness |
| WGPU/Vulkan/D3D12 reference | optional RTX 5070 Ti cross-platform lane | cross-platform reference proof when gated | CUDA proof |

## Required Product Invariants

Any strict CUDA answer receipt in this lane must preserve:

```text
requested_backend = nvidia-rtx-5070-ti-cuda
selected_backend = nvidia-rtx-5070-ti-cuda
runtime_api = cuda
fallback_used = false
model artifact identity present
tokenizer authority present
prompt template authority present
quality gate result present
execution_plan present
kernel stats present
speedup_claim = false unless benchmark-qualified
receipt path durable
claim boundary visible
```

BitNet receipts must also preserve BitNet-specific QK256 evidence:

```text
route = bitnet_qk256_cuda
qk256_gemv_cuda invocations > 0
bitnet linear CPU fallback count = 0
weights_uploaded_once = true
per_token_weight_upload = false
dense_regular_llm_cuda_proof = false
```

Dense SLM receipts must preserve dense-family evidence without inheriting
BitNet proof:

```text
route = dense_regular_llm_cuda
dense_regular_llm_cuda_proof = true only when the exact dense ladder passes
bitnet_packed_i2s_qk256_proof = false
speedup_claim = false unless benchmark-qualified
```

## Promotion Ladder

New dense SLM or small dense LLM candidates must graduate one model at a time:

```text
artifact contract
tokenizer and prompt authority
CPU answer sanity
dense all-layer plan
model-boundary fixtures
one-token strict CUDA proof
short-decode strict CUDA proof
warm-session strict CUDA proof
benchmark qualification
status matrix update
user guide or command surface
```

Do not batch-promote all candidate families. Qwen3 0.6B should be the first
candidate after Qwen2.5 because it can reuse the most dense Qwen infrastructure.
SmolLM2 is the next low-footprint model-family control. Llama 3.2, Gemma, and
Phi are broader tokenizer, prompt, architecture, and memory-pressure controls.

## CI Economics

This lane should respect BitNet-rs verification economics:

- Default docs/proposal/spec PRs stay docs-only and cheap.
- Runtime CUDA PRs run targeted crate checks and CUDA proof commands.
- Expensive hardware, model, benchmark, coverage, and mutation lanes run only
  when the changed surface warrants them.
- Skipped expensive lanes must report skipped-by-policy, not passed proof.
- Benchmark qualification is profile-specific; there is no global CUDA speedup
  claim.

## Non-Goals

- Do not change CUDA kernels in this proposal.
- Do not change runtime behavior, CLI behavior, workflows, receipts, model
  manifests, policy TOMLs, generated dashboards, or README product claims in
  this proposal.
- Do not create `.adze/goals`, `.bitnet/goals`, or another global active-work
  tracker.
- Do not claim generic `cuda` as RTX 5070 Ti proof.
- Do not claim dense Qwen proof as BitNet proof.
- Do not claim BitNet QK256 proof as dense SLM proof.
- Do not claim broad chat quality, production server readiness, or global
  speedup from one successful answer.

## Success Criteria

This CUDA productization lane succeeds when:

- Official BitNet 2B I2_S/QK256 has strict `model verify`, `ask`, warm-session
  or chat, `bench`, and `receipts explain` user paths with durable receipts.
- Dense Qwen2.5 0.5B Q8_0 has the same strict CUDA product path, with BitNet
  proof explicitly false.
- At least one additional dense SLM candidate advances through artifact, CPU,
  all-layer, CUDA, and short-decode proof without inheriting Qwen2.5 evidence.
- Benchmark receipts make profile-specific accepted or rejected speed decisions.
- Status docs and the model coverage matrix agree about claim tiers and
  forbidden claims.
- The `nvidia-5070ti` campaign `active.toml` records the next executable work
  items and their proof commands.

## Exit Criteria

The proposal can be closed when the lane has:

- A CUDA product contract spec.
- A CUDA product bench ADR.
- A CUDA productization plan.
- A reconciled current-state table for BitNet, dense Qwen, and candidate model
  rows.
- A campaign update with next CUDA productization work items.
- At least one user-facing CUDA quickstart whose claims match committed
  receipts and model coverage rows.

## Rollback

Rollback is documentation-only for this PR:

- Revert this proposal file.
- Leave runtime code, CUDA kernels, receipts, model manifests, policy ledgers,
  workflows, generated dashboards, and README product claims unchanged.
- If later CUDA productization specs or plans drift from receipts, repair those
  docs or demote the claim rather than editing proof receipts by hand.
