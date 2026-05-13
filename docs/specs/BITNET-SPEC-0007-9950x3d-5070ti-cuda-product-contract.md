# BITNET-SPEC-0007: 9950X3D + RTX 5070 Ti CUDA Product Contract

Status: proposed
Linked proposal:
[BITNET-PROP-0002](../proposals/BITNET-PROP-0002-9950x3d-5070ti-cuda-productization.md)
Applies to: RTX 5070 Ti CUDA product paths, official BitNet I2_S/QK256,
dense Qwen CUDA, dense SLM candidates, small dense LLM candidates, benchmark
qualification, CUDA receipt explanation

## Purpose

The 9950X3D + RTX 5070 Ti platform is the canonical BitNet-rs CUDA product
bench for x86 CPU reference plus NVIDIA CUDA target work. This spec defines
when that platform may be called product-ready for user-visible CUDA commands.

The platform is product-ready only when normal commands are boring,
fallback-free, and receipt-backed:

```bash
bitnet model verify <artifact>
bitnet ask --device nvidia-rtx-5070-ti-cuda --model <artifact> "..."
bitnet chat --device nvidia-rtx-5070-ti-cuda --model <artifact>
bitnet bench --device cuda --model <artifact>
bitnet receipts explain --latest
```

This spec does not implement those commands. It defines the contract that
future CUDA productization PRs must satisfy before claims move from diagnostic
or scoped proof to product documentation.

## Source-Of-Truth Authorities

This contract relies on existing BitNet authorities:

- [Source-of-truth and claim boundaries](BITNET-SPEC-0001-source-of-truth-and-claim-boundaries.md)
- [RTX 5070 Ti roadmap](nvidia-rtx-5070-ti-roadmap.md)
- [RTX 5070 Ti CUDA answer readiness](rtx5070ti-cuda-answer-readiness.md)
- [Answer Artifact Gate](../model-artifacts/ANSWER_ARTIFACT_GATE.md)
- [Hardware Matrix](../hardware/HARDWARE_MATRIX.md)
- [NVIDIA 5070 Ti Campaign](../tracking/campaigns/nvidia-5070ti/CAMPAIGN.md)
- `docs/tracking/campaigns/nvidia-5070ti/active.toml`
- `ci/model-artifacts/model-coverage-matrix.toml`
- `ci/hardware/windows-9950x3d-rtx5070ti/**`

If this spec and a receipt disagree, the receipt is the evidence for what
happened. If this spec and a policy or model coverage ledger disagree, repair
the doc or ledger before promoting the user-facing claim.

## Product-Ready Definition

A CUDA product path is ready only for the exact model family, artifact, route,
and profile whose receipts prove the claim.

For each promoted row, the product path must have:

- artifact identity and verifier surface;
- tokenizer and prompt-template authority;
- CPU reference evidence on the 9950X3D or accepted source of truth;
- strict CUDA selected-backend evidence on RTX 5070 Ti;
- fallback rejection under strict CUDA;
- answer or diagnostic quality gate result;
- durable receipt path;
- `bitnet receipts explain` summary;
- status or model coverage row that matches the receipt;
- speed claim explicitly accepted or rejected by profile.

## Strict CUDA Answer Receipt Invariants

Any strict CUDA answer receipt for this platform must preserve:

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
receipt path durable
claim boundary visible
speedup_claim = false unless benchmark-qualified
```

Strict CUDA must fail closed when:

- CUDA is unavailable;
- the selected backend is not `nvidia-rtx-5070-ti-cuda`;
- generic `cuda` is treated as strict RTX 5070 Ti proof;
- CPU fallback is attempted;
- tokenizer or prompt authority is missing or ambiguous;
- the model artifact is not acceptable for the claimed family;
- kernel stats needed for the claim are missing;
- answer quality gate fails for an answer-ready claim.

Even on failure, strict user commands should write enough receipt evidence to
explain the rejected claim when that is safe and the command reached receipt
creation.

## Route-Specific Contracts

### BitNet QK256 CUDA

Official BitNet 2B I2_S/QK256 CUDA proof is route-specific:

```text
model_class = bitnet
route = bitnet_qk256_cuda
artifact = official Microsoft I2_S GGUF
selected_backend = nvidia-rtx-5070-ti-cuda
cpu_reference = amd-9950x3d-cpu-avx512
bitnet_packed_i2s_qk256_proof = true
dense_regular_llm_cuda_proof = false
speedup_claim = false unless benchmark-qualified
```

Receipts for this route must include:

- official artifact identity;
- external tokenizer authority;
- prompt-template authority;
- QK256 CUDA kernel invocation counts;
- BitNet linear CPU fallback count;
- upload-once weight residency status;
- per-token weight upload status;
- answer quality result for answer claims;
- benchmark profile decision for speed claims.

The current source-of-truth ledgers mark this route as product CLI ready and
speed false. A later benchmark PR may only promote exact profiles, not global
CUDA speed.

### Dense Regular-LLM CUDA

Dense SLM and small dense LLM CUDA proof is route-specific:

```text
model_class = dense_slm | small_dense_llm
route = dense_regular_llm_cuda
selected_backend = nvidia-rtx-5070-ti-cuda
dense_regular_llm_cuda_proof = true only for the exact promoted artifact
bitnet_packed_i2s_qk256_proof = false
speedup_claim = false unless benchmark-qualified
```

Dense CUDA receipts must not inherit BitNet QK256 proof. BitNet receipts must
not satisfy dense SLM proof. Each dense model must carry its own artifact,
tokenizer, prompt, CPU, CUDA, and benchmark evidence.

Qwen2.5 0.5B Q8_0 is the first dense CUDA SLM product lane. Qwen3, SmolLM2,
Llama 3.2, Gemma, and Phi rows remain candidates until their own proof ladders
pass.

### CPU AVX-512 Reference

The same-box CPU reference identity is:

```text
amd-9950x3d-cpu-avx512
```

CPU evidence may prove CPU answer readiness or provide comparator evidence. It
does not prove CUDA execution, CUDA speed, WGPU execution, or dense/BitNet
cross-family claims.

### WGPU, Vulkan, And D3D12 Reference

WGPU, Vulkan, and D3D12 may be useful RTX 5070 Ti cross-platform reference
lanes. They are never CUDA proof. A receipt for those routes must preserve its
own selected backend and must not be summarized as
`nvidia-rtx-5070-ti-cuda`.

### Benchmark-Qualified Profiles

Benchmark qualification is profile-specific:

```text
one_token
short_decode_8
short_decode_32
warm_session_3_turns
warm_session_10_turns
```

Each profile decision must record:

- CPU mean, p50, and p95;
- CUDA mean, p50, and p95;
- prompt prefill time;
- first-token latency;
- steady decode time;
- kernel time;
- H2D timing source;
- D2H timing source;
- VRAM high-water mark;
- power and thermal context when available;
- fallback status;
- accepted or rejected speedup decision;
- reason.

`speedup_claim=true` for one profile does not imply speedup for another profile
or model family.

## User-Facing Command Contract

### `bitnet model verify`

`bitnet model verify <artifact>` should surface:

- artifact identity;
- artifact tier from the model coverage matrix or artifact manifests;
- tokenizer authority;
- prompt-template authority;
- supported, diagnostic, candidate, or unsupported state;
- next missing proof when not ready.

It must not turn structural validity into answer readiness.

### `bitnet ask`

Strict CUDA `ask` must surface or record:

- requested backend;
- selected backend;
- CUDA runtime visibility;
- artifact identity;
- tokenizer and prompt authority;
- route label;
- fallback status;
- answer quality result;
- default receipt path when the user does not pass one;
- speed claim status.

### `bitnet chat` Or Warm Session

Strict CUDA chat or warm-session proof must surface or record:

- model loaded once when claimed;
- tokenizer loaded once when claimed;
- CUDA context initialized once when claimed;
- upload-once buffers or weights when claimed;
- per-turn receipts;
- session summary receipt;
- full-residency claim status;
- speed claim status.

### `bitnet bench`

`bitnet bench --device cuda` must distinguish:

- existing governed benchmark receipt explanation;
- fresh benchmark execution;
- profile accepted or rejected status;
- proof source for timing, transfer, power, and thermal fields;
- scope of any accepted speed claim.

### `bitnet receipts explain`

`bitnet receipts explain --latest` should be the common proof cockpit. It must
summarize:

- model family;
- route;
- requested backend;
- selected backend;
- fallback status;
- quality status;
- benchmark status;
- residency status;
- speed claim status;
- durable receipt path;
- claim not allowed.

## Candidate Model Promotion Ladder

Dense SLM and small dense LLM candidates must follow this sequence:

```text
artifact contract
tokenizer and prompt authority
CPU answer sanity
all-layer plan
model-boundary fixtures
one-token strict CUDA proof
short-decode strict CUDA proof
warm-session strict CUDA proof
benchmark qualification
status/model-coverage update
user guide or command surface
```

Do not combine all candidates into one proof PR. The initial order is:

1. Qwen3 0.6B Q8/Q4.
2. SmolLM2 360M.
3. Llama 3.2 1B.
4. SmolLM2 1.7B.
5. Llama 3.2 3B.
6. Gemma or Phi small.

## Claim Boundary

| Proof type | What it means | Must not claim |
| --- | --- | --- |
| BitNet QK256 CUDA | Official BitNet I2_S/QK256 lane on RTX 5070 Ti CUDA | dense SLM proof, global speedup, server readiness |
| Dense regular-LLM CUDA | Dense SLM or small dense LLM lane for the exact artifact | BitNet, I2_S, QK256, or 1-bit proof |
| CPU AVX-512 | Same-box CPU reference or comparator evidence | CUDA proof or speedup |
| WGPU/Vulkan/D3D12 | Optional cross-platform RTX 5070 Ti reference | CUDA proof |
| Benchmark qualified | Exact profile accepted by governed receipt review | global speedup or other-profile speedup |
| Receipt explanation | Summary of committed receipt fields | new proof beyond the receipt |

## Proof Commands

Current docs-only validation:

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- campaign check nvidia-5070ti
```

Runtime PRs that promote product status must add the matching command from the
user path they affect, such as:

```bash
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- ask --device nvidia-rtx-5070-ti-cuda ...
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- bench --device cuda ...
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- receipts explain --latest
```

## Non-Goals

- Do not implement runtime CUDA behavior in this spec.
- Do not change CLI behavior in this spec.
- Do not alter receipts or model manifests in this spec.
- Do not edit generated dashboards by hand.
- Do not claim generic `cuda` as RTX 5070 Ti proof.
- Do not promote candidates without their own proof ladder.
- Do not use one successful answer as a broad chat, server, or speed claim.

## Related Policy Or Manifest Sources

- `ci/model-artifacts/model-coverage-matrix.toml`
- `ci/model-artifacts/artifact-manifest.toml`
- `ci/model-artifacts/tokenizer-authority.toml`
- `docs/tracking/campaigns/nvidia-5070ti/active.toml`
- `ci/hardware/windows-9950x3d-rtx5070ti/**`
- `policy/ci-lanes.toml`
- `policy/ci-budget.toml`
- `policy/ci-risk-packs.toml`
