# BITNET-PROP-0003: Native Rust Inference Product

Status: proposed
Owner: inference/product
Type: proposal

## Problem

Local inference tools usually answer the narrow question "did the model run?"
BitNet-rs should answer the support question users actually need:

```text
what model this is
what tier it has earned
which backend actually ran
which route was selected
whether fallback was rejected
whether answer quality passed
whether speed is qualified
whether server readiness is exact-profile or broad
what proof family applies
what claim remains forbidden
```

BitNet-rs already has strong proof surfaces for the 9950X3D + RTX 5070 Ti CUDA
lane, including official BitNet I2_S/QK256 CUDA proof, dense Qwen2.5 CUDA
product proof, Qwen3 candidate proof, model coverage ledgers, hardware
receipts, campaign trackers, server-smoke evidence, and receipt explainers.
The next product step is to make those surfaces compose into a Rust-native
local inference stack for BitNet models, dense SLMs, and selected small LLMs
without turning one proof family into another.

## Thesis

BitNet-rs should become the strongest Rust-native local inference stack for
BitNet models, dense SLMs, and selected small LLMs across CPU, AVX-512, and
CUDA.

Strong means infrastructure, not a single fast demo:

- accurate outputs;
- low time to first token;
- strong sustained decode;
- strict fallback rejection;
- model-aware routing;
- profile-scoped benchmark claims;
- clear server readiness;
- receipt-backed support claims.

The product is verified local inference. Every command that claims support must
carry model contracts, tokenizer and prompt authority, backend routing,
fallback status, quality result, benchmark qualification, model status, and
receipts.

## User-Facing End State

The 9950X3D + RTX 5070 Ti lane should support the normal user path:

```powershell
bitnet model status --device nvidia-rtx-5070-ti-cuda
bitnet model verify <model>
bitnet ask --device cuda --model <model> "..."
bitnet chat --device cuda --model <model>
bitnet bench --device cuda --model <model>
bitnet serve --device cuda --model <model>
bitnet receipts explain --latest
```

Those commands may accept friendly selectors such as `cuda`, but any proof
claim must resolve to the actual selected backend and route before it is
reported.

## Source-Of-Truth Links

This proposal relies on the BitNet source-of-truth stack. It does not create a
parallel tracker or hidden goal file.

- [Source-of-truth and claim boundaries](../specs/BITNET-SPEC-0001-source-of-truth-and-claim-boundaries.md)
- [9950X3D + RTX 5070 Ti CUDA productization proposal](BITNET-PROP-0002-9950x3d-5070ti-cuda-productization.md)
- [9950X3D + RTX 5070 Ti CUDA product contract](../specs/BITNET-SPEC-0007-9950x3d-5070ti-cuda-product-contract.md)
- [Server readiness proof boundary](../specs/BITNET-SPEC-0010-server-readiness-proof-boundary.md)
- [CUDA capability matrix](../status/CUDA_CAPABILITY_MATRIX.md)
- [Answer Artifact Gate](../model-artifacts/ANSWER_ARTIFACT_GATE.md)
- `ci/model-artifacts/model-coverage-matrix.toml`
- [Hardware Matrix](../hardware/HARDWARE_MATRIX.md)
- `ci/hardware/**`
- [NVIDIA 5070 Ti Campaign](../tracking/campaigns/nvidia-5070ti/CAMPAIGN.md)
- `docs/tracking/campaigns/nvidia-5070ti/active.toml`
- [CI Cost and Verification Policy](../ci/cost-and-verification-policy.md)
- `policy/docs-source-of-truth.toml`

If this proposal and a proof receipt disagree, the receipt is the evidence for
what happened. If this proposal and the model coverage matrix disagree, repair
the proposal or matrix before promoting the user-facing claim.

## Product Shape

BitNet-rs should provide proof-carrying local inference.

Each product row should expose:

- model identity and artifact contract;
- tokenizer and prompt-template authority;
- current tier from the model coverage ladder;
- selected backend and route;
- strict fallback rejection result;
- CPU/reference status;
- accelerator status;
- answer-quality status;
- speed qualification status;
- server readiness scope;
- proof family;
- forbidden claims.

The `bitnet model status` command should become the front door for this truth.
It should summarize support state without requiring users to read TOML.

## Model Coverage Ladder

Model families must graduate through the same ladder:

```text
registered
structurally_valid
reference_good
cpu_answer_ready
accelerator_answer_ready
benchmark_qualified
product_cli_ready
server_ready
```

No model may skip CPU/reference proof before accelerator answer claims. A model
may still be structurally valid while not answer-ready. Server readiness and
speed readiness are exact-profile claims unless a later spec explicitly
promotes broader scope.

## Current Product Rows

| Model lane | Current state | Still false |
| --- | --- | --- |
| Official BitNet 2B I2_S/QK256 | Product CLI-ready with BitNet QK256 CUDA proof and strict server-smoke evidence | speedup, full residency, broad server readiness |
| Qwen2.5 0.5B Q8_0 | Dense CUDA product CLI-ready and exact-profile server-ready | speedup, full residency, BitNet proof |
| Qwen3 0.6B Q8_0 | Accelerator-answer-ready candidate | product CLI-ready, server-ready, speedup, full residency |
| SmolLM2 360M | Structurally valid with an exact comparator contract | reference-good, CPU answer-ready, CUDA-ready |
| Llama/Gemma/Phi small | Registered candidates | claims beyond registration |
| BitNet TL1/TL2/GPU-int2 paths | Registered candidates or diagnostic lanes | inherited I2_S/QK256 proof |

This table is proposal context. The authoritative state remains
`ci/model-artifacts/model-coverage-matrix.toml`, status docs, and committed
receipts.

## Proof Families

Proof families are not interchangeable:

| Proof family | May prove | Must not prove |
| --- | --- | --- |
| BitNet I2_S/QK256 CUDA | Official packed BitNet QK256 route for the exact artifact/backend/profile | dense SLM CUDA, TL1/TL2, GPU-int2, broad server readiness |
| Dense regular-LLM CUDA | Exact dense SLM or small dense LLM route for one artifact/backend/profile | BitNet, I2_S, QK256, or 1-bit proof |
| CPU AVX-512 reference | CPU answer sanity or same-box comparator evidence | CUDA execution or CUDA speedup |
| Server readiness | One endpoint/profile/readiness envelope | broad serving, concurrency, streaming, speedup, or full residency |
| Benchmark qualification | One exact model/profile comparator | global speedup or other-profile speedup |

Dense CUDA evidence never proves BitNet packed I2_S/QK256. BitNet QK256
evidence never proves dense regular-LLM CUDA. Generic CUDA is not strict RTX
5070 Ti proof until a receipt resolves the selected backend.

## Model Onboarding Factory

Every new model family should graduate through a repeatable sequence:

```text
artifact contract
tokenizer and prompt authority
CPU answer sanity
accelerator all-layer plan
model-boundary fixtures
one-token proof
short decode proof
warm session proof
benchmark review
product CLI promotion
server readiness review
```

Qwen3 should be the next dense product promotion candidate after Qwen2.5.
SmolLM2 should first receive same-prompt comparator proof before CPU answer
readiness. Later Llama 3.2, Gemma, and Phi rows should be promoted one model at
a time through the same ladder.

## Runtime Performance Contract

Performance claims should be split by phase and profile. A future runtime
performance spec should require fields such as:

```text
model_load_ms
tokenizer_load_ms
prompt_render_ms
tokenize_ms
cuda_context_init_ms
weight_upload_ms
prefill_ms
first_token_ms
decode_total_ms
steady_tok_per_s
kernel_time_ms
launch_count
H2D_bytes
H2D_ms
D2H_bytes
D2H_ms
VRAM_high_water
fallback_used
```

TTFT claims require first-token breakdown. Throughput claims require decode
profile and token count. Speedup claims require an exact same-artifact CPU
comparator. Full residency claims require per-phase residency proof.

## CI Economics

The ordinary PR lane should stay cheap enough for swarm-scale work:

```text
Linux only
no macOS
no Windows
no GPU hardware
no Docker
no model downloads
no coverage
no broad feature matrix
crate/risk scoped
```

Expensive proof remains available on main, schedule, workflow dispatch,
release, labels, and hardware campaigns. Skipped expensive proof must be
reported as skipped by policy, not as passed evidence.

## Non-Goals

- Do not implement runtime inference behavior in this proposal.
- Do not change CUDA kernels, CPU kernels, server behavior, model manifests, CI
  workflows, policy TOMLs, hardware receipts, or generated dashboards.
- Do not promote any model row or claim boolean in this proposal.
- Do not create `.adze/goals`, `.bitnet/goals`, or another active work store.
- Do not claim broad server readiness from exact-profile server smoke.
- Do not claim speedup without governed exact-profile benchmark qualification.
- Do not claim full residency without per-phase residency proof.

## Success Criteria

This product lane succeeds when:

- `bitnet model status` is the user-facing model/hardware truth surface;
- `bitnet model verify`, `ask`, `chat`, `bench`, `serve`, and
  `receipts explain` share receipt-backed claim semantics;
- official BitNet I2_S/QK256, dense Qwen2.5, Qwen3, SmolLM2, and later small
  LLM candidates move through the same model onboarding ladder;
- speed, residency, and server readiness claims are exact-profile unless a
  later spec explicitly broadens them;
- CI routing keeps ordinary PRs cheap while preserving expensive proof on the
  lanes where it earns its cost.

## Rollback

Rollback is documentation-only:

- revert this proposal file;
- leave runtime code, receipts, model coverage rows, policy ledgers, workflows,
  generated dashboards, and README product claims unchanged;
- repair or demote any later docs that cite this proposal if the product
  direction changes.
