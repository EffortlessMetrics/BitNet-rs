# BITNET-SPEC-0013: Model Onboarding Proof Ladder

Status: proposed
Linked proposal:
[BITNET-PROP-0003](../proposals/BITNET-PROP-0003-native-rust-inference-product.md)
Applies to: model coverage rows, answer artifact gates, CPU reference proof,
accelerator proof, benchmark qualification, product CLI promotion, server
readiness claims, model onboarding plans

## Purpose

BitNet-rs supports several different model families and runtime routes. The
repo needs one promotion ladder so a model row cannot jump from "loads" to
"supported" without tokenizer, prompt, CPU/reference, accelerator, quality,
benchmark, and server evidence.

This spec defines the model onboarding proof ladder and the claim boundaries
for moving a model from registered candidate to product-ready local inference.

## Source-Of-Truth Authorities

This spec relies on existing authorities instead of replacing them:

- [Native Rust inference product proposal](../proposals/BITNET-PROP-0003-native-rust-inference-product.md)
- [Source-of-truth and claim boundaries](BITNET-SPEC-0001-source-of-truth-and-claim-boundaries.md)
- [9950X3D + RTX 5070 Ti CUDA product contract](BITNET-SPEC-0007-9950x3d-5070ti-cuda-product-contract.md)
- [CUDA Route Contract](BITNET-SPEC-CUDA-ROUTE-CONTRACT.md)
- [Server readiness proof boundary](BITNET-SPEC-0010-server-readiness-proof-boundary.md)
- [Answer Artifact Gate](../model-artifacts/ANSWER_ARTIFACT_GATE.md)
- [Model Coverage Matrix](../model-artifacts/MODEL_COVERAGE_MATRIX.md)
- `ci/model-artifacts/model-coverage-matrix.toml`
- `ci/model-artifacts/artifact-manifest.toml`
- `ci/model-artifacts/tokenizer-authority.toml`
- `ci/model-artifacts/model-kernel-compatibility.toml`
- `ci/hardware/**`
- [CUDA Capability Matrix](../status/CUDA_CAPABILITY_MATRIX.md)

If this spec and the model coverage matrix disagree, repair one before making
the user-facing claim. If this spec and a receipt disagree, the receipt remains
the evidence for what happened.

## Ladder

Model rows advance through these tiers:

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

`server_ready` is profile-scoped. A model may be product CLI-ready without
server readiness, and a server readiness claim may be exact-profile only.

## Tier Requirements

### registered

The model is known to the repo as a candidate.

Required evidence:

- model family and artifact identity;
- intended route or route candidate;
- expected tokenizer and prompt authority source;
- current forbidden claims.

Allowed claim:

- registered candidate only.

### structurally_valid

The artifact contract can be parsed and its high-level shape matches the
registered family.

Required evidence:

- artifact manifest entry or equivalent structural validation;
- architecture, tensor, quantization, tokenizer, and prompt metadata status;
- explicit missing proof list.

Must not claim:

- coherent answer quality;
- CPU answer readiness;
- accelerator answer readiness;
- speedup;
- server readiness.

### reference_good

A reference route or trusted comparator exists for the same prompt policy,
tokenizer policy, and artifact family.

Required evidence:

- same-prompt comparator or accepted reference evidence;
- tokenizer and prompt-template authority;
- first-token, top-k, decode, or answer sanity evidence appropriate to the
  model family;
- classification for mismatches: prompt policy, tokenizer, shared math, or
  reference mismatch.

Must not claim:

- CPU answer readiness unless the CPU answer gate passes;
- accelerator readiness unless the accelerator proof passes.

### cpu_answer_ready

The CPU route can produce acceptable answers for the exact model family and
artifact contract.

Required evidence:

- answer artifact gate pass or equivalent CPU answer receipt;
- artifact identity;
- tokenizer authority;
- prompt-template authority;
- CPU route identity;
- quality result;
- fallback status when relevant.

Must not claim:

- CUDA, WGPU, Metal, OpenCL, NPU, or other accelerator execution;
- accelerator speedup.

### accelerator_answer_ready

The accelerator route runs for the exact model family, artifact, backend, and
route without fallback and with an accepted answer or diagnostic quality gate.

Required evidence:

- selected backend identity;
- selected route identity;
- CPU/reference evidence from an earlier tier;
- fallback rejection;
- route-specific kernel or execution-plan evidence;
- answer quality result;
- durable receipt path.

Must not claim:

- product CLI readiness without normal user-path receipts;
- speedup without benchmark qualification;
- server readiness without the server readiness proof surface.

### benchmark_qualified

An exact model/backend/profile comparator has accepted a speed or performance
claim.

Required evidence:

- same artifact;
- same prompt/profile;
- CPU/reference comparator when claiming speedup;
- accelerator comparator;
- prefill, first-token, decode, transfer, kernel, and residency fields required
  by the applicable performance contract;
- accepted or rejected decision and reason.

Must not claim:

- global speedup;
- other-model speedup;
- other-profile speedup;
- full residency unless residency proof is separately accepted.

### product_cli_ready

Normal user-facing CLI paths are available for the exact model/backend/route
claim and explain their receipts.

Required evidence:

- `bitnet model status` or equivalent model status surface;
- `bitnet model verify` path;
- `bitnet ask` user-path receipt;
- `bitnet chat` or warm-session user-path receipt when claimed;
- `bitnet bench` review when benchmark status is shown;
- `bitnet receipts explain` summary;
- model coverage row aligned with status docs.

Must not claim:

- server readiness unless `server_ready` is separately accepted;
- speedup unless `benchmark_qualified` accepted the exact profile;
- broad residency unless full-residency proof is accepted.

### server_ready

The server path is ready for the exact endpoint, model, backend, route,
streaming mode, profile, and readiness envelope that was proven.

Required evidence:

- endpoint and method;
- streaming mode;
- selected backend and route;
- model coverage row;
- request/response receipt identifier;
- fallback rejection;
- readiness scope: `exact_profile` or broader scope if explicitly accepted;
- forbidden claims.

Must not claim:

- broad production readiness from an exact-profile smoke;
- streaming readiness from non-streaming proof;
- concurrency readiness without concurrency proof;
- speedup or full residency without separate proof.

## CUDA Route Contract

CUDA model rows and receipts that claim CUDA execution must follow the narrower
[BITNET-SPEC-CUDA-ROUTE-CONTRACT](BITNET-SPEC-CUDA-ROUTE-CONTRACT.md).
In particular, generic `cuda` must resolve to the strict backend before proof,
`selected_route` must use the documented CUDA route IDs, strict CUDA fallback is
a hard failure, and proof-family booleans must stay non-interchangeable.

## Promotion Rules

- A model must not skip `reference_good` or `cpu_answer_ready` before
  accelerator answer claims unless a later accepted spec defines a narrower
  substitute and updates the model coverage row.
- Dense CUDA evidence cannot satisfy BitNet I2_S/QK256 proof.
- BitNet I2_S/QK256 evidence cannot satisfy dense SLM or small dense LLM proof.
- CUDA receipts without a selected route and execution plan cannot promote
  accelerator, product, server, speed, or residency claims.
- Qwen2.5 evidence cannot satisfy Qwen3, SmolLM2, Llama, Gemma, or Phi rows.
- Structural validity cannot satisfy answer readiness.
- Hardware detection cannot satisfy selected-backend execution proof.
- Selected-backend execution cannot satisfy answer quality by itself.
- Speedup is exact-profile only.
- Server readiness is exact-profile unless a later spec explicitly promotes a
  broader scope.

## Required Claim Booleans

Model coverage rows that reach accelerator, benchmark, product, or server
tiers should keep explicit booleans for claims that users may otherwise infer:

```text
speedup_claim
server_ready
server_scope
full_residency_claim
bitnet_packed_i2s_qk256_proof
dense_regular_llm_cuda_proof
fallback_rejected
```

Boolean names may vary by manifest, but the row must make these claim states
machine-readable before user-facing status surfaces summarize them.

## Model Family Boundaries

| Family | Route examples | Separate proof required |
| --- | --- | --- |
| Official BitNet I2_S/QK256 | `bitnet_qk256_cuda`, CPU BitNet route | BitNet artifact, tokenizer, QK256 kernels, fallback rejection |
| BitNet TL1/TL2/GPU-int2 | TL or int2 diagnostic routes | Own artifact and kernel proof; no inherited I2_S/QK256 proof |
| Dense SLM | `dense_regular_llm_cuda` for Qwen/SmolLM rows | Own tokenizer, prompt, CPU, CUDA, benchmark, and server receipts |
| Small dense LLM | Llama, Gemma, Phi candidates | Own architecture, tokenizer, prompt, memory, and route proof |
| Server readiness | `/v1/chat/completions` or later endpoints | Own endpoint/profile/readiness proof |

## Proof Commands

Current docs-only validation:

```bash
git diff --check
cargo run --locked -p xtask --no-default-features -- check-model-coverage
cargo run --locked -p xtask --no-default-features -- campaign check nvidia-5070ti
```

Runtime PRs that promote a row must add the exact command that produced the
receipt and the command that explains it, such as:

```bash
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- model status --device nvidia-rtx-5070-ti-cuda --format json
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- ask --device cuda --model <model> "..."
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- receipts explain --latest --format json
```

## Non-Goals

- Do not implement runtime model loading, routing, CLI, benchmark, or server
  behavior in this spec.
- Do not promote any model coverage row in this spec.
- Do not edit model receipts, hardware receipts, policy TOMLs, CI workflows, or
  generated dashboards.
- Do not redefine the answer artifact gate.
- Do not collapse proof families to simplify status text.

## Related Policy Or Manifest Sources

- `ci/model-artifacts/model-coverage-matrix.toml`
- `ci/model-artifacts/artifact-manifest.toml`
- `ci/model-artifacts/tokenizer-authority.toml`
- `docs/tracking/campaigns/nvidia-5070ti/active.toml`
- `ci/hardware/windows-9950x3d-rtx5070ti/**`
- `policy/docs-source-of-truth.toml`
- `policy/ci-lanes.toml`
- `policy/ci-risk-packs.toml`
