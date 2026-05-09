# Apple M4 Dense SLM Model Support Matrix

This matrix defines the dense SLM model states for the Apple M4 Mac mini lane.
It is intentionally separate from BitNet, 1-bit / 1.58-bit, QK256, MPSGraph,
Neural Engine, and full Apple Metal inference work.

The current production path is a dense Qwen-class Apple CPU/NEON path. A model
is not user-supported on the M4 lane just because it is small, popular, or
loadable somewhere else. It must pass the M4 gates below.

## State Vocabulary

- `default`: selected by the Mac wrapper when no model ID is provided.
- `supported`: selectable by model ID for M4 dense SLM answers after reference
  and Rust M4 quality gates pass.
- `candidate`: exact artifact or family is worth evaluating, but it is not
  accepted for M4 answers yet.
- `diagnostic-only`: useful for loader, tokenizer, or architecture debugging;
  not an answer path.
- `rejected`: excluded from this lane until a separate architecture or artifact
  decision changes the state.

## Promotion Gates

To move from `candidate` or `diagnostic-only` to `supported`, a dense SLM must
record all of the following:

- source repository, revision, file name, byte size, and SHA256;
- GGUF architecture, quantization, tokenizer model, pre-tokenizer authority, and
  chat-template or prompt-template authority;
- strict model-cache verification with no binary committed to the repository;
- reference-runner output sanity on the M4 dense SLM prompt suite;
- Rust M4 `apple-m4-cpu-neon` output quality with `fallback_used = false`;
- generated text, prompt token IDs, generated token IDs, timing, and receipt
  validation;
- deterministic greedy behavior where the quality corpus expects it;
- explicit unsupported-backend and hidden-fallback failure behavior.

## Current Matrix

### `qwen2.5-0.5b-instruct-q8_0`

- State: `default`.
- Source: `Qwen/Qwen2.5-0.5B-Instruct-GGUF`.
- Revision: `9217f5db79a29953eb74d5343926648285ec7e67`.
- File: `qwen2.5-0.5b-instruct-q8_0.gguf`.
- Size: `675710816` bytes.
- SHA256: `ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e`.
- GGUF architecture: `qwen2`.
- Quantization: `Q8_0`.
- Tokenizer authority: `tokenizer.ggml.model = gpt2`,
  `tokenizer.ggml.pre = qwen2`.
- Prompt template: `qwen2.5`.
- Rust support status: `supported` through the dense Qwen CPU path.
- M4 support status: `default` for `bitnet mac ask`, `bitnet mac chat`,
  `bitnet mac smoke`, `bitnet mac doctor`, and M4 dense SLM validation.
- Cache policy: registered model-cache artifact; fetched and verified by
  `bitnet model fetch qwen2.5-0.5b-instruct-q8_0` and
  `bitnet model verify qwen2.5-0.5b-instruct-q8_0`.
- Quality status: passing M4 dense SLM local-answer, warm-session, quality,
  determinism, smoke, doctor, regression, and excellence receipt surfaces.
- Claim boundary: proves the dense Qwen Apple CPU/NEON answer path only; it does
  not prove BitNet, QK256, full Apple Metal inference, Neural Engine execution,
  MPSGraph model inference, CUDA, x86, or broad Apple Silicon performance.

### `qwen2.5-0.5b-instruct-q4_k_m`

- State: `supported`.
- Source: `Qwen/Qwen2.5-0.5B-Instruct-GGUF`.
- Revision: `9217f5db79a29953eb74d5343926648285ec7e67`.
- File: `qwen2.5-0.5b-instruct-q4_k_m.gguf`.
- Size: `491400032` bytes.
- SHA256: `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`.
- GGUF architecture: `qwen2`.
- Quantization: `Q4_K_M`.
- Tokenizer authority: `tokenizer.ggml.model = gpt2`,
  `tokenizer.ggml.pre = qwen2`.
- Prompt template: `qwen2.5`.
- Rust support status: `supported` through eager F32 dequantization for the
  standard GGUF tensor types used by this artifact: `Q5_0`, `Q4_K`, and `Q6_K`.
- M4 support status: supported non-default model for `apple-m4-cpu-neon` dense
  SLM answers.
- Cache policy: registered artifact may be fetched and verified for inspection,
  ask/chat/validate use through `--model-id qwen2.5-0.5b-instruct-q4_k_m`.
  It is not the default model.
- Quality status: reference-runner prompt sanity passed in `SLM-M4-002`; the
  `M4-SLM-EX-006` Rust M4 quality corpus passes with `fallback_used = false`
  and stable deterministic repeated prompt groups.
- Claim boundary: storage-conscious dense Qwen Apple CPU/NEON answer path; it
  does not prove BitNet, QK256, full Apple Metal inference, Neural Engine
  execution, MPSGraph model inference, CUDA, x86, or broad Apple Silicon
  performance.

### `qwen3-0.6b-q8_0`

- State: `diagnostic-only`.
- Source: `Qwen/Qwen3-0.6B-GGUF`.
- Revision: not recorded in the M4 lane.
- File: `Qwen3-0.6B-Q8_0.gguf`.
- Size: not recorded in the M4 lane.
- SHA256: `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`.
- GGUF architecture: `qwen3`.
- Quantization: `Q8_0`.
- Tokenizer authority: must be recorded from GGUF metadata before M4 promotion.
- Prompt template: Qwen3-compatible template required before M4 promotion.
- Rust support status: shared dense Qwen bring-up and first-token parity work is
  tracked outside the M4 Mac mini lane.
- M4 support status: not supported for M4 local answers.
- Cache policy: do not add to the M4 default cache path until exact M4 artifact
  metadata, tokenizer authority, and quality gates pass.
- Quality status: not accepted on M4.
- Claim boundary: useful as an upstream dense Qwen correctness candidate; not an
  M4 dense SLM product model.

### Qwen Small Instruct, 1B-ish Class

- State: `candidate`.
- Source: exact official GGUF repository required before evaluation.
- Revision: required before evaluation.
- File: required before evaluation.
- Size: required before evaluation; preferred under `1 GiB` for this lane.
- SHA256: required before evaluation.
- GGUF architecture: expected `qwen2` or `qwen3`, but exact metadata must decide.
- Quantization: `Q8_0` preferred for first Rust support; `Q4_K_M` only after
  strict support is proven.
- Tokenizer authority: required from GGUF metadata.
- Prompt template: required before reference and Rust M4 runs.
- Rust support status: candidate only.
- M4 support status: not supported until all promotion gates pass.
- Cache policy: no model-cache entry until an exact artifact is selected.
- Quality status: no M4 quality evidence.
- Claim boundary: candidate family only.

### Small Gemma, Phi, or SmolLM Instruct GGUF

- State: `candidate`.
- Source: exact official or trusted GGUF repository required before evaluation.
- Revision: required before evaluation.
- File: required before evaluation.
- Size: required before evaluation; preferred under `1 GiB`.
- SHA256: required before evaluation.
- GGUF architecture: required before evaluation.
- Quantization: `Q8_0` preferred for first support unless the Rust loader has
  exact support for the chosen quantization.
- Tokenizer authority: required from GGUF metadata.
- Prompt template: family-specific template required before reference and Rust
  M4 runs.
- Rust support status: candidate only; likely needs a family adapter before M4
  support.
- M4 support status: not supported.
- Cache policy: no model-cache entry until an exact artifact passes reference
  and Rust M4 gates.
- Quality status: no M4 quality evidence.
- Claim boundary: cross-family adapter candidate only.

### Qwen3.5, Hybrid, Vision, MoE, or State-Space Variants

- State: `rejected`.
- Source: not applicable for this lane until a separate architecture campaign
  selects an exact artifact.
- Revision: not applicable.
- File: not applicable.
- Size: not applicable.
- SHA256: not applicable.
- GGUF architecture: hybrid, vision, MoE, or state-space variants require
  architecture work outside this M4 dense SLM path.
- Quantization: not applicable.
- Tokenizer authority: not applicable.
- Prompt template: not applicable.
- Rust support status: rejected for this lane.
- M4 support status: rejected for this lane.
- Cache policy: do not register as a Mac default or supported dense SLM model.
- Quality status: no M4 quality evidence.
- Claim boundary: not part of the Apple M4 dense SLM appliance path.

### Random or Unpinned Community GGUFs

- State: `rejected`.
- Source: must become exact, trusted, and pinned before reconsideration.
- Revision: missing.
- File: missing or ambiguous.
- Size: missing or unverified.
- SHA256: missing or unverified.
- GGUF architecture: unverified.
- Quantization: unverified.
- Tokenizer authority: unverified.
- Prompt template: unverified.
- Rust support status: rejected.
- M4 support status: rejected.
- Cache policy: never register or fetch by default.
- Quality status: rejected before quality testing.
- Claim boundary: artifact hygiene failure, not a model-quality judgment.

## Current Supported Set

The supported M4 dense SLM set currently contains two models:

```text
qwen2.5-0.5b-instruct-q8_0
qwen2.5-0.5b-instruct-q4_k_m
```

`qwen2.5-0.5b-instruct-q8_0` remains the default. The Q4_K_M artifact is the
first non-default supported dense SLM for the M4 lane because it passes the
reference, Rust M4 quality, tokenizer, cache, and receipt gates in this matrix.

## Next Breadth Candidates

`M4-MODEL-001` selected the next exact candidate set for evaluation:

```text
qwen3-0.6b-q8_0
smollm2-360m-instruct-q8_0
```

See [apple-m4-slm-model-breadth-candidates.md](apple-m4-slm-model-breadth-candidates.md)
for exact source, revision, file, expected size, tokenizer expectations, prompt
template expectations, storage budget, and rejection criteria. These candidates
are not supported models until reference output sanity, Rust M4 quality, cache,
receipt, and deterministic gates pass.
