# CUDA-DENSE-040 Sampling Policy Implementation

`CUDA-DENSE-040` adds the governed dense GGUF logits-transfer and sampling
policy receipt after the KV-cache policy work in `CUDA-DENSE-039`.

The new `dense_gguf_sampling_policy` receipt records:

- LM-head fixture logits source, length, SHA-256, and top-k diagnostics;
- estimated device-to-host logits bytes per decode step;
- deterministic greedy CPU sampler settings;
- lowest-token-ID tie-break policy;
- the next proof gate after model-boundary policies are governed;
- claim-boundary rejection of runtime sampling integration, dense inference,
  Qwen token/decode/chat, speedup, full residency, and BitNet packed proof.

This is still a policy receipt, not runtime sampling integration or token
generation.

## Committed Hardware Receipt

The committed receipt was emitted from the verified Qwen2.5 0.5B Q8_0 GGUF on
the Windows 9950X3D + RTX 5070 Ti machine:

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-sampling-policy-qwen25-q8.json
```

Observed receipt summary:

```text
model_sha256: ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e
architecture: qwen2
logits_len: 151936
vocab_size: 151936
logits_sha256: 94b94c11e4b89d80b03989840b54efc73a8c31bd32a4c1425d73222876f7677a
logits_element_bytes: 4
logits_transfer_bytes_per_step_estimate: 607744
sampler_backend: bitnet-sampling
sampler_location: cpu
sampler_mode: greedy
temperature: 0.0
top_k_filter: 0
top_p: 1.0
repetition_penalty: 1.0
tie_break_policy: lowest_token_id
selected_token_id_from_fixture_logits: 3
next_required_proof: qwen_one_token_strict_cuda_proof
```

## Validation

```text
cargo fmt -p bitnet-cli -p bitnet-receipts-core -p bitnet-receipts -- --check
cargo test --locked -p bitnet-cli --lib --no-default-features --features cpu,cuda,full-cli dense_gguf_sampling -- --nocapture
cargo test --locked -p bitnet-receipts --test cuda_receipt_validation --no-default-features sampling_policy -- --nocapture
```

Additional campaign checks are recorded by the PR validation.

## Claim Boundary

May claim:

- dense GGUF logits-transfer and sampling policy is governed for the verified
  Qwen artifact;
- the receipt records fixture logits hash/top-k evidence, greedy CPU sampler
  settings, and estimated logits transfer bytes;
- the next dense CUDA proof gate is Qwen one-token strict CUDA proof.

Must not claim:

- runtime sampling integration exists;
- dense GGUF inference, Qwen one-token/short-decode/chat, speedup,
  persistent/full residency, or server readiness exists;
- dense regular-LLM CUDA evidence proves BitNet packed I2S/QK256 inference.
