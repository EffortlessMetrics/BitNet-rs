# CUDA-DENSE-039 KV Cache Policy Implementation

`CUDA-DENSE-039` adds the governed dense GGUF KV-cache policy receipt after
the model-boundary fixture work in `CUDA-DENSE-038`.

The new `dense_gguf_kv_cache_policy` receipt records:

- model-derived transformer layer count, context length, Q heads, KV heads, and
  key/value head dimensions;
- estimated KV bytes per token per layer and across all layers;
- prefill KV write policy;
- decode KV read/write policy;
- strict CUDA planned residency for future dense CUDA inference;
- the remaining sampling-policy gap before Qwen one-token proof.

This is still a policy receipt, not runtime KV allocation. It keeps runtime KV
cache allocation, measured transfer timing, dense GGUF inference, Qwen
one-token/short-decode/chat, speedup, persistent/full residency, server
readiness, BitNet packed proof, QK256, tokenizer, loader, transformer runtime,
and CUDA kernel math claims false.

## Committed Hardware Receipt

The committed receipt was emitted from the verified Qwen2.5 0.5B Q8_0 GGUF on
the Windows 9950X3D + RTX 5070 Ti machine:

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-kv-cache-policy-qwen25-q8.json
```

Observed receipt summary:

```text
model_sha256: ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e
architecture: qwen2
layers: 24
context_length: 32768
q_heads: 14
kv_heads: 2
key_head_dim: 64
value_head_dim: 64
kv_bytes_per_token_per_layer: 512
kv_bytes_per_token_all_layers: 12288
prefill_write_bytes_estimate: 49152
decode_read_bytes_per_step_estimate: 49152
decode_write_bytes_per_step_estimate: 12288
max_context_bytes_estimate: 402653184
planned_residency: cuda_required_for_strict_dense_cuda
observed_residency: not_allocated_policy_only
```

## Validation

```text
cargo fmt -p bitnet-cli -p bitnet-receipts-core -p bitnet-receipts -- --check
cargo test --locked -p bitnet-cli --lib --no-default-features --features cpu,cuda,full-cli dense_gguf_kv_cache -- --nocapture
cargo test --locked -p bitnet-receipts --test cuda_receipt_validation --no-default-features kv_cache -- --nocapture
```

Additional campaign checks are recorded by the PR validation.

## Claim Boundary

May claim:

- dense GGUF KV-cache policy is governed for the verified Qwen artifact;
- KV byte estimates are recorded for prefill, decode, and max context;
- sampling policy is now the next model-boundary gate before Qwen one-token
  proof.

Must not claim:

- runtime KV cache allocation exists;
- KV cache is actually CUDA resident at runtime;
- dense GGUF inference, Qwen one-token/short-decode/chat, speedup,
  persistent/full residency, or server readiness exists;
- dense regular-LLM CUDA evidence proves BitNet packed I2S/QK256 inference.
