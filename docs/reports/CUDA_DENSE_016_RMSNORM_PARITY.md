# CUDA-DENSE-016 Dense GGUF RMSNorm CUDA Parity

`CUDA-DENSE-016` adds the first dense GGUF non-linear CUDA parity fixture. It
extracts the verified Qwen2.5 0.5B Q8_0 `attention_norm` and `ffn_norm`
RMSNorm fixtures, runs them through a strict RTX 5070 Ti CUDA F32 RMSNorm
kernel, and validates the result against deterministic CPU references.

This is still fixture-level evidence. It is not dense GGUF inference, not a
Qwen one-token/decode/chat proof, not a server path, not a speedup claim, and
not BitNet packed I2_S/QK256 proof.

## Receipt

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-09/dense-gguf-rmsnorm-cuda-parity-qwen25-q8.json
```

## Evidence

| Field | Value |
| --- | --- |
| `artifact_kind` | `dense_gguf_norm_cuda_parity` |
| `claim` | `dense_gguf_norm_cuda_parity_tested` |
| `model.model_family` | `qwen` |
| `model.architecture` | `qwen2` |
| `execution_plan.selected_route` | `dense_regular_llm_cuda` |
| `execution_plan.cuda_dense_regular_llm_ops` | `2` |
| `kernel_stats[*].kernel_id` | `dense_rmsnorm_f32_cuda` |
| `parity.passed` | `true` |
| `parity.covered_roles` | `attention_norm`, `ffn_norm` |
| `timing.host_to_device_bytes` | `14336` |
| `timing.device_to_host_bytes` | `7168` |
| `fallback_used` | `false` |
| `speedup_claim` | `false` |
| `claim_boundary.dense_gguf_inference_claimed` | `false` |
| `claim_boundary.bitnet_packed_i2s_qk256_proof` | `false` |

## May Claim

- Dense GGUF `attention_norm` and `ffn_norm` RMSNorm fixtures from the verified
  Qwen2.5 0.5B Q8_0 artifact pass strict RTX 5070 Ti CUDA parity against CPU
  references.
- The receipt records `dense_regular_llm_cuda` routing, `fallback_used=false`,
  measured host/device byte counts, and `dense_rmsnorm_f32_cuda` kernel use.
- Dense RMSNorm CUDA parity remains rejected as BitNet packed I2_S/QK256 proof.

## Must Not Claim

- dense GGUF inference works;
- Qwen one-token, short decode, or chat works on CUDA;
- dense CUDA speedup is accepted;
- full dense CUDA residency is proven;
- dense regular-LLM CUDA proves BitNet packed I2_S/QK256 inference;
- tokenizer, loader, transformer, server, or QK256 math changed.

## Validation

```powershell
cargo test --locked -p bitnet-kernels --lib --no-default-features --features cuda rmsnorm -- --nocapture
cargo test --locked -p bitnet-cli --lib --no-default-features --features cpu,cuda,full-cli dense_gguf_norm -- --nocapture
cargo test --locked -p bitnet-receipts --test cuda_receipt_validation --no-default-features dense_gguf_norm -- --nocapture
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9'
$env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- dense-gguf-norm-cuda-parity --model <verified-qwen2.5-q8-gguf> --json-out ci\hardware\windows-9950x3d-rtx5070ti\2026-05-09\dense-gguf-rmsnorm-cuda-parity-qwen25-q8.json
cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- receipts explain ci\hardware\windows-9950x3d-rtx5070ti\2026-05-09\dense-gguf-rmsnorm-cuda-parity-qwen25-q8.json
```

## Next Step

The next scoped dense CUDA lane should move to the next one-layer gap only
after updating the planner gap state to mark RMSNorm CUDA parity available.
The likely next proof is RoPE fixture parity; it should remain below dense
GGUF one-token, decode, chat, server, speedup, and full-residency claims.
