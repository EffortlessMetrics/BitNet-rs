# CUDA-DENSE-037 All-Layer Dense GGUF Plan Implementation

`CUDA-DENSE-037` implements the all-layer execution-plan receipt boundary
defined by `CUDA-DENSE-036`. The new CLI command is:

```powershell
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- `
  dense-gguf-all-layer-plan `
  --model <verified-qwen2.5-q8-gguf> `
  --device-index 0 `
  --json-out ci\hardware\windows-9950x3d-rtx5070ti\2026-05-09\dense-gguf-all-layer-plan-qwen25-q8.json
```

The command is descriptor and planner only. It does not execute a full dense
GGUF model pass, generate tokens, benchmark speed, change CUDA kernel math, or
change tokenizer, loader, transformer runtime, server, BitNet CUDA, or QK256
behavior.

## Receipt Boundary

The implementation emits and validates:

```text
artifact_kind: dense_gguf_all_layer_execution_plan
claim: dense_gguf_all_layer_execution_plan_recorded
selected_backend: nvidia-rtx-5070-ti-cuda
runtime_api: cuda
fallback_used: false
speedup_claim: false
dense_gguf_inference_claimed: false
qwen_one_token_cuda_claimed: false
full_cuda_residency_claimed: false
bitnet_packed_i2s_qk256_proof: false
```

The `all_layer_plan` section records every detected Qwen-family transformer
layer, verifies that each layer has the same 14 governed dense CUDA block ops as
the layer-0 parity path, and requires zero unsupported strict CUDA ops for the
accepted receipt.

## Model-Boundary Gaps

The receipt keeps the remaining full-model blockers explicit:

| Gap | Status |
| --- | --- |
| `token_embedding` | not governed by all-layer block planning |
| `final_norm` | not governed by all-layer block planning |
| `lm_head_logits` | not governed by all-layer block planning |
| `kv_cache_policy` | pending residency and transfer policy |
| `sampling` | pending sampler integration and claim boundary |

These gaps continue to block Qwen one-token, short decode, chat, benchmark, and
server claims.

## Validation

Validated in this implementation PR:

```powershell
cargo fmt -p bitnet-cli -p bitnet-receipts-core -p bitnet-receipts -- --check
cargo test --locked -p bitnet-cli --lib --no-default-features --features cpu,cuda,full-cli dense_gguf_all_layer -- --nocapture
cargo test --locked -p bitnet-cli --lib --no-default-features --features cpu,cuda,full-cli dense_gguf_one_layer -- --nocapture
cargo test --locked -p bitnet-receipts --test cuda_receipt_validation --no-default-features all_layer -- --nocapture
cargo check --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli
cargo run --release --locked -p xtask --no-default-features -- campaign check nvidia-5070ti
cargo run --release --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

The real hardware command remains a template until a verified Qwen2.5 Q8 GGUF
artifact is available in the checkout.

