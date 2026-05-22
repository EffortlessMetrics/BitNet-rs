# CUDA-MODEL-017J Qwen3 Attention Norm Frontier Trace

Status: diagnostic trace added; first RMSNorm square operation is the next
frontier

## Scope

`CUDA-MODEL-017` requires repeated same-artifact Qwen3 CPU/CUDA comparator
receipts for the exact Qwen3 0.6B Q8_0 artifact on the Windows 9950X3D +
RTX 5070 Ti lane.

`CUDA-MODEL-017I` bypassed the strict CUDA one-token embedding `index_select`
frontier and moved the run into layer-0 transformer forward. This follow-up
splits the layer-0 attention norm frontier by tracing the RMSNorm sub-ops when
Qwen diagnostic tracing is enabled.

This is diagnostic observability only for the Qwen trace path. It does not
promote model coverage, receipts, route selection, server readiness, speed,
residency, or proof-family booleans.

## Change

When Qwen tracing is enabled and a block attention norm is RMSNorm without bias,
`TransformerBlock::forward_impl` now traces the RMSNorm operation as explicit
sub-steps:

```text
block.attention_norm_start
block.attention_norm_manual_start
block.attention_norm_to_dtype_start
block.attention_norm_to_dtype_finish
block.attention_norm_sqr_start
block.attention_norm_sqr_finish
block.attention_norm_sum_start
block.attention_norm_sum_finish
block.attention_norm_denom_start
block.attention_norm_denom_finish
block.attention_norm_div_start
block.attention_norm_div_finish
block.attention_norm_output_cast_start
block.attention_norm_output_cast_finish
block.attention_norm_weight_mul_start
block.attention_norm_weight_mul_finish
block.attention_norm_finish
```

The sub-ops mirror Candle RMSNorm math:

```text
x = x.to_dtype(internal_dtype)
norm_x = sum_keepdim(x.sqr(), -1) / hidden_size
x_normed = x / sqrt(norm_x + eps)
output = x_normed.to_dtype(input_dtype) * weight
```

Non-trace execution keeps using the existing Candle `LayerNorm::forward` path.
LayerNorm-with-mean and bias paths also keep using the existing Candle path.

The events include dimensions, dtype, device label, RMSNorm mode, epsilon, and
elapsed timings. They do not include prompt text or token IDs.

## Environment Observed

- GPU: NVIDIA GeForce RTX 5070 Ti
- Model: `Qwen3-0.6B-Q8_0.gguf`
- Local model path:
  `C:\Users\steven\AppData\Local\bitnet-rs\models\qwen3-0.6b-instruct-q8_0\Qwen3-0.6B-Q8_0.gguf`
- Built binary:
  `D:\codex-targets\bitnet-qwen-attention-norm-frontier-cuda\debug\bitnet.exe`
- CUDA build environment: Visual Studio 2022 x64 developer environment and
  CUDA 12.9 runtime path.

## Commands

Focused validation:

```powershell
cargo fmt -p bitnet-transformer -- --check
git diff --check
cargo test --locked -p bitnet-transformer --no-default-features --features cpu forward_full_single_token_produces_1_1_vocab_logits
```

The CUDA debug CLI was built with an isolated target directory:

```powershell
cargo build --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli
```

The bounded diagnostic run used the normal one-token strict CUDA capture command
with the Qwen3 prerequisite receipts from the 2026-05-15 RTX 5070 Ti campaign:

```powershell
D:\codex-targets\bitnet-qwen-attention-norm-frontier-cuda\debug\bitnet.exe dense-gguf-qwen-one-token-strict-cuda `
  --model C:\Users\steven\AppData\Local\bitnet-rs\models\qwen3-0.6b-instruct-q8_0\Qwen3-0.6B-Q8_0.gguf `
  --all-layer-plan ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-cuda-all-layer-plan.json `
  --model-boundary-fixtures ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-model-boundary-fixtures.json `
  --kv-cache-policy ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-kv-cache-policy.json `
  --sampling-policy ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-sampling-policy.json `
  --top-k 10 `
  --device-index 0 `
  --json-out D:\codex-runs\cuda-model-017j-attention-norm-frontier-1\qwen3-one-token.json `
  --phase-trace-jsonl D:\codex-runs\cuda-model-017j-attention-norm-frontier-1\qwen3-one-token-phase-trace.jsonl
```

The transformer trace was written next to the phase trace:

```text
D:\codex-runs\cuda-model-017j-attention-norm-frontier-1\qwen3-one-token-phase-trace.transformer.jsonl
```

## Result

No proof receipt JSON was emitted.

The phase trace contained 705 JSONL events. The transformer trace contained
16,699 JSONL events. The final phase events were:

```text
model_load_finish model_load_ms=14600.1945 elapsed_ms=30423.5581
kv_cache_start required_seq_len=8 elapsed_ms=30424.3301
kv_cache_finish elapsed_ms=30425.7446
prefill_start prefill_tokens=7 elapsed_ms=30426.2257
prefill_embed_start prefill_index=0 elapsed_ms=30426.6505
prefill_embed_finish embed_ms=1.4997 embedding_is_cuda=true prefill_index=0 elapsed_ms=30428.1508
prefill_forward_start prefill_index=0 elapsed_ms=30428.592
```

The transformer trace reached:

```text
model.forward_start step=0 dims=[1,1,1024] device=cuda layers=28
model.forward_layer_start step=0 layer=0 dims=[1,1,1024]
block.forward_start step=0 layer=0 dims=[1,1,1024] device=cuda
block.attention_norm_start step=0 layer=0 dims=[1,1,1024] dtype=F32 device=cuda remove_mean=false bias_present=false
block.attention_norm_manual_start step=0 layer=0 path=rms_norm_manual_trace hidden_size=1024 input_dtype=F32 internal_dtype=F32 eps=0.000001 weight_dims=[1024] weight_device=cuda
block.attention_norm_to_dtype_start step=0 layer=0 from=F32 to=F32
block.attention_norm_to_dtype_finish step=0 layer=0 op_ms=0.0783 dims=[1,1,1024] dtype=F32 device=cuda
block.attention_norm_sqr_start step=0 layer=0 dims=[1,1,1024] device=cuda
```

There was no `block.attention_norm_sqr_finish`, attention forward event, decode
event, logits event, or final receipt.

The bounded wrapper returned nonzero and the diagnostic process required cleanup
after the trace stopped advancing. Windows process inspection later reported no
running instance for the `bitnet.exe` child. No further CUDA run was attempted
from this branch.

## Blocker Classification

`CUDA-MODEL-017J` moves the trace frontier inside the first layer-0 RMSNorm:

```text
CUDA tensor loading completes
CUDA model construction completes
CUDA KV cache is created
prefill loop starts
prefill token 0 embedding completes
transformer forward starts
layer 0 attention norm starts
RMSNorm metadata and dtype path are confirmed
bounded run stops before CUDA sqr() finishes on [1, 1, 1024] F32
```

The next blocker is the first CUDA elementwise square used by RMSNorm over the
single-token hidden state.

The next diagnostic PR should isolate or replace this traced RMSNorm `sqr()`
frontier before spending effort on reductions, denominator construction,
attention projections, RoPE application, KV append, attention scores, softmax,
value mix, or output projection.

## Claim Boundary

This report proves only that current-source diagnostic tracing identifies the
Qwen3 strict CUDA one-token layer-0 RMSNorm square frontier. The following
remain false:

- `speedup_claim`
- `benchmark_qualified_speedup`
- `server_ready`
- `full_cuda_residency_claimed`
- broad dense GGUF readiness
- Qwen2.5 proof inheritance for Qwen3
- BitNet packed I2_S/QK256 proof
