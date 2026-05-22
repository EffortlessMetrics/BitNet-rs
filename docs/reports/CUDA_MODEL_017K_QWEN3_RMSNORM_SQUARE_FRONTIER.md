# CUDA-MODEL-017K Qwen3 RMSNorm Square Frontier

Status: diagnostic replacement added; CUDA still stops before the RMSNorm square
operation finishes

## Scope

`CUDA-MODEL-017` requires repeated same-artifact Qwen3 CPU/CUDA comparator
receipts for the exact Qwen3 0.6B Q8_0 artifact on the Windows 9950X3D +
RTX 5070 Ti lane.

`CUDA-MODEL-017J` moved the strict CUDA one-token trace to the first layer-0
RMSNorm square over a `[1, 1, 1024]` F32 CUDA tensor. This follow-up replaces
the trace-only RMSNorm square expression from `x.sqr()` to `x * x` so the run
can distinguish a unary `sqr()` frontier from a broader CUDA elementwise square
frontier.

This is diagnostic observability only for the Qwen trace path. It does not
promote model coverage, receipts, route selection, server readiness, speed,
residency, or proof-family booleans.

## Change

When Qwen tracing is enabled and a block attention norm is RMSNorm without bias,
the traced manual RMSNorm path now computes the square step as:

```text
squared = norm_input * norm_input
```

instead of:

```text
squared = norm_input.sqr()
```

The trace events for this step are now:

```text
block.attention_norm_square_mul_start
block.attention_norm_square_mul_finish
```

The normal non-trace execution path still uses the existing Candle
`LayerNorm::forward` path. LayerNorm-with-mean and bias paths also keep using
the existing Candle path.

A CPU fixture test asserts that `x * x` matches `x.sqr()` for the traced RMSNorm
math.

## Environment Observed

- GPU: NVIDIA GeForce RTX 5070 Ti
- Model: `Qwen3-0.6B-Q8_0.gguf`
- Local model path:
  `C:\Users\steven\AppData\Local\bitnet-rs\models\qwen3-0.6b-instruct-q8_0\Qwen3-0.6B-Q8_0.gguf`
- Built binary:
  `D:\codex-targets\bitnet-qwen-rmsnorm-square-frontier-cuda\debug\bitnet.exe`
- CUDA build environment: Visual Studio 2022 x64 developer environment and
  CUDA 12.9 runtime path.

## Commands

Focused validation:

```powershell
cargo fmt -p bitnet-transformer -- --check
git diff --check
cargo test --locked -p bitnet-transformer --no-default-features --features cpu rmsnorm_square_mul_matches_sqr_for_trace_path
```

The CUDA debug CLI was built with an isolated target directory:

```powershell
cargo build --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli
```

The bounded diagnostic run used the normal one-token strict CUDA capture command
with the Qwen3 prerequisite receipts from the 2026-05-15 RTX 5070 Ti campaign:

```powershell
D:\codex-targets\bitnet-qwen-rmsnorm-square-frontier-cuda\debug\bitnet.exe dense-gguf-qwen-one-token-strict-cuda `
  --model C:\Users\steven\AppData\Local\bitnet-rs\models\qwen3-0.6b-instruct-q8_0\Qwen3-0.6B-Q8_0.gguf `
  --all-layer-plan ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-cuda-all-layer-plan.json `
  --model-boundary-fixtures ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-model-boundary-fixtures.json `
  --kv-cache-policy ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-kv-cache-policy.json `
  --sampling-policy ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-sampling-policy.json `
  --top-k 10 `
  --device-index 0 `
  --json-out D:\codex-runs\cuda-model-017k-rmsnorm-square-frontier-1\qwen3-one-token.json `
  --phase-trace-jsonl D:\codex-runs\cuda-model-017k-rmsnorm-square-frontier-1\qwen3-one-token-phase-trace.jsonl
```

The transformer trace was written next to the phase trace:

```text
D:\codex-runs\cuda-model-017k-rmsnorm-square-frontier-1\qwen3-one-token-phase-trace.transformer.jsonl
```

## Result

No proof receipt JSON was emitted.

The phase trace contained 705 JSONL events. The transformer trace contained
16,699 JSONL events. The final phase events were:

```text
model_load_finish model_load_ms=13390.6251 elapsed_ms=28835.1521
kv_cache_start required_seq_len=8 elapsed_ms=28835.9332
kv_cache_finish elapsed_ms=28837.6841
prefill_start prefill_tokens=7 elapsed_ms=28838.275
prefill_embed_start prefill_index=0 elapsed_ms=28838.875
prefill_embed_finish embed_ms=1.7645 embedding_is_cuda=true prefill_index=0 elapsed_ms=28840.6381
prefill_forward_start prefill_index=0 elapsed_ms=28841.103
```

The transformer trace reached:

```text
model.forward_start step=0 dims=[1,1,1024] device=cuda layers=28
model.forward_layer_start step=0 layer=0 dims=[1,1,1024]
block.forward_start step=0 layer=0 dims=[1,1,1024] device=cuda
block.attention_norm_start step=0 layer=0 dims=[1,1,1024] dtype=F32 device=cuda remove_mean=false bias_present=false
block.attention_norm_manual_start step=0 layer=0 path=rms_norm_manual_trace hidden_size=1024 input_dtype=F32 internal_dtype=F32 eps=0.000001 weight_dims=[1024] weight_device=cuda
block.attention_norm_to_dtype_start step=0 layer=0 from=F32 to=F32
block.attention_norm_to_dtype_finish step=0 layer=0 op_ms=0.0801 dims=[1,1,1024] dtype=F32 device=cuda
block.attention_norm_square_mul_start step=0 layer=0 dims=[1,1,1024] device=cuda method=mul_self
```

There was no `block.attention_norm_square_mul_finish`, reduction event,
attention forward event, decode event, logits event, or final receipt.

The bounded wrapper returned nonzero after the timeout window. Process
inspection showed a stale `bitnet.exe` process object for the child, but
`taskkill` reported there was no running task instance for that PID.

## Blocker Classification

`CUDA-MODEL-017K` keeps the trace frontier inside the first layer-0 RMSNorm
square:

```text
CUDA tensor loading completes
CUDA model construction completes
CUDA KV cache is created
prefill loop starts
prefill token 0 embedding completes
transformer forward starts
layer 0 attention norm starts
RMSNorm metadata and dtype path are confirmed
bounded run stops before x * x finishes on [1, 1, 1024] F32 CUDA
```

Replacing unary `sqr()` with a self-multiply did not move the frontier. The next
blocker is therefore not just the public `Tensor::sqr()` call. It is the CUDA
elementwise square operation over the single-token hidden state, or a lower
runtime synchronization issue exposed by both square spellings.

The next diagnostic PR should inspect or bypass Candle CUDA elementwise square
for this RMSNorm step, preferably with a dedicated RMSNorm/sum-of-squares path
or a smaller CUDA elementwise fixture before moving on to reductions,
denominator construction, attention projections, RoPE application, KV append,
attention scores, softmax, value mix, or output projection.

## Claim Boundary

This report proves only that current-source diagnostic tracing identifies the
Qwen3 strict CUDA one-token layer-0 RMSNorm square frontier and that replacing
`sqr()` with `x * x` does not move it. The following remain false:

- `speedup_claim`
- `benchmark_qualified_speedup`
- `server_ready`
- `full_cuda_residency_claimed`
- broad dense GGUF readiness
- Qwen2.5 proof inheritance for Qwen3
- BitNet packed I2_S/QK256 proof
