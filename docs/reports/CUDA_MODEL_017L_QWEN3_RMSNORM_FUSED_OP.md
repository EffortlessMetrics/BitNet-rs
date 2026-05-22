# CUDA-MODEL-017L Qwen3 RMSNorm Fused Op Diagnostic

Status: diagnostic fused RMSNorm bypass added; CUDA still stops before the
first layer-0 fused RMSNorm operation finishes

## Scope

`CUDA-MODEL-017` requires repeated same-artifact Qwen3 CPU/CUDA comparator
receipts for the exact Qwen3 0.6B Q8_0 artifact on the Windows 9950X3D +
RTX 5070 Ti lane.

`CUDA-MODEL-017K` showed that the strict CUDA one-token path still stopped
before the first layer-0 RMSNorm square operation finished, even after the
trace-only expression changed from `x.sqr()` to `x * x`.

This follow-up bypasses the decomposed trace-only RMSNorm square/sum/divide path
by calling Candle's existing fused `ops::rms_norm` operation for RMSNorm-without-
bias trace captures. This is diagnostic routing only. It does not promote model
coverage, receipts, route selection, server readiness, speed, residency, or
proof-family booleans.

## Change

When Qwen tracing is enabled and a block attention norm is RMSNorm without bias,
the traced RMSNorm path now:

```text
1. records dtype conversion into the existing internal dtype trace fields;
2. calls candle_nn::ops::rms_norm(input, weight, eps);
3. records fused RMSNorm start/finish trace events.
```

The trace events for the bypass are:

```text
block.attention_norm_fused_rms_start
block.attention_norm_fused_rms_finish
```

The normal non-trace execution path still uses the existing Candle
`LayerNorm::forward` path. LayerNorm-with-mean and bias paths also keep using
the existing Candle path.

A CPU fixture test asserts that Candle's fused `ops::rms_norm` output matches
the `LayerNorm::rms_norm(...).forward(...)` semantics used by the current
transformer norm wrapper.

## Environment Observed

- GPU: NVIDIA GeForce RTX 5070 Ti
- Model: `Qwen3-0.6B-Q8_0.gguf`
- Local model path:
  `C:\Users\steven\AppData\Local\bitnet-rs\models\qwen3-0.6b-instruct-q8_0\Qwen3-0.6B-Q8_0.gguf`
- Built binary:
  `D:\codex-targets\bitnet-qwen-rmsnorm-fused-op-cuda\debug\bitnet.exe`
- CUDA build environment: Visual Studio 2022 x64 developer environment and
  CUDA 12.9 runtime path.

## Commands

Focused validation:

```powershell
cargo fmt -p bitnet-transformer -- --check
git diff --check
cargo test --locked -p bitnet-transformer --no-default-features --features cpu rmsnorm_fused_ops_matches_layernorm_rmsnorm_for_trace_path
```

The CUDA debug CLI was built with an isolated target directory:

```powershell
cargo build --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli
```

The bounded diagnostic run used the normal one-token strict CUDA capture command
with the Qwen3 prerequisite receipts from the 2026-05-15 RTX 5070 Ti campaign:

```powershell
D:\codex-targets\bitnet-qwen-rmsnorm-fused-op-cuda\debug\bitnet.exe dense-gguf-qwen-one-token-strict-cuda `
  --model C:\Users\steven\AppData\Local\bitnet-rs\models\qwen3-0.6b-instruct-q8_0\Qwen3-0.6B-Q8_0.gguf `
  --all-layer-plan ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-cuda-all-layer-plan.json `
  --model-boundary-fixtures ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-model-boundary-fixtures.json `
  --kv-cache-policy ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-kv-cache-policy.json `
  --sampling-policy ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-sampling-policy.json `
  --top-k 10 `
  --device-index 0 `
  --json-out D:\codex-runs\cuda-model-017l-rmsnorm-fused-op-1\qwen3-one-token.json `
  --phase-trace-jsonl D:\codex-runs\cuda-model-017l-rmsnorm-fused-op-1\qwen3-one-token-phase-trace.jsonl
```

The transformer trace was written next to the phase trace:

```text
D:\codex-runs\cuda-model-017l-rmsnorm-fused-op-1\qwen3-one-token-phase-trace.transformer.jsonl
```

## Result

No proof receipt JSON was emitted.

The phase trace contained 705 JSONL events. The transformer trace contained
14,459 JSONL events. The final phase events were:

```text
model_load_finish model_load_ms=14869.8537 elapsed_ms=32502.6846
kv_cache_start required_seq_len=8 elapsed_ms=32503.8865
kv_cache_finish elapsed_ms=32505.5445
prefill_start prefill_tokens=7 elapsed_ms=32506.0245
prefill_embed_start prefill_index=0 elapsed_ms=32506.6397
prefill_embed_finish embed_ms=1.942 embedding_is_cuda=true prefill_index=0 elapsed_ms=32508.5739
prefill_forward_start prefill_index=0 elapsed_ms=32509.2612
```

The transformer trace reached the CUDA target's first layer-0 fused RMSNorm
start:

```text
model.forward_start step=0 dims=[1,1,1024] device=cuda layers=28
model.forward_layer_start step=0 layer=0 dims=[1,1,1024]
block.forward_start step=0 layer=0 dims=[1,1,1024] device=cuda
block.attention_norm_start step=0 layer=0 dims=[1,1,1024] dtype=F32 device=cuda remove_mean=false bias_present=false
block.attention_norm_manual_start step=0 layer=0 path=rms_norm_manual_trace hidden_size=1024 input_dtype=F32 internal_dtype=F32 eps=0.000001 weight_dims=[1024] weight_device=cuda
block.attention_norm_to_dtype_start step=0 layer=0 from=F32 to=F32
block.attention_norm_to_dtype_finish step=0 layer=0 op_ms=0.0796 dims=[1,1,1024] dtype=F32 device=cuda
block.attention_norm_fused_rms_start step=0 layer=0 path=candle_ops_rms_norm hidden_size=1024 input_dims=[1,1,1024] input_dtype=F32 weight_dims=[1024] weight_dtype=F32 eps=0.000001 device=cuda
```

There was no CUDA-target `block.attention_norm_fused_rms_finish`, attention
forward event, decode event, logits event, or final receipt. The earlier CPU
comparator portion of the same command did produce fused RMSNorm finish events,
so the missing finish is specific to the CUDA target stage.

The bounded wrapper returned without a receipt. Process inspection showed a
stale `bitnet.exe` process object for the child; `taskkill` reported there was
no running task instance for that PID after terminating the child process.

## Blocker Classification

If the strict CUDA run moves past `block.attention_norm_fused_rms_finish`, the
previous frontier is narrowed to the decomposed CUDA RMSNorm elementwise/reduce
sequence rather than Qwen3 tensor loading, KV-cache construction, embedding, or
block entry.

This run stopped before `block.attention_norm_fused_rms_finish`, so the frontier
moved to Candle's fused CUDA RMSNorm reduce kernel or the lower CUDA runtime
synchronization path used by both fused and decomposed RMSNorm variants.

## Claim Boundary

This report and code path are diagnostic only. The following remain false:

- `speedup_claim`
- `benchmark_qualified_speedup`
- `server_ready`
- `full_cuda_residency_claimed`
- broad dense GGUF readiness
- Qwen2.5 proof inheritance for Qwen3
- BitNet packed I2_S/QK256 proof
