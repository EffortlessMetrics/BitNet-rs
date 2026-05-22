# CUDA-MODEL-017I Qwen3 Single-Token Embed Narrow

Status: diagnostic bypass added; first layer attention norm is the next
frontier

## Scope

`CUDA-MODEL-017` requires repeated same-artifact Qwen3 CPU/CUDA comparator
receipts for the exact Qwen3 0.6B Q8_0 artifact on the Windows 9950X3D +
RTX 5070 Ti lane.

`CUDA-MODEL-017H` localized the strict CUDA one-token run to the standard
embedding row-gather `index_select` path. This follow-up changes only the
standard `[vocab, hidden]` single-token embedding path to gather the selected
row with `narrow` instead of Candle CUDA `index_select`.

This is a diagnostic runtime frontier change only. It does not promote model
coverage, receipts, route selection, server readiness, speed, residency, or
proof-family booleans.

## Change

For `TransformerModel::embed` when all of the following are true:

```text
tokens.len() == 1
embed_transposed == false
embedding weight layout is [vocab, hidden]
```

the implementation now uses:

```text
weight.narrow(0, token_id, 1)
```

and then reshapes the result to `[1, 1, hidden]`.

The existing multi-token row-gather path still uses `index_select`. The
transposed embedding path is unchanged.

The Qwen trace now emits the single-token narrow boundary:

```text
model.embed_single_token_narrow_start
model.embed_single_token_narrow_finish
```

These events include only dimensions, dtype, path, device label, and timings.
They do not include prompt text or token IDs.

## Environment Observed

- GPU: NVIDIA GeForce RTX 5070 Ti
- Model: `Qwen3-0.6B-Q8_0.gguf`
- Local model path:
  `C:\Users\steven\AppData\Local\bitnet-rs\models\qwen3-0.6b-instruct-q8_0\Qwen3-0.6B-Q8_0.gguf`
- Built binary:
  `D:\codex-targets\bitnet-qwen-single-token-narrow-cuda\debug\bitnet.exe`
- CUDA build environment: Visual Studio 2022 x64 developer environment and
  CUDA 12.9 runtime path.

## Commands

Focused validation:

```powershell
cargo fmt -p bitnet-transformer -- --check
git diff --check
cargo test --locked -p bitnet-transformer --no-default-features --features cpu embed
```

The CUDA debug CLI was built with an isolated target directory:

```powershell
cargo build --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli
```

The bounded diagnostic run used the normal one-token strict CUDA capture command
with the Qwen3 prerequisite receipts from the 2026-05-15 RTX 5070 Ti campaign:

```powershell
D:\codex-targets\bitnet-qwen-single-token-narrow-cuda\debug\bitnet.exe dense-gguf-qwen-one-token-strict-cuda `
  --model C:\Users\steven\AppData\Local\bitnet-rs\models\qwen3-0.6b-instruct-q8_0\Qwen3-0.6B-Q8_0.gguf `
  --all-layer-plan ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-cuda-all-layer-plan.json `
  --model-boundary-fixtures ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-model-boundary-fixtures.json `
  --kv-cache-policy ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-kv-cache-policy.json `
  --sampling-policy ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-sampling-policy.json `
  --top-k 10 `
  --device-index 0 `
  --json-out D:\codex-runs\cuda-model-017i-single-token-narrow-1\qwen3-one-token.json `
  --phase-trace-jsonl D:\codex-runs\cuda-model-017i-single-token-narrow-1\qwen3-one-token-phase-trace.jsonl
```

The transformer trace was written next to the phase trace:

```text
D:\codex-runs\cuda-model-017i-single-token-narrow-1\qwen3-one-token-phase-trace.transformer.jsonl
```

## Result

No proof receipt JSON was emitted.

The phase trace contained 705 JSONL events. The transformer trace contained
13,335 JSONL events. The final phase events were:

```text
model_load_finish model_load_ms=13871.1593 elapsed_ms=28672.6953
kv_cache_start required_seq_len=8 elapsed_ms=28673.6817
kv_cache_finish elapsed_ms=28675.3734
prefill_start prefill_tokens=7 elapsed_ms=28676.2143
prefill_embed_start prefill_index=0 elapsed_ms=28676.8335
prefill_embed_finish embed_ms=1.8897 embedding_is_cuda=true prefill_index=0 elapsed_ms=28678.7153
prefill_forward_start prefill_index=0 elapsed_ms=28679.5529
```

The 017H frontier was passed. The transformer trace reached:

```text
model.embed_single_token_narrow_start path=row_gather_single_token_narrow dim=0 weight_dims=[151936,1024] device=cuda
model.embed_single_token_narrow_finish op_ms=0.1124 path=row_gather_single_token_narrow dims=[1,1024] dtype=F32 device=cuda
model.embed_reshape_start target_dims=[1,1,1024] device=cuda
model.embed_reshape_finish op_ms=0.0799 dims=[1,1,1024] dtype=F32 device=cuda
model.embed_finish path=row_gather dims=[1,1,1024] device=cuda
model.forward_start step=0 dims=[1,1,1024] device=cuda layers=28
model.forward_layer_start step=0 layer=0 dims=[1,1,1024]
block.forward_start step=0 layer=0 dims=[1,1,1024] device=cuda
block.attention_norm_start step=0 layer=0
```

There was no `block.attention_norm_finish`, attention forward event, decode
event, logits event, or final receipt.

The bounded wrapper returned nonzero and the diagnostic process required cleanup
after the trace stopped advancing. Windows process inspection later reported no
running instance for the `bitnet.exe` child. No further CUDA run was attempted
from this branch.

## Blocker Classification

`CUDA-MODEL-017I` moves the trace frontier from embedding row-gather into the
first transformer block:

```text
CUDA tensor loading completes
CUDA model construction completes
CUDA KV cache is created
prefill loop starts
prefill token 0 embedding starts
single-token CUDA row narrow completes
embedding reshape completes
transformer forward starts
layer 0 forward starts
layer 0 attention norm starts
bounded run stops before attention norm finishes
```

The next blocker is the first layer-0 attention norm forward call over the
single-token CUDA hidden state `[1, 1, 1024]`.

The next diagnostic PR should instrument or isolate `block.attention_norm` on
CUDA before spending effort on attention projections, RoPE application, KV
append, attention scores, softmax, value mix, or output projection.

## Claim Boundary

This report proves only that current-source diagnostic work bypasses the Qwen3
strict CUDA one-token embedding `index_select` frontier and identifies the next
runtime frontier at layer-0 attention norm. The following remain false:

- `speedup_claim`
- `benchmark_qualified_speedup`
- `server_ready`
- `full_cuda_residency_claimed`
- broad dense GGUF readiness
- Qwen2.5 proof inheritance for Qwen3
- BitNet packed I2_S/QK256 proof
