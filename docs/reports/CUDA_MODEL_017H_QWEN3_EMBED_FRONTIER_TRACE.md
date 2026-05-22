# CUDA-MODEL-017H Qwen3 Embed Frontier Trace

Status: diagnostic trace added; first CUDA embedding `index_select` is the next frontier

## Scope

`CUDA-MODEL-017` requires repeated same-artifact Qwen3 CPU/CUDA comparator
receipts for the exact Qwen3 0.6B Q8_0 artifact on the Windows 9950X3D +
RTX 5070 Ti lane.

`CUDA-MODEL-017G` localized the strict CUDA one-token run to the first prefill
embedding lookup. This follow-up splits `TransformerModel::embed` into internal
trace events so the next run can distinguish token tensor creation, flattening,
embedding weight access, row/column gather, transpose, and reshape.

This is diagnostic observability only. It does not change runtime math,
sampling, route selection, model coverage, receipts, server readiness, speed
claims, residency claims, or proof-family booleans.

## Change

`TransformerModel::embed` now emits structured Qwen trace events for:

```text
model.embed_start
model.embed_token_tensor_start
model.embed_token_tensor_finish
model.embed_flatten_start
model.embed_flatten_finish
model.embed_weight_start
model.embed_weight_finish
model.embed_index_select_start
model.embed_index_select_finish
model.embed_transpose_start
model.embed_transpose_finish
model.embed_reshape_start
model.embed_reshape_finish
model.embed_finish
```

The events include counts, dimensions, dtype, device label, gather path, gather
dimension, and elapsed timings. They do not include prompt text or token IDs.

## Environment Observed

- GPU: NVIDIA GeForce RTX 5070 Ti
- Model: `Qwen3-0.6B-Q8_0.gguf`
- Local model path:
  `C:\Users\steven\AppData\Local\bitnet-rs\models\qwen3-0.6b-instruct-q8_0\Qwen3-0.6B-Q8_0.gguf`
- Built binary:
  `D:\codex-targets\bitnet-qwen-embed-frontier-cuda\debug\bitnet.exe`
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

The first CUDA build attempt failed before compiling CUDA kernels because
`nvcc` could not find `cl.exe`. The second attempt reached the linker but failed
because the command overrode Visual Studio `LIB` search paths and `link.exe`
could not find `ntdll.lib`. Re-running with the Visual Studio x64 environment
intact and only `CUDA_PATH` set passed.

The bounded diagnostic run used the normal one-token strict CUDA capture command
with the Qwen3 prerequisite receipts from the 2026-05-15 RTX 5070 Ti campaign:

```powershell
D:\codex-targets\bitnet-qwen-embed-frontier-cuda\debug\bitnet.exe dense-gguf-qwen-one-token-strict-cuda `
  --model C:\Users\steven\AppData\Local\bitnet-rs\models\qwen3-0.6b-instruct-q8_0\Qwen3-0.6B-Q8_0.gguf `
  --all-layer-plan ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-cuda-all-layer-plan.json `
  --model-boundary-fixtures ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-model-boundary-fixtures.json `
  --kv-cache-policy ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-kv-cache-policy.json `
  --sampling-policy ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-sampling-policy.json `
  --top-k 10 `
  --device-index 0 `
  --json-out D:\codex-runs\cuda-model-017h-embed-frontier-1\qwen3-one-token.json `
  --phase-trace-jsonl D:\codex-runs\cuda-model-017h-embed-frontier-1\qwen3-one-token-phase-trace.jsonl
```

The transformer trace was written next to the phase trace:

```text
D:\codex-runs\cuda-model-017h-embed-frontier-1\qwen3-one-token-phase-trace.transformer.jsonl
```

## Result

No proof receipt JSON was emitted.

The phase trace contained 703 JSONL events. The transformer trace contained
13,327 JSONL events. The final phase events were:

```text
model_load_finish model_load_ms=14529.5261 elapsed_ms=32576.0571
kv_cache_start required_seq_len=8 elapsed_ms=32577.0292
kv_cache_finish elapsed_ms=32578.8489
prefill_start prefill_tokens=7 elapsed_ms=32579.5732
prefill_embed_start prefill_index=0 elapsed_ms=32580.291
```

There was no `prefill_embed_finish`, `prefill_forward_start`,
`model.forward_start`, decode event, logits event, or final receipt.

The transformer trace reached the embedding internals and stopped at CUDA
`index_select`:

```text
model.embed_start token_count=1 hidden_size=1024 embed_transposed=false device=cuda
model.embed_token_tensor_start token_count=1 device=cuda
model.embed_token_tensor_finish op_ms=0.2125 dims=[1,1] dtype=U32 device=cuda
model.embed_flatten_start dims=[1,1] device=cuda
model.embed_flatten_finish op_ms=0.0896 dims=[1] dtype=U32 device=cuda
model.embed_weight_start path=row_gather device=cuda
model.embed_weight_finish op_ms=0.0819 path=row_gather dims=[151936,1024] dtype=F32 device=cuda
model.embed_index_select_start path=row_gather dim=0 weight_dims=[151936,1024] id_dims=[1] device=cuda
```

There was no `model.embed_index_select_finish`, reshape event, `model.embed_finish`,
or `model.forward_start`.

The local bounded wrapper did not terminate the CUDA process cleanly after the
trace stopped advancing. Manual termination attempts left the process visible
through Windows process inspection. No further CUDA run was attempted from this
branch.

## Blocker Classification

`CUDA-MODEL-017H` moves the trace frontier inside the embedding lookup:

```text
CUDA tensor loading completes
CUDA model construction completes
CUDA KV cache is created
prefill loop starts
prefill token 0 embedding starts
token-id CUDA tensor creation completes
token-id flatten completes
embedding weight access completes
CUDA row-gather index_select starts
bounded run stops before index_select finishes
```

The next blocker is the first strict CUDA embedding row-gather
`weight.index_select(&flat_ids, 0)` for the Qwen3 `[151936, 1024]` F32 embedding
matrix and one CUDA U32 token ID.

The next diagnostic PR should avoid or replace this CUDA `index_select` path for
single-token embedding lookup before spending effort on transformer forward,
apply-time RoPE slice transfer, KV append, attention scores, softmax, value mix,
or output projection.

## Claim Boundary

This report proves only that current-source diagnostic tracing identifies the
Qwen3 strict CUDA one-token embedding frontier. The following remain false:

- `speedup_claim`
- `benchmark_qualified_speedup`
- `server_ready`
- `full_cuda_residency_claimed`
- broad dense GGUF readiness
- Qwen2.5 proof inheritance for Qwen3
- BitNet packed I2_S/QK256 proof
