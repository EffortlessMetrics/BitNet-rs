# CUDA-MODEL-017G Qwen3 Prefill Frontier Trace

Status: diagnostic trace added; first prefill embedding lookup is the next frontier

## Scope

`CUDA-MODEL-017` requires repeated same-artifact Qwen3 CPU/CUDA comparator
receipts for the exact Qwen3 0.6B Q8_0 artifact on the Windows 9950X3D +
RTX 5070 Ti lane.

`CUDA-MODEL-017F` moved the current-source CUDA capture past constructor-time
Qwen3 RoPE table upload by CPU-staging full RoPE tables for CUDA targets. This
follow-up adds bounded phase and transformer trace points around the next
frontier: the one-token strict CUDA prefill loop and transformer forward path.

This is diagnostic observability only. It does not change runtime math,
sampling, route selection, model coverage, receipts, server readiness, speed
claims, residency claims, or proof-family booleans.

## Change

The strict Qwen3 one-token CUDA command now emits phase trace events around each
prefill token:

```text
prefill_embed_start
prefill_embed_finish
prefill_forward_start
prefill_forward_finish
```

The transformer trace now emits lightweight structured events for:

```text
model.forward
model.forward_layer
block attention/norm/feed-forward boundaries
attention qkv/reshape/norm/rope/cache/scores/softmax/value/output boundaries
RoPE apply-time slice boundaries
```

The new events are names, counters, device labels, dimensions, layer numbers,
and elapsed timings. They do not include prompt text. The prefill CLI marker
records `prefill_index`, not the token ID.

## Environment Observed

- GPU: NVIDIA GeForce RTX 5070 Ti
- Model: `Qwen3-0.6B-Q8_0.gguf`
- Local model path:
  `C:\Users\steven\AppData\Local\bitnet-rs\models\qwen3-0.6b-instruct-q8_0\Qwen3-0.6B-Q8_0.gguf`
- Built binary:
  `D:\codex-targets\bitnet-qwen-prefill-frontier\debug\bitnet.exe`
- CUDA build environment: Visual Studio 2022 x64 developer environment and
  CUDA 12.9 runtime/library paths.

## Commands

Focused validation:

```powershell
cargo fmt -p bitnet-transformer -p bitnet-cli -- --check
git diff --check
cargo check --locked -p bitnet-transformer --no-default-features --features cpu
cargo check --locked -p bitnet-cli --no-default-features --features cpu,full-cli
cargo test --locked -p bitnet-cli --no-default-features --features cpu,full-cli qwen_one_token_phase_trace
cargo test --locked -p bitnet-transformer --no-default-features --features cpu rope
```

The CUDA debug CLI was built with an isolated target directory:

```powershell
cargo build --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli
```

The first CUDA build attempt failed before linking because the shell did not
have CUDA `LIB` paths set (`LNK1181: cannot open input file 'cuda.lib'`). After
setting CUDA 12.9 `PATH` and `LIB` and running under the Visual Studio x64
developer environment, the CUDA build passed.

The bounded diagnostic run used the normal one-token strict CUDA capture command
with the Qwen3 prerequisite receipts from the 2026-05-15 RTX 5070 Ti campaign:

```powershell
D:\codex-targets\bitnet-qwen-prefill-frontier\debug\bitnet.exe dense-gguf-qwen-one-token-strict-cuda `
  --model C:\Users\steven\AppData\Local\bitnet-rs\models\qwen3-0.6b-instruct-q8_0\Qwen3-0.6B-Q8_0.gguf `
  --all-layer-plan ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-cuda-all-layer-plan.json `
  --model-boundary-fixtures ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-model-boundary-fixtures.json `
  --kv-cache-policy ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-kv-cache-policy.json `
  --sampling-policy ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-sampling-policy.json `
  --top-k 10 `
  --device-index 0 `
  --json-out D:\codex-runs\cuda-model-017g-prefill-frontier-2\qwen3-one-token.json `
  --phase-trace-jsonl D:\codex-runs\cuda-model-017g-prefill-frontier-2\qwen3-one-token-phase-trace.jsonl
```

The transformer trace was written next to the phase trace:

```text
D:\codex-runs\cuda-model-017g-prefill-frontier-2\qwen3-one-token-phase-trace.transformer.jsonl
```

## Result

No proof receipt JSON was emitted.

The process was stopped by the bounded 240 second guard. The phase trace
contained 703 JSONL events. The transformer trace contained 13,223 JSONL
events. The final phase events were:

```text
model_loader_progress message="Model loaded successfully" progress=1.0 elapsed_ms=44044.3128
model_load_finish model_load_ms=24835.774 elapsed_ms=44052.5791
kv_cache_start required_seq_len=8 elapsed_ms=44053.7641
kv_cache_finish elapsed_ms=44055.748
prefill_start prefill_tokens=7 elapsed_ms=44056.5647
prefill_embed_start prefill_index=0 elapsed_ms=44057.3921
```

There was no `prefill_embed_finish`, `prefill_forward_start`,
`model.forward_start`, `decode_forward_start`, logits event, or final receipt.

The transformer trace again reached constructor completion:

```text
model_init.rope_table_storage layer=27 device=cuda target_device=cuda table_device=cpu reason=cpu_staged_to_avoid_constructor_full_table_cuda_upload
model_init.layers_finish layers=28
model_init.final_norm_finish
model_init.tied_embedding_transpose_finish transposed_dims=[1024,151936]
model_init.finish
```

## Blocker Classification

`CUDA-MODEL-017F` localized and removed the constructor-time full RoPE upload
blocker. `CUDA-MODEL-017G` moves the trace frontier one step further:

```text
CUDA tensor loading completes
CUDA model construction completes
CUDA KV cache is created
prefill loop starts
prefill token 0 embedding lookup starts
bounded run stops before embedding lookup finishes
```

The next blocker is the first strict CUDA prefill embedding lookup, before the
first `model.forward` call. The next diagnostic PR should instrument or repair
the Qwen3 CUDA embedding lookup path before spending effort on transformer
forward, apply-time RoPE slice transfer, KV append, attention scores, softmax,
value mix, or output projection.

## Claim Boundary

This report proves only that current-source diagnostic tracing now identifies
the Qwen3 strict CUDA one-token prefill frontier. The following remain false:

- `speedup_claim`
- `benchmark_qualified_speedup`
- `server_ready`
- `full_cuda_residency_claimed`
- broad dense GGUF readiness
- Qwen2.5 proof inheritance for Qwen3
- BitNet packed I2_S/QK256 proof
