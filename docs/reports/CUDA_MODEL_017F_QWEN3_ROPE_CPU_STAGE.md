# CUDA-MODEL-017F Qwen3 RoPE CPU Staging

Status: constructor blocker moved to prefill

## Scope

`CUDA-MODEL-017` requires repeated same-artifact Qwen3 CPU/CUDA comparator
receipts for the exact Qwen3 0.6B Q8_0 artifact on the Windows 9950X3D +
RTX 5070 Ti lane.

`CUDA-MODEL-017E` showed that current-source CUDA model construction stopped
inside the constructor-time host-to-device copy for the first full Qwen3 RoPE
sin table. This follow-up changes `RotaryEmbedding` construction so CUDA target
models keep the full sin/cos RoPE tables staged on CPU and move only the
narrowed position slice to the active tensor device when RoPE is applied.

This is a runtime unblocker plus diagnostic report. It does not emit a
one-token proof receipt and does not promote speed, server readiness, benchmark
qualification, full residency, broad dense GGUF support, Qwen2.5 inheritance, or
BitNet QK256 proof.

## Change

The RoPE table storage decision is now explicit:

```text
target device = cuda
RoPE table storage = cpu
apply-time slice device = input tensor device
reason = avoid constructor-time full-table CUDA upload
```

For non-CUDA target devices, RoPE table construction remains on the requested
target device.

The transformer trace now emits `model_init.rope_table_storage` so hardware
runs can distinguish constructor storage from the selected runtime target.

## Environment Observed

- GPU: NVIDIA GeForce RTX 5070 Ti
- Model: `Qwen3-0.6B-Q8_0.gguf`
- Model SHA-256:
  `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`
- Local model path:
  `C:\Users\steven\AppData\Local\bitnet-rs\models\qwen3-0.6b-instruct-q8_0\Qwen3-0.6B-Q8_0.gguf`
- Built binary:
  `D:\codex-targets\bitnet-qwen-rope-cpu-stage\debug\bitnet.exe`
- CUDA build environment: Visual Studio 2022 x64 developer environment, CUDA
  12.9, `NVCC_CCBIN` pinned to the x64 MSVC `cl.exe`

## Commands

Focused local validation:

```powershell
cargo fmt -p bitnet-transformer -- --check
cargo check --locked -p bitnet-transformer --no-default-features --features cpu
cargo test --locked -p bitnet-transformer --no-default-features --features cpu rope
cargo check --locked -p bitnet-transformer --no-default-features --features cpu,cuda
git diff --check
```

The current-source debug CLI was built with an isolated target directory on
`D:`:

```powershell
cmd /d /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && set "CARGO_TARGET_DIR=D:\codex-targets\bitnet-qwen-rope-cpu-stage" && set "CARGO_BUILD_JOBS=4" && set "CMAKE_GENERATOR=Ninja" && set "NVCC_CCBIN=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe" && set "LIB=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\lib\x64;%LIB%" && set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin;%PATH%" && cargo build --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli'
```

The bounded diagnostic run used the normal one-token strict CUDA capture command
with a phase trace:

```powershell
D:\codex-targets\bitnet-qwen-rope-cpu-stage\debug\bitnet.exe dense-gguf-qwen-one-token-strict-cuda `
  --model C:\Users\steven\AppData\Local\bitnet-rs\models\qwen3-0.6b-instruct-q8_0\Qwen3-0.6B-Q8_0.gguf `
  --all-layer-plan ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-cuda-all-layer-plan.json `
  --model-boundary-fixtures ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-model-boundary-fixtures.json `
  --kv-cache-policy ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-kv-cache-policy.json `
  --sampling-policy ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-sampling-policy.json `
  --top-k 10 `
  --device-index 0 `
  --json-out D:\codex-runs\cuda-model-017f-rope-cpu-stage\qwen3-one-token.json `
  --phase-trace-jsonl D:\codex-runs\cuda-model-017f-rope-cpu-stage\qwen3-one-token-phase-trace.jsonl
```

The transformer-constructor trace was written next to the phase trace:

```text
D:\codex-runs\cuda-model-017f-rope-cpu-stage\qwen3-one-token-phase-trace.transformer.jsonl
```

## Result

No proof receipt JSON was emitted.

The phase trace contained 674 JSONL events. The run got past CUDA model
construction and KV-cache creation. The final phase events were:

```text
model_loader_progress message="Model loaded successfully" progress=1.0 elapsed_ms=34136.2061
model_load_finish model_load_ms=17379.4899 elapsed_ms=34139.0427
kv_cache_start required_seq_len=8 elapsed_ms=34139.7101
kv_cache_finish elapsed_ms=34141.0436
prefill_start prefill_tokens=7 elapsed_ms=34141.6476
```

The transformer trace contained 2887 JSONL events. The CUDA trace reached all
28 layers and emitted `model_init.finish`. CUDA RoPE storage markers now show
CPU-staged tables for CUDA target construction:

```text
model_init.rope_table_storage layer=0 device=cuda target_device=cuda table_device=cpu reason=cpu_staged_to_avoid_constructor_full_table_cuda_upload
model_init.rope_table_storage layer=27 device=cuda target_device=cuda table_device=cpu reason=cpu_staged_to_avoid_constructor_full_table_cuda_upload
model_init.layers_finish layers=28
model_init.final_norm_finish
model_init.tied_embedding_transpose_finish transposed_dims=[1024,151936]
model_init.finish
```

The bounded process stopped after `prefill_start`; no `prefill_finish`,
`decode_forward_start`, logits event, or receipt was emitted.

## Blocker Classification

`CUDA-MODEL-017E` localized the previous blocker to full-table CUDA RoPE upload
during construction. This change removes that constructor-time blocker for the
Qwen3 CUDA path:

```text
CUDA tensor loading completes
CUDA model construction starts
CUDA layer 0 RoPE table storage is CPU-staged for CUDA target
CUDA layer 27 completes
CUDA model_init.finish is emitted
CUDA KV cache is created
prefill starts
bounded run stops before prefill finishes
```

The next blocker is no longer constructor-time RoPE upload. The next diagnostic
PR should instrument the one-token strict CUDA prefill loop and transformer
forward path so the lane can distinguish:

- embedding lookup;
- first prefill `model.forward`;
- apply-time RoPE slice transfer;
- KV append;
- attention score / softmax / value mix;
- output projection.

## Claim Boundary

This report proves only that the current-source Qwen3 one-token capture moves
past CUDA model construction after CPU-staging RoPE tables. The following remain
false:

- `speedup_claim`
- `benchmark_qualified_speedup`
- `server_ready`
- `full_cuda_residency_claimed`
- broad dense GGUF readiness
- Qwen2.5 proof inheritance for Qwen3
- BitNet packed I2_S/QK256 proof
