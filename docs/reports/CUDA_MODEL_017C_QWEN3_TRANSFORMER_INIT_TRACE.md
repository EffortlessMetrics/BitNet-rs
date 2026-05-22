# CUDA-MODEL-017C Qwen3 Transformer Init Trace

Status: diagnostic blocker narrowed

## Scope

CUDA-MODEL-017 requires repeated same-artifact Qwen3 CPU/CUDA comparator
receipts for the exact Qwen3 0.6B Q8_0 artifact on the Windows 9950X3D +
RTX 5070 Ti lane.

This report records the bounded follow-up after the Qwen one-token phase trace
was extended with opt-in transformer-constructor trace events. It narrows the
current-source capture blocker from "after CUDA tensor loading" to CUDA layer 0
transformer block construction.

This is a diagnostic report only. It does not emit a one-token proof receipt and
does not promote speed, server readiness, benchmark qualification, full
residency, broad dense GGUF support, Qwen2.5 inheritance, or BitNet QK256 proof.

## Environment Observed

- GPU: NVIDIA GeForce RTX 5070 Ti
- Model: `Qwen3-0.6B-Q8_0.gguf`
- Model SHA-256:
  `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`
- Local model path:
  `C:\Users\steven\AppData\Local\bitnet-rs\models\qwen3-0.6b-instruct-q8_0\Qwen3-0.6B-Q8_0.gguf`
- Swarm source: current `bitnet-rs-swarm/main` after PR #266
- Built binary:
  `D:\codex-targets\bitnet-qwen-init-trace\debug\bitnet.exe`
- CUDA build environment: Visual Studio 2022 x64 developer environment, CUDA
  12.9, `NVCC_CCBIN` pinned to the x64 MSVC `cl.exe`

## Commands

The current-source debug CLI was built with an isolated target directory on
`D:`:

```powershell
cmd /d /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && set "CARGO_TARGET_DIR=D:\codex-targets\bitnet-qwen-init-trace" && set "CARGO_BUILD_JOBS=4" && set "CMAKE_GENERATOR=Ninja" && set "NVCC_CCBIN=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe" && set "LIB=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\lib\x64;%LIB%" && set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin;%PATH%" && rtk cargo build --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli'
```

The bounded diagnostic run used:

```powershell
D:\codex-targets\bitnet-qwen-init-trace\debug\bitnet.exe dense-gguf-qwen-one-token-strict-cuda `
  --model C:\Users\steven\AppData\Local\bitnet-rs\models\qwen3-0.6b-instruct-q8_0\Qwen3-0.6B-Q8_0.gguf `
  --all-layer-plan ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-cuda-all-layer-plan.json `
  --model-boundary-fixtures ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-model-boundary-fixtures.json `
  --kv-cache-policy ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-kv-cache-policy.json `
  --sampling-policy ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-sampling-policy.json `
  --top-k 10 `
  --device-index 0 `
  --json-out D:\codex-runs\cuda-model-017-transformer-init\qwen3-one-token.json `
  --phase-trace-jsonl D:\codex-runs\cuda-model-017-transformer-init\qwen3-one-token-phase-trace.jsonl
```

The transformer-constructor trace was written next to the phase trace:

```text
D:\codex-runs\cuda-model-017-transformer-init\qwen3-one-token-phase-trace.transformer.jsonl
```

## Phase Trace Result

No proof receipt JSON was emitted.

The phase trace contained 669 JSONL events:

| Phase | Events |
| --- | ---: |
| `command` | 2 |
| `model_map` | 2 |
| `gguf_inspection` | 2 |
| `cuda_probe` | 2 |
| `prerequisites` | 2 |
| `tokenizer` | 2 |
| `cpu_reference` | 337 |
| `cuda_target` | 320 |

The final phase event was:

```json
{"command":"dense-gguf-qwen-one-token-strict-cuda","details":{"message":"Initializing model...","progress":0.8999999761581421},"elapsed_ms":26538.6656,"phase":"cuda_target","schema":1,"state":"model_loader_progress","timestamp_utc":"2026-05-22T04:11:54.135Z"}
```

## Transformer Trace Result

The transformer trace contained 79 JSONL events.

CPU reference initialization completed all 28 layers and finished model
construction in about 1.719 seconds:

```text
model_init.layers_finish elapsed_ms=1717.3286
model_init.final_norm_finish
model_init.lm_head_finish lm_head_present=false lm_head_weight_present=false
model_init.tied_embedding_transpose_finish
model_init.finish elapsed_ms=1718.784
lm_head.metadata has_cached_tied_weight=true has_tied_qk256_output=false
```

CUDA initialization reached layer 0 construction and did not emit a matching
layer 0 finish event:

```text
model_init.start device=cuda
model_init.embedding_start
model_init.embedding_finish
model_init.embed_transposed_flag_start
model_init.embed_transposed_flag_finish
model_init.layers_start
model_init.layer_start layer=0 elapsed_ms=0.6379
```

The stderr log places the CUDA stop point after `MultiHeadAttention::new`
reported the Qwen3 layer 0 dimensions and immediately before attention linear
construction returned:

```text
Successfully loaded 310 tensors (detected 0 QK256 tensors) with fingerprint: sha256-9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031
layer0: MultiHeadAttention dims: hidden=1024, n_heads=16, n_kv_heads=8, head_dim=128, q_out=2048, kv_out=1024, group_size=2
layer0: About to create linear layers with: q_proj([2048, 1024]), k_proj([1024, 1024]), v_proj([1024, 1024]), o_proj([1024, 2048])
```

## Blocker Classification

The CUDA-MODEL-017 one-token source receipt is currently blocked during CUDA
transformer construction:

```text
CUDA tensor loading completes
CUDA model construction starts
CUDA embedding setup completes
CUDA layer list construction starts
CUDA layer 0 starts
CUDA layer 0 attention linear construction does not return
```

The next diagnostic should instrument `TransformerBlock::new`,
`MultiHeadAttention::new`, and `linear_with_optional_bias` for the Qwen3 CUDA
path so the lane can identify whether the stall is in `q_proj`, `k_proj`,
`v_proj`, `o_proj`, or the tensor fetch/upload/reshape step that feeds those
linears.

## Claim Boundary

This report proves only the diagnostic location of the current-source Qwen3
one-token capture blocker. The following remain false:

- `speedup_claim`
- `benchmark_qualified_speedup`
- `server_ready`
- `full_cuda_residency_claimed`
- broad dense GGUF readiness
- Qwen2.5 proof inheritance for Qwen3
- BitNet packed I2_S/QK256 proof
