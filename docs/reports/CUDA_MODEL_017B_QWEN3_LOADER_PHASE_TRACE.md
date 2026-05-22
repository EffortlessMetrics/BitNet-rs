# CUDA-MODEL-017B Qwen3 Loader Phase Trace

Status: diagnostic blocker classified

## Scope

CUDA-MODEL-017 requires current-source repeated comparator receipts for the exact
Qwen3 0.6B Q8_0 artifact on the Windows 9950X3D + RTX 5070 Ti lane. This report
records a bounded diagnostic rerun after adding Qwen one-token phase tracing.

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
- Binary: current-source debug `bitnet-cli` built with `cpu,cuda,full-cli`
- CUDA build environment: Visual Studio 2022 x64 developer environment, CUDA
  12.9, `NVCC_CCBIN` pinned to the x64 MSVC `cl.exe`

## Commands

The current-source debug CLI was built with:

```powershell
cmd /d /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && set "CARGO_TARGET_DIR=target\cuda-model-017-phase-trace-debug" && set "CARGO_BUILD_JOBS=4" && set "CMAKE_GENERATOR=Ninja" && set "NVCC_CCBIN=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe" && set "LIB=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\lib\x64;%LIB%" && set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin;%PATH%" && rtk proxy cargo build --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli'
```

The bounded diagnostic run used:

```powershell
target\cuda-model-017-phase-trace-debug\debug\bitnet.exe dense-gguf-qwen-one-token-strict-cuda `
  --model C:\Users\steven\AppData\Local\bitnet-rs\models\qwen3-0.6b-instruct-q8_0\Qwen3-0.6B-Q8_0.gguf `
  --all-layer-plan ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-cuda-all-layer-plan.json `
  --model-boundary-fixtures ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-model-boundary-fixtures.json `
  --kv-cache-policy ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-kv-cache-policy.json `
  --sampling-policy ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-sampling-policy.json `
  --top-k 10 `
  --device-index 0 `
  --json-out target\cuda-model-017-phase-trace-loader\qwen3-one-token.json `
  --phase-trace-jsonl target\cuda-model-017-phase-trace-loader\qwen3-one-token-phase-trace.jsonl
```

## Result

The run did not emit `qwen3-one-token.json`. It was stopped after the diagnostic
trace identified the active long phase.

The phase trace contained 668 JSONL events:

| Phase | Events |
| --- | ---: |
| `command` | 1 |
| `model_map` | 2 |
| `gguf_inspection` | 2 |
| `cuda_probe` | 2 |
| `prerequisites` | 2 |
| `tokenizer` | 2 |
| `cpu_reference` | 337 |
| `cuda_target` | 320 |

Key observations:

- CPU reference completed.
- CUDA device creation completed with `is_cuda=true`.
- CUDA target model loading reached `Loading tensor 310/310`.
- The last trace event was:
  `cuda_target:model_loader_progress`, `message="Initializing model..."`,
  `progress=0.8999999761581421`, `elapsed_ms=27526.1621`.
- No `cuda_target:model_load_finish` event was emitted.
- No proof receipt JSON was emitted.

## Blocker Classification

The CUDA-MODEL-017 one-token source receipt is currently blocked after CUDA
tensor loading completes and before the CUDA model-load phase returns. The next
diagnostic should instrument or isolate `BitNetModel::from_gguf_with_dense_q8_sidecars`
for dense Qwen on CUDA, especially initialization work after all 310 tensors
have loaded.

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
