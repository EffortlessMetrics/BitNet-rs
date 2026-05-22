# CUDA-MODEL-017E Qwen3 RoPE Upload Probe

Status: diagnostic blocker narrowed

## Scope

`CUDA-MODEL-017` requires repeated same-artifact Qwen3 CPU/CUDA comparator
receipts for the exact Qwen3 0.6B Q8_0 artifact on the Windows 9950X3D +
RTX 5070 Ti lane.

`CUDA-MODEL-017D` showed that CUDA model construction reaches layer 0 attention
RoPE initialization, finishes CPU RoPE table generation, and then stops before
`Tensor::from_vec` returns for the CUDA sin table. This follow-up adds an
opt-in constructor trace probe that separates CUDA allocation, host-to-device
copy, and stream synchronization for the RoPE sin/cos tables before the existing
Candle tensor construction path.

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
- Built binary:
  `D:\codex-targets\bitnet-qwen-rope-upload-probe\debug\bitnet.exe`
- CUDA build environment: Visual Studio 2022 x64 developer environment, CUDA
  12.9, `NVCC_CCBIN` pinned to the x64 MSVC `cl.exe`

## Commands

The current-source debug CLI was built with an isolated target directory on
`D:`:

```powershell
cmd /d /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && set "CARGO_TARGET_DIR=D:\codex-targets\bitnet-qwen-rope-upload-probe" && set "CARGO_BUILD_JOBS=4" && set "CMAKE_GENERATOR=Ninja" && set "NVCC_CCBIN=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe" && set "LIB=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\lib\x64;%LIB%" && set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin;%PATH%" && rtk cargo build --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli'
```

The bounded diagnostic run used the normal one-token strict CUDA capture command
with a phase trace:

```powershell
D:\codex-targets\bitnet-qwen-rope-upload-probe\debug\bitnet.exe dense-gguf-qwen-one-token-strict-cuda `
  --model C:\Users\steven\AppData\Local\bitnet-rs\models\qwen3-0.6b-instruct-q8_0\Qwen3-0.6B-Q8_0.gguf `
  --all-layer-plan ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-cuda-all-layer-plan.json `
  --model-boundary-fixtures ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-model-boundary-fixtures.json `
  --kv-cache-policy ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-kv-cache-policy.json `
  --sampling-policy ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-sampling-policy.json `
  --top-k 10 `
  --device-index 0 `
  --json-out D:\codex-runs\cuda-model-017-rope-upload-probe\qwen3-one-token.json `
  --phase-trace-jsonl D:\codex-runs\cuda-model-017-rope-upload-probe\qwen3-one-token-phase-trace.jsonl
```

The transformer-constructor trace was written next to the phase trace:

```text
D:\codex-runs\cuda-model-017-rope-upload-probe\qwen3-one-token-phase-trace.transformer.jsonl
```

## Result

No proof receipt JSON was emitted.

The phase trace contained 669 JSONL events. The final phase event remained:

```json
{"command":"dense-gguf-qwen-one-token-strict-cuda","details":{"message":"Initializing model...","progress":0.8999999761581421},"elapsed_ms":28858.9606,"phase":"cuda_target","schema":1,"state":"model_loader_progress","timestamp_utc":"2026-05-22T06:46:50.692Z"}
```

The transformer trace contained 1573 JSONL events. The CUDA trace again reached
layer 0 attention initialization, q/k/v/o linears, attention norms, and CPU RoPE
table generation. The final CUDA transformer-constructor events were:

```text
model_init.rope_tables_finish layer=0 device=cuda tables_ms=61.7453 half_dim=64 sin_len=2621440 cos_len=2621440
model_init.rope_sin_tensor_start layer=0 device=cuda rows=40960 cols=64
model_init.rope_cuda_upload_probe_start layer=0 device=cuda table=sin rows=40960 cols=64 elements=2621440 bytes=10485760
model_init.rope_cuda_alloc_start layer=0 device=cuda table=sin elements=2621440
model_init.rope_cuda_alloc_finish layer=0 device=cuda table=sin alloc_ms=0.1107 elements=2621440
model_init.rope_cuda_htod_start layer=0 device=cuda table=sin bytes=10485760
```

No `model_init.rope_cuda_htod_finish`,
`model_init.rope_cuda_sync_start`, or `model_init.rope_sin_tensor_finish` event
was emitted before the bounded run was stopped.

## Blocker Classification

The CUDA-MODEL-017 one-token source receipt is currently blocked during CUDA
RoPE table upload:

```text
CUDA tensor loading completes
CUDA model construction starts
CUDA layer 0 starts
CUDA attention q/k/v/o linears complete
CUDA attention q/k norms complete
RoPE CPU table generation completes
direct CUDA allocation for the sin table succeeds
direct CUDA host-to-device copy for the 10 MiB sin table did not return before the bounded stop
```

The next fix should avoid repeated constructor-time H2D upload of identical full
RoPE tables. Candidate directions remain bounded and must not silently fall back:

- build RoPE tables only for the sequence length required by the proof profile;
- add a shared upload-once RoPE table cache for dense Qwen CUDA model
  construction;
- use a CUDA-side RoPE table builder or existing dense RoPE kernel path instead
  of host-building and uploading full sin/cos tables per layer.

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
