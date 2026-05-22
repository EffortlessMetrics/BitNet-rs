# CUDA-MODEL-017D Qwen3 RoPE Init Trace

Status: diagnostic blocker narrowed

## Scope

CUDA-MODEL-017 requires repeated same-artifact Qwen3 CPU/CUDA comparator
receipts for the exact Qwen3 0.6B Q8_0 artifact on the Windows 9950X3D +
RTX 5070 Ti lane.

CUDA-MODEL-017C narrowed the current-source one-token capture blocker to CUDA
layer 0 transformer block construction. This follow-up adds opt-in constructor
trace events around `TransformerBlock::new`, `MultiHeadAttention::new`,
attention q/k/v/o linears, and RoPE initialization. It narrows the blocker to
CUDA RoPE sin-table tensor materialization in the bounded run.

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
  `D:\codex-targets\bitnet-qwen-rope-init-trace\debug\bitnet.exe`
- CUDA build environment: Visual Studio 2022 x64 developer environment, CUDA
  12.9, `NVCC_CCBIN` pinned to the x64 MSVC `cl.exe`

## Commands

The current-source debug CLI was built with an isolated target directory on
`D:`:

```powershell
cmd /d /s /c 'call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && set "CARGO_TARGET_DIR=D:\codex-targets\bitnet-qwen-rope-init-trace" && set "CARGO_BUILD_JOBS=4" && set "CMAKE_GENERATOR=Ninja" && set "NVCC_CCBIN=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe" && set "LIB=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\lib\x64;%LIB%" && set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin;%PATH%" && rtk cargo build --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli'
```

The bounded diagnostic run used:

```powershell
D:\codex-targets\bitnet-qwen-rope-init-trace\debug\bitnet.exe dense-gguf-qwen-one-token-strict-cuda `
  --model C:\Users\steven\AppData\Local\bitnet-rs\models\qwen3-0.6b-instruct-q8_0\Qwen3-0.6B-Q8_0.gguf `
  --all-layer-plan ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-cuda-all-layer-plan.json `
  --model-boundary-fixtures ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-model-boundary-fixtures.json `
  --kv-cache-policy ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-kv-cache-policy.json `
  --sampling-policy ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-sampling-policy.json `
  --top-k 10 `
  --device-index 0 `
  --json-out D:\codex-runs\cuda-model-017-rope-init-trace\qwen3-one-token.json `
  --phase-trace-jsonl D:\codex-runs\cuda-model-017-rope-init-trace\qwen3-one-token-phase-trace.jsonl
```

The transformer-constructor trace was written next to the phase trace:

```text
D:\codex-runs\cuda-model-017-rope-init-trace\qwen3-one-token-phase-trace.transformer.jsonl
```

## Result

No proof receipt JSON was emitted.

The phase trace contained 669 JSONL events. The final phase event remained:

```json
{"command":"dense-gguf-qwen-one-token-strict-cuda","details":{"message":"Initializing model...","progress":0.8999999761581421},"elapsed_ms":28489.300499999998,"phase":"cuda_target","schema":1,"state":"model_loader_progress","timestamp_utc":"2026-05-22T05:51:15.787Z"}
```

The transformer trace contained 1457 JSONL events. CPU model initialization
completed, including tied embedding transpose and `model_init.finish`.

The CUDA trace reached layer 0 attention initialization and proved the q/k/v/o
attention linears were not the current stop point:

```text
model_init.linear_finish layer=0 device=cuda scope=attention linear=q_proj
model_init.linear_finish layer=0 device=cuda scope=attention linear=k_proj
model_init.linear_finish layer=0 device=cuda scope=attention linear=v_proj
model_init.linear_finish layer=0 device=cuda scope=attention linear=o_proj
model_init.attention_linears_finish layer=0 device=cuda
model_init.attention_norms_finish layer=0 device=cuda q_norm_present=true k_norm_present=true sub_layernorm_present=false
```

The final CUDA transformer-constructor events were:

```text
model_init.attention_rope_start layer=0 device=cuda head_dim=128 max_seq_len=40960
model_init.rope_start layer=0 device=cuda dim=128 max_seq_len=40960 theta=1000000
model_init.rope_tables_start layer=0 device=cuda dim=128 max_seq_len=40960
model_init.rope_tables_finish layer=0 device=cuda tables_ms=59.1173 half_dim=64 sin_len=2621440 cos_len=2621440
model_init.rope_sin_tensor_start layer=0 device=cuda rows=40960 cols=64
```

No `model_init.rope_sin_tensor_finish` event was emitted before the bounded run
was stopped.

## Blocker Classification

The CUDA-MODEL-017 one-token source receipt is currently blocked during CUDA
RoPE initialization:

```text
CUDA tensor loading completes
CUDA model construction starts
CUDA layer 0 starts
CUDA attention q/k/v/o linears complete
CUDA attention q/k norms complete
RoPE CPU table generation completes
CUDA Tensor::from_vec for the sin table did not return before the bounded stop
```

The next diagnostic should avoid or isolate CUDA materialization of identical
RoPE sin/cos tables per layer. The immediate questions are:

- whether `Tensor::from_vec` to CUDA stalls because it is invoked inside model
  construction for each layer;
- whether the dense Qwen CUDA path should keep RoPE tables on CPU during
  constructor proof, use a shared upload-once RoPE cache, or use a CUDA RoPE
  table builder;
- whether Qwen3 one-token capture can proceed with a guarded diagnostic mode
  that records RoPE table residency without silently falling back.

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
