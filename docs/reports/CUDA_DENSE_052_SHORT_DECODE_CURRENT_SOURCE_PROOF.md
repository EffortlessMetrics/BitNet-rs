# CUDA-DENSE-052 Current-Source Short-Decode Proof

Date: 2026-05-14
Campaign item: CUDA-DENSE-052
Platform: Windows 9950X3D + RTX 5070 Ti
Model: qwen2.5-0.5b-instruct-q8_0

## Summary

CUDA-DENSE-052 is unblocked by a current-source strict CUDA short-decode proof.

The proof was generated from current `origin/main` in a Visual Studio x64
developer environment with the CUDA 12.9 `bin` and `lib\x64` paths visible. It
uses the exact Qwen2.5 0.5B Q8_0 artifact and the refreshed CUDA-DENSE-051
one-token receipt as prerequisite evidence.

The resulting receipt records fallback-free RTX 5070 Ti CUDA execution,
CPU/CUDA generated-token equality, kernel stats, transfer stats, and bounded
decoded text:

```text
The answer is 4. What is
```

This supersedes the earlier 2026-05-14 diagnostic receipt produced by a stale
binary, which preserved CPU/CUDA token parity but decoded to non-product text.

## Receipt

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-14/dense-qwen25-q8-short-decode-current-source.json
```

Recorded facts:

```text
selected_backend = nvidia-rtx-5070-ti-cuda
runtime_api = cuda
route = dense_regular_llm_cuda
fallback_used = false
generated_tokens_count = 8
cpu_generated_token_ids = [576, 4226, 374, 220, 19, 13, 3555, 374]
cuda_generated_token_ids = [576, 4226, 374, 220, 19, 13, 3555, 374]
generated_token_ids_match = true
speedup_claim = false
bitnet_packed_i2s_qk256_proof = false
server_ready_claimed = false
```

## Command

```powershell
rtk powershell -NoProfile -Command '$cmd = ''"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 && set "LIB=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\lib\x64;%LIB%" && set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin;%PATH%" && rtk target\release\bitnet.exe dense-gguf-qwen-short-decode-strict-cuda --model C:\Users\steven\AppData\Local\bitnet-rs\models\qwen2.5-0.5b-instruct-q8_0\qwen2.5-0.5b-instruct-q8_0.gguf --one-token-proof ci\hardware\windows-9950x3d-rtx5070ti\2026-05-13\dense-qwen25-q8-one-token-cuda.json --json-out ci\hardware\windows-9950x3d-rtx5070ti\2026-05-14\dense-qwen25-q8-short-decode-current-source.json''; cmd /d /s /c $cmd'
```

The first build attempt without the Visual Studio developer environment failed
because `nvcc` could not find `cl.exe`. The second build attempt found `cl.exe`
but failed to link because `cuda.lib` was not in `LIB`. The successful proof run
used the Visual Studio x64 developer environment and explicitly added the CUDA
12.9 library path.

## Claim Boundary

This receipt may claim:

- exact Qwen2.5 0.5B Q8_0 artifact identity was checked;
- RTX 5070 Ti CUDA was selected;
- dense regular LLM CUDA route was used;
- CPU and CUDA generated token IDs matched across the recorded 8-token decode;
- fallback was not used;
- kernel and transfer evidence was recorded;
- CUDA-DENSE-052 is proven for the recorded bounded short-decode scope.

It must not claim:

- dense Qwen proof is BitNet packed I2_S/QK256 proof;
- broad dense Qwen chat quality;
- server readiness;
- speedup;
- full CUDA residency;
- general dense GGUF CUDA readiness.

## Validation

```powershell
rtk python -m json.tool ci\hardware\windows-9950x3d-rtx5070ti\2026-05-14\dense-qwen25-q8-short-decode-current-source.json
rtk cargo run --locked -p xtask --no-default-features -- campaign check nvidia-5070ti
rtk cargo run --locked -p xtask --no-default-features -- campaign generate --check
rtk git diff --check
```
