# CUDA-DENSE-053 Current-Source Warm-Session Proof

Date: 2026-05-14
Campaign item: CUDA-DENSE-053
Platform: Windows 9950X3D + RTX 5070 Ti
Model: qwen2.5-0.5b-instruct-q8_0

## Summary

CUDA-DENSE-053 records a current-source strict CUDA warm-session proof for the
exact Qwen2.5 0.5B Q8_0 artifact.

The proof was generated from current `origin/main` in a Visual Studio x64
developer environment with the CUDA 12.9 `bin` and `lib\x64` paths visible. It
uses the refreshed CUDA-DENSE-051 one-token receipt and CUDA-DENSE-052
current-source short-decode receipt as prerequisite evidence.

The resulting receipt records a bounded three-turn warm session through the
`dense_regular_llm_cuda` route on `nvidia-rtx-5070-ti-cuda`. It shows that the
model, tokenizer, and CUDA context were initialized once, intended runtime
buffers were reused, generated token IDs matched between CPU and CUDA for all
three turns, and no fallback was used.

## Receipt

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-14/dense-qwen25-q8-warm-session-current-source.json
```

Recorded facts:

```text
selected_backend = nvidia-rtx-5070-ti-cuda
runtime_api = cuda
route = dense_regular_llm_cuda
fallback_used = false
turns_count = 3
generated_tokens_total = 24
model_loaded_once = true
tokenizer_loaded_once = true
cuda_context_initialized_once = true
runtime_buffers_reused = true
weights_uploaded_once = true
per_turn_weight_upload = false
generated_token_ids_match = true
top_k_all_match = true
speedup_claim = false
full_cuda_residency_claimed = false
bitnet_packed_i2s_qk256_proof = false
server_ready_claimed = false
```

Per-turn decoded text:

```text
0: The answer is 4. What is
1: The color of the sky is blue.
2: Good morning! How can I assist you
```

## Command

The run used the locally cached artifact whose model identity and SHA-256 are
recorded in the receipt. The reproduction command is parameterized so it does
not depend on a user-specific cache path:

```powershell
$env:BITNET_QWEN25_Q8_GGUF = "<path-to-qwen2.5-0.5b-instruct-q8_0.gguf>"
rtk powershell -NoProfile -Command '$cmd = ''"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 && set "LIB=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\lib\x64;%LIB%" && set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin;%PATH%" && rtk cargo run --locked --release -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- dense-gguf-qwen-warm-session-strict-cuda --model "%BITNET_QWEN25_Q8_GGUF%" --one-token-proof ci\hardware\windows-9950x3d-rtx5070ti\2026-05-13\dense-qwen25-q8-one-token-cuda.json --short-decode-proof ci\hardware\windows-9950x3d-rtx5070ti\2026-05-14\dense-qwen25-q8-short-decode-current-source.json --json-out ci\hardware\windows-9950x3d-rtx5070ti\2026-05-14\dense-qwen25-q8-warm-session-current-source.json''; cmd /d /s /c $cmd'
```

## Claim Boundary

This receipt may claim:

- exact Qwen2.5 0.5B Q8_0 artifact identity was checked;
- RTX 5070 Ti CUDA was selected;
- dense regular LLM CUDA route was used;
- the bounded three-turn warm session reused the model, tokenizer, CUDA context,
  runtime buffers, and uploaded weights as recorded;
- CPU and CUDA generated token IDs matched for the recorded turns;
- fallback was not used;
- kernel and transfer evidence was recorded;
- CUDA-DENSE-053 is proven for the recorded warm-session scope.

It must not claim:

- dense Qwen proof is BitNet packed I2_S/QK256 proof;
- broad dense Qwen chat quality;
- server readiness;
- speedup;
- full CUDA residency;
- general dense GGUF CUDA readiness.

## Validation

```powershell
rtk python -m json.tool ci\hardware\windows-9950x3d-rtx5070ti\2026-05-14\dense-qwen25-q8-warm-session-current-source.json
rtk cargo run --locked -p xtask --no-default-features -- campaign check nvidia-5070ti
rtk cargo run --locked -p xtask --no-default-features -- campaign generate --check
rtk git diff --check
```
