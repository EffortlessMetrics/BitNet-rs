# CUDA-MODEL-006 Qwen3 0.6B Warm-Session Proof

Date: 2026-05-15
Campaign item: CUDA-MODEL-006
Platform: Windows 9950X3D + RTX 5070 Ti
Model: qwen3-0.6b-instruct-q8_0

## Summary

CUDA-MODEL-006 records bounded warm-session strict CUDA proof for the exact
Qwen3 0.6B Q8_0 artifact already covered by CUDA-MODEL-001 through
CUDA-MODEL-005.

The proof was generated from the CUDA-MODEL-006 branch in a Visual Studio x64
developer environment with CUDA 12.9 visible and `NVCC_CCBIN` pinned to the x64
MSVC host compiler. The receipt selects `nvidia-rtx-5070-ti-cuda`, uses the
`dense_regular_llm_cuda` route, rejects fallback, records one model load, one
tokenizer load, one CUDA context initialization, upload-once weights, reused
runtime buffers, and three deterministic turns. It keeps speed, server,
full-residency, broad dense GGUF, and BitNet QK256 claims false.

## Receipt

```text
ci/hardware/windows-9950x3d-rtx5070ti/2026-05-15/qwen3-0_6b-warm-session-cuda.json
```

Recorded facts:

```text
selected_backend = nvidia-rtx-5070-ti-cuda
runtime_api = cuda
route = dense_regular_llm_cuda
model_id = qwen3-0.6b-instruct-q8_0
model_sha256 = 9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031
fallback_used = false
turns_count = 3
generated_tokens_total = 24
generated_token_ids_match = true
top_k_all_match = true
model_loaded_once = true
tokenizer_loaded_once = true
cuda_context_initialized_once = true
weights_uploaded_once = true
per_turn_weight_upload = false
runtime_buffers_reused = true
speedup_claim = false
server_ready_claimed = false
full_cuda_residency_claimed = false
bitnet_packed_i2s_qk256_proof = false
```

Decoded per-turn text:

```text
turn 0:  3. What is 3+
turn 1:  1. Blue 2. Green
turn 2:  1. The user is a student
```

## Command

The run used the pinned Qwen3 artifact whose model identity and SHA-256 are
recorded in the receipt. The reproduction command is parameterized so it does
not depend on a user-specific temporary path:

```powershell
$env:BITNET_QWEN3_Q8_GGUF = "<path-to-Qwen3-0.6B-Q8_0.gguf>"

rtk powershell -NoProfile -Command 'cmd /d /s /c ''call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && set "CARGO_TARGET_DIR=<target-dir>" && set "CMAKE_GENERATOR=Ninja" && set "NVCC_CCBIN=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe" && set "LIB=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\lib\x64;%LIB%" && set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin;%PATH%" && rtk cargo run --locked --release -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- dense-gguf-qwen-warm-session-strict-cuda --model "%BITNET_QWEN3_Q8_GGUF%" --all-layer-plan ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-cuda-all-layer-plan.json --model-boundary-fixtures ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-model-boundary-fixtures.json --kv-cache-policy ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-kv-cache-policy.json --sampling-policy ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-sampling-policy.json --one-token-proof ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-one-token-cuda.json --short-decode-proof ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-short-decode-cuda.json --turns 3 --max-new-tokens 8 --top-k 10 --device-index 0 --json-out ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-warm-session-cuda.json'''
```

## Claim Boundary

This receipt may claim:

- exact Qwen3 0.6B Q8_0 artifact identity was checked;
- RTX 5070 Ti CUDA was selected;
- dense regular LLM CUDA route was used;
- bounded three-turn warm-session proof exists for the recorded prompts;
- CPU and CUDA generated token IDs match for all three turns;
- top-k membership matches for all recorded decode steps;
- fallback was not used;
- one model load, one tokenizer load, one CUDA context initialization, and
  upload-once weights were recorded;
- CUDA-MODEL-006 is proven for the recorded warm-session scope.

It must not claim:

- Qwen3 ask/chat product UX;
- Qwen3 broad product readiness;
- Qwen3 server readiness;
- Qwen3 speedup;
- persistent or full CUDA residency beyond the scoped receipt fields;
- dense Qwen proof is BitNet packed I2_S/QK256 proof;
- general dense GGUF CUDA readiness.

## Validation

```powershell
rtk python -m json.tool ci\hardware\windows-9950x3d-rtx5070ti\2026-05-15\qwen3-0_6b-warm-session-cuda.json
rtk cargo test --locked -p bitnet-receipts --test cuda_receipt_validation --no-default-features qwen_warm_session
rtk cargo check --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli
rtk cargo run --locked -p xtask --no-default-features -- campaign check nvidia-5070ti
rtk cargo run --locked -p xtask --no-default-features -- campaign generate --check
rtk cargo run --locked -p xtask --no-default-features -- campaign doctor
rtk git diff --check
```
