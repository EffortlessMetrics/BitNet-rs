# CUDA-DENSE-047 Qwen 1.5B Q4_K_M Route Audit

`CUDA-DENSE-047` records the first RTX 5070 Ti dense CUDA route audit for the
registered larger Qwen row:

```text
model id: qwen2.5-1.5b-instruct-q4_k_m
artifact: Qwen/Qwen2.5-1.5B-Instruct-GGUF
file: qwen2.5-1.5b-instruct-q4_k_m.gguf
sha256: 6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e
bytes: 1117320736
```

## Evidence

The artifact was fetched and verified through the model cache:

```text
cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- model fetch qwen2.5-1.5b-instruct-q4_k_m --json
cargo run --locked -p bitnet-cli --no-default-features --features cpu,full-cli -- model verify qwen2.5-1.5b-instruct-q4_k_m --json
```

The local 5070 Ti was visible:

```text
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
NVIDIA GeForce RTX 5070 Ti, 591.86, 16303 MiB
```

The CUDA toolchain was available only after entering the Visual Studio x64
environment and adding CUDA v12.9 to `PATH`:

```text
vcvars64.bat
nvcc: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe
cl: C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe
```

The existing strict all-layer planner was then run against the verified 1.5B
Q4_K_M artifact:

```text
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli -- dense-gguf-all-layer-plan --model C:\Users\steven\AppData\Local\bitnet-rs\models\qwen2.5-1.5b-instruct-q4_k_m\qwen2.5-1.5b-instruct-q4_k_m.gguf --device nvidia-rtx-5070-ti-cuda --device-index 0 --json-out ci\hardware\windows-9950x3d-rtx5070ti\2026-05-11\dense-gguf-all-layer-plan-qwen25-15b-q4km.json
```

The command selected the intended CUDA lane and rejected fallback:

```text
requested_backend=nvidia-rtx-5070-ti-cuda
selected_backend=nvidia-rtx-5070-ti-cuda
runtime_api=cuda
fallback_used=false
```

It then failed closed before writing a strict CUDA-ready receipt:

```text
Command failed: field `strict_cuda_ready` must be `true`, got `false`
```

## Interpretation

This is not a Qwen 0.5B Q8_0 regression. It is a model/artifact coverage
boundary: the already-validated dense Qwen 0.5B Q8_0 CUDA receipts do not prove
that the larger Qwen 1.5B Q4_K_M artifact is strict CUDA route-ready.

The next engineering gate for this row is one of:

```text
1. add explicit Q4_K_M dense CUDA route support and then rerun the all-layer plan;
2. add a validated unsupported-route receipt kind for larger dense GGUF candidates;
3. keep the row CPU-answer-ready only and leave CUDA claims false.
```

## Claim Boundary

This audit does not claim:

```text
dense_regular_llm_cuda_proof
cuda_answer_ready
Qwen one-token, short decode, or chat
speedup
full CUDA residency
server readiness
BitNet packed I2_S/QK256 proof
```
