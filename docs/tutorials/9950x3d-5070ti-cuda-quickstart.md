# 9950X3D + RTX 5070 Ti CUDA Quickstart

This quickstart is for the Windows 9950X3D + RTX 5070 Ti CUDA product bench.
It shows the normal operator path for checking the model matrix, verifying a
model artifact, running strict CUDA ask/chat-style commands, reading governed
benchmark receipts, and explaining the latest proof receipt.

The quickstart is a guide, not new proof. It summarizes the current
source-of-truth state from the model coverage matrix, NVIDIA 5070 Ti campaign,
and committed receipts.

## What This Bench Can Currently Claim

| Surface | Current claim | Must not claim |
| --- | --- | --- |
| Official BitNet 2B I2_S/QK256 | Product CLI ready on `nvidia-rtx-5070-ti-cuda`; route is `bitnet_qk256_cuda`; fallback is rejected in strict CUDA receipts. | Dense SLM proof, server readiness, global CUDA speedup, or generic GPU proof. |
| Qwen2.5 0.5B Q8_0 dense SLM | Product CLI ready for the bounded dense CUDA lane; route is `dense_regular_llm_cuda`; one-token, short-decode, warm-session, benchmark review, and exact-profile server-readiness receipts exist. | BitNet packed I2_S/QK256 proof, accepted speedup, full CUDA residency, broad dense GGUF support, or broad/concurrent/deployment server readiness. |
| Qwen3 0.6B Q8_0 dense SLM | Product CLI ready for bounded ask/chat CLI paths after its own artifact, CPU sanity, all-layer plan, strict CUDA one-token, short-decode, warm-session, benchmark-review, and user-path ask/chat receipts. | Inherited Qwen2.5 proof, BitNet proof, accepted speedup, server readiness, broad dense GGUF support, or full CUDA residency. |
| SmolLM2, Llama, Gemma, Phi candidates | SmolLM2 360M is structurally valid but CPU quality-blocked; later Llama, Gemma, and Phi rows are registered only. | CUDA answer readiness without their own ladder, inherited Qwen/Qwen3 proof, BitNet proof, speedup, or server readiness. |

## Use A CUDA-Capable Build

From a checkout, use this prefix for the examples when you have not installed a
`bitnet` binary:

```powershell
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli --
```

The examples below use `bitnet` for readability. Crates.io and docs.rs badges
remain pending until the project is actually published.

## 1. Check The CUDA Model Status

```powershell
bitnet model status --device nvidia-rtx-5070-ti-cuda
```

For automation:

```powershell
bitnet model status --device nvidia-rtx-5070-ti-cuda --format json
```

This command proves only that the local status surface can read the model
coverage source of truth. It does not require CUDA hardware and does not execute
inference. The important fields are the model class, route, current tier,
`speedup_claim`, `server_ready`, and claim boundary.

## 2. Verify A Model Artifact

Official BitNet I2_S/QK256:

```powershell
bitnet model verify microsoft-bitnet-b1.58-2B-4T-i2s
```

Dense Qwen2.5 0.5B Q8_0:

```powershell
bitnet model verify qwen2.5-0.5b-instruct-q8_0
```

Verification proves the named artifact contract can be resolved. It does not by
itself prove answer quality, CUDA execution, benchmark speed, or server
readiness.

## 3. Run The Official BitNet Strict CUDA Ask

Use the selected-device label for strict RTX 5070 Ti proof:

```powershell
bitnet --device nvidia-rtx-5070-ti-cuda ask `
  --model models\microsoft-bitnet-b1.58-2B-4T-gguf\ggml-model-i2_s.gguf `
  --tokenizer models\microsoft-bitnet-b1.58-2B-4T\tokenizer.json `
  --question "What is 2+2? Answer with only the number." `
  --max-new-tokens 8 `
  --temperature 0 `
  --strict-cuda `
  --receipt-out target\bitnet\receipts\cuda-answer-readiness\strict-cuda-ask-latest.json
```

This command can prove a bounded official BitNet CUDA answer only when the
receipt records:

- `selected_backend = nvidia-rtx-5070-ti-cuda`
- `runtime_api = cuda`
- `route = bitnet_qk256_cuda`
- `fallback_used = false`
- official Microsoft I2_S/QK256 model identity
- tokenizer and prompt authority
- answer-quality gate status
- `speedup_claim = false` unless a separate benchmark receipt accepts the exact profile

It does not prove dense Qwen, Qwen3, server readiness, or global speedup.

## 4. Run The Official BitNet Warm Session

```powershell
bitnet --device nvidia-rtx-5070-ti-cuda cuda-warm-session `
  --model models\microsoft-bitnet-b1.58-2B-4T-gguf\ggml-model-i2_s.gguf `
  --tokenizer models\microsoft-bitnet-b1.58-2B-4T\tokenizer.json `
  --prompt "What is 2+2? Answer with only the number." `
  --prompt "Name the route selected by this receipt." `
  --max-new-tokens 8 `
  --temperature 0 `
  --greedy `
  --deterministic `
  --strict-tokenizer `
  --strict-loader `
  --fail-on-quality `
  --json-out target\bitnet\receipts\cuda-answer-readiness\strict-cuda-warm-session.json
```

This proves the bounded CLI warm-session surface when the receipt validates. It
does not promote the server API, broad chat quality, full residency, or speed.

## 5. Run Dense Qwen Ask And Chat

Dense Qwen uses a different model family and route:

```powershell
bitnet ask `
  --device cuda `
  --model qwen2.5-0.5b-instruct-q8_0 `
  --max-new-tokens 8 `
  "What is 2+2?"
```

For a bounded chat-style session:

```powershell
@("What is 2+2?", "Name the previous answer in words.") | bitnet chat `
  --device cuda `
  --model qwen2.5-0.5b-instruct-q8_0 `
  --max-tokens 8
```

These commands are the dense SLM user path. Their receipts must resolve to
`dense_regular_llm_cuda` and the selected RTX 5070 Ti backend before they can
support the scoped dense CUDA claim. They do not prove BitNet packed
I2_S/QK256, QK256 kernels, accepted speedup, broad server readiness, or full
CUDA residency. Exact-profile dense Qwen server readiness is a separate
server-smoke claim; it is limited to the named shared-engine
`/v1/chat/completions` receipt and does not follow from every ask/chat receipt.
The committed dense Qwen hardware receipts remain the durable proof set until
direct user-path ask/chat receipts are committed under
`ci/hardware/windows-9950x3d-rtx5070ti`.

## 6. Read Governed Benchmark Receipts

Official BitNet benchmark qualification:

```powershell
bitnet bench --device cuda `
  --cuda-benchmark-receipt ci\hardware\windows-9950x3d-rtx5070ti\2026-05-13\cuda-prod-010-benchmark-qualification.json
```

Dense Qwen benchmark qualification:

```powershell
bitnet bench --device cuda `
  --cuda-benchmark-receipt ci\hardware\windows-9950x3d-rtx5070ti\2026-05-14\dense-qwen25-q8-benchmark-qualification-current-source.json
```

JSON and CSV inspection modes are available:

```powershell
bitnet bench --device cuda `
  --cuda-benchmark-receipt ci\hardware\windows-9950x3d-rtx5070ti\2026-05-13\cuda-prod-010-benchmark-qualification.json `
  --format json

bitnet bench --device cuda `
  --cuda-benchmark-receipt ci\hardware\windows-9950x3d-rtx5070ti\2026-05-13\cuda-prod-010-benchmark-qualification.json `
  --format csv
```

This is report-only receipt inspection. It is not a fresh benchmark run and it
does not select a live CUDA backend. The current BitNet and dense Qwen benchmark
reviews keep speedup rejected.

## 7. Explain The Latest Receipt

```powershell
bitnet receipts explain --latest
```

Or explain a specific receipt:

```powershell
bitnet receipts explain target\bitnet\receipts\cuda-answer-readiness\strict-cuda-ask-latest.json
```

The useful issue-report fields are:

- model coverage row or model identity
- route
- requested and selected backend
- fallback state
- quality gate state
- kernel and transfer timing fields
- speed qualification state
- server readiness state
- claim boundary

If the receipt says generic `cuda`, CPU fallback, WGPU, Vulkan, or a different
model family, it does not prove the strict RTX 5070 Ti CUDA claim.

For a pasteable support artifact, use:

```powershell
bitnet support bundle --latest --device nvidia-rtx-5070-ti-cuda --format json
```

The bundle combines model status, the latest receipt explanation, selected
backend/route, fallback state, quality gate, server-readiness scope,
speed/residency claims, proof-family booleans, binary identity, and runtime
identity when the receipt records it.

## Current Stop Lines

Do not use this quickstart to claim:

- crates.io or docs.rs publication
- generic CUDA speedup
- broad or inherited server readiness; exact-profile server readiness must name
  the accepted receipt and profile
- full CUDA residency
- Qwen3 server readiness, accepted speedup, full residency, broad dense GGUF
  support, or Qwen2.5 proof inheritance
- dense Qwen proof as BitNet proof
- official BitNet QK256 proof as dense SLM proof
- WGPU, Vulkan, CPU fallback, or generic `cuda` as selected RTX 5070 Ti proof

Use the status command and receipt explanation together. Status tells you what
the repo currently allows each model row to claim; receipts tell you what the
last command actually proved.
