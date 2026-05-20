# RTX 5070 Ti Dense Qwen CUDA Guide

This guide is for the Qwen2.5 0.5B Q8_0 dense SLM lane on the 9950X3D + RTX
5070 Ti bench. It shows the normal commands for checking support, verifying the
model artifact, running dense CUDA ask and chat-style commands, reading the
committed receipts, and understanding what the lane does not prove.

The claim is intentionally narrow:

- model family: dense SLM, Qwen2.5 0.5B Q8_0
- required selected backend for strict proof: `nvidia-rtx-5070-ti-cuda`
- route: `dense_regular_llm_cuda`
- fallback: rejected in strict CUDA receipts
- speed: reviewed, not accepted
- server readiness: exact-profile only
- BitNet proof: false

Official BitNet I2_S/QK256 receipts do not prove this dense lane. This dense
lane also does not prove BitNet packed I2_S/QK256 behavior, QK256 kernels,
global dense GGUF support, full CUDA residency, accepted speedup, or broad
server readiness.

## Prerequisites

Use a local checkout or installed `bitnet` binary built with CUDA support. From a
checkout, the equivalent prefix is:

```powershell
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli --
```

The examples below use `bitnet` for readability.

The dense Qwen row is identified by the model coverage matrix as:

```text
row: dense_qwen25_05b_q8_cuda
model: qwen2.5-0.5b-instruct-q8_0
route: dense_regular_llm_cuda
```

## Check Current Model Status

Start with the model status command. It reads the model coverage matrix and does
not require CUDA hardware:

```powershell
bitnet model status --device nvidia-rtx-5070-ti-cuda
```

For automation:

```powershell
bitnet model status --device nvidia-rtx-5070-ti-cuda --format json
```

The dense Qwen row should show `product_cli_ready`, route
`dense_regular_llm_cuda`, `speedup_claim=false`, `server_ready=true`,
`server_scope=exact_profile`, endpoint `/v1/chat/completions`, and
`server_streaming=false`. That readiness is scoped only to the exact
shared-engine profile named by the model coverage matrix.

## Verify The Model Artifact

Verify the registered dense artifact before running answer paths:

```powershell
bitnet model verify qwen2.5-0.5b-instruct-q8_0
```

Artifact verification is necessary, but it is not enough by itself. Dense CUDA
answer claims also need tokenizer authority, prompt authority, CPU reference,
strict CUDA route receipts, answer-quality evidence, and claim booleans.

## Run Dense Ask

Use the exact RTX 5070 Ti device label when you need a strict selected-device
claim:

```powershell
bitnet ask `
  --device nvidia-rtx-5070-ti-cuda `
  --model qwen2.5-0.5b-instruct-q8_0 `
  --max-new-tokens 8 `
  "What is 2+2?"
```

If a convenience command uses `--device cuda`, the receipt must still resolve to
`selected_backend = nvidia-rtx-5070-ti-cuda` before it can support this lane's
strict CUDA claim.

The receipt must preserve:

- `route = dense_regular_llm_cuda`
- `selected_backend = nvidia-rtx-5070-ti-cuda`
- `runtime_api = cuda`
- `fallback_used = false`
- tokenizer and prompt authority
- quality-gate status
- `speedup_claim = false`
- `bitnet_packed_i2s_qk256_proof = false`

## Run A Chat-Style Session

For a bounded chat-style session:

```powershell
@("What is 2+2?", "Name the previous answer in words.") | bitnet chat `
  --device nvidia-rtx-5070-ti-cuda `
  --model qwen2.5-0.5b-instruct-q8_0 `
  --max-tokens 8
```

The committed warm-session proof is a bounded CLI/session proof. It records
model, tokenizer, and CUDA context reuse for the proof run, but it does not
itself claim full CUDA residency, broad chat quality, server readiness, or
speedup.

## Explain Dense Receipts

Explain the latest receipt:

```powershell
bitnet receipts explain --latest
```

Or inspect the committed dense receipts directly:

```powershell
bitnet receipts explain ci\hardware\windows-9950x3d-rtx5070ti\2026-05-13\dense-qwen25-q8-one-token-cuda.json

bitnet receipts explain ci\hardware\windows-9950x3d-rtx5070ti\2026-05-14\dense-qwen25-q8-short-decode-current-source.json

bitnet receipts explain ci\hardware\windows-9950x3d-rtx5070ti\2026-05-14\dense-qwen25-q8-warm-session-current-source.json
```

The useful fields to paste into an issue are:

- model coverage row
- model id and artifact identity
- route
- requested and selected backend
- fallback status
- generated-token equality or first divergence
- quality gate
- kernel and transfer fields
- speed qualification state
- server readiness state
- claim boundary

For support issues, collect the status row and latest receipt explanation in one
artifact:

```powershell
bitnet support bundle --latest --device nvidia-rtx-5070-ti-cuda --format json
```

For Qwen2.5, the bundle should show `selected_route =
dense_regular_llm_cuda`, `server_ready = true`,
`server_ready_scope = exact_profile`, `speedup_claim = false`,
`full_residency_claim = false`, `bitnet_packed_i2s_qk256_proof = false`, and
`dense_regular_llm_cuda_proof = true`. Those Qwen2.5 claims do not transfer to
Qwen3 or any other dense model row.

## Read The Governed Benchmark Receipt

The current dense Qwen benchmark qualification review is committed as:

```powershell
bitnet bench --device cuda `
  --cuda-benchmark-receipt ci\hardware\windows-9950x3d-rtx5070ti\2026-05-14\dense-qwen25-q8-benchmark-qualification-current-source.json
```

This is report-only receipt inspection. It is not a fresh live CUDA benchmark
run. The current review keeps:

```text
speedup_claim = false
benchmark_qualified_speedup = false
accepted_profiles = []
```

Speed remains unqualified until a later governed review accepts an exact
same-artifact, fallback-free profile.

## Server Readiness Is Exact-Profile Only

The refreshed bounded server-smoke receipt promotes readiness only for the exact
dense Qwen non-streaming shared-engine chat-completions profile:

```powershell
bitnet receipts explain ci\hardware\windows-9950x3d-rtx5070ti\2026-05-17\server-strict-dense-qwen25-q8-smoke.json
```

That receipt carries the model SHA-256, `/v1/chat/completions` endpoint
profile, non-streaming request profile, greedy generation policy, strict
`nvidia-rtx-5070-ti-cuda` backend, dense route, and non-empty UTF-8 response
quality gate. Do not turn this exact-profile readiness into broad dense server
readiness or production-serving readiness.

## What This Guide Does Not Claim

This guide does not claim:

- BitNet I2_S, QK256, 1-bit, or packed-kernel proof
- Qwen3, SmolLM2, Llama, Gemma, or Phi support
- accepted CUDA speedup
- broad server readiness beyond the exact dense Qwen profile
- full CUDA residency
- global dense GGUF support
- generic `cuda`, WGPU, Vulkan, or CPU fallback as strict RTX 5070 Ti proof
- crates.io or docs.rs publication

The source-of-truth status remains the model coverage matrix, NVIDIA 5070 Ti
campaign tracker, committed receipts, and governed benchmark reports.
