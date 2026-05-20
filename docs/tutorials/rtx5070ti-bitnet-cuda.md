# RTX 5070 Ti BitNet CUDA Guide

This guide is for the strict official BitNet path on the 9950X3D + RTX 5070 Ti
bench. It shows the normal commands for checking support, verifying the model
artifact, running a strict CUDA ask, running a warm session, explaining receipts,
and reading the governed benchmark receipt.

The claim is intentionally narrow:

- model family: official Microsoft BitNet 2B I2_S/QK256
- required selected backend: `nvidia-rtx-5070-ti-cuda`
- route: `bitnet_qk256_cuda`
- fallback: rejected under strict CUDA
- speed: not accepted unless a benchmark qualification receipt says so
- server readiness: not claimed here

Dense SLM CUDA receipts, including Qwen2.5 0.5B Q8_0, do not prove this BitNet
packed I2_S/QK256 lane. This guide also does not promote generic `cuda`, WGPU,
CPU fallback, or server execution into RTX 5070 Ti CUDA proof.

## Prerequisites

Use a local checkout or installed `bitnet` binary built with CUDA support. From a
checkout, the equivalent prefix is:

```powershell
cargo run --locked -p bitnet-cli --no-default-features --features cpu,cuda,full-cli --
```

The examples below use `bitnet` for readability.

You need the accepted official I2_S GGUF and its external tokenizer available
locally. The receipt-backed lane uses these paths in the committed evidence:

```text
models\microsoft-bitnet-b1.58-2B-4T-gguf\ggml-model-i2_s.gguf
models\microsoft-bitnet-b1.58-2B-4T\tokenizer.json
```

## Check Current Model Status

Start with the status matrix. This does not require CUDA hardware; it reads the
model coverage source of truth and reports what each model row may claim.

```powershell
bitnet model status --device nvidia-rtx-5070-ti-cuda
```

For automation:

```powershell
bitnet model status --device nvidia-rtx-5070-ti-cuda --format json
```

The BitNet row should stay scoped to `bitnet_qk256_cuda`, show server smoke as
bounded evidence while keeping `server_ready=false` for broad production
readiness, and keep speed unqualified unless a profile-specific benchmark
review accepts it.

## Verify The Model Artifact

Verify the registered artifact before running answer paths:

```powershell
bitnet model verify microsoft-bitnet-b1.58-2B-4T-i2s
```

Artifact verification is necessary, but it is not sufficient by itself. Coherent
answer claims also need tokenizer authority, prompt authority, a strict backend
receipt, and answer-quality evidence.

## Check CUDA Readiness

Use CUDA doctor to confirm that the runtime can see the model/tokenizer pair and
write a diagnostic receipt:

```powershell
bitnet cuda doctor `
  --model models\microsoft-bitnet-b1.58-2B-4T-gguf\ggml-model-i2_s.gguf `
  --tokenizer models\microsoft-bitnet-b1.58-2B-4T\tokenizer.json `
  --json-out target\bitnet\receipts\cuda-answer-readiness\cuda-doctor.json
```

Doctor output is a readiness diagnostic. It is not an answer-quality receipt and
does not create a speed claim.

## Run Strict Ask

Use the exact RTX 5070 Ti device label and strict CUDA mode for answer receipts:

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

For this command to support the strict RTX 5070 Ti claim, the receipt must show:

- `selected_backend = nvidia-rtx-5070-ti-cuda`
- `runtime_api = cuda`
- `route = bitnet_qk256_cuda`
- `fallback_used = false`
- official BitNet I2_S/QK256 model identity
- tokenizer and prompt authority
- answer-quality gate result
- no accepted speedup claim unless a benchmark receipt separately qualifies it

If a receipt resolves to generic `cuda`, CPU, WGPU, or a fallback route, it is not
strict RTX 5070 Ti CUDA proof.

## Run A Warm Session

Use the warm-session path to prove that the user-facing route can keep a session
alive across multiple prompts without promoting it to server readiness:

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

Warm-session proof is still CLI proof. It does not claim `/v1/chat/completions`
server readiness.

## Explain Receipts

Explain the latest receipt:

```powershell
bitnet receipts explain --latest
```

Or explain a specific receipt:

```powershell
bitnet receipts explain target\bitnet\receipts\cuda-answer-readiness\strict-cuda-ask-latest.json
```

The explanation should make the route, selected backend, fallback state, quality
gate, and claim boundary obvious enough to paste into an issue.

For support issues, collect the status row and latest receipt explanation in one
artifact:

```powershell
bitnet support bundle --latest --device nvidia-rtx-5070-ti-cuda --format json
```

For this BitNet lane, the bundle should preserve `selected_route =
bitnet_qk256_cuda`, `bitnet_packed_i2s_qk256_proof = true`,
`dense_regular_llm_cuda_proof = false`, `server_ready = false`, and
`speedup_claim = false` unless a later exact-profile review explicitly changes
those claims.

## Read The Governed Benchmark Receipt

The current benchmark qualification review is committed as a governed receipt:

```powershell
bitnet bench --device cuda `
  --cuda-benchmark-receipt ci\hardware\windows-9950x3d-rtx5070ti\2026-05-13\cuda-prod-010-benchmark-qualification.json
```

JSON and CSV inspection modes are also available:

```powershell
bitnet bench --device cuda `
  --cuda-benchmark-receipt ci\hardware\windows-9950x3d-rtx5070ti\2026-05-13\cuda-prod-010-benchmark-qualification.json `
  --format json

bitnet bench --device cuda `
  --cuda-benchmark-receipt ci\hardware\windows-9950x3d-rtx5070ti\2026-05-13\cuda-prod-010-benchmark-qualification.json `
  --format csv
```

This report-only path reads an existing benchmark review. It is not a fresh live
CUDA benchmark run. The current review keeps `speedup_claim = false` and
`benchmark_qualified_speedup = false`.

## What This Guide Does Not Claim

This guide does not claim:

- dense SLM CUDA proof
- Qwen, Qwen3, SmolLM, Llama, Gemma, or Phi support
- broad chat quality
- server readiness
- global CUDA speedup
- full CUDA residency
- generic GPU proof
- crates.io or docs.rs publication

The source-of-truth status remains the model coverage matrix, NVIDIA 5070 Ti
campaign tracker, committed receipts, and governed benchmark reports.
