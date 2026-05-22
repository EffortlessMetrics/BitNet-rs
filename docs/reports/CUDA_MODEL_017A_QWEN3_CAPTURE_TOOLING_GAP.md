# CUDA-MODEL-017A Qwen3 Capture Tooling Gap

Status: source-capture tooling implemented for review

CUDA-MODEL-017 requires repeated same-artifact Qwen3 CPU/CUDA comparator source
receipts for five profiles:

```text
one_token
short_decode_8
short_decode_32
warm_session_3_turns
decode_128_from_warm_context
```

The aggregate receipt generator and manifest exist. CUDA-MODEL-017A closes the
source-capture tooling gap for the two profiles that were not executable through
current source before this work.

## Current Executable Surface

The CLI exposes dense Qwen strict CUDA proof commands for:

```text
dense-gguf-qwen-one-token-strict-cuda
dense-gguf-qwen-short-decode-strict-cuda
dense-gguf-qwen-warm-decode-strict-cuda
dense-gguf-qwen-warm-session-strict-cuda
```

Those cover `one_token`, `short_decode_8`, `short_decode_32`,
`warm_session_3_turns`, and `decode_128_from_warm_context`.

## Tooling Added

`short_decode_32` is produced with the existing short-decode command using an
explicit Qwen3-only capture profile:

```powershell
bitnet dense-gguf-qwen-short-decode-strict-cuda `
  --model <qwen3-0.6b-instruct-q8_0.gguf> `
  --capture-profile qwen3-short-decode-32 `
  --max-new-tokens 32 `
  --json-out <receipt.json>
```

`decode_128_from_warm_context` is produced with a distinct governed
warm-context decode command and receipt validator:

```powershell
bitnet dense-gguf-qwen-warm-decode-strict-cuda `
  --model <qwen3-0.6b-instruct-q8_0.gguf> `
  --max-new-tokens 128 `
  --json-out <receipt.json>
```

The warm-context source receipt uses the artifact kind:

```text
dense_gguf_qwen_warm_decode_strict_cuda_proof
```

The repeated-comparator aggregate now requires this explicit warm-decode source
artifact for `decode_128_from_warm_context`; it no longer accepts an ambiguous
short-decode receipt for that profile.

## CUDA-MODEL-017 Preconditions

The enabling work must make the CUDA-MODEL-017 profiles executable from current
source without broadening product ask/chat claims:

```text
short_decode_32:
  exact Qwen3 0.6B Q8_0 artifact
  dense_regular_llm_cuda route
  selected_backend = nvidia-rtx-5070-ti-cuda
  fallback_used = false
  32 generated tokens
  no ask/chat max-token expansion

decode_128_from_warm_context:
  exact Qwen3 0.6B Q8_0 artifact
  dense_regular_llm_cuda route
  selected_backend = nvidia-rtx-5070-ti-cuda
  fallback_used = false
  128 generated tokens
  explicit warm-context/session reuse evidence
  no speedup, benchmark-qualified, server-ready, or full-residency claim
```

## Claim Boundary

This report and tooling prove only that current source can emit or validate the
five CUDA-MODEL-017 source-capture profiles. They do not prove Qwen3 hardware
execution, repeated comparator evidence, speedup, benchmark-qualified speed,
server readiness, full residency, broad dense GGUF readiness, Qwen2.5
inheritance, or BitNet packed I2_S/QK256 proof.
