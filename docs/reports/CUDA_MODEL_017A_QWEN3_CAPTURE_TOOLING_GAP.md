# CUDA-MODEL-017A Qwen3 Capture Tooling Gap

Status: prerequisite blocker for CUDA-MODEL-017

CUDA-MODEL-017 requires repeated same-artifact Qwen3 CPU/CUDA comparator source
receipts for five profiles:

```text
one_token
short_decode_8
short_decode_32
warm_session_3_turns
decode_128_from_warm_context
```

The aggregate receipt generator and manifest exist, but current source-capture
commands do not yet cover every required profile.

## Current Executable Surface

The CLI currently exposes dense Qwen strict CUDA proof commands for:

```text
dense-gguf-qwen-one-token-strict-cuda
dense-gguf-qwen-short-decode-strict-cuda
dense-gguf-qwen-warm-session-strict-cuda
```

Those are enough for `one_token`, `short_decode_8`, and
`warm_session_3_turns`.

## Blockers

`short_decode_32` is not currently producible through the source-capture
command because the short-decode command and receipt validator reject generated
token counts outside `5..=16`.

`decode_128_from_warm_context` is not currently producible as a distinct source
receipt because there is no governed warm-decode source-capture command or
receipt validator for the expected 128-token warm-context profile.

## Required Tooling Before CUDA-MODEL-017

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

The preferred source artifact kind for the warm-context profile is:

```text
dense_gguf_qwen_warm_decode_strict_cuda_proof
```

If implementation review chooses a different artifact shape, the
`qwen3_cuda_repeated_comparator` aggregate contract must be updated in the same
PR so the source receipt meaning remains explicit.

## Claim Boundary

This report proves only that CUDA-MODEL-017 is blocked on capture tooling. It
does not prove Qwen3 hardware execution, repeated comparator evidence, speedup,
benchmark-qualified speed, server readiness, full residency, broad dense GGUF
readiness, Qwen2.5 inheritance, or BitNet packed I2_S/QK256 proof.
