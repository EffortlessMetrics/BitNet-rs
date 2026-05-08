# CUDA-PROD-002 Warm Session Receipts

## Summary

`CUDA-PROD-002` adds a strict RTX 5070 Ti CUDA warm-session path for the
official BitNet I2_S answer lane. The new CLI path runs multiple prompts in one
process while preserving the same claim boundary as the strict answer proof:
selected RTX 5070 Ti CUDA backend, `qk256_gemv_cuda`, no BitNet linear CPU
fallback, upload-once QK256 weight handles, and `speedup_claim=false`.

The command is intentionally receipt-first:

```powershell
bitnet --device nvidia-rtx-5070-ti-cuda cuda-warm-session `
  --model <official-i2s-gguf> `
  --tokenizer <official-tokenizer-json> `
  --prompt "What is 2+2? Answer with only the number." `
  --prompt "Answer yes or no: is water wet?" `
  --max-new-tokens 8 `
  --temperature 0 `
  --greedy `
  --deterministic `
  --strict-loader `
  --strict-tokenizer `
  --fail-on-quality `
  --json-out target\bitnet\receipts\cuda-answer-readiness\strict-cuda-warm-session.json
```

## Receipt Contract

The aggregate receipt uses:

```text
artifact_kind = bitnet_cuda_warm_session
selected_backend = nvidia-rtx-5070-ti-cuda
runtime_api = cuda
fallback_used = false
speedup_claim = false
```

It records:

- `session.model_loaded_once=true`
- `session.tokenizer_loaded_once=true`
- `session.qk256_weights_uploaded_once=true`
- `session.per_token_weight_upload=false`
- per-turn receipt paths
- total `qk256_gemv_cuda` invocation coverage
- `bitnet_linear_layers_cpu_fallback=0`
- per-turn answer quality
- prompt prefill status
- explicit claim boundaries

Each per-turn receipt uses `artifact_kind = bitnet_cuda_warm_session_turn` and
records prompt IDs, generated IDs, answer text, prompt rendering, backend
identity, QK256 CUDA coverage delta, quality, timing, and the same no-speed /
no-full-residency boundary.

## Claim Boundary

Allowed claim:

- Strict RTX 5070 Ti CUDA warm sessions can load the model and tokenizer once,
  reuse the process-local QK256 CUDA context and upload-once weight handles
  across multiple deterministic prompts, and emit per-turn plus aggregate
  receipts.

Not allowed:

- CUDA speedup.
- Broad chat quality beyond deterministic prompts.
- Production server readiness.
- Full CUDA residency for every transformer operation.
- KV-cache reuse across user turns.

The current warm-session receipt deliberately records
`kv_cache_reuse_policy = recreated_per_turn_for_prompt_isolation`. KV reuse is a
later product milestone; this PR does not claim it.

## Local Validation Evidence

The 9950X3D + RTX 5070 Ti lane ran a two-turn strict CUDA warm session on
2026-05-08 against the official Microsoft BitNet I2_S GGUF and external
tokenizer:

```text
turn 1: 4
turn 2: No. Water is not wet; it
```

The parsed aggregate receipt reported:

```text
artifact_kind = bitnet_cuda_warm_session
selected_backend = nvidia-rtx-5070-ti-cuda
runtime_api = cuda
fallback_used = false
turns = 2
model_loaded_once = true
tokenizer_loaded_once = true
qk256_weights_uploaded_once = true
per_token_weight_upload = false
qk256_cuda_invocations = 9030
bitnet_linear_layers_cpu_fallback = 0
quality_summary.passed = true
strict_session_validation.passed = true
speedup_claim = false
```

No generated target receipts or model binaries are committed by this report.

## Next Work

The next CUDA product gate should add explicit CUDA execution residency
accounting for the rest of the transformer loop. `CUDA-PROD-002` proves warm
strict session reuse for model/tokenizer/QK256 weight handles; it does not claim
that norms, RoPE, attention, KV cache, sampling, or every host-device transfer
are fully CUDA-resident.
