# Apple M4 Dense SLM Regression Guardrails

The Apple M4 dense SLM path is the practical Mac local-answer baseline for the validated Qwen2.5 0.5B Instruct GGUF. This document defines how regression work should compare future receipts against the published Apple M4 envelope without turning one machine run into a broad performance claim.

## Scope

This lane covers:

```text
Qwen2.5 0.5B Instruct dense SLM
apple-m4-cpu-neon
warm-session local answers
quality and determinism receipts
release-mode timing receipts
phase-scoped Metal receipts
```

It does not cover:

```text
BitNet local-answer quality
1-bit / 1.58-bit kernel correctness
I2_S / TL1 / TL2 BitNet proof
QK256 on Apple Silicon
full apple-m4-metal inference
MPSGraph model inference
Neural Engine execution
fleet-wide Apple Silicon performance
```

## Baseline Context

Regression comparisons are meaningful only when these fields match the published envelope:

```text
machine_id = apple-m4-mac-mini
model = Qwen2.5 0.5B Instruct Q8_0
model_sha256 = ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e
requested_backend = apple-m4-cpu-neon
selected_backend = apple-m4-cpu-neon
runtime_api = cpu
fallback_used = false
release_mode_observed = true
profiles = warm_16, warm_32, warm_64, warm_128
```

The current published envelope is documented in `docs/slm/apple-m4-slm-performance.md` and backed by the receipts under:

```text
ci/hardware/apple-m4-mac-mini/2026-05-08/slm-performance/
```

## Initial Thresholds

Quality, determinism, receipt schema, model identity, backend routing, and fallback status are correctness gates. Performance drift should be interpreted only after those pass.

`M4-SLM-REG-001` adds advisory comparison through the existing receipt checker:

```bash
bitnet mac receipts-check \
  target/apple-m4-slm-regression/current-release-baseline.json \
  --regression-baseline ci/hardware/apple-m4-mac-mini/2026-05-08/slm-performance/release-baseline.json \
  --json
```

The comparison first runs the normal Mac receipt validator. It then compares timing and memory only when both receipts are `apple_m4_slm_performance_profiles` with matching dense Qwen model identity, tokenizer metadata, Apple CPU/NEON backend routing, fallback status, release-mode evidence, profile set, and required profiles. Timing and memory drift are reported as advisory warnings rather than hard failures.

Initial advisory timing bands for matching receipts:

| Field | Advisory drift band |
|---|---:|
| `decode_tok_s` | more than 20% lower than baseline |
| `warm_prompt_tok_s` | more than 25% lower than baseline |
| `time_to_first_token_ms` / first-token mean | more than 25% higher than baseline |
| `total_session_ms` | more than 25% higher than baseline |
| `peak_memory_mb` | more than 15% higher than baseline |

These thresholds are intentionally conservative. They should become stricter only after repeated matching release-mode receipts exist from the scheduled Apple hardware lane.

## Failure Classes

Hard regression classes:

- Model, tokenizer, cache verification, profile, backend, fallback, or release-mode context does not match.
- The five-prompt dense SLM quality corpus fails valid UTF-8, non-empty text, non-degenerate output, or deterministic greedy token identity.
- Required receipt fields for timing, token counts, backend routing, fallback, model identity, or tokenizer authority are missing.
- A receipt claims BitNet quality, full Apple Metal inference, Neural Engine execution, MPSGraph model inference, QK256 support, or broad M4 performance from dense SLM evidence.

Advisory regression classes:

- Matching release receipt exceeds one of the initial timing or memory drift bands.
- Metal phase timing changes without breaking CPU/Metal parity or fallback visibility.
- Peak memory changes without a cache/model/backend mismatch.

## Claim Boundary

Dense Qwen SLM regression evidence validates the Apple M4 dense SLM UX and Apple CPU/NEON local-answer path. It does not validate BitNet, 1-bit / 1.58-bit kernels, QK256, Neural Engine execution, MPSGraph model inference, or full `apple-m4-metal` inference.
