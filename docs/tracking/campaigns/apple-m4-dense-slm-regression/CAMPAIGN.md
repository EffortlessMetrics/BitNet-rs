# Apple M4 Dense SLM Regression Guardrails

Campaign ID: `apple-m4-dense-slm-regression`

Status: active

## Objective

Turn the measured Apple M4 dense SLM local-answer envelope into repeatable regression guardrails for quality, determinism, receipt schema, model-cache identity, warm-session timing, and memory drift without broadening the claim beyond the recorded Qwen2.5 dense SLM Apple CPU/NEON path.

## Why This Exists

The `apple-m4-slm-answer`, `apple-m4-productization`, `apple-m4-slm-performance`, and `apple-m4-slm-hardening` campaigns made the Mac mini path practical: Qwen2.5 0.5B Instruct runs as a dense regular SLM through Rust-native `apple-m4-cpu-neon`, has model cache UX, warm-session receipts, a small deterministic quality corpus, release-mode timing envelopes, and phase-scoped Metal evidence.

That proof is useful only if regressions are visible. This campaign turns the recorded envelope into guardrails while keeping the claim boundary tight: Qwen dense SLM evidence validates Mac UX and dense Apple CPU/NEON behavior, not BitNet math, 1-bit layouts, QK256, Neural Engine execution, MPSGraph model inference, or full `apple-m4-metal` inference.

## End State

- Matching release-mode receipts can be compared against the published Apple M4 dense SLM envelope.
- Receipt comparisons first validate model hash, tokenizer metadata, profile, backend, fallback status, machine context, and release-mode evidence.
- Quality and determinism failures are treated as correctness failures before timing drift is interpreted.
- Warm-session profiles track `warm_16`, `warm_32`, `warm_64`, and `warm_128` timing and memory fields.
- Metal phase receipts remain phase-scoped and cannot imply full Apple Metal inference.
- Regression thresholds mature only after repeated matching hardware receipts exist.

## Initial Guardrails

Hard failures for matching Apple M4 dense SLM regression receipts:

- Model source, file, SHA256, tokenizer metadata, or cache verification differs from the published envelope without an explicit baseline refresh item.
- `requested_backend`, `selected_backend`, `runtime_api`, or `fallback_used` differs from the expected Apple CPU/NEON route.
- Release-mode evidence is missing for performance comparisons.
- The quality corpus fails valid UTF-8, non-empty output, non-degenerate output, or deterministic greedy token identity.
- A receipt claims BitNet local-answer quality, full `apple-m4-metal` inference, Neural Engine execution, MPSGraph model inference, QK256 support, or broad M4 performance.

Initial advisory timing drift bands for matching receipts:

| Field | Advisory drift band |
|---|---:|
| `decode_tok_s` | more than 20% lower than baseline |
| `warm_prompt_tok_s` | more than 25% lower than baseline |
| `time_to_first_token_ms` / first-token mean | more than 25% higher than baseline |
| `total_session_ms` | more than 25% higher than baseline |
| `peak_memory_mb` | more than 15% higher than baseline |

These bands are intentionally conservative and advisory until repeated scheduled Apple hardware receipts exist. They are not a broad M4 performance claim.

## Staged Hardware Workflow

`M4-SLM-REG-003` defines the Apple hardware workflow shape without enabling scheduled execution. The staged workflow is manual-dispatch only, defaults to `enable_run=false`, and requires a provisioned self-hosted runner with `self-hosted`, `macOS`, `ARM64`, and `apple-m4-dense-slm` labels before it can generate receipts.

The workflow shape covers model-cache verification, low-disk preflight, five-prompt quality and determinism receipts, release-mode performance receipts, advisory baseline comparison, artifact retention, and branch/commit/optional PR reporting. A future item may add a real `schedule:` trigger only after runner availability and artifact retention are confirmed.

## Work Items

| Work item | Status | Notes |
|---|---|---|
| M4-SLM-REG-001 | merged | Add advisory receipt comparison for the published performance envelope. |
| M4-SLM-REG-002 | merged | Gate quality and deterministic greedy drift for the five-prompt corpus. |
| M4-SLM-REG-003 | merged | Define or stage scheduled Apple hardware regression reporting. |
| M4-SLM-REG-004 | merged | Add compact trend-history artifacts for release receipts. |
| M4-SLM-REG-005 | proposed | Tighten thresholds only after repeated matching receipts exist. |

## Review Policy

Each PR owns one regression item. Performance changes require matching before/after receipts with the same model, tokenizer, profile, backend, generation settings, fallback status, and machine context. Dense SLM regression evidence must remain separate from BitNet, QK256, Neural Engine, MPSGraph, and full Metal inference claims.
