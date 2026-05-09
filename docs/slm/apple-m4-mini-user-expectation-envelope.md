# Apple M4 Mini Dense SLM User Expectation Envelope

This page defines what a healthy M4 Mac mini should do for the supported dense
SLM path. It is an operator expectation document, not a broad Apple Silicon
benchmark. The evidence is scoped to the recorded M4 Mac mini receipts and the
supported dense Qwen model family.

## Supported User Path

Primary commands:

```bash
bitnet mac ask "What is 2+2?"
bitnet mac chat
bitnet mac smoke
bitnet mac doctor
bitnet mac regression <receipt.json> --baseline <baseline.json>
```

The default model remains:

```text
model_id = qwen2.5-0.5b-instruct-q8_0
model = Qwen2.5 0.5B Instruct Q8_0
sha256 = ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e
size_bytes = 675710816
cache_size_mib = 644.41
tokenizer_model = gpt2
tokenizer_pre = qwen2
prompt_template = qwen2.5
backend = apple-m4-cpu-neon
fallback_used = false
```

The supported non-default storage-conscious model is:

```text
model_id = qwen2.5-0.5b-instruct-q4_k_m
model = Qwen2.5 0.5B Instruct Q4_K_M
sha256 = 74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db
size_bytes = 491400032
cache_size_mib = 468.64
tokenizer_model = gpt2
tokenizer_pre = qwen2
prompt_template = qwen2.5
backend = apple-m4-cpu-neon
fallback_used = false
```

## Release-Mode Warm Envelope

Evidence receipt:

```text
ci/hardware/apple-m4-mac-mini/2026-05-08/slm-performance/release-baseline.json
```

This receipt was recorded from the release-mode performance profile set. Cold
model load is separated from warm prompt timing. The profile set loads the model
and tokenizer once per profile and runs the bounded prompt group for that token
budget.

| Profile | Requested max tokens | Generated tokens | Cold model load ms | Tokenizer load ms | Warm prompt wall ms | Approx warm wall ms / prompt | TTFT mean ms | Decode tok/s | Total session ms | Peak memory MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `warm_16` | 16 | 34 | 3254.630 | 72.183 | 7666.475 | 2555.492 | 1885.000 | 14.962 | 12576.726 | 3772.469 |
| `warm_32` | 32 | 50 | 3118.775 | 49.173 | 8407.277 | 2802.426 | 1779.333 | 15.317 | 13117.047 | 4009.078 |
| `warm_64` | 64 | 82 | 3105.861 | 49.333 | 10459.755 | 3486.585 | 1763.000 | 15.269 | 15228.741 | 4026.438 |
| `warm_128` | 128 | 123 | 3174.666 | 54.097 | 13158.941 | 4386.314 | 1775.333 | 15.313 | 17896.347 | 4033.422 |

Healthy expectations for this M4 Mac mini:

- `requested_backend = apple-m4-cpu-neon`;
- `selected_backend = apple-m4-cpu-neon`;
- `runtime_api = cpu`;
- `fallback_used = false`;
- cold load is visible and separated from warm prompt timing;
- warm time-to-first-token is around the recorded release envelope for matching
  model, profile, and machine context;
- peak memory for the Q8_0 release profile is roughly 3.8-4.1 GB.

## Resident Soak Envelope

Evidence receipts:

```text
ci/hardware/apple-m4-mac-mini/2026-05-09/M4-SLM-EX-008/resident-25-64.json
ci/hardware/apple-m4-mac-mini/2026-05-09/M4-SLM-EX-008/resident-50-128.json
ci/hardware/apple-m4-mac-mini/2026-05-09/M4-SLM-EX-008/summary.json
```

These receipts exercise longer resident sessions. They are for stability and
memory/timing drift, not release-mode speed claims.

| Profile | Prompts | Max new tokens | Generated tokens | TTFT mean ms | Decode tok/s | Total session ms | Peak memory MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| `resident_25_prompt_64_budget` | 25 | 64 | 482 | 4630.360 | 6.327 | 198130.965 | 4013.234 |
| `resident_50_prompt_128_budget` | 50 | 128 | 1185 | 4682.900 | 6.313 | 424606.240 | 4020.250 |

Both long-session receipts record:

- `model_loaded_once = true`;
- `tokenizer_loaded_once = true`;
- `quality_summary.passed = true`;
- deterministic repeated prompt groups pass;
- `fallback_used = false`.

The 50-prompt receipt increased peak memory by 7.016 MB over the 25-prompt
receipt in this run. That is the current local soak reference, not a fleet-wide
memory guarantee.

## Health And Regression Commands

Use `doctor` for one local health verdict:

```bash
bitnet mac doctor
```

Use `smoke` for a compact answer/cache receipt:

```bash
bitnet mac smoke
```

Use `regression` for receipt-only drift checks against matching M4 dense SLM
envelopes:

```bash
bitnet mac regression \
  ci/hardware/apple-m4-mac-mini/2026-05-09/M4-SLM-EX-008/resident-25-64.json \
  --baseline ci/hardware/apple-m4-mac-mini/2026-05-09/M4-SLM-EX-008/resident-25-64.json
```

`bitnet mac regression` is advisory by default. Add `--fail-on-drift` for a
local operator hard failure when the matching receipt exceeds timing or memory
thresholds.

## Unsupported Claims

This envelope does not claim:

- BitNet local-answer quality;
- QK256 on Apple Silicon;
- Neural Engine execution;
- MPSGraph model inference;
- full `apple-m4-metal` inference;
- broad Apple Silicon or M4 fleet performance.

Metal remains phase-scoped unless a strict full-pipeline receipt proves
otherwise. Dense Qwen success proves the M4 dense SLM runner path, not 1-bit
BitNet math.
