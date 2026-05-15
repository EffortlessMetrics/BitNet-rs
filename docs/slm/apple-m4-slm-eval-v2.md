# Apple M4 Dense SLM Eval V2

This page defines the second dense SLM eval layer for the M4 Mac mini. It keeps
the existing `apple-m4-slm-eval-and-proof` artifacts as the v1 baseline and adds
a wider v2 path for repeatable quality, benchmark, and regression reporting.

## Corpus Contract

The v2 corpus is:

```text
ci/quality/apple-m4-slm-eval-seeded-corpus-v2.yaml
```

It contains 120 deterministic cases generated from seed `777331` across these
task families:

| Family | Cases | Primary scoring |
|---|---:|---|
| `arithmetic_exact` | 20 | `exact_match` |
| `numeric_tolerance` | 10 | `numeric_tolerance` |
| `fixed_table_qa` | 12 | `exact_match` |
| `format_constrained_json` | 10 | `json_schema` |
| `closed_label_classification` | 12 | `exact_match` |
| `synthetic_extraction` | 12 | `exact_match` |
| `ordering_sorting` | 12 | `normalized_match` |
| `copy_edit_rewrite` | 12 | `required_keywords` |
| `constrained_summary` | 10 | `required_keywords` |
| `instruction_following_required_forbidden` | 10 | `required_forbidden_tokens` |

`M4-SLM-EVAL2-001` validates only the corpus shape and deterministic scoring
metadata through `answer-corpus --dry-run`. It does not run live model
inference, does not create runtime pass-rate evidence, and does not make a broad
model-quality claim.

## Report Contract

Later v2 reports should publish one directory per supported dense model:

```text
ci/hardware/apple-m4-mac-mini/<date>/slm-eval-v2/<model-id>/summary.json
```

The live `answer-corpus` run must pass the matching supported dense model ID so
the aggregate receipt is pinned to the model catalog instead of inheriting the
default model block from the shared corpus YAML:

```bash
target/release/bitnet --device apple-m4-cpu-neon answer-corpus \
  --model <verified-cache-path>/<model-file>.gguf \
  --model-id <model-id> \
  --corpus ci/quality/apple-m4-slm-eval-seeded-corpus-v2.yaml \
  --json-out ci/hardware/apple-m4-mac-mini/<date>/slm-eval-v2/<model-id>/answer-corpus.json \
  --per-prompt-timeout-seconds 240
```

Each report should include:

- model source, file, SHA256, quantization, tokenizer authority, and prompt
  template;
- requested backend, selected backend, runtime API, and `fallback_used=false`;
- total strict score and task-family pass rates;
- failure taxonomy for stop-token, template, format, normalization, and
  answer-content misses;
- generated text and generated token IDs for each case;
- TTFT, input token throughput, output token throughput, decode throughput,
  total wall time, peak memory, and memory drift;
- claim-boundary fields stating that the report is dense SLM only.

Strict scoring still reports exact `failed_rules`. V2 taxonomy is additive and
groups those failures under stable labels so reports can separate failure
families without hiding the strict result:

| Taxonomy | Meaning |
|---|---|
| `raw_special_token_tail` | Raw special-token text such as ChatML/header markers reached the decoded answer. |
| `template_or_stop` | Output suggests prompt-template or stop-token handling leaked into the answer. |
| `fenced_json` | A JSON-scored answer was wrapped in a Markdown code fence. |
| `punctuation_casing_normalization` | Strict exact-match failed, but normalized punctuation/case/spacing would match. |
| `format_only` | The answer shape failed, such as JSON parse/schema/type or missing numeric form. |
| `answer_content` | The answer content missed the expected value, label, keyword, forbidden token, enum, or numeric tolerance. |

Per-case receipts expose `quality.failure_taxonomy` and
`quality.scoring.failure_taxonomy`; aggregate receipts expose
`scoring_summary.failure_taxonomy` counts.

## Published M4 Reports

`M4-SLM-EVAL2-003` publishes 2026-05-14 reports for every supported dense M4
model ID:

```text
ci/hardware/apple-m4-mac-mini/2026-05-14/slm-eval-v2/<model-id>/summary.json
```

The runs use `apple-m4-cpu-neon`, `fallback_used=false`, the catalog-pinned
GGUF SHA256 for each model, strict GGUF tokenizer authority, the v2 seed
`777331`, and 120 deterministic cases. The answer-corpus quality path strips
Qwen's `<|im_end|>` as a known stop marker before strict scoring, matching the
existing warm-session answer normalization. The raw generated text and generated
token IDs remain recorded in the case receipts and compact summary case results.

| Model | Strict score | Quality gate | TTFT p50 | TTFT p90 | Input tok/s p50 | Output tok/s p50 | Decode tok/s p50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen2.5-0.5b-instruct-q8_0` | 62 / 120 | 62 / 120 | 3857.5 ms | 4975.9 ms | 12.164 | 1.052 | 9.064 |
| `qwen2.5-0.5b-instruct-q4_k_m` | 66 / 120 | 66 / 120 | 3793.0 ms | 4944.4 ms | 12.412 | 1.097 | 9.052 |
| `qwen2.5-1.5b-instruct-q4_k_m` | 59 / 120 | 59 / 120 | 13771.5 ms | 18186.3 ms | 3.314 | 0.290 | 3.117 |

Task-family strict pass rates:

| Family | Qwen 0.5B Q8_0 | Qwen 0.5B Q4_K_M | Qwen 1.5B Q4_K_M |
|---|---:|---:|---:|
| `arithmetic_exact` | 19 / 20 | 19 / 20 | 20 / 20 |
| `numeric_tolerance` | 0 / 10 | 0 / 10 | 0 / 10 |
| `fixed_table_qa` | 2 / 12 | 0 / 12 | 2 / 12 |
| `format_constrained_json` | 0 / 10 | 0 / 10 | 0 / 10 |
| `closed_label_classification` | 2 / 12 | 6 / 12 | 0 / 12 |
| `synthetic_extraction` | 12 / 12 | 12 / 12 | 8 / 12 |
| `ordering_sorting` | 0 / 12 | 0 / 12 | 0 / 12 |
| `copy_edit_rewrite` | 8 / 12 | 9 / 12 | 11 / 12 |
| `constrained_summary` | 9 / 10 | 10 / 10 | 9 / 10 |
| `instruction_following_required_forbidden` | 10 / 10 | 10 / 10 | 9 / 10 |

The remaining failures are real reportable gaps, not hidden by the report
schema. Current v2 failure taxonomy is dominated by `answer_content`, with
`format_only` and `fenced_json` misses for JSON/numeric cases and
`punctuation_casing_normalization` misses where strict exact scoring still
rejects the output. The published reports therefore support bounded regression
tracking and targeted repair work; they do not prove broad dense-model quality.

## Benchmark Contract

The v2 benchmark profile set should include:

```text
short_prompt_16_out
short_prompt_64_out
long_prompt_16_out
long_prompt_128_out
context_1k
context_4k
resident_25
resident_50
```

Reports should summarize p50, p90, and p99 for:

```text
cold_load_ms
tokenizer_load_ms
prompt_tokenize_ms
prefill_ms
time_to_first_token_ms
input_tokens_per_second
output_tokens_per_second
decode_tokens_per_second
total_wall_ms
peak_memory_mb
memory_drift_mb
```

The operator command is:

```bash
target/release/bitnet mac benchmark \
  --model-id <model-id> \
  --device apple-m4-cpu-neon \
  --profile short_prompt_16_out \
  --profile short_prompt_64_out \
  --profile long_prompt_16_out \
  --profile long_prompt_128_out \
  --profile context_1k \
  --profile context_4k \
  --profile resident_25 \
  --profile resident_50 \
  --json-out ci/hardware/apple-m4-mac-mini/<date>/slm-benchmark-v2/<model-id>/summary.json
```

The receipt kind is `apple_m4_slm_benchmark_v2`. Each profile is one
resident warm-session run with model/tokenizer reuse visible inside that
profile. The memory drift field is based on `getrusage.ru_maxrss`, so it is a
process peak delta and not a live RSS measurement.

## Published M4 Benchmark Reports

`M4-SLM-EVAL2-004` publishes 2026-05-15 benchmark reports for every supported
dense M4 model ID:

```text
ci/hardware/apple-m4-mac-mini/2026-05-15/slm-benchmark-v2/<model-id>/summary.json
```

The runs use `apple-m4-cpu-neon`, `fallback_used=false`, the catalog-pinned
model identity for each dense model, and the full benchmark profile set listed
above. Each summary validated with `bitnet mac receipts-check` as
`apple_m4_slm_benchmark_v2` and records 101 prompts. These receipts are a
recorded M4 Mac mini benchmark envelope for the supported dense model IDs, not a
broad Apple Silicon benchmark.

| Model | Prompts | Generated | TTFT p50 | TTFT p90 | TTFT p99 | Input tok/s p50 | Output tok/s p50 | Decode tok/s p50 | Peak MB p99 | Memory drift MB p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen2.5-0.5b-instruct-q8_0` | 101 | 1522 | 3770.0 ms | 5564.0 ms | 465267.0 ms | 12.329 | 1.009 | 8.884 | 4150.891 | 3997.438 |
| `qwen2.5-0.5b-instruct-q4_k_m` | 101 | 1615 | 3777.0 ms | 6300.0 ms | 470645.0 ms | 12.311 | 1.942 | 8.860 | 4158.281 | 3998.719 |
| `qwen2.5-1.5b-instruct-q4_k_m` | 101 | 1458 | 14203.0 ms | 18334.0 ms | 1433677.0 ms | 3.384 | 0.260 | 2.805 | 8673.922 | 7559.922 |

The overall p99 TTFT is dominated by the `context_4k` profile. The profile
receipts make that tail explicit:

| Model | Profile | Prompts | Generated | Prompt tokens p50 | TTFT p50 | TTFT p99 | Input tok/s p50 | Decode tok/s p50 | Peak MB p99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen2.5-0.5b-instruct-q8_0` | `context_4k` | 3 | 45 | 4075.0 | 465267.0 ms | 470174.0 ms | 8.762 | 5.537 | 4047.078 |
| `qwen2.5-0.5b-instruct-q8_0` | `resident_50` | 50 | 430 | 45.0 | 3764.0 ms | 4926.0 ms | 12.314 | 8.867 | 4150.891 |
| `qwen2.5-0.5b-instruct-q4_k_m` | `context_4k` | 3 | 36 | 4075.0 | 470645.0 ms | 471484.0 ms | 8.664 | 5.517 | 4045.281 |
| `qwen2.5-0.5b-instruct-q4_k_m` | `resident_50` | 50 | 473 | 45.0 | 3772.0 ms | 3986.0 ms | 12.313 | 8.863 | 4158.281 |
| `qwen2.5-1.5b-instruct-q4_k_m` | `context_4k` | 3 | 33 | 4075.0 | 1433677.0 ms | 1458291.0 ms | 2.844 | 2.002 | 8673.922 |
| `qwen2.5-1.5b-instruct-q4_k_m` | `resident_50` | 50 | 403 | 45.0 | 13812.0 ms | 15500.0 ms | 3.388 | 2.766 | 8673.922 |

## Regression Dashboard

`M4-SLM-EVAL2-005` wires v2 reports into the receipt-only regression path:

```bash
target/release/bitnet mac regression \
  ci/hardware/apple-m4-mac-mini/<current-date>/slm-eval-v2/<model-id>/summary.json \
  --baseline ci/hardware/apple-m4-mac-mini/<baseline-date>/slm-eval-v2/<model-id>/summary.json \
  --json

target/release/bitnet mac regression \
  ci/hardware/apple-m4-mac-mini/<current-date>/slm-benchmark-v2/<model-id>/summary.json \
  --baseline ci/hardware/apple-m4-mac-mini/<baseline-date>/slm-benchmark-v2/<model-id>/summary.json \
  --json
```

The comparison is advisory by default. `--fail-on-drift` turns advisory
warnings into a non-zero exit for release or nightly gates. Matching requires
the same M4 machine, model identity, tokenizer authority, prompt template or
benchmark profile set, backend, fallback status, and claim-boundary flags. A
different model, tokenizer, corpus, backend, profile set, or claim boundary is a
new baseline rather than a regression comparison.

The v2 eval comparison watches:

- strict seeded-corpus totals and scoring-summary passed count;
- task-family `cases_passed`, `pass_rate`, `quality_gate_cases_passed`, and
  `quality_gate_pass_rate`;
- input, output, and decode throughput;
- cold load, tokenizer load, prompt tokenization, prefill, TTFT, sampling, and
  total wall time;
- peak memory.

The v2 benchmark comparison watches p50, p90, and p99 for:

- cold load, tokenizer load, prompt tokenization, prefill, TTFT, total wall
  time, and per-profile sampling overhead;
- input, output, and decode throughput;
- peak memory and process-peak memory drift.

Generic PR CI remains model-free through:

```text
.github/workflows/apple-m4-slm-eval-tier0.yml
```

Tier 0 validates parser/scoring tests, v1 and v2 corpus dry-runs, committed v1
and v2 summary receipt schemas, and self-baseline v2 regression comparison
coverage under `--fail-on-drift`. It does not fetch models, run live M4
inference, produce new timing evidence, or publish quality/performance claims.

## Claim Boundary

This lane may claim only bounded, recorded dense SLM evidence for the M4 Mac
mini. It must not claim BitNet quality, full `apple-m4-metal` inference, QK256,
Neural Engine execution, MPSGraph inference, MacBook behavior, broad Apple
Silicon performance, or broad model quality.
