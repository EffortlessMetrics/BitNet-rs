# Apple M4 Dense SLM Eval V2

This page defines the second dense SLM eval layer for the M4 Mac mini. It keeps
the existing `apple-m4-slm-eval-and-proof` artifacts as the v1 baseline and adds
a wider v2 path for repeatable quality, benchmark, and regression reporting.

## Corpus Contract

The v2 corpus is:

```text
ci/quality/apple-m4-slm-eval-seeded-corpus-v2.yaml
```

It contains 500 deterministic cases generated from seed `777331` across these
task families:

| Family | Cases | Primary scoring |
|---|---:|---|
| `arithmetic_exact` | 84 | `exact_match` |
| `numeric_tolerance` | 42 | `numeric_tolerance` |
| `fixed_table_qa` | 50 | `exact_match` |
| `format_constrained_json` | 42 | `json_schema` |
| `closed_label_classification` | 50 | `exact_match` |
| `synthetic_extraction` | 50 | `exact_match` |
| `ordering_sorting` | 50 | `normalized_match` |
| `copy_edit_rewrite` | 50 | `required_keywords` |
| `constrained_summary` | 41 | `required_keywords` |
| `instruction_following_required_forbidden` | 41 | `required_forbidden_tokens` |

`M4-SLM-EVAL2-001` validates only the corpus shape and deterministic scoring
metadata through `answer-corpus --dry-run`. It does not run live model
inference, does not create runtime pass-rate evidence, and does not make a broad
model-quality claim.

`M4-ACCURACY-000` freezes the v2 corpus/scorer contract before any larger
corpus expansion. The YAML records `metadata.corpus_contract` with:

```text
contract_version: m4-eval-corpus-scorer-contract-v1
corpus_id: apple-m4-slm-eval-seeded-corpus-v2
corpus_version: 2.2.0
seed: 777331
generator_policy: deterministic-static-fixture-v2
scoring_schema: answer_corpus_mechanical_scoring_v1
receipt_contract: answer_corpus_aggregate_receipt_v1
```

Expected outputs are closed-form fixture answers from the YAML prompt data, not
model outputs or LLM-judge labels. `answer-corpus` aggregate receipts propagate
the contract under `corpus.contract` and `scoring_contract` so later pass rates
can be compared only when corpus, scorer, normalization, and receipt contracts
match.

`M4-ACCURACY-002` expands the static corpus from the earlier 120-case fixture to
500 deterministic cases. This is a corpus/scoring-contract change only; the
historical runtime reports below still describe the 120-case runs that produced
their committed receipts.

`M4-ACCURACY-003` keeps the 500 cases and expected answers unchanged while
tightening deterministic scoring normalization. The answer-corpus scorer now
normalizes known Qwen ChatML stop tails, the leading Qwen assistant separator
observed in resident receipts, fenced or embedded JSON payloads for
`json_schema` scoring, and keyword/forbidden-token boundaries. Generated text
and token IDs remain recorded unchanged; this is a scorer/harness repair, not a
runtime pass-rate refresh.

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
| `fenced_json` | A JSON-scored answer used a Markdown code fence that could not be mechanically extracted and validated. |
| `punctuation_casing_normalization` | Strict exact-match failed, but normalized punctuation/case/spacing would match. |
| `format_only` | The answer shape failed, such as JSON parse/schema/type or missing numeric form. |
| `answer_content` | The answer content missed the expected value, label, keyword, forbidden token, enum, or numeric tolerance. |

Per-case receipts expose `quality.failure_taxonomy` and
`quality.scoring.failure_taxonomy`; aggregate receipts expose
`scoring_summary.failure_taxonomy` counts.

`M4-ACCURACY-005` adds explicit mechanical failure-category fields alongside
those legacy labels. Per-case receipts expose
`quality.failure_category_labels`, `quality.failure_categories`,
`quality.scoring.failure_category_labels`, and
`quality.scoring.failure_categories`; aggregate receipts expose category counts
under `quality_summary.failure_categories`,
`scoring_summary.failure_categories`, task-family summaries, and profile
summaries.

| Category field | Mechanical trigger |
|---|---|
| `formatting` | Raw special-token/template tails, malformed shape, fenced JSON parse issues, non-text output, or other format-only failures. |
| `factual_table` | A failed fixed-table or factual QA family case. |
| `extraction` | A failed synthetic extraction family case. |
| `refusal` | A failed answer containing deterministic refusal phrases such as `I cannot`, `can't answer`, or `unable to answer`. |
| `timeout` | A timed-out child run or timeout failure rule. |
| `schema` | A JSON/schema scoring failure, including parse/type/required/additional/const/enum rules. |
| `normalization` | Strict exact-match failed while normalized punctuation/case/spacing would match. |

These fields are deterministic triage signals only. They do not replace
`failed_rules`, do not judge broad semantic quality, and do not make dense SLM
evidence apply to BitNet.

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

## Second Matching Refresh

`M4-EXCELLENCE-001` records a second dense SLM eval-v2 refresh for the same
supported M4 dense model identities:

```text
ci/hardware/apple-m4-mac-mini/2026-05-16T0240Z/slm-eval-v2/<model-id>/summary.json
```

The refresh keeps the same 120-case seeded corpus, `apple-m4-cpu-neon`
backend, `fallback_used=false`, catalog-pinned GGUF SHA256 values, strict GGUF
tokenizer authority, Qwen2.5 prompt template, and dense-SLM-only claim
boundary. It adds matching-history receipts for the eval-v2 family; dashboard
status is refreshed separately by `M4-EXCELLENCE-003`.

| Model | Strict score | Quality gate | TTFT p50 | TTFT p90 | Input tok/s p50 | Output tok/s p50 | Decode tok/s p50 | Regression note |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `qwen2.5-0.5b-instruct-q8_0` | 62 / 120 | 62 / 120 | 3747.0 ms | 4656.0 ms | 13.101 | 1.803 | 9.067 | matched, no warnings |
| `qwen2.5-0.5b-instruct-q4_k_m` | 66 / 120 | 66 / 120 | 2916.0 ms | 11424.0 ms | 18.445 | 1.803 | 11.912 | advisory `ttft_ms_p90` warning |
| `qwen2.5-1.5b-instruct-q4_k_m` | 59 / 120 | 59 / 120 | 14027.0 ms | 38583.0 ms | 3.368 | 0.311 | 3.149 | advisory `ttft_ms_p90` warning |

The strict scores and task-family pass counts match the 2026-05-14 eval-v2
baseline. The timing warnings are preserved as advisory regression evidence,
not hidden or converted into a speed claim.

## Task-Family Pass-Rate Publication

`M4-ACCURACY-004` publishes a machine-readable task-family pass-rate rollup for
the committed 2026-05-16T0240Z dense SLM eval-v2 refresh:

```text
ci/hardware/apple-m4-mac-mini/2026-05-16T0240Z/slm-eval-v2/task-family-pass-rates.json
```

The rollup is derived from the three sibling `summary.json` receipts. It
preserves each supported model identity, GGUF SHA256, tokenizer authority,
requested and selected backend, runtime API, prompt template, `fallback_used`,
source answer-corpus receipt, scoring summary, quality summary, task-family
counts, and claim boundary. It is not a fresh runtime run, not a 500-case
pass-rate refresh, and not a broad dense-model benchmark.

| Model | Source cases | Backend | Fallback | Tokenizer authority | Prompt template |
|---|---:|---|---|---|---|
| `qwen2.5-0.5b-instruct-q8_0` | 120 | `apple-m4-cpu-neon` | `false` | `gguf_metadata` / `qwen2` | `qwen2.5` |
| `qwen2.5-0.5b-instruct-q4_k_m` | 120 | `apple-m4-cpu-neon` | `false` | `gguf_metadata` / `qwen2` | `qwen2.5` |
| `qwen2.5-1.5b-instruct-q4_k_m` | 120 | `apple-m4-cpu-neon` | `false` | `gguf_metadata` / `qwen2` | `qwen2.5` |

Task-family strict pass rates from that rollup:

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

The 500-case static corpus created by `M4-ACCURACY-002` and the scorer repairs
from `M4-ACCURACY-003` remain separate from this rollup. A fresh 500-case
runtime pass-rate refresh must be recorded as its own evidence item before the
larger corpus has runtime pass rates.

## Full 500-Case Runtime Refresh

`M4-ACCURACY-006` records the first full 500-case dense SLM eval-v2 runtime
refresh for every supported M4 dense model identity:

```text
ci/hardware/apple-m4-mac-mini/2026-05-16T1711Z/slm-eval-v2/<model-id>/answer-corpus.json
ci/hardware/apple-m4-mac-mini/2026-05-16T1711Z/slm-eval-v2/<model-id>/summary.json
ci/hardware/apple-m4-mac-mini/2026-05-16T1711Z/slm-eval-v2/task-family-pass-rates.json
```

The run keeps the bounded eval context unchanged: `apple-m4-cpu-neon`,
`fallback_used=false`, supported Qwen dense model IDs only, strict GGUF
tokenizer authority, Qwen2.5 prompt template, deterministic corpus seed
`777331`, corpus contract `2.2.0`, generated text, generated token IDs, and the
dense-SLM-only claim boundary. This is not BitNet evidence, not a broad
quality claim, and not a broad performance benchmark. The summary memory and
resident-stability fields continue to point at the prior matching dense
warm-session proof because `answer-corpus` child receipts do not record fresh
process RSS.

| Model | Strict score | Quality gate | TTFT p50 | TTFT p90 | Input tok/s p50 | Output tok/s p50 | Decode tok/s p50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen2.5-0.5b-instruct-q8_0` | 299 / 500 | 299 / 500 | 4214.0 ms | 11488.1 ms | 12.298 | 1.095 | 8.957 |
| `qwen2.5-0.5b-instruct-q4_k_m` | 297 / 500 | 297 / 500 | 2202.0 ms | 2793.8 ms | 21.988 | 2.237 | 15.626 |
| `qwen2.5-1.5b-instruct-q4_k_m` | 246 / 500 | 245 / 500 | 8724.0 ms | 11114.7 ms | 5.525 | 0.571 | 4.971 |

Task-family strict pass rates from the 500-case rollup:

| Family | Qwen 0.5B Q8_0 | Qwen 0.5B Q4_K_M | Qwen 1.5B Q4_K_M |
|---|---:|---:|---:|
| `arithmetic_exact` | 71 / 84 | 71 / 84 | 84 / 84 |
| `numeric_tolerance` | 0 / 42 | 0 / 42 | 1 / 42 |
| `fixed_table_qa` | 14 / 50 | 4 / 50 | 15 / 50 |
| `format_constrained_json` | 24 / 42 | 20 / 42 | 2 / 42 |
| `closed_label_classification` | 16 / 50 | 28 / 50 | 0 / 50 |
| `synthetic_extraction` | 49 / 50 | 49 / 50 | 36 / 50 |
| `ordering_sorting` | 13 / 50 | 13 / 50 | 0 / 50 |
| `copy_edit_rewrite` | 40 / 50 | 40 / 50 | 30 / 50 |
| `constrained_summary` | 37 / 41 | 37 / 41 | 41 / 41 |
| `instruction_following_required_forbidden` | 35 / 41 | 35 / 41 | 37 / 41 |

`M4-ACCURACY-007` records the second matching full 500-case dense SLM eval-v2
runtime refresh:

```text
ci/hardware/apple-m4-mac-mini/2026-05-17T0045Z/slm-eval-v2/<model-id>/answer-corpus.json
ci/hardware/apple-m4-mac-mini/2026-05-17T0045Z/slm-eval-v2/<model-id>/summary.json
ci/hardware/apple-m4-mac-mini/2026-05-17T0045Z/slm-eval-v2/task-family-pass-rates.json
ci/hardware/apple-m4-mac-mini/2026-05-17T0045Z/report-refresh/report-refresh-manifest.json
ci/hardware/apple-m4-mac-mini/2026-05-17T0045Z/regression-dashboard/regression-dashboard.json
```

The repeated run keeps the same bounded identity: supported Qwen dense model
IDs, `apple-m4-cpu-neon`, `fallback_used=false`, GGUF tokenizer authority,
Qwen2.5 prompt template, deterministic corpus seed `777331`, and corpus
contract `2.2.0`. Each model summary passes `bitnet mac receipts-check`; each
model regression against the first 500-case refresh reports
`matched_context=true` with zero warnings. The regression dashboard now reports
three ready dense SLM eval-v2 groups for the recorded identities.

| Model | Strict score | Quality gate | TTFT p50 | TTFT p90 | Input tok/s p50 | Output tok/s p50 | Decode tok/s p50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen2.5-0.5b-instruct-q8_0` | 299 / 500 | 299 / 500 | 2203 ms | 2771 ms | 22.010 | 2.206 | 15.628 |
| `qwen2.5-0.5b-instruct-q4_k_m` | 297 / 500 | 297 / 500 | 2201 ms | 2784 ms | 21.989 | 2.243 | 15.630 |
| `qwen2.5-1.5b-instruct-q4_k_m` | 246 / 500 | 245 / 500 | 8809 ms | 11336 ms | 5.505 | 0.574 | 4.949 |

Task-family strict pass rates from the repeated 500-case rollup:

| Family | Qwen 0.5B Q8_0 | Qwen 0.5B Q4_K_M | Qwen 1.5B Q4_K_M |
|---|---:|---:|---:|
| `arithmetic_exact` | 71 / 84 | 71 / 84 | 84 / 84 |
| `numeric_tolerance` | 0 / 42 | 0 / 42 | 1 / 42 |
| `fixed_table_qa` | 14 / 50 | 4 / 50 | 15 / 50 |
| `format_constrained_json` | 24 / 42 | 20 / 42 | 2 / 42 |
| `closed_label_classification` | 16 / 50 | 28 / 50 | 0 / 50 |
| `synthetic_extraction` | 49 / 50 | 49 / 50 | 36 / 50 |
| `ordering_sorting` | 13 / 50 | 13 / 50 | 0 / 50 |
| `copy_edit_rewrite` | 40 / 50 | 40 / 50 | 30 / 50 |
| `constrained_summary` | 37 / 41 | 37 / 41 | 41 / 41 |
| `instruction_following_required_forbidden` | 35 / 41 | 35 / 41 | 37 / 41 |

This moves the 500-case dense SLM eval-v2 groups from insufficient history to
comparable matching history for the recorded identities. It does not prove
BitNet behavior, broad dense model quality, or broad M4 performance.

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
resident_100
```

Reports should summarize p50, p90, and p99 for:

```text
cold_load_ms
tokenizer_load_ms
prompt_tokenize_ms
prefill_ms
time_to_first_token_ms
sampling_ms_per_token
input_tokens_per_second
output_tokens_per_second
decode_tokens_per_second
total_wall_ms
peak_memory_mb
memory_drift_mb
```

The operator command is:

```bash
bitnet mac benchmark --calibrate \
  --json-out ci/hardware/apple-m4-mac-mini/<date>/benchmark/calibration.json

target/release/bitnet --device apple-m4-cpu-neon mac benchmark \
  --model-id <model-id> \
  --profile short_prompt_16_out \
  --profile short_prompt_64_out \
  --profile long_prompt_16_out \
  --profile long_prompt_128_out \
  --profile context_1k \
  --profile context_4k \
  --profile resident_25 \
  --profile resident_50 \
  --profile resident_100 \
  --json-out ci/hardware/apple-m4-mac-mini/<date>/slm-benchmark-v2/<model-id>/summary.json
```

The current published `M4-BENCH-002` full-profile summaries are:

| Model | Receipt | Prompt runs | Generated tokens | Receipt check |
|---|---|---:|---:|---|
| `qwen2.5-0.5b-instruct-q8_0` | `ci/hardware/apple-m4-mac-mini/2026-05-15T1845Z/slm-benchmark-v2/qwen2.5-0.5b-instruct-q8_0/summary.json` | 201 | 2382 | pass |
| `qwen2.5-0.5b-instruct-q4_k_m` | `ci/hardware/apple-m4-mac-mini/2026-05-15T1845Z/slm-benchmark-v2/qwen2.5-0.5b-instruct-q4_k_m/summary.json` | 201 | 2543 | pass |
| `qwen2.5-1.5b-instruct-q4_k_m` | `ci/hardware/apple-m4-mac-mini/2026-05-15T1845Z/slm-benchmark-v2/qwen2.5-1.5b-instruct-q4_k_m/summary.json` | 201 | 2262 | pass |

Each summary covers `short_prompt_16_out`, `short_prompt_64_out`,
`long_prompt_16_out`, `long_prompt_128_out`, `context_1k`, `context_4k`,
`resident_25`, `resident_50`, and `resident_100`. The reports are comparable
only inside their recorded model, tokenizer, backend, runtime API, fallback
state, profile set, and M4 Mac mini identity.

The calibration receipt kind is `apple_m4_benchmark_calibration`; it records
synthetic harness timing only and does not load a model or claim model speed.

The live benchmark receipt kind is `apple_m4_slm_benchmark_v2`. Each profile is one
resident warm-session run with model/tokenizer reuse visible inside that
profile. The memory drift field is based on `getrusage.ru_maxrss`, so it is a
process peak delta and not a live RSS measurement.

## Published M4 Benchmark Reports

`M4-SLM-EVAL2-004` publishes 2026-05-15 benchmark reports for every supported
dense M4 model ID:

```text
ci/hardware/apple-m4-mac-mini/2026-05-15/slm-benchmark-v2/<model-id>/summary.json
```

Those reports cover the original v2 profile set through `resident_50`. The
follow-on durable evidence refresh adds `resident_100` to the contract and
publishes a first full nine-profile live refresh under:

```text
ci/hardware/apple-m4-mac-mini/2026-05-15T1845Z/slm-benchmark-v2/<model-id>/summary.json
```

The 2026-05-15 runs use `apple-m4-cpu-neon`, `fallback_used=false`, the
catalog-pinned model identity for each dense model, and the original v2
benchmark profile set. Each summary validated with `bitnet mac receipts-check`
as `apple_m4_slm_benchmark_v2` and records 101 prompts. These receipts are a
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

The 2026-05-15T1845Z durable refresh keeps the same dense model IDs, backend,
fallback status, and claim boundary, but it changes the benchmark profile set by
adding `resident_100`. Each summary validated with `bitnet mac receipts-check`
as `apple_m4_slm_benchmark_v2` and records 201 prompts. Direct strict
`bitnet mac regression` against the earlier 2026-05-15 summaries stops with
`profiles_required mismatch`; that is expected because the previous baseline did
not include `resident_100`.

| Model | Prompts | Generated | TTFT p50 | TTFT p99 | Input tok/s p50 | Output tok/s p50 | Decode tok/s p50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen2.5-0.5b-instruct-q8_0` | 201 | 2382 | 2150.0 ms | 262573.0 ms | 21.701 | 1.708 | 15.652 |
| `qwen2.5-0.5b-instruct-q4_k_m` | 201 | 2543 | 2150.0 ms | 262456.0 ms | 21.698 | 3.079 | 15.653 |
| `qwen2.5-1.5b-instruct-q4_k_m` | 201 | 2262 | 8184.0 ms | 822688.0 ms | 5.773 | 0.357 | 4.808 |

The refreshed profile receipts make the new long-context and `resident_100`
boundaries explicit:

| Model | Profile | Prompts | Generated | TTFT p50 | TTFT p99 | Input tok/s p50 | Decode tok/s p50 | Peak MB p50 | Memory drift MB p50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen2.5-0.5b-instruct-q8_0` | `context_4k` | 3 | 45 | 262608.0 ms | 262615.0 ms | 15.526 | 9.958 | 4051.297 | 0.000 |
| `qwen2.5-0.5b-instruct-q8_0` | `resident_100` | 100 | 860 | 2150.0 ms | 2246.0 ms | 21.698 | 15.650 | 4156.750 | 1.875 |
| `qwen2.5-0.5b-instruct-q4_k_m` | `context_4k` | 3 | 36 | 262519.0 ms | 262698.0 ms | 15.529 | 9.951 | 4053.719 | 0.000 |
| `qwen2.5-0.5b-instruct-q4_k_m` | `resident_100` | 100 | 928 | 2151.0 ms | 2246.0 ms | 21.694 | 15.650 | 4159.609 | 0.968 |
| `qwen2.5-1.5b-instruct-q4_k_m` | `context_4k` | 3 | 33 | 822691.0 ms | 823143.0 ms | 4.954 | 3.572 | 8395.047 | 0.000 |
| `qwen2.5-1.5b-instruct-q4_k_m` | `resident_100` | 100 | 804 | 8078.0 ms | 8966.0 ms | 5.816 | 4.780 | 8395.047 | 0.000 |

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
