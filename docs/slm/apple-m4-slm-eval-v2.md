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

## Claim Boundary

This lane may claim only bounded, recorded dense SLM evidence for the M4 Mac
mini. It must not claim BitNet quality, full `apple-m4-metal` inference, QK256,
Neural Engine execution, MPSGraph inference, MacBook behavior, broad Apple
Silicon performance, or broad model quality.
