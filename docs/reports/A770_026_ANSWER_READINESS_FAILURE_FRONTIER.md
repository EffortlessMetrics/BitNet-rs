# A770-026 Answer-Readiness Failure Frontier

Date: 2026-05-22

Campaign: `intel-a770`

Source work item: `A770-025`

Source receipts:

- `ci/hardware/amd-5700x-intel-a770/2026-05-22/a770-answer-readiness/cpu-avx2-answer-readiness.json`
- `ci/hardware/amd-5700x-intel-a770/2026-05-22/a770-answer-readiness/a770-opencl-answer-readiness.json`
- `ci/hardware/amd-5700x-intel-a770/2026-05-22/a770-answer-readiness/cpu-avx2-vs-a770-answer-readiness-parity.json`

## Claim Boundary

This report is diagnostic-only. It does not change runtime math and does not
promote strict A770 answer readiness, broad A770 answer quality, CPU/A770 answer
parity, reference parity, selected attention residency, resident KV, full
residency, performance speedup, trusted-partial acceleration, or full BitNet
inference.

## Receipt Summary

The seeded A770 answer-readiness run completed on both lanes with
`fallback_used=false`:

| Lane | Selected backend | Runtime API | Quality |
| --- | --- | --- | --- |
| AMD 5700X CPU AVX2 | `cpu-rust` | `cpu` | 8 passed, 12 failed, 0 timeout |
| Intel A770 OpenCL | `intel-a770-opencl` | `opencl` | 8 passed, 12 failed, 0 timeout |

Both lanes have the same failure-category summary:

| Category | Count |
| --- | ---: |
| `normalization` | 7 |
| `extraction` | 2 |
| `factual_table` | 1 |

The CPU/A770 parity receipt remains divergent:

| Field | Value |
| --- | --- |
| `summary.passed` | 0 / 20 |
| `logits_topk_frontier.classification` | `logits_topk_frontier_generated_output_divergence` |
| `logits_topk_mismatch_count` | 20 / 20 |
| `same_chosen_token_count` | 20 / 20 |
| `same_generated_output_count` | 18 / 20 |
| `generated_output_mismatch_count` | 2 / 20 |
| `generated_output_frontier.classification` | `generated_output_frontier_first_mismatch_missing_logit_context` |
| `generated_output_logit_margin_frontier.classification` | `generated_output_logit_margin_frontier_missing_context` |

## Failure Classification

The quality failures are shared by CPU AVX2 and Intel A770 OpenCL. That makes
them a prompt/model/scoring readiness frontier, not A770-only backend evidence.

| Class | Cases | Interpretation |
| --- | ---: | --- |
| Mechanical normalization mismatch | 7 | The generated answer contains the expected answer with casing, punctuation, or exact-match normalization differences. |
| Answer-content failure | 5 | The generated answer misses the requested content, ordering, yes/no polarity, or unknown handling. |
| A770-only quality failure | 0 | No quality failure appears only on the A770 receipt. |

## Failed Cases

| Case | Category | Shared answer | Failure class |
| --- | --- | --- | --- |
| `a770_classify_seed770024_negative_011` | `closed_label_classification` | `Negative` | normalization |
| `a770_classify_seed770024_positive_010` | `closed_label_classification` | `Positive` | normalization |
| `a770_context_seed770024_password_019` | `context_conditioned` | `Tulip` | normalization |
| `a770_extract_seed770024_code_009` | `synthetic_extraction` | fenced `R` fragment | extraction |
| `a770_extract_seed770024_owner_008` | `synthetic_extraction` | `Mira` | extraction + normalization |
| `a770_fact_seed770024_capital_018` | `short_factual` | `Paris` | factual_table + normalization |
| `a770_sort_seed770024_words_012` | `ordering_sorting` | `ant, dog, cat` | answer_content |
| `a770_table_seed770024_code_006` | `context_table_lookup` | `Delta` | normalization |
| `a770_table_seed770024_color_005` | `context_table_lookup` | `Blue` | normalization |
| `a770_unknown_seed770024_unknown_020` | `unknown_handling` | `The note "Nova owns the ticket"` | answer_content |
| `a770_yes_no_seed770024_sky_016` | `yes_no` | `No. The daytime` | answer_content |
| `a770_yes_no_seed770024_water_015` | `yes_no` | CPU: `No. N/A`; A770: `No. .` | answer_content + generated-output divergence |

## Passing Cases

The current seeded corpus passes arithmetic, numeric tolerance, copy, required
token, summary-keyword, and JSON-schema gates on both CPU AVX2 and A770 OpenCL.
The JSON case emits fenced JSON text but passes because the scoring contract is
`json_schema`, not exact trimmed text.

## Parity Frontier

The parity frontier is separate from the shared quality frontier:

- Every case has top-k logit mismatch.
- Every compared first step has the same chosen token on CPU and A770.
- 18 of 20 generated outputs match exactly.
- The two generated-output divergences are `a770_summary_seed770024_keywords_014`
  and `a770_yes_no_seed770024_water_015`.
- Both generated-output divergences lack logit context at the first mismatch
  because the A770-025 run dumped only one logit step.

This means the next parity diagnostic should not infer a first-mismatch numeric
root cause from A770-025 alone. It needs more dumped steps for the two divergent
cases, or a focused parity replay that covers the first differing generated
token.

## Next Frontier

The next work should stay split:

1. Answer-quality readiness: decide whether the seeded readiness scorer should
   keep strict exact-match behavior or add explicit normalized-label and
   one-word-answer handling for cases where both CPU and A770 generated the
   expected semantic answer.
2. CPU/A770 parity: add focused multi-step logits for the two A770-025
   generated-output divergent cases before changing runtime math.

Neither path justifies a runtime fix from this report alone.
