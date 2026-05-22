# A770-028 Readiness Normalization Scoring

Date: 2026-05-22

Campaign: `intel-a770`

Source work items: `A770-025`, `A770-026`, `A770-027`

Source receipts:

- `ci/hardware/amd-5700x-intel-a770/2026-05-22/a770-answer-readiness-normalized-scoring/cpu-avx2-answer-readiness-normalized-scoring.json`
- `ci/hardware/amd-5700x-intel-a770/2026-05-22/a770-answer-readiness-normalized-scoring/a770-opencl-answer-readiness-normalized-scoring.json`
- `ci/hardware/amd-5700x-intel-a770/2026-05-22/a770-answer-readiness-normalized-scoring/cpu-avx2-vs-a770-answer-readiness-normalized-scoring-parity.json`

## Claim Boundary

This report is scoring-contract evidence only. It does not change runtime math,
QK256 dispatch, OpenCL execution, tokenizer behavior, prompt templates, or model
artifacts. It does not promote strict A770 answer readiness, broad A770 answer
quality, CPU/A770 answer parity, reference parity, selected attention residency,
resident KV, full residency, performance speedup, trusted-partial acceleration,
or full BitNet inference.

## Scoring Contract Change

The A770 seeded answer-readiness corpus moves from `corpus_version = "1.0.0"` to
`corpus_version = "1.0.1"`.

A770-028 changes exactly seven cases that committed CPU and A770 receipts
classified as shared punctuation/casing normalization failures from
`exact_match` to `normalized_match`:

| Case | Expected normalized answer |
| --- | --- |
| `a770_table_seed770024_color_005` | `blue` |
| `a770_table_seed770024_code_006` | `delta` |
| `a770_extract_seed770024_owner_008` | `mira` |
| `a770_classify_seed770024_positive_010` | `positive` |
| `a770_classify_seed770024_negative_011` | `negative` |
| `a770_fact_seed770024_capital_018` | `paris` |
| `a770_context_seed770024_password_019` | `tulip` |

The corpus also contains pre-existing `normalized_match` cases. A770-028 does
not broaden those contracts. The known content-failure cases remain failures
under their existing scoring rules.

## Readiness Receipt Summary

Both lanes now pass 15 of 20 cases under the normalized scoring contract. The
remaining five failures are still `answer_content`, not normalization misses.

| Lane | Selected backend | Runtime API | Selected kernel/runtime | Fallback | Passed | Failed |
| --- | --- | --- | --- | --- | ---: | ---: |
| AMD 5700X CPU AVX2 | `cpu-rust` | `cpu` | `i2_s-avx2-reference` | `false` | 15 | 5 |
| Intel A770 OpenCL | `intel-a770-opencl` | `opencl` | `a770_opencl_qk256_i2s_i8s_scaled_dispatch_candidate` | `false` | 15 | 5 |

The five remaining failing cases are:

| Case | CPU AVX2 answer | Intel A770 OpenCL answer |
| --- | --- | --- |
| `a770_extract_seed770024_code_009` | starts with fenced R package text | starts with fenced R package text |
| `a770_sort_seed770024_words_012` | `ant, dog, cat` | `ant, dog, cat` |
| `a770_yes_no_seed770024_water_015` | `No. N/A` | `No. .` |
| `a770_yes_no_seed770024_sky_016` | `No. The daytime` | `No. The daytime` |
| `a770_unknown_seed770024_unknown_020` | `The note "Nova owns the ticket"` | `The note "Nova owns the ticket"` |

## Parity Receipt Summary

The CPU/A770 parity receipt still fails, as expected. Normalizing the seven
label-like scoring cases improves the quality-gate count but does not prove
CPU/A770 output or logit parity.

| Field | Value |
| --- | --- |
| `summary.passed` | 0 / 20 |
| `logits_topk_frontier.classification` | `logits_topk_frontier_generated_output_divergence` |
| `logits_topk_mismatch_count` | 20 / 20 |
| `same_chosen_token_count` | 20 / 20 |
| `same_generated_output_count` | 18 / 20 |
| `generated_output_mismatch_count` | 2 / 20 |
| `generated_output_frontier.classification` | `generated_output_frontier_first_mismatch_missing_logit_context` |

The full readiness receipts dump one logit step per case. That is enough for the
quality-gate scoring refresh, but it does not replace the focused A770-027
first-mismatch logit-context receipt for the two generated-output divergent
cases.

## Interpretation

A770-028 removes seven false strict-case quality failures from the seeded
readiness corpus. This is a scoring-contract cleanup, not a backend correctness
fix. The live CPU and A770 receipts now agree on the quality count: 15 passed
and 5 failed.

The remaining answer-readiness blocker is content quality, and the remaining
CPU/A770 backend blocker is still parity. The two generated-output divergent
cases remain:

- `a770_summary_seed770024_keywords_014`
- `a770_yes_no_seed770024_water_015`

A770-027 remains the focused receipt for their first-mismatch logit context.

## Next Frontier

Keep the quality and parity tracks separate:

1. Answer-quality readiness: decide whether the remaining content failures need
   prompt/scoring contract changes or model/runtime fixes.
2. CPU/A770 parity: continue first-mismatch logit source attribution for the two
   generated-output divergent cases before changing runtime math.

Neither path justifies a runtime fix or claim promotion from A770-028 alone.
