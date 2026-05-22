# A770-027 Readiness Divergence Logit Context

Date: 2026-05-22

Campaign: `intel-a770`

Source work items: `A770-025`, `A770-026`

Source receipts:

- `ci/hardware/amd-5700x-intel-a770/2026-05-22/a770-answer-readiness-divergence-logits/cpu-avx2-readiness-divergent-cases.json`
- `ci/hardware/amd-5700x-intel-a770/2026-05-22/a770-answer-readiness-divergence-logits/a770-opencl-readiness-divergent-cases.json`
- `ci/hardware/amd-5700x-intel-a770/2026-05-22/a770-answer-readiness-divergence-logits/cpu-avx2-vs-a770-readiness-divergent-cases-parity.json`

## Claim Boundary

This report is diagnostic-only. It does not change runtime math and does not
promote strict A770 answer readiness, broad A770 answer quality, CPU/A770 answer
parity, reference parity, selected attention residency, resident KV, full
residency, performance speedup, trusted-partial acceleration, or full BitNet
inference.

## Focused Receipt Scope

A770-026 classified the seeded readiness run as two separate frontiers:

- shared CPU/A770 answer-quality failures, not A770-only backend failures;
- CPU/A770 generated-output parity divergence with missing first-mismatch logit
  context for two cases.

A770-027 reran only those two generated-output divergent cases with
`--dump-logit-steps 24` and `--logits-topk 20` on both CPU AVX2 and Intel A770
OpenCL.

| Lane | Selected backend | Runtime API | Cases | Fallback |
| --- | --- | --- | ---: | --- |
| AMD 5700X CPU AVX2 | `cpu-rust` | `cpu` | 2 | `false` |
| Intel A770 OpenCL | `intel-a770-opencl` | `opencl` | 2 | `false` |

## Parity Summary

The focused parity receipt still fails, as expected, because both selected cases
remain generated-output divergent:

| Field | Value |
| --- | --- |
| `summary.passed` | 0 / 2 |
| `generated_output_frontier.classification` | `generated_output_frontier_first_mismatch_has_logit_context` |
| `generated_output_logit_margin_frontier.classification` | `generated_output_logit_margin_frontier_opposite_argmax_right_near_tie` |
| `generated_output_mismatch_count` | 2 / 2 |
| `mismatch_with_logit_context_count` | 2 / 2 |
| `missing_logit_context_count` | 0 / 2 |
| `opposite_argmax_count` | 2 / 2 |
| `right_near_tie_count` | 1 / 2 |

## First-Mismatch Rows

| Case | First mismatch | CPU chosen | A770 chosen | Classification | Key margin |
| --- | ---: | ---: | ---: | --- | --- |
| `a770_summary_seed770024_keywords_014` | 9 | 40599 | 27252 | `generated_output_logit_margin_first_mismatch_opposite_argmax` | CPU margin over A770-chosen on CPU: 0.009021759033199572; A770 margin over CPU-chosen on A770: 0.16872406005859375 |
| `a770_yes_no_seed770024_water_015` | 2 | 452 | 662 | `generated_output_logit_margin_first_mismatch_opposite_argmax_right_near_tie` | CPU margin over A770-chosen on CPU: 0.4544386863708496; A770 margin over CPU-chosen on A770: 0.004253387451171875 |

## Generated Answers

| Case | CPU AVX2 answer | Intel A770 OpenCL answer | Quality relation |
| --- | --- | --- | --- |
| `a770_summary_seed770024_keywords_014` | `Rust enables fast and safe software development by eliminating null pointer exceptions and memory leaks.` | `Rust enables fast and safe software development by preventing common programming errors.` | Both pass the required-keyword scoring gate. |
| `a770_yes_no_seed770024_water_015` | `No. N/A` | `No. .` | Both remain quality failures under the current readiness scoring contract. |

## Interpretation

A770-027 closes the missing-context gap from A770-026. The remaining two
generated-output divergences are no longer opaque first-mismatch rows: both have
cross-chosen logits, both are opposite-argmax decisions at the first differing
generated token, and the `yes_no_water` A770-side choice is a near tie under the
current `0.01` threshold.

This is still not a runtime fix receipt. It narrows the next CPU/A770 parity work
to first-mismatch logit source attribution for the two focused cases, especially
the summary case where the A770-side margin is not a near tie.

## Next Frontier

Keep the readiness work split:

1. Answer-quality readiness: decide whether shared CPU/A770 scoring failures
   should remain strict exact-match failures or be handled by explicit
   normalized-label and one-word-answer scoring rules.
2. CPU/A770 parity: localize the first-mismatch logit source for the two focused
   generated-output divergent cases before changing runtime math.

Neither path justifies a runtime fix from this report alone.
