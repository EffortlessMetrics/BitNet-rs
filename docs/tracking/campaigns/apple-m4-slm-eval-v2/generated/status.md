<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Apple M4 dense SLM eval v2 Campaign Status

- Campaign: `apple-m4-slm-eval-v2`
- State: `active`
- Objective: Move the Apple M4 dense SLM lane from bounded 10-case proof into broader, reproducible quality and benchmark reporting with a 100-500 case deterministic corpus, task-family pass rates, full latency/throughput profiles, and regression dashboards without broad model-quality or Apple Silicon benchmark claims.

## Work Items

| Item | State | PR | Branch | Review | Merge | Human gate | Acceptance |
|---|---|---:|---|---|---|---|---|
| M4-SLM-EVAL2-001 | merged | #4777 | `codex/apple-m4-slm-eval-v2/M4-SLM-EVAL2-001-campaign-corpus` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Define the apple-m4-slm-eval-v2 campaign and add a 120-case seeded deterministic dense SLM eval corpus v2 that dry-runs through answer-corpus parsing/scoring with no live model run and no runtime accuracy or broad quality claim. |
| M4-SLM-EVAL2-002 | merged | #4792 | `codex/apple-m4-slm-eval-v2/M4-SLM-EVAL2-002-failure-taxonomy` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add stop-token/template/scoring failure taxonomy so v2 reports distinguish raw special-token tails, fenced JSON, punctuation/casing/normalization differences, format-only failures, and answer-content failures without hiding strict scoring results. |
| M4-SLM-EVAL2-003 | proposed | TBD | `codex/apple-m4-slm-eval-v2/M4-SLM-EVAL2-003-task-family-reports` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Run the v2 corpus for every supported dense M4 model ID and publish per-model reports with task-family pass rates, strict scoring totals, generated text/token IDs, backend/fallback identity, catalog-pinned aggregate model identity, and dense-SLM-only claim boundaries. |
| M4-SLM-EVAL2-004 | proposed | TBD | `codex/apple-m4-slm-eval-v2/M4-SLM-EVAL2-004-benchmark-profiles` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Refresh dense M4 benchmark profiles for every supported model with cold load, tokenizer load, prompt tokenization, prefill, TTFT, input tok/s, output tok/s, decode tok/s, total wall time, peak memory, memory drift, and p50/p90/p99 summaries. |
| M4-SLM-EVAL2-005 | proposed | TBD | `codex/apple-m4-slm-eval-v2/M4-SLM-EVAL2-005-regression-dashboard` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Wire v2 eval and benchmark reports into advisory/nightly regression dashboards with thresholds for task-family pass rates, strict scoring totals, TTFT, input/output/decode throughput, total wall time, peak memory, and memory drift while keeping generic PR CI model-free. |

## Hard Constraints

- This is an M4 Mac mini dense SLM campaign.
- Do not reopen completed apple-m4, apple-m4-operational, apple-m4-slm-answer, apple-m4-productization, apple-m4-slm-performance, apple-m4-slm-excellence, apple-m4-slm-hardening, apple-m4-continuity, apple-m4-dense-slm-regression, or apple-m4-slm-eval-and-proof campaigns.
- Do not use dense Qwen evidence as BitNet local-answer evidence.
- Do not claim broad model quality or broad Apple Silicon benchmark performance.
- Do not claim full apple-m4-metal inference, QK256 support, Neural Engine execution, MPSGraph model inference, or MacBook evidence.
- Do not add live model downloads, hardware timing runs, or long resident soaks to generic required PR CI.
- Never commit model binaries.
