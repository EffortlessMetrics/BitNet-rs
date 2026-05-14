<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Apple M4 dense SLM eval and proof Campaign Status

- Campaign: `apple-m4-slm-eval-and-proof`
- State: `active`
- Objective: Turn the usable Apple M4 dense SLM path into a structured, receipt-backed local model runner proof with seeded quality eval, first-class speed/stability metrics, per-model reports, and lightweight regression economics without broad benchmark claims.

## Work Items

| Item | State | PR | Branch | Review | Merge | Human gate | Acceptance |
|---|---|---:|---|---|---|---|---|
| M4-SLM-EVAL-001 | merged | #4656 | `codex/apple-m4-slm-eval-and-proof/M4-SLM-EVAL-001-seeded-corpus` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Define the apple-m4-slm-eval-and-proof campaign and add a seeded, deterministic dense SLM eval corpus spec covering arithmetic, fixed-table QA, JSON-shaped output, closed-label classification, synthetic extraction, ordering/sorting, rewrite, constrained summary, and required/forbidden-token instruction following. Validate the corpus shape with the existing answer-corpus dry-run path, but do not claim runtime accuracy or broad model quality. |
| M4-SLM-EVAL-002 | merged | #4660 | `codex/apple-m4-slm-eval-and-proof/M4-SLM-EVAL-002-scoring` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add deterministic dense SLM eval scoring for exact, normalized, JSON/schema-style, numeric tolerance, required keyword, and forbidden-token checks, with fixture tests that do not invoke live models. |
| M4-SLM-EVAL-003 | merged | #4663 | `codex/apple-m4-slm-eval-and-proof/M4-SLM-EVAL-003-report-schema` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Define and validate the per-model Apple M4 dense SLM eval summary report schema with first-class TTFT, input token throughput, output/decode throughput, total wall time, memory, stability, and claim-boundary fields. |
| M4-SLM-EVAL-004 | merged | #4670 | `codex/apple-m4-slm-eval-and-proof/M4-SLM-EVAL-004-supported-model-reports` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Run and publish per-model Apple M4 dense SLM eval reports for every supported dense model ID under the seeded corpus and report schema, preserving fallback=false and dense-only claim boundaries. |
| M4-SLM-EVAL-005 | merged | #4675 | `codex/apple-m4-slm-eval-and-proof/M4-SLM-EVAL-005-ci-tiers` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Document and wire lightweight Tier 0 parser/schema/report checks for generic PR CI, with advisory/nightly/release M4 tiers defined separately for live model runs. |
| M4-SLM-EVAL-006 | in_progress | TBD | `codex/apple-m4-slm-eval-and-proof/M4-SLM-EVAL-006-regression` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Wire `bitnet mac regression` or an adjacent checker to compare matching dense SLM eval summary reports over time with explicit thresholds and no broad benchmark claims. |

## Hard Constraints

- This is an M4 Mac mini dense SLM campaign.
- Do not reopen completed apple-m4, apple-m4-operational, apple-m4-slm-answer, apple-m4-productization, apple-m4-slm-performance, apple-m4-slm-excellence, apple-m4-slm-hardening, apple-m4-continuity, or apple-m4-dense-slm-regression campaigns.
- Do not use dense Qwen evidence as BitNet local-answer evidence.
- Do not claim broad model quality or broad Apple Silicon benchmark performance.
- Do not claim full apple-m4-metal inference, QK256 support, Neural Engine execution, MPSGraph model inference, or MacBook evidence.
- Do not add live model downloads, hardware timing runs, or long resident soaks to generic required PR CI.
- Never commit model binaries.
