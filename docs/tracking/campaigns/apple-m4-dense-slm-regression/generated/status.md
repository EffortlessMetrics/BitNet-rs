<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Apple M4 dense SLM regression guardrails Campaign Status

- Campaign: `apple-m4-dense-slm-regression`
- State: `active`
- Objective: Turn the measured Apple M4 dense SLM local-answer envelope into repeatable regression guardrails for quality, determinism, receipt schema, model-cache identity, warm-session timing, and memory drift without broadening the claim beyond the recorded Qwen2.5 dense SLM Apple CPU/NEON path.

## Work Items

| Item | State | PR | Branch | Review | Merge | Human gate | Acceptance |
|---|---|---:|---|---|---|---|---|
| M4-SLM-REG-001 | merged | #4163 | `codex/apple-m4-dense-slm-regression/M4-SLM-REG-001-receipt-diff` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add an advisory Apple M4 dense SLM receipt comparison surface for the published performance envelope, validating context identity and reporting quality, timing, and memory drift without installing nightly infrastructure or failing broad CI on performance. |
| M4-SLM-REG-002 | merged | #4166 | `codex/apple-m4-dense-slm-regression/M4-SLM-REG-002-quality-drift` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Turn the five-prompt dense SLM quality and determinism corpus into a regression receipt gate that checks valid UTF-8, non-empty output, non-degenerate output, stable greedy token IDs, backend identity, and fallback status. |
| M4-SLM-REG-003 | merged | #4170 | `codex/apple-m4-dense-slm-regression/M4-SLM-REG-003-scheduled-hardware` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Define the scheduled Apple hardware regression workflow shape for dense SLM receipts, including artifact retention, branch/PR reporting, low-disk behavior, and required context fields, without requiring the workflow to be active until runner availability is confirmed. |
| M4-SLM-REG-004 | merged | #4177 | `codex/apple-m4-dense-slm-regression/M4-SLM-REG-004-trend-history` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add a compact dense SLM trend-history artifact format for Apple M4 release receipts, preserving per-run context and drift summaries without committing model binaries or large raw receipt bundles. |
| M4-SLM-REG-005 | proposed | TBD | `codex/apple-m4-dense-slm-regression/M4-SLM-REG-005-threshold-tightening` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Tighten Apple M4 dense SLM performance thresholds only after multiple matching release-mode receipts exist, with separate bands for timing noise, quality failures, memory drift, and backend/fallback mismatches. |

## Hard Constraints

- Do not reopen the completed apple-m4, apple-m4-slm-answer, apple-m4-productization, or apple-m4-slm-performance campaigns.
- Do not weaken the blocked BitNet apple-m4-local-answer gates.
- Do not claim BitNet local-answer quality from dense SLM evidence.
- Do not claim full apple-m4-metal inference from a named Metal phase.
- Do not claim Neural Engine execution, MPSGraph model inference, QK256 support, or broad M4 performance.
- Do not turn a single M4 Mac mini envelope into a fleet-wide Apple Silicon performance guarantee.
- Do not make performance drift a hard CI failure until a scheduled Apple hardware lane produces repeatable receipts.
- Never commit model binaries.
