<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Apple M4 BitNet eval and benchmark Campaign Status

- Campaign: `apple-m4-bitnet-eval-and-benchmark`
- State: `active`
- Objective: Move Apple M4 BitNet from bounded one-shot and fixed-warm proof into repeatable eval and benchmark reporting with accepted artifact/tokenizer authority, reference-vs-Rust comparison fields, one-shot and warm timing envelopes, and regression dashboards without enabling chat, serve, Metal, QK256, Neural Engine, MPSGraph, MacBook, or broad Apple Silicon claims.

## Work Items

| Item | State | PR | Branch | Review | Merge | Human gate | Acceptance |
|---|---|---:|---|---|---|---|---|
| M4-BITNET-EVAL-001 | in_progress | TBD | `codex/apple-m4-bitnet-eval-and-benchmark/M4-BITNET-EVAL-001-campaign-corpus` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Define the apple-m4-bitnet-eval-and-benchmark campaign and add a 100-case seeded deterministic BitNet eval corpus that dry-runs through answer-corpus parsing/scoring with accepted artifact/tokenizer metadata, no live model run, no runtime BitNet accuracy claim, and no performance claim. |
| M4-BITNET-EVAL-002 | proposed | TBD | `codex/apple-m4-bitnet-eval-and-benchmark/M4-BITNET-EVAL-002-report-schema` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add BitNet eval/report schema support with explicit reference-vs-Rust comparison fields, accepted model/tokenizer/prompt authority, task-family pass rates, timeout/failure taxonomy, generated text/token IDs, backend identity, fallback=false, and no chat/serve enablement. |
| M4-BITNET-EVAL-003 | proposed | TBD | `codex/apple-m4-bitnet-eval-and-benchmark/M4-BITNET-EVAL-003-m4-eval-reports` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Run the BitNet seeded corpus on the M4 Mac mini through the accepted I2_S GGUF, external tokenizer, bitnetcpp-answer prompt authority, and apple-m4-cpu-neon backend, then publish eval reports with generated text/token IDs, valid UTF-8, task-family pass rates, timeout/failure taxonomy, reference-vs-Rust comparison fields, and fallback_used=false. |
| M4-BITNET-EVAL-004 | proposed | TBD | `codex/apple-m4-bitnet-eval-and-benchmark/M4-BITNET-EVAL-004-benchmark-reports` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Benchmark Apple M4 BitNet one-shot and fixed-warm paths with accepted artifact/tokenizer authority, recording model load, tokenizer load, prompt tokenize, prefill, TTFT, input tok/s, output tok/s, decode tok/s, total wall, peak memory, timeout boundaries, and p50/p90/p99 summaries without enabling chat or serve. |
| M4-BITNET-EVAL-005 | proposed | TBD | `codex/apple-m4-bitnet-eval-and-benchmark/M4-BITNET-EVAL-005-regression-dashboard` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Wire BitNet eval and benchmark reports into advisory/nightly regression dashboards with thresholds for task-family pass rates, strict scoring totals, TTFT, input/output/decode throughput, total wall time, peak memory, memory drift, timeout boundaries, and exact model/tokenizer/backend identity while keeping generic PR CI model-free. |

## Hard Constraints

- This is an M4 Mac mini BitNet campaign.
- Use only the accepted Microsoft I2_S GGUF paired with external Microsoft tokenizer authority for answer claims.
- Use the bitnetcpp-answer prompt authority for BitNet M4 answer/eval reports.
- Do not use dense Qwen or dense SLM evidence as BitNet quality, performance, or UX evidence.
- Do not reopen completed dense SLM eval/proof, dense SLM v2, Apple M4 local-answer, server, operational, Metal, or productization campaigns unless a new regression proves they are wrong.
- Do not enable BitNet chat or BitNet serve in this campaign.
- Do not claim full apple-m4-metal inference, QK256 support, Neural Engine execution, MPSGraph model inference, MacBook evidence, or broad Apple Silicon performance.
- Do not add live model downloads, long hardware timing runs, variable warm-session soaks, or model binaries to generic required PR CI.
- Never commit model binaries.
