# Apple M4 Seven-Day Trend History

Generated: `2026-05-22T05:30:00Z`

Work item: `M4-TREND-001`

Source artifacts:

| Kind | Path | SHA256 |
|---|---|---|
| report refresh manifest | `ci/hardware/apple-m4-mac-mini/2026-05-17T0045Z/report-refresh/report-refresh-manifest.json` | `f573eaa03bc02239bc36c0f8a4d5fd5b1b577d9c04d311de935519d20e520ee9` |
| regression dashboard | `ci/hardware/apple-m4-mac-mini/2026-05-17T0045Z/regression-dashboard/regression-dashboard.json` | `84b6b14070527a04911225aeb69896eb86b57a6398f3f38b6cacff8532015119` |
| BitNet repaired eval context delta | `ci/hardware/apple-m4-mac-mini/2026-05-18T1806Z/bitnet-eval-250-repaired/regression-vs-2026-05-17T1903Z.json` | `d4dd4c4ddcddf64c00d55b86f05ae8e38f058e0bf841a5deb1a99907b119f52a` |

## Summary

The committed seven-day window keeps five M4 dashboard families in matching-history
state: dense SLM eval-v2, dense SLM benchmark-v2, BitNet eval, BitNet benchmark,
and BitNet variable warm. The dashboard has nine matching groups and all nine are
`ready` for latest-vs-baseline comparison.

One advisory threshold needs operator attention: BitNet variable warm resident
memory increased from `2140225536` bytes to `2688778240` bytes, a `25.63%`
increase against the existing `10%` higher-is-worse resident-memory threshold.
Quality remained passing and timing improved, so this is an advisory memory
follow-up only. It does not enable BitNet chat or serve.

## Dashboard Groups

| Family | Model | Reports | Latest | Baseline | Threshold outcome | Operator impact |
|---|---|---:|---|---|---|---|
| `dense_slm_eval_v2` | `qwen2.5-0.5b-instruct-q4_k_m` | 4 | `2026-05-17T0045Z/slm-eval-v2/qwen2.5-0.5b-instruct-q4_k_m/summary.json` | `2026-05-16T1711Z/slm-eval-v2/qwen2.5-0.5b-instruct-q4_k_m/summary.json` | within threshold | no class change |
| `dense_slm_eval_v2` | `qwen2.5-0.5b-instruct-q8_0` | 4 | `2026-05-17T0045Z/slm-eval-v2/qwen2.5-0.5b-instruct-q8_0/summary.json` | `2026-05-16T1711Z/slm-eval-v2/qwen2.5-0.5b-instruct-q8_0/summary.json` | within threshold | no class change |
| `dense_slm_eval_v2` | `qwen2.5-1.5b-instruct-q4_k_m` | 4 | `2026-05-17T0045Z/slm-eval-v2/qwen2.5-1.5b-instruct-q4_k_m/summary.json` | `2026-05-16T1711Z/slm-eval-v2/qwen2.5-1.5b-instruct-q4_k_m/summary.json` | within threshold | no class change |
| `dense_slm_benchmark_v2` | `qwen2.5-0.5b-instruct-q4_k_m` | 2 | `2026-05-15T1845Z/slm-benchmark-v2/qwen2.5-0.5b-instruct-q4_k_m/summary.json` | `2026-05-15/slm-benchmark-v2/qwen2.5-0.5b-instruct-q4_k_m/summary.json` | within threshold | no class change |
| `dense_slm_benchmark_v2` | `qwen2.5-0.5b-instruct-q8_0` | 2 | `2026-05-15T1845Z/slm-benchmark-v2/qwen2.5-0.5b-instruct-q8_0/summary.json` | `2026-05-15/slm-benchmark-v2/qwen2.5-0.5b-instruct-q8_0/summary.json` | within threshold | no class change |
| `dense_slm_benchmark_v2` | `qwen2.5-1.5b-instruct-q4_k_m` | 2 | `2026-05-15T1845Z/slm-benchmark-v2/qwen2.5-1.5b-instruct-q4_k_m/summary.json` | `2026-05-15/slm-benchmark-v2/qwen2.5-1.5b-instruct-q4_k_m/summary.json` | within threshold | no class change |
| `bitnet_eval` | `microsoft-bitnet-b1.58-2B-4T-i2s` | 2 | `2026-05-15T2214Z/bitnet-eval/answer-corpus.json` | `2026-05-15/bitnet-eval/answer-corpus.json` | within threshold | BitNet eval only |
| `bitnet_benchmark` | `microsoft-bitnet-b1.58-2B-4T-i2s` | 2 | `2026-05-15T2214Z/bitnet-benchmark/summary.json` | `2026-05-15/bitnet-benchmark/summary.json` | within threshold | BitNet benchmark only |
| `bitnet_variable_warm` | `microsoft-bitnet-b1.58-2B-4T-i2s` | 2 | `2026-05-16T0626Z/bitnet-productization/variable-warm-session.json` | `2026-05-15/bitnet-productization/variable-warm-session.json` | advisory memory warning | no chat or serve enablement |

## Skipped Days

| Date | Status | Reason |
|---|---|---|
| 2026-05-15 | covered | Dense benchmark, BitNet eval, BitNet benchmark, and BitNet variable-warm baseline/latest receipts were retained in matching dashboard groups. |
| 2026-05-16 | covered | Dense SLM eval-v2 and BitNet variable-warm receipts added matching-history refreshes. |
| 2026-05-17 | covered | Dense SLM eval-v2 latest refresh and report-refresh/regression-dashboard summaries were committed. Later BitNet 250-case eval evidence started a separate context. |
| 2026-05-18 | skipped for matching dashboard trend | BitNet 250-case repaired eval was committed as a new context-only baseline with scoring-summary mismatch against the prior 250-case run. |
| 2026-05-19 | skipped for matching dashboard trend | Setup and benchmark-variance receipts were committed for separate operator surfaces, not as replacements for the five matching dashboard families. |
| 2026-05-20 | skipped for matching dashboard trend | Context and reliability-drill receipts were committed for separate route and failure-mode evidence. |
| 2026-05-21 | skipped for matching dashboard trend | Serve failure-semantics evidence was committed for local-server behavior and does not replace dense SLM or BitNet eval/benchmark/warm trend evidence. |

## Claim Boundary

This artifact is a dashboard and receipt summary only. The per-run receipts
remain authoritative. It does not run live inference, download models, replace
per-run receipts, enable BitNet chat, enable BitNet serve, prove full Metal,
claim QK256, claim Neural Engine or MPSGraph inference, claim MacBook behavior,
make broad Apple Silicon quality or performance claims, claim speedup, or prove
future performance.
