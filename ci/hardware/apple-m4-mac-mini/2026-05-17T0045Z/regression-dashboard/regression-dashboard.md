# Apple M4 Inference Regression Dashboard

Model-free dashboard generated from committed Apple M4 receipts only.

| Family | Evidence | Model | Reports | Status | Latest | Baseline |
|---|---|---|---:|---|---|---|
| `dense_slm_eval_v2` | `dense_slm` | `qwen2.5-0.5b-instruct-q4_k_m` | 4 | `ready` | `ci/hardware/apple-m4-mac-mini/2026-05-17T0045Z/slm-eval-v2/qwen2.5-0.5b-instruct-q4_k_m/summary.json` | `ci/hardware/apple-m4-mac-mini/2026-05-16T1711Z/slm-eval-v2/qwen2.5-0.5b-instruct-q4_k_m/summary.json` |
| `dense_slm_eval_v2` | `dense_slm` | `qwen2.5-0.5b-instruct-q8_0` | 4 | `ready` | `ci/hardware/apple-m4-mac-mini/2026-05-17T0045Z/slm-eval-v2/qwen2.5-0.5b-instruct-q8_0/summary.json` | `ci/hardware/apple-m4-mac-mini/2026-05-16T1711Z/slm-eval-v2/qwen2.5-0.5b-instruct-q8_0/summary.json` |
| `dense_slm_eval_v2` | `dense_slm` | `qwen2.5-1.5b-instruct-q4_k_m` | 4 | `ready` | `ci/hardware/apple-m4-mac-mini/2026-05-17T0045Z/slm-eval-v2/qwen2.5-1.5b-instruct-q4_k_m/summary.json` | `ci/hardware/apple-m4-mac-mini/2026-05-16T1711Z/slm-eval-v2/qwen2.5-1.5b-instruct-q4_k_m/summary.json` |
| `dense_slm_benchmark_v2` | `dense_slm` | `qwen2.5-0.5b-instruct-q4_k_m` | 2 | `ready` | `ci/hardware/apple-m4-mac-mini/2026-05-15T1845Z/slm-benchmark-v2/qwen2.5-0.5b-instruct-q4_k_m/summary.json` | `ci/hardware/apple-m4-mac-mini/2026-05-15/slm-benchmark-v2/qwen2.5-0.5b-instruct-q4_k_m/summary.json` |
| `dense_slm_benchmark_v2` | `dense_slm` | `qwen2.5-0.5b-instruct-q8_0` | 2 | `ready` | `ci/hardware/apple-m4-mac-mini/2026-05-15T1845Z/slm-benchmark-v2/qwen2.5-0.5b-instruct-q8_0/summary.json` | `ci/hardware/apple-m4-mac-mini/2026-05-15/slm-benchmark-v2/qwen2.5-0.5b-instruct-q8_0/summary.json` |
| `dense_slm_benchmark_v2` | `dense_slm` | `qwen2.5-1.5b-instruct-q4_k_m` | 2 | `ready` | `ci/hardware/apple-m4-mac-mini/2026-05-15T1845Z/slm-benchmark-v2/qwen2.5-1.5b-instruct-q4_k_m/summary.json` | `ci/hardware/apple-m4-mac-mini/2026-05-15/slm-benchmark-v2/qwen2.5-1.5b-instruct-q4_k_m/summary.json` |
| `bitnet_eval` | `bitnet` | `microsoft-bitnet-b1.58-2B-4T-i2s` | 2 | `ready` | `ci/hardware/apple-m4-mac-mini/2026-05-15T2214Z/bitnet-eval/answer-corpus.json` | `ci/hardware/apple-m4-mac-mini/2026-05-15/bitnet-eval/answer-corpus.json` |
| `bitnet_benchmark` | `bitnet` | `microsoft-bitnet-b1.58-2B-4T-i2s` | 2 | `ready` | `ci/hardware/apple-m4-mac-mini/2026-05-15T2214Z/bitnet-benchmark/summary.json` | `ci/hardware/apple-m4-mac-mini/2026-05-15/bitnet-benchmark/summary.json` |
| `bitnet_variable_warm` | `bitnet` | `microsoft-bitnet-b1.58-2B-4T-i2s` | 2 | `ready` | `ci/hardware/apple-m4-mac-mini/2026-05-16T0626Z/bitnet-productization/variable-warm-session.json` | `ci/hardware/apple-m4-mac-mini/2026-05-15/bitnet-productization/variable-warm-session.json` |

Claim boundary: dashboard only; no live model run, no model download, no BitNet chat/serve, no full Metal, QK256, Neural Engine, MPSGraph, MacBook, broad quality, broad performance, or speedup claim.
