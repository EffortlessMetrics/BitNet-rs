# Apple M4 Operator Envelope V2

This envelope maps supported M4 Mac mini operator commands to their receipt
requirements, gates, report families, and unsupported claim boundaries. It is a
local operator contract, not a broad Apple Silicon benchmark.

## Global Boundaries

Every supported M4 inference receipt must keep these fields true unless a later
campaign explicitly proves otherwise:

```text
selected_backend=apple-m4-cpu-neon
runtime_api=cpu
fallback_used=false
machine_id or machine.id=apple-m4-mac-mini
```

Still unsupported by this envelope:

```text
BitNet chat
BitNet serve
full apple-m4-metal inference
QK256 on Apple Silicon
Neural Engine execution
MPSGraph model inference
MacBook evidence
broad Apple Silicon performance claims
broad model quality claims
speedup claims
```

Dense SLM evidence and BitNet evidence are separate. A dense Qwen receipt never
proves BitNet behavior, and a BitNet receipt never broadens dense SLM support.

## Command Map

| Command | Family | Receipt or artifact | Gate | Report family |
|---|---|---|---|---|
| `bitnet mac models` | operator | model-free catalog JSON when `--json` is used | supported model matrix and cache state only | none |
| `bitnet mac status` | operator | `apple_m4_inference_status` | no live model run; dense/BitNet readiness separated | local report inventory |
| `bitnet mac report-refresh` | operator | `apple_m4_report_refresh_manifest` | committed reports only; no model downloads | dense eval, dense benchmark, BitNet eval, BitNet benchmark, BitNet warm |
| `bitnet mac regression-dashboard` | operator | `apple_m4_regression_dashboard` plus Markdown | matching identity before comparison | same as report-refresh |
| `bitnet mac ask` with dense model | dense SLM | strict local-answer receipt from the ask path | supported dense model ID, tokenizer, CPU/NEON, fallback false | dense answer/runtime receipts |
| `bitnet mac chat` with dense model | dense SLM | `slm_apple_m4_warm_session` | resident session, per-turn token IDs, fallback false | dense warm-session receipts |
| `bitnet mac serve` | dense SLM | server health/ready/completion receipts | local server only; readiness before completion | dense local-server receipts |
| `bitnet mac smoke` | dense SLM | `apple_m4_slm_golden_smoke` | compact quality smoke, generated text and token IDs | dense smoke |
| `bitnet mac doctor` | dense SLM | `apple_m4_slm_doctor` | cache, model, backend, and receipt checks | dense health |
| `bitnet mac validate` | dense SLM | `apple_m4_slm_operator_profiles` or `apple_m4_slm_performance_profiles` | bounded corpus/profile set only | dense quality/profile receipts |
| `bitnet mac benchmark` | dense SLM | `apple_m4_slm_benchmark_v2` | release-mode profile set; no broad benchmark claim | `slm-benchmark-v2` |
| `bitnet mac regression` | dense SLM and BitNet | comparison output, optionally via `receipts-check --regression-baseline` | context match before drift checks | matching report pairs only |
| `bitnet mac ask --model-id microsoft-bitnet-b1.58-2B-4T-i2s ...` | BitNet | one-shot BitNet ask receipt or `bitnet_apple_m4_mac_ask_failure` | accepted I2_S GGUF, external tokenizer SHA, CPU/NEON, fallback false | BitNet ask/runtime receipts |
| `bitnet mac bitnet-warm` | BitNet | `bitnet_apple_m4_warm_session` or `bitnet_apple_m4_warm_session_failure` | accepted artifact, tokenizer authority, per-turn receipts, timeout evidence | BitNet warm/productization |
| `bitnet mac bitnet-chat-gate` | BitNet | `bitnet_apple_m4_chat_gate` | warm, failure, and streaming-semantics receipts all valid | BitNet productization gate |
| `bitnet mac smoke --model-family bitnet` | BitNet | `bitnet_apple_m4_mac_smoke` | accepted artifact and bounded smoke only | BitNet smoke |
| `bitnet mac bitnet-benchmark` | BitNet | `bitnet_apple_m4_benchmark_v1` | one-shot plus fixed-warm benchmark receipts | `bitnet-benchmark` |
| `bitnet mac bitnet-proof` | BitNet | `apple_m4_bitnet_proof_preflight` | proof input validation only | BitNet proof preflight |
| `bitnet mac receipts-check` | all | receipt-check summary | schema, backend, fallback, token, and claim-boundary validation | all supported receipt kinds |

## Report Families

The report-refresh manifest and regression dashboard know these committed
families:

| Family | Evidence | Expected artifact kind | Comparison boundary |
|---|---|---|---|
| `dense_slm_eval_v2` | dense SLM | `apple_m4_slm_eval_summary` | same model ID, model SHA, tokenizer, backend, fallback |
| `dense_slm_benchmark_v2` | dense SLM | `apple_m4_slm_benchmark_v2` | same model cache identity, profile set, backend, fallback |
| `bitnet_eval` | BitNet | `bitnet_apple_m4_local_answer_corpus` | same accepted GGUF SHA, tokenizer authority, prompt template, backend, fallback |
| `bitnet_benchmark` | BitNet | `bitnet_apple_m4_benchmark_v1` | same BitNet artifact/tokenizer, benchmark set, backend, fallback |
| `bitnet_variable_warm` | BitNet | `bitnet_apple_m4_warm_session` | same artifact/tokenizer, warm-session scope, backend, fallback |

When a family has only one matching report identity, the dashboard must report
`insufficient_history`. It may offer the self-baseline command for validation,
but it must not describe a trend until at least two matching reports exist.

## Operator Sequence

Use this order when refreshing the M4 appliance state:

```bash
bitnet mac models
bitnet mac status
bitnet mac report-refresh
bitnet mac regression-dashboard
bitnet mac receipts-check <receipt.json> --json
bitnet mac regression <current.json> --baseline <baseline.json>
```

Run live dense or BitNet model commands only in local, advisory, scheduled, or
release lanes. Generic PR CI stays model-free: schema checks, receipt checks,
manifest generation, and dashboard generation only.

## Claim Language

Allowed:

```text
The M4 Mac mini has receipt-backed dense SLM local ask/chat/server paths for the
supported Qwen model IDs.
The M4 Mac mini has receipt-backed BitNet one-shot ask and warm-session paths
for the accepted Microsoft I2_S artifact and external tokenizer.
The report-refresh manifest and regression dashboard can validate committed M4
receipt families without live model runs.
```

Not allowed:

```text
BitNet chat works.
BitNet serve works.
Full Apple M4 Metal inference works.
QK256, Neural Engine, MPSGraph, MacBook, or broad Apple Silicon performance is proven.
The current reports are a broad model quality benchmark.
Dense SLM receipts prove BitNet behavior.
```
