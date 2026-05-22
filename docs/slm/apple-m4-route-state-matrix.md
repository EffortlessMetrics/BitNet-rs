# Apple M4 Route State Matrix

`M4-ROUTE-MATRIX-001` makes route availability explicit for the M4 Mac mini
appliance. The matrix is embedded in both operator receipts:

```bash
bitnet mac status --json
bitnet mac evidence --json
```

Look for `route_state_matrix` on schema-`1.3.0` status/evidence receipts. It is
model-free: the commands inventory cache, docs, and committed receipts only.
They do not fetch models, run live generation, enable a disabled route, or turn
dense SLM evidence into BitNet evidence. Older operator receipts remain valid
without this matrix.

## State Contract

| State | Meaning |
|---|---|
| `enabled` | The route may run when its cache and gate preconditions pass. |
| `disabled_without_ready_gate` | The route exists, but must fail closed unless a ready gate receipt is supplied. |
| `batch_only` | The route is valid but should be treated as slow or unattended from the recorded evidence. |
| `unsupported` | No accepted M4 Mac mini appliance receipt supports this route or claim. |

## Route Matrix

| Family | Surface | State | Class | Evidence item | Receipt family |
|---|---|---|---|---|---|
| dense SLM | ask | `enabled` | interactive or advisory by selected model | `M4-DENSE-CHAT-001`, `M4-OPS-SLO-001`, `M4-CONTEXT-001` | strict local-answer receipt, `slm_apple_m4_warm_session`, `apple_m4_context_guardrail` |
| dense SLM | chat | `enabled` | interactive or advisory by selected model | `M4-DENSE-CHAT-001`, `M4-OPS-SLO-001`, `M4-CONTEXT-001` | `apple_m4_slm_chat_smoke`, `slm_apple_m4_warm_session`, `apple_m4_context_guardrail` |
| dense SLM | warm session | `enabled` | interactive or advisory by selected model | `M4-DENSE-CHAT-001`, `M4-BENCH-002`, `M4-OPS-SLO-001` | `slm_apple_m4_warm_session`, `apple_m4_slm_benchmark_v2` |
| dense SLM | serve | `enabled` | advisory | `M4-SERVE-EX-001`, `M4-SERVE-EX-002`, `M4-SERVE-EX-003`, `M4-SERVE-EX-004` | `bitnet_apple_m4_local_server_health`, `bitnet_apple_m4_local_server_ready`, `bitnet_apple_m4_local_server_completion`, `apple_m4_serve_failure_semantics`, `apple_m4_serve_backpressure_smoke` |
| dense SLM | streaming | `enabled` | advisory | `M4-DENSE-CHAT-001`, `M4-SERVE-EX-002` | `apple_m4_slm_chat_smoke`, `bitnet_apple_m4_local_server_completion`, `apple_m4_serve_failure_semantics` |
| dense SLM | long context | `batch_only` | batch | `M4-CONTEXT-001`, `M4-CONTEXT-002` | `apple_m4_context_guardrail`, `apple_m4_slm_eval_summary`, `apple_m4_slm_benchmark_v2` |
| BitNet | ask | `enabled` | batch | `M4-BITNET-EX-003`, `M4-BITNET-EX-011`, `M4-BITNET-EX-014`, `M4-BITNET-EX-015` | `strict_bitnet_cpu_profile`, `bitnet_apple_m4_mac_ask_failure`, `bitnet_apple_m4_local_answer_corpus` |
| BitNet | warm session | `enabled` | batch | `M4-BITNET-EX-004`, `M4-BITNET-EX-005`, `M4-BENCH-006`, `M4-BITNET-EX-015` | `bitnet_apple_m4_warm_session`, `bitnet_apple_m4_warm_session_failure`, `bitnet_apple_m4_benchmark_v1` |
| BitNet | chat | `disabled_without_ready_gate` | disabled | `M4-BITNET-EX-006` | `bitnet_apple_m4_chat_gate`, `bitnet_apple_m4_chat_session` |
| BitNet | serve | `disabled_without_ready_gate` | disabled | `M4-BITNET-EX-007`, `M4-SERVE-EX-002`, `M4-SERVE-EX-004` | `bitnet_apple_m4_serve_gate`, `bitnet_apple_m4_serve_completion`, `apple_m4_serve_failure_semantics`, `apple_m4_serve_backpressure_smoke` |
| BitNet | streaming | `disabled_without_ready_gate` | disabled | `M4-BITNET-EX-006`, `M4-BITNET-EX-007`, `M4-SERVE-EX-002` | `bitnet_apple_m4_chat_gate`, `bitnet_apple_m4_chat_session`, `bitnet_apple_m4_serve_gate`, `bitnet_apple_m4_serve_completion` |
| BitNet | long context | `batch_only` | batch | `M4-CONTEXT-001`, `M4-CONTEXT-002`, `M4-BITNET-EX-015` | `apple_m4_context_guardrail`, `bitnet_apple_m4_warm_session`, `bitnet_apple_m4_local_answer_corpus` |
| all | full Metal route | `unsupported` | unsupported | none | none |
| all | QK256, Neural Engine, MPSGraph, MacBook, or broad Apple Silicon route | `unsupported` | unsupported | none | none |

## Boundaries

The route matrix is an operator truth table, not a proof generator. It preserves
these boundaries:

- BitNet chat and BitNet serve are disabled by default and require ready gate
  receipts.
- Dense SLM serve remains local loopback appliance evidence, not production
  hosting or broad OpenAI compatibility.
- Long-context dense and BitNet paths are batch-only inside their recorded
  prompt envelopes and unsupported beyond those envelopes.
- Full `apple-m4-metal`, Apple QK256, Neural Engine, MPSGraph, MacBook, broad
  Apple Silicon, broad model quality, broad performance, and speedup claims
  remain unsupported without separate receipts.
