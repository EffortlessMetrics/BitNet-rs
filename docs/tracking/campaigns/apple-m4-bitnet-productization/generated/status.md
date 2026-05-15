<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Apple M4 BitNet productization Campaign Status

- Campaign: `apple-m4-bitnet-productization`
- State: `active`
- Objective: Move Apple M4 BitNet from fixed one-shot and fixed-warm proof into operator-ready warm sessions, then chat and serve only after receipt-backed correctness, determinism, timeout, streaming, and failure-mode gates pass.

## Work Items

| Item | State | PR | Branch | Review | Merge | Human gate | Acceptance |
|---|---|---:|---|---|---|---|---|
| M4-BITNET-PROD-001 | merged | #4946 | `codex/apple-m4-bitnet-productization/M4-BITNET-PROD-001-variable-warm-prompts` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Allow `bitnet mac bitnet-warm` to run operator-supplied repeated prompts in one resident Apple M4 CPU/NEON BitNet process while preserving the fixed proof prompt default, accepted model/tokenizer checks, repeated-prompt determinism, per-turn receipts, aggregate prompt-source metadata, and disabled chat/serve/Metal/QK256/Neural Engine/MPSGraph/broad-claim boundaries. |
| M4-BITNET-PROD-002 | merged | #4950 | `codex/apple-m4-bitnet-productization/M4-BITNET-PROD-002-variable-warm-runtime` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Run the operator-prompt BitNet warm route on the M4 Mac mini with accepted model/tokenizer identity, at least five variable prompts including one exact repeat, fallback_used=false, generated text/token IDs, repeated-prompt determinism, per-turn and aggregate timing/memory receipts, timeout boundary, and no chat/serve/Metal/broad claims. |
| M4-BITNET-PROD-003 | merged | #4954 | `codex/apple-m4-bitnet-productization/M4-BITNET-PROD-003-timeout-failure-ux` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Add BitNet warm-session progress, timeout, and partial-failure receipts that identify model load, tokenizer load, prefill, first-token, decode, and receipt-write stages with repair guidance, while preserving disabled chat/serve/Metal/broad-claim boundaries. |
| M4-BITNET-PROD-004 | in_progress | TBD | `codex/apple-m4-bitnet-productization/M4-BITNET-PROD-004-chat-gate` | `codex_premerge` | `automerge_when_green` | `on_blocker_only` | Define and enforce the BitNet chat enablement gate from variable warm-session receipts, repeated-prompt determinism, timeout/failure evidence, streaming semantics, and claim boundaries before any `bitnet mac chat --model-family bitnet` route is enabled. |

## Hard Constraints

- This is an M4 Mac mini BitNet campaign.
- Use only the accepted Microsoft I2_S GGUF paired with external Microsoft tokenizer authority for answer claims.
- Use the bitnetcpp-answer prompt authority for BitNet M4 answer and warm-session reports.
- Do not use dense Qwen or dense SLM evidence as BitNet quality, performance, or UX evidence.
- Do not enable BitNet chat until variable warm-session receipts prove stable correctness, determinism, timeout behavior, and failure boundaries.
- Do not enable BitNet serve until chat and streaming/request receipts pass.
- Do not claim full apple-m4-metal inference, QK256 support, Neural Engine execution, MPSGraph model inference, MacBook evidence, speedup, broad BitNet quality, or broad Apple Silicon performance.
- Do not add live model downloads, long hardware timing runs, variable warm-session soaks, or model binaries to generic required PR CI.
