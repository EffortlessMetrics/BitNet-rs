<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| apple-m4-inference-excellence | M4-CONTEXT-001 | #6130 | `codex/apple-m4-inference-excellence/M4-CONTEXT-001-long-context-guardrails` | Add long-context envelope guardrails for M4 CLI and server routes so requests beyond recorded dense SLM or BitNet context/profile evidence produce clear advisory, batch, disabled, or unsupported states with receipt fields instead of silent overclaiming. |
| apple-m4-inference-excellence | M4-STABILITY-HARNESS-001 | #6131 | `codex/apple-m4-inference-excellence/M4-STABILITY-HARNESS-001-mixed-model-profile` | Implement the `bitnet mac benchmark --profile mixed_model_switch` harness and receipt validation required by M4-STABILITY-001 before any mixed dense-model switch soak evidence is recorded: run one resident_25 child benchmark summary per supported dense M4 model identity, preserve per-model child receipts, validate cache/model identity separation, fallback=false backend selection, and memory summary fields, and keep the change harness-only without publishing live soak evidence. |
