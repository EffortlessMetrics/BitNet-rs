<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| apple-m4-inference-excellence | M4-CONTEXT-001 | #6130 | `codex/apple-m4-inference-excellence/M4-CONTEXT-001-long-context-guardrails` | Add long-context envelope guardrails for M4 CLI and server routes so requests beyond recorded dense SLM or BitNet context/profile evidence produce clear advisory, batch, disabled, or unsupported states with receipt fields instead of silent overclaiming. |
| apple-m4-inference-excellence | M4-STABILITY-001 | #6134 | `codex/apple-m4-inference-excellence/M4-STABILITY-001-mixed-model-soak` | Run and record a mixed dense-model switch soak across supported M4 dense SLM identities, proving cache reuse, model unload/reload behavior, fallback=false backend selection, bounded memory drift, and receipt separation per model identity. |
