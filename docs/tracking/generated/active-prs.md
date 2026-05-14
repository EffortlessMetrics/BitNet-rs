<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| apple-m4-local-answer | M4-BITNET-WARM-001 | #4701 | `codex/apple-m4-local-answer/M4-BITNET-WARM-001-route` | Add a narrow `bitnet mac bitnet-warm` route that uses the accepted Microsoft I2_S GGUF and accepted external tokenizer, loads model/tokenizer once through the resident warm-session engine, runs three fixed prompts with one repeated prompt for determinism, writes per-turn receipts and an aggregate `bitnet_apple_m4_warm_session` receipt, records fallback_used=false, generated text/token IDs, timing and memory fields, validates the receipt, and keeps BitNet chat, serve, full Metal, QK256, Neural Engine, MPSGraph, broad quality, and broad performance claims disabled. |
