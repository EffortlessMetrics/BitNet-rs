<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| apple-m4-local-answer | M4-BITNET-ASK-006 | #4691 | `codex/apple-m4-local-answer/M4-BITNET-ASK-006-bitnet-smoke` | Add `bitnet mac smoke --model-family bitnet` as a narrow one-shot BitNet ask smoke that defaults to the accepted Microsoft I2_S model id and external tokenizer path, preserves explicit --model-path/--tokenizer overrides only for BitNet, writes a compact aggregate smoke receipt after a successful one-shot answer, leaves durable BitNet ask failure receipts on setup failures, and keeps BitNet chat, serve, full Metal, QK256, Neural Engine, MPSGraph, broad quality, and performance claims disabled. |
