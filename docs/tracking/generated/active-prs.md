<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| apple-m4-continuity | M4-CONT-001 | #4253 | `codex/apple-m4-continuity/M4-CONT-001-mac-chat` | Add a resident `bitnet mac chat` wrapper for the supported dense Qwen Apple M4 CPU/NEON path, accepting repeated prompts or stdin, streaming by default, reusing the existing warm-session runner so the model/tokenizer load once, writing aggregate and per-prompt receipts, and preserving device-boundary errors before cache/model work. |
