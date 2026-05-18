<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-040 | #5715 | `codex/slm-cpu-040-workspace-down-proj-storage` | Use the SLM-CPU-039 `feed_forward.output` workspace-owned boundary to attempt one reusable workspace-backed `FeedForward::down_proj` output storage hook, or explicitly prove why Candle's current linear/output API still prevents safe reuse at that exact boundary. The slice must preserve generated IDs, decoded text, strict GGUF tokenizer authority, selected CPU backend/kernel, model SHA, and fallback=false when behavior artifacts are regenerated; it must update allocation-boundary evidence and must not claim speedup or sustained throughput. |
