<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-041 | #5754 | `codex/slm-cpu-041-linear-output-storage-api` | Add or adopt a behavior-preserving dense linear output-storage API boundary that can fill a caller-provided workspace slot for the Qwen3 Q8_0 Kaby FeedForward::down_proj output, or prove with a narrower implementation note why Candle's current Tensor/Linear surface still prevents this without changing math semantics. The slice must preserve generated IDs, decoded text, strict GGUF tokenizer authority, selected CPU backend/kernel, model SHA, and fallback=false when behavior artifacts are regenerated; it must keep SLM-CPU-040's no-speedup/no-throughput boundary unless reusable storage is actually implemented and proven with before/after receipts. |
