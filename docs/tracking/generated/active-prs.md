<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-065 | #6155 | `codex/slm-cpu-065-single-tensor-runtime-promotion-gate` | Open the first BitNet-rs release-surface runtime-promotion gate for the accepted single-tensor packed Q8_0 sidecar candidate returned from bitnet-rs-swarm. The item may implement or explicitly block opt-in exact-tensor runtime promotion for `layers.0.attention.q_proj.weight` only, and must preserve the Qwen3 Q8_0 strict CPU oracle with before/after receipts proving identical model SHA, strict GGUF tokenizer authority, prompt IDs, generated IDs, decoded text, selected CPU backend/kernel identity, dense hook-selection identity, and fallback_used=false. Runtime promotion must remain disabled by default unless the receipt gate proves behavior, and the item must not claim speedup, sustained throughput, broad answer quality, Q4/Q5 runtime support, server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5 support, or BitNet QK256 changes. |
