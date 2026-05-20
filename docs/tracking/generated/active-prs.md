<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-066 | #87 | `codex/slm-cpu-066-swarm-runtime-hook-handoff` | Record the bitnet-rs-swarm handoff after the BitNet-rs SLM-CPU-065 release-surface gate accepted the single-tensor sidecar evidence but blocked packed Q8_0 runtime promotion. The item must preserve the accepted evidence for `layers.0.attention.q_proj.weight`, direct exact-tensor runtime hook implementation back to bitnet-rs-swarm, and require future return evidence with before/after strict CPU receipts proving unchanged model SHA, strict GGUF tokenizer authority, prompt IDs, generated IDs, decoded text, selected CPU backend/kernel identity, dense hook-selection identity, and fallback_used=false. This slice must not implement runtime compute or claim speedup, sustained throughput, broad answer quality, Q4/Q5 runtime support, server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5 support, or BitNet QK256 changes. |
