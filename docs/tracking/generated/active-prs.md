<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-052 | #5921 | `codex/slm-cpu-052-selector-update-impl` | Implement the first behavior-preserving packed Q8_0 sidecar selector update after SLM-CPU-051. The slice may allow the dense-linear selector to choose a packed Q8_0 sidecar candidate only for the exact evidence-scoped Qwen3 Q8_0 appliance profile where generated-ID/text equivalence, strict tokenizer authority, selected CPU backend/kernel, model SHA, and fallback=false are proven against the eager F32 Candle oracle. If runtime sidecar execution remains blocked, the slice must record the blocker explicitly. Any before/after receipt must preserve prompt IDs, generated IDs, decoded text, loader/tokenizer provenance, backend/kernel identity, and fallback=false before claiming even a bounded improvement. It must not claim sustained throughput, broad answer quality, Q4/Q5 runtime support, server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5 support, or BitNet QK256 changes. |
