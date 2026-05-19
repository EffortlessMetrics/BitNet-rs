<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-056 | #6018 | `codex/slm-cpu-056-production-dense-linear-hook-impl` | Implement the first production dense-linear hook boundary defined by SLM-CPU-055 without enabling packed Q8_0 sidecar compute by default. The hook must let transformer dense linear calls report an explicit eager-F32 selection or a selected Q8_0 sidecar descriptor while preserving the current Qwen3 Q8_0 Kaby behavior oracle. Any before/after artifact must prove identical model SHA, tokenizer source/strictness, prompt IDs, generated IDs, decoded text, selected CPU backend/kernel identity, and fallback_used=false. If packed runtime compute remains disabled, emit a machine-checkable blocker artifact naming the remaining runtime or receipt gap. It must not claim speedup, sustained throughput, broad answer quality, Q4/Q5 runtime support, server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5 support, or BitNet QK256 changes. |
