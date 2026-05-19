<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-054 | #5983 | `codex/slm-cpu-054-sidecar-runtime-hook` | Burn down the SLM-CPU-053 packed Q8_0 sidecar runtime blocker by adding the narrow production dense-linear runtime hook or proving the exact remaining hook/API gap. The slice must keep eager_f32_candle as the default Qwen3 Q8_0 Kaby behavior oracle unless before/after receipts prove identical model SHA, tokenizer source/strictness, prompt IDs, generated IDs, decoded text, selected CPU backend/kernel identity, and fallback_used=false. If packed sidecar compute is not safely enabled, emit an updated blocker artifact that names the remaining gap and keeps sidecar_runtime_compute_allowed=false. It must preserve speedup_claim=false unless a later performance item records separate bounded timing evidence, and must not claim sustained throughput, broad answer quality, Q4/Q5 runtime support, server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5 support, or BitNet QK256 changes. |
