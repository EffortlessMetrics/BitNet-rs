<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-045 | #5845 | `codex/slm-cpu-045-q8-sidecar-carrier` | Add the first production-facing Qwen3 Q8_0 packed sidecar carrier after SLM-CPU-044 by preserving packed Q8_0 block metadata from strict GGUF tensor loading into an inert model-side sidecar registry or descriptor while keeping the existing eager F32 Candle tensors as the only runtime compute path. The slice must prove the sidecar carrier is metadata-only/inert for generation, preserve prompt IDs, generated IDs, decoded text, model SHA, strict GGUF tokenizer authority, selected CPU backend/kernel, and fallback=false in behavior evidence or fixture-equivalent gates, and name the next runtime API hook needed before any dense-linear sidecar compute can be selected. It must not claim speedup, sustained throughput, Q4/Q5 runtime support, server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5 support, or BitNet QK256 changes. |
