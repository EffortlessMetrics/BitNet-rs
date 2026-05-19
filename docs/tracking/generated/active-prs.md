<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| apple-m4-inference-excellence | M4-RECEIPT-001 | #5811 | `codex/apple-m4-inference-excellence/M4-RECEIPT-001-schema-compat` | Add receipt-schema compatibility and negative fixtures for M4 eval, benchmark, warm, chat-gate, serve-gate, dashboard, and failure receipts so missing identity fields, fallback ambiguity, malformed timing, absent token IDs, and unsupported claim fields fail validation clearly. |
| slm-cpu | SLM-CPU-044 | #5810 | `codex/slm-cpu-044-q8-sidecar-runtime-plan` | Define the first production-integration boundary after the SLM-CPU-043 fixture-level Q8_0 packed sidecar prototype. The slice must specify how a packed Q8_0 sidecar can be carried from GGUF load metadata toward a runtime dense-linear path without replacing the existing eager F32 Candle tensors yet, including exact behavior-preservation gates for prompt IDs, generated IDs, decoded text, model SHA, strict GGUF tokenizer authority, selected CPU backend/kernel, and fallback=false. It must name any remaining API/layout blockers before runtime use and must not claim speedup, sustained throughput, Q4/Q5 runtime support, server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5 support, or BitNet QK256 changes. |
