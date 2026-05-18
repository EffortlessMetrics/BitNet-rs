<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| apple-m4-inference-excellence | M4-DENSE-CHAT-001 | #5675 | `codex/apple-m4-inference-excellence/M4-DENSE-CHAT-001-cli-chat-conformance` | Prove dense SLM CLI ask/chat conformance on M4 for supported model identities: prompt template and stop behavior, bounded multi-turn history, timeout/cancel behavior, per-turn receipts, generated text, token IDs, backend, fallback state, and model/tokenizer identity. |
| slm-cpu | SLM-CPU-038 | #5679 | `codex/slm-cpu-038-typed-transformer-forward-workspace` | Use the SLM-CPU-037 no-reuse classification to introduce the first typed transformer forward workspace API boundary, or explicitly prove the next narrower API hook needed before replacing owned tensor outputs. The slice must preserve generated IDs, decoded text, strict GGUF tokenizer authority, selected CPU backend/kernel, model SHA, and fallback=false when behavior artifacts are regenerated. It must not claim sustained throughput, broad answer quality, Q4/Q5 runtime support, server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5 support, or BitNet QK256 changes. |
