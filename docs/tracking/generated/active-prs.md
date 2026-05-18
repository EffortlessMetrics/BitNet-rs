<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-031 | #5499 | `codex/slm-cpu-031-kv-cache-session-reuse` | Reuse a single CPU KV cache across Qwen3 Q8_0 Kaby warm-session prompts by clearing it before each prompt. The slice must preserve prompt isolation, generated IDs, decoded text, strict GGUF tokenizer authority, selected CPU backend/kernel, model SHA, and fallback=false when real artifacts are regenerated. Receipts must distinguish the session-level KV cache allocation from per-prompt prompt_setup.kv_cache clear/reset work and must not claim sustained throughput, broad answer quality, Q4/Q5 runtime support, server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5 support, or BitNet QK256 changes. |
