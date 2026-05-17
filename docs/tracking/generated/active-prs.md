<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-024 | #5342 | `codex/slm-cpu-024-greedy-sampler-fast-path` | Add a guarded greedy no-penalty sampler fast path that bypasses sampler logits scratch copying for temperature=0.0 and inactive repetition penalty, then record a real i5-8250U Qwen3 Q8_0 warm-session after artifact proving the same model SHA, generated IDs, decoded text, strict GGUF tokenizer authority, selected CPU backend, fallback=false, and zero sampler decode allocations. This slice must not claim sustained throughput, broad answer quality, Q4/Q5 quant expansion, a second dense model, server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5 support, or BitNet QK256 changes. |
