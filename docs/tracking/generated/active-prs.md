<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-008T | #4611 | `codex/slm-cpu-008t-first-token-parity` | Fix the post-008 Qwen3-0.6B Q8_0 first-token parity candidate by selecting the dedicated GGUF output.weight head before tied-embedding fallback, then require a real i5-8250U artifact refresh to prove whether bitnet-rs now matches the reference token 19 / '4' for the same model SHA, rendered prompt, prompt IDs, greedy settings, strict tokenizer, selected CPU backend, and fallback=false. This slice must not claim answer quality, tiny corpus, multi-token stability, warm-session performance, Q4/Q5 quant expansion, server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5 support, or BitNet QK256 changes. |
