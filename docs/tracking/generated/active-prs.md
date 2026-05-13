<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-008S | #4606 | `codex/slm-cpu-008s-qwen3-output-head-root-cause` | Use the SLM-CPU-008R post-#4434 artifact and trace to localize the remaining Qwen3-0.6B Q8_0 first-token divergence, prioritizing output head/vocab indexing and shared transformer math. The item must identify the next concrete drift point or add the smallest missing diagnostic needed to do so, while preserving strict loader/tokenizer provenance and fallback=false. No answer-quality, tiny corpus, sustained throughput, server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5, Q4/Q5 quant expansion, or BitNet QK256 claim is allowed. |
