<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| nvidia-5070ti | CUDA-DENSE-052 | #4695 | `cuda/dense-qwen25-short-decode-unblock` | Add an 8-32 token deterministic Qwen2.5 0.5B Q8_0 short-decode strict CUDA receipt with fallback_used=false, stable greedy token sequence, valid UTF-8 answer, no raw special-token garbage, and recorded prefill, KV, logits, sampler, kernel, and transfer evidence. The current-source rerun supersedes the stale-binary diagnostic blocker and records decoded text `The answer is 4. What is` with CPU/CUDA generated-token equality. |
| slm-cpu | SLM-CPU-008YB | #4699 | `codex/slm-cpu-008yb-qwen-think-special-tokenizer` | Parse Qwen thinking control tokens `<think>` and `</think>` from the GGUF BPE vocabulary as special tokens when `parse_special=true`, so `--no-think` rendered prompts can be compared against known-good reference tokenization without literalizing thinking markers. This support slice must not claim first-token parity, known-good checkpoint capture, answer quality, tiny corpus success, multi-token stability, warm-session performance, Q4/Q5 expansion, server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5 support, or BitNet QK256 changes. |
