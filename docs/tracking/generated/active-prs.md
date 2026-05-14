<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| slm-cpu | SLM-CPU-008YB | #4699 | `codex/slm-cpu-008yb-qwen-think-special-tokenizer` | Parse Qwen thinking control tokens `<think>` and `</think>` from the GGUF BPE vocabulary as special tokens when `parse_special=true`, so `--no-think` rendered prompts can be compared against known-good reference tokenization without literalizing thinking markers. This support slice must not claim first-token parity, known-good checkpoint capture, answer quality, tiny corpus success, multi-token stability, warm-session performance, Q4/Q5 expansion, server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5 support, or BitNet QK256 changes. |
