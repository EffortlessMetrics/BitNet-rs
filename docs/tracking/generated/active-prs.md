<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | CPU258V-021 | #4315 | `codex/lunar-lake/CPU258V-021-reference-instrumentation` | Instrument or script the external BitNet reference runner boundary so the fixed 258V prompts expose prompt token IDs, first generated token IDs, decoded first tokens, and first-token logits/top-k when available; if the reference cannot expose logits or token IDs, record the blocker precisely without inferring parity. |
| nvidia-5070ti | CUDA-DENSE-032 | #4316 | `codex/cuda-dense-032-reference-harness-tracker` | Define the full dense GGUF one-layer CPU reference harness contract after CUDA-DENSE-031 made every one-layer op CUDA-routable, covering deterministic Qwen2.5 0.5B Q8_0 input fixture sources, CPU reference op ordering and numeric tolerances, receipt fields, and next CUDA comparison boundaries while keeping CUDA execution, dense GGUF inference, Qwen token/decode/chat, speedup, full-residency, BitNet packed proof, tokenizer, loader, transformer, QK256, server, and CUDA kernel math claims false. |
