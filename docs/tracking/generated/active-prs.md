<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| nvidia-5070ti | CUDA-DENSE-040 | #4370 | `codex/cuda-dense-040-sampling-policy` | Implement governed dense GGUF logits-transfer and sampling-policy receipts after CUDA-DENSE-039, recording LM-head logits source/hash/length/top-k evidence, deterministic greedy CPU sampler policy, estimated D2H logits bytes per decode step, remaining model-boundary policy gaps, and claim-boundary rejection of runtime sampling integration, dense GGUF inference, Qwen one-token/short decode/chat, speedup, persistent/full residency, server readiness, BitNet packed proof, tokenizer behavior, loader behavior, transformer runtime behavior, QK256, BitNet CUDA, and CUDA kernel math claims. |
