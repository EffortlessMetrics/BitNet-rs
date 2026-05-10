<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| nvidia-5070ti | CUDA-DENSE-042 | #4392 | `codex/cuda-dense-042-one-token-impl` | Add the strict dense Qwen one-token CUDA proof receipt validator and synthetic rejection tests required by CUDA-DENSE-041, requiring future receipts to prove one deterministic greedy token through dense_regular_llm_cuda with fallback_used=false, CPU/CUDA selected-token agreement, prerequisite all-layer/model-boundary/KV/sampling receipt hashes, tokenizer/prompt authority, kernel/residency/timing evidence, and claim-boundary rejection of short-decode, chat, speedup, full-residency, server, BitNet packed proof, QK256, tokenizer behavior, loader behavior, transformer runtime behavior, CUDA kernel math, and dense GGUF broad-inference claims. |
