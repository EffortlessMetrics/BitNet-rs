<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| nvidia-5070ti | CUDA-DENSE-041 | #4381 | `codex/cuda-dense-041-one-token` | Define the governed Qwen one-token strict CUDA proof contract after CUDA-DENSE-040, requiring the future implementation to consume the verified Qwen2.5 0.5B Q8_0 artifact, all-layer CUDA route receipt, model-boundary fixtures, KV-cache policy, and sampling-policy receipt; execute exactly one deterministic greedy token through the dense_regular_llm_cuda route with fallback_used=false; compare CPU/CUDA selected token and logits/top-k evidence; record tokenizer and prompt authority, selected backend, kernel/residency/timing evidence, and preserve speedup, short-decode, chat, server, full-residency, BitNet packed proof, QK256, tokenizer behavior, loader behavior, transformer runtime behavior, and CUDA kernel math non-claims. |
