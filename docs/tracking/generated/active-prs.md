<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| apple-m4-slm-metal-phases | M4-METAL-004 | #4385 | `codex/apple-m4-slm-metal-phases/M4-METAL-004-runtime-boundary` | Record the resident-routing feasibility boundary for the validated Q/K/V Metal phase, explicitly documenting that live dispatch is currently test-local, choosing the required runtime extraction path, and blocking resident routing until a non-dev Metal runtime API exists. |
| nvidia-5070ti | CUDA-DENSE-041 | #4381 | `codex/cuda-dense-041-one-token` | Define the governed Qwen one-token strict CUDA proof contract after CUDA-DENSE-040, requiring the future implementation to consume the verified Qwen2.5 0.5B Q8_0 artifact, all-layer CUDA route receipt, model-boundary fixtures, KV-cache policy, and sampling-policy receipt; execute exactly one deterministic greedy token through the dense_regular_llm_cuda route with fallback_used=false; compare CPU/CUDA selected token and logits/top-k evidence; record tokenizer and prompt authority, selected backend, kernel/residency/timing evidence, and preserve speedup, short-decode, chat, server, full-residency, BitNet packed proof, QK256, tokenizer behavior, loader behavior, transformer runtime behavior, and CUDA kernel math non-claims. |
