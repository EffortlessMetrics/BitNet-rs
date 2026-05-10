<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| nvidia-5070ti | CUDA-DENSE-043 | #4407 | `codex/cuda-dense-043-one-token-runtime` | Define the governed CUDA-DENSE-044 runtime proof gate after CUDA-DENSE-042, documenting that the future implementation must consume the SHA-verified qwen2.5-0.5b-instruct-q8_0 artifact plus all-layer plan, model-boundary fixture, KV-cache policy, and sampling-policy receipts; execute exactly one deterministic greedy token through dense_regular_llm_cuda on the RTX 5070 Ti with fallback_used=false; compare CPU/CUDA selected token and logits/top-k evidence; emit and validate a dense_gguf_qwen_one_token_strict_cuda_proof hardware receipt; and preserve short-decode, chat, speedup, persistent/full-residency, server, BitNet packed proof, QK256, tokenizer behavior, loader behavior, transformer runtime behavior, and CUDA kernel math non-claims. |
