<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| nvidia-5070ti | CUDA-DENSE-045 | #4420 | `codex/cuda-dense-045-short-decode-proof` | Extend the governed dense Qwen strict CUDA runtime from the CUDA-DENSE-044 one-token proof to a bounded deterministic short-decode proof, generating 5-16 tokens through dense_regular_llm_cuda on the RTX 5070 Ti with fallback_used=false, recording prompt/tokenizer authority, generated token IDs, CPU/CUDA comparison or first divergence evidence, kernel/residency/timing/transfer summaries, prerequisite receipt hashes, and a validated dense_gguf_qwen_short_decode_strict_cuda_proof receipt while preserving chat, speedup, persistent/full-residency, server, BitNet packed proof, QK256, tokenizer behavior, loader behavior, transformer runtime behavior, and CUDA kernel math non-claims. |
