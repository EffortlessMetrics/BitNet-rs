<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| nvidia-5070ti | CUDA-UX-004 | #4427 | `codex/cuda-ux-004-dense-chat` | Promote the validated dense Qwen CUDA warm-session runtime receipts into the user-facing `bitnet chat --device cuda --model qwen2.5-0.5b-instruct-q8_0` path, resolving the verified artifact/cache entry, selecting dense_regular_llm_cuda, accepting 2-4 bounded user turns with 5-16 generated tokens per turn, rejecting hidden fallback, producing assistant responses plus a compact proof summary and receipt path, and preserving speedup, full-residency, server, broad dense GGUF inference, BitNet packed proof, QK256, tokenizer behavior, loader behavior, transformer runtime behavior, and CUDA kernel math non-claims. |
