<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| nvidia-5070ti | CUDA-DENSE-046 | #4422 | `codex/cuda-dense-046-warm-session` | Extend the governed dense Qwen strict CUDA runtime from the CUDA-DENSE-045 short-decode proof to a warm-session proof, loading the SHA-verified qwen2.5-0.5b-instruct-q8_0 artifact and tokenizer once, initializing CUDA once, reusing dense CUDA weights and governed runtime buffers across multiple deterministic turns, emitting per-turn and session-summary dense_gguf_qwen_warm_session_strict_cuda receipts with fallback_used=false, generated-token evidence, kernel/residency/timing/transfer summaries, prerequisite receipt hashes, and preserving ask/chat UX, speedup, persistent/full-residency beyond the warm-session scope, server, BitNet packed proof, QK256, tokenizer behavior, loader behavior, transformer runtime behavior, and CUDA kernel math non-claims. |
