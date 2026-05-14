<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| nvidia-5070ti | CUDA-DENSE-053 | #4713 | `cuda/dense-qwen25-warm-session-proof` | Add a Qwen2.5 0.5B Q8_0 warm-session strict CUDA proof showing model_loaded_once=true, tokenizer_loaded_once=true, CUDA context initialized once, intended buffers reused, per-turn receipts plus session summary, fallback_used=false, and full_cuda_residency_claimed=false unless every phase is proven device-resident. The current-source receipt records three turns, 24 total generated tokens, generated-token equality, runtime buffer reuse, upload-once weights, and speedup/full-residency/BitNet proof claims false. |
