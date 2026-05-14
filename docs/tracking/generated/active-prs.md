<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | CPU-BITNET-REF-003 | #4715 | `codex/lunar-lake/CPU-BITNET-REF-003-direct-reference-tokens` | Record direct Microsoft BitNet.cpp generated-token IDs and first-token top-k/logit evidence for the fixed 258V prompts using a patched local llama-server boundary, then rerun the first-token divergence classifier so direct reference token evidence validates against the corrected 258V scalar/AVX2 CPU receipts without claiming broad answer quality, speed, Arc/NPU execution, QK256 changes, or full model correctness. |
| nvidia-5070ti | CUDA-DENSE-053 | #4713 | `cuda/dense-qwen25-warm-session-proof` | Add a Qwen2.5 0.5B Q8_0 warm-session strict CUDA proof showing model_loaded_once=true, tokenizer_loaded_once=true, CUDA context initialized once, intended buffers reused, per-turn receipts plus session summary, fallback_used=false, and full_cuda_residency_claimed=false unless every phase is proven device-resident. The current-source receipt records three turns, 24 total generated tokens, generated-token equality, runtime buffer reuse, upload-once weights, and speedup/full-residency/BitNet proof claims false. |
