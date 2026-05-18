<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | LNL258V-QUAL-007 | #5656 | `codex/lunar-lake/LNL258V-QUAL-007-keyword-term-normalization` | Normalize bounded dense Qwen answer-corpus required/forbidden term checks so regular plural forms satisfy the same token-boundary policy as singular terms, rerun the 258V CPU corpus-v2 receipt, refresh the CPU diagnosis plus route-profile/regression/comparison indexes, and preserve no route promotion, no speedup, no power advantage, no Arc/NPU acceleration, no native OpenCL/NPU claim, and no BitNet QK256/I2_S behavior change. |
| nvidia-5070ti | CUDA-MODEL-011 | #5645 | `codex/cuda-model-011-qwen3-chat-user-path` | Record Qwen3 0.6B Q8_0 through the normal bitnet chat user path with model/tokenizer/CUDA context/weights loaded once across multiple prompts, selected_backend=nvidia-rtx-5070-ti-cuda, selected_route=dense_regular_llm_cuda, fallback_used=false, quality gate evidence, and no product, server, speedup, full-residency, or BitNet QK256 promotion. |
