<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | LNL258V-ASK-001 | #4644 | `codex/lunar-lake/LNL258V-ASK-001-operator-ask` | Add a policy-gated `bitnet lunar-lake ask` wrapper for the dense Qwen CPU default route. The command must read the committed operator-readiness receipt before generation, refuse non-default or accelerator routes, run through the CPU/dense-Qwen path with strict tokenizer and loader behavior, emit an operator ask receipt with source run evidence, route reason, answer gate result, timing fields, fallback_used=false, and no broad dense quality, speedup, Arc/NPU execution, acceleration, full BitNet accelerator inference, or BitNet QK256/I2_S claim. |
| nvidia-5070ti | CUDA-DENSE-051 | #4645 | `cuda/dense-qwen25-one-token-proof` | Refresh or add the real RTX 5070 Ti Qwen2.5 0.5B Q8_0 one-token strict CUDA proof so the receipt records dense_regular_llm_cuda routing, selected RTX 5070 Ti CUDA backend identity, fallback_used=false, CPU/CUDA selected token IDs, kernel and transfer stats, quality gate result, speedup_claim=false, and bitnet_packed_i2s_qk256_proof=false. |
