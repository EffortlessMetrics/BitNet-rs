<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | LNL258V-ASK-002 | #4654 | `codex/lunar-lake/LNL258V-ASK-002-live-operator-ask` | Restore the live Lunar Lake dense Qwen2.5 CPU operator ask route so bounded math prompts produce the expected answer with fallback_used=false, preserve the dedicated Qwen2.5 output.weight receipt identity, and add an optional bounded answer gate for operator ask receipts without claiming broad dense SLM quality, speedup, Arc/NPU execution, acceleration, full BitNet accelerator inference, or BitNet QK256/I2_S proof. |
| nvidia-5070ti | CUDA-DENSE-051 | #4645 | `cuda/dense-qwen25-one-token-proof` | Refresh or add the real RTX 5070 Ti Qwen2.5 0.5B Q8_0 one-token strict CUDA proof so the receipt records dense_regular_llm_cuda routing, selected RTX 5070 Ti CUDA backend identity, fallback_used=false, CPU/CUDA selected token IDs, kernel and transfer stats, quality gate result, speedup_claim=false, and bitnet_packed_i2s_qk256_proof=false. |
