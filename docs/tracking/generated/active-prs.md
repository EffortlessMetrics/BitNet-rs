<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| apple-m4-local-answer | M4-BITNET-ASK-002 | #4651 | `codex/apple-m4-local-answer/M4-BITNET-ASK-002-progress` | Add explicit progress/status UX for slow one-shot `bitnet mac ask` runs so operators can see tokenizer/model verification, model/tokenizer loading, prompt tokenization, prefill, first-token, decode completion, and receipt validation milestones on stderr while generated text remains on stdout; add `--quiet` suppression and keep BitNet chat/serve/Metal claim boundaries unchanged. |
| nvidia-5070ti | CUDA-DENSE-051 | #4645 | `cuda/dense-qwen25-one-token-proof` | Refresh or add the real RTX 5070 Ti Qwen2.5 0.5B Q8_0 one-token strict CUDA proof so the receipt records dense_regular_llm_cuda routing, selected RTX 5070 Ti CUDA backend identity, fallback_used=false, CPU/CUDA selected token IDs, kernel and transfer stats, quality gate result, speedup_claim=false, and bitnet_packed_i2s_qk256_proof=false. |
