<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| apple-m4-local-answer | M4-BITNET-ASK-001 | #4647 | `codex/apple-m4-local-answer/M4-BITNET-ASK-001-runtime-receipt` | Prove the user-facing `bitnet mac ask` BitNet route completes through the accepted Microsoft I2_S GGUF and accepted external tokenizer in strict mode, selects apple-m4-cpu-neon, records fallback_used=false, valid UTF-8, non-empty coherent output, generated token IDs, timing receipt fields, and a documented operator command; if the route is too slow, record load, tokenizer, prefill, first-token, decode timing, timeout boundary, and where the delay occurs without claiming success. |
| nvidia-5070ti | CUDA-DENSE-051 | #4645 | `cuda/dense-qwen25-one-token-proof` | Refresh or add the real RTX 5070 Ti Qwen2.5 0.5B Q8_0 one-token strict CUDA proof so the receipt records dense_regular_llm_cuda routing, selected RTX 5070 Ti CUDA backend identity, fallback_used=false, CPU/CUDA selected token IDs, kernel and transfer stats, quality gate result, speedup_claim=false, and bitnet_packed_i2s_qk256_proof=false. |
