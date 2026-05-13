<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| nvidia-5070ti | CUDA-DENSE-051 | #4645 | `cuda/dense-qwen25-one-token-proof` | Refresh or add the real RTX 5070 Ti Qwen2.5 0.5B Q8_0 one-token strict CUDA proof so the receipt records dense_regular_llm_cuda routing, selected RTX 5070 Ti CUDA backend identity, fallback_used=false, CPU/CUDA selected token IDs, kernel and transfer stats, quality gate result, speedup_claim=false, and bitnet_packed_i2s_qk256_proof=false. |
| slm-cpu | SLM-CPU-008W | #4641 | `codex/slm-cpu-008w-tied-head-shared-math-root-cause` | Use the post-SLM-CPU-008U artifact refresh and output-head audit to localize the remaining first-token divergence after the official Qwen3 GGUF is proven to use tied token embeddings rather than a dedicated output.weight head. Prioritize tied embedding logits, final norm input, vocab-index handoff, and shared transformer math; only move backward into RoPE, GQA, MLP ordering, tensor orientation, or Q8_0 dequant if the new evidence points there. This item must not claim answer quality, corpus success, performance, Q4/Q5 expansion, server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5 support, or BitNet QK256 changes. |
