<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | CPU258V-024 | #4342 | `codex/lunar-lake/CPU258V-024-observed-logits-length` | Capture observed runtime logits vector length evidence from the real 258V CPU generation/eval path, compare it with tokenizer/output-head expected vocab length and decoded top-k IDs, and preserve the no answer-quality, no speed, no Arc/NPU, and no full-model-correctness claim boundary. |
| nvidia-5070ti | CUDA-DENSE-038 | #4348 | `codex/cuda-dense-038-model-boundary-fixtures` | Implement governed dense GGUF model-boundary fixture receipts after CUDA-DENSE-037, covering token embedding lookup, final model norm, LM head/output projection, logits shape/hash/top-k diagnostics, and the selected dense_regular_llm_cuda route boundary, while keeping KV cache policy, sampling integration, Qwen one-token/short decode/chat, speedup, persistent/full residency, server readiness, BitNet packed proof, tokenizer behavior, loader behavior, transformer runtime behavior, QK256, BitNet CUDA, and CUDA kernel math claims false. |
