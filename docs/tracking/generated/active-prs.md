<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| intel-258v-platform | CPU258V-023 | #4329 | `codex/lunar-lake/CPU258V-023-output-head-logits-audit` | Audit the 258V CPU output-head and logits-index boundary by recording embedding tensor identity, selected output head or tied-head policy, vocab size, logits vector length, EOS/stop IDs, top-k token IDs/decoded strings, and any mismatch without claiming answer quality, speed, Arc/NPU execution, or full model correctness. |
| nvidia-5070ti | CUDA-DENSE-037 | #4333 | `codex/cuda-dense-037-all-layer-plan-impl` | Implement the governed dense GGUF all-layer execution-plan receipt command after CUDA-DENSE-036, inspecting every Qwen-family transformer layer from descriptors, reporting per-layer routed op counts and graph differences, preserving dense_regular_llm_cuda route separation, explicitly listing token embedding, final norm, LM head/logits, KV cache, and sampling model-boundary gaps, and keeping dense GGUF inference, Qwen one-token/decode/chat, speedup, persistent/full residency, BitNet packed proof, tokenizer, loader, transformer runtime, QK256, server, and CUDA kernel math claims false. |
