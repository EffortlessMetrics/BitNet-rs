<!-- GENERATED: do not edit by hand. Run cargo run -p xtask --no-default-features -- campaign generate. -->
# Active Campaign PRs

| Campaign | Item | PR | Branch | Notes |
|---|---|---:|---|---|
| nvidia-5070ti | CUDA-DENSE-036 | #4331 | `codex/cuda-dense-036-all-layer-plan` | Define the governed dense GGUF all-layer execution-plan receipt contract after CUDA-DENSE-035 proved integrated layer-0 CUDA parity, requiring the future implementation to inspect every Qwen-family transformer layer, report per-layer routed op counts and any graph differences, preserve dense_regular_llm_cuda route separation, explicitly list model-boundary gaps such as token embeddings, final norm, LM head/logits, KV cache, and sampling, and keep dense GGUF inference, Qwen one-token/decode/chat, speedup, persistent/full residency, BitNet packed proof, tokenizer, loader, transformer runtime, QK256, server, and CUDA kernel math claims false. |
