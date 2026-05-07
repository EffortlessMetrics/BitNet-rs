# 5070 Ti-class hardware fit

This policy assumes the largest local accelerator is a 5070 Ti-class 16 GB GPU. Memory estimates are planning inputs and exclude KV cache, allocator overhead, runtime buffers, multimodal projector memory, and software overhead unless explicitly stated.

## Tier A: locally testable soon

- Gemma 4 E2B/E4B text-only quantized proof plans.
- Qwen3.5 0.8B/2B/4B/9B text-only GGUF proof plans.
- OpenAI privacy-filter CPU/ONNX token-classification smoke plans.

## Tier B: partial/offload possible

- Qwen3.6 27B text-only reduced-context/offload design.
- Qwen3.5 27B/35B-A3B offload or one-token proof only.
- Gemma 4 31B and 26B-A4B quantized/offload plans.
- Mistral Medium 3.5 only with external/offload reference receipts.

## Tier C: design-only locally

- Kimi K2.6.
- GLM-5.1.
- DeepSeek V4 Flash/Pro.
- Full local Mistral Medium 3.5 128B.

No tier implies quality, speed, long-context, multimodal, or full-inference support. Those claims require receipts.
