# Hardware fit policy: 5070 Ti-class 16 GB GPU

This policy assumes the largest local accelerator is a 5070 Ti-class GPU with 16 GB VRAM. Memory estimates are planning inputs, not benchmarks.

## Tier A: locally testable soon

- Gemma 4 E2B/E4B text-only quantized proof targets.
- Qwen3.5 0.8B, 2B, 4B, and 9B text-only quantized proof targets.
- OpenAI privacy-filter CPU/ONNX/safetensors token-classification smoke.

## Tier B: partial/offload possible

- Qwen3.6 27B text-only reduced-context/offload design target.
- Qwen3.5 27B and 35B-A3B quantized/offload targets.
- Gemma 4 31B and 26B-A4B quantized/offload targets.
- Mistral Medium 3.5 only as external or offload reference unless a future receipt proves otherwise.

## Tier C: design-only locally

- Kimi K2.6.
- GLM-5.1.
- DeepSeek V4 Flash/Pro.
- Mistral Medium 3.5 128B full path.

## Hard boundary

Tier membership is not runtime proof. Design-only docs must not be converted into local execution, speed, quality, long-context, multimodal, or MoE support claims without receipts.
