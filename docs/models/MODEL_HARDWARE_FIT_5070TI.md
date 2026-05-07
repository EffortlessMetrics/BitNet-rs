# Hardware fit policy: 5070 Ti-class 16 GB GPU

The largest local GPU target for this planning layer is treated as a 5070 Ti-class 16 GB device. Memory estimates in family docs are planning inputs and do not include KV cache, runtime allocator overhead, multimodal projector memory, host offload pressure, or long-context cache growth unless explicitly stated.

## Tier A: locally testable soon

- Gemma 4 E2B/E4B text-only quantized paths.
- Qwen3.5 0.8B, 2B, 4B, and 9B quantized text-only paths.
- OpenAI privacy-filter CPU/ONNX/token-classification smoke.

## Tier B: partial/offload possible

- Qwen3.6 27B text-only reduced-context/offload proof.
- Qwen3.5 27B and 35B-A3B reduced-context/offload proof.
- Gemma 4 31B and 26B-A4B quantized/offload proof.
- Mistral Medium 3.5 only with external/offload reference receipts.

## Tier C: design-only locally

- Kimi K2.6.
- GLM-5.1.
- DeepSeek V4 Flash/Pro.
- Mistral Medium 3.5 128B full path.

Design-only docs are not local inference, throughput, quality, full-context, or residency claims.

