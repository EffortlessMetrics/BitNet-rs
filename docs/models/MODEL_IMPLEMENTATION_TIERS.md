# Model implementation tiers

The current local planning target is a 5070 Ti-class 16 GB GPU. Tiers describe what can be planned locally without over-claiming.

| Tier | Meaning | Families |
| --- | --- | --- |
| Tier A: locally testable soon | Reasonable to smoke locally with small/quantized models or CPU/GPU offload. | Gemma 4 E2B/E4B text-only; Qwen3.5 0.8B/2B/4B/9B; OpenAI privacy-filter. |
| Tier B: partial/offload possible | May run with quantization, CPU/RAM offload, reduced context, or single-token proof only. | Qwen3.6 27B; Qwen3.5 27B/35B-A3B; Gemma 4 26B-A4B/31B quantized; Mistral Medium 3.5 only if external/offload reference exists. |
| Tier C: design-only locally | Too large for direct local proof; document architecture, prompt, loader, receipts, and reference commands. | Kimi K2.6, GLM-5.1, DeepSeek V4 Flash/Pro, Mistral Medium 3.5 128B full path. |

Design-only is a positive state: it records rails for future agents while explicitly refusing local inference claims.
