# Model Hardware Fit for 5070 Ti-Class 16 GB GPU

The current largest local device class is treated as a 5070 Ti-class 16 GB GPU. This policy prevents large-model documentation from becoming local execution claims.

| Local tier | Expected proof shape | Families |
|---|---|---|
| `tier_a_local` | Small/quantized local smoke or CPU/GPU offload. | Qwen3.5 0.8B/2B/4B/9B; Gemma 4 E2B/E4B text-only; OpenAI privacy-filter. |
| `tier_b_partial` | Reduced-context, offload, quantized, or one-token proof only. | Qwen3.6 27B; Qwen3.5 27B/35B-A3B; Gemma 4 31B/26B-A4B quantized; Mistral Medium 3.5 external/offload reference only. |
| `tier_c_design_only` | Documentation, prompt/tokenizer/loader notes, and external-reference receipt shapes only. | Kimi K2.6; GLM-5.1; DeepSeek V4; Mistral Medium 3.5 full path. |

No document in this tree claims full local inference, full context, performance, or quality on this GPU unless a named receipt backs that exact claim.
