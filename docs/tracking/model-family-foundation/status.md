# Model-family foundation status

Updated: 2026-05-07

This tracker is separate from the BitNet/hardware alignment tracker. It records model-family documentation, implementation lanes, status values, hardware-fit tiers, and receipt requirements without changing runtime behavior.

## Current queue

| Item | State | Notes |
|---|---|---|
| MF-001 | ready | Add model docs tree, schemas, source index, statuses, and receipt templates. |
| MF-002 | proposed | Add shared architecture glossary and lane docs. |
| MF-003 | proposed | Add 5070 Ti-class local hardware fit policy. |
| GEMMA4-DOC-001 | proposed | Gemma 4 plan; no Gemma 4 loader claim. |
| QWEN35-DOC-001 | proposed | Qwen3.5 small-model-first plan. |
| PRIVACY-DOC-001 | proposed | Token-classification plan separate from generation. |
| QWEN36-DOC-001 | proposed | Qwen3.6 partial/offload design. |
| KIMI26-DOC-001 | proposed | Design-only huge-model rails. |
| GLM51-DOC-001 | proposed | Design-only huge MoE rails. |
| DSV4-DOC-001 | proposed | Design-only DeepSeek V4 rails. |
| MM35-DOC-001 | proposed | Design-only Mistral Medium 3.5 full path. |

## Claim boundary

Docs do not mean loaders exist; loader scaffold does not mean inference works; one-token proof does not mean multimodal, long-context, MoE, speed, or quality support; design-only families remain explicit non-claims.
