# Model-family foundation status

This tracker is separate from `docs/tracking/bitnet-alignment/` so model-family planning does not become mixed with CPU, GPU, NPU, or BitNet backend proof items.

## Current state

- MF-001 is ready: the documentation structure, schemas, status values, source index, and receipt templates are in place.
- MF-002 is proposed: architecture glossary and implementation lanes define shared vocabulary and claim boundaries.
- MF-003 is proposed: 5070 Ti-class hardware-fit tiers separate locally testable, partial/offload, and design-only targets.

## First practical model-family targets

1. Qwen3.5 0.8B or 2B text-only GGUF one-token proof.
2. Gemma 4 E2B/E4B text-only quantized one-token proof.
3. OpenAI privacy-filter token-classification CPU/ONNX smoke.

## Design-only targets

Kimi K2.6, GLM-5.1, DeepSeek V4 Flash/Pro, and the full Mistral Medium 3.5 path are design-only on current local hardware until external or local receipts prove a narrower claim.
