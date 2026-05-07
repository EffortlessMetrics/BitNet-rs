# Model-family foundation

This area is the documentation and control-plane layer for future model-family support in bitnet-rs. It records architecture facts, prompt and tokenizer contracts, implementation lanes, receipt requirements, and explicit non-claims before any runtime behavior is added.

The model-family tracker is intentionally separate from the BitNet/hardware tracker. Hardware lanes prove CPU, CUDA, OpenCL, Metal, NPU, and fallback identity; this area proves named model families, variants, modalities, tokenizers, prompt templates, loaders, and receipts.

## Claim rule

A family document is not a support claim. Claims advance only through named statuses in `MODEL_STATUS_VALUES.md` and the machine-readable catalog in `docs/tracking/model-family-foundation/model-status.yaml`.

## First practical lanes

- Qwen3.5 small text-only GGUF models.
- Gemma 4 E2B/E4B text-only quantized proofs.
- OpenAI privacy-filter token classification.

## Design-only lanes

Kimi K2.6, GLM-5.1, DeepSeek V4 Flash/Pro, and the full Mistral Medium 3.5 path are documented as design-only rails for current 5070 Ti-class local hardware.
