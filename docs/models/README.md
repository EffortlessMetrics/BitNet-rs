# Model Family Foundation

This directory is a documentation and planning control plane for future non-BitNet model-family support. It records architecture facts, prompt/tokenizer contracts, local hardware fit, implementation lanes, receipt requirements, and explicit non-claims before runtime code exists.

The model-family lane is intentionally separate from the BitNet/hardware alignment tracker. Hardware tracker receipts prove devices and BitNet kernels; this area proves named model family, variant, backend, task, tokenizer, prompt template, fallback status, and receipt coverage.

## First practical targets

- Qwen3.5 small models as regular dense local LLM targets.
- Gemma 4 E2B/E4B text-only as newer hybrid/multimodal decoder targets with multimodal future-gated.
- OpenAI privacy-filter as a non-generative token-classification target.

## Design-only targets

Kimi K2.6, GLM-5.1, DeepSeek V4 Flash/Pro, and the full Mistral Medium 3.5 path are design-only locally until external/offload receipts or future hardware prove narrower claims.

## Claim rule

Docs do not mean loaders exist; loaders do not mean inference works; one-token proof does not mean multimodal, long-context, MoE, quality, speed, or broad variant support.
