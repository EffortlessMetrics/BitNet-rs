# Model-family foundation

This area is a documentation and control-plane layer for future model-family support in bitnet-rs. It records model facts, implementation lanes, status gates, local hardware fit, and receipt requirements without adding runtime behavior.

## Claim rule

A family document is not a support claim. Claims only advance through the status values in [MODEL_STATUS_VALUES.md](MODEL_STATUS_VALUES.md) and the machine-readable tracker in `docs/tracking/model-family-foundation/model-status.yaml`.

## First practical non-BitNet targets

1. Qwen3.5 small models for a regular dense local LLM path.
2. Gemma 4 E2B/E4B text-only for a newer hybrid-thinking, multimodal-family decoder with text-only first proof.
3. OpenAI privacy-filter for a non-generative token-classification path.

Huge models are still documented as design-only rails so future agents have prompt, loader, receipt, and non-claim boundaries without implying local proof on a 5070 Ti-class 16 GB GPU.

