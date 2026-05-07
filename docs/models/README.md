# Model-family foundation

This directory is a documentation and control-plane layer for future non-BitNet model support. It records source-backed facts, implementation contracts, local hardware fit, receipt requirements, and explicit non-claims before runtime code is added.

The model-family lane is intentionally separate from the BitNet hardware/backend tracker. Backend proof for CPU, CUDA, OpenCL, OpenVINO, Metal, or NPU paths remains in `docs/tracking/bitnet-alignment/`; model-family readiness lives in `docs/tracking/model-family-foundation/`.

## Claim rule

A family document is not a support claim. Claims advance only through named statuses such as `documented`, `design_scaffold`, `loader_scaffold`, `one_token_smoke_tested`, and `receipt_backed`. A claim applies only to the exact family, variant, backend, format, modality, task, and receipt that proved it.

## First practical targets

- Qwen3.5 small text-only GGUF path.
- Gemma 4 E2B/E4B text-only quantized path.
- OpenAI privacy-filter token classification path.

## Design-only rails

Kimi K2.6, GLM-5.1, DeepSeek V4, and the full Mistral Medium 3.5 path are documented as design-only or external-reference targets on a 5070 Ti-class 16 GB local GPU.
