# Qwen3.6 Source Map

Status: draft
Owner: BitNet-rs contributors
Created: 2026-05-19
Linked proposal: docs/proposals/BITNET-PROP-0017-qwen36-modern-dense-model-family.md
Linked specs: docs/specs/BITNET-SPEC-QWEN36-*
Linked ADRs: n/a
Linked plan: plans/qwen36/implementation-plan.md
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: candidate registration only
Policy impact: claim-boundary enforcement

Qwen3.6 is tracked as a separate modern dense/multimodal/agentic family lane.

## Claim boundary

- Not a BitNet I2_S/TL lane.
- Not inheriting Qwen2.5 or Qwen3 0.6B dense SLM proofs.
- External engines (Transformers/vLLM/SGLang/Ollama/MLX/API) are reference-only until native receipts exist.
- Initial status is registered candidate only.
