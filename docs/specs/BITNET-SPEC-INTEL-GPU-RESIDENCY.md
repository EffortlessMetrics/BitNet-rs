# BITNET-SPEC-INTEL-GPU-RESIDENCY

Status: proposed
Owner: intel-gpu/product
Created: 2026-05-18
Linked proposal: `docs/proposals/BITNET-PROP-0006-intel-gpu-productization.md`
Linked specs: `docs/specs/BITNET-SPEC-INTEL-GPU-ROUTE-CONTRACT.md`, `docs/specs/BITNET-SPEC-INTEL-GPU-BITNET-QK256.md`, `docs/specs/BITNET-SPEC-INTEL-GPU-PERFORMANCE.md`
Linked ADRs: n/a
Linked plan: `plans/intel-gpu/implementation-plan.md`
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: Defines residency classes; no residency promotion.
Policy impact: No exception.

## Purpose

Partial acceleration must not become "full GPU" by implication. Residency is a
named-phase claim and must be reported phase by phase.

## Residency classes

```text
none
kernel_smoke
qk256_linears_only
bitnet_trusted_partial
dense_graph_runtime
support_ops_partial
decode_full
full_device_resident
```

## Phase table

Receipts that discuss residency must include this shape or an equivalent
versioned table:

```json
{
  "residency": {
    "weights": "gpu|cpu|mixed|unknown",
    "qk256_linears": "gpu|cpu|mixed|unknown",
    "dense_linears": "gpu|cpu|mixed|unknown",
    "embedding": "gpu|cpu|mixed|unknown",
    "lm_head": "gpu|cpu|mixed|unknown",
    "kv_cache": "gpu|cpu|mixed|unknown",
    "rmsnorm": "gpu|cpu|mixed|unknown",
    "rope": "gpu|cpu|mixed|unknown",
    "attention_scores": "gpu|cpu|mixed|unknown",
    "softmax": "gpu|cpu|mixed|unknown",
    "attention_value_mix": "gpu|cpu|mixed|unknown",
    "sampling": "gpu|cpu|host_logits_only|unknown"
  }
}
```

## Hard rules

- QK256 linears on A770 are trusted partial acceleration, not full residency.
- OpenVINO GPU LLMPipeline output is dense graph/runtime proof, not native
  OpenCL residency.
- `resident_proven` proves only the named phase or operation.
- `complete` requires every required operation, residency phase, and server gate
  for the selected route.
