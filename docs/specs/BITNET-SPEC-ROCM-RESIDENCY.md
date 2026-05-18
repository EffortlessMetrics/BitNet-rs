# BITNET-SPEC-ROCM-RESIDENCY: ROCm Residency Claims

Status: proposed
Owner: rocm/productization
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0008](../proposals/BITNET-PROP-0008-amd-rocm-productization.md)
Linked specs: [ROCm route contract](BITNET-SPEC-ROCM-ROUTE-CONTRACT.md), [ROCm performance](BITNET-SPEC-ROCM-PERFORMANCE.md)
Linked ADRs: [BITNET-ADR-0005](../adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md)
Linked plan: [AMD ROCm implementation plan](../../plans/rocm/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: defines residency classes; no residency promotion
Policy impact: no CI policy exception

## Purpose

Prevent partial ROCm acceleration from becoming a "full ROCm" claim. Residency
must be per-phase evidence, not a synonym for one kernel or one tensor op.

## Residency Classes

```text
none
runtime_detected
source_compiled
kernel_smoke
qk256_linears_only
dense_linears_only
decode_partial
warm_session_partial
full_decode_candidate
full_device_resident
```

## Phase Table

```json
{
  "residency": {
    "weights": "rocm|cpu|mixed|unknown",
    "qk256_linears": "rocm|cpu|mixed|unknown",
    "dense_linears": "rocm|cpu|mixed|unknown",
    "embedding": "rocm|cpu|mixed|unknown",
    "kv_cache": "rocm|cpu|mixed|unknown",
    "rmsnorm": "rocm|cpu|mixed|unknown",
    "rope": "rocm|cpu|mixed|unknown",
    "attention_scores": "rocm|cpu|mixed|unknown",
    "softmax": "rocm|cpu|mixed|unknown",
    "attention_value_mix": "rocm|cpu|mixed|unknown",
    "mlp_activation": "rocm|cpu|mixed|unknown",
    "lm_head": "rocm|cpu|mixed|unknown",
    "sampling": "rocm|cpu|host_logits_only|unknown"
  }
}
```

## Hard Rule

```text
QK256 linears on ROCm are not full ROCm residency.
Dense linear kernels on ROCm are not full model residency.
```

Full device residency requires phase evidence for every applicable phase and
must remain independent from speedup and server-readiness claims.
