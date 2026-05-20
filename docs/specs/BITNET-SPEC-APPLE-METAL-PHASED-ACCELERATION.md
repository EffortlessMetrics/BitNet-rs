# BITNET-SPEC-APPLE-METAL-PHASED-ACCELERATION

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0005 Apple Silicon productization](../proposals/BITNET-PROP-0005-apple-silicon-productization.md)
Linked specs: [Apple Silicon route contract](BITNET-SPEC-APPLE-SILICON-ROUTE-CONTRACT.md)
Linked ADRs: n/a
Linked plan: [Apple Silicon implementation plan](../../plans/apple-silicon/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no support promotion; phase-scoped Metal proof only
Policy impact: no policy exception

## Purpose

Keep Apple Metal work honest and useful by proving explicit kernels/subgraphs
with CPU reference parity and receipts before any generation contribution or
full route claim.

## Phases

1. `metal_visibility`;
2. `tiny_compute_smoke`;
3. `tiny_add_cpu_parity`;
4. `i2s_packed_parity`;
5. `i2s_prefill_projection_fixture`;
6. `i2s_projection_residual_subgraph`;
7. `dense_prefill_linear_fixture`;
8. `dense_prefill_qkv_fixture`;
9. `real_generation_contribution_candidate`;
10. `full_route_candidate`.

## Required phase receipt fields

```json
{
  "requested_backend": "apple-m4-metal",
  "selected_backend": "apple-m4-metal",
  "runtime_api": "metal",
  "fallback_used": false,
  "kernel_id": "tiny_metal_i2s_prefill_contribution",
  "phase_scope": "prefill_projection_fixture",
  "full_autoregressive_decode": false,
  "cpu_reference_tokens_match": true,
  "decoded_text_match": true,
  "full_metal_inference_claimed": false,
  "speedup_claim": false
}
```

## Promotion rule

A Metal phase may enter real generation only if CPU-only greedy output before
and after the phase matches generated token IDs and decoded text. Until a full
route candidate proves end-to-end autoregressive decode on Metal with fallback
false, Metal receipts must set `full_metal_inference_claimed = false`.

## Not-claims

Metal visibility is not Metal execution. CPU fallback is not Metal execution.
MPSGraph smoke is not native Metal proof. Metal subgraph parity is not full
Metal inference. Phase-local timing is not a broad speedup claim unless a later
spec explicitly allows and proves that claim.
