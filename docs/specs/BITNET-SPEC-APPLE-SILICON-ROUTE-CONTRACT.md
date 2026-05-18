# BITNET-SPEC-APPLE-SILICON-ROUTE-CONTRACT

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0005 Apple Silicon productization](../proposals/BITNET-PROP-0005-apple-silicon-productization.md)
Linked specs: n/a
Linked ADRs: n/a
Linked plan: [Apple Silicon implementation plan](../../plans/apple-silicon/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no support promotion; route labels and receipt rules only
Policy impact: no policy exception

## Purpose

Define Apple Silicon backend labels, proof-family IDs, and receipt fields so
CPU/NEON, Metal, MPSGraph, Neural Engine, MacBook, dense SLM, and BitNet
evidence cannot be conflated.

## Route IDs

| Proof family | Requested/selected backend label | Runtime API | Machine scope | Model family |
| --- | --- | --- | --- | --- |
| `apple_m4_cpu_neon_dense_slm` | `apple-m4-cpu-neon` | `cpu` | `apple-m4-mac-mini` | `dense_slm` |
| `apple_m4_cpu_neon_bitnet` | `apple-m4-cpu-neon` | `cpu` | `apple-m4-mac-mini` | `bitnet` |
| `apple_m4_metal_phase` | `apple-m4-metal` | `metal` | `apple-m4-mac-mini` | phase-specific |
| `apple_m4_mpsgraph_reference` | `apple-m4-mpsgraph` | `mpsgraph` | `apple-m4-mac-mini` | reference/graph |
| `apple_m4_neural_engine_research` | future explicit label | `neural_engine` only if receipt-proven | `apple-m4-mac-mini` | research |
| `apple_macbook_cpu_neon_bitnet` | MacBook CPU/NEON route | `cpu` | `apple-silicon-macbook` | `bitnet` |
| `apple_macbook_metal_phase` | MacBook Metal route | `metal` | `apple-silicon-macbook` | phase-specific |

## Required receipt fields

Apple route receipts must include enough information to reject aliasing and
fallback confusion. Minimal fields are:

```json
{
  "requested_backend": "apple-m4-cpu-neon",
  "selected_backend": "apple-m4-cpu-neon",
  "runtime_api": "cpu",
  "machine_id": "apple-m4-mac-mini",
  "fallback_used": false,
  "model_family": "dense_slm|bitnet",
  "proof_family": "apple_m4_cpu_neon_dense_slm",
  "metal_proof": false,
  "mpsgraph_proof": false,
  "neural_engine_proof": false,
  "broad_apple_silicon_claim": false
}
```

## Hard rules

- `apple-m4-metal` cannot alias to CPU/NEON, MPSGraph, or Neural Engine.
- `apple-m4-mpsgraph` cannot count as native Metal proof.
- `apple-m4-cpu-neon` cannot count as acceleration.
- MacBook receipts must use a different `machine_id` and `proof_family` from M4
  Mac Mini receipts.
- `fallback_used = true` invalidates backend-specific proof for the requested
  backend unless the proof is explicitly a fallback-behavior test.
- A receipt with `runtime_api = "cpu"` cannot set `metal_proof = true`.
- A receipt with `runtime_api = "mpsgraph"` cannot set `neural_engine_proof = true`
  unless the resolved target is independently receipt-backed.
- No single M4 Mac Mini receipt may set `broad_apple_silicon_claim = true`.

## Failure behavior

Unsupported or unavailable backends must fail with a clear selected-backend and
fallback receipt. Silent fallback is not proof. Operator-facing output must make
requested backend, selected backend, runtime API, and fallback status visible.
