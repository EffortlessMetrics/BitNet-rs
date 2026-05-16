# M3 MacBook Air Backend Visibility Preflight

Work item: `M3MBA-016`

This report defines the bounded M3 Air backend visibility preflight contract.
It is a no-model receipt shape for recording Apple runtime visibility before
any dense SLM or BitNet candidate execution.

## Receipt Scope

The preflight may record:

- `machine_id = apple-m3-macbook-air`
- `artifact_kind = backend_visibility_preflight`
- `requested_backend = apple-m3-air-metal` or `apple-m3-air-mpsgraph`
- `selected_backend` only when the matching M3 Air runtime is visible
- `runtime_api = metal` or `mpsgraph`
- resolved Apple device facts such as chip, GPU cores, and unified memory
- Metal visibility and MPSGraph graph/API visibility
- fallback status and fallback reason

The preflight must not record model downloads, model loads, model inference,
speedups, timing claims, Neural Engine execution, full Metal inference, or
MPSGraph model inference.

## Claim Boundary

The shared `AppleVisibilityClaimBoundary::bounded_preflight()` contract fixes
these fields to false:

- `model_downloaded`
- `model_loaded`
- `model_inference`
- `metal_inference_claimed`
- `mpsgraph_model_inference_claimed`
- `neural_engine_claimed`
- `performance_claimed`

Validation rejects any preflight receipt that flips one of those fields to
true. Validation also rejects generic `metal` or `mpsgraph` backend aliases for
`apple-m3-macbook-air`; M3 Air receipts must use explicit M3 Air labels.

## Runtime Meaning

`apple-m3-air-metal` visibility means the host can expose a Metal runtime under
the M3 Air identity. It does not prove a BitNet kernel, a dense SLM answer, or a
full Metal inference path.

`apple-m3-air-mpsgraph` visibility means the preflight can record MPSGraph
graph/API visibility under the M3 Air identity. It does not prove Neural Engine
execution and does not prove MPSGraph model inference.

## Next Use

This contract is intended to be consumed by later local M3 Air evidence runs and
CLI/reporting paths. Those later items can emit concrete receipt files under a
target or campaign evidence directory, but generic required CI must remain
no-model and no-download.
