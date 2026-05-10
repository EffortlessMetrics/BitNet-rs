# Apple M4 SLM Metal Resident Routing Boundary

`M4-METAL-004` records the boundary between the validated dense Qwen-shaped
Q/K/V Metal phase and real resident-session routing.

## Current State

The project has a real Apple M4 Metal proof for the selected phase:

- `M4-METAL-002` added an opt-in live Metal fixture for dense prefill Q/K/V
  projection parity.
- `M4-METAL-003` added a durable checked receipt at
  `ci/hardware/apple-m4-mac-mini/2026-05-10/slm-metal-phases/metal-dense-prefill-qkv.json`.
- `bitnet mac receipts-check` validates that receipt as a phase contribution
  with `selected_backend=apple-m4-metal`, `runtime_api=metal`, and
  `fallback_used=false`.

That evidence is phase-scoped. It does not mean the resident dense SLM
generation path can route Q/K/V through Metal yet.

## Runtime Boundary

The live Metal dispatch helper is currently test-local:

```text
crates/bitnet-kernels/tests/metal_tiny_smoke.rs
```

The dependencies required to dispatch it (`wgpu` and `pollster`) are currently
dev-dependencies for macOS/aarch64 test code, not non-dev runtime dependencies
available to `bitnet mac chat` or `bitnet mac validate`.

Because of that, `M4-METAL-004` must not claim:

```text
Metal participates in resident dense SLM generation.
CPU+Metal resident answers are proven.
Apple Metal accelerates the full SLM answer path.
```

## Required Path

Resident routing needs one more implementation step before it is honest:

```text
M4-METAL-005:
  promote the Q/K/V Metal dispatch from test-only code into a non-dev runtime API
  with target/feature-gated dependencies and CI-safe tests

M4-METAL-006:
  route that runtime API through resident sessions only where parity holds
  and record per-turn phase receipts

M4-METAL-007:
  record phase-local timing deltas without full-pipeline speedup claims
```

`M4-METAL-005` must keep ordinary CI efficient. Generic CI can validate schema,
fixture construction, and CPU-side behavior. Live Apple M4 Metal dispatch should
remain opt-in or Mac-runner scoped.

`M4-METAL-005` implements that first prerequisite as a fixture-scoped runtime
API in `bitnet_kernels::metal::dense_prefill_qkv`, gated behind the opt-in
`metal-runtime` feature so ordinary CI can keep using lightweight `metal`
receipt checks without pulling live dispatch dependencies. That API is
intentionally not wired into `bitnet mac chat`, `bitnet mac validate`, or
resident generation; `M4-METAL-006` owns that routing decision after the runtime
API is available.

## Allowed Claim After M4-METAL-004

```text
The M4 dense SLM Metal resident-routing prerequisite is documented:
the Q/K/V phase is validated, but resident routing waits for a non-dev Metal
runtime API.
```

## Claim Boundary

Dense SLM Metal phase evidence does not prove BitNet quality, QK256 on Apple
Silicon, Neural Engine execution, MPSGraph model inference, full
`apple-m4-metal` inference, or broad M4 performance.
