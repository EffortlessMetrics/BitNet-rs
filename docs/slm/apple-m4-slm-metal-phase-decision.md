# Apple M4 SLM Metal Phase Decision

`SLM-M4-007` selects the first safe Apple Metal contribution for the validated dense SLM lane. This is a decision record only; it does not add Metal kernels, route inference through Metal, or claim full `apple-m4-metal` model inference.

## Decision

The first safe Metal target is a **prefill-only dense linear projection microphase** for the validated Qwen2.5 0.5B Q8_0 artifact.

This phase is selected because it is easier to bound than autoregressive decode:

- prefill shapes are known before dispatch;
- CPU-only greedy output already has quality and determinism receipts;
- projection parity can be checked before changing generated text;
- KV-cache mutation and sampling stay on CPU until a later proof explicitly moves them.

## Required Future Proof

A future Metal implementation PR must prove all of the following before making a user-facing acceleration claim:

```text
CPU-only greedy corpus receipt
CPU+Metal-phase greedy corpus receipt
same generated token IDs, or a recorded divergence that blocks the claim
Metal phase receipt with fallback_used=false
rest-of-pipeline backend recorded as apple-m4-cpu-neon
layout handling recorded: direct Q8_0 consume vs conversion/dequantization
phase timing recorded without broad performance language
```

The Metal phase receipt must include:

```json
{
  "artifact_kind": "slm_apple_m4_metal_phase",
  "model_family": "qwen",
  "requested_backend": "apple-m4-metal",
  "selected_backend": "apple-m4-metal",
  "runtime_api": "metal",
  "execution_phase": "prefill_linear_projection",
  "reference_backend": "apple-m4-cpu-neon",
  "fallback_used": false,
  "cpu_pipeline_for_remaining_phases": true,
  "greedy_token_ids_match_cpu_reference": true
}
```

## Explicit Deferrals

The following remain deferred:

- full `bitnet run --device apple-m4-metal` or SLM `apple-m4-metal` model inference;
- decode-loop Metal routing;
- KV-cache mutation on Metal;
- MPSGraph or Neural Engine claims;
- QK256 on Apple Silicon;
- broad M4 performance claims.

## Claim Boundary

After this decision, the campaign may claim only:

```text
A safe first Metal phase route has been selected for future proof work.
```

It must not claim:

```text
Apple Metal runs SLM inference.
Metal is faster.
Neural Engine is used.
QK256 is supported on Apple Silicon.
```

## M4-PROD-005 Proof Shape

`M4-PROD-005` implements the first proof as a dense f32 prefill linear projection
microphase. The proof uses a deterministic fixture, runs the same projection
through the CPU reference and native Metal, compares greedy argmax token IDs, and
emits a split receipt:

```text
slm_pipeline.selected_backend = apple-m4-cpu-neon
metal_phase.selected_backend = apple-m4-metal
metal_phase.runtime_api = metal
metal_phase.fallback_used = false
metal_phase.execution_phase = prefill_linear_projection
layout.consumes_dense_f32_directly = true
layout.dequantizes_before_compute = false
parity.greedy_token_ids_match_cpu_reference = true
```

Run the live Apple M4 proof explicitly:

```bash
BITNET_RUN_M4_METAL_DENSE_PREFILL_LINEAR=1 \
BITNET_M4_METAL_DENSE_PREFILL_LINEAR_RECEIPT=ci/hardware/apple-m4-mac-mini/<date>/metal-dense-prefill-linear.json \
cargo test --locked -p bitnet-kernels \
  --no-default-features \
  --features metal \
  --test metal_tiny_smoke \
  tiny_m4_metal_dense_prefill_linear_projection_matches_cpu_reference_when_enabled \
  -- --nocapture
```

Validate the emitted receipt:

```bash
cargo run --locked -p bitnet-cli \
  --no-default-features \
  --features cpu,full-cli \
  -- mac receipts-check ci/hardware/apple-m4-mac-mini/<date>/metal-dense-prefill-linear.json --json
```

This proof still does not claim full `apple-m4-metal` inference, decode-loop
Metal routing, KV-cache behavior on Metal, Neural Engine execution, MPSGraph
execution, QK256 on Apple Silicon, or broad performance.
