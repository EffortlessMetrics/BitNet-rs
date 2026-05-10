# Apple M4 SLM Metal Next Phase

`M4-METAL-001` selects the next Apple Metal phase target for the dense SLM
lane. This is a decision record only. It does not add kernels, route
generation through Metal, download models, or claim full `apple-m4-metal`
inference.

## Decision

The next target is a **prefill Q/K/V projection triplet** for the supported
Qwen2.5 0.5B dense SLM path.

The phase consumes the attention-normalized prefill activation for one layer
and computes the three independent attention input projections:

```text
q = hidden_norm @ q_proj
k = hidden_norm @ k_proj
v = hidden_norm @ v_proj
```

The first proof should keep RoPE, attention scores, softmax, V mixing,
output projection, KV-cache writes, decode, sampling, and detokenization on
the CPU side or outside the fixture. That keeps the phase narrow enough to
debug and prevents a successful projection fixture from becoming a full
Metal inference claim.

## Why This Phase

This is the smallest useful expansion beyond the existing dense f32 prefill
linear projection microphase:

- it exercises the real attention projection boundary instead of an isolated
  generic linear layer;
- Q/K/V projections are independent dense matrix products and can be compared
  before mutating KV cache state;
- the phase covers Qwen GQA-sensitive shapes, including rectangular K/V
  outputs;
- the fixture can compare CPU and Metal tensors directly before checking any
  greedy token behavior;
- the rest of the generation path remains the already validated
  `apple-m4-cpu-neon` route.

## CPU Reference Scope

The CPU reference for `M4-METAL-002` should be a deterministic dense f32
fixture first, with metadata-derived Qwen2.5 dimensions recorded in the
receipt. A later artifact-derived fixture may use the verified cached Qwen
GGUF, but the first parity fixture should not require a model download in
ordinary CI.

The CPU reference must compute Q, K, and V separately using the same input
activation, layout, bias policy, and tolerance as the Metal path. A mismatch
in any one of Q, K, or V fails the phase instead of falling back to CPU.

## Expected Shape Contract

For the current Qwen2.5 0.5B dense SLM support surface, the shape contract is:

```text
hidden_size = 896
attention_heads = 14
kv_heads = 2
head_dim = 64
q_dim = 896
kv_dim = 128

input activation: [prefill_tokens, hidden_size]
q_proj weight:    [q_dim, hidden_size]
k_proj weight:    [kv_dim, hidden_size]
v_proj weight:    [kv_dim, hidden_size]
q output:         [prefill_tokens, q_dim]
k output:         [prefill_tokens, kv_dim]
v output:         [prefill_tokens, kv_dim]
```

`M4-METAL-002` should assert these values from the fixture metadata instead of
silently assuming them. If a later supported dense model has different
dimensions, it needs its own phase receipt or a receipt that records the
model-specific dimensions explicitly.

## Parity Method

The first parity fixture should require:

```text
same q output within tolerance
same k output within tolerance
same v output within tolerance
same q/k/v output shapes
same layout interpretation
no CPU fallback for the Metal phase
```

The fixture may also record a greedy-relevant checksum or argmax summary for
each output, but direct tensor parity is the gate. Resident generation routing
is deferred until `M4-METAL-004`, after phase receipt validation exists.

## Receipt Fields

The phase receipt should extend the existing dense Metal phase shape with:

```json
{
  "artifact_kind": "slm_apple_m4_metal_phase",
  "model_family": "qwen2.5",
  "requested_backend": "apple-m4-metal",
  "selected_backend": "apple-m4-metal",
  "runtime_api": "metal",
  "reference_backend": "apple-m4-cpu-neon",
  "rest_of_pipeline_backend": "apple-m4-cpu-neon",
  "fallback_used": false,
  "execution_phase": "prefill_qkv_projection",
  "phase_scope": "qwen2_5_dense_prefill_qkv_projection_fixture",
  "kernel_family": "dense_f32",
  "layout_source": "fixture_dense_f32_row_major",
  "prefill_tokens": 0,
  "hidden_size": 896,
  "attention_heads": 14,
  "kv_heads": 2,
  "head_dim": 64,
  "q_dim": 896,
  "kv_dim": 128,
  "parity": {
    "q_matches_cpu_reference": true,
    "k_matches_cpu_reference": true,
    "v_matches_cpu_reference": true,
    "max_abs_error": 0.0,
    "mean_abs_error": 0.0
  },
  "timing": {
    "cpu_reference_ms": 0.0,
    "metal_phase_ms": 0.0,
    "metal_q_ms": 0.0,
    "metal_k_ms": 0.0,
    "metal_v_ms": 0.0,
    "dispatch_readback_ms": 0.0,
    "timing_delta_ms": 0.0,
    "speedup_claim": false
  }
}
```

Timing is phase-local. It must not be reported as full-pipeline speedup.

## Fallback Rules

- If Metal is unavailable, the live Metal proof is skipped or fails as a
  missing capability; it must not emit a passing CPU fallback receipt.
- If Q, K, or V mismatch, the phase fails and blocks the claim.
- Non-Metal phases remain explicitly routed to `apple-m4-cpu-neon`.
- Generic CI may validate schema and CPU fixture construction without running
  live Metal dispatch.

## Explicit Deferrals

The following remain out of scope for this item:

- adding a new Metal kernel;
- routing Q/K/V through resident generation;
- RoPE, attention-score, softmax, V-mix, output projection, or MLP Metal
  routing;
- KV-cache mutation on Metal;
- decode-loop Metal routing;
- full `apple-m4-metal` model inference;
- Neural Engine or MPSGraph execution;
- QK256 or BitNet claims;
- broad M4 performance claims.

## Allowed Claim

After `M4-METAL-001`, the project may claim only:

```text
The next dense SLM Metal phase target is selected: a prefill Q/K/V projection
triplet with CPU/Metal parity and phase-local receipts required before any
routing or acceleration claim.
```

It must not claim that Apple Metal runs dense SLM inference, improves answer
speed, or covers the full answer path.
