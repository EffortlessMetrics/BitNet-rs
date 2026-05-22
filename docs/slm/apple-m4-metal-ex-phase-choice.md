# M4 Metal Attention-Score Phase Choice

`M4-METAL-EX-001` chooses the next phase target after the completed dense
prefill Q/K/V contribution. This is a decision record only. It does not add a
kernel, change resident generation, download a model, or widen any public M4
expectation envelope.

## Decision

The next target is a **prefill attention-score logits microphase** for the
supported Qwen2.5 dense SLM path.

The phase consumes already computed and RoPE-applied Q and K tensors for one
layer, then computes scaled attention-score logits:

```text
scores[h, q, k] = dot(q_rot[h, q, :], k_rot[kv_head(h), k, :]) / sqrt(head_dim)
```

The first proof should use a deterministic dense f32 fixture. RoPE application,
causal masking, softmax, V mixing, output projection, KV-cache writes, decode,
sampling, detokenization, and all BitNet paths remain outside this phase.

## Why This Phase

The completed Metal evidence already covers dense prefill linear projection,
Q/K/V projection parity, opt-in resident Q/K/V contribution receipts, and
phase-local timing for that Q/K/V contribution. The next narrow boundary is the
attention score calculation because it:

- uses Q and K tensors from the previous phase without mutating KV cache state;
- exercises Qwen grouped-query attention head mapping;
- produces a tensor that can be compared directly against a CPU reference before
  any softmax or generated-token behavior is involved;
- keeps the rest of the answer path explicitly on `apple-m4-cpu-neon`;
- avoids treating dense SLM phase evidence as BitNet evidence.

## CPU Reference Scope

`M4-METAL-EX-002` should implement the CPU reference first as a deterministic
fixture with metadata-derived Qwen2.5 dimensions:

```text
attention_heads = 14
kv_heads = 2
head_dim = 64
prefill_tokens = fixture-defined, at least 4
q input = [prefill_tokens, attention_heads, head_dim]
k input = [prefill_tokens, kv_heads, head_dim]
score output = [attention_heads, prefill_tokens, prefill_tokens]
```

The CPU reference must apply the same head-to-KV-head mapping, scale factor,
layout interpretation, precision policy, and tolerance as the phase under test.
Any mismatch in score shape, head mapping, or logits blocks the phase.

Generated token IDs and text are not required for the fixture-only proof because
the selected phase does not route generation. If a later item wires the phase
into resident generation, that later evidence must record generated token IDs
and text that match the CPU/NEON reference or keep the route disabled.

## Receipt Requirements

The phase receipt should use the existing `slm_apple_m4_metal_phase` family and
add an execution phase for attention-score logits:

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
  "execution_phase": "prefill_attention_scores",
  "phase_scope": "qwen2_5_dense_prefill_attention_scores_fixture",
  "kernel_family": "dense_f32",
  "layout_source": "fixture_dense_f32_qk_rope_applied",
  "prefill_tokens": 4,
  "attention_heads": 14,
  "kv_heads": 2,
  "head_dim": 64,
  "parity": {
    "scores_match_cpu_reference": true,
    "score_shape_matches_cpu_reference": true,
    "head_mapping_matches_cpu_reference": true,
    "max_abs_error": 0.0,
    "mean_abs_error": 0.0
  },
  "timing": {
    "cpu_reference_ms": 0.0,
    "metal_phase_ms": 0.0,
    "dispatch_readback_ms": 0.0,
    "timing_delta_ms": 0.0,
    "speedup_claim": false
  },
  "claim_boundary": {
    "metal_phase_contribution_only": true,
    "cpu_pipeline_for_remaining_phases": true,
    "softmax_on_metal_claimed": false,
    "kv_cache_on_metal_claimed": false,
    "full_metal_inference_claimed": false,
    "bitnet_claimed": false
  }
}
```

Timing is phase-local. It must not be reported as full-answer acceleration or a
public performance envelope.

## Proof Shape

`M4-METAL-EX-002` should be split so ordinary CI stays cheap:

```bash
cargo test --locked -p bitnet-kernels \
  --no-default-features \
  --test <selected-phase-test> \
  dense_prefill_attention_scores_fixture_matches_cpu_reference
```

Live dispatch on the M4 Mac mini should remain opt-in or Mac-runner scoped:

```bash
BITNET_RUN_M4_METAL_DENSE_PREFILL_ATTENTION_SCORES=1 \
BITNET_M4_METAL_DENSE_PREFILL_ATTENTION_SCORES_RECEIPT=ci/hardware/apple-m4-mac-mini/<date>/slm-metal-phases/metal-dense-prefill-attention-scores.json \
cargo test --locked -p bitnet-kernels \
  --no-default-features \
  --features metal-runtime \
  --test <selected-phase-test> \
  dense_prefill_attention_scores_match_cpu_reference_when_enabled \
  -- --nocapture
```

Receipt validation must reject CPU fallback, missing score parity, missing
head-mapping parity, missing phase timing, `speedup_claim=true`, or any claim
that softmax, V mixing, KV cache mutation, decode, sampling, BitNet, QK256,
Neural Engine, MPSGraph, MacBook, or broad Apple Silicon behavior is proven by
this phase.

## Explicit Deferrals

The following remain out of scope:

- resident generation routing for the selected phase;
- causal mask, softmax, V mixing, output projection, MLP, KV-cache mutation,
  decode-loop, sampling, or detokenization work;
- full `apple-m4-metal` model inference;
- BitNet, QK256, Neural Engine, MPSGraph, MacBook, broad Apple Silicon, broad
  quality, broad performance, or speedup claims.

## Allowed Claim

After `M4-METAL-EX-001`, the project may claim only:

```text
The next M4 dense SLM phase target is selected: prefill attention-score logits
with CPU reference parity, fallback-free phase receipts, phase-local timing, and
CPU/NEON for the rest of the answer path required before implementation claims.
```
