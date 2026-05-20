# BITNET-SPEC-NPU-RESIDENCY

Status: draft
Proposal: `docs/proposals/BITNET-PROP-0007-npu-productization.md`
Plan: `plans/npu/implementation-plan.md`

## Purpose

Prevent partial graph execution from becoming a full-NPU or full-residency
claim. Residency must be per phase and per profile.

## Residency classes

- `none`
- `runtime_detected`
- `static_graph`
- `selected_subgraph`
- `dense_slm_pipeline`
- `warm_resident_pipeline`
- `full_decode_candidate`
- `full_device_resident`

## Phase table

```json
{
  "residency": {
    "weights": "npu|cpu|mixed|unknown",
    "kv_cache": "npu|cpu|mixed|unknown",
    "embedding": "npu|cpu|mixed|unknown",
    "norms": "npu|cpu|mixed|unknown",
    "rope": "npu|cpu|mixed|unknown",
    "attention_scores": "npu|cpu|mixed|unknown",
    "softmax": "npu|cpu|mixed|unknown",
    "av": "npu|cpu|mixed|unknown",
    "mlp": "npu|cpu|mixed|unknown",
    "lm_head": "npu|cpu|mixed|unknown",
    "sampling": "npu|cpu|host_logits_only|unknown"
  }
}
```

## Hard rules

- OpenVINO GenAI pipeline on NPU is not native NPU kernel proof.
- Selected static subgraph parity is not full device residency.
- Warm pipeline reuse is not full residency unless phase placement is recorded.
- Host sampling or host logits handling must be recorded rather than hidden.
