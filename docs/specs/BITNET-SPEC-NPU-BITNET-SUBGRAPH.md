# BITNET-SPEC-NPU-BITNET-SUBGRAPH

Status: draft
Proposal: `docs/proposals/BITNET-PROP-0007-npu-productization.md`
Plan: `plans/npu/implementation-plan.md`

## Purpose

Define BitNet NPU work without overclaiming. The BitNet NPU lane is initially a
static subgraph / graph-lowering research lane. It is not full BitNet inference
and not packed QK256 decode.

## Already proved or tracked

- Tiny static OpenVINO graph.
- RMSNorm subgraph parity.
- Linear projection subgraph parity.
- FFN/ReLU2 subgraph parity.

## Next candidates

- sub-layernorm
- RoPE static slice
- embedding/gather static fixture
- attention score static fixture
- softmax static fixture
- A×V static fixture
- LM-head static fixture
- prefill block static graph

## Required receipt shape

```json
{
  "bitnet": {
    "subgraph": "rmsnorm|linear_projection|ffn_relu2|...",
    "shape_mode": "static",
    "full_bitnet_inference": false,
    "qk256_decode": false,
    "packed_i2s_direct": false
  },
  "parity": {
    "reference_backend": "cpu",
    "target_backend": "intel-npu-openvino",
    "max_abs_error": 0.0,
    "mean_abs_error": 0.0,
    "tolerance": 0.0001
  }
}
```

## Hard rule

Static BitNet-shaped subgraph parity does not prove full BitNet inference. It
also does not prove packed I2_S × I8_S QK256 decode, native NPU packed kernels,
BitNet acceleration, or full device residency.
