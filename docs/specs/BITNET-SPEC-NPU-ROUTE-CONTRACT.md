# BITNET-SPEC-NPU-ROUTE-CONTRACT

Status: draft
Proposal: `docs/proposals/BITNET-PROP-0007-npu-productization.md`
Plan: `plans/npu/implementation-plan.md`
Campaign: `intel-npu`

## Purpose

Define NPU backend labels, proof-family separation, receipt fields, and hard
non-conflation rules. This spec prevents Intel OpenVINO NPU evidence from being
mistaken for GPU proof, dense SLM proof from being mistaken for BitNet QK256
proof, or detection from being mistaken for execution.

## Route IDs

| Route ID | Scope |
| --- | --- |
| `intel_lunar_lake_openvino_npu_probe` | Intel Lunar Lake OpenVINO NPU runtime visibility only. |
| `intel_lunar_lake_openvino_npu_static_graph` | Selected tiny static OpenVINO graph execution on NPU. |
| `intel_lunar_lake_openvino_npu_bitnet_subgraph` | Selected static BitNet-shaped subgraph parity on NPU. |
| `intel_lunar_lake_openvino_npu_dense_slm` | Dense SLM OpenVINO GenAI candidate route on NPU. |
| `intel_lunar_lake_openvino_npu_warm_resident` | Dense SLM warm/resident exact-profile route on NPU. |
| `apple_neural_engine_research` | Future Apple Neural Engine research; no Intel proof inheritance. |
| `qualcomm_hexagon_research` | Future Qualcomm Hexagon research; no Intel proof inheritance. |
| `amd_ryzen_ai_research` | Future AMD Ryzen AI research; no Intel proof inheritance. |


## Shape and runtime constraints

Intel Lunar Lake OpenVINO NPU route receipts must record shape mode. Full
dynamic autoregressive decode is out of scope for early NPU proof because the
current route is static-shape governed. Static smoke and static subgraph parity
therefore cannot be promoted to full inference.

## Required receipt fields

Receipts for NPU routes must include route/proof-family metadata equivalent to:

```json
{
  "requested_backend": "intel-npu",
  "selected_backend": "intel-npu-openvino",
  "runtime_api": "openvino",
  "runtime_device": "NPU",
  "fallback_used": false,
  "fallback_reason": null,
  "proof_family": "intel_lunar_lake_openvino_npu_dense_slm",
  "model_family": "dense_slm",
  "bitnet_qk256_proof": false,
  "native_npu_kernel_proof": false,
  "full_npu_inference_claim": false
}
```

If `AUTO`, `HETERO`, or multi-device selection is used, the receipt must record
actual execution devices or mark the route as not selected-NPU proof.

## Hard rules

- NPU detection is not NPU execution.
- OpenVINO NPU smoke is not full inference.
- OpenVINO GPU is not NPU proof.
- Arc 140V OpenCL is not NPU proof.
- Intel NPU proof is not Apple Neural Engine proof.
- Intel NPU proof is not Qualcomm Hexagon proof.
- Intel NPU proof is not AMD Ryzen AI proof.
- Dense SLM NPU proof is not BitNet QK256 proof.
- CPU fallback cannot count as NPU execution.
- AUTO/HETERO proof is not selected NPU proof unless execution devices are recorded.
