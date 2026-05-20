# BITNET-SPEC-NPU-STATUS-SURFACE

Status: draft
Proposal: `docs/proposals/BITNET-PROP-0007-npu-productization.md`
Plan: `plans/npu/implementation-plan.md`

## Purpose

Make NPU status visible to users and reviewers without overclaiming. Status
surfaces must show proof level, route, fallback status, not-claims, and next
proof required.

## User commands

```bash
bitnet npu doctor --format json
bitnet model status --device intel-npu-openvino
bitnet receipts explain <receipt>
bitnet lunar-lake routes --format json
```

## Required status fields

Status should show:

- NPU detected,
- OpenVINO NPU visible,
- driver/compiler,
- static graph smoke,
- BitNet subgraph parity,
- dense SLM route status,
- cold/cache/warm timing,
- quality status,
- speed/power status,
- residency status,
- not-claims,
- next proof required.

## Example JSON shape

```json
{
  "intel_npu": {
    "detected": true,
    "openvino_visible": true,
    "selected_backend": "intel-npu-openvino",
    "driver_version": "...",
    "compiler_version": "...",
    "latest_static_smoke": "pass",
    "latest_subgraph_parity": "pass",
    "dense_slm_status": "candidate",
    "fallback_used": false,
    "not_claimed": [
      "full_bitnet_inference",
      "packed_qk256_decode",
      "native_npu_packed_kernels",
      "broad_speedup",
      "full_residency"
    ]
  }
}
```

## Receipts explanation language

For static BitNet-shaped subgraph parity, `receipts explain` should communicate:

```text
This is an Intel NPU OpenVINO static subgraph parity receipt.
This is not full BitNet inference.
This is not packed QK256 decode.
This is not native NPU kernel proof.
This is not OpenVINO GPU proof.
Fallback was false.
```
