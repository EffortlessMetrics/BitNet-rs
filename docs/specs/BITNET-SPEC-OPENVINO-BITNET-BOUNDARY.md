# BITNET-SPEC-OPENVINO-BITNET-BOUNDARY: OpenVINO and BitNet Proof Boundary

Status: draft
Owner: intel/openvino
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0004](../proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md)
Linked specs: [BITNET-SPEC-OPENVINO-ROUTE-CONTRACT](BITNET-SPEC-OPENVINO-ROUTE-CONTRACT.md), [BITNET-SPEC-OPENVINO-DENSE-SLM](BITNET-SPEC-OPENVINO-DENSE-SLM.md), [BITNET-SPEC-OPENVINO-ROUTE-PROMOTION](BITNET-SPEC-OPENVINO-ROUTE-PROMOTION.md)
Linked ADRs: n/a
Linked plan: [OpenVINO Lunar Lake implementation plan](../../plans/openvino-lunar-lake/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no promotion; defines BitNet claim boundaries
Policy impact: no policy exception

## Purpose

Define the boundary between OpenVINO dense SLM evidence and BitNet QK256/I2_S
evidence on Lunar Lake. This spec prevents dense Qwen OpenVINO receipts,
OpenVINO GPU receipts, NPU receipts, or static graph smokes from being treated
as BitNet packed-kernel, full-inference, or acceleration proof.

This spec does not run inference, promote routes, claim BitNet execution on
Arc/NPU, claim QK256 decode on accelerators, or change BitNet semantics.

## Proof Families

Use separate proof families for separate claims:

| Proof family | May prove | Must not prove |
| --- | --- | --- |
| `bitnet_cpu_reference` | 258V CPU BitNet strict reference path | Arc/NPU execution, dense SLM quality |
| `bitnet_qk256_i2s_cpu` | CPU QK256/I2_S/I8_S semantic and kernel evidence | GPU/NPU acceleration |
| `openvino_dense_slm_cpu` | OpenVINO dense SLM CPU route | BitNet QK256/I2_S |
| `openvino_dense_slm_gpu_arc140v` | Dense SLM OpenVINO GPU route | native OpenCL or BitNet QK256 |
| `openvino_dense_slm_npu` | Dense SLM OpenVINO NPU route | native NPU kernels, dynamic BitNet decode, QK256 |
| `openvino_bitnet_subgraph_reference` | selected static BitNet-shaped subgraph parity | full BitNet inference, decode loop, speedup |
| `arc140v_native_opencl_kernel` | native OpenCL kernel parity | OpenVINO GPU proof, full BitNet inference |
| `openvino_model_server` | exact-profile server endpoint proof | broad server readiness, BitNet QK256 unless exact route proves it |

Receipts must not use one proof family as evidence for another without an
explicit bridge spec and route-profile review.

## Required Boundary Fields

OpenVINO dense SLM receipts must include explicit negative claim fields:

```json
{
  "model_family": "qwen",
  "proof_family": "openvino_dense_slm_gpu_arc140v",
  "bitnet_qk256_proof": false,
  "bitnet_i2s_proof": false,
  "bitnet_full_inference_proof": false,
  "native_opencl_proof": false,
  "native_npu_kernel_proof": false,
  "acceleration_claim": false,
  "does_not_prove": [
    "BitNet QK256/I2_S behavior",
    "full BitNet inference on Arc or NPU",
    "native OpenCL execution",
    "NPU dynamic decode"
  ]
}
```

BitNet receipts must identify whether they are CPU reference, static subgraph,
native accelerator kernel, or full-inference candidates.

## BitNet CPU Truth Plate

The current BitNet truth plate is the 258V CPU reference bundle and its linked
evidence:

- prompt/token/generated-token boundaries;
- bitnet.cpp reference boundaries;
- scalar/AVX2 answer parity;
- QK256/I2_S/I8_S semantic checks;
- output-head/logit audits;
- transformer-layer parity ladder;
- I2_S GEMV/GEMM tuning receipts;
- fallback status.

Accelerator BitNet work must compare against that CPU reference plate. Dense SLM
receipts cannot replace it.

## Dense SLM Separation

Dense SLM success may inform user-facing Lunar Lake usability, but it is not
BitNet proof.

Examples:

- Qwen2.5 OpenVINO GPU passing corpus v2 does not prove BitNet QK256 decode.
- Qwen2.5 NPU hot decode timing does not prove BitNet NPU dynamic decode.
- OpenVINO CPU dense SLM timing does not prove BitNet I2_S GEMV/GEMM speed.
- Dense SLM route promotion does not promote `bitnet_strict_reference`.

Any receipt that uses dense SLM evidence to claim BitNet behavior must fail
validation.

## OpenVINO BitNet Subgraph Boundary

`openvino_bitnet_subgraph_reference` may prove only a bounded static subgraph
against the CPU reference:

```json
{
  "proof_family": "openvino_bitnet_subgraph_reference",
  "subgraph": "rmsnorm|linear_projection|ffn_relu2|attention_prefill_candidate",
  "shape_mode": "static",
  "runtime_api": "openvino_runtime",
  "runtime_device": "NPU|GPU.0|CPU",
  "cpu_reference_receipt": "<path>",
  "fallback_used": false,
  "max_abs_error": 0.0,
  "mean_abs_error": 0.0,
  "full_bitnet_inference_claim": false,
  "qk256_decode_claim": false
}
```

Static subgraph proof must not claim:

- tokenizer execution;
- prompt rendering;
- sampler correctness;
- dynamic decode loop;
- KV-cache behavior;
- packed QK256 decode;
- full model inference;
- speedup unless timing and parity are both present for the exact subgraph.

## Native OpenCL Boundary

Arc 140V native OpenCL evidence is separate from OpenVINO GPU evidence.

Native OpenCL receipts must record:

- runtime API `opencl`;
- selected Arc 140V device identity;
- kernel name;
- input/output shapes;
- CPU reference receipt;
- tolerance;
- fallback status;
- timing if any speed claim is made.

OpenVINO GPU receipts must not set `native_opencl_proof=true`. Native OpenCL
receipts must not imply OpenVINO GenAI route readiness.

## NPU Boundary

NPU OpenVINO dense SLM evidence is not native NPU kernel proof and not BitNet
QK256 proof. NPU BitNet claims require:

- exact static subgraph definition or later full-route spec;
- CPU reference comparison;
- selected NPU device identity;
- `fallback_used=false`;
- static-shape proof for current subgraph work;
- no beam search, no parallel sampling, and no unsupported dynamic decode claim.

Packed QK256 decode on NPU remains blocked until a dedicated spec, model
artifact contract, graph/kernel proof, parity receipt, and timing receipt exist.

## Shared BitNet Semantic Intake

When another hardware lane lands a shared BitNet semantic fix, Lunar Lake must
rerun affected CPU reference evidence before changing route policy. Examples:

- tokenizer special-token handling;
- prompt template or generation prompt policy;
- output-head or vocab-index mapping;
- RMSNorm or sub-layernorm placement;
- ReLU2 or FFN arithmetic;
- RoPE or attention/value mixing;
- KV-cache position handling;
- QK256/I2_S/I8_S scale order;
- sampler or stop behavior.

Required reruns depend on the changed surface but may include:

```text
BitNet CPU answer corpus
scalar/AVX2 parity
first-token classifier
phase receipts
operator readiness
route comparison
```

Dense SLM receipts do not need to rerun for BitNet-only semantic fixes unless a
shared tokenizer, sampler, or operator routing surface changes.

## Rejection Examples

| Evidence | Required decision |
| --- | --- |
| Qwen OpenVINO GPU corpus v2 passes | No BitNet QK256 claim |
| NPU dense SLM warm decode is fast | No BitNet NPU decode claim |
| OpenVINO RMSNorm subgraph parity passes | No full BitNet inference claim |
| Arc OpenCL vector add parity passes | No OpenVINO GPU or BitNet kernel claim |
| BitNet CPU parity passes | No Arc/NPU acceleration claim |
| Dense SLM server endpoint works | No BitNet server readiness claim |

## Acceptance

This spec is complete when it defines:

1. Separate proof families for CPU BitNet, dense SLM OpenVINO, BitNet subgraph,
   native OpenCL, NPU, and server evidence.
2. Required negative-claim fields for dense SLM OpenVINO receipts.
3. CPU BitNet truth-plate requirements for accelerator comparison.
4. Static subgraph, native OpenCL, and NPU boundaries.
5. Shared BitNet semantic intake rerun rules.
6. Rejection examples that prevent dense SLM evidence from becoming BitNet
   proof.
