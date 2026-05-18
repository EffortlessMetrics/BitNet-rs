# BITNET-SPEC-OPENVINO-ROUTE-CONTRACT: OpenVINO Route Contract

Status: proposed
Linked proposal:
[BITNET-PROP-0004](../proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md)
Applies to: OpenVINO dense SLM receipts, OpenVINO subgraph receipts,
OpenVINO server receipts, `bitnet model status`, `bitnet receipts explain`,
future OpenVINO ask/chat/bench/server surfaces

## Purpose

OpenVINO routes must be explicit enough to prevent CPU fallback, AUTO/HETERO
ambiguity, native OpenCL claim leakage, NPU/GPU conflation, and dense-SLM-to-
BitNet proof leakage. This spec defines the route IDs, proof families, receipt
fields, and hard rejection rules required before OpenVINO evidence can support
status docs, model coverage rows, or CLI summaries.

This spec is a route identity contract. It does not promote OpenVINO GPU/NPU,
claim speedup, claim broad answer quality, or claim BitNet QK256 proof.

## Required Route IDs

The governed OpenVINO route IDs are:

```text
openvino_dense_slm_cpu
openvino_dense_slm_gpu_arc140v
openvino_dense_slm_npu
openvino_bitnet_subgraph_reference
openvino_server_exact_profile
```

Receipts may contain campaign-local candidate IDs, but any user-facing summary
must map the evidence to one of these governed route IDs or explicitly mark the
receipt as diagnostic/unmapped.

## Route Identity Table

| Route ID | Required selected device | Runtime API | Proof family | Product role |
| --- | --- | --- | --- | --- |
| `openvino_dense_slm_cpu` | `CPU` with CPU plugin identity | `openvino_genai` or `openvino_runtime` | `openvino_dense_slm_cpu` | dense SLM OpenVINO correctness/control route |
| `openvino_dense_slm_gpu_arc140v` | `GPU.0` or exact Arc 140V GPU device with full name | `openvino_genai` or `openvino_runtime` | `openvino_dense_slm_gpu_arc140v` | exact-profile Arc 140V candidate route |
| `openvino_dense_slm_npu` | `NPU` with Intel AI Boost/NPU identity | `openvino_genai` or `openvino_runtime` | `openvino_dense_slm_npu` | cached/warm/resident NPU candidate route |
| `openvino_bitnet_subgraph_reference` | explicitly selected OpenVINO device for a static subgraph | `openvino_runtime` | `openvino_bitnet_subgraph_reference` | selected static BitNet-shaped subgraph parity |
| `openvino_server_exact_profile` | exact device from server receipt | server runtime plus OpenVINO runtime/API | `openvino_model_server` | exact endpoint/profile server proof |

## Required Receipt Fields

Every strict OpenVINO receipt must include the following route identity fields
or a schema-versioned equivalent:

```json
{
  "requested_backend": "openvino-gpu",
  "selected_backend": "openvino-gpu",
  "runtime_api": "openvino_genai",
  "runtime_device": "GPU.0",
  "resolved_device": "Intel(R) Arc(TM) 140V GPU",
  "fallback_used": false,
  "route_id": "openvino_dense_slm_gpu_arc140v",
  "proof_family": "openvino_dense_slm_gpu_arc140v",
  "model_family": "qwen",
  "model_or_subgraph_contract": {
    "source_model": "Qwen/Qwen2.5-0.5B-Instruct",
    "export_format": "openvino_ir",
    "weight_format": "int4",
    "symmetric": true,
    "group_size": 128,
    "model_binary_committed": false
  },
  "bitnet_qk256_proof": false,
  "native_opencl_proof": false,
  "speedup_claimed": false,
  "power_advantage_claimed": false,
  "full_residency_claimed": false
}
```

If the route is an OpenVINO NPU route, the receipt must also reserve or populate
NPU cold/cache/warm fields before promotion is attempted:

```json
{
  "npu_timing": {
    "first_ever_compile_and_infer_ms": null,
    "pipeline_construct_ms": null,
    "cache_dir": null,
    "cache_mode": "unknown",
    "cache_hit": null,
    "cached_pipeline_construct_ms": null,
    "first_streamed_text_chunk_ms": null,
    "time_to_first_token_ms": null,
    "decode_total_ms": null,
    "steady_tok_per_s": null,
    "warm_ask_total_ms": null,
    "resident_session_total_ms": null
  }
}
```

Null means not measured or not applicable only when accompanied by a reason in
the receipt. Missing fields must not be interpreted as zero.

## Strict Device Selection Rules

```text
--device openvino-cpu must select OpenVINO CPU or fail.
--device openvino-gpu must not silently select CPU.
--device openvino-gpu must resolve GPU.0/GPU.1 and the full device name.
--device openvino-gpu is Arc 140V proof only when resolved_device identifies Arc 140V or the receipt records the exact accepted 258V GPU identity.
--device openvino-npu must not silently select CPU/GPU.
--device openvino-npu must resolve NPU and the full device name/properties available from OpenVINO.
--device openvino-auto is diagnostic unless execution devices are recorded.
--device openvino-hetero is diagnostic unless every executed device partition is recorded.
```

A strict GPU/NPU route with `fallback_used=true`, missing `runtime_device`, or a
resolved CPU fallback must be marked failed for that proof family. It can be
useful diagnostic evidence, but it cannot satisfy route execution.

## Proof Family Claim Boundaries

| Proof family | May prove | Must not prove |
| --- | --- | --- |
| `openvino_dense_slm_cpu` | OpenVINO CPU execution for exact dense SLM export/profile | GPU/NPU execution, BitNet QK256 |
| `openvino_dense_slm_gpu_arc140v` | OpenVINO GenAI execution on Arc 140V / `GPU.0` for exact dense SLM profile | Native OpenCL proof, NPU proof, BitNet QK256 |
| `openvino_dense_slm_npu` | OpenVINO GenAI execution on Intel AI Boost NPU for exact dense SLM profile | Cold-route promotion, native NPU custom kernels, BitNet packed QK256 |
| `openvino_bitnet_subgraph_reference` | Selected static BitNet-shaped subgraph parity | Full BitNet inference, QK256 decode, speedup |
| `openvino_model_server` | Exact endpoint/profile server proof | Broad server readiness, streaming/concurrency, speedup |

## Permanent Hard Rails

Receipts, status docs, campaign rows, and CLI explainers must preserve these
rails:

```text
OpenVINO GPU is not native OpenCL proof.
OpenVINO NPU is not Arc 140V proof.
OpenVINO dense SLM proof is not BitNet QK256 proof.
BitNet QK256 CPU/CUDA proof is not OpenVINO proof.
Generic AUTO/HETERO OpenVINO is not selected-device proof unless the receipt exposes execution devices.
OpenVINO CPU fallback cannot count as GPU/NPU execution.
Retokenized generated text is not the same as direct pipeline-internal generated token IDs.
OpenVINO speedup is exact-profile only.
NPU promotion requires cold/cache/warm/resident separation.
Full residency is false until every relevant phase is proven resident.
```

## Quality And Token Evidence Requirements

OpenVINO answer receipts used for route promotion must record:

- direct case outputs;
- profile and category summaries;
- prompt hash;
- rendered prompt;
- prompt token IDs;
- generation configuration;
- pass/fail result and failure taxonomy;
- `fallback_used=false`;
- generated token evidence.

If OpenVINO pipeline-internal generated token IDs are unavailable and generated
text is retokenized after the fact, the receipt must label those IDs as
retokenized. Retokenized IDs are useful diagnostics but do not satisfy a direct
pipeline-internal generated-token-ID requirement.

## Timing And Promotion Guardrails

OpenVINO timing receipts must name the profile and distinguish the measured
phases. At minimum, timing summaries must not infer:

```text
case_elapsed_ms_sum as a decode throughput claim
pipeline_load_ms as a model-load comparison unless model/export type is same
GPU/NPU timing promotion if prompt token count is missing
speedup claim if the CPU comparator is not same-profile
cold one-off NPU usability from hot-path generation timing
power advantage without measured telemetry or an explicit power-proxy policy
full residency from a subset of phases
```

A route can become promotion-eligible only when the route-specific spec and
promotion review prove quality, selected-device identity, fallback rejection,
profile-specific timing, unchanged model/export contract, and benchmark-
qualified speed or power advantage when those claims are made.

## Receipt Explainer Contract

`bitnet receipts explain` should be able to summarize an OpenVINO receipt as:

```text
route_id: openvino_dense_slm_gpu_arc140v
proof_family: openvino_dense_slm_gpu_arc140v
requested_backend: openvino-gpu
selected_backend: openvino-gpu
runtime_api: openvino_genai
runtime_device: GPU.0
resolved_device: Intel(R) Arc(TM) 140V GPU
fallback_used: false
model/export: Qwen2.5 0.5B Instruct OpenVINO IR INT4 symmetric, group_size=128
quality: candidate / failed / passed for exact profile
phase timing: smoke / profile-specific / cache-warm-resident
promotion: candidate / quality_candidate / benchmark_candidate / promoted_for_profile
speedup_claimed: false
power_advantage_claimed: false
full_residency_claimed: false
what this does not prove: native OpenCL, NPU, BitNet QK256, broad quality, global speedup
```

If any required identity field is missing, the explainer must say which field is
missing instead of promoting a claim.

## Acceptance

A receipt, status row, or CLI summary satisfies this route contract only if:

- it records requested and selected backend identity;
- it records the OpenVINO runtime API;
- it records selected OpenVINO device and resolved device name when applicable;
- strict GPU/NPU routes reject CPU fallback;
- AUTO/HETERO routes are diagnostic unless execution devices are exposed;
- dense SLM receipts set BitNet QK256 and native OpenCL proof booleans false;
- OpenVINO GPU receipts do not replace native OpenCL proof;
- OpenVINO NPU receipts do not replace Arc 140V proof;
- retokenized generated token IDs are marked as retokenized;
- speed, power, and residency booleans remain false unless their exact specs are
  satisfied.

## Non-Goals

This spec does not define the full dense SLM model/export contract, NPU
cold/warm/cache timing contract, corpus-v2 quality contract, route promotion
state machine, Rust bridge, or server readiness. Those are follow-on specs in
the OpenVINO Lunar Lake plan.

## Source-Of-Truth Links

- [OpenVINO Lunar Lake productization proposal](../proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md)
- [Repo source-of-truth system](../reference/SPEC_SYSTEM.md)
- [Intel 258V validation profile](../hardware/intel-258v-validation.md)
- [Intel Lunar Lake NPU roadmap](intel-lunar-lake-npu-roadmap.md)
- [Intel Lunar Lake GPU roadmap](intel-lunar-lake-gpu-roadmap.md)
- [OpenVINO Lunar Lake implementation plan](../../plans/openvino-lunar-lake/implementation-plan.md)
- `docs/tracking/campaigns/intel-258v-platform/active.toml`
- `ci/hardware/intel-258v/**`

## How To Revert

Revert this spec and remove or supersede the related OpenVINO plan and campaign
work items. Existing receipts remain evidence; after revert they must not be
summarized as governed OpenVINO route claims unless another accepted route
contract replaces this one.
