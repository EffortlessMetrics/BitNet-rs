# BITNET-SPEC-OPENVINO-ROUTE-CONTRACT: OpenVINO Route Identity Contract

Status: proposed
Owner: intel/openvino
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0004](../proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md)
Linked specs: [Intel Lunar Lake 258V platform roadmap](intel-lunar-lake-258v-platform-roadmap.md), [Intel Lunar Lake NPU roadmap](intel-lunar-lake-npu-roadmap.md)
Linked ADRs: n/a
Linked plan: [OpenVINO Lunar Lake implementation plan](../../plans/openvino-lunar-lake/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no promotion; defines receipt identity and claim boundaries
Policy impact: none

## Purpose

Define OpenVINO route identity for Lunar Lake dense SLM and future BitNet-shaped
reference work. This contract prevents OpenVINO CPU, Arc 140V GPU, Intel AI
Boost NPU, native OpenCL, CUDA, and BitNet QK256 proofs from being collapsed
into one another.

## Required Route IDs

Receipts, status rows, campaign items, and explainers must use these governed
route identifiers for OpenVINO work:

| Route ID | Applies to | May prove | Must not prove |
| --- | --- | --- | --- |
| `openvino_dense_slm_cpu` | OpenVINO CPU execution for an exact dense SLM export/profile | OpenVINO CPU execution | GPU/NPU execution, BitNet QK256 |
| `openvino_dense_slm_gpu_arc140v` | OpenVINO GenAI execution on Arc 140V / `GPU.0` or explicit `GPU.x` | OpenVINO GPU execution for exact dense SLM profile | native OpenCL, NPU, BitNet QK256 |
| `openvino_dense_slm_npu` | OpenVINO GenAI execution on Intel AI Boost NPU | OpenVINO NPU execution for exact dense SLM profile | cold-route promotion, native NPU custom kernels, BitNet packed QK256 |
| `openvino_bitnet_subgraph_reference` | Selected static BitNet-shaped OpenVINO graph/subgraph parity | static subgraph parity | full BitNet inference, QK256 decode, speedup |
| `openvino_server_exact_profile` | Exact endpoint/profile server proof | endpoint/profile server readiness | broad server readiness, streaming/concurrency, speedup |

Legacy or campaign-local route IDs such as `dense_slm_openvino_gpu_candidate`
may remain as route instance names, but they must map to one of the proof
families above before any user-facing claim is reported.

## Required Receipt Fields

Strict OpenVINO receipts must expose the route identity fields below. Additional
fields may be added by later specs for model contracts, quality, timing, NPU
cache/warm proof, server status, or Rust bridge provenance.

```json
{
  "requested_backend": "openvino-gpu",
  "selected_backend": "openvino-gpu",
  "runtime_api": "openvino_genai",
  "runtime_device": "GPU.0",
  "resolved_device": "Intel(R) Arc(TM) 140V GPU",
  "fallback_used": false,
  "route_id": "dense_slm_openvino_gpu_candidate",
  "proof_family": "openvino_dense_slm_gpu_arc140v",
  "model_family": "qwen",
  "bitnet_qk256_proof": false,
  "native_opencl_proof": false
}
```

### Field Semantics

- `requested_backend` is the user or harness request, such as `openvino-cpu`,
  `openvino-gpu`, `openvino-npu`, or diagnostic `openvino-auto`.
- `selected_backend` is the backend that actually executed the route. It must
  not be rewritten to the requested backend when fallback occurred.
- `runtime_api` is `openvino_genai` for GenAI `LLMPipeline` paths or
  `openvino_runtime` for conventional Runtime graph/subgraph paths.
- `runtime_device` is the OpenVINO device selector that executed work, such as
  `CPU`, `GPU.0`, `GPU.1`, or `NPU`.
- `resolved_device` is the full device name where OpenVINO exposes it.
- `fallback_used` is `true` if any CPU/GPU/NPU substitution was accepted after
  the requested strict device could not execute.
- `route_id` is the route instance or campaign candidate identifier.
- `proof_family` is one of the governed proof families in this spec.
- `model_family` records dense SLM or model-family context without changing the
  proof family.
- `bitnet_qk256_proof` and `native_opencl_proof` must remain `false` for dense
  SLM OpenVINO receipts.

## Device Selection Rules

- `--device openvino-cpu` must resolve to OpenVINO `CPU` and must not count as
  GPU or NPU proof.
- `--device openvino-gpu` must not silently select CPU.
- `--device openvino-npu` must not silently select CPU or GPU.
- `--device openvino-auto` is diagnostic unless the receipt records execution
  devices and fallback behavior precisely enough to prove selected-device
  execution.
- `--device openvino-gpu` must resolve `GPU.0`/`GPU.1` and the full device name
  before it can satisfy Arc 140V route proof.
- `--device openvino-npu` must resolve `NPU` and the full Intel AI Boost device
  name before it can satisfy NPU route proof.
- Generic `AUTO` or `HETERO` OpenVINO execution is not selected-device proof
  unless every execution device relevant to the claim is exposed in the receipt.

## Runtime API Rules

OpenVINO GenAI and conventional OpenVINO Runtime proofs are related but not
interchangeable:

- Dense SLM ask/chat/corpus/bench proofs should use `runtime_api=openvino_genai`
  unless a later spec explicitly defines a Runtime-only path.
- Static graph or subgraph parity proofs should use `runtime_api=openvino_runtime`
  unless GenAI is truly executing the graph under test.
- A conventional Runtime subgraph receipt cannot promote an OpenVINO GenAI LLM
  route.
- A GenAI dense SLM receipt cannot promote a BitNet subgraph reference route.

## Permanent Hard Rails

These rails apply to specs, status docs, receipts, campaign rows, route
comparison ledgers, and explainers:

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

## Promotion Preconditions

This route contract does not promote any route. Later promotion specs and
receipts may promote exact profiles only when all of the following are true:

- model/export contract exists and is unchanged for the compared evidence;
- requested and selected devices match the strict route;
- `fallback_used=false`;
- proof family is explicit;
- corpus/profile quality passes for the promoted profile;
- prompt token count and generated token evidence are present;
- retokenized generated token IDs, if used, are marked as retokenized and not
  pipeline-internal IDs;
- timing evidence is profile-specific;
- telemetry is present or explicitly unavailable;
- speed, power, and residency claims are benchmark-qualified where claimed.

## NPU-Specific Constraints

NPU route receipts must preserve OpenVINO constraints that affect product
promotion:

- NPU execution requires an installed NPU driver and explicit compilation or
  pipeline selection for `NPU`.
- OpenVINO model caching must be recorded when used to reduce startup delay.
- Receipts must distinguish first-ever compile/inference, cached compile or
  pipeline construction, first inference, and steady decode.
- Static-shape constraints mean NPU static graph/subgraph proof is not dynamic
  autoregressive BitNet decode proof.
- NPU low-power promotion requires power telemetry or an explicit power-proxy
  policy accepted by a later spec.

## Examples

### Valid GPU Candidate Receipt Summary

```json
{
  "requested_backend": "openvino-gpu",
  "selected_backend": "openvino-gpu",
  "runtime_api": "openvino_genai",
  "runtime_device": "GPU.0",
  "resolved_device": "Intel(R) Arc(TM) 140V GPU",
  "fallback_used": false,
  "proof_family": "openvino_dense_slm_gpu_arc140v",
  "promotion_eligible": false,
  "blockers": [
    "corpus/profile quality failures remain",
    "profile-specific timing missing",
    "benchmark-qualified speedup_or_power_advantage missing"
  ],
  "bitnet_qk256_proof": false,
  "native_opencl_proof": false
}
```

### Invalid GPU Claim

```text
Requested openvino-gpu, selected CPU, fallback_used=true, proof_family=openvino_dense_slm_gpu_arc140v.
```

This is invalid because CPU fallback cannot satisfy GPU proof.

### Invalid BitNet Claim

```text
Qwen2.5 0.5B OpenVINO INT4 symmetric IR passed a GPU smoke, therefore BitNet QK256 works on Arc 140V.
```

This is invalid because dense SLM OpenVINO proof is not BitNet QK256 proof and
OpenVINO GPU proof is not native OpenCL proof.

## Acceptance For This Spec PR

- The OpenVINO proof families and route IDs are named.
- Required route receipt identity fields are defined.
- CPU/GPU/NPU strict device selection rules reject silent fallback.
- AUTO/HETERO routes remain diagnostic unless execution devices are recorded.
- Dense SLM OpenVINO receipts cannot prove BitNet QK256 or native OpenCL.
- No OpenVINO GPU/NPU runtime route is promoted by this docs-only PR.

## Source-Of-Truth Links

- [OpenVINO productization proposal](../proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md)
- [OpenVINO Lunar Lake plan](../../plans/openvino-lunar-lake/implementation-plan.md)
- [Intel Lunar Lake 258V platform roadmap](intel-lunar-lake-258v-platform-roadmap.md)
- [Intel Lunar Lake GPU roadmap](intel-lunar-lake-gpu-roadmap.md)
- [Intel Lunar Lake NPU roadmap](intel-lunar-lake-npu-roadmap.md)
- `docs/tracking/campaigns/intel-258v-platform/active.toml`
