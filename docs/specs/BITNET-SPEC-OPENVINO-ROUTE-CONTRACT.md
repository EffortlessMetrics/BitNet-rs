# BITNET-SPEC-OPENVINO-ROUTE-CONTRACT: OpenVINO Route Identity and Claim Boundary

Status: draft
Owner: intel/openvino
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0004](../proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md)
Linked specs: n/a
Linked ADRs: n/a
Linked plan: [OpenVINO Lunar Lake implementation plan](../../plans/openvino-lunar-lake/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no promotion; defines future receipt gates
Policy impact: no policy exception

## Purpose

Define the route identity contract for OpenVINO on Lunar Lake so CPU, GPU.0 /
Arc 140V, NPU / Intel AI Boost, BitNet subgraph reference, and exact-profile
server proof cannot be conflated.

This spec governs receipt fields, strict device selection, fallback handling,
and proof-family claim boundaries. It does not promote any OpenVINO route.

## Route IDs

The canonical OpenVINO route IDs are:

| Route ID | Scope | Initial product posture |
| --- | --- | --- |
| `openvino_dense_slm_cpu` | OpenVINO CPU execution for exact dense SLM export/profile | control/candidate |
| `openvino_dense_slm_gpu_arc140v` | OpenVINO GenAI execution on Arc 140V / `GPU.0` for exact dense SLM profile | candidate |
| `openvino_dense_slm_npu` | OpenVINO GenAI execution on Intel AI Boost NPU for exact dense SLM profile | warm/resident candidate |
| `openvino_bitnet_subgraph_reference` | selected static BitNet-shaped subgraph parity | research/reference |
| `openvino_server_exact_profile` | exact endpoint/profile server proof | gated after ask/chat readiness |

Legacy or campaign-local route names may appear in old receipts, but new
receipts and validators must map them to one of these canonical route IDs before
promotion review.

## Proof Families

| Proof family | May prove | Must not prove |
| --- | --- | --- |
| `openvino_dense_slm_cpu` | OpenVINO CPU execution for exact dense SLM export/profile | GPU/NPU execution, BitNet QK256 |
| `openvino_dense_slm_gpu_arc140v` | OpenVINO GenAI execution on Arc 140V / `GPU.0` for exact dense SLM profile | native OpenCL proof, NPU proof, BitNet QK256 |
| `openvino_dense_slm_npu` | OpenVINO GenAI execution on Intel AI Boost NPU for exact dense SLM profile | cold-route promotion, native NPU custom kernels, BitNet packed QK256 |
| `openvino_bitnet_subgraph_reference` | selected static BitNet-shaped subgraph parity | full BitNet inference, QK256 decode, speedup |
| `openvino_model_server` | exact endpoint/profile server proof | broad server readiness, streaming/concurrency, speedup |

## Required Receipt Fields

Every strict OpenVINO route receipt must include the following identity fields or
an explicitly versioned equivalent consumed by the receipt validator:

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
  "bitnet_qk256_proof": false,
  "native_opencl_proof": false
}
```

Receipts should also include the exact model/export contract, prompt/template
identity, generation config, quality result, timing scope, promotion status,
and "does not prove" boundary required by the later dense SLM, quality, timing,
and promotion specs.

## Strict Device Selection Rules

- `--device openvino-cpu` must resolve to OpenVINO CPU execution and must not be
  presented as GPU or NPU evidence.
- `--device openvino-gpu` must not silently select CPU.
- `--device openvino-gpu` must record `GPU.0` or `GPU.1` plus the full resolved
  device name. On Lunar Lake promotion paths, the accepted target is Arc 140V /
  `GPU.0` unless a spec adds a different exact GPU identity.
- `--device openvino-npu` must not silently select CPU or GPU.
- `--device openvino-npu` must record `NPU` plus the full resolved device name,
  driver/compiler properties when available, and `fallback_used=false`.
- `--device openvino-auto` is diagnostic unless the receipt records the actual
  execution devices for every relevant phase. AUTO/HETERO receipts cannot be
  selected-device proof without that execution-device evidence.
- CPU fallback cannot count as GPU or NPU execution.

## Runtime API Rules

Accepted runtime API values are:

| `runtime_api` | Use |
| --- | --- |
| `openvino_genai` | OpenVINO GenAI `LLMPipeline` dense SLM or small LLM execution |
| `openvino_runtime` | Conventional OpenVINO runtime graph/subgraph proof or non-GenAI smoke |
| `openvino_model_server` | Exact endpoint/profile server proof |

A receipt must not use `openvino_runtime` graph proof to imply OpenVINO GenAI
LLM readiness, and must not use `openvino_genai` dense SLM proof to imply native
Rust BitNet inference.

## Model and Export Identity Minimum

Strict dense SLM OpenVINO receipts must bind route identity to a model/export
contract that includes at least:

```json
{
  "source_model": "Qwen/Qwen2.5-0.5B-Instruct",
  "source_revision": "<revision-or-explicit-unknown>",
  "export_tool": "optimum-cli export openvino",
  "format": "openvino_ir",
  "weight_format": "int4",
  "symmetric": true,
  "group_size": 128,
  "ratio": 1.0,
  "tokenizer_source": "hf_tokenizer_export",
  "prompt_template": "qwen2.5",
  "model_binary_committed": false
}
```

Missing or unknown fields must be explicit. A changed model/export contract
resets promotion eligibility until the exact route/profile proof ladder is rerun.

## Permanent Hard Rails

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

This route contract only defines identity. A route can become promotion-eligible
for a profile only after later specs require and validators confirm:

- exact model/export contract exists;
- selected device identity is exact;
- `fallback_used=false`;
- quality passes for every case in that profile or an explicit spec excludes a
  diagnostic-only case;
- prompt hash, prompt token IDs, rendered prompt, and generation config are
  recorded;
- retokenized generated token IDs are clearly marked when direct pipeline-
  internal IDs are unavailable;
- phase timing is profile-specific and includes prompt/output token counts;
- NPU routes distinguish first-ever compile, cached startup, warm second ask,
  and resident session timing before warm/resident promotion;
- speed or power claims are benchmark-qualified for the exact profile;
- the receipt states what it does not prove.

## Rejection Examples

| Receipt condition | Required handling |
| --- | --- |
| Requested `openvino-gpu`, selected CPU | Fail strict route validation; no GPU proof |
| Requested `openvino-npu`, selected GPU | Fail strict route validation; no NPU proof |
| AUTO route with no execution-device breakdown | Diagnostic only; no selected-device proof |
| Dense Qwen receipt sets `bitnet_qk256_proof=true` | Fail validation |
| OpenVINO GPU receipt sets `native_opencl_proof=true` | Fail validation |
| NPU hot decode receipt lacks cold/cache/resident timing | Candidate only; no warm/resident promotion |
| Retokenized generated IDs are recorded as direct internal IDs | Fail validation or mark route non-promotable until corrected |

## Acceptance

- CPU, GPU.0 / Arc 140V, NPU / Intel AI Boost, BitNet subgraph, and server
  route IDs are defined separately.
- Required receipt fields include requested/selected backend, runtime API,
  device identity, fallback status, route ID, proof family, model family, and
  false BitNet/OpenCL claim booleans.
- Strict rules reject silent CPU fallback for GPU/NPU routes.
- AUTO/HETERO is diagnostic unless execution devices are recorded.
- No runtime route is promoted by this spec.
