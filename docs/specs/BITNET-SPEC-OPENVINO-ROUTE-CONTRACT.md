# BITNET-SPEC-OPENVINO-ROUTE-CONTRACT: OpenVINO Route Identity Contract

Status: proposed
Owner: intel-runtime/product
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0004](../proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md)
Linked specs:
- [Intel Lunar Lake NPU roadmap](intel-lunar-lake-npu-roadmap.md)
Linked ADRs: n/a
Linked plan: [OpenVINO Lunar Lake implementation plan](../../plans/openvino-lunar-lake/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: Defines route identity fields before any OpenVINO support-tier promotion.
Policy impact: No policy exception.

## Purpose

This spec defines the minimum route identity, selected-device, fallback, and
claim-boundary contract for OpenVINO receipts on Lunar Lake. It prevents
OpenVINO CPU, Arc 140V GPU, Intel AI Boost NPU, server, and BitNet subgraph
reference evidence from being treated as interchangeable.

OpenVINO proof is exact-route proof. It is not generic acceleration proof.

## Required Route IDs

OpenVINO receipts and status surfaces must use one of these governed route IDs
when they make a route claim:

```text
openvino_dense_slm_cpu
openvino_dense_slm_gpu_arc140v
openvino_dense_slm_npu
openvino_bitnet_subgraph_reference
openvino_server_exact_profile
```

Campaign-local candidate IDs may remain more specific, such as
`dense_slm_openvino_gpu_candidate`, but they must map to one of the governed
route IDs and proof families before a status, receipt explainer, or promotion
review summarizes the result.

## Required Proof Families

| Route ID | Proof family | Required runtime scope | May prove | Must not prove |
| --- | --- | --- | --- | --- |
| `openvino_dense_slm_cpu` | `openvino_dense_slm_cpu` | OpenVINO Runtime or OpenVINO GenAI on `CPU` | exact dense SLM OpenVINO CPU execution/profile | GPU/NPU execution, BitNet QK256 |
| `openvino_dense_slm_gpu_arc140v` | `openvino_dense_slm_gpu_arc140v` | OpenVINO GenAI or Runtime on `GPU.0`/`GPU.1` resolving to Arc 140V | exact dense SLM OpenVINO GPU execution/profile | native OpenCL, NPU, BitNet QK256 |
| `openvino_dense_slm_npu` | `openvino_dense_slm_npu` | OpenVINO GenAI or Runtime on `NPU` resolving to Intel AI Boost | exact dense SLM OpenVINO NPU execution/profile | cold-route promotion, native NPU custom kernels, BitNet packed QK256 |
| `openvino_bitnet_subgraph_reference` | `openvino_bitnet_subgraph_reference` | static OpenVINO graph/subgraph receipt | selected static BitNet-shaped subgraph parity | full BitNet inference, QK256 decode, speedup |
| `openvino_server_exact_profile` | `openvino_model_server` | one endpoint/profile/server receipt | exact endpoint/profile server proof | broad server readiness, concurrency, global speedup |

## Required Receipt Identity Fields

Every strict OpenVINO route receipt must include these fields or an equivalent
schema location that a validator can read unambiguously:

```json
{
  "requested_backend": "openvino-gpu",
  "selected_backend": "openvino-gpu",
  "runtime_api": "openvino_genai",
  "runtime_device": "GPU.0",
  "resolved_device": "Intel(R) Arc(TM) 140V GPU",
  "fallback_used": false,
  "route_id": "dense_slm_openvino_gpu_candidate",
  "governed_route_id": "openvino_dense_slm_gpu_arc140v",
  "proof_family": "openvino_dense_slm_gpu_arc140v",
  "model_family": "qwen",
  "bitnet_qk256_proof": false,
  "native_opencl_proof": false
}
```

The same information may be nested under `route`, `device`, `model`, or
`claim_boundary` only if receipt validators and `receipts explain` can recover
it without special-case guessing.

## Device Selection Rules

Strict OpenVINO device selectors must behave as follows:

```text
--device openvino-cpu must select OpenVINO CPU or fail.
--device openvino-gpu must not silently select CPU.
--device openvino-npu must not silently select CPU/GPU.
--device openvino-auto is diagnostic unless execution devices are recorded.
--device openvino-gpu must resolve GPU.0/GPU.1 and full device name.
--device openvino-npu must resolve NPU and full device name.
```

A strict receipt for `openvino-gpu` or `openvino-npu` is invalid if CPU fallback
was used, even if the generated answer looks correct.

## Runtime API Values

Receipts must identify the OpenVINO API family because proof scope differs:

| `runtime_api` | Valid use | Boundary |
| --- | --- | --- |
| `openvino_genai` | LLM/SLM `LLMPipeline` proof, streaming chunks, generation metrics | not native Rust BitNet inference |
| `openvino_runtime` | conventional OpenVINO Runtime graph/subgraph proof | not GenAI pipeline proof unless wrapped by a GenAI receipt |
| `openvino_model_server` | endpoint/profile server proof | exact profile only, not broad serving |

The runtime API must not be omitted from receipts that make device or route
claims.

## Dense SLM Route Receipt Fields

Dense SLM OpenVINO receipts must identify the model/export contract at least by
reference, and later dense SLM specs may require additional fields. The route
contract minimum is:

```json
{
  "model_contract": {
    "source_model": "Qwen/Qwen2.5-0.5B-Instruct",
    "source_revision": "<revision-or-explicit-unknown>",
    "format": "openvino_ir",
    "weight_format": "int4",
    "symmetric": true,
    "group_size": 128,
    "ratio": 1.0,
    "tokenizer_source": "hf_tokenizer_export",
    "prompt_template": "qwen2.5",
    "model_binary_committed": false
  }
}
```

Missing source revision is allowed only when the receipt explicitly marks it as
unknown and the route remains candidate-only.

## Quality and Token-ID Rules

A receipt that supports route promotion must include quality evidence for the
exact profile being promoted. For answer-corpus evidence, the receipt must
record:

- direct case outputs;
- profile and category summary;
- pass/fail rules;
- prompt hash;
- prompt token IDs when available;
- rendered prompt;
- generation config;
- `fallback_used=false`;
- whether generated token IDs are direct pipeline-internal IDs or retokenized
  generated text.

Retokenized generated text is useful diagnostic evidence but is not the same as
direct pipeline-internal generated token IDs. A promotion review must carry that
gap forward unless an explicit spec accepts it for the exact profile.

## Timing and Promotion Identity Rules

OpenVINO timing must be scoped to the exact profile. Receipts and route reviews
must not use:

- `case_elapsed_ms_sum` as a decode throughput claim;
- pipeline construction time as a model-load comparison unless the model/export
  type is the same;
- GPU/NPU timing for promotion if prompt token count is missing;
- speedup claims if the CPU comparator is not same-profile;
- NPU hot-path generation timing as cold one-off proof.

NPU receipts that support warm/resident or low-power claims must distinguish
first-ever compile/load, cached load, first streamed chunk or first token, steady
decode, warm second ask, and resident session timing.

## AUTO and HETERO Rules

OpenVINO `AUTO`, `MULTI`, or `HETERO` routing is diagnostic by default. It may
support selected-device proof only when the receipt exposes execution devices
and the route validator can prove that the claimed device executed the relevant
phase.

If execution devices are unavailable, the route state must remain diagnostic or
candidate-only and must not promote CPU/GPU/NPU support.

## Fallback Rules

Fallback handling must be explicit:

| Requested route | Invalid fallback for strict proof | Required behavior |
| --- | --- | --- |
| `openvino_dense_slm_cpu` | non-OpenVINO CPU when OpenVINO CPU is claimed | fail or mark non-OpenVINO fallback |
| `openvino_dense_slm_gpu_arc140v` | CPU, NPU, native OpenCL, CUDA | fail strict validation |
| `openvino_dense_slm_npu` | CPU, GPU, native NPU custom path, CUDA | fail strict validation |
| `openvino_bitnet_subgraph_reference` | CPU-only result presented as OpenVINO subgraph result | fail strict validation |
| `openvino_server_exact_profile` | any backend other than the endpoint/profile's selected backend | fail strict validation |

`fallback_used=false` is required for promotion-eligible strict route proof.

## Promotion States

OpenVINO status surfaces should use these route states until a dedicated route
promotion spec supersedes this list:

```text
unsupported
artifact_ready
runtime_detected
smoke_passed
candidate
quality_candidate
benchmark_candidate
promoted_for_profile
server_exact_profile_ready
```

Docs-only PRs may add or update route states only as planned work. They must not
promote GPU or NPU routes.

## Claim Boundary Fields

Receipts that make an OpenVINO route claim must include explicit booleans or
claim-boundary entries equivalent to:

```json
{
  "bitnet_qk256_proof": false,
  "bitnet_i2s_proof": false,
  "native_opencl_proof": false,
  "native_cuda_proof": false,
  "openvino_selected_device_proof": true,
  "fallback_used": false,
  "speedup_claimed": false,
  "power_advantage_claimed": false,
  "full_residency_claimed": false
}
```

A receipt may set `speedup_claimed`, `power_advantage_claimed`, or
`full_residency_claimed` to true only when a later performance, telemetry, or
residency spec's proof requirements are satisfied for the exact profile.

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

## Acceptance Examples

### Valid GPU candidate receipt

A GPU candidate receipt is valid when it records `requested_backend=openvino-gpu`,
`selected_backend=openvino-gpu`, `runtime_api=openvino_genai`,
`runtime_device=GPU.0`, a resolved Arc 140V device name, `fallback_used=false`,
`proof_family=openvino_dense_slm_gpu_arc140v`, model/export identity, and claim
booleans that keep BitNet QK256 and native OpenCL false.

### Invalid GPU proof with CPU fallback

A receipt is invalid for GPU proof when `--device openvino-gpu` requested GPU but
runtime execution silently used CPU. It may be retained as a failure artifact,
but it must not count as OpenVINO GPU execution or route promotion evidence.

### Valid NPU warm candidate caveat

An NPU receipt may support warm/resident candidate review when it records NPU
selection, fallback rejection, quality evidence, cache/resident configuration,
and separate cold, cached, warm, and resident timings. It still must not imply
cold one-off promotion unless cold compile/load is acceptable for that exact
profile.

### Invalid BitNet inference claim

A dense Qwen OpenVINO receipt is invalid as BitNet proof even if it ran on CPU,
GPU, or NPU. BitNet OpenVINO evidence must come from the separate subgraph or
model-path ladder.

## Validation Expectations

The first docs PR for this spec is validated with campaign and whitespace checks:

```bash
cargo run --locked -p xtask --no-default-features -- campaign check intel-258v-platform
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

Future receipt validators should enforce this route contract before route
promotion, `model status`, or `receipts explain` can report OpenVINO CPU/GPU/NPU
support.
