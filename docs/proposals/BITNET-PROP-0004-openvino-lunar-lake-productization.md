# BITNET-PROP-0004: OpenVINO Lunar Lake Productization

Status: proposed
Owner: intel-runtime/product
Type: proposal

## Problem

BitNet-rs already has meaningful Lunar Lake 258V evidence for CPU, Arc 140V
GPU, Intel AI Boost NPU, and Qwen2.5 0.5B Instruct through OpenVINO GenAI.
That evidence is useful only if the repo keeps the proof families separate and
turns the current receipts into governed product surfaces instead of treating
OpenVINO as generic acceleration.

Users need to know which model/export contract ran, which OpenVINO device was
selected, whether fallback was rejected, which corpus/profile quality gate
passed, which timing profile was measured, and which claims remain false. A
bounded OpenVINO GPU or NPU smoke must not become proof of BitNet QK256, native
OpenCL, global speedup, server readiness, or cold one-off NPU usability.

## Thesis

OpenVINO is BitNet-rs's Intel-runtime lane for dense SLMs and selected small
LLMs on Lunar Lake CPU/GPU/NPU. It gives the repo a way to compare Rust CPU,
OpenVINO CPU, Arc 140V GPU, and Intel AI Boost NPU on the same machine with
strict receipts, quality gates, route promotion, and profile-scoped performance
claims.

OpenVINO also gives BitNet-rs a separate reference lane for future static
BitNet-shaped graph and subgraph experiments. That reference lane is not a full
BitNet model path and is not QK256 decode proof.

## User-Facing End State

The intended user path is proof-carrying and device-specific:

```powershell
bitnet model status --device intel-258v-openvino
bitnet ask --device openvino-gpu --model qwen2.5-0.5b-instruct "..."
bitnet ask --device openvino-npu --model qwen2.5-0.5b-instruct "..."
bitnet bench --device openvino-npu --profile warm_resident "..."
bitnet receipts explain --latest
```

Those commands should eventually produce receipts that expose:

```text
exact model/export contract
selected OpenVINO device: CPU / GPU.0 Arc 140V / NPU Intel AI Boost
runtime_api=openvino_genai or openvino_runtime
fallback_used=false
answer corpus/profile quality result
cold/warm/cache timing split
route promotion status
speedup/power/full-residency claim status
what this does not prove
```

Friendly device selectors are acceptable only after the receipt records the
selected backend, selected OpenVINO device, resolved device name, runtime API,
route identity, proof family, fallback status, and forbidden-claim booleans.

## Current Supported And Candidate Rows

This proposal records product direction. The authoritative evidence remains the
committed receipts, model coverage ledgers, campaign tracker, and generated
status.

| Lane | Current state | Product meaning | Still false |
| --- | --- | --- | --- |
| Qwen2.5 0.5B Instruct OpenVINO CPU | Candidate/control route with LLMPipeline smoke and corpus-v2 evidence | Correctness/reference route for the dense SLM OpenVINO export | Broad dense SLM quality, speedup, GPU/NPU proof, BitNet QK256 |
| Qwen2.5 0.5B Instruct OpenVINO GPU.0 Arc 140V | Candidate route with bounded smoke, operator ask, corpus-v2, and phase evidence | Likely first cold interactive OpenVINO candidate after quality and timing gaps close | Native OpenCL proof, NPU proof, speedup, broad quality, BitNet QK256 |
| Qwen2.5 0.5B Instruct OpenVINO NPU Intel AI Boost | Warm/resident candidate with bounded smoke and promising hot-path evidence | Candidate for cached, resident, and low-power profiles after cache/warm proof and quality pass | Cold one-off promotion, generic NPU kernels, speedup/power advantage, BitNet QK256 |
| BitNet-shaped OpenVINO subgraphs | Research/reference lane only | Selected static subgraph parity exploration | Full BitNet inference, packed QK256 decode, route speedup |
| OpenVINO exact-profile server | Future lane after ask/chat proof | Exact endpoint/profile proof only | Broad server readiness, concurrency, streaming, speedup |

## First Product Target: Qwen2.5 0.5B Instruct

Qwen2.5 0.5B Instruct is the first OpenVINO product candidate because the repo
already has an OpenVINO INT4 symmetric IR export manifest, CPU/GPU/NPU bounded
smokes, operator ask evidence, corpus-v2 candidate-route evidence, phase
comparison receipts, and route-profile comparison artifacts for that model.
It is small enough to exercise CPU, Arc 140V GPU, and Intel AI Boost NPU on the
same Lunar Lake laptop without turning the work into a large-model serving
campaign.

The governing product split is:

```text
Qwen2.5 0.5B Instruct OpenVINO on Lunar Lake:
  CPU = correctness/reference route
  GPU.0 / Arc 140V = likely first interactive speed candidate
  NPU / Intel AI Boost = warm/resident low-power candidate
```

The CPU route remains the reference/control lane until profile-specific quality
and timing proof justifies a different promoted route. GPU and NPU remain
candidate routes in docs/spec PRs.

## Follow-On Dense Small Model Targets

The second target is Qwen3, SmolLM, Llama, Gemma, and Phi small models through
the same OpenVINO proof ladder. Each model must carry its own source revision,
export contract, tokenizer authority, prompt template, corpus/profile quality,
timing profile, fallback status, and promotion review. Qwen2.5 proof may inform
tooling but must not promote another model row.

## BitNet OpenVINO Research Target

The third target is deliberately separate:

```text
BitNet-shaped OpenVINO subgraphs, then maybe BitNet model-path experiments.
```

OpenVINO dense SLM success does not prove BitNet I2_S, QK256, packed decode,
native Rust inference, or BitNet model semantics. The BitNet OpenVINO research
ladder should start with static RMSNorm, ReLU2/FFN, linear projection, and
attention-block experiments before any model-path review. Selected subgraph
parity receipts are useful reference evidence, not full inference proof.

## OpenVINO Runtime Realities To Encode

The route contract and follow-on specs must account for current OpenVINO NPU
constraints rather than rediscovering them during promotion reviews:

- OpenVINO NPU execution requires the NPU plugin and driver-visible `NPU`
  device, and conventional runtime use compiles models for `NPU` through
  OpenVINO Runtime APIs.
- OpenVINO documents model caching and distinguishes first-ever latency from
  later first-inference latency; BitNet-rs must therefore separate first-ever
  compile/load, cached construction, first token, steady decode, warm second
  ask, and resident session timing.
- OpenVINO NPU documentation exposes NPU driver/compiler and memory properties
  that receipts should preserve when available.
- OpenVINO NPU static-shape limitations make full dynamic autoregressive BitNet
  decode the wrong first NPU target.
- OpenVINO GenAI NPU guidance for small LLMs recommends symmetric compression,
  INT4 or NF4 weights, a maximized 4-bit ratio, and group-size choices such as
  128 for smaller models, matching the Qwen2.5 0.5B direction.
- OpenVINO GenAI NPU configuration includes `LLMPipeline(model_path, "NPU")`,
  prompt/response length controls such as `MAX_PROMPT_LEN` and
  `MIN_RESPONSE_LEN`, performance hints, and caching through `CACHE_DIR`.

References:

- [OpenVINO NPU Device documentation](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html)
- [OpenVINO GenAI on NPU documentation](https://docs.openvino.ai/2025/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html)

## Proof Families

| Proof family | May prove | Must not prove |
| --- | --- | --- |
| `openvino_dense_slm_cpu` | OpenVINO CPU execution for exact dense SLM export/profile | GPU/NPU execution, BitNet QK256 |
| `openvino_dense_slm_gpu_arc140v` | OpenVINO GenAI execution on Arc 140V / `GPU.0` for exact dense SLM profile | Native OpenCL proof, NPU proof, BitNet QK256 |
| `openvino_dense_slm_npu` | OpenVINO GenAI execution on Intel AI Boost NPU for exact dense SLM profile | Cold-route promotion, native NPU custom kernels, BitNet packed QK256 |
| `openvino_bitnet_subgraph_reference` | Selected static BitNet-shaped subgraph parity | Full BitNet inference, QK256 decode, speedup |
| `openvino_model_server` | Exact endpoint/profile server proof | Broad server readiness, streaming/concurrency, speedup |

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

## Non-Goals

This proposal does not claim or implement:

- BitNet QK256 proof from OpenVINO dense SLM receipts;
- native OpenCL proof from OpenVINO GPU receipts;
- CUDA proof;
- broad dense SLM answer quality;
- global OpenVINO speedup;
- cold one-off NPU route usability from hot-path timing;
- broad server readiness;
- model binaries committed to the repo;
- removal of Python proof harnesses before Rust surfaces emit equivalent
  receipts and pass the same validators.

## Source-Of-Truth Links

- [Repo source-of-truth system](../reference/SPEC_SYSTEM.md)
- [OpenVINO route contract](../specs/BITNET-SPEC-OPENVINO-ROUTE-CONTRACT.md)
- [Intel Lunar Lake 258V platform roadmap](../specs/intel-lunar-lake-258v-platform-roadmap.md)
- [Intel Lunar Lake 258V buildout plan](../specs/intel-lunar-lake-258v-buildout-plan.md)
- [Intel Lunar Lake GPU roadmap](../specs/intel-lunar-lake-gpu-roadmap.md)
- [Intel Lunar Lake NPU roadmap](../specs/intel-lunar-lake-npu-roadmap.md)
- [Intel 258V validation profile](../hardware/intel-258v-validation.md)
- [OpenVINO Lunar Lake plan](../../plans/openvino-lunar-lake/implementation-plan.md)
- `docs/tracking/campaigns/intel-258v-platform/active.toml`
- `ci/hardware/intel-258v/**`

If this proposal and a receipt disagree, the receipt is the evidence for what
actually happened. If this proposal and a status/model-coverage surface
disagree, repair the source-of-truth stack before promoting the claim.

## Alternatives Considered

### Treat OpenVINO As Generic Acceleration

Rejected. OpenVINO CPU, GPU, NPU, AUTO, HETERO, GenAI, and conventional runtime
paths have different fallback, device, timing, and claim boundaries. A generic
acceleration label would hide exactly the fields users need.

### Promote GPU/NPU From Existing Smokes

Rejected. Existing smokes and bounded asks are valuable, but current route
comparison still records quality failures, missing direct generated-token IDs,
missing profile-specific timing, and no benchmark-qualified speed or power
advantage. Docs/spec PRs must not promote runtime routes.

### Start With BitNet Model-Path OpenVINO

Rejected. The safer path is dense SLM productization plus selected static
BitNet-shaped subgraph parity. OpenVINO dense SLM receipts cannot prove packed
BitNet QK256 behavior.

### Start With Server Readiness

Rejected. Server readiness should follow ask/chat route readiness and remain
exact-profile. Server receipts must not bypass model/export, quality, timing,
cache/resident, and fallback gates.

## How To Revert

Revert this proposal, the linked route contract, and the OpenVINO Lunar Lake
plan entries. Existing hardware receipts remain immutable evidence; reverting
this proposal only removes the productization direction and claim-governance
language built on top of them.
