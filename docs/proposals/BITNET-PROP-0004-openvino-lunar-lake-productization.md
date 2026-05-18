# BITNET-PROP-0004: OpenVINO Lunar Lake Productization

Status: proposed
Owner: intel-runtime/product
Created: 2026-05-18
Linked proposal: n/a
Linked specs:
- [BITNET-SPEC-OPENVINO-ROUTE-CONTRACT](../specs/BITNET-SPEC-OPENVINO-ROUTE-CONTRACT.md)
Linked ADRs: n/a
Linked plan: [OpenVINO Lunar Lake implementation plan](../../plans/openvino-lunar-lake/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: No support-tier promotion; this proposal defines the governed lane and claim boundaries.
Policy impact: No policy exception.

## Problem

BitNet-rs already has Lunar Lake 258V proof artifacts for CPU, Arc 140V GPU,
Intel AI Boost NPU, dense SLM OpenVINO exports, bounded OpenVINO GenAI smokes,
corpus-v2 candidate-route evidence, and route-profile comparison receipts. The
missing product artifact is not another smoke run. The missing artifact is a
clear source-of-truth statement that explains what OpenVINO is for, which claims
it may support, and which claims it must never inherit from adjacent CPU,
OpenCL, CUDA, or BitNet QK256 lanes.

Without that proposal, a successful OpenVINO Qwen run could be misread as a
generic accelerator result, a native OpenCL result, an NPU cold-route result, or
BitNet packed QK256 proof. Those interpretations are false. OpenVINO needs a
separate governed lane so users can compare exact dense SLM and selected small
LLM routes on the same Lunar Lake system without collapsing proof families.

## Thesis

OpenVINO is BitNet-rs's Intel-runtime lane for dense SLMs and selected small
LLMs on Lunar Lake CPU/GPU/NPU. It gives the repo a way to compare Rust CPU,
OpenVINO CPU, Arc 140V GPU, and Intel AI Boost NPU on the same machine with
strict receipts, quality gates, route promotion, and profile-scoped performance
claims.

The lane has two distinct product tracks and one research track:

1. Qwen2.5 0.5B Instruct OpenVINO on Lunar Lake:
   - CPU is the correctness/reference OpenVINO route.
   - GPU.0 / Arc 140V is the likely first cold interactive candidate.
   - NPU / Intel AI Boost is a warm/resident low-power candidate, not a cold
     one-off default.
2. Qwen3, SmolLM, Llama/Gemma/Phi small dense models through the same OpenVINO
   proof ladder.
3. BitNet-shaped OpenVINO subgraphs as a separate reference/research lane before
   any full BitNet model-path experiment.

These targets must not collapse into one another.

## User-Facing End State

The intended OpenVINO user path is proof-carrying and receipt-backed:

```powershell
bitnet model status --device intel-258v-openvino
bitnet ask --device openvino-gpu --model qwen2.5-0.5b-instruct ...
bitnet ask --device openvino-npu --model qwen2.5-0.5b-instruct ...
bitnet bench --device openvino-npu --profile warm_resident ...
bitnet receipts explain --latest
```

The receipt or receipt explanation must identify:

- exact model/export contract;
- selected OpenVINO device, such as `CPU`, `GPU.0` / Arc 140V, or `NPU` / Intel
  AI Boost;
- runtime API, such as `openvino_genai` or `openvino_runtime`;
- `fallback_used=false` for strict CPU/GPU/NPU route proof;
- answer corpus/profile quality result;
- cold/warm/cache timing split when NPU or residency is involved;
- route promotion status;
- speedup, power, and full-residency claim status;
- what the receipt does not prove.

## Current Lunar Lake OpenVINO Baseline

The Intel 258V platform campaign already treats CPU AVX2, Arc 140V GPU, and
Intel AI Boost NPU proof labels as separate. The existing hard rails remain in
force:

- OpenVINO GPU smoke is not packed BitNet kernel proof.
- Arc 140V OpenCL proof is not NPU proof.
- WSL only counts for NPU validation if OpenVINO reports NPU inside WSL.

The campaign has already merged substantial OpenVINO evidence:

- Qwen2.5 OpenVINO INT4 symmetric IR export manifest;
- OpenVINO CPU LLMPipeline smoke;
- OpenVINO GPU / Arc 140V bounded smoke;
- OpenVINO NPU / Intel AI Boost bounded smoke;
- OpenVINO CPU/GPU/NPU phase comparison;
- OpenVINO GenAI phase runner;
- OpenVINO GPU/NPU operator ask helper;
- OpenVINO corpus-v2 candidate-route execution;
- Lunar Lake route profile comparison and promotion ledger.

Those artifacts make OpenVINO productization a governance and proof-quality
problem first. They do not justify route promotion by themselves.

## Current Supported and Candidate Rows

| Model/export | Route | Current role | Quality state | Timing state | Promoted? | Claim boundary |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5 0.5B Instruct OpenVINO INT4 symmetric IR | OpenVINO CPU | correctness/control candidate | corpus-v2 partial pass; not broad quality | smoke/phase evidence exists | no OpenVINO default promotion | dense SLM only |
| Qwen2.5 0.5B Instruct OpenVINO INT4 symmetric IR | OpenVINO GPU.0 / Arc 140V | cold interactive candidate | corpus-v2 failures remain | promising bounded ask timing; profile-specific proof still required | no | not native OpenCL, not NPU, not BitNet QK256 |
| Qwen2.5 0.5B Instruct OpenVINO INT4 symmetric IR | OpenVINO NPU / Intel AI Boost | warm/resident low-power candidate | corpus-v2 failures remain | hot path promising; cold load/compile/cache/residency proof still required | no | not cold one-off promotion, not native NPU kernels, not BitNet QK256 |
| BitNet-shaped static subgraphs | OpenVINO NPU/runtime reference | research/reference lane | selected subgraph parity only when proven | no full model path | no | not full BitNet inference, not QK256 decode |

The current route profile comparison keeps `dense_slm_default_cpu` as the
promoted route and OpenVINO GPU/NPU as candidates. This proposal preserves that
state.

## Why Qwen2.5 0.5B Is First

Qwen2.5 0.5B Instruct is the first OpenVINO product candidate because it is
small enough for repeatable Lunar Lake CPU/GPU/NPU experiments, already has an
OpenVINO INT4 symmetric IR operating path, already appears in the dense SLM
corpus and phase receipts, and is large enough to exercise the user-facing ask,
corpus, timing, and promotion machinery.

The first OpenVINO lane should prove the ladder, not chase the largest model.
After the Qwen2.5 route contract is trustworthy, Qwen3, SmolLM, and selected
Llama/Gemma/Phi small dense models can reuse the same proof requirements.

## Why GPU Is the First Cold Interactive Candidate

Arc 140V / `GPU.0` is the likely first cold interactive OpenVINO candidate
because GPU bounded ask timing is promising while still avoiding the NPU's large
cold compile/load cost. However, GPU promotion remains blocked until receipts
show profile-specific timing, selected-device identity, fallback rejection,
quality pass for the promoted profile, and a same-profile CPU comparator when a
speed or UX advantage is claimed.

OpenVINO GPU receipts must never be summarized as native OpenCL proof. Native
OpenCL evidence remains a separate Arc 140V proof family.

## Why NPU Is Warm/Resident First

Intel AI Boost NPU is a warm/resident low-power candidate because existing
bounded asks show promising hot-path behavior but also show that cold load or
compile can dominate one-off usage. The OpenVINO NPU plugin's model caching
model distinguishes first-ever inference from later first inference, which maps
directly to BitNet-rs's required cold, cached, warm, and resident proof split.

NPU promotion must therefore be profile-specific. It may become eligible for
warm/resident or low-power profiles only after quality passes, selected NPU
identity and fallback rejection are recorded, cache/residency timing is split,
and power telemetry or an explicit power-proxy policy supports any low-power
claim. It must not be promoted for cold one-off asks using hot-path numbers.

## Why OpenVINO BitNet Is Subgraph/Reference First

OpenVINO dense SLM success does not prove BitNet packed QK256/I2_S behavior.
The BitNet OpenVINO path should start with selected static subgraphs, such as
RMSNorm, ReLU2/FFN, linear projection, and later attention-block experiments,
with CPU parity receipts and explicit tolerance. Only after graph-lowering
feasibility is reviewed should a full BitNet model-path experiment be proposed.

This preserves the boundary already established by the Lunar Lake NPU roadmap:
selected static subgraph parity may be useful evidence, but it is not full
BitNet inference or packed QK256 decode proof.

## External OpenVINO Constraints to Encode

OpenVINO NPU documentation makes the cold/warm/cache split a first-class proof
concern. The NPU plugin requires an installed NPU driver, supports compiling a
model for `NPU`, exposes device and compiler properties, and documents model
caching as a way to reduce startup delays. It distinguishes first-ever inference
latency from later first inference latency, so BitNet-rs receipts must not merge
first-ever compile, cached load, first inference, and steady decode into one
number. See OpenVINO NPU documentation:
<https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html>.

OpenVINO GenAI-on-NPU documentation recommends symmetric 4-bit or NF4 weight
compression for LLMs, group-wise or channel-wise quantization, high 4-bit weight
ratio, and group size 128 for smaller models. It also documents
`LLMPipeline(model_path, "NPU")`, NPU prompt/response shape settings such as
`MAX_PROMPT_LEN` and `MIN_RESPONSE_LEN`, performance hints, and `CACHE_DIR` for
cache-backed startup behavior. See OpenVINO GenAI NPU documentation:
<https://docs.openvino.ai/2025/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html>.

These external constraints are reference constraints, not BitNet-rs proof. A
BitNet-rs claim still requires committed receipts.

## Proof Families

| Proof family | May prove | Must not prove |
| --- | --- | --- |
| `openvino_dense_slm_cpu` | OpenVINO CPU execution for exact dense SLM export/profile | GPU/NPU execution, BitNet QK256 |
| `openvino_dense_slm_gpu_arc140v` | OpenVINO GenAI execution on Arc 140V / `GPU.0` for exact dense SLM profile | native OpenCL proof, NPU proof, BitNet QK256 |
| `openvino_dense_slm_npu` | OpenVINO GenAI execution on Intel AI Boost NPU for exact dense SLM profile | cold-route promotion, native NPU custom kernels, BitNet packed QK256 |
| `openvino_bitnet_subgraph_reference` | selected static BitNet-shaped subgraph parity | full BitNet inference, QK256 decode, speedup |
| `openvino_model_server` | exact endpoint/profile server proof | broad server readiness, streaming/concurrency, speedup |

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

This proposal does not claim or promote:

- BitNet QK256/I2_S proof from OpenVINO dense SLM receipts;
- native OpenCL proof from OpenVINO GPU receipts;
- CUDA readiness;
- broad server readiness;
- global OpenVINO speedup;
- broad dense SLM quality;
- cold one-off NPU usability from hot-path timing;
- model binaries committed to the repository.

## Alternatives Considered

### Treat OpenVINO as generic acceleration

Rejected. Generic acceleration would hide selected-device identity and allow CPU
fallback or AUTO/HETERO routing to be misread as GPU/NPU proof.

### Start with BitNet full-model OpenVINO experiments

Rejected for productization. Dense SLM OpenVINO and BitNet packed QK256 have
different proof families. BitNet-shaped OpenVINO work should begin as static
subgraph parity and research feasibility.

### Promote GPU/NPU from existing smokes

Rejected. Existing receipts are useful candidate evidence but still have quality
failures, missing direct generated-token IDs, missing profile-specific timing,
missing benchmark-qualified speed/power evidence, and NPU cold/cache/resident
proof gaps.

## Acceptance for the First Docs PR

The first PR in this lane is docs/spec only. It is accepted when:

- this proposal defines why OpenVINO exists as a governed lane;
- the route contract defines CPU/GPU/NPU identities and claim boundaries;
- the implementation plan lists PR-sized next steps;
- no runtime route is promoted;
- the campaign tracker records docs/spec work items only.
