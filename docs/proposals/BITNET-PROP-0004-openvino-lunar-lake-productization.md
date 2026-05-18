# BITNET-PROP-0004: OpenVINO Lunar Lake Productization

Status: proposed
Owner: intel/openvino
Created: 2026-05-18
Linked proposal: n/a
Linked specs: [BITNET-SPEC-OPENVINO-ROUTE-CONTRACT](../specs/BITNET-SPEC-OPENVINO-ROUTE-CONTRACT.md)
Linked ADRs: n/a
Linked plan: [OpenVINO Lunar Lake implementation plan](../../plans/openvino-lunar-lake/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no promotion; defines future proof lanes only
Policy impact: none

## Problem

BitNet-rs already has a Lunar Lake 258V campaign with CPU AVX2, Arc 140V GPU,
Intel AI Boost NPU, and dense SLM OpenVINO receipts. Without a productization
proposal, future work can accidentally treat any OpenVINO success as generic
acceleration, BitNet QK256 proof, native OpenCL proof, or an NPU cold-route
promotion. That would make receipts less useful precisely where the platform is
most valuable: one machine can compare Rust CPU, OpenVINO CPU, Arc 140V GPU,
and Intel AI Boost NPU under strict route identity.

## Thesis

OpenVINO is BitNet-rs's Intel-runtime lane for dense SLMs and selected small
LLMs on Lunar Lake CPU/GPU/NPU. It gives the repo a way to compare Rust CPU,
OpenVINO CPU, Arc 140V GPU, and Intel AI Boost NPU on the same machine with
strict receipts, quality gates, route promotion, and profile-scoped performance
claims.

OpenVINO is also a clearly separate reference lane for future BitNet-shaped
static graph and subgraph experiments. It is not generic acceleration and does
not prove BitNet I2_S/QK256 decode.

## Product End State

The eventual user path should be able to emit receipts for commands such as:

```powershell
bitnet model status --device intel-258v-openvino
bitnet ask --device openvino-gpu --model qwen2.5-0.5b-instruct ...
bitnet ask --device openvino-npu --model qwen2.5-0.5b-instruct ...
bitnet bench --device openvino-npu --profile warm_resident ...
bitnet receipts explain --latest
```

Those receipts must expose:

- exact model/export contract;
- selected OpenVINO device: `CPU`, `GPU.0` / Arc 140V, or `NPU` / Intel AI
  Boost;
- runtime API: `openvino_genai` or `openvino_runtime`;
- `fallback_used=false` for strict device routes;
- answer corpus or profile quality result;
- cold, warm, cache, and resident timing split where relevant;
- route promotion status;
- speedup, power, and full-residency claim status;
- what the receipt does not prove.

## Current Supported And Candidate Rows

| Lane | Current role | Product direction | Still false |
| --- | --- | --- | --- |
| Rust dense SLM CPU | Promoted dense SLM default/control route | Correctness and same-profile comparator | OpenVINO proof, GPU/NPU proof |
| OpenVINO dense SLM CPU | Candidate/control route for exact OpenVINO export | Reference OpenVINO CPU execution | GPU/NPU execution, broad dense quality |
| OpenVINO dense SLM GPU.0 / Arc 140V | Candidate route with promising bounded timing | First likely cold interactive OpenVINO candidate | native OpenCL proof, BitNet QK256 proof, speedup |
| OpenVINO dense SLM NPU / Intel AI Boost | Warm/resident candidate with hot-path promise and cold-start blockers | Low-power or resident-session candidate | cold one-off promotion, native NPU kernels, BitNet QK256 proof |
| OpenVINO BitNet-shaped subgraph | Future research/reference lane | Static subgraph parity before model-path experiments | full BitNet inference, QK256 decode, speedup |
| OpenVINO exact-profile server | Future server lane after ask/chat proof | Endpoint-specific proof | broad server readiness, streaming/concurrency claims |

This proposal does not promote any OpenVINO GPU or NPU route. Current Lunar
Lake receipts keep OpenVINO accelerator rows candidate-only until quality,
profile-specific timing, direct-token evidence, and benchmark qualification gaps
are closed.

## Why Qwen2.5 0.5B Is First

Qwen2.5 0.5B Instruct is the first OpenVINO product candidate because the
repository already has an OpenVINO INT4 symmetric IR export manifest and
CPU/GPU/NPU Lunar Lake receipts for that dense SLM operating path. It is small
enough to exercise CPU, Arc 140V GPU, and Intel AI Boost NPU on one 258V laptop
without turning the exercise into a broad small-LLM claim.

The required export shape is compatible with the OpenVINO GenAI NPU guidance
for small LLMs: symmetric compression, 4-bit INT4 or NF4 weights, group-wise or
channel-wise quantization, and group size 128 for smaller models. That makes the
Qwen2.5 0.5B lane the right first proof ladder before Qwen3, SmolLM,
Llama/Gemma/Phi small models, or any BitNet-shaped OpenVINO research lane.

## Why GPU Is Likely The First Cold Interactive Candidate

The Arc 140V GPU route is the likely first cold interactive candidate because
GPU.0 can be selected through OpenVINO GenAI without inheriting NPU compile/cache
costs. The route can become product-ready only for exact profiles where:

- `--device openvino-gpu` resolves to `GPU.0`/`GPU.1` with a full Arc 140V device
  name;
- CPU fallback is rejected and recorded as `fallback_used=false`;
- corpus/profile quality passes;
- prompt and generated token counts are present;
- timing is profile-specific;
- any speed or UX advantage is benchmark-qualified.

OpenVINO GPU proof is not native OpenCL proof. Native OpenCL proof remains a
separate Arc 140V lane.

## Why NPU Is A Warm/Resident Low-Power Candidate

The Intel AI Boost NPU route is not a cold one-off default candidate. OpenVINO
NPU model caching exists specifically to reduce startup delays and distinguishes
first-ever inference latency from later first inference latency, which maps to
BitNet-rs's cold/cache/warm/resident proof split. The NPU plugin also has device
properties, cache modes, performance hints, compiler/driver versions, memory
fields, and compilation options that receipts must preserve.

OpenVINO's NPU documentation currently limits NPU support to static-shape
models, so dynamic autoregressive BitNet decode is not the first NPU target. The
first NPU product target should be warm/resident dense SLM asks or chats where
quality passes and cached/resident timing, plus power or accepted power-proxy
evidence, justifies the exact profile.

## Why OpenVINO BitNet Starts As Subgraph Reference

OpenVINO BitNet work starts as a reference lane for selected static subgraphs:
RMSNorm, ReLU2/FFN, linear projection, and later attention-block experiments.
Only after static parity and graph-lowering feasibility are proven should a
model-path experiment be considered.

This keeps the BitNet and dense SLM lanes separate:

- OpenVINO Qwen success is dense SLM proof only.
- OpenVINO INT4 symmetric IR is not BitNet I2_S/QK256.
- OpenVINO GenAI `LLMPipeline` is not native Rust BitNet inference.
- OpenVINO NPU subgraph parity is not full BitNet inference.
- OpenVINO GPU execution is not native OpenCL execution.

## Non-Goals

This proposal does not claim or authorize:

- BitNet QK256 proof from OpenVINO dense SLM receipts;
- native OpenCL proof from OpenVINO GPU receipts;
- CUDA proof;
- broad OpenVINO server readiness;
- global OpenVINO speedup;
- broad dense SLM quality;
- cold one-off NPU usability from hot-path numbers;
- CPU fallback as GPU or NPU execution;
- retokenized generated text as direct pipeline-internal generated token IDs;
- committed model binaries.

## Alternatives Considered

### Treat OpenVINO As Generic Acceleration

Rejected. Generic acceleration would erase the distinction between OpenVINO CPU,
OpenVINO GPU, OpenVINO NPU, native OpenCL, CUDA, and BitNet QK256 proof.

### Start With BitNet Full Inference On NPU

Rejected. The NPU lane has static-shape constraints and cold/cache behavior that
must be measured first. Static BitNet-shaped subgraph parity is the safer and
more honest research ladder.

### Promote GPU Or NPU From Existing Smokes

Rejected. Existing OpenVINO receipts are valuable but still candidate-only for
accelerator routes because quality, direct-token, profile timing, cache/resident,
and benchmark-qualified advantage gaps remain.

## Source-Of-Truth Links

- [Source-of-truth system](../reference/SPEC_SYSTEM.md)
- [OpenVINO route contract](../specs/BITNET-SPEC-OPENVINO-ROUTE-CONTRACT.md)
- [Intel Lunar Lake 258V platform roadmap](../specs/intel-lunar-lake-258v-platform-roadmap.md)
- [Intel Lunar Lake GPU roadmap](../specs/intel-lunar-lake-gpu-roadmap.md)
- [Intel Lunar Lake NPU roadmap](../specs/intel-lunar-lake-npu-roadmap.md)
- [OpenVINO Lunar Lake plan](../../plans/openvino-lunar-lake/implementation-plan.md)
- `docs/tracking/campaigns/intel-258v-platform/active.toml`
- OpenVINO NPU device documentation: <https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html>
- OpenVINO GenAI on NPU documentation: <https://docs.openvino.ai/2025/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html>

If this proposal and a committed receipt disagree, the receipt is evidence for
what happened. If this proposal and the route contract disagree, update the
proposal or contract before promoting user-facing claims.
