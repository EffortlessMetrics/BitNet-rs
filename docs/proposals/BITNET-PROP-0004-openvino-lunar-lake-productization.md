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
Support-tier impact: no promotion; docs/spec governance only
Policy impact: no policy exception

## Thesis

OpenVINO is BitNet-rs's Intel-runtime lane for dense SLMs and selected small
LLMs on the Lunar Lake 258V CPU/GPU/NPU platform. It gives the repository a way
to compare Rust CPU, OpenVINO CPU, Arc 140V GPU, and Intel AI Boost NPU on the
same machine with strict receipts, quality gates, route promotion, and
profile-scoped performance claims.

OpenVINO is not generic acceleration evidence. OpenVINO dense SLM proof is not
BitNet QK256 proof. OpenVINO GPU proof is not native OpenCL proof. OpenVINO NPU
proof is not a cold one-off route proof until the receipts separate first-ever
compile, cached startup, warm asks, and resident sessions.

## Product Value

The OpenVINO lane should eventually let a user run strict Intel-runtime product
surfaces such as:

```powershell
bitnet model status --device intel-258v-openvino
bitnet ask --device openvino-gpu --model qwen2.5-0.5b-instruct ...
bitnet ask --device openvino-npu --model qwen2.5-0.5b-instruct ...
bitnet bench --device openvino-npu --profile warm_resident ...
bitnet receipts explain --latest
```

A valid receipt must explain the exact model/export contract, selected
OpenVINO device, runtime API, fallback status, quality result, cold/cache/warm
or phase timing split, promotion state, speed/power/full-residency claim state,
and what the receipt does not prove.

This proposal is a governance lane first. It preserves the existing candidate
receipts and adds the source-of-truth rails required before route promotion,
status UX, Rust bridge work, or server readiness can become product claims.

## Current State Ledger

| Target | Current state | Existing evidence class | Product posture | Claim boundary |
| --- | --- | --- | --- | --- |
| Qwen2.5 0.5B Instruct OpenVINO CPU | candidate/control | OpenVINO GenAI CPU smoke, corpus-v2 attempt, phase comparison | correctness/reference route candidate, not default promotion | dense SLM only; not GPU, NPU, or BitNet QK256 proof |
| Qwen2.5 0.5B Instruct OpenVINO GPU.0 / Arc 140V | candidate | bounded ask, GPU smoke, corpus-v2 attempt, route-profile comparison | likely first cold interactive speed candidate after quality and profile timing pass | not native OpenCL, not NPU, not BitNet QK256 |
| Qwen2.5 0.5B Instruct OpenVINO NPU / Intel AI Boost | candidate | bounded ask, NPU smoke, corpus-v2 attempt, phase comparison | likely warm/resident low-power candidate after cache/resident and quality proof | not cold-route promotion, not Arc 140V, not BitNet QK256 |
| Qwen3, SmolLM, Llama/Gemma/Phi small models | future candidates | none in this lane until each model passes the same ladder | second OpenVINO product target | no inherited Qwen2.5 proof |
| BitNet-shaped OpenVINO subgraphs | research/reference | future selected static subgraph parity only | third and separate research target | no full BitNet inference, packed QK256 decode, or speedup claim |

The Lunar Lake campaign already keeps the CPU AVX2, Arc 140V GPU, and Intel AI
Boost NPU proof labels separate. Existing OpenVINO receipts are useful but
candidate-only: GPU/NPU have quality blockers, missing direct pipeline-internal
generated token IDs, incomplete profile timing, and no benchmark-qualified
speed/power advantage. CPU remains the promoted dense SLM default route until
receipt evidence justifies a profile-scoped change.

## First Product Target: Qwen2.5 0.5B Instruct

Qwen2.5 0.5B Instruct is the right first OpenVINO product candidate because the
repository already has an OpenVINO INT4 symmetric IR export path and Lunar Lake
CPU/GPU/NPU receipts for that model family. It is small enough for repeatable
Lunar Lake CPU/GPU/NPU proof, close to the OpenVINO GenAI NPU guidance for
small LLMs, and useful as a dense SLM control distinct from BitNet b1.58.

The required product split is:

```text
Qwen2.5 0.5B Instruct OpenVINO on Lunar Lake:
  CPU = correctness/reference route
  GPU.0 / Arc 140V = likely first interactive speed candidate
  NPU / Intel AI Boost = warm/resident low-power candidate
```

The route can promote only per exact profile after fallback is false, quality
passes for that profile, timing is profile-specific, telemetry context exists or
is explicitly unavailable, and speed/power advantage is benchmark-qualified
where claimed.

## Why GPU Is the First Cold Interactive Candidate

OpenVINO GPU.0 / Arc 140V is the most plausible first cold interactive
candidate because the existing bounded ask timing is promising and does not
inherit the NPU cold compile/cache problem. That promise is not a claim. The
route remains blocked until corpus-v2 profile failures are diagnosed or fixed,
direct token evidence is available or marked unavailable, prompt/output token
counts are present, and same-profile CPU comparators support any speed claim.

OpenVINO GPU proof must remain OpenVINO GenAI proof. It must not be used as
native OpenCL proof, Arc 140V BitNet proof, or packed QK256 proof.

## Why NPU Is a Warm/Resident Candidate

OpenVINO NPU should be treated as a warm/resident low-power candidate, not as a
cold one-off default. OpenVINO's NPU documentation requires an installed NPU
driver, compiles models for the `NPU` device, supports model caching to reduce
startup delay, and distinguishes first-ever inference latency from later first
inference latency. Its NPU limitations also include static-shape support, which
is why full dynamic autoregressive BitNet decode is not the first NPU target.

OpenVINO GenAI on NPU exposes LLM pipeline controls such as `LLMPipeline` on
`NPU`, `MAX_PROMPT_LEN`, `MIN_RESPONSE_LEN`, `PREFILL_HINT`, `GENERATE_HINT`,
`CACHE_DIR`, and `CACHE_MODE`. The BitNet-rs NPU receipts therefore need to
separate first-ever compile and infer, cached pipeline construction, first text
chunk or token, steady decode, warm second ask, and resident session timing.

## Why BitNet OpenVINO Is a Separate Reference Lane

BitNet-shaped OpenVINO work should start as selected static graph or subgraph
parity, then possibly graph-lowering feasibility, and only then model-path
experiments. Dense OpenVINO success for Qwen does not prove BitNet I2_S, BitNet
QK256, native Rust inference, packed CPU/CUDA kernels, or a dynamic BitNet NPU
decode path.

The reference ladder should remain separate:

```text
static RMSNorm parity
static ReLU2 / FFN parity
static linear projection parity
attention block experiment
external OpenVINO/llama.cpp GGUF reference
native graph-lowering feasibility review
only then model-path experiment
```

## Non-Goals

This proposal does not:

- promote OpenVINO GPU or NPU routes;
- claim OpenVINO speedup, power advantage, or full residency;
- claim broad dense SLM answer quality;
- claim BitNet QK256, I2_S, or 1-bit correctness from dense SLM receipts;
- claim native OpenCL proof from OpenVINO GPU receipts;
- claim CUDA proof;
- claim broad OpenVINO Model Server readiness;
- commit model binaries;
- delete Python proof harnesses before equivalent Rust receipt emitters exist.

## Alternatives Considered

| Alternative | Why rejected for this lane |
| --- | --- |
| Treat OpenVINO as generic acceleration | Conflates CPU/GPU/NPU devices, hides fallback risk, and makes speed claims impossible to audit. |
| Promote GPU/NPU from bounded ask timing | Existing quality, token-ID, profile-timing, and benchmark blockers still exist. |
| Start with BitNet full model path on NPU | OpenVINO NPU static-shape constraints and current BitNet QK256 semantics make subgraph/reference proof the honest first step. |
| Wait for Rust-native OpenVINO before product proof | Existing Python OpenVINO GenAI harnesses already emit valuable receipts; Rust should wrap and validate before replacing them. |
| Start with server readiness | Ask/chat quality and route identity must be stable before server endpoint claims are meaningful. |

## Acceptance for This Proposal PR

- Defines why OpenVINO exists as a governed Intel-runtime lane.
- Separates dense SLM product targets, future small LLM candidates, and BitNet
  subgraph/reference research.
- Records CPU/GPU/NPU candidate state without route promotion.
- Links to the route contract and implementation plan.
- Preserves claim boundaries and non-goals.
