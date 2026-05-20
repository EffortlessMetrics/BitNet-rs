# BITNET-SPEC-OPENVINO-DENSE-SLM: Dense SLM OpenVINO Model and Proof Contract

Status: draft
Owner: intel/openvino
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0004](../proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md)
Linked specs: [BITNET-SPEC-OPENVINO-ROUTE-CONTRACT](BITNET-SPEC-OPENVINO-ROUTE-CONTRACT.md)
Linked ADRs: n/a
Linked plan: [OpenVINO Lunar Lake implementation plan](../../plans/openvino-lunar-lake/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no promotion; defines dense SLM proof ladder
Policy impact: no policy exception

## Purpose

Define the dense SLM contract for OpenVINO on Lunar Lake. This spec covers
Qwen2.5 0.5B Instruct as the first model and future small LLM candidates that
enter the same OpenVINO CPU/GPU/NPU proof ladder.

This spec is a contract for artifacts, receipts, route identity, and promotion
preconditions. It does not run inference, promote OpenVINO GPU or NPU, claim
speedup, claim broad dense SLM quality, prove cold one-off NPU usability, or
prove any BitNet QK256/I2_S behavior.

## Initial Model Family

The first governed OpenVINO dense SLM target is:

```text
source_model: Qwen/Qwen2.5-0.5B-Instruct
model_family: qwen
model_architecture: qwen2
prompt_template: qwen2.5
tokenizer_family: qwen2
export_format: openvino_ir
preferred_weight_format: int4
preferred_quantization: symmetric
model_binary_committed: false
```

Qwen2.5 0.5B is the control model because the repository already carries
Lunar Lake GGUF CPU evidence and OpenVINO CPU/GPU/NPU candidate evidence for
that model family. A later candidate such as Qwen3, SmolLM, Llama, Gemma, or
Phi must start at the manifest step. It cannot inherit Qwen2.5 quality,
timing, promotion, or tokenizer/template proof.

## Artifact Manifest

Every dense SLM OpenVINO candidate needs a manifest before smoke, quality, or
phase receipts can count toward route promotion.

Minimum manifest fields:

```json
{
  "artifact_kind": "openvino_dense_slm_manifest",
  "model_id": "qwen2_5_0_5b_instruct_openvino_int4_sym",
  "source_model": "Qwen/Qwen2.5-0.5B-Instruct",
  "source_revision": "<pinned-revision-or-explicit-unknown>",
  "export_tool": "optimum-cli export openvino",
  "export_command": "optimum-cli export openvino --weight-format int4 --sym --model Qwen/Qwen2.5-0.5B-Instruct <output_folder>",
  "openvino_version": "<version>",
  "optimum_intel_version": "<version-or-explicit-unknown>",
  "format": "openvino_ir",
  "weight_format": "int4",
  "symmetric": true,
  "group_size": 128,
  "ratio": 1.0,
  "model_xml_sha256": "<sha256-or-not-committed>",
  "model_bin_sha256": "<sha256-or-not-committed>",
  "tokenizer_source": "hf_tokenizer_export",
  "tokenizer_sha256": "<sha256-or-explicit-unknown>",
  "chat_template_source": "tokenizer_config",
  "prompt_template": "qwen2.5",
  "context_length": 32768,
  "model_binary_committed": false,
  "accepted_for_lunar_lake": false
}
```

Unknown fields must be explicit. Model binaries must not be committed unless a
future policy item approves that exposure. A changed model, export command,
OpenVINO version, tokenizer, chat template, or quantization resets promotion
eligibility until the proof ladder is rerun for the exact artifact.

## Proof Ladder

Dense SLM OpenVINO work advances through this ladder:

| Stage | Receipt | Required result | Promotion effect |
| --- | --- | --- | --- |
| manifest | artifact/export manifest | exact model/export/tokenizer identity recorded | no promotion |
| smoke | CPU/GPU/NPU route smoke | selected device, runtime API, fallback=false, bounded output | no promotion |
| operator ask | exact route ask receipt | route ID, prompt/template, answer gate, cold timing | candidate evidence only |
| corpus v2 | bounded profile quality receipt | per-case answer gates and failure taxonomy | profile can be considered only if passing |
| phase profile | cold/warm profile timing receipt | prompt/output counts, TTFT, decode, total, cache context | timing can be compared |
| route profile | route-profile comparison | CPU/GPU/NPU compared under named profiles | promotion review input |
| promotion review | route ledger update | quality pass, fallback=false, timing/power/stability advantage | exact profile may promote |
| model status | user-facing capability matrix | current route state and gaps are visible | no new proof by itself |
| exact-profile server | optional server receipt | endpoint identity, same model/profile gates | server proof only |

No stage may skip the prior identity and fallback checks. A passing smoke or
operator ask is not a corpus-v2 quality claim. A passing corpus-v2 receipt is
not a speedup claim. A phase timing receipt is not a quality claim.

## Route Receipts

All dense SLM OpenVINO receipts must comply with
`BITNET-SPEC-OPENVINO-ROUTE-CONTRACT` and include these dense-model fields:

```json
{
  "route_id": "openvino_dense_slm_gpu_arc140v",
  "proof_family": "openvino_dense_slm_gpu_arc140v",
  "requested_backend": "openvino-gpu",
  "selected_backend": "openvino-gpu",
  "runtime_api": "openvino_genai",
  "runtime_device": "GPU.0",
  "resolved_device": "Intel(R) Arc(TM) 140V GPU",
  "fallback_used": false,
  "model_id": "qwen2_5_0_5b_instruct_openvino_int4_sym",
  "model_family": "qwen",
  "model_architecture": "qwen2",
  "prompt_template": "qwen2.5",
  "tokenizer_source": "hf_tokenizer_export",
  "generation_config": {
    "do_sample": false,
    "max_new_tokens": 32
  },
  "bitnet_qk256_proof": false,
  "native_opencl_proof": false
}
```

OpenVINO CPU, GPU, and NPU receipts must not share a single ambiguous route
label. CPU fallback cannot count as GPU or NPU proof. `AUTO` and `HETERO`
receipts are diagnostic unless every execution device is recorded by phase.

## Prompt, Token, and Stop Policy

Dense SLM OpenVINO receipts must record:

- rendered prompt or prompt hash;
- prompt template source;
- tokenizer source and tokenizer hash when available;
- prompt token IDs when direct API access exists;
- prompt token count even when direct IDs are unavailable;
- generated token IDs when direct pipeline-internal IDs are available;
- retokenized generated IDs only if labeled as retokenized;
- decoded output preview;
- stop/EOS policy and whether stop/EOS was observed;
- generation config including greedy/sampling, max new tokens, temperature,
  top-p/top-k when present, and repetition settings when present.

Retokenized generated text is weaker than direct pipeline-internal generated
token IDs and must not be described as direct parity.

## Quality Corpus Requirements

The dense SLM quality lane must use bounded profile cases, not broad chat
claims. The minimum profiles are:

```text
regression_tiny
ask_short
ask_normal
prefill_heavy
decode_heavy
structured
low_power
warm_resident
```

Each case must record:

- case ID, category, profile, and task family;
- prompt token count;
- generated token count;
- decoded answer preview;
- answer gate result;
- scoring result when separate from the gate;
- failure taxonomy when failed;
- stop/EOS behavior;
- route identity and fallback status.

A route can be considered for a profile only if all required cases for that
profile pass or a later spec explicitly marks a case diagnostic-only.

## Phase Timing Requirements

OpenVINO dense SLM phase receipts must separate:

```text
pipeline construction
model or IR load
device compile
cache lookup / cache hit / cache miss
tokenization
prefill or first generate setup
time to first token or first text chunk
decode total
steady decode throughput
total response
memory context
power/thermal context or explicit unavailable reason
```

Every timing profile must include the prompt and output token counts used to
judge profile applicability. Proxy timing can be indexed, but it cannot promote
a route for profiles whose token bounds it does not satisfy.

## NPU Constraints

OpenVINO NPU dense SLM routes are warm/resident candidates until proven
otherwise. Receipts must record:

- INT4 symmetric export status;
- `NPU` selected device and resolved device details;
- greedy/simple generation config;
- no beam search or parallel sampling claim unless a later NPU spec proves it;
- first-ever compile/load timing;
- cached cold-process timing when model cache is used;
- warm second ask timing;
- resident multi-ask timing;
- cache directory and cache mode when available.

Hot-path first-token or decode speed alone cannot promote NPU for cold one-off
ask. Low-power promotion requires quality pass, fallback=false, warm/resident
latency evidence, and measured power or accepted energy-proxy evidence.

## GPU Constraints

OpenVINO GPU dense SLM routes are Arc 140V OpenVINO candidates, not native
OpenCL proof. Receipts must record `GPU.0` plus the resolved Arc 140V device
identity. A GPU route may promote for an exact profile only after quality
passes, fallback=false is recorded, prompt/output token timing is
profile-specific, and the route beats or otherwise justifies replacing the
current CPU default for that profile.

## Promotion Preconditions

A dense SLM OpenVINO route may be promoted only for a named profile when all of
the following are true:

1. The manifest exactly identifies the model/export/tokenizer/template.
2. The selected OpenVINO device matches the requested backend.
3. `fallback_used=false`.
4. The profile's corpus-v2 cases pass or are explicitly excluded by spec.
5. Prompt/template/token/stop policy is recorded.
6. Timing is profile-specific and includes prompt/output token counts.
7. A same-profile CPU comparator exists.
8. The route proves latency, throughput, power, or stability advantage for the
   profile, or an approved policy reason explains the promotion.
9. Known gaps are recorded and do not contradict the promotion.
10. The receipt states what the proof does not cover.

Promotion is exact-profile only. A route promoted for `ask_short` is still a
candidate for `prefill_heavy`, `decode_heavy`, `low_power`, and server use
until those profiles pass separately.

## Model Status Requirements

User-facing status must report, at minimum:

```text
model_id
source_model
export format and quantization
route IDs
route state per profile: promoted / candidate / blocked
last evidence date
fallback status
quality status
phase timing status
token visibility status
promotion blockers
claim boundary
```

Status output is an index. It is not proof unless it links to the underlying
receipts.

## Server Ladder

OpenVINO Model Server or any local server proof is optional and exact-profile
only. Server readiness requires prior ask/corpus/phase proof for the same
model/export/profile and then separate endpoint receipts for:

- model load and selected device;
- request/response schema;
- prompt/template policy;
- fallback=false;
- answer gate;
- first-token or first chunk timing when streaming;
- total latency and throughput;
- concurrency only if explicitly tested.

Server proof must not be generalized to broad server readiness.

## Rejection Examples

| Condition | Required handling |
| --- | --- |
| Manifest lacks source revision and does not mark it unknown | Block model acceptance |
| GPU receipt selected CPU | Fail strict route validation |
| NPU receipt lacks cache/warm/resident split but requests low-power promotion | Block promotion |
| Retokenized generated IDs are recorded as direct internal IDs | Fail or mark non-promotable |
| Corpus-v2 passes one profile and route promotes another | Reject promotion |
| Phase timing lacks prompt token count for numeric profile bounds | Block timing applicability |
| Dense SLM receipt claims BitNet QK256 proof | Fail validation |
| OpenVINO GPU receipt claims native OpenCL proof | Fail validation |
| Server receipt lacks exact profile link | Server proof is diagnostic only |

## Non-Goals

This spec does not:

- promote OpenVINO CPU, GPU, or NPU routes;
- run inference;
- define model-server readiness as complete;
- claim speedup or power advantage;
- prove broad dense SLM quality;
- prove cold one-off NPU usability;
- prove native OpenCL execution;
- prove BitNet QK256/I2_S behavior;
- commit model binaries.

## Acceptance

- Defines the dense SLM model/export contract for Qwen2.5 0.5B Instruct and
  future small LLM candidates.
- Defines the manifest, smoke, operator ask, corpus-v2, phase profile, route
  profile, promotion review, model status, and exact-profile server ladder.
- Requires prompt/template/token/stop policy, fallback status, device identity,
  profile quality, and phase timing evidence before promotion.
- Keeps OpenVINO GPU/NPU candidate routes unpromoted.
- Keeps dense SLM OpenVINO proof separate from BitNet QK256/I2_S proof.
