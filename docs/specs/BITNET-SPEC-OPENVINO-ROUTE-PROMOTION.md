# BITNET-SPEC-OPENVINO-ROUTE-PROMOTION: OpenVINO Route Promotion Contract

Status: draft
Owner: intel/openvino
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0004](../proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md)
Linked specs: [BITNET-SPEC-OPENVINO-ROUTE-CONTRACT](BITNET-SPEC-OPENVINO-ROUTE-CONTRACT.md), [BITNET-SPEC-OPENVINO-DENSE-SLM](BITNET-SPEC-OPENVINO-DENSE-SLM.md), [BITNET-SPEC-OPENVINO-QUALITY-CORPUS](BITNET-SPEC-OPENVINO-QUALITY-CORPUS.md), [BITNET-SPEC-OPENVINO-PHASE-TIMING](BITNET-SPEC-OPENVINO-PHASE-TIMING.md), [BITNET-SPEC-OPENVINO-NPU-COLD-WARM-CACHE](BITNET-SPEC-OPENVINO-NPU-COLD-WARM-CACHE.md)
Linked ADRs: n/a
Linked plan: [OpenVINO Lunar Lake implementation plan](../../plans/openvino-lunar-lake/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no promotion; defines promotion review gates
Policy impact: no policy exception

## Purpose

Define how an OpenVINO Lunar Lake route moves from candidate to promoted for a
specific workload profile. This contract prevents visibility, smoke tests,
bounded asks, or hot-path timing from becoming broad route claims.

This spec does not promote OpenVINO CPU, GPU, or NPU routes; run benchmarks;
claim speedup; claim power advantage; claim broad dense SLM quality; or prove
BitNet QK256/I2_S behavior.

## Route States

Route ledgers must use these states:

| State | Meaning | User-facing routing effect |
| --- | --- | --- |
| `proposed` | route or profile exists only as a plan | never selected |
| `candidate` | route has some evidence but lacks promotion package | selectable only by explicit request |
| `promoted` | route passed promotion review for exact profiles | eligible for `--device auto` on those profiles |
| `blocked` | route has a known failed gate or policy blocker | never selected automatically |
| `retired` | route was superseded or invalidated | never selected |

`candidate` is not a weak promotion. Candidate routes must remain out of
automatic selection unless the user explicitly requests them.

## Exact-Profile Rule

Promotion is exact-profile only. A route promoted for `ask_short` remains a
candidate or blocked for:

```text
ask_normal
prefill_heavy
decode_heavy
structured
low_power
warm_resident
server
BitNet
```

unless each profile has its own promotion package.

The canonical profile IDs are defined by
`BITNET-SPEC-OPENVINO-PHASE-TIMING`. Route ledgers must not invent profile
aliases such as `fast`, `gpu_default`, or `npu_mode` without a spec update.

## Promotion Package

A route can be promoted only when one machine-readable package links:

```json
{
  "artifact_kind": "openvino_route_promotion_review",
  "route_id": "openvino_dense_slm_gpu_arc140v",
  "profile": "ask_normal",
  "status": "promoted|candidate|blocked",
  "model_id": "qwen2_5_0_5b_instruct_openvino_int4_sym",
  "quality_receipt": "<path>",
  "phase_timing_receipt": "<path>",
  "telemetry_receipt": "<path-or-unavailable>",
  "cpu_reference_receipt": "<path>",
  "regression_bundle": "<path>",
  "route_profile_comparison": "<path>",
  "fallback_used": false,
  "answer_gate_passed": true,
  "timing_applicable": true,
  "advantage": {
    "latency": "proven|not_proven|not_claimed",
    "throughput": "proven|not_proven|not_claimed",
    "power": "proven|not_proven|not_claimed",
    "stability": "proven|not_proven|not_claimed"
  },
  "route_reason": "<why this route should or should not be selected>",
  "known_gaps": []
}
```

The promotion package may be embedded in the route-promotion ledger, but the
ledger must still link every evidence receipt by path.

## Promotion Gates

All promoted routes require:

1. Exact model/export/tokenizer/template identity.
2. Exact workload profile.
3. Quality corpus pass for that profile.
4. `fallback_used=false` for every required receipt.
5. Phase timing that satisfies profile prompt/output token bounds.
6. Same-profile CPU default or reference comparison.
7. Regression bundle coverage.
8. Route reason and known gaps.
9. Claim-boundary fields proving no dense-SLM-to-BitNet or OpenVINO-to-native
   kernel leakage.

At least one advantage must be proven or explicitly accepted by policy:

- lower total response latency;
- higher decode or total throughput;
- lower power or energy proxy;
- better repeated-run stability;
- explicit product policy for a specialized profile.

Without one of those advantages, a correct accelerator route remains a
candidate.

## GPU Promotion

OpenVINO GPU / Arc 140V promotion requires:

- `route_id = openvino_dense_slm_gpu_arc140v`;
- selected device identity resolves to Arc 140V / `GPU.0` unless a later spec
  adds a different exact GPU;
- quality corpus pass for the exact profile;
- profile-applicable timing;
- comparison against Qwen CPU or the current default route;
- `fallback_used=false`;
- no native OpenCL claim unless a separate native OpenCL proof is linked.

GPU promotion can be for `ask_short`, `ask_normal`, `prefill_heavy`,
`decode_heavy`, or `structured`, but each profile needs separate evidence.

## NPU Promotion

OpenVINO NPU promotion requires all GPU-style dense SLM gates plus the NPU
contract:

- `route_id = openvino_dense_slm_npu`;
- selected device identity resolves to NPU / Intel AI Boost;
- `BITNET-SPEC-OPENVINO-NPU-COLD-WARM-CACHE` timing mode is recorded;
- no beam search;
- no parallel sampling;
- no unsupported dynamic decode claim;
- cold-start caveat is explicit for warm/resident selections;
- power or energy evidence is present for `low_power` claims.

NPU can be promoted for warm/resident profiles while remaining blocked for cold
one-off asks.

## CPU Promotion

The dense SLM CPU default remains promoted only while:

- its corpus-v2 quality gates pass for the promoted profiles;
- fallback is false;
- phase timing remains recorded;
- no semantic or runtime change invalidates the current evidence;
- no accelerator route has a stronger promotion package for the same profile.

CPU promotion is not an acceleration claim.

## Invalidation Rules

Any of these changes reset promotion eligibility for affected routes:

- model artifact or export command changes;
- tokenizer, chat template, or stop policy changes;
- generation config changes;
- OpenVINO version or driver changes for accelerator routes;
- profile prompt/output bounds change;
- route receipt schema changes;
- quality scorer changes;
- fallback appears;
- semantic fixes land in shared tokenizer, loader, model, sampler, or BitNet
  runtime paths.

After invalidation, the route must return to `candidate` or `blocked` until its
promotion package is rerun.

## Auto-Route Contract

`--device auto` may select only routes whose ledger state is `promoted` for the
requested profile.

Every auto-route receipt must include:

```json
{
  "requested_device": "auto",
  "selected_route": "dense_slm_default_cpu",
  "profile": "ask_normal",
  "promotion_status": "promoted",
  "fallback_used": false,
  "route_reason": "<why selected>",
  "why_not_cpu": "<required when CPU not selected>",
  "why_not_gpu": "<required when GPU not selected>",
  "why_not_npu": "<required when NPU not selected>"
}
```

If no accelerator route is promoted for the profile, auto must choose the
promoted CPU route or fail closed.

## Rejection Examples

| Evidence | Required decision |
| --- | --- |
| GPU smoke passes and timing looks faster | Candidate only; no promotion without corpus/profile evidence |
| NPU hot decode is fast but cold/cache/resident evidence is missing | Candidate only |
| GPU passes `regression_tiny` but not `ask_normal` | Promote at most `regression_tiny` |
| Timing fits `ask_short` but quality passed only on a different profile | Block promotion |
| Power telemetry unavailable for low-power route | No power-advantage claim |
| Route has one fallback case | Block promotion |
| Dense SLM route is promoted | Do not claim BitNet QK256/I2_S behavior |

## Acceptance

This spec is complete when it defines:

1. Route states and exact-profile promotion rules.
2. Required promotion package fields and evidence links.
3. CPU, GPU, and NPU promotion gates.
4. Invalidation and auto-route behavior.
5. Rejection examples that preserve candidate-only accelerator boundaries.
