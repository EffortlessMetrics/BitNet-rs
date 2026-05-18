# BITNET-SPEC-OPENVINO-NPU-COLD-WARM-CACHE: NPU Cold, Cache, Warm, and Resident Contract

Status: draft
Owner: intel/openvino
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0004](../proposals/BITNET-PROP-0004-openvino-lunar-lake-productization.md)
Linked specs: [BITNET-SPEC-OPENVINO-ROUTE-CONTRACT](BITNET-SPEC-OPENVINO-ROUTE-CONTRACT.md), [BITNET-SPEC-OPENVINO-DENSE-SLM](BITNET-SPEC-OPENVINO-DENSE-SLM.md)
Linked ADRs: n/a
Linked plan: [OpenVINO Lunar Lake implementation plan](../../plans/openvino-lunar-lake/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no promotion; defines NPU timing and residency proof gates
Policy impact: no policy exception

## Purpose

Define the OpenVINO NPU timing contract for Lunar Lake dense SLM routes. This
spec separates first-ever cold startup, cached process startup, warm same-process
asks, and resident-session operation so NPU hot-path evidence cannot be mistaken
for cold one-off usability.

This spec does not run inference, promote NPU, claim NPU speedup, claim NPU
power advantage, prove broad dense SLM quality, prove dynamic decode support,
or prove BitNet QK256/I2_S behavior.

## Route Scope

This contract applies to:

```text
route_id = openvino_dense_slm_npu
proof_family = openvino_dense_slm_npu
runtime_api = openvino_genai
runtime_device = NPU
model_family = dense_slm
```

The initial governed model is the exact OpenVINO IR dense SLM export defined by
`BITNET-SPEC-OPENVINO-DENSE-SLM`. Future small LLMs must satisfy the same
model/export contract before their NPU timing receipts can count toward route
promotion review.

## Timing Modes

Receipts must classify NPU timing into one of these modes:

| Mode | Meaning | May support |
| --- | --- | --- |
| `first_ever_cold` | No trusted prior compiled artifact or process-resident pipeline | cold-start diagnosis only |
| `cached_cold_process` | New process with an explicit OpenVINO cache directory and stable exported model path | cold-with-cache comparison |
| `warm_same_process` | Second or later ask using the same process and already constructed pipeline | warm interactive profile evidence |
| `resident_session` | Long-lived route that keeps the NPU pipeline loaded across a bounded request loop | warm/resident route evidence |

Receipts must not collapse these modes into a single `load_ms` or
`total_response_ms` value. A route can be fast in `warm_same_process` and still
blocked for `first_ever_cold`.

## Required Receipt Fields

All NPU cold/cache/warm receipts must include the OpenVINO route identity fields
from `BITNET-SPEC-OPENVINO-ROUTE-CONTRACT` plus:

```json
{
  "artifact_kind": "openvino_npu_cold_warm_cache",
  "route_id": "openvino_dense_slm_npu",
  "proof_family": "openvino_dense_slm_npu",
  "requested_backend": "openvino-npu",
  "selected_backend": "openvino-npu",
  "runtime_api": "openvino_genai",
  "runtime_device": "NPU",
  "resolved_device": "Intel(R) AI Boost",
  "fallback_used": false,
  "model_id": "qwen2_5_0_5b_instruct_openvino_int4_sym",
  "timing_mode": "first_ever_cold",
  "profile": "ask_short",
  "cold_start_policy": "blocked_for_one_off|candidate_with_cache|warm_only",
  "cache": {
    "cache_dir": "<path-or-none>",
    "cache_enabled": true,
    "cache_writable": true,
    "cache_key_basis": ["model_xml_sha256", "model_bin_sha256", "openvino_version", "device", "config"],
    "cache_hit": false,
    "cache_evidence": "<runtime-log|file-mtime|explicit-unavailable>"
  },
  "genai_config": {
    "prefill_hint": "<value-or-unset>",
    "generate_hint": "<value-or-unset>",
    "max_prompt_len": 512,
    "min_response_len": 1,
    "max_new_tokens": 32,
    "sampling": "greedy",
    "beam_search": false,
    "parallel_sampling": false
  },
  "phase_ms": {
    "process_start": null,
    "pipeline_construct": null,
    "tokenizer_load": null,
    "model_read": null,
    "compile_model": null,
    "cache_lookup": null,
    "first_generate_call": null,
    "first_token": null,
    "decode": null,
    "total_response": null
  },
  "quality": {
    "answer_gate_passed": false,
    "corpus_v2_profile_passed": false
  },
  "promotion": {
    "status": "candidate",
    "eligible_profiles": [],
    "blocked_for": ["cold_one_off"],
    "blockers": []
  }
}
```

Unknown fields must be explicit. If OpenVINO GenAI cannot expose an internal
phase such as `compile_model` or `cache_lookup`, the receipt must record
`phase_source = "not_exposed"` or an equivalent gap, not infer the phase from
total latency.

## Cache Contract

NPU cache evidence must bind to an exact model/export/runtime/device tuple.
Minimum cache identity:

```json
{
  "openvino_version": "<version>",
  "driver_version": "<version-or-unavailable>",
  "model_xml_sha256": "<sha256>",
  "model_bin_sha256": "<sha256>",
  "tokenizer_sha256": "<sha256-or-unavailable>",
  "device": "NPU",
  "cache_dir": "<absolute-or-repo-relative-path>",
  "cache_policy": "enabled|disabled|unavailable",
  "cache_permission": "writable|read_only|missing",
  "cache_hit_evidence": "runtime_metric|runtime_log|file_reuse|not_available"
}
```

Any change to model files, tokenizer files, OpenVINO version, driver, device
selection, or compile configuration invalidates prior cache timing for promotion
review. If the cache directory is not writable, the route remains blocked for
cached cold-start claims and may only use warm/resident evidence.

## Generation Config Contract

NPU receipts must record the GenAI and device configuration that affects compile
shape, prompt capacity, and generation behavior:

- `PREFILL_HINT` value or explicit `unset`;
- `GENERATE_HINT` value or explicit `unset`;
- `MAX_PROMPT_LEN` value and whether the profile prompt fits it;
- `MIN_RESPONSE_LEN` value and whether it affects stop behavior;
- `max_new_tokens` and profile output-token budget;
- greedy sampling only unless a later spec approves more modes;
- `beam_search=false`;
- `parallel_sampling=false`.

If a receipt uses a static or bounded compile shape, it must record whether the
profile prompt and expected response fit that shape. A profile cannot promote
from timing evidence gathered outside its prompt/output token bounds.

## Mode-Specific Acceptance

### First-Ever Cold

First-ever cold receipts must capture:

- pipeline construction and model read timing where available;
- NPU compile or explicit compile-not-exposed gap;
- cache disabled or cache miss evidence;
- total response timing;
- answer gate result;
- `fallback_used=false`.

First-ever cold evidence may diagnose cost. It must not promote NPU for
cold one-off asks unless a later promotion review proves both quality and a
material latency or power advantage against the exact CPU profile.

### Cached Cold Process

Cached cold-process receipts must run a new process after an explicit cache
priming run and record:

- cache directory identity and writability;
- cache hit or explicit cache-hit-unavailable evidence;
- total response timing compared with first-ever cold;
- same model/export and generation config;
- answer gate result and fallback status.

If the second process does not materially improve or cache-hit evidence is
unavailable, the NPU remains blocked for cached cold-start claims.

### Warm Same-Process

Warm same-process receipts must run at least:

```text
ask_1 = cold or cache-primed setup ask
ask_2 = warm same-process ask
```

They must report the first and second ask separately. The second ask may count
toward warm interactive evidence only if:

- the pipeline object is reused;
- no model reload or recompile is observed, or the gap is explicit;
- answer gate passes;
- `fallback_used=false`;
- timing fits the named profile.

### Resident Session

Resident-session receipts must run a bounded request loop:

```text
warmup: 1 request
measurement: at least 10 requests for the profile under test
```

They must record:

- per-request answer gate status;
- fallback drift;
- route/device drift;
- first-token and decode timing distribution;
- memory growth;
- power/thermal context or explicit unavailable gap;
- whether the session stayed resident for the full run.

Resident evidence can support only warm/resident/low-power promotion reviews. It
does not prove cold one-off usability.

## Low-Power Promotion Inputs

Low-power NPU promotion requires more than hot decode timing. A promotion review
must have:

- corpus-v2 pass for the exact profile;
- `fallback_used=false`;
- warm or resident timing for the exact profile;
- cold-start caveat recorded in the route reason;
- power or energy proxy evidence compared against CPU and GPU;
- memory/thermal context or explicit blocker;
- no beam search, no parallel sampling, and no unsupported dynamic decode claim.

If power telemetry is unavailable, a route may remain a low-power candidate but
must not claim a power advantage.

## Route Reason Requirements

Any route ledger or operator receipt that selects or rejects NPU must explain:

```json
{
  "route_id": "openvino_dense_slm_npu",
  "profile": "warm_resident",
  "promotion_status": "candidate|promoted|blocked",
  "route_reason": "<why selected or not selected>",
  "why_not_cpu": "<required when NPU selected>",
  "why_not_gpu": "<required when NPU selected>",
  "why_not_npu": "<required when NPU not selected>",
  "cold_start_caveat": "<required for any warm/resident NPU selection>"
}
```

A user-facing `auto` route must not choose NPU from hot-path evidence alone.

## Rejection Examples

| Evidence | Required decision |
| --- | --- |
| NPU first-token/decode is fast but cold startup is 40 seconds and no resident proof exists | Candidate only; no cold or warm promotion |
| NPU receipt lacks cache directory and cache hit/miss evidence | No cached cold-start claim |
| NPU warm loop passes quality but lacks power evidence | May support warm latency review; no low-power claim |
| AUTO receipt does not expose actual execution device by phase | Diagnostic only; no selected-device proof |
| NPU corpus-v2 passes `ask_short` but timing was measured on a 9-token smoke prompt | Block `ask_short` promotion until profile timing applies |
| NPU receipt uses beam search or parallel sampling | Reject for current NPU route promotion |

## Non-Goals

This spec does not:

- define OpenVINO GPU promotion;
- define dense SLM corpus-v2 quality gates;
- define the full route-promotion ledger;
- claim NPU cold one-off usability;
- claim NPU speedup or power advantage;
- claim dynamic decode support;
- claim packed BitNet QK256 decode or full BitNet NPU inference;
- approve committing model binaries;
- define server readiness.

## Acceptance

This spec is complete when:

1. NPU first-ever cold, cached cold-process, warm same-process, and
   resident-session modes are defined separately.
2. Required cache, GenAI configuration, phase timing, quality, route, and
   promotion fields are listed.
3. `PREFILL_HINT`, `GENERATE_HINT`, `MAX_PROMPT_LEN`, and `MIN_RESPONSE_LEN`
   handling is explicit.
4. NPU low-power promotion inputs require quality, fallback-free execution,
   warm/resident timing, cold-start caveats, and power/energy evidence.
5. The spec blocks cold one-off NPU usability claims from hot-path timing alone.
6. Dense SLM NPU proof remains separate from BitNet QK256/I2_S proof.
