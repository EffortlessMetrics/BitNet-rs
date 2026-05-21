# Kaby SLM Swarm Artifact Intake Gate

Status: `SLM-CPU-063`

This document defines how Kaby Lake SLM performance work produced outside this
release repository can be promoted back into BitNet-rs. Development experiments
for the next packed Q8_0 compute candidate live in `bitnet-rs-swarm`; BitNet-rs
remains the release and evidence surface.

## Scope

This gate accepts only audited Kaby Lake SLM artifacts for the established
small dense GGUF CPU lane:

```text
host = Intel Core i5-8250U
backend = cpu-rust
primary oracle = Qwen3-0.6B Q8_0 appliance profile
secondary sanity = Qwen2.5-0.5B Q8_0 strict CPU receipt
fallback_used = false
```

The gate is for release-surface intake. It does not implement runtime compute,
enable packed Q8_0 sidecar execution by default, or promote a speed claim by
itself.

## Required Package

A returned artifact package must include:

```text
candidate_summary.json
before_receipt.json
after_receipt.json
equivalence_report.json
timing_report.json or timing_not_claimed.json
source_commit.txt
```

The package may include additional diagnostics, but the files above are the
minimum release-surface evidence.

## Candidate Summary

`candidate_summary.json` must record:

```text
schema
artifact_kind = slm_cpu_swarm_candidate
source_repo = bitnet-rs-swarm
source_commit
model_sha256
model_family
quant_format
candidate_path
selected_cpu_backend
selected_kernel
dense_hook_selection
packed_q8_tensor_scope
runtime_enabled_by_default = false
speedup_claim = false unless timing_report.json is accepted separately
```

The `packed_q8_tensor_scope` must be exact. A candidate for one tensor path
cannot be described as general packed Q8_0 support.

## Before/After Equivalence

The before and after receipts must use the same:

```text
model SHA
model file identity
tokenizer source
tokenizer strictness
prompt template policy
prompt IDs
generated IDs
decoded text
selected CPU backend
selected kernel
dense hook-selection identity
fallback_used = false
```

The equivalence report must fail closed if any of those fields drift.

Allowed differences are limited to:

```text
candidate implementation identity
bounded timing fields
allocation counters
diagnostic counters
explicitly unavailable host telemetry fields
```

## Timing Evidence

Timing evidence is optional for intake. If present, it must be bounded to:

```text
host = Intel Core i5-8250U
model = verified Qwen3-0.6B Q8_0 GGUF
thread count
prompt corpus
generated token count
power mode when available
thermal fields present or explicitly unavailable
storage/free-space context present
```

Timing evidence must separate:

```text
cold load
warm-session load-once
prefill
first-token latency
steady decode
per-prompt wall time
allocation counters where available
```

No sustained-throughput claim is accepted unless cold/warm context, thread
count, token counts, memory, and thermal/power fields are present or explicitly
unavailable.

## Validation Checklist

Before a candidate can be promoted into BitNet-rs, the reviewer must verify:

```text
same model SHA
same strict GGUF tokenizer authority
same prompt IDs
same generated IDs
same decoded text
same selected CPU backend/kernel identity
same fallback_used=false
same dense hook-selection identity or an explicitly approved narrowed selector
no broad answer-quality claim
no sustained-throughput claim
no Q4/Q5 runtime claim
no server/GPU/NPU/OpenVINO/UHD 620 claim
no Qwen3.5 or hybrid architecture claim
no BitNet QK256/I2_S change
```

If any required equivalence field differs, the candidate remains a swarm-side
experiment and cannot be promoted into BitNet-rs.

## Promotion Path

A valid intake PR may touch:

```text
docs/slm/**
docs/tracking/campaigns/slm-cpu/**
docs/tracking/generated/**
ci/slm-cpu/** only for committed release/evidence artifacts
```

Runtime code changes in BitNet-rs require a separate item after this intake gate
accepts the evidence package. That later item must repeat the same before/after
receipt discipline and preserve the Qwen3 Q8_0 behavior oracle.

## Rejection Conditions

Reject the artifact package if it:

```text
changes generated IDs or decoded text
hides fallback behavior
uses a guessed tokenizer source
omits model SHA
omits prompt IDs
claims broad chat quality
claims portable CPU performance
claims sustained throughput without bounded telemetry
promotes Q4/Q5 as supported before the Q4 gates pass
mixes in server, GPU, NPU, OpenVINO, UHD 620, Qwen3.5, or BitNet QK256 work
```

This keeps BitNet-rs boring: release artifacts come in only when they preserve
the strict CPU proof contract.

## First Intake Review

`SLM-CPU-064` is the first release-surface review item for an audited package
returned from `bitnet-rs-swarm`. It may accept, reject, or block a package
against this gate, but it does not promote runtime code into BitNet-rs by
itself.

If the package passes, the follow-up runtime promotion item must still be
separate and must repeat the same before/after receipt discipline.

## SLM-CPU-064 Review

The first returned package is committed under:

```text
ci/slm-cpu/intel-i5-8250u/2026-05-20/slm-cpu-062-swarm-export/
```

The release-surface review is:

```text
ci/slm-cpu/intel-i5-8250u/2026-05-20/slm-cpu-064-swarm-intake-review.json
```

The package is accepted as release-surface evidence only. It preserves model
SHA, strict GGUF tokenizer authority, prompt IDs, generated IDs, decoded text,
selected CPU backend/kernel identity, dense hook-selection identity, and
`fallback_used=false`; it also keeps packed Q8_0 runtime compute disabled and
records `speedup_claim=false`.

Runtime promotion remains a separate follow-up item. `SLM-CPU-065` is the
first queued promotion gate, limited to the exact single-tensor packed Q8_0
sidecar candidate accepted through this review. It must keep runtime promotion
disabled by default unless before/after strict CPU receipts prove identical
behavior.
