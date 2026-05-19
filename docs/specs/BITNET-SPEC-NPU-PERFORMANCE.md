# BITNET-SPEC-NPU-PERFORMANCE

Status: draft
Proposal: `docs/proposals/BITNET-PROP-0007-npu-productization.md`
Plan: `plans/npu/implementation-plan.md`

## Purpose

Define NPU performance and efficiency evidence. NPU performance claims must be
profile-specific, quality-gated, fallback-strict, and split by cold/cache/warm
phase.

## Profiles

- `tiny_graph_smoke`
- `static_subgraph_parity`
- `dense_slm_bounded_ask`
- `ask_short`
- `ask_normal`
- `prefill_heavy`
- `decode_heavy`
- `warm_second_ask`
- `resident_10x_ask_short`
- `resident_25`
- `low_power_short_answer`
- `server_nonstream_exact_profile`

## Required timing and context

Receipts must record:

- model/export load,
- OpenVINO runtime init,
- pipeline construct,
- compile,
- cache hit/miss,
- first token,
- decode total,
- steady tok/s,
- total response,
- power context,
- thermal context,
- NPU utilization if available,
- memory context.

## Promotion requirements

- `quality_passed=true`.
- `fallback_used=false`.
- `profile_timing_applicable=true`.
- Same-profile CPU/GPU comparator recorded.
- Cold/cache/warm split recorded.
- Power/thermal context recorded or explicit unavailable reason provided.

## Boundary rule

Do not claim broad NPU speedup. Claim only the exact model + route + profile
that passed quality and timing gates.
