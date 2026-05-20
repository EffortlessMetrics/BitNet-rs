# BITNET-SPEC-NPU-QUALITY

Status: draft
Proposal: `docs/proposals/BITNET-PROP-0007-npu-productization.md`
Plan: `plans/npu/implementation-plan.md`

## Purpose

Ensure NPU answer routes are useful and governed, not merely fast or visible.
Quality gates apply to dense SLM answer profiles and any later full-inference
candidate route.

## Failure taxonomy

- `exact_answer_overgenerated`
- `exact_answer_instruction_not_followed`
- `missing_required_keyword`
- `forbidden_token_observed`
- `raw_special_token_seen`
- `empty_answer`
- `repetition`
- `stop_policy_failed`
- `context_sensitivity_failed`
- `structured_output_failed`
- `retokenized_token_id_boundary`
- `runtime_error`
- `timeout`
- `cold_compile_timeout`

## Promotion requirements

NPU route promotion requires:

- profile-specific corpus pass,
- `fallback_used=false`,
- selected NPU device,
- prompt/template/tokenizer identity,
- retokenized-vs-direct-token-ID boundary recorded,
- generation config recorded,
- failure taxonomy populated for every failed case,
- explicit not-claims preserved in status surfaces.

## Boundary rule

A route that is fast but fails profile quality is a diagnostic candidate only. It
must not become a default user route or promoted profile.
