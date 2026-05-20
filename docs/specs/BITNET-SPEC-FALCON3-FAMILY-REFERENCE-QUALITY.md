# BITNET-SPEC-FALCON3-FAMILY-REFERENCE-QUALITY: Falcon3 Reference Quality Contract

Status: draft
Owner: model-artifacts
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0012](../proposals/BITNET-PROP-0012-falcon3-family-supported-models.md)
Linked specs: n/a
Linked ADRs: [BITNET-ADR-0005](../adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md)
Linked plan: [Falcon3 family implementation plan](../../plans/falcon3-family/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: defines future gates only; no promotion
Policy impact: no policy exception

## Purpose

Define reference-runner quality gates before any Rust CPU or accelerator claim. Reference-good is a prerequisite, not a product claim.

## Corpus Levels

| Corpus | Purpose |
| --- | --- |
| `tiny_smoke` | Can the artifact answer at all? |
| `answer_corpus_v1` | Bounded answer-ready gate. |
| `behavior_suite_v1` | Prompt conditioning, stop behavior, non-repetition. |
| `long_decode_v1` | Warm-session and stability. |

## Minimum Cases

```text
math_2_plus_2
capital_france
copy_color_sequence
yes_no_clear_sky
short_continuation
prompt_conditioning_pair
chat_template_sanity
stop_token_behavior
special_token_garbage_check
```

## Pass Criteria

```text
non-empty output
printable UTF-8
no raw special-token garbage
no uncontrolled repetition
constrained answers satisfy gates
prompt-conditioned pairs differ appropriately
stop policy respected
```

## Receipt Fields

Reference-quality receipts must record artifact ID, source revision, artifact hash, tokenizer/prompt authority receipt, reference-runner command, deterministic generation settings, prompt IDs, generated IDs when available, decoded text, case-level pass/fail, and claim boundary with CPU/backend/speed/server readiness false.
