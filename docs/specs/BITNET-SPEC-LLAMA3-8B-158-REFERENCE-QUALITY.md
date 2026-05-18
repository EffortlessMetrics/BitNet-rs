# BITNET-SPEC-LLAMA3-8B-158-REFERENCE-QUALITY

Status: proposed
Owner: model-artifacts
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0011](../proposals/BITNET-PROP-0011-llama3-8b-158-supported-model.md)
Linked plan: [Llama3 8B 1.58 implementation plan](../../plans/llama3-8b-158/implementation-plan.md)
Support-tier impact: reference-good candidate only after corpus passes
Policy impact: no policy exception

## Purpose

Define reference-quality success before Rust CPU or accelerator answer claims.

## Reference levels

| Corpus | Purpose |
| --- | --- |
| `tiny_smoke` | Can it answer at all? |
| `answer_corpus_v1` | Bounded answer-ready gate. |
| `behavior_suite_v1` | Prompt conditioning, stop behavior, repetition. |
| `long_decode_v1` | Stability and warm-session behavior. |

## Minimum cases

The suite must include `math_2_plus_2`, `capital_france`,
`copy_color_sequence`, `yes_no_clear_sky`, `short_continuation`,
`prompt_conditioning_pair`, Llama3 chat-template sanity, `stop_token_behavior`,
and `special_token_garbage_check`.

## Pass criteria

Output must be non-empty, printable UTF-8, free of raw special-token garbage,
free of uncontrolled repetition, constrained-answer correct, prompt-conditioned
where required, and stop-policy compliant.

## Hard rules

Reference-good requires an approved artifact or approved runner path. A loading
receipt, conversion receipt, or tokenizer audit alone does not prove answer
quality.
