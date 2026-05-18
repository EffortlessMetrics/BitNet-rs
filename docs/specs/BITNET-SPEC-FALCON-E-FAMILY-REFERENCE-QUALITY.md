# BITNET-SPEC-FALCON-E-FAMILY-REFERENCE-QUALITY

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: docs/proposals/BITNET-PROP-0013-falcon-e-family-supported-models.md
Linked specs: docs/specs/BITNET-SPEC-FALCON-E-FAMILY-TOKENIZER-PROMPT.md
Linked ADRs: n/a
Linked plan: plans/falcon-e-family/implementation-plan.md
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: reference quality only; CPU/backend claims still require runtime receipts
Policy impact: no policy exception

## Reference levels

| Corpus | Purpose |
|---|---|
| `tiny_smoke` | Can the artifact answer at all? |
| `answer_corpus_v1` | Bounded answer-ready gate. |
| `behavior_suite_v1` | Prompt conditioning, stop behavior, repetition. |
| `long_decode_v1` | Warm-session and stability. |

## Minimum cases

```text
math_2_plus_2
capital_france
copy_color_sequence
yes_no_clear_sky
short_continuation
prompt_conditioning_pair
chat_or_conversation_mode_sanity
stop_token_behavior
special_token_garbage_check
```

## Pass criteria

```text
non-empty output
printable UTF-8
no raw special-token garbage
no uncontrolled repetition
constrained answers satisfy gates
prompt-conditioned pairs differ appropriately
stop policy respected
```

## Required receipt fields

Reference-quality receipts must record artifact SHA256, tokenizer/prompt
authority receipt IDs, reference command, runner version, decode parameters,
prompt text, prompt token IDs, generated IDs, decoded output, pass/fail per
case, failure taxonomy, and cleanup status.

## Hard rules

Reference-good for Falcon-E 1B does not prove Falcon-E 3B. Reference-good does
not prove Rust CPU, CUDA, Apple, A770, speed, server, or full-residency support.
