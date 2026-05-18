# BITNET-SPEC-B158-3B-REFERENCE-QUALITY

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0010](../proposals/BITNET-PROP-0010-bitnet-b158-3b-tl-model.md)
Linked specs: [3B conversion](BITNET-SPEC-B158-3B-CONVERSION.md), [3B tokenizer/prompt](BITNET-SPEC-B158-3B-TOKENIZER-PROMPT.md)
Linked ADRs: n/a
Linked plan: [3B TL implementation plan](../../plans/bitnet-b158-3b/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no support promotion; reference-quality contract only
Policy impact: no policy exception

## Purpose

Define the reference-good gate that must pass before Rust CPU or accelerator
work can claim useful answers for the 3B lane. Reference quality is
route-specific and artifact-specific.

## Reference levels

| Corpus | Purpose |
| --- | --- |
| `tiny_smoke` | Can the model answer at all? |
| `answer_corpus_v1` | Bounded answer-ready gate. |
| `behavior_suite_v1` | Prompt conditioning, stop behavior, non-repetition. |
| `long_decode_v1` | Stability and warm-session behavior. |

## Minimum reference cases

- `math_2_plus_2`
- `capital_france`
- `copy_color_sequence`
- `yes_no_clear_sky`
- `short_continuation`
- `prompt_conditioning_pair`
- `stop_token_behavior`
- `special_token_garbage_check`

## Pass criteria

A reference-quality receipt must show:

- non-empty output;
- printable UTF-8;
- no raw special-token garbage;
- no uncontrolled repetition;
- constrained answers satisfy gates;
- prompt-conditioned pair changes appropriately;
- stop policy is respected;
- deterministic decoding configuration is recorded.

## Hard rules

- Reference-good output requires an approved artifact or approved reference
  route, tokenizer authority, prompt authority, and deterministic settings.
- Transformers, vLLM, SGLang, or bitnet.cpp reference success does not prove
  BitNet-rs Rust CPU support.
- Reference-good for TL1 does not prove TL2. Reference-good for TL2 does not
  prove TL1.
