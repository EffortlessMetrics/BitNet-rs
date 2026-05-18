# BITNET-SPEC-B158-LARGE-TOKENIZER-PROMPT

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0009 bitnet_b1_58-large control model](../proposals/BITNET-PROP-0009-bitnet-b158-large-control-model.md)
Linked specs: [artifact contract](BITNET-SPEC-B158-LARGE-ARTIFACT-CONTRACT.md), [conversion](BITNET-SPEC-B158-LARGE-CONVERSION.md), [reference quality](BITNET-SPEC-B158-LARGE-REFERENCE-QUALITY.md)
Linked ADRs: n/a
Linked plan: [bitnet_b1_58-large implementation plan](../../plans/bitnet-b158-large/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no support promotion; tokenizer/prompt contract only
Policy impact: no policy exception

## Purpose

Define tokenizer and prompt authority for `1bitLLM/bitnet_b1_58-large`. The
model is older and may not share the exact prompt or Llama-BPE authority story
used for the official Microsoft 2B artifact. BitNet-rs must discover and record
that authority instead of assuming it.

## Required audit fields

A tokenizer/prompt authority receipt must record:

- `tokenizer.json` hash;
- `tokenizer.model` hash;
- `tokenizer_config.json` hash;
- `added_tokens.json` hash when present;
- `special_tokens_map.json` hash;
- model `config.json` hash;
- BOS, EOS, PAD, and UNK token IDs;
- added/special token inventory;
- chat or instruct template when present;
- completion prompt template when no chat template is authoritative;
- stop-token policy;
- reference prompt rendering;
- prompt token IDs for each corpus case;
- tokenizer/pre-tokenizer implementation source;
- whether the runner defaulted, inferred, or explicitly loaded tokenizer data.

## Prompt modes

Receipts must classify each supported mode:

```text
completion_mode
instruction_mode
chat_mode_if_supported
```

If a mode is unsupported or ambiguous, mark it unsupported or blocked. Do not
invent a chat template for convenience.

## Hard rule

Do not use the official Microsoft 2B `bitnetcpp-answer` prompt template unless a
reference-runner prompt-authority audit proves it is correct for
`bitnet_b1_58-large`.

## Promotion dependency

Tokenizer/prompt authority is required before:

- reference-good promotion;
- CPU answer-ready receipts;
- CUDA answer-ready receipts;
- Apple local-answer receipts;
- CLI `ask`/`chat` exposure;
- server exact-profile readiness;
- benchmark comparisons based on generated text quality.
