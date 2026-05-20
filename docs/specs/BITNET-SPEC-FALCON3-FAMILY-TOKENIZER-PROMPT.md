# BITNET-SPEC-FALCON3-FAMILY-TOKENIZER-PROMPT: Falcon3 Tokenizer and Prompt Contract

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

Define the tokenizer, chat-template, prompt rendering, and stop-policy audit required before Falcon3 answer claims. Falcon3 must not copy Microsoft BitNet 2B or Falcon-E prompt authority without evidence.

## Required Audit Fields

```text
tokenizer.json hash
tokenizer.model hash if present
tokenizer_config.json hash
special_tokens_map hash
BOS/EOS/PAD/UNK IDs
chat template text
completion template fallback
stop-token policy
prompt rendering
prompt token IDs
reference-runner command
conversation mode / -cnv policy
```

## Prompt Receipt Requirements

Each tokenizer/prompt authority receipt must bind:

- source repository and revision;
- tokenizer file names, byte sizes, and SHA256 hashes;
- exact chat template text or explicit absence;
- rendered prompts for every corpus item;
- prompt token IDs for every corpus item;
- stop-token IDs and decoded stop strings;
- reference runner and command line;
- deterministic generation settings used for prompt verification.

## Hard Rules

```text
Do not assume Microsoft BitNet 2B prompt authority applies to Falcon3.
Do not assume Falcon-E prompt authority applies to Falcon3.
No tokenizer/prompt authority, no answer claim.
A changed tokenizer file, chat template, stop policy, or conversation mode resets answer-proof eligibility.
```
