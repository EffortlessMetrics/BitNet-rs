# BITNET-SPEC-FALCON-E-FAMILY-TOKENIZER-PROMPT

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: docs/proposals/BITNET-PROP-0013-falcon-e-family-supported-models.md
Linked specs: docs/specs/BITNET-SPEC-FALCON-E-FAMILY-ARTIFACT-CONTRACT.md
Linked ADRs: n/a
Linked plan: plans/falcon-e-family/implementation-plan.md
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no answer promotion until tokenizer/prompt authority passes
Policy impact: no policy exception

## Prompt evidence boundary

The Falcon-E model cards show a BitNet-style conversation-mode example with
`ggml-model-i2_s.gguf`; that is prompt-mode evidence, not BitNet-rs prompt
authority. BitNet-rs must audit tokenizer and prompt behavior before answer
claims.

## Required audit

```text
GGUF tokenizer metadata
tokenizer.json if available
tokenizer.model if available
tokenizer_config if available
BOS/EOS/PAD/UNK IDs
chat template if embedded
conversation mode / -cnv policy
completion prompt fallback
stop-token policy
prompt rendering
prompt token IDs
reference-runner command
```

## Receipts

Tokenizer/prompt receipts must include artifact ID, source revision, tokenizer
source, pre-tokenizer source, chat template source, rendered prompts, prompt
IDs, stop IDs, decode configuration, reference runner, and a claim boundary that
keeps `answer_ready=false` until reference-quality gates pass.

## Hard rules

```text
Do not assume Microsoft BitNet 2B bitnetcpp-answer prompt authority applies to Falcon-E.
Do not assume Falcon3 prompt authority applies to Falcon-E.
Do not claim answer readiness from a model-card command alone.
```
