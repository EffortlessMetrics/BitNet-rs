# BITNET-SPEC-LLAMA3-8B-158-TOKENIZER-PROMPT

Status: proposed
Owner: model-artifacts
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0011](../proposals/BITNET-PROP-0011-llama3-8b-158-supported-model.md)
Linked plan: [Llama3 8B 1.58 implementation plan](../../plans/llama3-8b-158/implementation-plan.md)
Support-tier impact: no answer promotion until audit passes
Policy impact: no policy exception

## Purpose

Define tokenizer and prompt authority for the Llama3-derived BitNet-family
candidate. The model must not inherit the official Microsoft 2B
`bitnetcpp-answer` prompt template without proof.

## Required audit

The audit must record `tokenizer.json` hash, `tokenizer_config.json` hash, chat
template text, BOS/EOS/PAD/UNK IDs, special tokens, generation config, reference
prompt render, prompt token IDs, stop-token policy, whether a Meta Llama3
tokenizer is used externally, whether the model repo tokenizer is authoritative,
and whether chat mode or completion mode is intended.

## Prompt modes to test

- `completion_mode`
- `chat_mode`
- `instruction_mode`
- `bitnetcpp_run_inference_mode`
- `transformers_apply_chat_template_mode`

## Required receipt boundary

Tokenizer/prompt receipts must include rendered prompt text, prompt token IDs,
reference runner identity, deterministic decoding config, stop token list, and
whether any special-token garbage appears in decoded output.

## Hard rule

Do not use the official Microsoft 2B `bitnetcpp-answer` prompt template unless a
prompt-authority audit proves it is correct for this Llama3-derived model.
