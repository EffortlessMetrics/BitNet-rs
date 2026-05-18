# BITNET-SPEC-B158-3B-TOKENIZER-PROMPT

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0010](../proposals/BITNET-PROP-0010-bitnet-b158-3b-tl-model.md)
Linked specs: [3B artifact contract](BITNET-SPEC-B158-3B-ARTIFACT-CONTRACT.md), [3B reference quality](BITNET-SPEC-B158-3B-REFERENCE-QUALITY.md)
Linked ADRs: n/a
Linked plan: [3B TL implementation plan](../../plans/bitnet-b158-3b/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no support promotion; tokenizer/prompt contract only
Policy impact: no policy exception

## Purpose

Define tokenizer and prompt authority for deterministic 3B reference and Rust
runs. The 3B lane must not inherit the official Microsoft 2B
`bitnetcpp-answer` prompt template or tokenizer compatibility decision.

## Required audit fields

A tokenizer/prompt receipt must record:

- `tokenizer.json` SHA256;
- `tokenizer.model` SHA256;
- `tokenizer_config.json` SHA256;
- `added_tokens.json` SHA256 when present;
- `special_tokens_map.json` SHA256;
- BOS, EOS, PAD, and UNK token IDs;
- chat template if present;
- completion template when no chat template is used;
- stop-token policy;
- rendered prompt text;
- prompt token IDs;
- deterministic decoding settings;
- reference runner command;
- runner conversation mode, yes or no.

## Prompt authority levels

| Level | Meaning | Claim |
| --- | --- | --- |
| `files_hashed` | Tokenizer files exist and hashes are recorded. | Tokenizer inventory only. |
| `template_identified` | Chat or completion template is selected and rendered. | Prompt candidate only. |
| `runner_accepted` | Reference runner accepts tokenizer and prompt settings. | Reference-run candidate. |
| `reference_good` | Reference corpus passes with the selected tokenizer/prompt. | Reference quality candidate. |
| `rust_matched` | Rust tokenization and prompt IDs match bounded reference evidence. | Rust route candidate. |

## Hard rules

- Do not assume the official Microsoft 2B `bitnetcpp-answer` prompt template
  applies to `1bitLLM/bitnet_b1_58-3B`.
- No tokenizer hashes, no answer claim.
- No prompt rendering and prompt token IDs, no deterministic answer receipt.
- A different tokenizer revision invalidates previous prompt-token receipts
  until rerun.
