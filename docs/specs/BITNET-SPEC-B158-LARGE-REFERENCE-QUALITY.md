# BITNET-SPEC-B158-LARGE-REFERENCE-QUALITY

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0009 bitnet_b1_58-large control model](../proposals/BITNET-PROP-0009-bitnet-b158-large-control-model.md)
Linked specs: [artifact contract](BITNET-SPEC-B158-LARGE-ARTIFACT-CONTRACT.md), [conversion](BITNET-SPEC-B158-LARGE-CONVERSION.md), [tokenizer/prompt](BITNET-SPEC-B158-LARGE-TOKENIZER-PROMPT.md), [answer artifact gate](../model-artifacts/ANSWER_ARTIFACT_GATE.md)
Linked ADRs: n/a
Linked plan: [bitnet_b1_58-large implementation plan](../../plans/bitnet-b158-large/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no support promotion until reference receipts pass
Policy impact: no policy exception

## Purpose

Define reference-runner success for `1bitLLM/bitnet_b1_58-large`. Backend work
must not blame CPU, CUDA, Metal, or Apple kernels until an authoritative
reference runner can produce coherent bounded output for the exact artifact,
tokenizer, and prompt template.

## Corpus levels

| Corpus | Required when |
| --- | --- |
| `tiny_smoke` | Initial artifact probe. |
| `answer_corpus_v1` | Reference-good promotion. |
| `behavior_suite_v1` | Product CLI promotion. |
| `long_decode_v1` | Warm/chat promotion. |

The shared answer corpus should be used where possible. If the model is
completion-only, the receipt must adapt prompts explicitly and record why chat
or instruction modes are blocked.

## Minimum cases

```text
math_2_plus_2
capital_france
copy_color_sequence
yes_no_clear_sky
short_continuation
prompt_conditioning_pair
stop_token_behavior
special_token_garbage_check
```

## Reference receipt shape

```json
{
  "reference_runner": "microsoft_bitnet_cpp|transformers|vllm|sglang|other",
  "runner_version": "...",
  "model_artifact_sha256": "...",
  "tokenizer_authority": "...",
  "prompt_template": "...",
  "cases_total": 8,
  "passed": 8,
  "failed": 0,
  "claim": "reference_good",
  "backend_claim": false
}
```

Receipts must include generated text summaries and per-case pass/fail outcomes.
Store enough output to review special-token garbage, repetition failures, empty
responses, and prompt-conditioning failures without committing model binaries.

## Hard rule

If the reference runner cannot produce coherent bounded output, do not blame
CPU, CUDA, Metal, Apple CPU/NEON, OpenCL, NPU, server, or CLI code yet. The
artifact remains blocked or rejected at the reference-quality gate until the
exact failure is understood.
