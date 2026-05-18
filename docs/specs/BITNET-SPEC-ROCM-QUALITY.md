# BITNET-SPEC-ROCM-QUALITY: ROCm Answer Quality

Status: proposed
Owner: rocm/productization
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0008](../proposals/BITNET-PROP-0008-amd-rocm-productization.md)
Linked specs: [ROCm BitNet QK256](BITNET-SPEC-ROCM-BITNET-QK256.md), [ROCm dense SLM](BITNET-SPEC-ROCM-DENSE-SLM.md)
Linked ADRs: [BITNET-ADR-0005](../adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md)
Linked plan: [AMD ROCm implementation plan](../../plans/rocm/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: defines quality gates; no answer-ready promotion
Policy impact: no CI policy exception

## Purpose

Ensure "working on ROCm" means intelligible, bounded outputs with fallback
false, route identity, tokenizer/template authority, and CPU/ROCm parity or
classified divergence.

## BitNet ROCm Quality Set

```text
tiny deterministic answer corpus
expanded answer corpus
prompt-conditioning pairs
copy/repeat
yes/no
format following
stop-token behavior
long decode
CPU/ROCm generated-token parity or first divergence classification
```

## Dense SLM ROCm Quality Set

```text
same dense SLM corpus used by CUDA/CPU lanes
direct prompt IDs when available
generated IDs
decoded text
failure taxonomy
warm-session behavior
```

## Failure Taxonomy

```text
exact_answer_overgenerated
exact_answer_instruction_not_followed
missing_required_keyword
forbidden_token_observed
raw_special_token_seen
empty_answer
repetition
stop_policy_failed
context_sensitivity_failed
structured_output_failed
tokenizer_mismatch
runtime_error
timeout
first_divergence_logits_topk
```

## Promotion Rule

Answer-ready ROCm promotion requires all applicable quality cases to pass or an
exact blocker to be recorded. Benchmark, speed, residency, and server readiness
remain false unless their separate specs are satisfied.
