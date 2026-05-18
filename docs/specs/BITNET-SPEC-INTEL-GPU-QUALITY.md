# BITNET-SPEC-INTEL-GPU-QUALITY

Status: proposed
Owner: intel-gpu/product
Created: 2026-05-18
Linked proposal: `docs/proposals/BITNET-PROP-0006-intel-gpu-productization.md`
Linked specs: `docs/specs/BITNET-SPEC-INTEL-GPU-ROUTE-CONTRACT.md`, `docs/specs/BITNET-SPEC-INTEL-GPU-BITNET-QK256.md`, `docs/specs/BITNET-SPEC-INTEL-GPU-DENSE-SLM.md`
Linked ADRs: n/a
Linked plan: `plans/intel-gpu/implementation-plan.md`
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: Defines quality gates; no promotion.
Policy impact: No exception.

## Purpose

Intel GPU "up and running" means the route produces intelligible, useful,
route-scoped answers. Kernel execution, graph execution, or fast timing alone
is not answer readiness.

## BitNet A770/OpenCL gates

BitNet native OpenCL answer proof requires:

```text
official answer corpus
prompt conditioning
paired context changes answer
copy/repeat
yes/no
format following
stop-token behavior
long decode
CPU/A770 generated-token parity or first divergence classification
```

## Dense SLM OpenVINO GPU gates

Dense SLM OpenVINO GPU proof requires:

```text
lunar-lake answer corpus v2
profile summaries
category summaries
failure taxonomy
generation-budget sensitivity
stop/EOS diagnosis
retokenized-vs-direct-token-ID boundary
```

## Failure taxonomy

Failures must use one or more of:

```text
exact_answer_instruction_not_followed
exact_answer_overgenerated
missing_required_keyword
forbidden_token_observed
raw_special_token_seen
empty_answer
repetition
stop_policy_failed
context_sensitivity_failed
structured_output_failed
timeout
runtime_error
```

## Promotion rule

A route with unclassified quality failures cannot be promoted to
`answer_ready`, `behavior_proven`, `performance_proven`, or `complete`.
