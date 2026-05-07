# Model Artifacts Campaign

Campaign ID: `model-artifacts`

Status: active

## Objective

Define and maintain shared answer-artifact authority so CPU, CUDA, Apple, NPU,
SLM, server, and other hardware lanes do not claim coherent local answers from a
GGUF that only proves structural validity or backend execution.

## End State

- Answer-capable model artifacts are distinguished from structurally valid GGUFs.
- Rejected artifacts record exact identity, hash, tokenizer authority, reference
  runner result, prompt-suite result, and claim boundary.
- Hardware answer-readiness lanes depend on an `answer_ready` artifact before
  coherent local-answer claims.
- Diagnostic-only receipts remain allowed against rejected artifacts when they
  do not claim answer readiness or speedup.

## Hard Constraints

- Do not weaken CPU, CUDA, Apple, NPU, SLM, server, or quality gates.
- Do not touch kernels, transformer routing, tokenizer behavior, model loader
  behavior, server inference, or accelerator runtime code in tracker-only PRs.
- Do not claim a structurally valid GGUF is answer-ready without reference
  runner prompt-suite evidence.
- Do not turn backend execution proof into answer-readiness proof.
- Do not make speed claims from answer artifacts.

## Work Items

| Work item | Status | Notes |
|---|---|---|
| MODEL-ARTIFACT-001 | merged | Shared answer-artifact gate merged in #3922; the current Microsoft BitNet I2_S GGUF is rejected for coherent local-answer claims. |
| MODEL-ARTIFACT-002 | blocked | Shared search merged in #3928 and records no `answer_ready` BitNet artifact. Backend local-answer lanes remain blocked or diagnostic-only until a future artifact passes the reference prompt suite. |
| MODEL-ARTIFACT-003 | merged | PR #3932 records latest stock llama.cpp reference-runner compatibility for official-derived candidates. This is diagnostic-only and does not unblock backend answer claims. |
| MODEL-ARTIFACT-004 | merged | PR #3939 records intended `ik_llama.cpp` runner evidence for `tdh111` IQ2_BN_R4 and the official Microsoft I2_S artifact. This is diagnostic-only and does not promote an `answer_ready` artifact. |

## Review Policy

Model-artifact PRs are non-stackable when they change answer-claim boundaries or
accepted/rejected artifact state. Runtime fixes belong in their owning lane only
after the artifact state makes that lane's claim meaningful.
