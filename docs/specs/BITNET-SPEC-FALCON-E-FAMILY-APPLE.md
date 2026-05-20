# BITNET-SPEC-FALCON-E-FAMILY-APPLE

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: docs/proposals/BITNET-PROP-0013-falcon-e-family-supported-models.md
Linked specs: docs/specs/BITNET-SPEC-FALCON-E-FAMILY-CPU.md, docs/specs/BITNET-SPEC-FALCON-E-FAMILY-I2S.md
Linked ADRs: n/a
Linked plan: plans/falcon-e-family/implementation-plan.md
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: Apple claims only after machine-specific receipts
Policy impact: no policy exception

## Apple route

```text
MacBook artifact inventory
M4 / MacBook reference runner
ARM I2_S CPU/NEON strict answer proof
ARM TL1 candidate after layout proof
Metal phase candidates only after CPU/NEON proof
```

## Required evidence

Apple receipts must record exact machine identity, artifact identity, tokenizer
and prompt authority, reference-good receipt, selected backend, selected kernel,
NEON/CPU route, fallback status, generated IDs, decoded text, quality corpus
result, and whether the evidence applies to MacBook, M4 Mac mini, or another
Apple machine.

## Hard rules

```text
MacBook proof does not prove M4 Mac Mini.
M4 proof does not prove MacBook.
Apple CPU/NEON proof does not prove Metal.
Metal phase proof does not prove full Metal inference.
```
