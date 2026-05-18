# BITNET-SPEC-FALCON3-FAMILY-CUDA: Falcon3 CUDA Contract

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

Define the gated CUDA path for Falcon3. CUDA proof starts only after CPU answer-ready proof for the exact Falcon3 artifact/route.

## First CUDA Ladder

```text
Falcon3 1B I2_S CPU answer-ready
→ Falcon3 I2_S CUDA fixture parity
→ one-token CUDA
→ short-decode CUDA
→ warm-session CUDA
→ exact-profile benchmark review
```

## Required CUDA Fields

```text
all-layer execution plan
Falcon3 tensor role classification
QK256/I2_S invocation count
unsupported op count
CPU fallback count = 0
weights uploaded once
per-token weight upload = false
CPU/CUDA answer parity or first divergence classification
speedup=false until review
```

## Hard Rules

```text
No CUDA proof before CPU answer-ready.
No TL CUDA before TL scalar oracle.
No dense Falcon3 CUDA inheritance.
No Microsoft 2B CUDA inheritance.
No speedup claim before exact-profile benchmark review.
```
