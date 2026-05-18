# BITNET-SPEC-FALCON-E-FAMILY-CUDA

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: docs/proposals/BITNET-PROP-0013-falcon-e-family-supported-models.md
Linked specs: docs/specs/BITNET-SPEC-FALCON-E-FAMILY-CPU.md, docs/specs/BITNET-SPEC-FALCON-E-FAMILY-I2S.md
Linked ADRs: n/a
Linked plan: plans/falcon-e-family/implementation-plan.md
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: CUDA support only after CPU and CUDA receipts
Policy impact: no policy exception

## CUDA ladder

```text
Falcon-E 1B I2_S CPU answer-ready
→ Falcon-E I2_S CUDA fixture parity
→ one-token CUDA
→ short-decode CUDA
→ warm-session CUDA
→ exact-profile benchmark review
```

## Required evidence

```text
all-layer execution plan
Falcon-E tensor role classification
QK256/I2_S invocation count
unsupported op count
CPU fallback count = 0
weights uploaded once
per-token weight upload = false
CPU/CUDA answer parity or first divergence classification
speedup=false until review
```

## Receipt requirements

CUDA receipts must record device identity, driver/runtime, artifact SHA,
tokenizer/prompt receipt, CPU comparator receipt, selected backend, selected
kernel, kernel invocation counts, fallback counts, H2D/D2H bytes, upload policy,
generated token IDs, decoded text, divergence class, and speed claim boundary.

## Hard rules

```text
No CUDA proof before CPU answer-ready.
No TL CUDA before TL scalar oracle.
No Microsoft 2B CUDA inheritance.
No Falcon3 CUDA inheritance.
```
