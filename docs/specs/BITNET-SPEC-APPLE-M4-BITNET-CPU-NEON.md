# BITNET-SPEC-APPLE-M4-BITNET-CPU-NEON

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0005 Apple Silicon productization](../proposals/BITNET-PROP-0005-apple-silicon-productization.md)
Linked specs: [Apple Silicon route contract](BITNET-SPEC-APPLE-SILICON-ROUTE-CONTRACT.md)
Linked ADRs: n/a
Linked plan: [Apple Silicon implementation plan](../../plans/apple-silicon/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no support promotion; defines proof ladder for accepted BitNet artifact
Policy impact: no policy exception

## Purpose

Define the accepted BitNet artifact path on Apple M4 CPU/NEON. This is the
first BitNet productization target for M4 and is deliberately separate from
dense SLM, Metal, MPSGraph, Neural Engine, QK256, and broad Apple Silicon
claims.

## Required proof ladder

The `apple_m4_cpu_neon_bitnet` proof family advances only through this ladder:

1. model verify;
2. tokenizer authority;
3. strict one-shot ask;
4. 100-case BitNet corpus;
5. reference-vs-Rust comparison;
6. one-shot benchmark;
7. variable warm-session;
8. 25/50/100 prompt warm soaks;
9. progress/timeout/failure receipts;
10. chat gate;
11. serve gate;
12. task-family taxonomy;
13. 250-case corpus;
14. 500-case decision.

Each rung must preserve model identity, tokenizer identity, prompt/template
identity, requested backend, selected backend, runtime API, fallback status,
proof family, generated token/text receipts where applicable, and timing or
failure taxonomy appropriate to the rung.

## Not-claims

BitNet CPU/NEON receipts are:

- not Metal;
- not QK256 acceleration;
- not Neural Engine;
- not MPSGraph;
- not broad Apple Silicon;
- not dense SLM.

Scalar fallback cannot count as Apple CPU/NEON proof if fallback is selected.
Fallback behavior may be tested, but such tests must be labeled as fallback
behavior, not as product route proof.
