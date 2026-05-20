# BITNET-SPEC-APPLE-REPRODUCIBLE-RUN-IDENTITY

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0005 Apple Silicon productization](../proposals/BITNET-PROP-0005-apple-silicon-productization.md)
Linked specs: [Apple Silicon route contract](BITNET-SPEC-APPLE-SILICON-ROUTE-CONTRACT.md)
Linked ADRs: n/a
Linked plan: [Apple Silicon implementation plan](../../plans/apple-silicon/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no support promotion; run identity contract only
Policy impact: no policy exception

## Purpose

Make Apple Silicon comparisons reproducible by requiring every receipt family to
record machine, software, model, tokenizer, prompt, backend, fallback, corpus,
profile, seed, and timing identity.

## Required fields

Apple receipts must record:

- `machine_id`;
- SoC;
- OS name/version;
- git commit;
- binary hash or build profile;
- command class;
- model ID and SHA;
- tokenizer authority and SHA;
- prompt template ID and hash;
- requested backend;
- selected backend;
- runtime API;
- fallback used;
- corpus/profile/seed identity;
- timing source.

## Comparison rules

A matching-history comparison is valid only when identities match for the
fields relevant to the receipt family. If any identity changes, the result may
be a new baseline, compatibility refresh, or diagnostic run, but it is not a
same-identity trend point.

Receipts must expose enough identity to distinguish M4 Mac Mini from MacBook,
dense SLM from BitNet, CPU/NEON from Metal, Metal from MPSGraph, and fallback
behavior from requested-backend execution.
