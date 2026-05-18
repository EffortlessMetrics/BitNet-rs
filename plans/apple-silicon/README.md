# Apple Silicon Plan

Status: active
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: planned `docs/proposals/BITNET-PROP-0005-apple-silicon-productization.md`
Linked specs: planned Apple Silicon contract specs under `docs/specs/`
Linked ADRs: n/a
Linked plan: `plans/apple-silicon/implementation-plan.md`
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no support-tier promotion; docs-only sequencing
Policy impact: none

This plan sequences the Apple Silicon docs/spec rails that keep M4 Mac Mini,
MacBook, dense SLM, BitNet CPU/NEON, Metal phase, MPSGraph reference, and Neural
Engine research evidence from being conflated.

## Source-Of-Truth Links

| Surface | Path |
| --- | --- |
| Source-of-truth map | `docs/apple-silicon/README.md` |
| M4 roadmap | `docs/specs/apple-m4-mac-mini-roadmap.md` |
| Dense SLM support matrix | `docs/slm/apple-m4-dense-slm-model-support-matrix.md` |
| M4 inference narrative | `docs/slm/apple-m4-inference-excellence.md` |
| Active M4 campaign | `docs/tracking/campaigns/apple-m4-inference-excellence/active.toml` |
| Machine receipts | `ci/hardware/apple-m4-mac-mini/**` |

## Files

| File | Owns |
| --- | --- |
| `implementation-plan.md` | PR order, dependencies, proof commands, rollback |
| `docs/apple-silicon/README.md` | Apple proof-family map and authority hierarchy |

## Operating Rules

- Do not create a new competing current-truth page for M4 runtime status.
- Do not delete historical Apple campaign docs.
- Keep dense SLM, BitNet, Metal, MPSGraph, Neural Engine, M4 Mac Mini, and
  MacBook receipts in distinct proof families.
- Keep live hardware/model timing out of ordinary generic PR CI.
- Do not touch model binaries, QK256, Metal kernels, server runtime, or runtime
  routes from docs/spec PRs unless a later work item explicitly allows it.
