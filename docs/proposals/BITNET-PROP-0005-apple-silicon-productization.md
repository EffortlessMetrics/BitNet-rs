# BITNET-PROP-0005: Apple Silicon productization

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: n/a
Linked specs: [Apple Silicon route contract](../specs/BITNET-SPEC-APPLE-SILICON-ROUTE-CONTRACT.md), [Apple M4 dense SLM appliance](../specs/BITNET-SPEC-APPLE-M4-DENSE-SLM-APPLIANCE.md), [Apple M4 BitNet CPU/NEON](../specs/BITNET-SPEC-APPLE-M4-BITNET-CPU-NEON.md), [Apple Metal phased acceleration](../specs/BITNET-SPEC-APPLE-METAL-PHASED-ACCELERATION.md)
Linked ADRs: n/a
Linked plan: [Apple Silicon implementation plan](../../plans/apple-silicon/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no support promotion by this proposal
Policy impact: no policy exception

## Thesis

Apple Silicon gives BitNet-rs a Mac-native local inference appliance path:
supported dense SLMs and accepted BitNet artifacts can run on widely available
unified-memory Macs with strict receipts, useful local-answer behavior,
operator health checks, and phase-scoped Metal acceleration work.

## Motivation

The M4 Mac Mini lane already has strong local inference foundations, but Apple
Silicon claims are easy to over-broaden. A dense Qwen-class CPU/NEON result does
not prove BitNet; a BitNet CPU/NEON result does not prove Metal; a MacBook run
does not prove M4 Mac Mini behavior; and a Metal smoke test does not prove full
autoregressive Metal inference. This proposal creates a product lane that is
useful now while preserving those boundaries.

## Product order

1. **Dense SLM appliance path first.** Supported, artifact-pinned Qwen-class
   dense SLMs are the first Mac-native user path. They must keep tokenizer,
   prompt-template, backend, fallback, quality, and timing receipts.
2. **BitNet CPU/NEON path second.** The accepted BitNet artifact should run on
   Apple CPU/NEON before any acceleration claim expands.
3. **Metal acceleration is phase-scoped.** Metal work remains explicit kernels
   or subgraphs with CPU parity and receipt boundaries until a full route proof
   exists.
4. **MPSGraph is a reference/graph lane.** MPSGraph can inform graph experiments
   but does not count as native Metal proof.
5. **Neural Engine is not claimed.** No ANE or Neural Engine claim exists until
   a future route records the selected target and proves it with receipts.
6. **MacBook is auxiliary.** A MacBook can help with larger artifacts, external
   reference comparisons, and longer soaks, but MacBook proof never counts as
   M4 Mac Mini proof.

## Non-goals

- Full Metal BitNet inference.
- QK256-on-Metal proof.
- Neural Engine execution.
- Broad Apple Silicon support or performance from one M4 Mac Mini profile.
- Runtime, kernel, server, or model-binary changes in proposal/spec PRs.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| A successful dense SLM run is treated as BitNet evidence. | Route contract requires `model_family` and `proof_family` receipts. |
| CPU fallback is counted as Metal execution. | Metal specs require `runtime_api = "metal"` and `fallback_used = false`. |
| MacBook storage solves are mistaken for M4 runtime proof. | MacBook receipts must record `counts_as_m4_mac_mini_proof = false`. |
| Hardware-only timing becomes generic CI. | Plans keep live hardware/model timing out of ordinary PR CI. |

## Acceptance

This proposal is accepted when linked specs define backend labels, proof
families, dense SLM gates, BitNet CPU/NEON gates, phase-scoped Metal promotion,
quality corpus separation, benchmark envelopes, reproducible identity, MacBook
auxiliary boundaries, and Mac service-surface readiness without promoting new
runtime claims.
