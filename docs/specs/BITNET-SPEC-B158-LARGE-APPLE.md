# BITNET-SPEC-B158-LARGE-APPLE

Status: proposed
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0009 bitnet_b1_58-large control model](../proposals/BITNET-PROP-0009-bitnet-b158-large-control-model.md)
Linked specs: [artifact contract](BITNET-SPEC-B158-LARGE-ARTIFACT-CONTRACT.md), [reference quality](BITNET-SPEC-B158-LARGE-REFERENCE-QUALITY.md), [Apple MacBook auxiliary lane](BITNET-SPEC-APPLE-MACBOOK-AUXILIARY-LANE.md), [Apple M4 BitNet CPU/NEON](BITNET-SPEC-APPLE-M4-BITNET-CPU-NEON.md), [Apple Metal phased acceleration](BITNET-SPEC-APPLE-METAL-PHASED-ACCELERATION.md)
Linked ADRs: n/a
Linked plan: [bitnet_b1_58-large implementation plan](../../plans/bitnet-b158-large/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no Apple support promotion until receipts pass
Policy impact: no policy exception

## Purpose

Define Apple MacBook and M4 support boundaries for `1bitLLM/bitnet_b1_58-large`.
The model is a storage-friendly smaller control candidate, but Apple claims must
still pass the shared artifact and answer gates.

## Apple routes

```text
macbook_artifact_probe
m4_cpu_neon_answer
m4_metal_phase_candidate
m4_mpsgraph_reference_candidate
```

## Route order

The MacBook lane is used first for larger artifact exploration and reference
runs. Accepted candidates go back to M4 for strict Apple CPU/NEON proof. MacBook
evidence never counts as M4 Mac mini proof.

## Acceptance

Apple receipts must record:

- MacBook artifact inventory;
- storage/free-space context;
- exact artifact SHA256;
- tokenizer and prompt authority;
- reference output;
- cleanup status;
- M4 strict CPU/NEON answer receipt before M4 answer claims;
- selected backend;
- generated IDs and decoded text;
- timing;
- `fallback_used = false`;
- no Metal claim from CPU/NEON proof;
- Metal phase only after CPU proof.

## Hard rules

- No M4 local-answer claim from MacBook receipts.
- No full Metal claim from artifact inventory, CPU/NEON proof, or phase-level
  kernel work.
- No dense Qwen or SLM proof inheritance.
- No Neural Engine, MPSGraph model-inference, or broad Apple Silicon performance
  claim without separate specs and receipts.
