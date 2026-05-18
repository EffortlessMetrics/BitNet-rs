# Apple Silicon Docs/Rails Implementation Plan

Status: active
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: planned `docs/proposals/BITNET-PROP-0005-apple-silicon-productization.md`
Linked specs: planned Apple Silicon contract specs under `docs/specs/`
Linked ADRs: n/a
Linked plan: self
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no support-tier promotion; docs/spec sequencing only
Policy impact: none

This plan lays down the PR-sized docs/spec rollout for Apple Silicon proof-family
rails. It does not promote runtime support, benchmark status, service readiness,
Metal inference, MPSGraph execution, Neural Engine execution, MacBook proof, or
broad Apple Silicon claims.

## Sequence

| Order | PR title | Adds | Updates | Purpose |
| --- | --- | --- | --- | --- |
| 0 | `docs(apple): add Apple Silicon source-of-truth map` | `docs/apple-silicon/README.md`, `plans/apple-silicon/README.md`, `plans/apple-silicon/implementation-plan.md` | `docs/specs/INDEX.md`, active/generated M4 campaign status | Define proof families and authority hierarchy without claim promotion. |
| 1 | `docs(proposal): add Apple Silicon productization proposal` | `docs/proposals/BITNET-PROP-0005-apple-silicon-productization.md` | spec index as needed | Explain why Apple Silicon is a product lane. |
| 2 | `docs(spec): add Apple Silicon route contract` | `docs/specs/BITNET-SPEC-APPLE-SILICON-ROUTE-CONTRACT.md` | spec index | Define backend labels, receipt fields, fallback rules, and proof-family separation. |
| 3 | `docs(spec): define Apple M4 dense SLM appliance path` | `docs/specs/BITNET-SPEC-APPLE-M4-DENSE-SLM-APPLIANCE.md` | spec index | Contractualize the dense SLM model gates and supported states. |
| 4 | `docs(spec): define Apple M4 BitNet CPU/NEON path` | `docs/specs/BITNET-SPEC-APPLE-M4-BITNET-CPU-NEON.md` | spec index | Define accepted BitNet CPU/NEON proof ladder and not-claims. |
| 5 | `docs(spec): define phase-scoped Apple Metal acceleration` | `docs/specs/BITNET-SPEC-APPLE-METAL-PHASED-ACCELERATION.md` | spec index | Keep Metal proof phase-scoped with CPU parity and no full-inference claim. |
| 6 | `docs(spec): add Apple quality and benchmark envelope contracts` | `docs/specs/BITNET-SPEC-APPLE-QUALITY-CORPUS.md`, `docs/specs/BITNET-SPEC-APPLE-BENCHMARK-ENVELOPE.md` | spec index | Separate dense and BitNet corpora and define exact-profile benchmark envelopes. |
| 7 | `docs(spec): add Apple reproducibility and MacBook auxiliary lane specs` | `docs/specs/BITNET-SPEC-APPLE-REPRODUCIBLE-RUN-IDENTITY.md`, `docs/specs/BITNET-SPEC-APPLE-MACBOOK-AUXILIARY-LANE.md` | spec index | Make run identity contractual and keep MacBook proof distinct from M4 proof. |
| 8 | `docs(spec): define Apple ask chat serve readiness` | `docs/specs/BITNET-SPEC-APPLE-SERVICE-SURFACE.md` | spec index | Define `bitnet mac` operator and service-surface readiness contracts. |

## Work Item: APPLE-SILICON-DOCS-000

Status: ready
Linked proposal: planned `docs/proposals/BITNET-PROP-0005-apple-silicon-productization.md`
Linked specs: `docs/specs/apple-m4-mac-mini-roadmap.md`, planned Apple Silicon specs
Linked ADRs: n/a
Campaign item: `APPLE-SILICON-DOCS-000`
Blocked by: none
Blocks: Apple Silicon proposal and contract spec PRs

### Goal

Add the Apple Silicon source-of-truth map and implementation plan so future
Codex work can find the correct authority for M4 dense SLM, M4 BitNet CPU/NEON,
Metal phase, MPSGraph reference, Neural Engine research, MacBook auxiliary,
quality, benchmark, reproducibility, and service-surface evidence.

### Production Delta

Docs only. No runtime behavior, model support, service readiness, benchmark,
receipt, or support-tier state changes.

### Non-Goals

- Do not add model binaries.
- Do not edit QK256, Metal kernels, server runtime, or runtime routes.
- Do not claim full `apple-m4-metal` inference.
- Do not claim Neural Engine execution.
- Do not claim MPSGraph as native Metal.
- Do not claim dense SLM evidence proves BitNet.
- Do not claim BitNet CPU/NEON evidence proves Metal.
- Do not claim MacBook evidence proves M4 Mac Mini behavior.
- Do not require live Apple hardware/model timing in generic CI.

### Acceptance

- `docs/apple-silicon/README.md` lists Apple proof families and source-of-truth
  authorities.
- `plans/apple-silicon/README.md` and this plan define the docs/spec rollout.
- `docs/specs/INDEX.md` points readers to the Apple Silicon map and planned
  contract specs.
- The active M4 campaign records this docs/rails work item.
- Generated campaign status is refreshed by the campaign generator.

### Proof Commands

```bash
cargo run --locked -p xtask --no-default-features -- campaign check apple-m4-inference-excellence
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

### Rollback

Revert `docs/apple-silicon/`, `plans/apple-silicon/`, the `docs/specs/INDEX.md`
Apple Silicon entry, and the campaign work-item/status updates. No runtime or
model artifacts are involved.
