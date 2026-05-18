# Apple Silicon docs and rails implementation plan

Status: active
Owner: BitNet-rs maintainers
Created: 2026-05-18
Linked proposal: [BITNET-PROP-0005 Apple Silicon productization](../../docs/proposals/BITNET-PROP-0005-apple-silicon-productization.md)
Linked specs: [Apple Silicon route contract](../../docs/specs/BITNET-SPEC-APPLE-SILICON-ROUTE-CONTRACT.md), [Apple M4 dense SLM appliance](../../docs/specs/BITNET-SPEC-APPLE-M4-DENSE-SLM-APPLIANCE.md), [Apple M4 BitNet CPU/NEON](../../docs/specs/BITNET-SPEC-APPLE-M4-BITNET-CPU-NEON.md), [Apple Metal phased acceleration](../../docs/specs/BITNET-SPEC-APPLE-METAL-PHASED-ACCELERATION.md), [Apple quality corpus](../../docs/specs/BITNET-SPEC-APPLE-QUALITY-CORPUS.md), [Apple benchmark envelope](../../docs/specs/BITNET-SPEC-APPLE-BENCHMARK-ENVELOPE.md), [Apple reproducible run identity](../../docs/specs/BITNET-SPEC-APPLE-REPRODUCIBLE-RUN-IDENTITY.md), [Apple MacBook auxiliary lane](../../docs/specs/BITNET-SPEC-APPLE-MACBOOK-AUXILIARY-LANE.md), [Apple service surface](../../docs/specs/BITNET-SPEC-APPLE-SERVICE-SURFACE.md)
Linked ADRs: n/a
Linked plan: n/a
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: no support promotion; docs/spec contracts only
Policy impact: no policy exception

## Scope

Lay down Apple Silicon proposal/spec/plan rails that route every Apple proof
family to the correct authority. This plan does not change runtime code,
default models, backend selection, server behavior, kernels, or support tiers.

## PR sequence

1. **Source-of-truth map**: add `docs/apple-silicon/README.md`, this plan
   directory, index links, and active-campaign references. Acceptance: proof
   families and historical/current authorities are explicit.
2. **Productization proposal**: add the Apple Silicon proposal. Acceptance:
   dense SLM first, BitNet CPU/NEON second, Metal phase-scoped, MacBook
   auxiliary, no Neural Engine or broad Apple Silicon promotion.
3. **Route contract**: add route IDs, receipt fields, fallback semantics, and
   proof-family separation.
4. **Dense SLM appliance spec**: elevate the supported dense model matrix gates
   into a contractual M4 appliance spec.
5. **BitNet CPU/NEON spec**: define the accepted BitNet artifact proof ladder
   and not-claims.
6. **Metal phased acceleration spec**: define visibility, smoke, parity,
   phase-contribution, real-generation-candidate, and full-route-candidate
   boundaries.
7. **Quality and benchmark specs**: define dense/BitNet corpus separation and
   benchmark-envelope fields/profiles.
8. **Identity and MacBook specs**: define reproducible run identity and the
   MacBook auxiliary lane.
9. **Service surface spec**: define Mac doctor/evidence/ask/chat/serve,
   receipts-check, regression, report-refresh, and benchmark semantics.

A single documentation PR may include multiple steps only when it remains a
rails-only change and does not promote runtime claims.

## Validation

Run these proof commands for docs/rails changes:

```bash
cargo run --locked -p xtask --no-default-features -- campaign check apple-m4-inference-excellence
cargo run --locked -p xtask --no-default-features -- campaign generate --check
git diff --check
```

If a command cannot run because of environment limitations, record the command,
reason, substitute evidence, and whether it blocks merge.

## Claim boundaries

- Do not promote new runtime claims in docs/spec PRs.
- Do not claim full `apple-m4-metal` inference.
- Do not claim Neural Engine execution.
- Do not claim MPSGraph as native Metal.
- Do not claim dense SLM evidence proves BitNet.
- Do not claim BitNet CPU/NEON evidence proves Metal.
- Do not claim MacBook evidence proves M4 Mac Mini behavior.
- Do not touch QK256, Metal kernels, server runtime, or model binaries unless a
  later work item explicitly allows it.
- Keep live hardware/model timing out of ordinary generic CI.
