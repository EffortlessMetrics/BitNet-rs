# BITNET-SPEC-CLAIM-BOUNDARY-REVIEW: PR Claim-Boundary Review

Status: proposed
Owner: release/model-support
Created: 2026-05-20
Linked proposal:
[BITNET-PROP-0003](../proposals/BITNET-PROP-0003-native-rust-inference-product.md)
Linked specs:
[BITNET-SPEC-MODEL-READINESS-STATUS-SURFACE](BITNET-SPEC-MODEL-READINESS-STATUS-SURFACE.md),
[BITNET-SPEC-RECEIPT-EXPLAIN-SCHEMA](BITNET-SPEC-RECEIPT-EXPLAIN-SCHEMA.md),
[BITNET-SPEC-SUPPORT-BUNDLE](BITNET-SPEC-SUPPORT-BUNDLE.md),
[BITNET-SPEC-PROOF-FAMILY-NON-INHERITANCE](BITNET-SPEC-PROOF-FAMILY-NON-INHERITANCE.md),
[BITNET-SPEC-0013](BITNET-SPEC-0013-model-onboarding-proof-ladder.md),
[BITNET-SPEC-0014](BITNET-SPEC-0014-runtime-performance-contract.md)
Linked ADRs:
[BITNET-ADR-0005](../adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md)
Linked plan: n/a
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: review gate only; no tier promotion
Policy impact: none

## Purpose

BitNet-rs PR review must preserve the difference between evidence, diagnostics,
route visibility, and user-facing support claims. A PR may make the repo more
observable or easier to support without proving new execution, quality, speed,
residency, server readiness, or model-family coverage.

This spec defines the cross-cutting claim-boundary review rules for PRs that
touch model status, receipt explanation, support bundles, diagnostics,
hardware/backend rows, performance evidence, or support docs.

## Source-Of-Truth Authorities

Claim-boundary truth is owned by:

- model coverage rows in `ci/model-artifacts/model-coverage-matrix.toml`;
- status surfaces governed by
  [BITNET-SPEC-MODEL-READINESS-STATUS-SURFACE](BITNET-SPEC-MODEL-READINESS-STATUS-SURFACE.md);
- receipt summaries governed by
  [BITNET-SPEC-RECEIPT-EXPLAIN-SCHEMA](BITNET-SPEC-RECEIPT-EXPLAIN-SCHEMA.md);
- support bundles governed by
  [BITNET-SPEC-SUPPORT-BUNDLE](BITNET-SPEC-SUPPORT-BUNDLE.md);
- proof-family non-inheritance governed by
  [BITNET-SPEC-PROOF-FAMILY-NON-INHERITANCE](BITNET-SPEC-PROOF-FAMILY-NON-INHERITANCE.md);
- proof-family rules in
  [BITNET-ADR-0005](../adr/BITNET-ADR-0005-proof-families-are-not-interchangeable.md);
- backend, model-family, and performance specs for the exact lane being
  changed.

Docs prose, issue text, PR titles, branch names, and generated dashboards may
summarize claims. They do not promote claims by themselves.

## Review Rule

Every PR that touches a support-facing surface must name the allowed claim and
the forbidden promotion. If the PR cannot name a proof source for a promotion,
the claim remains false or unknown.

Required review questions:

```text
What exact model/artifact/profile is changed?
What exact backend and route are changed?
What receipt or test proves the changed claim?
What support surface displays the claim?
What adjacent claims must remain false or unknown?
Does fallback status prove strict execution, disprove strict execution, or stay unknown?
```

The answer may be "no claim changes." In that case the PR must preserve
existing status rows, receipt fields, support-bundle fields, and support-tier
boundaries unless it explicitly narrows them.

## Allowed Claims And Forbidden Promotions

| Change type | Allowed claim | Forbidden promotion |
| --- | --- | --- |
| diagnostic trace | diagnostic evidence or observability | semantic quality, answer readiness, speed, or residency |
| route label or route visibility | route/status visibility | selected execution or fallback-free execution |
| one-token receipt | bounded one-token evidence | broad product readiness, chat readiness, server readiness, or speedup |
| short-decode or warm-session receipt | exact-profile decode evidence | unrelated model family, endpoint readiness, or benchmark speedup |
| exact-profile server smoke | exact endpoint/profile server evidence | broad server readiness, streaming, concurrency, or deployment readiness |
| CUDA dense Qwen2.5 proof | exact Qwen2.5 dense CUDA row | Qwen3, BitNet packed I2_S/QK256, TL1/TL2, or generic CUDA support |
| Qwen3 dense proof | exact Qwen3 row and route | Qwen2.5, BitNet, or broad dense small-LLM support |
| BitNet packed I2_S/QK256 proof | exact packed BitNet route | dense SLM, TL1/TL2, GPU-int2, or broad BitNet-family support |
| A770 OpenCL diagnostic status | non-claiming A770 diagnostic evidence | quality, selected execution, speed, residency, or support readiness |
| NPU static-subgraph proof | static subgraph parity evidence | full inference, QK256 decode, speed, residency, or generic NPU support |
| Apple CPU/NEON receipt | exact Apple CPU/NEON evidence | Metal, ANE/NPU, speedup, or non-Apple proof |
| CUDA support bundle | issue triage context | new inference proof, hardware probe proof, speed, residency, or server readiness |
| performance microbench | microbench result for named kernel/profile | product speedup or end-to-end decode speedup |

## Fallback Semantics

Fallback status must be explicit:

- `fallback_used=false` may support strict execution only when the receipt also
  identifies the selected backend, selected route, and claimed operation.
- `fallback_used=true` disproves strict selected-route execution for that run.
- missing fallback data is unknown. It is not proof of fallback-free execution.

A PR must not replace unknown fallback data with `false` to unlock a claim.

## Proof-Family Non-Inheritance

Proof never inherits across model family, route, backend, or profile unless the
target spec explicitly says so and a promotion PR records that decision.

Hard boundaries:

- Qwen2.5 proof is not Qwen3 proof.
- Dense regular-LLM CUDA proof is not BitNet packed I2_S/QK256 proof.
- BitNet QK256 proof is not TL1, TL2, GPU-int2, or dense SLM proof.
- A CUDA receipt is not OpenCL, ROCm, Metal, OpenVINO, NPU, or CPU proof.
- One device profile does not prove another device profile.
- One exact server profile does not prove broad server readiness.
- A microbench does not prove product speedup.

## Support Surface Requirements

When a PR changes a support-facing claim, these surfaces must remain aligned:

```text
ci/model-artifacts/model-coverage-matrix.toml
bitnet model status --device <device> --format json
bitnet receipts explain <receipt> --format json
bitnet support bundle --latest --device <device> --format json
docs/status/* when the public matrix describes the claim
```

If a PR only improves diagnostics, docs, schemas, issue templates, or support
bundle context, it must not update model coverage rows or public status
matrices to imply a stronger support tier.

## Acceptance Examples

| Case | Required handling |
| --- | --- |
| PR adds a diagnostic trace for A770 | Claim diagnostic evidence only; keep A770 support booleans false |
| PR adds a CUDA support-bundle field | Preserve existing claim booleans; do not imply a new proof command ran |
| PR proves Qwen2.5 dense CUDA one-token output | Update only the exact row/tier allowed by the lane spec |
| PR proves Qwen3 ask path | Keep Qwen2.5 and BitNet proof-family fields independent |
| PR adds an exact-profile server smoke | Keep broad `server_ready` false unless server promotion gates pass |
| PR reports an AVX2 microbench improvement | Record microbench evidence; keep `speedup_claim=false` until product benchmark review |
| PR exposes route labels | Say route visibility improved; do not claim selected execution |

## Proof Commands

Docs-only changes to this spec should run:

```bash
cargo run --locked -p xtask --no-default-features -- check-file-policy --report-dir target/bitnet/reports
cargo run --locked -p xtask --no-default-features -- policy-report --report-dir target/bitnet/reports
git diff --check
```

PRs that change claim-bearing code or data must also run the proof commands in
the exact backend/model/performance spec they touch.

## Non-Goals

- Do not promote any current model, backend, speed, residency, or server claim.
- Do not encode today's open PR queue or current PR order.
- Do not replace backend-specific, model-specific, or performance-specific specs.
- Do not make support bundles or generated dashboards a proof source.
- Do not require hardware CI for docs-only claim-boundary changes.
