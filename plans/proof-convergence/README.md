# Proof Convergence Plan

The proof-convergence lane makes BitNet-rs claim boundaries explicit across
model artifacts, hardware lanes, CI economics, campaign execution, and user
status docs.

The goal is a proof-first repo operating model: a user, maintainer, or agent can
find which model families are usable, which are diagnostic only, which hardware
lanes have proof, what CI may claim, what PR should happen next, and which
commands prove or disprove a claim.

## Source-Of-Truth Stack

| Layer | BitNet source |
| --- | --- |
| Direction | `ROADMAP.md` |
| Why | `docs/proposals/` |
| What must be true | `docs/specs/` |
| Durable decisions | `docs/adr/` |
| User-facing status | `docs/status/` |
| Model answer-readiness | `docs/model-artifacts/ANSWER_ARTIFACT_GATE.md` and `ci/model-artifacts/*.toml` |
| Hardware proof | `docs/hardware/HARDWARE_MATRIX.md`, proof-stage docs, and `ci/hardware/**` |
| CI economics | `docs/ci/cost-and-verification-policy.md` and `policy/ci-*.toml` |
| Active work | `docs/tracking/campaigns/<campaign>/active.toml` |
| Lifecycle history | `docs/tracking/campaigns/<campaign>/events/` |
| Evidence | Receipts, reports, artifacts, and closeouts |

## BitNet-Specific Rule

Do not import an Adze-style goal store. BitNet-rs already has a campaign tracker
with campaign-local manifests. Agents should use:

```text
docs/tracking/campaigns/<campaign>/CAMPAIGN.md
docs/tracking/campaigns/<campaign>/active.toml
docs/tracking/campaigns/<campaign>/events/
docs/tracking/campaigns/<campaign>/generated/
```

## First PRs

1. Define the source-of-truth documentation model.
2. Add the proof convergence proposal.
3. Define source-of-truth and claim boundary specs.
4. Add a capability matrix source of truth.
5. Specify default PR CI economics.

Later PRs should add model/hardware specs, ADRs, plan files, campaign state, a
policy ledger, and optional `xtask` enforcement after the docs model is stable.

## Claim Boundary

This plan does not claim:

- model answer readiness,
- BitNet coherent local answers,
- dense SLM proof as BitNet proof,
- CPU, CUDA, Metal, MPSGraph, OpenCL, WGPU, or NPU speed,
- hardware validation,
- CI budget enforcement.

Those claims require the relevant artifact gate, hardware receipt, policy TOML,
campaign item, and proof command.
