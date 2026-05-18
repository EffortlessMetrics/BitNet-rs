# Proposals

Proposals explain why a BitNet-rs effort exists, what success means, and which
current repo authorities it will use. They are durable context for humans and
agents before a lane becomes a spec, ADR, plan, or campaign item.

Proposals do not own implementation status, active work, or product claims.
Those belong to campaign manifests, plans, status documents, policy ledgers,
and proof receipts.

## Source-Of-Truth Role

| Layer | Owns |
| --- | --- |
| Proposal | Why the effort exists, user value, constraints, success criteria |
| Spec | What must be true before a claim or behavior is accepted |
| ADR | Durable architecture or proof decision |
| Plan | PR order, proof commands, rollback path |
| Campaign `active.toml` | Current executable work item state |
| Handoff | Operator transfer context, closeout summary, remaining work |
| Status document | User-facing claim tier, proof command, artifact link |
| Policy TOML | Enforceable CI, exception, allowlist, or routing ledger |
| Receipt or artifact | Evidence for what actually happened |

## BitNet Rule

Use proposals to frame a lane, then connect implementation to active manifests
instead of reconstructing state from chat logs or README prose. BitNet-rs uses a
repo-level active-goal entrypoint when present:

```text
.bitnet-rs/goals/active.toml
```

BitNet-rs also keeps campaign-local tracking for campaign execution:

```text
docs/tracking/campaigns/<campaign>/CAMPAIGN.md
docs/tracking/campaigns/<campaign>/active.toml
docs/tracking/campaigns/<campaign>/events/
docs/tracking/campaigns/<campaign>/generated/
```

A proposal may link either authority, but it must not become the live work queue.

## Proposal Shape

New proposals should include:

- `Status`
- `Owner`
- `Problem`
- `Goals`
- `Non-goals`
- `Source-of-truth links`
- `Success criteria`
- `Rollback or exit criteria`

Every proposal that affects user-visible capability claims should link to the
status, model-artifact, hardware, CI, and campaign surfaces that will carry the
actual proof.

## Current proposals

- [BITNET-PROP-0009 bitnet_b1_58-large control model](BITNET-PROP-0009-bitnet-b158-large-control-model.md)
  defines why `1bitLLM/bitnet_b1_58-large` starts as an artifact authority and
  conversion-lane project before any CPU, CUDA, Apple, server, or speed claim.
