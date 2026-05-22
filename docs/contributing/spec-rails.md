# Contributing to source-of-truth rails

Status: active
Owner: repo-architecture
Created: 2026-05-21
Linked proposal: n/a
Linked specs: n/a
Linked ADRs: n/a
Linked plan: n/a
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: none
Policy impact: references-only

## Scope

Use the existing source-of-truth stack for durable repo-owned artifacts:

- `docs/proposals/` explains why a lane exists.
- `docs/specs/` defines what must be true.
- `docs/adr/` records durable decisions.
- `plans/<lane>/` sequences PR-sized work.
- `docs/tracking/campaigns/<campaign>/active.toml` records campaign-local
  execution state.
- `policy/*.toml` records live policy and enforcement ledgers.
- Receipts and reports prove claims.

Do not create a parallel namespace for durable proposals, specs, ADRs, plans,
policy, or closeouts unless a future ADR explicitly changes the source-of-truth
model.

## Do not edit as durable rails

The following paths are external or tool/session-specific state:

- `.codex/`
- `.spec/`
- `.claude/`
- `.jules/`

These may coexist conceptually, but this system does not migrate, rewrite,
validate, or depend on them as source-of-truth artifacts.

## Practical rules

- Keep the full source-of-truth stack (proposal -> spec -> ADR -> plan -> proof
  -> support/policy -> closeout).
- Link durable artifacts through `docs/reference/SPEC_SYSTEM.md`, the relevant
  proposal/spec/ADR/plan headers, campaign manifests, and policy ledgers.
- Keep lane trackers focused and PR-sized; avoid giant shared active queues.
- Keep support-tier and policy claims referenced to their authoritative surfaces.
