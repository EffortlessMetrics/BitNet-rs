# BitNet-rs spec style and source-of-truth doctrine

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

## Durable home

BitNet-rs stores durable source-of-truth rails in the existing repository
surfaces named by `docs/reference/SPEC_SYSTEM.md`:

- `docs/proposals/`
- `docs/specs/`
- `docs/adr/`
- `plans/<lane>/`
- `docs/tracking/campaigns/<campaign>/active.toml`
- `policy/*.toml`
- receipt and report locations named by the relevant plan or campaign

This includes the complete chain:

- roadmap
- proposal / PRD
- spec
- ADR
- lane tracker
- implementation plan
- proof expectations
- support and policy references
- closeout memory

## Separation of concerns

- Proposals explain why.
- Specs define required behavior, evidence, and claim boundaries.
- ADRs record durable decisions.
- Plans sequence PRs, proof commands, and rollback.
- Campaign manifests track active execution.
- `policy/*.toml` remains live policy enforcement state.
- Receipts and reports prove claims.

External/tool-specific directories are awareness-only for this lane:

- `.codex/`
- `.spec/`
- `.claude/`
- `.jules/`

Do not treat those directories as durable rails for proposals/specs/ADRs/plans.
Do not add a second durable namespace for source-of-truth artifacts unless a
future ADR deliberately changes the model.

## Authoring guidance

When adding or updating durable artifacts:

1. Keep "why" in proposals.
2. Keep required behavior in specs.
3. Keep durable architecture decisions in ADRs.
4. Keep execution sequencing in lane trackers and implementation plans.
5. Keep support-tier claims and live policy in their existing source surfaces,
   referenced from specs or plans when needed.
6. Keep closeouts as durable records of what landed, what proved it, and what
   remains.
