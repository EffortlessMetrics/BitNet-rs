# ADR-0004: Preserve PR identity by default

Status: Accepted
Owner: BitNet maintainers
Created: 2026-05-19
Linked proposal: n/a
Linked specs: docs/specs/BITNET-SPEC-PR-QUEUE-DISPOSITION.md
Linked ADRs: docs/adr/0003-pr-closure-creates-backlog.md
Linked plan: n/a
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: none
Policy impact: policy/pr-dispositions.toml

## Context

Review threads, CI history, receipts, and branch context are durable assets. Recreating PRs by default discards this context and inflates queue churn.

## Decision

Preserve original PR identity by default. Preferred action is rebase/restack/retarget existing PR.

Replacement PR is allowed only when:

1. original branch cannot be safely updated,
2. scope must be narrowed, or
3. explicit consolidation is planned.

When replacement happens, source PR must link successor and stay open until successor lands unless a tracking issue exists.

## Consequences

- Less review context loss and less CI churn.
- Clear lineage between source PRs and successor ports.
