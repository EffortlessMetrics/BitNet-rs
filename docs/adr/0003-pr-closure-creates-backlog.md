# ADR-0003: PR closure creates backlog unless durable disposition conditions are met

Status: Accepted
Owner: BitNet maintainers
Created: 2026-05-19
Linked proposal: n/a
Linked specs: docs/specs/BITNET-SPEC-PR-QUEUE-DISPOSITION.md
Linked ADRs: n/a
Linked plan: n/a
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: none
Policy impact: policy/pr-dispositions.toml

## Context

Past queue cleanup actions closed PRs for operational convenience (stale stack, parent closure, restack required) without proving that durable value was merged, superseded, or intentionally rejected after audit.

## Decision

Adopt closure law as a durable repo operating decision:

- Closing is not backlog reduction unless the disposition satisfies explicit durable reasons.
- Operational states (stale base, restack needed, parent closed) route to repair actions, not disposal.
- If future work remains, closure requires successor link or tracking issue.

## Consequences

- Queue cleanup must preserve evidence lineage.
- PR disposition comments must include durable reason and links.
- Future automation can validate closure metadata against policy.
