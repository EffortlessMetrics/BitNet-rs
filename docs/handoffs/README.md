# Handoffs

Handoffs carry operator transfer context after a PR, branch, campaign slice, or
proof lane changes state. They explain what landed, what evidence exists, what
is still uncertain, and what the next maintainer or agent should do.

Handoffs do not own product claims, active work state, or policy enforcement.
Those remain in model coverage ledgers, hardware receipts, status documents,
campaign manifests, and policy TOMLs.

## Source-Of-Truth Role

| Layer | Owns |
| --- | --- |
| Proposal | Why the effort exists |
| Spec | What must be true |
| ADR | Durable decision |
| Plan | PR sequence, proof commands, rollback |
| Campaign `active.toml` | Current executable work |
| Campaign events | Append-only lifecycle state |
| Handoff | Operator transfer context, validation notes, remaining work |
| Closeout | What landed, what did not, and which follow-up owns the gap |
| Status document | User-facing claim tier, proof command, artifact link |
| Policy TOML | Enforceable CI, exception, allowlist, or routing ledger |
| Receipt or artifact | Evidence for what actually happened |

## When To Write One

Write a handoff when:

- a branch or PR is passed to another operator,
- a campaign slice closes but follow-up work remains,
- validation is incomplete for a specific, named reason,
- a blocker depends on external credentials, hardware, policy, or human review,
- a proof lane produced receipts that need a durable operator summary.

Campaign-local `events/*.toml` remain the lifecycle audit trail. Handoffs are
the human-readable transfer layer that points to those events and receipts.

## Handoff Shape

Use this shape when practical:

```md
# <Lane or PR> Handoff

Status:
Linked proposal:
Linked specs:
Linked ADRs:
Linked plan:
Campaign:
PRs:

## Landed

## Evidence

## Validation

## Remaining Work

## Blockers

## Claim Boundary

## Next Operator Commands
```

## Boundaries

Handoffs must not:

- promote a model tier without updating the model coverage matrix,
- claim hardware execution without a hardware receipt,
- claim server readiness without the server readiness proof surface,
- rewrite active campaign state instead of updating campaign manifests/events,
- replace policy TOMLs or workflow gates as CI enforcement,
- hand-edit generated dashboards.
