# Plans

Plans translate proposals, specs, and ADRs into PR-sized implementation work.
They should tell a maintainer or agent what to do next, what not to touch, and
which commands prove or disprove the claim.

Plans are not active global goals. The active source of truth for executable
work is still the campaign tracker:

```text
docs/tracking/campaigns/<campaign>/active.toml
```

Use plans for sequencing and proof commands. Use campaign manifests for live
state, ownership, branch names, allowed paths, and merge policy.

## Source-Of-Truth Role

| Layer | Owns |
| --- | --- |
| Proposal | Why |
| Spec | What must be true |
| ADR | Durable decision |
| Plan | PR sequence, proof commands, rollback |
| Campaign `active.toml` | Active work state |
| Campaign events | Append-only lifecycle history |
| Closeout | What landed and what remains |

## Work Item Shape

Plan work items should use this shape when practical:

```md
## Work item: <id>

Status: ready
Linked proposal:
Linked specs:
Linked ADRs:
Campaign item:
Blocked by:
Blocks:

### Goal

### Production delta

### Non-goals

### Acceptance

### Proof commands

### Rollback
```

## Boundaries

Plans must not:

- duplicate generated dashboards,
- create `.adze/goals` or `.bitnet/goals`,
- claim model answer readiness without the answer artifact gate,
- claim hardware validation without lane-specific receipts,
- claim CI budget enforcement unless policy TOMLs and workflow gates enforce it.
