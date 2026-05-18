# Plans

Plans translate proposals, specs, and ADRs into PR-sized implementation work.
They tell a maintainer or agent what to do next, what not to touch, which
commands prove or disprove the claim, and how to roll back safely.

Plans are queues, not product strategy. Product rationale belongs in proposals;
required behavior belongs in specs; durable choices belong in ADRs; live
execution state belongs in `.bitnet/goals/active.toml` or the selected
campaign-local `active.toml`.

## Source-of-truth role

| Layer | Owns |
| --- | --- |
| Proposal | Why |
| Spec | What must be true |
| ADR | Durable decision |
| Plan | PR sequence, proof commands, rollback |
| Active goal or campaign manifest | Active work state |
| Campaign events | Append-only lifecycle history |
| Closeout | What landed and what remains |

## Work item shape

Plan work items should use this shape when practical:

````md
## Work item: <id>

Status: ready | active | blocked | completed | superseded
Linked proposal:
Linked specs:
Linked ADRs:
Active goal or campaign item:
Blocked by:
Blocks:

### Goal

### Production delta

### Non-goals

### Acceptance

### Proof commands

```bash
git diff --check
```

### Rollback

### Notes
````

## Boundaries

Plans must not:

- duplicate generated dashboards;
- claim model answer readiness without the answer artifact gate;
- claim hardware validation without lane-specific receipts;
- claim CI budget enforcement unless policy TOMLs and workflow gates enforce it;
- turn specs into task lists or ADRs into implementation queues.
