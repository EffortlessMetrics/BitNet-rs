# Repo source-of-truth system

BitNet-rs uses a linked source-of-truth stack so humans and agents can find the
right kind of truth in one place without reconstructing intent from chat logs,
stale status notes, or generated reports.

## Stack

```text
Roadmap
  -> Proposal
    -> Spec
      -> ADR
        -> Implementation plan
          -> Active goal
            -> Issue / PR
              -> Proof command
              -> CI receipt
              -> Support-tier update
              -> Policy ledger update
```

## Artifact roles

| Artifact | Owns | Does not own |
| --- | --- | --- |
| Roadmap | Release direction, milestone framing, lane inventory | Detailed PR queue, live proof state |
| Proposal | Why, users, alternatives, risks, success criteria | Behavior contract, PR order, generated metrics |
| Spec | Required behavior, acceptance examples, proof requirements | Product rationale, PR sequence |
| ADR | Durable architecture or operating decision | Current task list, metric state |
| Plan | Work-item order, dependencies, proof commands, rollback | Product rationale, durable architecture |
| Active goal | Current machine-readable lane, selected work items, claim boundaries | Generated dashboards, long prose, durable decisions |
| Support tiers | Public claim level, proof pointer, limitations, promotion rule | Feature design, implementation queue |
| Policy ledgers | Exceptions, CI intent, owner, reason, coverage, review date | Broad architecture, informal allowlists |
| Receipts / CI | Evidence for what actually ran | Product intent or future promises |

## Source-of-truth map

| Question | Source of truth |
| --- | --- |
| Why are we doing this? | `docs/proposals/` |
| What must be true? | `docs/specs/` |
| What durable decision constrains it? | `docs/adr/` |
| What PR lands next? | `plans/<lane>/implementation-plan.md` |
| What is the agent actively executing? | `.bitnet-rs/goals/active.toml` when active, or an explicitly linked campaign `active.toml` while a legacy lane still owns execution |
| What proves a public claim? | `docs/status/`, receipts, and CI artifacts |
| What exception exists? | `policy/*.toml` |

## Rules

1. One kind of truth per artifact.
2. One semantic artifact per PR unless the selected plan item explicitly says otherwise.
3. Specs define behavior; plans define sequencing.
4. Proposals explain why; ADRs record durable choices.
5. Active goals tell agents what to do now.
6. Generated status is updated by tools, not by hand.
7. Public claims require support-tier proof or an equivalent proof pointer.
8. Policy exceptions require owner, reason, coverage, creation date, and review date.
9. Runtime/code PRs must link to the spec and plan work item they implement.
10. Proof commands must be run before claiming success, or explicitly recorded as unavailable with the blocking reason.

## Required headers

Use `n/a` when a field does not apply. Proposal, spec, ADR, and plan files
should include the fields relevant to their role.

### Proposal headers

```text
Status:
Owner:
Created:
Target milestone:
Linked specs:
Linked ADRs:
Linked plan:
Support/status impact:
Policy impact:
```

### Spec headers

```text
Status:
Owner:
Created:
Linked proposal:
Linked ADRs:
Linked plan:
Linked issues:
Linked PRs:
Support-tier impact:
Policy impact:
```

### ADR headers

```text
Status:
Date:
Owner:
Linked proposal:
Linked specs:
Linked plan:
```

### Plan headers

```text
Status:
Owner:
Linked proposal:
Linked specs:
Linked ADRs:
Active goal:
```

## Agent workflow

Agents must:

1. Read `AGENTS.md` / `CLAUDE.md` and this file.
2. Read `.bitnet-rs/goals/active.toml` if present; otherwise read the campaign manifest named by the task.
3. Read the linked implementation plan.
4. Select exactly one ready work item.
5. Read the linked proposal only for why.
6. Read the linked spec for acceptance and proof requirements.
7. Read linked ADRs for constraints.
8. Inspect git status before editing.
9. Implement only the selected work item.
10. Run the proof commands listed in the plan item.
11. Update receipts, status, or policy ledgers only when the plan item requires it.
12. Commit and open or update one focused PR.

## Stop conditions

Stop and report instead of guessing when:

- no active goal or campaign work item can be identified;
- linked files do not exist;
- a linked spec or plan is missing;
- requested work conflicts with an ADR;
- generated status is dirty and no generator/checker is specified;
- proof commands cannot run and no substitute evidence is specified;
- unrelated staged changes exist;
- a public claim lacks support-tier proof;
- a policy exception lacks owner, reason, coverage, or review date.

## Active goal lifecycle

### Activate

Create or update:

```text
.bitnet-rs/goals/active.toml
```

with:

```toml
status = "active"
```

### Pause

Use a paused manifest instead of deleting execution state:

```toml
status = "paused"
reason = "No selected implementation lane."
```

### Archive

Move an old active manifest to:

```text
.bitnet-rs/goals/archive/YYYY-MM-DD-<lane>.toml
```

Then create a new active manifest. Do not leave multiple active goals.

## Work item shape

Plans should define PR-sized work items with explicit acceptance and proof:

````md
## Work item: short-id

Status: ready | active | blocked | completed | superseded
Linked proposal:
Linked spec:
Linked ADR:
Blocks:
Blocked by:

### Goal

### Production delta

### Non-goals

### Acceptance

### Proof commands

```bash
git diff --check
```

### Rollback
````

## Closeout format

At the end of a lane, add:

```text
plans/<lane>/closeout.md
```

with:

```md
# Lane closeout: <lane>

Status: completed
Date:
Owner:
Linked proposal:
Linked specs:
Linked ADRs:
Linked plan:
Active goal archive:

## What shipped

## Proof

## Receipts

## PRs

## CI runs

## Support-tier updates

## Policy updates

## What did not ship

## Deferred work

## Claim boundary

## Next lane recommendation
```

## Common failure modes

### Spec becomes a task list

Move PR order to `plans/<lane>/implementation-plan.md`; keep the spec focused
on behavior, examples, proof, and claim boundaries.

### Plan becomes product rationale

Move why and user pain to `docs/proposals/`; keep the plan focused on work
items, dependencies, proof commands, and rollback.

### Active goal becomes prose

Keep `.bitnet-rs/goals/active.toml` machine-readable. Link to prose artifacts
instead of embedding long narrative tables.

### Generated status is hand-edited

Run the named generator/checker. If none exists, stop and add or request a plan
item for the generator/checker rather than mutating generated status by hand.

### Support claims drift

Require a support-tier row or equivalent proof pointer before broadening README,
release, or product claims.

### Policy exceptions become silent debt

Every policy exception must have an owner, reason, `covered_by`, `created`, and
`review_after`; temporary exceptions should also have an expiry.

### Mega PR

Split by source-of-truth role or by one implementation work item. Do not mix a
new proposal, a new spec, an ADR, a plan, an active goal, and runtime behavior
unless the selected plan item explicitly requires that combined change.

## What good looks like

A new contributor or agent can arrive cold and answer:

```text
What are we doing?
Why?
What must be true?
What decision constrains it?
What PR lands next?
What command proves it?
What may we claim?
What must we not claim?
```

If the repo answers those questions without chat history, the source-of-truth
system is working.
