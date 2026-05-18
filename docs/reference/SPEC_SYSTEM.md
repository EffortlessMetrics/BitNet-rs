# Repo source-of-truth system

BitNet-rs uses a linked source-of-truth stack so humans and agents can find the
right kind of truth in the right place without relying on chat history.

## Stack

```text
Roadmap
  -> Proposal
    -> Spec
      -> ADR
        -> Implementation plan
          -> Active goal
            -> PR
              -> Proof
```

## Artifact roles

| Artifact | Owns | Does not own |
| --- | --- | --- |
| Roadmap | Release direction, milestone framing, lane inventory | Detailed PR queue, live status, proof receipts |
| Proposal | Why the effort exists, users, alternatives, success criteria | Behavior contract, implementation queue, generated status |
| Spec | Required behavior, acceptance, proof, claim boundaries | Product rationale, PR order, active queue |
| ADR | Durable architecture or operating decision | Task list, current metric state, implementation queue |
| Plan | PR order, work items, proof commands, rollback | Product rationale, durable decisions, generated status truth |
| Active goal | Current machine-readable work, claim boundaries, proof pointers | Long prose, generated metrics, durable decisions |
| Support tiers | Public claim tier, proof commands, limitations, promotion rule | Feature design or task sequencing |
| Policy ledgers | Exceptions, CI/policy receipts, owners, review dates | Broad architecture or product strategy |

## Rules

1. One kind of truth per artifact.
2. One semantic artifact per PR unless the plan says otherwise.
3. Proposals explain why; specs define what must be true.
4. ADRs record durable decisions; plans define sequencing.
5. Active goals tell agents what to do now.
6. Generated status is updated by tools, not by hand.
7. Public claims require support-tier proof or an equivalent receipt pointer.
8. Policy exceptions require owner, reason, coverage, and review date.

## Required headers

Use `n/a` when a header is not applicable. New proposals, specs, ADRs, and
plans should include the relevant subset of these fields:

```text
Status:
Owner:
Created:
Linked proposal:
Linked specs:
Linked ADRs:
Linked plan:
Linked issues:
Linked PRs:
Support-tier impact:
Policy impact:
```

## Agent workflow

Agents must:

1. read repo instructions (`AGENTS.md` and/or `CLAUDE.md`);
2. read this file;
3. read `.bitnet/goals/active.toml` when present, or the campaign-local
   `docs/tracking/campaigns/<campaign>/active.toml` named by the task;
4. choose exactly one ready work item;
5. read the linked plan, spec, and ADRs;
6. implement only that item;
7. run the listed proof commands and `git diff --check`;
8. update receipts, status, policy, or support tiers only when the work item
   requires it;
9. stop on missing or contradictory source-of-truth artifacts.

## Stop conditions

Stop and report instead of guessing when:

- the active goal or campaign item is missing or stale;
- linked files do not exist;
- generated status is dirty;
- proof commands cannot run and no substitute evidence is defined;
- unrelated staged changes exist;
- requested work conflicts with an ADR;
- a public claim lacks support-tier proof.

## Active goal lifecycle

The repo-level active goal, when used, lives at:

```text
.bitnet/goals/active.toml
```

Campaign-local trackers may also use:

```text
docs/tracking/campaigns/<campaign>/active.toml
```

Use exactly one active execution authority for a work item. If a new repo-level
active goal replaces an old one, archive the old manifest at:

```text
.bitnet/goals/archive/YYYY-MM-DD-<lane>.toml
```

A paused repo-level goal should use:

```toml
status = "paused"
reason = "No selected implementation lane."
```

## Closeout format

At the end of a lane, write the closeout where the plan lives, normally:

```text
plans/<lane>/closeout.md
```

A closeout should record what shipped, proof commands, receipts, PRs, CI runs,
support-tier updates, policy updates, deferred work, claim boundaries, and the
next lane recommendation.

## Common failure modes

### Spec becomes a task list

Move PR order to `plans/<lane>/implementation-plan.md`; keep the spec focused
on behavior, examples, proof, and claim boundaries.

### Plan becomes product rationale

Move the why to `docs/proposals/`; keep plans focused on work items, proof, and
rollback.

### Active goal becomes prose

Keep the active goal as TOML. Link to docs instead of embedding long generated
tables or design rationale.

### Agent hand-edits generated status

Run the named generator/checker. If no generator exists, stop and record the
missing tool rather than silently editing generated state.

### Support claims drift

Require a support-tier impact header and a support-tier row or proof pointer
before adding README or marketing claims.

### Policy exceptions become silent debt

Every exception must have an owner, reason, `covered_by`, `created`, and
`review_after`; temporary exceptions should also have `expires`.

### Mega PR

Keep one semantic artifact or one implementation work item per PR unless the
plan explicitly authorizes bundling.

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

If the repository answers those questions without chat history, the source-of-
truth system is working.
