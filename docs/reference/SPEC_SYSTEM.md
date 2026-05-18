# Repo source-of-truth system

BitNet-rs uses a linked source-of-truth stack so humans and agents can find the
right kind of truth in the right artifact. Do not make every document do every
job: separate why, what, durable decisions, sequencing, active execution state,
and proof.

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
| Roadmap | Release direction, milestone framing, lane discovery | Detailed PR queue or proof receipts |
| Proposal | Why a lane exists, user pain, alternatives, success criteria | Behavior contract or implementation order |
| Spec | Required behavior, acceptance examples, proof requirements | Product rationale or PR sequencing |
| ADR | Durable architecture, proof, or operating decision | Task lists or current metric state |
| Plan | PR order, work items, proof commands, rollback | Product strategy or durable decisions |
| Active goal | Current machine-readable objective and work item pointers | Generated dashboards or long-form rationale |
| Support tiers | Public claims and proof pointers | Feature design |
| Policy ledgers | Exceptions, CI intent, owners, coverage, review dates | Broad architecture |

## Canonical locations

| Question | Source of truth |
| --- | --- |
| Why are we doing this? | `docs/proposals/` |
| What must be true? | `docs/specs/` |
| What durable decision constrains it? | `docs/adr/` |
| What PR lands next? | `plans/<lane>/implementation-plan.md` |
| What is the agent actively executing? | `.bitnet-rs/goals/active.toml` when present, otherwise the linked campaign `active.toml` named by the plan |
| What proves a public claim? | `docs/status/`, proof receipts, and CI artifacts |
| What exceptions exist? | `policy/*.toml` |

## Rules

1. One kind of truth per artifact.
2. One semantic artifact per PR unless the selected plan work item says otherwise.
3. Specs define behavior; plans define sequencing.
4. Proposals explain why; ADRs record durable decisions.
5. Active goals tell agents what to do now.
6. Generated status is updated by tools, not by hand.
7. Public claims require support-tier proof or an equivalent proof pointer.
8. Policy exceptions require owner, reason, coverage, and review date.

## Required headers

Use `n/a` when a header does not apply. New proposal, spec, ADR, and plan
artifacts should declare the applicable subset of these fields near the top:

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

1. Read `AGENTS.md` or `CLAUDE.md`.
2. Read this file.
3. Read `.bitnet-rs/goals/active.toml` if it exists.
4. Read the linked implementation plan.
5. Read the linked proposal only for why.
6. Read the linked spec for acceptance.
7. Read linked ADRs for constraints.
8. Inspect current git status before editing.
9. Pick exactly one ready work item.
10. Implement only that work item.
11. Run the proof commands listed by the work item.
12. Update status, receipts, support tiers, or policy ledgers only when the work item requires it.
13. Commit and open or update one focused PR.

If an agent cannot identify a ready work item, it must not invent one. It should
write a handoff or report the missing rail.

## Stop conditions

Stop and report instead of guessing when:

- the active goal is missing, stale, or contradictory;
- linked files do not exist;
- the branch contains unrelated staged changes;
- generated status is dirty and the generator/checker is not identified;
- proof commands cannot run;
- requested work conflicts with an ADR;
- a public claim lacks support-tier proof;
- a policy exception lacks owner, reason, coverage, or review date.

## Active goal lifecycle

The preferred cross-lane active goal path is:

```text
.bitnet-rs/goals/active.toml
```

A paused active goal should state:

```toml
status = "paused"
reason = "No selected implementation lane."
```

Archive replaced active goals under:

```text
.bitnet-rs/goals/archive/YYYY-MM-DD-<lane>.toml
```

Do not leave multiple active cross-lane goal manifests. Existing campaign-local
manifests under `docs/tracking/campaigns/<campaign>/active.toml` remain valid
when a plan or active goal links to them.

## Plan work item shape

```md
## Work item: <id>

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

### Rollback
```

## Closeout format

At the end of a lane, create or update `plans/<lane>/closeout.md`:

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

## Generated status

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

Move the why to `docs/proposals/`; keep the plan focused on work items,
dependencies, proof commands, and rollback.

### Active goal becomes prose

Keep active goals as TOML and link out to docs. Do not put generated tables or
long rationale in the manifest.

### Generated status is hand-edited

Run the generator/checker named by the plan and record receipts rather than
manually changing generated truth.

### Support claims drift

Require support-tier impact headers and proof pointers before broadening README,
release, or user-facing claims.

### Policy exceptions become silent debt

Every exception must have owner, reason, `covered_by`, `created`,
`review_after`, and an expiry when temporary.

### Mega PR

Split the work: one semantic artifact per PR and one implementation work item
per runtime PR unless the selected plan item explicitly allows otherwise.

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
