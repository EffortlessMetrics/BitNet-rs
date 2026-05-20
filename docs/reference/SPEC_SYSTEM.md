# Repo source-of-truth system

BitNet-rs uses a linked source-of-truth stack so humans and agents can find the
right authority without reconstructing intent from chat history, stale status
notes, or generated dashboards.

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
| Roadmap | Release direction, milestone framing, lane discovery | Detailed PR order or proof receipts |
| Proposal | Why the work exists, users, alternatives, risks | Behavior contracts or active queues |
| Spec | Required behavior, acceptance examples, proof requirements | Product rationale or PR sequencing |
| ADR | Durable architecture, proof, or operating decisions | Current task lists or generated status |
| Plan | PR order, dependencies, proof commands, rollback | Product strategy or durable decisions |
| Active goal | Machine-readable current lane and work items | Generated metrics or long prose |
| Campaign tracker | Campaign-local execution state, events, generated dashboards | Cross-lane product rationale |
| Support tiers | Public claim tier and proof pointer | Feature design |
| Policy ledgers | Exceptions, CI routing, owners, coverage, review dates | Broad architecture |

## BitNet-rs authority mapping

- Proposals live in `docs/proposals/` and answer why a lane exists.
- Specs live in `docs/specs/` and define what must be true.
- ADRs live in `docs/adr/` and record decisions that should still matter after
  the implementation plan is complete.
- Plans live in `plans/<lane>/` and sequence PR-sized work.
- Active goal manifests live in `.bitnet-rs/goals/` when a lane needs a
  repo-level agent entrypoint; campaign-local active manifests remain in
  `docs/tracking/campaigns/<campaign>/active.toml` for campaign execution.
- Generated campaign dashboards and events remain under
  `docs/tracking/campaigns/<campaign>/generated/` and
  `docs/tracking/campaigns/<campaign>/events/`.
- Public support claims belong in `docs/status/`, hardware matrices, model
  artifact gates, and receipts.
- Policy exceptions and CI routing belong in `policy/*.toml`.
- PR queue disposition rules belong in
  `docs/specs/BITNET-SPEC-PR-QUEUE-DISPOSITION.md`,
  `docs/adr/BITNET-ADR-0006-pr-closure-creates-backlog.md`,
  `docs/tracking/PR_QUEUE_DISPOSITION.md`, and
  `policy/pr-dispositions.toml`.
- Generated tracking conflict rules belong in
  `docs/specs/BITNET-SPEC-GENERATED-TRACKING.md` and
  `policy/generated-tracking.toml`.

## Rules

1. Keep one kind of truth per artifact.
2. Use one semantic artifact per PR unless the selected plan item says otherwise.
3. Specs define behavior; plans define sequencing.
4. Proposals explain why; ADRs record durable decisions.
5. Active goals tell agents what to do now and link to the plan/spec/ADR.
6. Generated status is updated by tools, not by hand; conflicts are resolved by
   repairing source manifests, events, generators, or checkers before
   regenerating.
7. Public claims require support-tier, hardware, model-artifact, or receipt proof.
8. Policy exceptions require owner, reason, coverage, and review date.
9. Closing a PR is a disposition event, not backlog reduction, unless the close
   reason is valid under the PR queue disposition spec and policy ledger.

## Required headers

New proposals, specs, ADRs, and plans should declare the applicable source links
near the top of the file. Use `n/a` where a field does not apply.

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

Existing legacy artifacts may use older headings, but new work should converge
on these fields instead of adding another status vocabulary.

## Agent workflow

Before changing files, agents should:

1. read `AGENTS.md` or `CLAUDE.md`;
2. read this file;
3. read `.bitnet-rs/goals/active.toml` if present, otherwise the relevant
   campaign `active.toml` named by the task;
4. read the linked plan;
5. read the linked spec for acceptance;
6. read linked ADRs for constraints;
7. inspect `git status --short` for unrelated staged or modified files;
8. pick exactly one ready work item;
9. implement only that item;
10. run the listed proof commands plus `git diff --check`;
11. update only required status, receipt, policy, or tracking files.

## Stop conditions

Stop and report instead of guessing when:

- no active goal, campaign item, or explicit task scope exists;
- a linked proposal, spec, ADR, or plan is missing;
- proof commands cannot run and no substitute evidence is allowed;
- generated status is dirty but no generator command is provided;
- unrelated staged changes exist;
- requested work conflicts with an ADR or claim boundary;
- a public claim lacks support-tier, receipt, hardware, or model-artifact proof.

## Active goal lifecycle

A repo-level active goal lives at:

```text
.bitnet-rs/goals/active.toml
```

Use `status = "active"` for an executable lane and `status = "paused"` when no
lane is selected. Archive retired manifests under:

```text
.bitnet-rs/goals/archive/YYYY-MM-DD-<lane>.toml
```

Do not leave multiple repo-level active manifests. If a campaign-local
`docs/tracking/campaigns/<campaign>/active.toml` is the executable authority,
link it from the repo-level manifest or from the selected plan item.

## Closeout format

At the end of a lane, write a closeout under `plans/<lane>/closeout.md` with:

- what shipped;
- proof commands and receipts;
- PRs and CI runs;
- generated status, support-tier, and policy updates;
- deferred work;
- claim boundary;
- next lane recommendation.

## Common failure modes

- If a spec becomes a task list, move PR order to `plans/<lane>/implementation-plan.md`.
- If a plan becomes product rationale, move why-text to `docs/proposals/`.
- If an active goal becomes prose, move details to the plan and keep TOML linked.
- If generated status is hand-edited, add or run the generator/checker.
- If support claims drift, add support-tier proof or narrow the claim.
- If policy exceptions become silent debt, add owner, reason, coverage, and review date.
- If a PR grows into multiple semantic changes, split it by artifact or work item.

## What good looks like

A new contributor or agent can answer these questions from repository files
alone:

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
