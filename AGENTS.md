# AGENTS.md

This file provides Codex-specific guidance for working in the BitNet-rs
repository. `CLAUDE.md` remains the broader repository guide; this file records
the campaign authority model that Codex should apply while operating work-item
branches.


## Repo source-of-truth stack

BitNet-rs uses a linked source-of-truth stack:

```text
Roadmap -> Proposal -> Spec -> ADR -> Plan -> Active goal -> PR -> Proof
```

Before changing files, read:

1. `docs/reference/SPEC_SYSTEM.md`
2. `.bitnet-rs/goals/active.toml` when present, or the campaign manifest named by the selected plan item
3. The linked implementation plan
4. The linked spec for the selected work item
5. Linked ADRs

### Scope rule

Implement one work item per PR. Docs-only artifacts are separate PRs unless the
selected plan item explicitly says otherwise:

- proposal PRs explain why;
- spec PRs define behavior;
- ADR PRs record durable decisions;
- plan PRs define sequencing;
- active goal PRs define current execution state.

Runtime/code PRs must link to the spec and plan item they implement.

### Proof rule

Run the proof commands listed in the plan item. If a proof command cannot run,
record the command, why it was unavailable, substitute evidence if any, and
whether the missing proof blocks merge. Do not claim success without proof.

### Generated status rule

Do not hand-edit generated status. Run the generator/checker named in the plan
or stop and report that the source-of-truth stack needs a generator/checker
work item.

### Policy rule

If you add an exception, add or update the relevant `policy/*.toml` ledger with
owner, reason, `covered_by`, `created`, `review_after`, and `expires` when the
exception is temporary.

### Stop conditions

Stop and report instead of guessing when the active goal is missing or stale,
linked specs or plans are missing, proof commands cannot run, unrelated staged
changes exist, generated status is dirty, or requested behavior contradicts an
ADR.

## Campaign Work Item Authority

Campaign work items are the source of truth for review, PR, and merge flow.
For items with:

- `review_mode = "codex_premerge"`
- `merge_policy = "automerge_when_green"`
- `human_gate = "on_blocker_only"`

Codex agents are authorized and expected to:

1. edit files within the item scope,
2. run scoped validation,
3. commit,
4. push,
5. open or update the PR,
6. address CI, bot, and reviewer feedback,
7. merge the PR when required checks are green and GitHub reports it mergeable,
8. create and merge closeout tracker PRs when required.

Commit, push, PR creation, CI/bot/reviewer repair, merge, and tracker closeout
are agent responsibilities for those items. They are not human approval gates.

## Human Gates

Human involvement is required only for true blockers:

- GitHub permissions or branch protection prevent the merge.
- Destructive data loss or secret/model-binary exposure is possible.
- Kernel, math, tokenizer, or loader semantics are in unresolved conflict.
- Acceptance criteria conflict with repository policy.
- A cost, exposure, or release decision is genuinely outside the ticket scope.

Older runbook language that routes ordinary commit, push, PR creation, CI
repair, merge, or tracker closeout to manual intervention is superseded by the
campaign work item policy above.
