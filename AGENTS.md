# AGENTS.md

This file provides Codex-specific guidance for working in the BitNet-rs
repository. `CLAUDE.md` remains the broader repository guide; this file records
the campaign authority model that Codex should apply while operating work-item
branches. See
[`docs/development/AGENTIC_PR_OPERATIONS.md`](docs/development/AGENTIC_PR_OPERATIONS.md)
for the durable agentic PR operations reference.

## Repo Source-Of-Truth Stack

BitNet-rs uses a linked source-of-truth stack:

```text
Roadmap → Proposal → Spec → ADR → Plan → Active goal → PR → Proof
```

Before changing files, Codex agents must read:

1. `docs/reference/SPEC_SYSTEM.md`;
2. `.bitnet-rs/goals/active.toml` when present, otherwise the campaign
   `active.toml` explicitly named by the task;
3. the linked implementation plan;
4. the linked spec for the selected work item;
5. linked ADRs for durable constraints.

Work on exactly one ready work item per PR. Proposal PRs explain why, spec PRs
define behavior, ADR PRs record durable decisions, plan PRs define sequencing,
active-goal PRs define current execution state, and runtime PRs must link to
the spec and plan item they implement.

Run the proof commands listed by the selected plan or active goal, plus
`git diff --check`. If a proof command cannot run, record the unavailable
command, why it cannot run, substitute evidence if any, and whether that blocks
merge. Do not hand-edit generated status; run the named generator or checker.

Policy exceptions must update the relevant `policy/*.toml` ledger with owner,
reason, coverage, creation date, review date, and expiry when temporary.

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
6. refresh the agent-owned PR branch when needed, including merge-from-main,
   rebase, `gh pr update-branch`, or `--force-with-lease` after branch, status,
   and diff inspection,
7. address CI, bot, and reviewer feedback,
8. merge the PR when required checks are green and GitHub reports it mergeable,
9. create and merge closeout tracker PRs when required.

Commit, push, PR creation, agent-owned PR branch refresh, CI/bot/reviewer
repair, merge, and tracker closeout are agent responsibilities for those items.
They are not human approval gates.

## Human Gates

Human involvement is required only for true blockers:

- GitHub permissions or branch protection prevent the merge.
- Direct mutation of `origin/main`, destructive cleanup, or secret/model-binary
  exposure is possible.
- Kernel, math, tokenizer, or loader semantics are in unresolved conflict.
- Acceptance criteria conflict with repository policy.
- A cost, exposure, or release decision is genuinely outside the ticket scope.

Older runbook language that routes ordinary commit, push, PR creation, CI
repair, PR branch refresh, merge, or tracker closeout to manual intervention is
superseded by the campaign work item policy above.
