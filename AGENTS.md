# AGENTS.md

This file provides Codex-specific guidance for working in the BitNet-rs
repository. `CLAUDE.md` remains the broader repository guide; this file records
the campaign authority model that Codex should apply while operating work-item
branches. See
[`docs/development/AGENTIC_PR_OPERATIONS.md`](docs/development/AGENTIC_PR_OPERATIONS.md)
for the durable agentic PR operations reference.

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
