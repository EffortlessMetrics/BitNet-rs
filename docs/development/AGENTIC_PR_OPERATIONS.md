# Agentic PR Operations

This document defines the normal Codex/swarm operating model for BitNet-rs PR
lanes. It exists to prevent older handoff language from turning ordinary PR
operations into human approval gates.

The campaign work item remains the execution authority. Use this document to
interpret work items whose tracker fields are:

```toml
review_mode = "codex_premerge"
merge_policy = "automerge_when_green"
human_gate = "on_blocker_only"
```

## Normal Autonomous Operations

For those work items, Codex agents are expected to carry the branch through the
review loop:

1. inspect the current branch, status, diff, and relevant source-of-truth docs,
2. preserve unrelated user work,
3. create or reuse a scoped feature branch,
4. port, merge from `main`, rebase, or use `gh pr update-branch` when the
   agent-owned PR branch needs refresh,
5. stage only scoped files,
6. commit,
7. push the feature branch,
8. open or update the PR,
9. address CI, bot, and reviewer feedback,
10. merge when required checks are green and GitHub reports the PR mergeable,
11. create and merge tracker closeout PRs when the work item requires them.

These actions are not human gates for `codex_premerge` +
`automerge_when_green` + `on_blocker_only` items. They are the expected agent
workflow.

## Required Safeguards

Before any git operation that changes state, inspect branch, status, and the
relevant diff. Before staging, identify unrelated or uncommitted user work.

Agent-owned branch refresh is allowed when scoped to the PR branch and after
inspection. This includes merge-from-main, rebase, `gh pr update-branch`, and
`--force-with-lease` for the agent-owned feature branch.

Do not mutate `origin/main` directly. Merge through GitHub PRs and branch
protection.

## Human Gates

Human involvement is required only for true blockers:

- GitHub permissions or branch protection prevent the merge.
- The operation would directly mutate `origin/main`.
- Destructive cleanup, branch deletion, or history rewrite outside the
  agent-owned PR branch is required.
- Unrelated user work would be overwritten, deleted, or staged.
- Secrets, credentials, private payloads, or large model binaries may be
  committed or exposed.
- Kernel, math, tokenizer, loader, or claim semantics conflict with the work
  item acceptance criteria.
- Acceptance criteria conflict with repository policy.
- A cost, exposure, publish, release, or external decision is genuinely outside
  the ticket scope.

## Not A Blocker

The following are normal PR work, not blockers by themselves:

- creating a feature branch,
- committing scoped changes,
- pushing a feature branch,
- opening or updating a PR,
- refreshing an agent-owned PR branch,
- fixing CI or bot feedback,
- self-reviewing the diff,
- merging a green, mergeable PR,
- closing tracker items after the PR lands.

Older runbooks, handoffs, or notes that require explicit human phrasing for
ordinary PR operations are superseded by the campaign work item fields and this
document. Keep branch safety, user-work preservation, and claim boundaries
strict; do not reintroduce exact-phrase approval gates for routine swarm ops.
