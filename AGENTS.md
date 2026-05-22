# AGENTS.md

This file provides Codex-specific guidance for working in the BitNet-rs
repository. `CLAUDE.md` remains the broader repository guide; this file records
the campaign authority model that Codex should apply while operating work-item
branches. See
[`docs/development/AGENTIC_PR_OPERATIONS.md`](docs/development/AGENTIC_PR_OPERATIONS.md)
for the durable agentic PR operations reference.

## Repository Role

`EffortlessMetrics/BitNet-rs` remains the public source-of-truth, release, and
publish repository until an explicit sync/cutover says otherwise.

`EffortlessMetrics/bitnet-rs-swarm` is the high-throughput same-repo
development and proof execution repository. Normal feature, hardware,
diagnostic, performance, campaign, refactor, and agent-swarm work lands here
first, then promotes back to `BitNet-rs` through an explicit
release-promotion or sync PR with source swarm commits, included PRs, proof
manifest, changelog, and excluded work.

Do not open normal development PRs in `BitNet-rs` unless explicitly directed,
or unless the PR is a source-repo promotion, sync, release, publish, or
emergency hotfix.

## Merge Method Boundary

Normal swarm PRs into `bitnet-rs-swarm/main` use squash merge.

Source-history repair, source-to-swarm sync, and swarm-to-source promotion PRs
must preserve ancestry. They must land by regular merge commit or by an
explicitly approved fast-forward/direct update. Do not squash these
repository-boundary PRs, because squash can copy the file tree while dropping
the contributor and promotion history needed by the source/swarm cutover.

Do not hard-reset swarm main to source main, squash a history import, or copy a
source tree as a single content commit across the repository boundary. Those
operations make the files look fresh while losing the reachable history that
contributors, active PRs, and promotion audits need.

Never force-push `main`. Release, signing, publish, secrets-heavy workflows,
full-platform matrices, large model-cache workflows, GPU lanes, and public-fork
self-hosted paths remain source-owned until a separate approved migration moves
them.

## Machine Clone Boundary

Machine cutover uses side-by-side clones. Do not retarget an existing
`EffortlessMetrics/BitNet-rs` clone by editing its `origin` remote to point at
`EffortlessMetrics/bitnet-rs-swarm`.

Before moving a machine to swarm work:

1. inspect and checkpoint dirty work in the source clone;
2. commit and push any intended source-repo branch;
3. open or update the source PR if needed;
4. clone `EffortlessMetrics/bitnet-rs-swarm` into a separate directory;
5. start new active development from the swarm clone.

Keep the source clone available for release, publish, signing, emergency
hotfix, and explicit promotion work.

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

## Lane Ownership and Collision Rules

Every swarm PR must declare its lane, campaign, work item, orchestrator,
branch, allowed paths, shared surfaces touched, and whether closeout is
required. The durable contract lives in
[`docs/tracking/LANE_OWNERSHIP.md`](docs/tracking/LANE_OWNERSHIP.md).

The campaign manifest is the source of truth. GitHub labels are navigation
metadata only.

Required PR body fields:

- Lane:
- Campaign:
- Work item:
- Orchestrator:
- Branch:
- Base main SHA:
- Allowed paths:
- Shared surfaces touched:
- Closeout required:

Branch names must use:

- `codex/<lane>/<work-item>-<slug>`
- `claude/<lane>/<work-item>-<slug>`
- `droid/<lane>/<work-item>-<slug>`
- `dependabot/<ecosystem>/<dependency>`

Hardware and runtime lanes are non-stackable by default. Do not combine A770,
CUDA, Apple, Lunar Lake, NPU, server, model-family, and CI-routing work unless
the campaign manifest explicitly allows the overlap.

Generated dashboards are shared surfaces. Do not hand-edit them as the source
of truth. Change the campaign-local source files, run the generator, and
preserve other lanes' current rows when rebasing.

If main moves under a PR and the only conflicts are generated dashboards, keep
both campaign-source changes and regenerate. Do not overwrite another lane's
state.

Closing PRs is not backlog reduction. Do not close for age, stale branch,
branch distance, old stack, or restack need. Close only when content landed,
was clean-ported, is a true duplicate, is historical-only and ledgered, or was
explicitly rejected after content review.

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
