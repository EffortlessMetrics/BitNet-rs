# Lane Ownership

Status: active
Owner: swarm maintainers
Created: 2026-05-20

This document defines the coordination contract for swarm PRs. It is meant to
prevent multiple orchestrators from advancing nearby tracker state, generated
dashboards, or branch stacks without naming ownership.

## Required PR Fields

Every swarm PR must declare:

```text
Lane:
Campaign:
Work item:
Orchestrator:
Branch:
Base main SHA:
Allowed paths:
Shared surfaces touched:
Closeout required:
```

Repository-boundary PRs must also declare:

```text
Promotion or sync packet path:
Source commit:
Swarm base commit:
Merge method:
Source impact:
Release/publish/signing impact:
Excluded work:
Machine clone or cutover impact:
```

Example:

```text
Lane: intel-a770
Campaign: intel-a770
Work item: A770-011
Orchestrator: codex-a770
Branch: codex/intel-a770/A770-011-qk256-runtime-contract
Base main SHA: <sha>
Allowed paths:
- docs/tracking/campaigns/intel-a770/**
- docs/specs/a770-bitnet-claim-boundary.md
Shared surfaces touched:
- docs/tracking/generated/**
Closeout required: yes
```

## Labels Mirror The Manifest

GitHub labels help humans and queues, but they are not the source of truth.
Campaign manifests and append-only events remain authoritative:

```text
docs/tracking/campaigns/<campaign>/active.toml
docs/tracking/campaigns/<campaign>/events/*.toml
```

Use lane labels for navigation:

```text
lane:intel-a770
lane:intel-258v
lane:apple-m4
lane:cuda
lane:slm-cpu
lane:qwen
lane:ci
lane:repo-boundary
lane:deps
```

Use status labels for queue state:

```text
state:campaign-ready
state:pr-open
state:closeout-needed
state:blocked
state:mergeable
state:ci-rerun-needed
```

Use repo-boundary and merge-method labels for merge discipline:

```text
merge:squash              # ordinary swarm PR
merge:regular             # source-sync or promotion PR
sync:source-to-swarm      # imports public source history/content into swarm
promote:swarm-to-source   # exports swarm history/content into source
repo-boundary             # crosses the source/swarm boundary
swarm-only                # stays in bitnet-rs-swarm
source-only               # source repo release/hotfix/publish work
pre-sync                  # branch opened before a source-sync repair landed
```

Ordinary swarm PRs default to `merge:squash`. PRs labeled
`sync:source-to-swarm` or `promote:swarm-to-source` must use `merge:regular` or
an explicitly approved fast-forward/direct update that preserves ancestry.

Use shared-surface labels when the PR touches collision-prone files:

```text
shared:generated-dashboard
shared:ci-routing
shared:model-status
shared:release-boundary
```

## Branch Namespaces

Branch names must match the lane:

```text
codex/<lane>/<work-item>-<slug>
claude/<lane>/<work-item>-<slug>
droid/<lane>/<work-item>-<slug>
dependabot/<ecosystem>/<dependency>
```

Examples:

```text
codex/intel-a770/A770-011-qk256-runtime-contract
codex/apple-m4/M4-SERVE-EX-001-dense-serve-conformance
claude/ci/CI-PLANNER-002-gpu-native-emission
codex/repo-boundary/SWARM-AUTHORITY-001-lane-contract
```

Avoid ambiguous branches such as:

```text
codex/fix-stuff
codex/update-docs
claude/pr-backlog-cleanup
codex/a770-random-next
```

## Shared Surfaces

These paths are collision-prone:

```text
docs/tracking/generated/**
docs/tracking/generated/global-dashboard.md
docs/tracking/generated/lane-dashboard.md
docs/tracking/generated/blocked-items.md
docs/tracking/generated/active-prs.md
AGENTS.md
README.md
.github/**
xtask/**
ci/hardware/device-kernel-routing.toml
```

Repository-boundary docs are also shared surfaces:

```text
docs/development/SWARM_DEVELOPMENT_AUTHORITY.md
docs/development/SWARM_HISTORY_REPAIR.md
docs/release/PROMOTE_TO_BITNET_RS.md
docs/tracking/LANE_OWNERSHIP.md
policy/repo-boundary.toml
```

Changes to those files should use `lane:repo-boundary` and must preserve the
source/swarm split, the no-hard-reset/no-squash-import rule, and the
release/publish/signing boundary.

If the PR changes promotion, source-sync, machine cutover, or branch-protection
interpretation, it must include a promotion or sync packet field even when the
value is `n/a` for a swarm-only documentation update.

For source-to-swarm sync PRs, the PR body may be the sync packet. It must name
the exact source commit imported, the swarm base being preserved, the merge
method, included source PRs, validation commands, release-workflow impact, and
excluded swarm work. Do not rely on chat transcripts or local notes as the only
record of a repository-boundary decision.

`policy/repo-boundary.toml` is the machine-readable summary of those repository
roles and history invariants. Keep prose docs and that ledger aligned when
changing promotion, sync, or release-boundary behavior.

A lane may edit its campaign-local `active.toml` and event files directly.
Generated dashboards must be produced by the generator, not hand-edited as the
source of truth.

If a PR touches generated dashboards, the PR body must state which campaign
source changed and include `campaign generate --check` evidence.

If two PRs collide only on generated dashboards, preserve both campaign-source
changes and regenerate. Do not overwrite another lane's state.

## Merge Windows

Ordinary isolated PRs should keep flowing. A PR needs a short merge window only
when it touches shared surfaces that can race with other lanes:

- `docs/tracking/generated/**`;
- `docs/tracking/campaigns/**/active.toml`;
- `.github/workflows/**`;
- `Cargo.lock`;
- root policy files;
- repository authority docs;
- source-sync or promotion packets.

For a shared-surface PR, refresh from current `main`, rerun the relevant
generator or checker, wait for the normalized routed result, and merge promptly
when green. Hold other non-critical shared-surface merges only for that window,
then release the hold.

Source-history repair, source-to-swarm sync, and swarm-to-source promotion
windows are exclusive. Do not merge unrelated `main` updates during those
windows unless the update is urgent and is named in the sync or promotion
packet.

## Lightweight Leases

When a work item touches shared surfaces or blocks related lane work, record a
short-lived lease near the campaign state. Prefer campaign-local leases in
`active.toml`:

```toml
[[lease]]
lane = "intel-a770"
item = "A770-011"
orchestrator = "codex-a770"
branch = "codex/intel-a770/A770-011-qk256-runtime-contract"
scope = "qk256-runtime-contract"
expires_after_hours = 12
shared_surfaces = ["docs/tracking/generated/**"]
```

Leases do not replace PRs or campaign events. They are a coordination hint for
nearby orchestrators before they edit the same shared surfaces.

## Non-Stackable Lanes

Hardware and runtime lanes are non-stackable by default:

```text
intel-a770
intel-258v
apple-m4
cuda
npu
opencl
```

Overlap is allowed only when:

- one PR is implementation work and the other is tracker closeout for an
  already merged item;
- one PR is generated-dashboard repair after main moved;
- the campaign explicitly allows the dependency;
- both PR bodies name the overlap and the newer PR rebases and regenerates.

## Closeout PRs

Closeout is not implementation. Use:

```text
lane:<lane>
state:closeout-needed
type:tracker-closeout
```

Title format:

```text
docs(a770): close <work item>
```

The body must include:

```text
Source PR:
Head SHA:
Merge SHA:
Claim boundary:
Generator validation:
Campaign check:
```

## Dependabot Quarantine

Dependabot PRs should not become hidden lane work.

Label dependency PRs with:

```text
lane:deps
shared:ci-routing   # only for workflow, action, or runtime bumps
runtime:node24      # for Node 24 action bumps
```

Do not merge action-runtime bumps until runner compatibility is confirmed.
Batch related Node 24 action PRs mentally. Dependabot PRs must not update
generated campaign dashboards.

## Future Checker

A lightweight checker may later enforce this contract:

```text
cargo run -p xtask --no-default-features -- lane-check
```

The first version should catch only obvious hazards:

- PR title or body includes `Lane:`;
- PR body includes `Campaign:` or explicitly says `Campaign: none`;
- branch name starts with `codex/<lane>/`, `claude/<lane>/`,
  `droid/<lane>/`, or `dependabot/`;
- generated dashboard changes also include campaign-source changes;
- `pr_open` state includes a PR number;
- `pr_open` events include `head_sha`;
- event type is known;
- generated dashboards do not contain conflict markers.

Do not turn the checker into a second workflow engine.
