# bitnet-rs to bitnet-rs-swarm Handoff

Status: migration cutoff handoff
Owner: Codex
Created: 2026-05-20
Linked proposal: n/a
Linked specs:

- `docs/reference/SPEC_SYSTEM.md`
- `docs/specs/BITNET-SPEC-PR-QUEUE-DISPOSITION.md`
- `docs/specs/a770-bitnet-claim-boundary.md`
- `docs/specs/intel-arc-a770-gpu-roadmap.md`

Linked ADRs:

- `docs/adr/BITNET-ADR-0006-pr-closure-creates-backlog.md`

Linked plan:

- `plans/a770-bitnet-claim-boundary-implementation.md`

Campaign:

- `docs/tracking/campaigns/intel-a770/active.toml`

PRs:

- active and draft PRs remain authoritative in GitHub until content-audited

## Migration Posture

Active feature, diagnostic, and performance expansion is moving from
`bitnet-rs` to `bitnet-rs-swarm`. Treat `bitnet-rs` as the source repository at
the migration cutoff, not as the place to continue broad queue burn-down.

`bitnet-rs` should receive only:

- migration blocker fixes;
- already-scoped merge candidates;
- source-of-truth handoff documentation;
- CI or routing fixes required to keep existing proof reliable;
- safety fixes for the current repository state.

Move new A770 diagnostics, A770 QK256 OpenCL implementation, CPU AVX2 proof,
behavior tooling, benchmark tooling, device-history ledgers, model matrix work,
Rust-native proof tooling, and broad model-family expansion to
`bitnet-rs-swarm`.

## Operating Rules

Do not close PRs because they are old, behind `main`, from an old stack, noisy,
diagnostic-only, inconvenient, or in need of restack.

Valid close reasons only:

- exact useful content already landed;
- exact useful content was clean-ported and the successor landed;
- true duplicate of a named kept PR;
- historical-only diagnostic evidence was captured in a committed ledger or
  report;
- explicit content rejection after review.

If future work remains, keep the PR open or create and link a tracking issue
before closing. Wrong base means rebase or restack where feasible. Wrongly
closed valid PRs should be restored as the same PR when feasible because PR
identity, review history, and comments are useful state.

Do not bulk-close, bulk-reopen, bulk-rebase, bulk-label, or mass-comment. CI is
scarce; do not trigger CI for queue archaeology or cosmetics.

## Current Cutoff State

Recent A770 hardware-lane facts:

- A770-005 landed a selected-device tiny OpenCL smoke receipt with
  `fallback_used=false` and no BitNet inference, QK256, performance, residency,
  or semantic-quality claim.
- A770-006 landed a minimal `matmul_i2s` OpenCL parity receipt on the selected
  A770 route, still without official BitNet QK256 production semantics or
  BitNet inference claims.
- A770-007 is the selected-device receipt identity step. If still open at
  migration cutoff, keep it scoped to receipt identity and generated tracking
  repair only.

Recent mainline diagnostic and hygiene facts:

- Durable A770 layer-trace diagnostic tooling is on `main`.
- SLM dense Q8 hook receipts remain observational: eager F32 Candle remains the
  selected runtime path and `speedup_claim=false`.
- Apple M4 benchmark-variance and M3 Air accuracy-profile wiring are metadata
  or report surfaces, not new live speed or quality proof.

## Proof-Gated Work

Keep draft and proof-gated work in that state unless the stated proof lands.

- Windows BitNet.cpp fetch/build/install hardening remains proof-gated until a
  full Windows fetch/build/install completes, or the PR is explicitly narrowed
  to script-hardening-only with remaining install proof tracked separately.
- AVX2 QK256 performance work remains proof-gated until official-shape parity,
  hot-path counters, behavior receipts, and repeatable benchmark evidence exist.
- A770 diagnostic and runtime PRs remain content-bearing until an exact
  successor, duplicate, clean port, historical ledger capture, or explicit
  content rejection is proven.

## A770 Continuation Boundary

Do not treat older A770 diagnostic or runtime PRs as disposable. Some old PRs
contain durable reports, tests, receipt schemas, instrumentation, or runtime
correctness fixes.

A770 continuation in `bitnet-rs-swarm` should preserve this order:

1. finish numerical attribution;
2. apply one narrow runtime fix only if the receipt chain proves it;
3. prove CPU/reference behavior;
4. implement real A770 QK256 execution;
5. add strict fallback rejection and counters;
6. prove CPU/A770 parity;
7. prove multi-token behavior;
8. add residency receipts;
9. benchmark only after behavior passes.

No A770 semantic quality, selected attention, resident KV, attention score
residency, softmax residency, value-mix residency, full residency, performance,
or completion claim may be promoted from diagnostic-only evidence.

## Swarm Continuation Targets

Move these lanes to `bitnet-rs-swarm`:

- A770 diagnostic continuation;
- A770 QK256 OpenCL implementation;
- CPU/5700X correctness and AVX2 proof;
- behavior-suite runner;
- benchmark runner;
- device-history ledger;
- model support matrix;
- Rust-native proof tooling;
- broad model-family expansion.

## Cutoff Goal

At cutoff:

- worktree is clean;
- no hidden local `target` artifacts are required for truth;
- drafts remain drafts;
- open PRs remain open unless content-disposition proof exists;
- current active work is documented;
- swarm orchestrators inherit the queue rules and continuation lanes.

## What Not To Do

Do not spend the migration window making GitHub look tidy.

Avoid:

- closing old PRs to reduce count;
- creating replacement PRs for valid old PRs by default;
- rebasing huge branch stacks just before migration;
- opening new diagnostic slices in `bitnet-rs`;
- running expensive CI to classify archaeology;
- promoting partial A770 evidence;
- turning drafts into merge candidates to clean up the queue.

## Next Operator Commands

Use these only for local cutoff hygiene, not for broad queue mutation:

```powershell
git status --short --branch
git worktree list
Get-Process | Where-Object { $_.ProcessName -match '^(cargo|rustc|cl|link|bitnet|xtask)$' }
```

If a local generator or build attempt times out, stop the helper processes and
clear local build artifacts before handoff. Do not rely on uncommitted `target`
state as proof.
