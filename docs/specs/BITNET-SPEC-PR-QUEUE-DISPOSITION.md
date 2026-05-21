# BITNET-SPEC-PR-QUEUE-DISPOSITION: PR Queue Disposition And Closure Law

Status: proposed
Owner: release/ci
Created: 2026-05-19
Linked proposal: n/a
Linked specs:
[BITNET-SPEC-0001](BITNET-SPEC-0001-source-of-truth-and-claim-boundaries.md)
Linked ADRs:
[BITNET-ADR-0006](../adr/BITNET-ADR-0006-pr-closure-creates-backlog.md)
Linked plan: n/a
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: none
Policy impact: `policy/pr-dispositions.toml`

## Purpose

BitNet-rs PR queue work must not reduce visible backlog by hiding useful work.
Closing a PR is a disposition event, not proof that the work no longer exists.
This spec defines valid close reasons, invalid close reasons, routing states,
tracking requirements, and acceptance examples for stale, restacked, diagnostic,
and successor PRs.

This spec does not define today's queue order. Plans and campaign active
manifests own sequencing. This spec defines the rule that survives the current
queue.

## Source-Of-Truth Authorities

PR disposition truth lives in:

- this spec;
- [PR closure creates backlog ADR](../adr/BITNET-ADR-0006-pr-closure-creates-backlog.md);
- `policy/pr-dispositions.toml`;
- [PR Queue Disposition](../tracking/PR_QUEUE_DISPOSITION.md);
- PR bodies, comments, and linked issues that record the actual disposition.

Generated campaign dashboards may summarize PR state, but they do not create a
close reason. If a generated dashboard conflicts with this spec, repair the
source manifest or generator.

## Valid Close Reasons

A PR close counts as backlog reduction only when the close comment or linked
record proves one of these reasons:

- the PR merged;
- the PR is an exact duplicate of work that remains available elsewhere;
- the PR is superseded by a linked successor that already landed;
- the PR was clean-ported and the port already landed;
- the PR is historical-only evidence and that evidence was captured in a
  committed report or ledger;
- content audit found no unique durable value and records the audit result.

Any close that leaves future work behind must link a live successor PR or a
tracking issue before the source PR closes.

## Invalid Close Reasons

The following are not close reasons:

- old;
- stale;
- behind `main`;
- root or parent closed;
- needs restack;
- not based on `main`;
- diagnostic-only;
- noisy;
- inconvenient.

These facts may change routing. They do not prove that the underlying work has
no value.

## Routing States

Routing states tell maintainers and agents what action is allowed next:

| Routing state | Meaning | Allowed action |
| --- | --- | --- |
| stale stack | Direct merge is unsafe | Rebase, restack, retarget, or successor-port |
| needs restack | Branch shape is outdated | Refresh the same PR when feasible |
| closed parent | Stack ancestry is broken | Review the child content on its own merits |
| diagnostic-only | Evidence or tooling is diagnostic | Classify durable vs transient value |
| wrong base | Base branch is wrong | Retarget or rebase where safe |

None of these states permits closure by itself.

## PR Identity Preservation

Original PR identity is useful work product. Review comments, CI history, body
text, source branch, and linked evidence are assets.

Default action is to preserve the PR identity by rebasing, restacking,
retargeting, or merging from current `main` when that is feasible and safe.

A replacement PR is allowed only when:

- the original branch cannot be safely updated;
- scope must be narrowed for review;
- consolidation is explicit;
- the source PR links to the successor;
- the source remains open until the successor lands, unless a tracking issue
  exists for the remaining work.

## Diagnostic PRs

Diagnostic PRs are not disposable by default. A diagnostic PR has durable value
when it contains reusable trace instrumentation, reference-runner tooling,
comparator logic, receipt schema fields, focused regression tests, final
synthesis reports, or runtime/model correctness signals.

Durable diagnostic content must be merged as-is when valid, restacked when the
base is wrong, clean-ported when the original cannot be made usable, or linked
to a landed successor before closure.

Transient diagnostic evidence may remain closed only when captured by a
committed report, ledger, or newer landed synthesis.

## Acceptance Examples

| Case | Required disposition |
| --- | --- |
| PR targets a stale base but is otherwise valid | Rebase or restack the same PR |
| PR was wrongly closed but still has durable value | Reopen the same PR |
| Clean port landed on current `main` | Close the source PR with successor link |
| Future work remains and no successor exists | Create or link a tracking issue before close |
| Diagnostic PR has durable tooling | Keep open, port, or merge; do not close as old |
| PR has no unique durable value after audit | Close with audit note and evidence links |

## Close Comment Requirements

A close comment or linked disposition record must include:

```text
disposition:
reason:
source_pr:
landed_pr:
landed_commit:
duplicate_of:
historical_report:
tracking_issue:
audit_summary:
remaining_work:
```

Fields that do not apply may be `n/a`, but at least one valid close reason must
be populated.

## Proof Commands

Current contract validation:

```bash
cargo run --locked -p xtask --no-default-features -- check-file-policy --report-dir target/bitnet/reports
cargo run --locked -p xtask --no-default-features -- policy-report --report-dir target/bitnet/reports
git diff --check
```

Future enforcement should live behind:

```bash
cargo run --locked -p xtask --no-default-features -- check-pr-dispositions
```

The future checker should fail if a PR close comment lacks a valid close reason,
landed successor, duplicate link, historical report link, audit record, or
tracking issue when future work remains.

## Non-Goals

- Do not encode today's exact PR queue.
- Do not encode current open PR order.
- Do not encode temporary CI failures.
- Do not encode one agent's command transcript.
- Do not require CI for archaeology.
- Do not convert stale, diagnostic, or non-main PRs into disposable source
  material by default.

