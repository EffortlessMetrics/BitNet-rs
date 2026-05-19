# BITNET-SPEC-PR-QUEUE-DISPOSITION

Status: Draft
Owner: BitNet maintainers
Created: 2026-05-19
Linked proposal: n/a
Linked specs: docs/reference/SPEC_SYSTEM.md
Linked ADRs: docs/adr/0003-pr-closure-creates-backlog.md, docs/adr/0004-preserve-pr-identity.md
Linked plan: n/a
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: none
Policy impact: policy/pr-dispositions.toml

## Purpose

Define durable disposition law for pull requests so queue burndown cannot reinterpret valuable work as disposable backlog reduction.

## Required behavior

### Closure does not equal backlog reduction

Closing a PR counts as backlog reduction only when one of the following is true:

1. PR is merged.
2. PR is duplicate of another open/merged PR with explicit link.
3. PR is superseded by a linked landed equivalent.
4. PR was clean-ported and the successor port is landed.
5. PR is historical-only and evidence is captured in a committed report or ledger.
6. PR was explicitly rejected after content audit with no unique durable value.

If future work remains, closure requires a linked tracking issue or live successor PR.

### Invalid close reasons

The following are invalid as stand-alone closure reasons:

- old/stale age,
- behind main,
- parent/root closed,
- needs restack,
- not based on main,
- diagnostic-only,
- noisy,
- inconvenient.

### Routing states

- stale stack means no direct merge, not mandatory close;
- needs restack means repair path, not close;
- parent/root closed does not force descendant closure;
- diagnostic-only does not imply disposable work.

### PR identity preservation

Default action is to preserve PR identity and update the same PR through rebase/restack/retarget. Replacement PRs are allowed only when branch repair is unsafe, scope narrowing is required, or explicit consolidation is documented. Source PR must link successor and remain open until successor lands unless a tracking issue exists.

## Acceptance examples

1. PR targets stale base but valid content -> rebase/restack same PR.
2. PR wrongly closed but still valid -> reopen same PR.
3. Clean-port landed -> close source PR with successor link.
4. Future work remains with no successor -> create tracking issue before close.
5. Diagnostic PR includes durable tooling -> keep open, port, or merge; do not close as old.

## Proof requirements

- `cargo run --locked -p xtask --no-default-features -- check-file-policy --report-dir target/bitnet/reports`
- `cargo run --locked -p xtask --no-default-features -- policy-report --report-dir target/bitnet/reports`
- `git diff --check`
