# BITNET-SPEC-A770-DIAGNOSTIC-LINEAGE: A770 Diagnostic Lineage Handling

Status: proposed
Owner: release/runtime
Created: 2026-05-19
Linked proposal: n/a
Linked specs:
[BITNET-SPEC-PR-QUEUE-DISPOSITION](BITNET-SPEC-PR-QUEUE-DISPOSITION.md),
[A770 BitNet Claim Boundary](a770-bitnet-claim-boundary.md)
Linked ADRs:
[BITNET-ADR-0007](../adr/BITNET-ADR-0007-a770-diagnostics-are-lineage.md)
Linked plan:
[A770 diagnostic lineage implementation plan](../../plans/a770-diagnostic-lineage/implementation-plan.md)
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: none
Policy impact: `policy/a770-diagnostic-lineage.toml`

## Purpose

The A770 diagnostic branch chain is lineage. It is not a disposable pile of old
PRs, and it is not product proof by itself. This spec defines how to classify
diagnostic content, preserve PR identity, close only with valid disposition, and
avoid batch operations that hide useful runtime evidence.

This spec governs operational handling of A770 diagnostic PRs. Runtime behavior,
quality promotion, residency, performance, and user-facing support claims remain
owned by the A770 claim-boundary specs, runtime salvage specs, support-tier
surfaces, and receipts.

## Source-Of-Truth Authorities

A770 diagnostic lineage truth lives in:

- this spec;
- [A770 diagnostics are lineage ADR](../adr/BITNET-ADR-0007-a770-diagnostics-are-lineage.md);
- [A770 diagnostic lineage plan](../../plans/a770-diagnostic-lineage/implementation-plan.md);
- `policy/a770-diagnostic-lineage.toml`;
- the general [PR queue disposition spec](BITNET-SPEC-PR-QUEUE-DISPOSITION.md);
- PR bodies, comments, linked successor PRs, and tracking issues.

Generated dashboards may summarize A770 PR state, but they do not decide
whether diagnostic content is durable or transient.

## Durable Diagnostic Content

Diagnostic content is durable when it contains reusable or claim-relevant work,
including:

- trace instrumentation;
- reference-runner tooling;
- comparator logic;
- receipt schema fields;
- focused regression tests;
- final synthesis reports;
- runtime or model correctness signals.

Durable diagnostic content must be handled by one of these dispositions:

- merge the original PR as-is when the branch is valid and proof is current;
- rebase, restack, retarget, or merge from current `main` when the base is
  wrong but the PR identity remains usable;
- clean-port the exact durable subset when the original branch cannot be made
  usable;
- keep the source PR open until a linked successor lands;
- close only after a landed successor, duplicate, committed historical report,
  audit rejection, or tracking issue satisfies the PR queue disposition spec.

## Transient Diagnostic Content

Diagnostic content is transient when it is only:

- a one-off target-local observation;
- intermediate evidence that a mismatch moved to another location;
- a contradicted hypothesis already captured by newer proof;
- a manual receipt without reusable code, schema, tooling, or synthesis.

Transient evidence may remain closed only when captured by a committed report,
ledger, newer landed synthesis, or audited disposition comment. A transient
classification must explain why no durable code, schema, test, or runtime signal
remains.

## Forbidden Inferences

The following inferences are invalid:

- diagnostic-only means disposable;
- root closed means descendants should close;
- closed ancestor means source material by default;
- open descendant means replacement by default;
- stale stack means close;
- needs restack means close;
- wrong base means close;
- diagnostic evidence means A770 support, quality, speed, or residency proof.

Diagnostic evidence may justify a narrow runtime salvage review. It does not
promote product claims without the applicable runtime, support-tier, receipt,
and claim-boundary gates.

## Frontier Model

A770 diagnostic review moves through one narrow frontier at a time:

| Frontier state | Meaning | Required action |
| --- | --- | --- |
| closed ancestor | Older branch or parent closed | Audit content before treating it as disposed |
| open descendant | Later PR remains open | Compare exact content before treating it as successor |
| stale base | PR cannot merge directly | Rebase/restack same PR where feasible |
| successor candidate | Clean port or narrowed PR exists | Keep source open until successor lands |
| historical evidence | No reusable durable value remains | Link committed report or ledger before close |

A landed successor must be exact enough to close its source. "Inspired by" or
"source material" is not enough.

## Prohibited Batch Actions

Agents must not process the A770 diagnostic stack as a bulk object:

- no bulk close;
- no bulk reopen;
- no bulk recreate;
- no bulk restack;
- no one-for-one replacement wave;
- no CI for archaeology.

One dependency slice may be reviewed at a time after current-main merge
candidates are drained. The default is to preserve the original PR identity and
create a successor only when the original cannot be made usable.

## Acceptance Examples

| Case | Required handling |
| --- | --- |
| PR contains reusable comparator code but targets a stale stack | Restack or clean-port; do not close as stale |
| PR only records a contradicted local mismatch and a newer synthesis landed | Link the synthesis and close as historical evidence captured |
| Closed parent blocks direct merge of a valid child | Review child content on its own merits |
| Descendant PR overlaps but lacks exact tests from the source | Keep source open or create tracking for missing tests |
| Clean successor lands with exact durable content | Close source with successor link and landed commit |
| A diagnostic trace improves visibility | Claim diagnostic evidence only, not A770 product readiness |

## Proof Commands

Current contract validation:

```bash
cargo run --locked -p xtask --no-default-features -- check-file-policy --report-dir target/bitnet/reports
cargo run --locked -p xtask --no-default-features -- policy-report --report-dir target/bitnet/reports
git diff --check
```

Future enforcement may be added to the PR disposition checker by loading
`policy/a770-diagnostic-lineage.toml` and requiring an explicit durable or
transient classification for A770 diagnostic closures.

## Non-Goals

- Do not encode today's exact A770 queue.
- Do not claim A770 support, quality, speed, residency, or selected execution.
- Do not port runtime fixes in this spec PR.
- Do not replace the A770 runtime salvage spec.
- Do not turn generated dashboard rows into hand-authored truth.
