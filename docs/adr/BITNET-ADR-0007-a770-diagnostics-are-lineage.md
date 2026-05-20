# BITNET-ADR-0007: A770 Diagnostics Are Lineage

- **Status:** Accepted
- **Date:** 2026-05-19
- **Linked proposal/spec:**
  [BITNET-SPEC-A770-DIAGNOSTIC-LINEAGE](../specs/BITNET-SPEC-A770-DIAGNOSTIC-LINEAGE.md)

## Context

The A770 branch chain contains diagnostic traces, reference-runner work,
comparators, receipts, tests, and runtime correctness signals. Some entries are
old, stacked, noisy, or based on closed parents. Those facts change routing, but
they do not prove the content has no durable value.

Treating diagnostic PRs as disposable source material loses review history,
breaks successor accounting, and can convert unresolved runtime work into
invisible backlog. Treating diagnostic evidence as product proof creates the
opposite failure: unsupported A770 claims can leak from traces or one-off
receipts.

## Decision

A770 diagnostic PRs are lineage records. Each PR must be classified on content:
durable diagnostic content is preserved, restacked, clean-ported, merged, or
linked to a landed successor before closure; transient evidence may close only
when captured in committed reports, ledgers, newer landed synthesis, or an audit
record.

Bulk close, bulk reopen, bulk recreate, bulk restack, and one-for-one
replacement waves are not normal A770 diagnostic handling. The stack is reviewed
one narrow dependency slice at a time.

Diagnostic lineage does not promote product claims. A diagnostic trace can prove
diagnostic evidence, not quality, speed, residency, selected execution, or broad
A770 support.

## Consequences

- A770 queue work stays slower but more truthful.
- Closed ancestors must be audited before descendants are considered disposed.
- Open descendants are not automatic replacements.
- Successor PRs need exact enough landed content before source PRs close.
- Runtime salvage can use diagnostic lineage as evidence, but claim promotion
  still requires route-specific specs and receipts.
- Generated dashboards summarize state; they do not decide lineage value.

## Alternatives Considered

- **Bulk-close old diagnostic PRs.** Rejected because stale, noisy, and closed
  parent states are routing states, not value judgments.
- **Bulk-recreate successors.** Rejected because replacement waves lose review
  history and create duplicate CI churn.
- **Treat every diagnostic PR as product proof.** Rejected because diagnostics
  can improve visibility without proving execution, quality, speed, or
  residency.
- **Keep every diagnostic PR open forever.** Rejected because transient evidence,
  duplicates, landed successors, and audited no-value content can close with a
  real disposition.

## How To Revert

Reverting this ADR requires another durable operating decision that protects
A770 diagnostic evidence from both false closure and false product promotion.
Existing lineage links, successor records, and closure comments remain evidence.
