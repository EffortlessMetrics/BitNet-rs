# BITNET-ADR-0006: PR Closure Creates Backlog Unless Disposed

- **Status:** Accepted
- **Date:** 2026-05-19
- **Linked proposal/spec:**
  [BITNET-SPEC-PR-QUEUE-DISPOSITION](../specs/BITNET-SPEC-PR-QUEUE-DISPOSITION.md)

## Context

BitNet-rs uses PRs as durable work records. A PR can contain implementation,
review comments, CI history, evidence links, diagnostic tooling, and branch
state that is not fully captured in generated dashboards or chat.

During queue burn-down, older PRs can be stale, behind `main`, stacked on closed
parents, diagnostic, noisy, or inconvenient. Those are routing states. Treating
them as close reasons hides work and creates invisible backlog: the underlying
behavior, evidence, or tooling still needs a disposition, but the queue no
longer shows it.

## Decision

Closing a PR is valid backlog reduction only when the work is merged,
duplicated, superseded by a linked landed successor, clean-ported by a linked
landed successor, captured as historical-only evidence in a committed report or
ledger, or explicitly rejected after content audit as having no unique durable
value.

If future work remains, the PR must not close without a live successor PR or a
tracking issue.

The following are invalid close reasons:

```text
old
stale
behind main
root closed
parent closed
needs restack
not based on main
diagnostic-only
noisy
inconvenient
```

Original PR identity is useful work product. The default action is to preserve
that identity by rebasing, restacking, retargeting, or merging from current
`main` where feasible. Replacement PRs are allowed only when the original branch
cannot be safely updated, scope must be narrowed, consolidation is explicit, the
source links to the successor, and the source remains open until the successor
lands unless a tracking issue exists.

## Consequences

- Queue size may remain larger while work is being classified truthfully.
- Agents must review PR content before closing or replacing it.
- Wrong base and closed-parent stacks route to restack, retarget, or successor
  review rather than closure.
- Clean ports close source PRs only after the successor lands.
- Diagnostic PRs require durable/transient classification before closure.
- Bulk close, bulk reopen, bulk recreate, and bulk restack remain outside the
  normal autonomous burn-down model unless explicitly approved.

## Alternatives Considered

- **Close stale PRs and mine them later as source material.** Rejected because
  it loses review history, hides remaining work, and makes future agents infer
  intent from archaeology.
- **Always create replacement PRs for old stacks.** Rejected because PR
  identity, comments, evidence, and CI history are durable assets.
- **Keep every PR open forever.** Rejected because duplicates, landed
  successors, historical-only evidence, and audited no-value changes can be
  closed truthfully.

## How To Revert

Reverting this ADR requires replacing it with another durable operating
decision that prevents queue closure from hiding unlanded work. Existing PRs and
comments remain evidence for what happened even if future policy changes the
allowed disposition categories.
