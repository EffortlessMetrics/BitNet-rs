# PR Queue Disposition

Status: proposed
Owner: release/ci
Created: 2026-05-19
Linked proposal: n/a
Linked specs:
[BITNET-SPEC-PR-QUEUE-DISPOSITION](../specs/BITNET-SPEC-PR-QUEUE-DISPOSITION.md)
Linked ADRs:
[BITNET-ADR-0006](../adr/BITNET-ADR-0006-pr-closure-creates-backlog.md)
Linked plan: n/a
Linked issues: n/a
Linked PRs: n/a
Support-tier impact: none
Policy impact: `policy/pr-dispositions.toml`

## Operator Rule

Closing a PR is not backlog reduction unless the work is merged, duplicated,
superseded by a linked landed successor, clean-ported by a linked landed
successor, captured as historical-only evidence, or explicitly rejected after a
content audit as having no unique durable value.

If future work remains, create or link a tracking issue or keep a live successor
PR before closing the source PR.

## Do Not Close Because

- old;
- stale;
- behind `main`;
- root or parent closed;
- needs restack;
- not based on `main`;
- diagnostic-only;
- noisy;
- inconvenient.

These are routing states, not close reasons.

## Default Routing

| State | Next action |
| --- | --- |
| Valid PR on stale base | Rebase or restack the same PR |
| Wrongly closed valid PR | Reopen the same PR |
| Original branch cannot be made usable | Create explicit successor and link source |
| Successor has landed | Close source with successor PR and commit link |
| Durable diagnostic tooling exists | Keep open, port, or merge |
| Transient evidence only | Close only after committed report or ledger capture |

## Close Comment Template

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

Use `n/a` only when a field does not apply. Do not leave `reason`,
`audit_summary`, or `remaining_work` ambiguous.

## Review Checklist

- The close reason is valid under `policy/pr-dispositions.toml`.
- Any successor PR has landed before the source closes.
- Any duplicate or historical-only claim links the durable evidence.
- Any future work has a live successor PR or tracking issue.
- Diagnostic content was classified as durable or transient.
- The close comment records the disposition in a form future agents can audit.
