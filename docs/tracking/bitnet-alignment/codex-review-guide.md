# Codex Review Guide

## Purpose

Use this guide when reviewing Codex-generated PRs for the bitnet-rs alignment burndown.

## Review order

1. Identify the linked work item in `workstream-ledger.yaml`.
2. Verify the PR touches only `scope.allowed_paths`.
3. Verify the PR avoids `scope.forbidden_paths`.
4. Confirm the implementation satisfies every acceptance criterion.
5. Confirm the PR updates `status.md`.
6. Confirm the PR updates the work item state.
7. Confirm verification commands were actually run.
8. If verification was not run, the PR must say so plainly.
9. If follow-up work was discovered, it must be added as a new ledger item.
10. Reject opportunistic "while here" edits.

## Hard rejects

Reject or request changes if a PR:

- mixes crate collapse with runtime behavior changes
- claims GPU/server/QK256 production readiness without receipts
- silently enables GGUF minimal fallback
- returns simulated inference from production server endpoints
- maps requested backend identity to the wrong backend
- modifies unrelated CI while doing runtime work
- removes ignored-test justification strings
- claims a test passed without showing the command

## Review comment template

```md
## Review result

Decision: approve / request changes / comment only

Work item:
- `ITEM-ID`

Scope:
- Allowed paths respected: yes/no
- Forbidden paths touched: yes/no

Acceptance:
- [ ] ...
- [ ] ...

Verification:
- Commands run:
  - ...
- Missing verification:
  - ...

Follow-up ledger items needed:
- ...
```
