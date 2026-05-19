# PR Queue Disposition Tracker Notes

This tracker note operationalizes `BITNET-SPEC-PR-QUEUE-DISPOSITION` for queue reviews.

## Reviewer checklist

1. Record one valid close reason from `policy/pr-dispositions.toml`.
2. Add required evidence link:
   - merged commit/PR,
   - duplicate PR,
   - landed successor,
   - historical report/ledger, or
   - content-audit note.
3. If future work remains, link successor PR or tracking issue before close.
4. Do not close for stale/restack/parent-closed states.

## Routing reminders

- stale stack -> restack/rebase.
- invalid closure -> reopen and continue repair.
- diagnostic with durable tooling -> keep open, port, or merge.
