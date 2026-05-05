Use the in-repo tracker:

- `docs/tracking/bitnet-alignment/workstream-ledger.yaml`
- `docs/tracking/bitnet-alignment/status.md`
- `docs/tracking/bitnet-alignment/verification-gates.md`
- `.codex/bitnet-alignment.md`

Task:

1. Select the first `ready` work item with no unmet dependencies.
2. Implement only that item.
3. Stay inside `scope.allowed_paths`.
4. Do not touch `scope.forbidden_paths`.
5. Update the item state and `status.md`.
6. Run the verification commands listed for that item.
7. Open one focused PR.
