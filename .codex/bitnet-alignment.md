# Codex: bitnet-rs Alignment Work

You are working on the bitnet-rs pre-publish alignment burndown.

Use `docs/tracking/bitnet-alignment/workstream-ledger.yaml` as the control plane:
pick the next ready item with no unmet dependencies, stay inside its allowed paths,
avoid forbidden paths, update the tracker, and report only verification commands
actually run. The goal is to make bitnet-rs smaller, stricter, greener, and more
honest by collapsing excess public crate seams into SRP modules, preserving
explicit feature gates, removing fake or ambiguous runtime paths, and requiring
receipt-backed proof for working claims. Keep the sequence disciplined: truth
boundary first, then crate inventory, then consolidation, then CPU runtime proof,
then server inference and GPU validation; when work grows beyond the item, add a
follow-up ledger entry instead of broadening the PR.

Primary tracker:

- `docs/tracking/bitnet-alignment/workstream-ledger.yaml`

Before starting:

1. Pick the first `ready` item with no unmet dependencies.
2. Keep the PR within `scope.allowed_paths`.
3. Do not touch `scope.forbidden_paths`.
4. If the task is too large, split it by adding follow-up ledger items.
5. Update `status.md`.
6. Run the verification gate listed on the item.
7. Report commands actually run.

Every work item may also update:

- `docs/tracking/bitnet-alignment/status.md`
- `docs/tracking/bitnet-alignment/workstream-ledger.yaml`

Use that implicit tracker exception only for item state, PR number, verification notes,
and follow-up items. Do not reshape unrelated tracker sections inside implementation PRs.

Default command baseline:

```bash
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets --no-default-features --features cpu -- -D warnings
cargo nextest run --locked --workspace --no-default-features --features cpu
```

Do not claim GPU, server inference, QK256 performance, or production readiness unless receipt-backed.
