# Codex: bitnet-rs Alignment Work

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

Default command baseline:

```bash
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets --no-default-features --features cpu -- -D warnings
cargo nextest run --locked --workspace --no-default-features --features cpu
```

Do not claim GPU, server inference, QK256 performance, or production readiness unless receipt-backed.
