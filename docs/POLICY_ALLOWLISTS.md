# BitNet-rs Policy Allowlists

A concise index of every policy ledger in `policy/`, who owns it, and
which `xtask` command checks it.

| File                                  | Purpose                                                | Checker                               | Owner          |
| ------------------------------------- | ------------------------------------------------------ | ------------------------------------- | -------------- |
| `policy/ci-lane-whitelist.toml`       | Inventory every CI lane with cost, trigger, owner      | `xtask ci-lane-whitelist check`       | release/ci     |
| `policy/ci-whitelist-exceptions.toml` | Receipted exceptions to whitelist defaults             | (same)                                | release/ci     |
| `policy/clippy-lints.toml`            | Future Clippy lints scheduled by MSRV                  | `xtask check-lint-policy` (PR 08)     | core/rust      |
| `policy/clippy-exceptions.toml`       | Receipted `#[expect(clippy::...)]` exceptions          | `xtask check-clippy-exceptions`       | core/rust      |
| `policy/clippy-debt.toml`             | Crate/path-scoped Clippy debt blocking promotion       | `xtask check-clippy-exceptions`       | core/rust      |
| `policy/no-panic-allowlist.toml`      | Receipted panic-family exceptions                      | `xtask check-no-panic-family`         | testing/policy |
| `policy/non-rust-allowlist.toml`      | Allowlisted non-Rust files (with owner / reason)       | `xtask check-file-policy`             | release/ci     |
| `policy/ripr-suppressions.toml`       | Suppressed `ripr` findings (PR 13)                     | (advisory)                            | testing/oracle |

## Receipt expiry

Every entry in every ledger has an `expires` field. Expired entries
fail the corresponding policy gate. The intent is that nothing is a
permanent allow — every exception either gets renewed (with a real
review) or removed.

## Adding a receipt

1. Try to remove the issue. The first option for every "I want an
   exception" is "is the underlying code wrong?".
2. If the exception is genuinely temporary, add a TOML entry with:
   * a stable `id`
   * a real `owner` (team / area, not an individual)
   * a real `reason` (what is acceptable today, not just "TODO")
   * a real `expires` date (typical: 60–90 days; never longer than
     the surrounding rollout phase)
3. Run the corresponding `xtask check-...` command locally to confirm
   the receipt resolves the finding.

## Aggregated report

`xtask policy-report` runs every checker in sequence and writes:

```
target/bitnet/reports/policy-report.md
target/bitnet/reports/ci-lane-whitelist.json
target/bitnet/reports/lint-inheritance.json
target/bitnet/reports/file-policy.json
target/bitnet/reports/no-panic.json
target/bitnet/reports/clippy-exceptions.json
target/bitnet/reports/no-panic-proposed-allowlist.toml   # advisory
```

The aggregator never fails the build by itself; each individual
checker's `--fail-on-error` flag is what gates merges.
