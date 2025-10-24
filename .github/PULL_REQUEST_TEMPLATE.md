## Summary
<!-- 1-3 sentences: What changed and why? Link related issues with #123 -->

## Type
<!-- Check one -->
- [ ] feat - New feature
- [ ] fix - Bug fix
- [ ] docs - Documentation only
- [ ] test - Test changes
- [ ] refactor - Code refactoring
- [ ] perf - Performance improvement
- [ ] chore - Build/tooling changes

## Test Plan
<!-- Describe how you tested this change -->

## Breaking Changes
<!-- List any breaking API changes, migration steps, or mark "None" -->

## Pre-Submission Checklist

### Build & Format
- [ ] **MSRV 1.90.0** (Rust 2024 edition) - no newer features used
- [ ] **Format**: `cargo fmt --all` passes
- [ ] **Clippy**: `cargo clippy --workspace --all-targets --all-features -- -D warnings` passes

### Tests
- [ ] **CPU tests**: `cargo nextest run --workspace --no-default-features --features cpu --no-fail-fast` green
- [ ] **Doctests**: `cargo test --doc --workspace --no-default-features --features cpu` green
- [ ] **GPU tests** (if applicable): Tested with `--features gpu`

### Fixtures & EnvGuard
- [ ] **Fixtures unchanged** OR **updated with receipts** (updated `ci/fixtures/qk256/SHA256SUMS`)
- [ ] Fixture integrity tests pass: `cargo test -p bitnet-models --test fixture_integrity_tests`
- [ ] **Ignore hygiene**: All `#[ignore]` includes reason (`#[ignore = "reason"]` or comment)
- [ ] **EnvGuard**: No raw `std::env::set_var/remove_var`; tests use `EnvGuard + #[serial(bitnet_env)]`

### Supply Chain & Links
- [ ] **Supply chain**: `cargo deny check advisories bans licenses sources` clean
- [ ] **Links**: `lychee --config .lychee.toml "README.md" "docs/**/*.md"` clean OR `LINK_DEBT.md` updated

### Documentation
- [ ] **Docs updated** (if user-visible change)
- [ ] **CHANGELOG** updated (if user-visible change)

## Receipts / Evidence
<!-- Paste CI summaries, clippy output, test results, or benchmarks -->
