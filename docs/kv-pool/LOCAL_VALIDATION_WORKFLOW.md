# Local Validation Workflow (CI Offline)

## Overview

With GitHub Actions CI disabled, this project uses `scripts/ci-local.sh` for local validation.
This ensures all changes meet quality gates before merge.

---

## Quick Reference

### Full Workspace Validation

```bash
# Run all workspace checks (default mode)
./scripts/ci-local.sh

# Same as:
./scripts/ci-local.sh workspace

# Or use Nix flake (reproducible, hermetic)
nix flake check .#workspace
```

**Checks performed:**
- ✅ Baseline build with `-Dwarnings`
- ✅ Workspace tests (`--lib`)
- ✅ Clippy (strict)
- ✅ Format check
- ✅ Documentation build
- ✅ MSRV (1.89.0)

**Duration:** ~5-10 minutes (with clean build)

---

### Focused bitnet-server Receipts Validation

```bash
# Run focused bitnet-server validation
./scripts/ci-local.sh bitnet-server-receipts

# Or use Nix flake (reproducible, hermetic)
nix flake check .#bitnet-server-receipts
```

**Checks performed:**
1. ✅ Baseline CPU check (`cpu` feature only)
2. ✅ Clippy (CPU only, strict)
3. ✅ Format check
4. ✅ Documentation build
5. ✅ MSRV (1.89.0)
6. ✅ Feature combo: `cpu,receipts`
7. ✅ Feature combo: `cpu,receipts,tuning`
8. ✅ Test: happy path (receipts enabled)
9. ✅ Test: guard path (receipts disabled)

**Duration:** ~2-4 minutes (incremental build)

**Use case:** Fast validation for `bitnet-server` changes without rebuilding entire workspace

---

## What "Working Local Validation" Means

In the context of BitNet.rs with CI offline, **local validation is working** when:

1. **Repeatable:** Commands are captured in a script, not run ad-hoc
2. **Documented:** Validation steps are explicitly listed
3. **Passing:** All checks pass on current branch
4. **Scoped:** Validation matches the change surface (workspace vs single crate)
5. **Traceable:** Results are captured in PR summaries

For `feat/kv-receipts-phase2`, this means:

- ✅ All 9 validation steps pass (see `bitnet-server-receipts` mode)
- ✅ Commands are in `scripts/ci-local.sh`
- ✅ Results are documented in `PR_525_KV_RECEIPTS_PHASE2_SUMMARY.md`
- ✅ Feature combinations are explicitly tested (`cpu,receipts,tuning`)

---

## Usage Patterns

### Before Creating a PR

```bash
# 1. Run focused validation on changed crate
./scripts/ci-local.sh bitnet-server-receipts

# 2. If passing, run full workspace validation
./scripts/ci-local.sh workspace

# 3. Document results in PR summary
```

### During Development (Fast Feedback)

```bash
# Quick format + clippy check (no full rebuild)
cargo fmt --all
cargo clippy -p bitnet-server --no-default-features --features cpu -- -D warnings

# Run specific test
cargo test -p bitnet-server --no-default-features --features "cpu,receipts,tuning" \
  -- emits_eviction_receipt_with_correct_payload
```

### After Fixing Validation Failures

```bash
# Re-run just the failed step
# Example: If clippy failed, fix and re-run:
cargo clippy -p bitnet-server --all-targets --no-default-features --features cpu -- -D warnings

# Then re-run full validation
./scripts/ci-local.sh bitnet-server-receipts
```

---

## Validation Results Format

Each validation run should produce:

1. **Console output:** All checks passing with `✅` markers
2. **PR summary:** Checklist of all validation steps
3. **Git status:** Clean working tree (or explicit list of modified files)

Example PR summary checklist:

```markdown
## Local Validation (CI Offline)

| Check                               | Status |
|-------------------------------------|--------|
| Baseline CPU build                  | ✅ Pass |
| Clippy (cpu only)                   | ✅ Pass |
| Formatting                          | ✅ Pass |
| Documentation                       | ✅ Pass |
| MSRV (1.89.0)                       | ✅ Pass |
| Feature: cpu,receipts               | ✅ Pass |
| Feature: cpu,receipts,tuning        | ✅ Pass |
| Test: happy path (receipts enabled) | ✅ Pass |
| Test: guard path (receipts off)     | ✅ Pass |
```

---

## Extending for New Crates

To add validation for a new crate:

1. **Add a new mode** to `scripts/ci-local.sh`:

```bash
elif [[ "${MODE}" == "my-new-crate" ]]; then
  echo "== my-new-crate: validation sequence =="

  # Baseline
  RUSTFLAGS="-Dwarnings" \
    cargo +stable check -p my-new-crate --locked --no-default-features --features cpu

  # Clippy
  cargo +stable clippy -p my-new-crate --all-targets --no-default-features --features cpu \
    -- -D warnings

  # Tests
  cargo +stable test -p my-new-crate --no-default-features --features cpu

  echo "✅ All my-new-crate checks passed."
```

2. **Update usage message** in the `else` block

3. **Document in PR summary** with checklist of validation steps

---

## Troubleshooting

### Script fails with "RUSTC_WRAPPER=: command not found"

This was fixed in the current version. Make sure you're using the latest `scripts/ci-local.sh`.

### Validation passes locally but fails in PR review

Check for:
- Uncommitted files (run `git status`)
- Feature flag mismatches
- MSRV issues (ensure `rustup show` includes `1.89.0`)

### Tests pass individually but fail in validation script

This indicates test isolation issues. Check for:
- Shared mutable state
- Environment variable dependencies
- Filesystem state assumptions

Use `#[serial(bitnet_env)]` for env-mutating tests (see `CLAUDE.md`).

---

## See Also

- `CLAUDE.md` – Project instructions and development workflow
- `NIX_FLAKE_USAGE.md` – Nix flake guide for reproducible environments
- `PR_525_KV_RECEIPTS_PHASE2_SUMMARY.md` – Example PR summary with validation checklist
- `scripts/ci-local.sh` – Local validation script
