# P0 Tasks Quick Reference

**Generated**: 2025-10-23
**Context**: Post-PR #475 hardening tasks

---

## TL;DR

**3 P0 tasks** identified for post-PR #475 CI hardening:

1. **Build Script Hygiene** (2-3 hrs) - Fix `cargo:warning=` directives
2. **EnvGuard Rollout** (4-6 hrs) - Fix 45 unprotected env-mutating tests
3. **All-Features CI** (1-2 hrs) - Fix module path blocking PR #475

**All specs ready** in `/home/steven/code/Rust/BitNet-rs/docs/explanation/specs/`
**GitHub issues ready** in `/home/steven/code/Rust/BitNet-rs/GITHUB_ISSUES_P0_TASKS.md`

---

## Quick Commands

### Issue #1: Build Script Hygiene

**Fix**:
```bash
# Edit crates/bitnet-ggml-ffi/build.rs lines 8-18
# Replace eprintln!() with println!("cargo:warning=...")
```

**Verify**:
```bash
! grep -n "unwrap()" crates/*/build.rs
env -u HOME cargo build --features cpu -p bitnet-kernels
```

---

### Issue #2: EnvGuard Rollout

**Fix Priority 1**:
```bash
# Edit crates/bitnet-kernels/tests/device_features.rs
# Add #[serial(bitnet_env)] to 14 tests (lines 87-594)
# Change #[path = "../support/mod.rs"] to #[path = "support/mod.rs"] (line 83)
```

**Verify**:
```bash
cargo run -p xtask -- check-env-guards
RUST_TEST_THREADS=4 cargo test -p bitnet-kernels --test device_features
```

**Full List**: See `ENV_VAR_MUTATION_AUDIT_REPORT.md` (45 tests across 30 files)

---

### Issue #3: All-Features CI (BLOCKER)

**Fix**:
```bash
# Edit crates/bitnet-kernels/tests/device_features.rs line 83
-    #[path = "../support/mod.rs"]
+    #[path = "support/mod.rs"]
```

**Verify**:
```bash
cargo check --workspace --all-features
cargo clippy --workspace --all-features --all-targets -- -D warnings
cargo test --doc --workspace --all-features
```

---

## File Locations

### Specifications
- `/home/steven/code/Rust/BitNet-rs/docs/explanation/specs/SPEC-2025-002-build-script-hygiene-hardening.md`
- `/home/steven/code/Rust/BitNet-rs/docs/explanation/specs/SPEC-2025-003-envguard-serial-rollout.md`
- `/home/steven/code/Rust/BitNet-rs/docs/explanation/specs/SPEC-2025-004-all-features-ci-failure-investigation.md`

### GitHub Issues
- `/home/steven/code/Rust/BitNet-rs/GITHUB_ISSUES_P0_TASKS.md`

### Supporting Docs
- `/home/steven/code/Rust/BitNet-rs/ENV_VAR_MUTATION_AUDIT_REPORT.md` (45 test analysis)
- `/home/steven/code/Rust/BitNet-rs/PR_475_FINAL_SUCCESS_REPORT.md` (100% test pass)
- `/home/steven/code/Rust/BitNet-rs/P0_TASKS_SPEC_GENERATION_SUMMARY.md` (full summary)

---

## Priority Order

1. **Issue #3** (1-2 hrs) - Blocks PR #475 merge
2. **Issue #2** (4-6 hrs) - Eliminates CI flakiness
3. **Issue #1** (2-3 hrs) - Hardens build system

**Total**: 7-11 hours

---

## Acceptance Criteria Summary

### Issue #1: Build Script Hygiene
- [ ] No `unwrap()` in build scripts
- [ ] `cargo:warning=` directives visible
- [ ] Builds succeed without `$HOME`
- [ ] CI panics on missing markers

### Issue #2: EnvGuard Rollout
- [ ] 45/45 tests protected (`check-env-guards`)
- [ ] Parallel execution stable (10 iterations)
- [ ] No raw env mutations
- [ ] Documentation complete

### Issue #3: All-Features CI
- [ ] Compilation succeeds with `--all-features`
- [ ] Clippy clean with `--all-features`
- [ ] Tests compile with `--all-features`
- [ ] Doctest passes

---

## Impact

### Before
- Build warnings invisible in CI (bitnet-ggml-ffi)
- 45 env tests unprotected (8-12% CI flakiness)
- PR #475 blocked by all-features compilation error

### After
- 100% build warning visibility
- 0% env test flakiness
- PR #475 unblocked + all feature combinations validated

---

## Next Steps

1. Review specifications for completeness
2. Create GitHub issues (copy from `GITHUB_ISSUES_P0_TASKS.md`)
3. Assign to developers
4. Track progress via issue milestones
