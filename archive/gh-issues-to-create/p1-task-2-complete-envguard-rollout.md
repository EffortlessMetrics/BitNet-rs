# Issue: Complete EnvGuard Rollout for Parallel Test Safety

## Context

BitNet.rs has extensive test infrastructure with ~1935+ tests, including many that mutate environment variables for deterministic testing (e.g., `BITNET_DETERMINISTIC`, `BITNET_STRICT_MODE`, `BITNET_GPU_FAKE`). Currently, ~21 test files still use unsafe `std::env::set_var`/`remove_var` operations without proper isolation, creating race conditions during parallel test execution.

Following PR #475, which established the EnvGuard pattern with 7 passing isolation tests, we need to complete the rollout to all env-mutating tests across the workspace.

**Affected Components:**
- `tests/` - 6 files needing migration to EnvGuard
- `tests-new/` - 7 files needing migration
- `xtask/` - 5 test files needing migration
- `crates/*/tests/` - Additional files requiring review
- `docs/development/test-suite.md` - Documentation updates

**Inference Pipeline Impact:**
- Testing infrastructure - Ensures deterministic inference validation
- GPU/CPU device selection - Safe testing of device fallback mechanisms
- Strict mode validation - Isolated testing of production safety enforcement

**Performance Implications:**
- EnvGuard overhead: < 1ms per test (negligible)
- Parallel test safety: Zero flaky failures from env races
- CI stability: Deterministic test execution with `--test-threads=4`

## User Story

As a test maintainer, I need all env-mutating tests to use proper isolation so that parallel test execution is safe and deterministic without race conditions.

## Acceptance Criteria

AC1: Migrate all 21 files with unsafe env operations to use `EnvGuard` + `#[serial(bitnet_env)]` pattern
AC2: Add mini-guide (< 200 lines) in `docs/development/test-suite.md` documenting EnvGuard usage with code examples
AC3: Zero unsafe env mutations outside `env_guard.rs` module (verified by CI check)
AC4: All env-mutating tests include `#[serial(bitnet_env)]` marker for process-level serialization
AC5: Add CI enforcement check that fails on unsafe env patterns (`unsafe.*set_var|unsafe.*remove_var`)
AC6: Verify no test regressions - all 1935+ tests still pass with parallel execution
AC7: No performance degradation - CI run time within ±5% of baseline (current: ~2-3 minutes)
AC8: Document both RAII and closure-based EnvGuard patterns in `tests/support/env_guard.rs` module docs

## Technical Implementation Notes

- **Affected crates**: `tests` (6 files), `tests-new` (7 files), `xtask` (5 files), `crates/*/tests` (remaining files)
- **Pipeline stages**: Test infrastructure - affects all inference pipeline validation stages
- **Performance considerations**:
  - EnvGuard overhead: < 1ms per test (negligible for ~72 env-mutating tests)
  - `#[serial(bitnet_env)]` only affects env-mutating tests; other 1935+ tests run in parallel
  - CI stability: Zero flaky test failures from environment variable races
- **Quantization requirements**: N/A (test infrastructure change only)
- **Cross-validation**: Ensures deterministic cross-validation tests via `BITNET_DETERMINISTIC=1 BITNET_SEED=42 RAYON_NUM_THREADS=1`
- **Feature flags**: No new feature flags required - test infrastructure only
- **GGUF compatibility**: N/A (test infrastructure change only)
- **Testing strategy**:
  - TDD with `// AC:ID` tags for each acceptance criterion
  - Parallel test execution validation: `cargo nextest run --workspace --test-threads=8`
  - Serial test execution validation: `cargo nextest run --workspace --test-threads=1 --profile ci`
  - CI enforcement: Automated check for unsafe env patterns
  - No test regressions: All 1935+ tests pass before/after migration

**Migration Pattern:**
```rust
// BEFORE (unsafe, no isolation)
#[test]
fn test_strict_mode_enabled() {
    unsafe { std::env::set_var("BITNET_STRICT_MODE", "1"); }
    let config = StrictModeConfig::from_env();
    assert!(config.enabled);
    // ❌ No cleanup - pollutes other tests!
}

// AFTER (safe, isolated)
use serial_test::serial;
use tests::helpers::env_guard::EnvGuard;

#[test]
#[serial(bitnet_env)]  // AC4: Process-level serialization
fn test_strict_mode_enabled() {
    let _guard = EnvGuard::new("BITNET_STRICT_MODE");
    _guard.set("1");

    let config = StrictModeConfig::from_env();
    assert!(config.enabled);
    // ✅ AC1: Guard drops here, restoring original value
}

// PREFERRED (closure-based, cleaner)
use serial_test::serial;
use temp_env::with_var;

#[test]
#[serial(bitnet_env)]
fn test_strict_mode_enabled() {
    with_var("BITNET_STRICT_MODE", Some("1"), || {
        let config = StrictModeConfig::from_env();
        assert!(config.enabled);
    });
    // ✅ AC8: Automatic restoration on scope exit
}
```

**Validation Commands:**
```bash
# AC3: Verify no unsafe env operations remain
! grep -r "unsafe.*set_var\|unsafe.*remove_var" --include="*.rs" \
  crates/ tests/ tests-new/ xtask/ | grep -v "env_guard.rs"

# AC6: Run full test suite with parallel execution
cargo nextest run --workspace --test-threads=4 --profile ci

# AC6: Verify serial tests still work
cargo nextest run --workspace --test-threads=1 --profile ci

# AC7: Verify no performance degradation (baseline: ~2-3 minutes)
time cargo nextest run --workspace --profile ci
```

**CI Enforcement (AC5):**
```yaml
# .github/workflows/ci.yml
- name: Verify EnvGuard rollout
  run: |
    # Check for unsafe env mutations outside EnvGuard
    if grep -r "unsafe.*set_var\|unsafe.*remove_var" \
       --include="*.rs" crates/ tests/ tests-new/ xtask/ | \
       grep -v "env_guard.rs"; then
      echo "❌ Found unsafe env mutations outside EnvGuard pattern"
      exit 1
    fi
    echo "✅ All env mutations use EnvGuard pattern"
```

**Estimate**: 2 hours

---

<!-- gates:start -->
| Gate | Status | Evidence |
|------|--------|----------|
| spec | ✅ pass | Feature spec created in docs/explanation/specs/SPEC-2025-001-next-steps-priority-implementation.md (Story 2) |
| format | pending | Code formatting with cargo fmt --all --check |
| clippy | pending | Linting with cargo clippy --workspace --all-targets --no-default-features --features cpu -- -D warnings |
| tests | pending | TDD scaffolding with cargo test --workspace --no-default-features --features cpu |
| build | pending | Build validation with cargo build --release --no-default-features --features cpu |
| features | pending | Feature smoke testing: cpu feature combo |
| benchmarks | pending | CI performance baseline validation (±5% threshold) |
| docs | pending | Documentation updates in docs/development/test-suite.md |
<!-- gates:end -->

<!-- hoplog:start -->
### Hop log
- Created feature spec: Story 2 in docs/explanation/specs/SPEC-2025-001-next-steps-priority-implementation.md
<!-- hoplog:end -->

<!-- decision:start -->
**State:** in-progress
**Why:** Feature spec created and validated, ready for implementation
**Next:** NEXT → implementation with TDD workflow (AC1-AC8 migration pattern)
<!-- decision:end -->
