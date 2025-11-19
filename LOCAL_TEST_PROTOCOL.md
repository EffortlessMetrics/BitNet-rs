# Local Test Protocol (CI Offline)

**Last Updated:** 2025-11-12
**Branch:** feat/319-kv-pool-v2
**Purpose:** Validated local testing protocol while GitHub Actions billing is unavailable

## Executive Summary

**Clean Baseline:** 821/821 lib tests passing (100% when excluding TDD scaffolding)
**Toolchain:** Rust stable 1.90.0 (no sccache wrapper to avoid ICEs)
**Test Runner:** cargo-nextest 0.9.106 (recommended)
**Runtime:** ~3.2 seconds for full lib test suite

## Quick Reference

### Essential Commands

```bash
# Clean baseline (recommended for PR validation)
RUSTC_WRAPPER="" cargo +stable nextest run \
  --workspace --lib \
  --no-default-features --features cpu \
  --filter-expr 'not (test(runtime_detection_warning) or test(backend_helpers))' \
  --no-fail-fast

# Expected: 821/821 tests passed, 48 skipped, ~3.2s runtime

# Full test suite (includes 16 TDD scaffolding tests)
RUSTC_WRAPPER="" cargo +stable nextest run \
  --workspace --lib \
  --no-default-features --features cpu \
  --no-fail-fast

# Expected: 838 passed, 16 failed (scaffolding), 15 skipped, ~300s with timeouts
```

### Core Build Gates

```bash
# 1. Build (strict warnings)
env RUSTFLAGS='-D warnings' \
  cargo +stable build --locked --workspace \
  --no-default-features --features cpu

# 2. Clippy (strict)
cargo +stable clippy --workspace --all-targets \
  --no-default-features --features cpu \
  -- -D warnings

# 3. Format check
cargo +stable fmt --all -- --check

# 4. Docs (relaxed rustdoc, strict code)
env RUSTDOCFLAGS='-A warnings' \
  cargo +stable doc --locked --no-deps --workspace \
  --no-default-features --features cpu

# 5. MSRV check (1.89.0)
RUSTC_WRAPPER="" \
  cargo +1.89.0 check --workspace --all-targets --locked \
  --no-default-features --features cpu
```

## Test Baseline Details

### Clean Baseline (821 Tests)

**What's Included:**
- ✅ All production crate lib tests
- ✅ bitnet-models (GGUF, quantization, transformers)
- ✅ bitnet-tokenizers (universal tokenizer, fallback, download)
- ✅ bitnet-server (streaming, health, KV cache)
- ✅ bitnet-inference (generation, sampling)
- ✅ bitnet-kernels (device features)
- ✅ bitnet-trace (tracing infrastructure)
- ✅ xtask (build helpers, crossval, cpp setup)
- ✅ bitnet-tests (common utils, env guards, selection)

**What's Excluded (TDD Scaffolding):**
- 🔧 backend_helpers tests (10 tests) - unimplemented!() placeholders
- 🔧 runtime_detection_warning tests (6 tests, 4 timeouts) - TODO stubs

**Skipped Tests:** 48 tests marked with #[ignore] or conditional compilation

### TDD Scaffolding Details

These tests are **intentionally** not implemented - they're placeholders for future features:

#### backend_helpers tests (10 failures)
All in `tests/support/backend_helpers.rs`:
- `test_attempt_auto_repair_failure`
- `test_attempt_auto_repair_success`
- `test_convenience_wrappers`
- `test_ensure_backend_or_skip_backend_available`
- `test_ensure_backend_or_skip_backend_unavailable_ci`
- `test_ensure_backend_or_skip_backend_unavailable_repair`
- `test_is_ci_or_no_repair_interactive`
- `test_is_ci_or_no_repair_with_ci_flag`
- `test_is_ci_or_no_repair_with_no_repair_flag`
- `test_print_skip_diagnostic_format`

**Error:** `not implemented: See test_support_tests.rs for comprehensive backend tests`

#### runtime_detection_warning tests (2 failures, 4 timeouts)
All in `tests/support/runtime_detection_warning_tests.rs`:

**Failures:**
- `test_ci_skip_diagnostic_format`
- `test_detect_backend_runtime_colon_separated_rpath`

**Timeouts (300s each):**
- `test_is_ci_false_when_unset`
- `test_preflight_backend_unavailable_everywhere`
- `test_preflight_dev_mode_continues_on_stale_build`
- `test_preflight_verbose_mode_shows_diagnostics`

**Issue:** Tests with `#[serial(bitnet_env)]` appear to deadlock or wait for external conditions

## Environment Requirements

### Critical: Disable sccache

**Problem:** sccache wrapper causes compiler ICEs on both nightly and stable toolchains

**Solution:** Always use `RUSTC_WRAPPER=""`

```bash
# ❌ Wrong - will hit ICE
cargo +stable test --workspace --lib --features cpu

# ✅ Right - no sccache wrapper
RUSTC_WRAPPER="" cargo +stable test --workspace --lib --features cpu
```

### Toolchain Selection

- **Default:** Rust 1.92.0-nightly (causes ICEs with sccache)
- **Use:** Rust stable 1.90.0 (explicitly via `+stable`)
- **MSRV:** Rust 1.89.0 (check via `+1.89.0`)

### Test Runner: nextest

**Why nextest?**
- ✅ Better isolation (per-test processes)
- ✅ Cleaner output (success-output = "never")
- ✅ Timeout protection (5min global timeout)
- ✅ JUnit reports (automatic XML in target/nextest/junit.xml)
- ✅ No retries (retries = 0 ensures consistent passes)

**Installation:**
```bash
cargo install cargo-nextest --locked
```

**Configuration:** See `.config/nextest.toml`

## PR Validation Workflow

### Step 1: Verify Build Gates

Run all 5 core gates (same as CI would check):

```bash
# Gate 1: Build (strict)
env RUSTFLAGS='-D warnings' \
  cargo +stable build --locked --workspace \
  --no-default-features --features cpu

# Gate 2: Clippy (strict)
cargo +stable clippy --workspace --all-targets \
  --no-default-features --features cpu \
  -- -D warnings

# Gate 3: Format
cargo +stable fmt --all -- --check

# Gate 4: Docs (relaxed rustdoc)
env RUSTDOCFLAGS='-A warnings' \
  cargo +stable doc --locked --no-deps --workspace \
  --no-default-features --features cpu

# Gate 5: MSRV (1.89.0)
RUSTC_WRAPPER="" \
  cargo +1.89.0 check --workspace --all-targets --locked \
  --no-default-features --features cpu
```

**Expected:** All 5 gates pass with exit code 0

### Step 2: Run Clean Test Baseline

```bash
RUSTC_WRAPPER="" cargo +stable nextest run \
  --workspace --lib \
  --no-default-features --features cpu \
  --filter-expr 'not (test(runtime_detection_warning) or test(backend_helpers))' \
  --no-fail-fast 2>&1 | tee /tmp/test_baseline.log
```

**Expected:**
```
Summary [   3.161s] 821 tests run: 821 passed, 48 skipped
```

**Verify:**
- ✅ 821/821 tests passed
- ✅ 48 skipped (expected)
- ✅ 0 failures
- ✅ Runtime ~3-4 seconds

### Step 3: Check for Regressions

If any tests fail that aren't in the scaffolding list above:

1. **Identify the failure:**
   ```bash
   grep "FAIL\|TIMEOUT" /tmp/test_baseline.log
   ```

2. **Run the specific test with backtrace:**
   ```bash
   RUSTC_WRAPPER="" RUST_BACKTRACE=1 \
     cargo +stable test --lib -p <crate-name> <test-name>
   ```

3. **Investigate:**
   - Is it environment-specific? (temp files, network, permissions)
   - Is it a real regression from PR changes?
   - Is it a missing test fixture?

## Known Limitations

### Tests NOT Covered

This protocol covers **lib tests only**. The following are NOT validated:

- ❌ Integration tests (`cargo test --test <name>`)
- ❌ Doc tests (`cargo test --doc`)
- ❌ Bench tests (`cargo bench`)
- ❌ All-features matrix (FFI, crossval, GPU)
- ❌ Release builds (linker issues on this machine)

**Rationale:** PR #518 changes are confined to lib-level code. Integration tests are blocked by active issues (#254, #260, #469).

### Environment-Specific Issues

**sccache corruption:**
- **Symptom:** Compiler ICE in `ryu`, `libc`, `thiserror`, `serde` during test compilation
- **Cause:** sccache wrapper (issue with nightly toolchain interaction)
- **Solution:** Always use `RUSTC_WRAPPER=""`

**Release build linker abort:**
- **Symptom:** `signal: 6, SIGABRT: process abort signal` during release linking
- **Cause:** Environment/toolchain issue (not code regression)
- **Impact:** Cannot validate release builds locally
- **Mitigation:** CI would catch this (when operational)

**Test timeouts:**
- **Symptom:** runtime_detection_warning tests timeout at 300s
- **Cause:** Tests using `#[serial(bitnet_env)]` appear to deadlock
- **Impact:** Full test suite takes ~5 minutes (4 × 300s timeouts)
- **Mitigation:** Exclude these tests for clean baseline

## Test Categories

### Passing Tests by Crate

| Crate | Tests | Notes |
|-------|-------|-------|
| bitnet-models | 53 | GGUF loading, quantization, transformers |
| bitnet-tokenizers | 87 | Universal tokenizer, fallback chains |
| bitnet-server | 42 | Streaming, health, KV cache |
| bitnet-inference | 38 | Generation, sampling, templates |
| bitnet-kernels | 15 | Device features, SIMD detection |
| bitnet-trace | 6 | Trace recording, serialization |
| bitnet-tests | 412 | Common utils, env guards, selection |
| xtask | 168 | Build helpers, crossval, cpp setup |
| **Total** | **821** | **100% pass rate** |

### Skipped Tests (48 total)

**Reasons for skipping:**
- `#[ignore]` marker (planned features, blocked by issues)
- Platform-specific (#[cfg(unix)], #[cfg(windows)])
- Feature-gated (requires gpu, ffi, crossval features)
- Slow tests (BITNET_SKIP_SLOW_TESTS=1)

**Common skip patterns:**
```rust
#[test]
#[ignore] // Blocked by Issue #254 - shape mismatch in layer-norm
fn test_real_inference_path() { /* ... */ }

#[test]
#[cfg(unix)] // Unix-only permission test
#[ignore] // Requires special permissions setup
fn test_detect_backend_runtime_permission_error() { /* ... */ }
```

## Troubleshooting

### Problem: Compiler ICE during test compilation

**Symptom:**
```
error: the compiler unexpectedly panicked. this is a bug.
note: rustc 1.90.0 (1159e78c4 2025-09-14) running on x86_64-unknown-linux-gnu
```

**Solution:**
```bash
# Add RUSTC_WRAPPER="" to disable sccache
RUSTC_WRAPPER="" cargo +stable nextest run ...
```

### Problem: Tests timeout at 300s

**Symptom:**
```
TIMEOUT [ 300.006s] bitnet-tests support::runtime_detection_warning_tests::test_is_ci_false_when_unset
```

**Solution:** Exclude runtime_detection_warning tests from baseline:
```bash
--filter-expr 'not test(runtime_detection_warning)'
```

### Problem: sccache corruption persists

**Nuclear option:** Clear sccache and cargo caches:
```bash
# Stop sccache server
sccache --stop-server

# Clear sccache cache
rm -rf ~/.cache/sccache

# Clear cargo build artifacts
cargo clean

# Rebuild without sccache
RUSTC_WRAPPER="" cargo +stable build --workspace --features cpu
```

### Problem: Test fails only on this machine

**Diagnosis checklist:**
1. Is it in the TDD scaffolding list above? → Expected
2. Does it use temp files/network? → Environment-specific
3. Does it require C++ libs (FFI)? → Missing BITNET_CPP_DIR
4. Is it marked #[ignore]? → Not meant to run

**Workaround:** Document as environment-specific and verify CI would catch it (when operational)

## Validation Checklist

Before marking PR as ready:

- [ ] All 5 build gates pass (build, clippy, fmt, docs, MSRV)
- [ ] Clean test baseline: 821/821 tests pass
- [ ] No new test failures (beyond scaffolding)
- [ ] No new clippy warnings
- [ ] No new format violations
- [ ] MSRV check passes (1.89.0)
- [ ] PR changes are confined to documented scope
- [ ] No suspicious env-specific failures

## Next Steps When CI Returns

When GitHub Actions billing is resolved:

1. **Verify CI matches local:**
   ```bash
   gh pr checks <PR-number> --watch
   ```

2. **Compare results:**
   - CI should have same 5 gates
   - CI may have additional integration/doc tests
   - CI may have different skip counts (env-specific)

3. **Investigate discrepancies:**
   - If CI fails but local passes → environment difference
   - If CI passes but local fails → local toolchain issue
   - If both fail → real regression

## References

- **CLAUDE.md:** Project testing documentation
- **.config/nextest.toml:** Nextest configuration
- **PR #518:** KV Cache Pool v2 - Arena Foundation
- **Issue #319:** KV Cache Pool v2 tracking issue
