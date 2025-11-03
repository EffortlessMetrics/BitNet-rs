# GitHub Issues for P0 Tasks - BitNet.rs Post-PR-475

## Issue 1: Real GGUF Fixtures for QK256 and BitNet-32 Testing

### Problem Statement

BitNet.rs currently generates GGUF test fixtures in-memory during test execution, which adds ~50-100ms overhead per test run and creates potential for CI/CD instability. We need persistent, disk-based GGUF v3 fixtures stored in version control to ensure stable, reproducible testing of QK256 and BitNet32F16 quantization formats.

The current approach in `qk256_dual_flavor_tests.rs` dynamically generates fixtures using helper functions, which:
- Adds unnecessary runtime overhead (150ms across 3 key tests)
- Creates potential for cross-platform variance in generated fixtures
- Lacks explicit validation of GGUF v3 alignment and metadata requirements
- Makes test debugging harder (no persistent artifacts to inspect)

### Affected Files

- `ci/fixtures/qk256/` - **NEW DIRECTORY** - Persistent GGUF fixtures storage
- `crates/bitnet-models/tests/qk256_dual_flavor_tests.rs:1-250` - Migrate from in-memory generation to disk-based loading
- `crates/bitnet-models/tests/helpers/qk256_fixtures.rs:1-150` - Keep for regeneration tooling only
- `ci/fixtures/qk256/README.md` - **NEW FILE** - Fixture documentation and regeneration guide
- `scripts/regenerate-fixtures.sh` - **NEW FILE** - Reproducible fixture generation script

### Acceptance Criteria

#### AC1: Fixture Generation and Storage
- [ ] Generate 3 persistent GGUF v3 fixtures with deterministic seeds:
  - `qk256_4x256_seed42.gguf` - Single-block QK256 (4×256 = 256 bytes quantized)
  - `qk256_3x300_seed44.gguf` - Multi-block QK256 with tail (3×300 = 384 bytes)
  - `bitnet32_2x64_seed43.gguf` - BitNet32F16 format (2×64 = 40 bytes with inline F16 scales)
- [ ] Store fixtures in `ci/fixtures/qk256/` directory
- [ ] Each fixture must be < 50KB in size
- [ ] Add SHA256 checksums in `ci/fixtures/qk256/checksums.txt`

#### AC2: GGUF v3 Compliance
- [ ] All fixtures pass GGUF v3 validation:
  - Magic: `GGUF` (0x46554747)
  - Version: 3 (little-endian u32)
  - 32-byte alignment for tensor data section
  - Required metadata KV pairs: `general.name`, `tokenizer.ggml.tokens`, `bitnet-b1.58.embedding_length`
  - Dual tensor structure: `tok_embeddings.weight` (I2_S) + `output.weight` (F16)

#### AC3: Test Migration
- [ ] Update `qk256_dual_flavor_tests.rs` to load fixtures from `ci/fixtures/qk256/` instead of in-memory generation
- [ ] Add `test_all_fixtures_exist_and_valid()` test to verify fixture presence and validity
- [ ] All 12 existing QK256 tests pass with disk-based fixtures
- [ ] Tests use relative path resolution via `env!("CARGO_MANIFEST_DIR")`

#### AC4: Validation Integration
- [ ] All fixtures pass `cargo run -p bitnet-cli --features cpu,full-cli -- compat-check <fixture.gguf>`
- [ ] Alignment validation tests in `helpers/alignment_validator.rs` pass against disk fixtures
- [ ] Strict mode validation: `BITNET_STRICT_MODE=1 cargo run -p bitnet-cli -- inspect --ln-stats --gate auto <fixture.gguf>` succeeds

#### AC5: Documentation and Regeneration
- [ ] Create `ci/fixtures/qk256/README.md` documenting:
  - Fixture format specifications
  - Regeneration procedure
  - SHA256 checksum verification
  - Cross-platform testing requirements
- [ ] Create `scripts/regenerate-fixtures.sh` for reproducible fixture updates
- [ ] Update `crates/bitnet-models/tests/helpers/qk256_fixtures.rs` module docs to indicate "for regeneration only"

#### AC6: CI Integration
- [ ] Add fixture existence check in `.github/workflows/ci.yml`
- [ ] CI validates SHA256 checksums before running tests
- [ ] Cross-platform validation: Fixtures load identically on x86_64, ARM64, WASM

### Verification Steps

```bash
# 1. Generate fixtures (one-time script)
cargo run -p bitnet-models --test qk256_dual_flavor_tests --features fixtures -- \
  test_dump_fixture_for_debug --nocapture

# 2. Copy to ci/fixtures/ (manually or via script)
./scripts/regenerate-fixtures.sh

# 3. Verify checksum integrity
sha256sum -c ci/fixtures/qk256/checksums.txt

# 4. Validate with bitnet-cli
for f in ci/fixtures/qk256/*.gguf; do
  cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- \
    compat-check "$f"
done

# 5. Run fixture-based tests
cargo test -p bitnet-models --features cpu,fixtures test_qk256_detection_by_size

# 6. CI integration test with nextest
cargo nextest run -p bitnet-models --features cpu,fixtures --profile ci

# 7. Cross-platform validation (requires targets)
cargo test -p bitnet-models --features cpu,fixtures --target x86_64-unknown-linux-gnu
cargo test -p bitnet-models --features cpu,fixtures --target aarch64-unknown-linux-gnu

# 8. Performance verification (should be ~150ms faster than in-memory generation)
time cargo nextest run -p bitnet-models --features cpu,fixtures
```

### Estimated Effort

**2-3 hours** (including testing and documentation)

### Related Issues/PRs

- PR #475 - Comprehensive integration with QK256 AVX2 and fixtures feature (merged to main)
- Issue #439 - Feature gate consistency (resolved in PR #475)
- Related spec: `docs/explanation/specs/SPEC-2025-001-next-steps-priority-implementation.md` (Story 1)

### Technical Implementation Notes

**Quantization Constraints:**
- QK256 block size: 256 elements → 64 bytes packed (2-bit quantization)
- BitNet32F16 block size: 32 elements → 10 bytes (8 bytes packed + 2 bytes F16 scale)
- GGUF v3 alignment: 32-byte alignment for tensor data section
- Metadata requirements: 8 required KV pairs for GGUF v3 compliance

**Feature Flags:**
- Uses existing `fixtures` feature flag
- Tests run with `--features cpu,fixtures`
- Conditional compilation: `#[cfg_attr(not(feature = "fixtures"), ignore)]` unchanged

**Performance Requirements:**
- CI speedup: Eliminate ~150ms per test run (3 tests × ~50ms generation overhead)
- Determinism: Fixtures must produce identical results across platforms
- Size constraints: Each fixture < 50KB (typical: 1-3KB)

**Risk Mitigation:**
- Git LFS consideration if fixtures grow > 1MB (current: all < 50KB)
- SHA256 checksums prevent corruption
- Regeneration script ensures reproducibility
- Cross-platform validation prevents platform-specific bugs

---

## Issue 2: Complete EnvGuard Rollout Across Test Suite

### Problem Statement

BitNet.rs test suite contains ~21 files with unsafe environment variable mutations that can cause race conditions during parallel test execution. These unsafe mutations use `unsafe { std::env::set_var() }` without proper cleanup, polluting the environment for other tests running in parallel.

The existing `EnvGuard` pattern provides safe environment isolation with automatic restoration, but rollout is incomplete. Key tests for deterministic inference (`BITNET_DETERMINISTIC`, `BITNET_SEED`), strict mode validation (`BITNET_STRICT_MODE`), and GPU override (`BITNET_GPU_FAKE`) are at risk of flaky failures due to env races.

### Affected Files

- `tests/` - **6 files** needing EnvGuard migration
- `tests-new/` - **7 files** needing EnvGuard migration
- `xtask/` - **5 test files** needing EnvGuard migration
- `crates/*/tests/` - **~3 remaining files** needing review
- `docs/development/test-suite.md:250-350` - Add EnvGuard mini-guide section
- `.github/workflows/ci.yml:180-200` - Add unsafe env pattern detection check
- `tests/support/env_guard.rs:1-50` - Update module docs with rollout status

### Acceptance Criteria

#### AC1: Complete Migration to EnvGuard Pattern
- [ ] All 21 files with unsafe env operations migrated to use `EnvGuard` or `temp_env::with_var`
- [ ] Zero remaining instances of `unsafe { std::env::set_var() }` or `unsafe { std::env::remove_var() }` outside `env_guard.rs`
- [ ] All env-mutating tests include `#[serial(bitnet_env)]` marker for process-level serialization

#### AC2: Pattern Standardization
- [ ] Use RAII-based pattern for simple tests:
  ```rust
  let _guard = EnvGuard::new("BITNET_STRICT_MODE");
  _guard.set("1");
  ```
- [ ] Use closure-based pattern for complex tests:
  ```rust
  with_var("BITNET_DETERMINISTIC", Some("1"), || {
      // test code
  });
  ```
- [ ] All test files include proper imports:
  ```rust
  use serial_test::serial;
  use tests::helpers::env_guard::EnvGuard;  // or temp_env::with_var
  ```

#### AC3: Documentation and Mini-Guide
- [ ] Add EnvGuard mini-guide section to `docs/development/test-suite.md` (< 200 lines)
- [ ] Mini-guide includes:
  - Problem explanation (race conditions in parallel tests)
  - RAII pattern example with `EnvGuard`
  - Closure-based pattern example with `temp_env::with_var`
  - When to use `#[serial(bitnet_env)]` marker
  - Common pitfalls and best practices
- [ ] Update `tests/support/env_guard.rs` module docs with rollout completion status

#### AC4: CI Enforcement
- [ ] Add CI check in `.github/workflows/ci.yml` to detect unsafe env patterns:
  ```bash
  if grep -r "unsafe.*set_var\|unsafe.*remove_var" \
     --include="*.rs" crates/ tests/ tests-new/ xtask/ | \
     grep -v "env_guard.rs"; then
    echo "❌ Found unsafe env mutations outside EnvGuard pattern"
    exit 1
  fi
  ```
- [ ] CI check runs before test execution
- [ ] Failure provides clear error message with violating files

#### AC5: Test Suite Stability
- [ ] All 1935+ tests pass with parallel execution: `cargo nextest run --workspace --test-threads=4`
- [ ] Zero flaky test failures from env races in 10 consecutive CI runs
- [ ] EnvGuard overhead < 1ms per test (negligible performance impact)

#### AC6: Validation and Coverage
- [ ] All env-mutating test categories covered:
  - Deterministic testing: `BITNET_DETERMINISTIC`, `BITNET_SEED`, `RAYON_NUM_THREADS`
  - Validation flags: `BITNET_STRICT_MODE`, `BITNET_VALIDATION_GATE`
  - GPU overrides: `BITNET_GPU_FAKE`
  - Test infrastructure: `BITNET_SKIP_SLOW_TESTS`, `BITNET_RUN_IGNORED_TESTS`
- [ ] Manual review checklist completed for all 21 target files

### Verification Steps

```bash
# 1. Verify no unsafe env operations remain (excluding env_guard.rs)
! grep -r "unsafe.*set_var\|unsafe.*remove_var" --include="*.rs" \
  crates/ tests/ tests-new/ xtask/ | grep -v "env_guard.rs"

# 2. Verify all env-mutating tests have #[serial(bitnet_env)] marker
# (Manual review - automated check difficult due to test macro variations)
grep -r "EnvGuard::new\|with_var" --include="*.rs" crates/ tests/ tests-new/ xtask/ | \
  while read -r line; do
    file=$(echo "$line" | cut -d: -f1)
    grep -B5 "EnvGuard::new\|with_var" "$file" | grep "#\[serial(bitnet_env)\]" || \
      echo "⚠️  Missing #[serial(bitnet_env)] in $file"
  done

# 3. Run full test suite with parallel execution (detect race conditions)
cargo nextest run --workspace --test-threads=8 --no-default-features --features cpu

# 4. Run serial execution for comparison (should have identical results)
cargo nextest run --workspace --test-threads=1 --no-default-features --features cpu

# 5. CI stability test (10 consecutive runs)
for i in {1..10}; do
  echo "=== CI Run $i/10 ==="
  cargo nextest run --workspace --profile ci || exit 1
done

# 6. Verify EnvGuard tests still pass (7 existing tests)
cargo test -p tests --test env_guard_tests -- --nocapture

# 7. Performance benchmark (should be within ±5% of baseline)
hyperfine --warmup 2 --runs 5 \
  'cargo nextest run --workspace --profile ci' \
  --export-markdown /tmp/envguard_perf.md
```

### Estimated Effort

**2 hours** (migration pattern is well-established, mostly mechanical changes)

### Related Issues/PRs

- PR #475 - Introduced EnvGuard infrastructure with 7 tests (merged to main)
- Issue #439 - Feature gate consistency affecting GPU override tests (resolved)
- Related spec: `docs/explanation/specs/SPEC-2025-001-next-steps-priority-implementation.md` (Story 2)

### Technical Implementation Notes

**Affected Environment Variables:**
- `BITNET_DETERMINISTIC` - Enables deterministic inference for reproducibility
- `BITNET_SEED` - Sets RNG seed for deterministic tests
- `RAYON_NUM_THREADS` - Controls parallelism for deterministic execution
- `BITNET_STRICT_MODE` - Enables strict validation (fail on warnings)
- `BITNET_VALIDATION_GATE` - Validation mode selection (none/auto/policy)
- `BITNET_GPU_FAKE` - Override GPU detection for testing (cuda/none)
- `BITNET_SKIP_SLOW_TESTS` - Skip slow tests (QK256 scalar kernels)
- `BITNET_RUN_IGNORED_TESTS` - Include ignored tests in test run

**Migration Priority Order:**
1. **High Priority (10 files)**: Tests for deterministic inference and strict mode
2. **Medium Priority (7 files)**: GPU override and validation tests
3. **Low Priority (4 files)**: Test infrastructure and utility tests

**Pattern Preference:**
- **Prefer `temp_env::with_var`** for new tests (cleaner closure-based idiom)
- **Keep `EnvGuard`** for existing tests where refactoring is complex
- **Always use `#[serial(bitnet_env)]`** for process-level serialization

**Risk Mitigation:**
- EnvGuard overhead < 1ms per test (negligible)
- `#[serial(bitnet_env)]` only serializes env-mutating tests (~72 tests)
- Other tests (1935+ total) continue to run in parallel
- Manual review checklist ensures no missed migrations

---

## Issue 3: FFI Hygiene - Zero Warnings in bitnet-ggml-ffi Build

### Problem Statement

The `bitnet-ggml-ffi` crate currently produces compiler warnings during build due to vendored GGML C++ code. These warnings pollute CI build output and make it difficult to detect regressions in BitNet.rs shim code. The vendored GGML headers should use `-isystem` flag (GCC/Clang) or `/external:I` (MSVC) to suppress third-party warnings while keeping warnings visible for local shim code.

Current state:
- Vendored GGML code in `csrc/ggml/` produces warnings (sign-compare, unused-parameter, unused-function)
- `-isystem` flag format in `build.rs` may not work correctly across all platforms
- No CI enforcement to prevent new warnings from creeping in
- Shim code warnings (`ggml_quants_shim.c`, `ggml_consts.c`) should remain visible

### Affected Files

- `crates/bitnet-ggml-ffi/build.rs:32-50` - Fix `-isystem` flag handling with platform detection
- `crates/bitnet-ggml-ffi/csrc/VENDORED_GGML_COMMIT` - Add CI enforcement for tracked commit
- `.github/workflows/ci.yml:150-180` - Add zero-warning check for FFI build
- `xtask/tests/ffi_build_hygiene_test.rs` - **NEW FILE** - Regression test for build hygiene
- `crates/bitnet-ggml-ffi/README.md:80-120` - Document `-isystem` rationale and platform handling

### Acceptance Criteria

#### AC1: Platform-Aware Build System
- [ ] Implement platform-aware `-isystem` flag handling in `build.rs`:
  - Detect compiler type: `build.get_compiler().is_like_msvc()`
  - Use `/external:I` for MSVC (Visual Studio 2019+)
  - Use `-isystem` for GCC/Clang (Linux, macOS)
- [ ] Add `/external:W0` flag for MSVC to suppress all external warnings
- [ ] Keep local includes (`csrc/`) using `-I` flag (warnings visible)

#### AC2: Zero-Warning Builds
- [ ] Clean build produces zero compiler warnings:
  ```bash
  cargo clean -p bitnet-ggml-ffi
  cargo build -p bitnet-ggml-ffi --no-default-features --features iq2s-ffi 2>&1 | \
    grep -i "warning" && exit 1 || echo "✅ Zero warnings"
  ```
- [ ] Multi-compiler validation (if available):
  - `CC=gcc cargo build -p bitnet-ggml-ffi --features iq2s-ffi` - zero warnings
  - `CC=clang cargo build -p bitnet-ggml-ffi --features iq2s-ffi` - zero warnings
  - MSVC build (Windows CI) - zero warnings

#### AC3: Vendored Commit Tracking
- [ ] Add CI enforcement in `build.rs`:
  ```rust
  if std::env::var("CI").is_ok() && commit == "unknown" {
      panic!("VENDORED_GGML_COMMIT is 'unknown' in CI. Run: cargo xtask vendor-ggml --commit <sha>");
  }
  ```
- [ ] `csrc/VENDORED_GGML_COMMIT` file exists and contains valid commit SHA
- [ ] `build.rs` reads commit and sets `BITNET_GGML_COMMIT` env var

#### AC4: CI Enforcement
- [ ] Add zero-warning check in `.github/workflows/ci.yml`:
  ```yaml
  - name: Build FFI crate with zero warnings
    run: |
      cargo clean -p bitnet-ggml-ffi
      BUILD_OUTPUT=$(cargo build -p bitnet-ggml-ffi --no-default-features --features iq2s-ffi 2>&1)

      if echo "$BUILD_OUTPUT" | grep -i "warning:"; then
        echo "❌ FFI build has warnings:"
        echo "$BUILD_OUTPUT" | grep -i "warning:"
        exit 1
      fi

      echo "✅ FFI build clean (zero warnings)"
  ```
- [ ] CI check runs on every PR targeting main branch
- [ ] Failure provides clear error message with warning details

#### AC5: Regression Test
- [ ] Create `xtask/tests/ffi_build_hygiene_test.rs` with automated build validation:
  ```rust
  #[test]
  #[cfg(feature = "iq2s-ffi")]
  fn test_ffi_build_zero_warnings() {
      // Clean build and capture output
      let output = Command::new("cargo")
          .args(&["build", "-p", "bitnet-ggml-ffi", "--features", "iq2s-ffi"])
          .output()
          .expect("Failed to build FFI crate");

      let stderr = String::from_utf8_lossy(&output.stderr);
      let has_warnings = stderr.contains("warning:");

      if has_warnings {
          panic!("FFI crate build produced warnings (expected zero)");
      }
  }
  ```
- [ ] Test runs in CI pipeline as part of standard test suite

#### AC6: Documentation
- [ ] Update `crates/bitnet-ggml-ffi/README.md` with:
  - Rationale for `-isystem` vs `-I` flags
  - Platform-specific handling (MSVC vs GCC/Clang)
  - Build hygiene policy (zero warnings requirement)
  - How to regenerate fixtures if GGML vendor updated
  - Troubleshooting guide for build warnings

### Verification Steps

```bash
# 1. Clean build test (zero warnings)
cargo clean -p bitnet-ggml-ffi
cargo build -p bitnet-ggml-ffi --no-default-features --features iq2s-ffi 2>&1 | \
  tee /tmp/ffi_build.log

# 2. Verify zero warnings
if grep -i "warning" /tmp/ffi_build.log; then
  echo "❌ FFI build has warnings"
  exit 1
fi
echo "✅ FFI build clean (zero warnings)"

# 3. Test with different compilers (Linux/macOS)
CC=gcc cargo build -p bitnet-ggml-ffi --no-default-features --features iq2s-ffi
CC=clang cargo build -p bitnet-ggml-ffi --no-default-features --features iq2s-ffi

# 4. Test MSVC (Windows - requires Windows CI or local machine)
# cargo build -p bitnet-ggml-ffi --no-default-features --features iq2s-ffi

# 5. Verify vendored commit enforcement (should fail in CI if missing)
rm crates/bitnet-ggml-ffi/csrc/VENDORED_GGML_COMMIT
cargo build -p bitnet-ggml-ffi --features iq2s-ffi 2>&1 | \
  grep "VENDORED_GGML_COMMIT is 'unknown'" && echo "✅ CI enforcement works"

# 6. Restore vendored commit file (from git)
git checkout crates/bitnet-ggml-ffi/csrc/VENDORED_GGML_COMMIT

# 7. Run regression test
cargo test -p xtask --test ffi_build_hygiene_test --features iq2s-ffi -- --nocapture

# 8. Cross-platform validation (requires cross-compilation setup)
rustup target add x86_64-unknown-linux-gnu x86_64-apple-darwin x86_64-pc-windows-msvc
cargo build -p bitnet-ggml-ffi --features iq2s-ffi --target x86_64-unknown-linux-gnu
cargo build -p bitnet-ggml-ffi --features iq2s-ffi --target x86_64-apple-darwin
# Windows MSVC requires Windows machine or cross-compilation toolchain
```

### Estimated Effort

**2-3 hours** (including platform testing and documentation)

### Related Issues/PRs

- PR #475 - Comprehensive integration with FFI feature gate improvements (merged to main)
- Issue #469 - FFI build hygiene and tokenizer parity (in progress)
- Related spec: `docs/explanation/specs/SPEC-2025-001-next-steps-priority-implementation.md` (Story 4)

### Technical Implementation Notes

**Platform-Specific Flag Handling:**
```rust
// build.rs enhancement
let is_msvc = build.get_compiler().is_like_msvc();
let isystem_flag = if is_msvc {
    "/external:I"  // MSVC external headers (VS 2019+)
} else {
    "-isystem"     // GCC/Clang system includes
};

build
    .include("csrc")  // Local shim code (warnings visible via -I)
    .flag(&format!("{}csrc/ggml/include", isystem_flag))  // Vendored GGML (warnings suppressed)
    .flag(&format!("{}csrc/ggml/src", isystem_flag))
    .flag_if_supported("/external:W0")  // MSVC: suppress external warnings
```

**Shim Code (Warnings Visible):**
- `csrc/ggml_quants_shim.c` - BitNet.rs IQ2_S FFI bridge
- `csrc/ggml_consts.c` - GGML constant exports

**Vendored Code (Warnings Suppressed):**
- `csrc/ggml/include/` - GGML public headers
- `csrc/ggml/src/` - GGML internal implementation

**Risk Mitigation:**
1. **Platform incompatibility**: Detect compiler type at build time, use appropriate flags
2. **Vendored updates**: Pin commit in `VENDORED_GGML_COMMIT`, test updates in separate PR
3. **Latent shim warnings**: CI regression test catches new warnings immediately

**Feature Flags:**
- Uses existing `iq2s-ffi` feature flag
- No new feature flags required
- Build system changes apply when `iq2s-ffi` feature is enabled

---

## Summary

These three P0 issues establish production-grade infrastructure for BitNet.rs:

1. **Issue 1 (GGUF Fixtures)**: CI stability and test performance (~150ms speedup)
2. **Issue 2 (EnvGuard)**: Parallel test safety and deterministic execution
3. **Issue 3 (FFI Hygiene)**: Clean builds and regression visibility

**Total Estimated Effort**: 6-8 hours (can be parallelized across multiple developers)

**Implementation Order**:
1. Issue 3 (FFI Hygiene) - Quick win, independent work stream (2-3h)
2. Issue 1 (GGUF Fixtures) - Foundation for CI stability (2-3h)
3. Issue 2 (EnvGuard) - Parallel test safety (2h)

All issues have clear acceptance criteria, verification steps, and measurable success metrics aligned with BitNet.rs TDD practices and cargo + xtask workflow automation.
