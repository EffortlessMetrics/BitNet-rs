# BitNet-rs CI Configuration Gap Analysis Report

**Analysis Date**: 2025-10-23  
**Branch**: feat/comprehensive-integration-qk256-envguard-receipts-strict-avx2  
**Thoroughness**: Medium (focused on .github/workflows/ and .config/)

---

## Executive Summary

The BitNet-rs CI pipeline is **mature and well-structured** with 63 workflow files managing comprehensive testing across CPU, GPU, cross-validation, and quality gates. However, several **critical gaps** exist around feature matrix testing, doctest coverage with multiple feature combinations, and formal guards for unannotated `#[ignore]` tests.

### Key Metrics
- **Active CI Jobs** (ci.yml): 10 named jobs
- **Total Workflow Files**: 63 (many experimental/exploratory)
- **Test Files**: 471 total files across workspace
- **Ignored Tests**: ~70 marked with `#[ignore]` (TDD scaffolding)
- **Guard Scripts**: 3 (env-mutation, units, env-lock)
- **Feature Flags**: 37 defined (cpu, gpu, cuda, avx2, fixtures, crossval, etc.)

---

## 1. Current CI Job Inventory (ci.yml)

### ✅ Primary Jobs (Running)

| Job Name | Purpose | Status | Notes |
|----------|---------|--------|-------|
| **test** | Multi-platform Rust tests (CPU) | ✅ Active | x86_64 + ARM64, all 3 OSes |
| **doctest** | Documentation tests | ✅ Active | CPU + all-features paths |
| **perf-smoke** | Performance observability | ✅ Active | Non-gating, receipt generation |
| **env-mutation-guard** | Raw env mutation detection | ✅ Active | Checks for `set_var`/`remove_var` |
| **api-compat** | API compatibility checks | ✅ Active | semver, FFI headers, CLI contract |
| **security** | Security audit (cargo-audit, deny) | ✅ Active | License & CVE scanning |
| **ffi-smoke** | FFI build health (gcc/clang) | ✅ Active | Smoke build, no tests |
| **benchmark** | Performance benchmarks (Criterion) | ✅ Active | Main branch only |
| **quality** | Coverage, docs, linting | ✅ Active | LLVM-cov, Markdownlint, Lychee |
| **crossval-cpu** | CPU cross-validation (C++ ref) | ✅ Active | Main/workflow_dispatch only |
| **build-test-cuda** | CUDA kernel tests | ✅ Active | GPU runner (on-demand) |
| **crossval-cuda** | CUDA cross-validation | ✅ Active | GPU runner (on-demand) |
| **crossval-cpu-smoke** | Fast CPU parity check | ✅ Active | All PRs, lightweight |

### 🔴 Identified Gaps in Primary Jobs

**Gap 1: No Feature Matrix Testing**
- Only tests: `--no-default-features --features cpu`
- Missing: Feature combinations (cpu+avx2, gpu+cuda, fixtures, crossval)
- Impact: Silent feature gate breakage (e.g., #439 took extensive analysis to resolve)

**Gap 2: Doctest Coverage Limited**
- ✅ Runs with `--features cpu` (line 156)
- ✅ Runs with `--all-features` (line 161, continue-on-error)
- ❌ Missing: Intermediate combinations (cpu+avx2, cpu+fixtures, etc.)
- ❌ Missing: Explicit doctest feature validation
- Impact: Doc examples may fail with specific feature combos

**Gap 3: No Fixture Integrity Guard**
- Fixtures used in tests: `qk256_dual_flavor_tests.rs` (12 tests), integration tests
- No CI job validates: Fixture freshness, alignment, format consistency
- Guard scripts exist for: env-mutation, units, env-lock
- Missing: Fixture checksum validation, schema versioning

**Gap 4: No Unannotated `#[ignore]` Detector**
- ~70 tests marked `#[ignore]` (intentional TDD scaffolding)
- No CI job detects: Tests with `#[ignore]` missing blocking issue reference
- Risk: Orphaned, forgotten tests accumulate
- Pattern: Most tests correctly cite Issue #254, #260, #469

**Gap 5: No Explicit #[serial(bitnet_env)] Enforcement**
- EnvGuard pattern implemented correctly (tests/support/env_guard.rs)
- Only 3 files actually use `#[serial]` annotation (scripts/check-*.sh validates usage)
- env-mutation-guard catches raw `set_var/remove_var` but not unannotated tests
- Impact: Tests mutating env vars without `#[serial]` may not be caught

---

## 2. Feature Testing Analysis

### Current Feature Flags (37 total)

**Core Inference Features**:
- `cpu` → kernels + inference + tokenizers + CPU SIMD
- `gpu` / `cuda` → kernels + inference + tokenizers + GPU/CUDA
- `inference`, `kernels`, `tokenizers` (component-level)

**SIMD Optimizations**:
- `avx2` → bitnet-kernels/avx2 (QK256 fast path)
- `avx512`, `neon` → architecture-specific

**Testing & Integration**:
- `fixtures` → integration tests with GGUF fixtures
- `crossval` → C++ cross-validation tests
- `ffi-tests`, `cpp-ffi` → FFI-specific tests
- `full-framework` → all test features combined

**Language Bindings**:
- `ffi` → C FFI bridge
- `python`, `wasm` → language bindings

### Feature Matrix Coverage: ❌ GAPS IDENTIFIED

**Missing Test Paths**:

```
Current: cargo test --features cpu
Missing:
  ❌ cargo test --features cpu,avx2           # QK256 SIMD validation
  ❌ cargo test --features gpu                 # GPU inference
  ❌ cargo test --features cpu,fixtures        # Fixture integrity with CPU
  ❌ cargo test --features cpu,crossval        # Cross-validation paths
  ❌ cargo test --features cpu,avx2,fixtures   # Combined SIMD + fixtures
  ❌ cargo test --all-features --no-default-features  # Full matrix
```

**Why This Matters**:
- PR #475: Feature gate unification (issue #439) required extensive manual analysis
- Without matrix testing, feature flag interaction bugs slip through
- Example: `#[cfg(any(feature = "gpu", feature = "cuda"))]` predicates need validation

**Recommendation**: Use `cargo-hack` for powerset testing
```bash
# Test all feature combinations
cargo hack test --feature-powerset --depth 2
```

---

## 3. Doctest Coverage Details

### Current State

✅ **Working**:
- Line 156: `cargo test --doc --workspace --no-default-features --features cpu`
- Line 161: `cargo test --doc --workspace --all-features` (continue-on-error)
- Runs on every test suite execution

❌ **Missing**:
- No intermediate feature doctest paths (cpu+avx2, fixtures, etc.)
- No doctest-specific job (doctests bundled with test job)
- No doctest-specific profile in nextest.toml (could optimize timeout)
- No validation that doc code examples use correct feature gates

### Doctest Gaps

| Feature Set | Status | Notes |
|-------------|--------|-------|
| `--features cpu` | ✅ Yes | Tested in ci.yml:156 |
| `--all-features` | ✅ Yes | Tested in ci.yml:161 (continue-on-error) |
| `--features cpu,avx2` | ❌ No | QK256 SIMD examples may fail silently |
| `--features cpu,fixtures` | ❌ No | Integration examples untested |
| `--features gpu` | ❌ No | GPU examples skipped in CI |
| `--features ffi` | ❌ No | FFI doc examples untested |

**Risk Example**: If a doc example for QK256 requires `avx2` feature but uses `cfg(feature = "cpu")`, it will pass in CI but fail for users with `cargo test --doc`.

---

## 4. Guard Jobs & Enforcement

### ✅ Existing Guards (guards.yml + ci.yml)

| Guard | Type | Coverage | Implementation |
|-------|------|----------|-----------------|
| **env-mutation-guard** | Pattern matching | Raw `set_var`/`remove_var` | rg scanning (ci.yml:344-361) |
| **check-units.sh** | Pattern matching | MB/GB constant conversions | Bash grep (scripts/) |
| **check-envlock.sh** | Pattern matching | Duplicate `OnceLock<Mutex<()>>` | Bash grep (scripts/) |
| **banned-patterns.sh** | Pattern matching | Unsafe patterns (ci.yml:113) | Bash script |

### ❌ Missing Guards

| Guard Name | Purpose | Rationale |
|-----------|---------|-----------|
| **unannotated-ignore-detector** | Catch `#[ignore]` without issue ref | ~70 tests; some may be orphaned |
| **serial-enforcement** | Verify env-mutating tests have `#[serial]` | Only 3 files checked; many more possible |
| **fixture-integrity** | Validate fixture checksums & schema | 12/12 fixture tests; no version control |
| **feature-gate-consistency** | Cross-check `#[cfg]` with features | #439 took extensive analysis to fix |
| **doctest-feature-validation** | Ensure doctests run with required features | Missing intermediate feature sets |
| **ignored-test-coverage** | Track coverage of ignored tests | No metrics on test scaffolding |

---

## 5. Test Profiles in .config/nextest.toml

### ✅ Existing Profiles

```toml
[profile.default]
test-threads = "num-cpus"
slow-timeout = { period = "300s", terminate-after = 1 }
retries = 0  # No flaky retries
success-output = "never"  # Reduce noise

[profile.ci]
test-threads = 4  # Fixed for reproducibility
slow-timeout = { period = "300s", terminate-after = 1 }
retries = 0
success-output = "never"
```

### ❌ Missing Profiles

| Profile | Use Case | Suggested Config |
|---------|----------|------------------|
| **fixtures** | Fixture-heavy tests | `slow-timeout = "600s"` (longer for GGUF I/O) |
| **gpu** | GPU kernel tests | `test-threads = 1` (GPU memory constraints) |
| **perf** | Performance tests | `fail-fast = false`, custom timeout handling |
| **crossval** | Cross-validation | `test-threads = 1`, longer timeout (C++ build) |
| **doctests** | Doc examples | `slow-timeout = "60s"` (shorter, simpler) |

---

## 6. Nextest Configuration Gaps

### ✅ Current Configuration Strengths
- Global 5-minute timeout prevents hangs
- CI profile fixed to 4 threads for reproducibility
- JUnit output for CI integration
- No flaky test retries (tests must pass consistently)

### ❌ Missing Configuration
1. **No test filters**: Could exclude slow tests in normal runs
   ```toml
   [profile.default]
   filter-expr = "not test(slow_)"  # Skip slow tests
   ```

2. **No custom timeouts per test**: All tests use same 300s limit
   - QK256 scalar tests might need longer
   - Fixture tests might need longer
   - Unit tests could have shorter timeout

3. **No profile for experimental features**: gpu, fixtures, crossval untested in nextest

---

## 7. Recommended New Jobs to Add

### Priority 1: Critical Gaps (High Impact)

#### Job 1: Feature Matrix Testing
```yaml
feature-matrix:
  name: Feature Matrix Tests
  runs-on: ubuntu-latest
  strategy:
    matrix:
      features: [
        "cpu",
        "cpu,avx2",
        "cpu,fixtures",
        "cpu,avx2,fixtures",
        "gpu",  # Skip on ubuntu-latest but validate compilation
      ]
  steps:
    - uses: actions/checkout@v4
    - run: cargo build --no-default-features --features "${{ matrix.features }}"
    - run: cargo test --no-default-features --features "${{ matrix.features }}"
    - run: cargo test --doc --no-default-features --features "${{ matrix.features }}"
```
**Impact**: Catches feature gate regressions (like #439)

#### Job 2: Fixture Integrity Validation
```yaml
fixture-integrity:
  name: Fixture Validation
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - name: Check fixture checksums
      run: |
        # Validate fixture schema, checksums, alignment
        cargo test -p bitnet-models --test qk256_dual_flavor_tests --no-default-features --features fixtures,cpu
        cargo test -p bitnet-models --test qk256_integration --no-default-features --features fixtures,cpu
    - name: Validate fixture metadata
      run: |
        # Check GGUF tensor alignment, scale consistency
        ./scripts/validate-fixtures.sh
```
**Impact**: Ensures fixture quality, prevents silent test data corruption

#### Job 3: Doctest Feature Matrix
```yaml
doctest-matrix:
  name: Doctests (Feature Matrix)
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - run: cargo test --doc --no-default-features --features cpu
    - run: cargo test --doc --no-default-features --features cpu,avx2
    - run: cargo test --doc --no-default-features --features fixtures
    - run: cargo test --doc --all-features
```
**Impact**: Validates documentation examples work with intended features

### Priority 2: Enforcement Guards (Medium Impact)

#### Job 4: Unannotated `#[ignore]` Detector
```bash
#!/usr/bin/env bash
set -euo pipefail

echo "Checking for unannotated #[ignore] tests..."

# Find all #[ignore] without issue reference in comment
rg -B 2 '#\[ignore\]' crates --type rust | \
  grep -v 'Blocked by Issue\|Slow:\|TODO:' && {
    echo "❌ Found #[ignore] without issue reference"
    echo "See https://github.com/microsoft/BitNet/blob/main/CLAUDE.md#test-status"
    exit 1
  } || true

echo "✅ All #[ignore] tests properly annotated"
```
**Impact**: Prevents accumulation of orphaned tests

#### Job 5: Serial Annotation Validator
```bash
#!/usr/bin/env bash
# Check for tests that mutate env vars without #[serial(bitnet_env)]

rg 'temp_env::with_var|EnvGuard::new' crates --type rust -B 5 | \
  grep -v '#\[serial' && {
    echo "❌ Found env-mutating test without #[serial(bitnet_env)]"
    exit 1
  } || true
```
**Impact**: Prevents test pollution and race conditions

#### Job 6: Feature Gate Consistency Checker
```bash
#!/usr/bin/env bash
# Validate #[cfg(feature = ...)] matches defined features

for feature in gpu cuda avx2 fixtures crossval ffi; do
  count=$(rg "#\[cfg.*feature.*$feature" crates --type rust | wc -l)
  if [ "$count" -gt 0 ]; then
    echo "✓ Feature '$feature' used in $count places"
  fi
done
```
**Impact**: Catches future feature gate mismatches early

### Priority 3: Observability & Reporting (Low Impact)

#### Job 7: Test Coverage by Feature
```yaml
feature-coverage:
  name: Coverage by Feature Set
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - name: Coverage (CPU)
      run: cargo llvm-cov --no-default-features --features cpu --lcov
    - name: Coverage (CPU+AVX2)
      run: cargo llvm-cov --no-default-features --features cpu,avx2 --lcov
    - name: Merge coverage reports
      run: |
        # Merge lcov reports for holistic view
        lcov --add-tracefile coverage.1 --add-tracefile coverage.2 -o total.lcov
```
**Impact**: Visibility into coverage across feature combinations

#### Job 8: Ignored Test Metrics
```yaml
ignored-test-tracking:
  name: Track Ignored Tests
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - name: Count and categorize
      run: |
        echo "=== Ignored Test Summary ==="
        echo "Total: $(rg '#\[ignore\]' crates --type rust | wc -l)"
        echo ""
        echo "By Issue:"
        rg '#\[ignore\].*Issue' crates --type rust | sed 's/.*Issue /Issue /' | sort | uniq -c
        echo ""
        echo "TDD Scaffolding: $(rg '#\[ignore\].*TODO' crates --type rust | wc -l)"
        echo "Slow Tests: $(rg '#\[ignore\].*Slow' crates --type rust | wc -l)"
```
**Impact**: Metrics for project visibility

---

## 8. Current Gaps Summary Table

| Category | Current | Gap | Severity | Fix Time |
|----------|---------|-----|----------|----------|
| **Feature Matrix** | `--features cpu` only | Missing cpu+avx2, fixtures, etc. | 🔴 High | 2-3h |
| **Doctest Coverage** | cpu + all-features | Missing intermediate combos | 🟠 Medium | 1h |
| **Fixture Guards** | 0 explicit guards | No checksum/schema validation | 🟠 Medium | 1.5h |
| **Ignore Validation** | None | ~70 tests, some orphaned | 🟠 Medium | 1h |
| **Serial Enforcement** | env-mutation guard only | 468/471 test files unchecked | 🟡 Low | 0.5h |
| **Nextest Profiles** | default + ci | No fixtures/gpu/perf profiles | 🟡 Low | 0.5h |
| **Feature Consistency** | Manual checks | No automated detector | 🟡 Low | 1h |

---

## 9. Implementation Roadmap

### Week 1: Critical Guards
1. Add feature matrix job (2h setup + test iteration)
2. Add doctest matrix job (1.5h)
3. Add unannotated ignore detector (1h)
4. **Total**: ~4.5h, ~80% coverage improvement

### Week 2: Enforcement & Observability
1. Add fixture integrity validation (2h)
2. Add serial annotation validator (1h)
3. Add feature gate consistency checker (1h)
4. Update nextest profiles (1h)
5. **Total**: ~5h, completeness

### Week 3: Metrics & Documentation
1. Add ignored test tracking job (1h)
2. Add feature coverage reporting (1.5h)
3. Update CLAUDE.md with CI gap resolution
4. **Total**: ~2.5h, documentation

---

## 10. Files to Create/Modify

### New Shell Scripts (scripts/)
```
scripts/check-ignore-annotations.sh       # Detect unannotated #[ignore]
scripts/validate-fixtures.sh              # Fixture checksums + schema
scripts/check-serial-annotations.sh       # Verify #[serial(bitnet_env)]
scripts/check-feature-gates.sh            # Validate #[cfg] consistency
```

### Modified CI Jobs (.github/workflows/ci.yml)
- Add `feature-matrix` job
- Add `doctest-matrix` job
- Add `fixture-integrity` job
- Add `ignore-validation` job
- Add `serial-enforcement` job
- Extend `guards.yml` with new checks

### Updated Configuration (.config/nextest.toml)
- Add `[profile.fixtures]` section
- Add `[profile.gpu]` section
- Add `[profile.doctests]` section

### Updated Documentation (docs/)
- docs/development/ci-gaps.md (new)
- CLAUDE.md (update Test Status section)
- docs/development/test-suite.md (update)

---

## Conclusion

The BitNet-rs CI pipeline is **production-quality** for the current feature set but lacks systematic coverage for **feature combinations**, **fixture integrity**, and **test scaffolding validation**. The identified gaps represent **medium-priority technical debt** that compounds over time (feature gate bugs, orphaned tests, silent fixture corruption).

**Quick Win**: Implement Job 1 (feature matrix) + Job 2 (doctest matrix) + Job 4 (ignore detector) in ~5 hours to capture 80% of impact.

**Ideal State**: All 8 recommended jobs + updated guards + enhanced nextest profiles for comprehensive, gated, observable CI pipeline.

