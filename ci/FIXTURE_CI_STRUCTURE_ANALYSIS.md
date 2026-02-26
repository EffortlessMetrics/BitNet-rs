# BitNet-rs Fixture Tests and CI Workflow Structure - Comprehensive Analysis

## Executive Summary

The BitNet-rs project implements a sophisticated CI workflow with:
- **3 committed GGUF test fixtures** for QK256 and BitNet32 quantization format validation
- **5 independent guard jobs** ensuring code quality and correctness
- **18 named CI jobs** with complex dependency chains for comprehensive testing
- **Parallel guard execution** to prevent race conditions in tests
- **Pre-commit hooks** mirroring CI gates for local enforcement

---

## 1. FIXTURE TESTS - Location and Implementation

### 1.1 Test File: fixture_integrity_tests.rs

**Path**: `/home/steven/code/Rust/BitNet-rs/crates/bitnet-models/tests/fixture_integrity_tests.rs`

**Purpose**: Validates structural integrity of GGUF fixtures without requiring shell execution

**Key Tests**:
- `test_qk256_4x256_header_integrity()` - Validates 10,816 byte single-block QK256 fixture
- `test_qk256_3x300_header_integrity()` - Validates 10,696 byte multi-block QK256 with tail
- `test_bitnet32_2x64_header_integrity()` - Validates 8,832 byte BitNet32-F16 fixture
- `test_all_fixtures_present()` - Ensures all 3 fixtures are committed
- `test_sha256sums_file_present()` - Verifies SHA256SUMS manifest exists

**Validation Checks**:
1. GGUF magic number (bytes 0-3 must be "GGUF")
2. GGUF version (bytes 4-7 as little-endian u32, must be 2 or 3)
3. Expected file sizes (from exploratory generation)
4. SHA256SUMS file presence with all three fixtures

**Execution Context**:
- Runs with `--features fixtures` flag
- Uses cargo test framework (no nextest profile)
- Early detection of fixture corruption before CI jobs

### 1.2 Validation Script: validate-fixtures.sh

**Path**: `/home/steven/code/Rust/BitNet-rs/scripts/validate-fixtures.sh`

**Purpose**: Shell-based validation with GGUF structure and metadata inspection

**Validation Stages**:
```
Stage 1: SHA256 Checksum Verification
├─ Reads ci/fixtures/qk256/SHA256SUMS
├─ Verifies each .gguf against checksum
└─ Fails if any file corrupted or modified

Stage 2: GGUF Binary Structure Validation
├─ Magic number check (first 4 bytes = "GGUF")
├─ Version validation (bytes 4-7 must be 2 or 3)
└─ Uses od command for little-endian u32 extraction

Stage 3: GGUF Metadata Inspection (if jq available)
├─ Runs: cargo run -p bitnet-cli --features cpu,full-cli -- inspect
├─ Validates required metadata keys (general.architecture, general.name)
├─ Checks tensor alignment (GGUF v3 requires 32-byte alignment)
└─ Verifies tensor count >= 2 (realistic fixtures)

Stage 4: Output & Error Handling
├─ Uses GitHub Actions workflow annotations (::error::, ::warning::)
└─ Provides regeneration instructions if validation fails
```

**Error Messages**:
- Missing checksums file → Exit 1 (blocking)
- Checksum mismatch → Exit 1 (blocking), suggests regeneration
- Invalid magic/version → Exit 1 (blocking)
- Missing metadata keys → Warning (non-blocking)
- Alignment issues → Error (v3 only, blocking)
- Tensor count < 2 → Warning (non-blocking)

### 1.3 Fixture Files and Checksums

**Location**: `/home/steven/code/Rust/BitNet-rs/ci/fixtures/qk256/`

**Files**:
```
bitnet32_2x64.gguf        8,832 bytes   BitNet32-F16 quantization (2x64 tensor)
qk256_3x300.gguf         10,696 bytes   QK256 with tail (3x300 tensor)
qk256_4x256.gguf         10,816 bytes   QK256 single block (4x256 tensor)
SHA256SUMS                 251 bytes   Manifest with all three checksums
README.md                4,900 bytes   Fixture documentation and regeneration guide
QUICK_REFERENCE.md       3,000 bytes   Quick lookup for fixture specs
```

**SHA256 Checksums**:
```
c1568a0a08e38ef2865ce0816bfd2c617e5589c113114cd731e4c5014b7fbb20  bitnet32_2x64.gguf
6e5a4f21607c0064affbcb86133627478eb34d812b59807a7123ff386c63bd3e  qk256_3x300.gguf
a41cc62c893bcf1d4c03c30ed3da12da03c339847c4d564e9e5794b5d4c6932a  qk256_4x256.gguf
```

**Generation Details**:
- GGUF Version: 3
- Tensors per fixture: 2 (tok_embeddings.weight + output.weight)
- Deterministic seeds: 42, 43, 44 for reproducibility
- Code mapping: 0 → -2.0, 1 → -1.0, 2 → +1.0, 3 → +2.0
- Alignment: 32-byte GGUF v3 compliance

**Regeneration Process**:
```bash
# Generate new fixtures
cargo test -p bitnet-models --test qk256_dual_flavor_tests \
  --no-default-features --features fixtures test_dump_fixture_for_debug -- --nocapture

# Copy to ci/fixtures/qk256/
cp /tmp/test_qk256_4x256.gguf ci/fixtures/qk256/qk256_4x256.gguf
cp /tmp/test_bitnet32_2x64.gguf ci/fixtures/qk256/bitnet32_2x64.gguf
cp /tmp/test_qk256_3x300.gguf ci/fixtures/qk256/qk256_3x300.gguf

# Update checksums
cd ci/fixtures/qk256 && sha256sum *.gguf > SHA256SUMS
```

---

## 2. CI WORKFLOW STRUCTURE

### 2.1 Job Dependency Graph (DAG)

**Complete Dependency Tree**:
```
PRIMARY GATE
└─ test (6 matrix configs, multi-platform)
   ├─ feature-matrix (gates)
   │  └─ feature-hack-check (non-blocking observe)
   ├─ doctest-matrix (gates)
   ├─ doctest (gates, explicit needs: test)
   ├─ ffi-smoke (gates)
   │  ├─ ffi-zero-warning-windows (gates)
   │  └─ ffi-zero-warning-linux (gates, 2 compiler configs)
   ├─ perf-smoke (non-blocking observe)
   ├─ crossval-cpu-smoke (gates)
   │  └─ crossval-cpu (main/dispatch only, gates)
   ├─ build-test-cuda (GPU only, gates)
   └─ crossval-cuda (GPU/main only, gates)

INDEPENDENT GUARDS (run immediately, in parallel)
├─ guard-fixture-integrity (gates) ────────────────→ CI job #301
├─ guard-serial-annotations (gates) ───────────────→ CI job #317
├─ guard-feature-consistency (gates) ───────────────→ CI job #332
├─ guard-ignore-annotations (observe) ──────────────→ CI job #348
└─ env-mutation-guard (gates, inline ripgrep) ─────→ CI job #557

ALWAYS-RUN JOBS (no dependencies)
├─ api-compat (PR only, gates)
├─ security (gates)
├─ quality (gates)
└─ benchmark (main only, no gating)
```

**Critical Path**: test → feature-matrix/doctest-matrix → downstream tests

**Parallel Execution**: Guards run independently from primary test path

### 2.2 Guard Job Specifications

#### Guard 1: guard-fixture-integrity (Line 301)

**Name**: Guard - Fixture Integrity  
**Platform**: ubuntu-latest  
**Dependency**: None (independent)  
**Blocking**: YES (must pass for merge)  

**Steps**:
1. Checkout repository
2. Install Rust (stable)
3. Setup cache
4. Run: `bash scripts/validate-fixtures.sh`

**What It Validates**:
- SHA256 checksums for all 3 fixtures
- GGUF v2/v3 magic number
- GGUF version bytes
- (Optional) bitnet-cli inspect for metadata/alignment
- Tensor alignment (32-byte for GGUF v3)
- Tensor count >= 2

**Failure Scenarios**:
- Any checksum mismatch → STOP
- Invalid magic/version → STOP
- Alignment violations (GGUF v3) → STOP
- Missing metadata keys → Warning only

#### Guard 2: guard-serial-annotations (Line 317)

**Name**: Guard - Serial Annotations  
**Platform**: ubuntu-latest  
**Dependency**: None (independent)  
**Blocking**: YES (must pass for merge)  

**Steps**:
1. Checkout
2. Install ripgrep: `sudo apt-get install ripgrep`
3. Run: `bash scripts/check-serial-annotations.sh`

**What It Validates**:
- All env-mutating tests have `#[serial(bitnet_env)]`
- Uses ripgrep pattern: `EnvGuard::new|temp_env::with_var`
- Context backtrack: 10 lines before mutation
- Searches: `#[serial(bitnet_env)]` in context

**Failure Scenarios**:
- EnvGuard usage without `#[serial]` → STOP
- Lists offending file:line numbers
- Suggests proper pattern usage

#### Guard 3: guard-feature-consistency (Line 332)

**Name**: Guard - Feature Consistency  
**Platform**: ubuntu-latest  
**Dependency**: None (independent)  
**Blocking**: YES (must pass for merge)  

**Steps**:
1. Checkout
2. Install ripgrep
3. Run: `bash scripts/check-feature-gates.sh`

**What It Validates**:
1. Extract defined features from Cargo.toml `[features]` section
2. Find all `#[cfg(feature = "...")]` usage in code
3. Cross-check: all used features are defined
4. Anti-pattern detection: bare `#[cfg(feature = "gpu")]` without fallback

**Failure Scenarios**:
- Undefined feature in `#[cfg(...)]` → STOP
- GPU feature without cuda fallback → Warning
- Provides fix suggestions

#### Guard 4: guard-ignore-annotations (Line 348)

**Name**: Guard - Ignore Annotations  
**Platform**: ubuntu-latest  
**Dependency**: None (independent)  
**Blocking**: NO (observational, non-gating)  

**Steps**:
1. Checkout
2. Install ripgrep
3. Run: `bash scripts/check-ignore-annotations.sh`

**What It Validates**:
- All `#[ignore]` attributes have justification
- Valid patterns: "Blocked by Issue #NNN", "Slow: reason", "TODO: reason"
- Context extraction: 2 lines before `#[ignore]`
- Status: 144 total ignores, all properly annotated

**Reporting**:
- Lists unannotated ignores if found
- Suggests proper patterns
- Non-blocking (doesn't fail build)

#### Guard 5: env-mutation-guard (Line 557)

**Name**: Guard - No raw env mutations  
**Platform**: ubuntu-latest  
**Dependency**: None (independent)  
**Blocking**: YES (must pass for merge)  

**Implementation**: Inline ripgrep (no script file)

**What It Validates**:
- Detects: `std::env::set_var(` and `std::env::remove_var(` in code
- Scope: `crates/` directory only
- Excludes: test helpers, support dirs, env_guard.rs
- Pattern: `rg -n '(std::env::set_var|std::env::remove_var)\(' crates`

**Failure Scenarios**:
- Raw env mutations found → STOP
- Lists offenders with context
- Requires EnvGuard pattern

### 2.3 Feature-Specific Test Matrix

**Feature Matrix Tests** (Line 187):
```yaml
Features tested:
  - cpu
  - cpu,avx2
  - cpu,fixtures          # Tests fixture loading
  - cpu,avx2,fixtures     # Both optimizations + fixtures
  - ffi
  - gpu (compile-only, no runtime tests)

Nextest Profile: 
  - fixtures profile used when "fixtures" feature present
  - 2 threads, 600s timeout, junit output
  - Prevents I/O contention with fixture loading
```

**Doctest Matrix** (Line 259):
```yaml
Features tested:
  - cpu
  - cpu,avx2
  - all-features (with continue-on-error for GPU)

Purpose: Validate documentation examples
```

### 2.4 Fixture-Specific Test Configuration

**Nextest Profile** (`.config/nextest.toml` line 43):
```toml
[profile.fixtures]
fail-fast = false
test-threads = 2              # Limited I/O contention
retries = 0                    # No flaky retries
slow-timeout = { period = "600s", terminate-after = 1 }  # 10min timeout
success-output = "never"
status-level = "fail"
junit = "target/nextest/fixtures/junit.xml"
```

**When Applied**:
- Automatically selected in `feature-matrix` when features include "fixtures"
- Also manually selectable via `--profile fixtures` flag

---

## 3. GIT HOOKS - Local Enforcement

### 3.1 Pre-commit Hook

**Path**: `/home/steven/code/Rust/BitNet-rs/.githooks/pre-commit`

**Enable Locally**:
```bash
git config core.hooksPath .githooks
```

**Check 1: Bare #[ignore] Markers**
```bash
Pattern: rg -n '#\[ignore\](?!\s*(=|//))' --type rust crates tests tests-new xtask

Accepts:
  1. #[ignore = "reason"]                          # Attribute style
  2. #[ignore] // reason                            # Inline comment
  3. // reason + next line #[ignore]                # Preceding comment

Fails if none of above found within 2 lines
```

**Check 2: Raw Environment Mutations**
```bash
Pattern: rg -n '(std::env::set_var|std::env::remove_var)\(' crates \
  --glob '!**/tests/support/**' --glob '!**/support/**' \
  --glob '!**/helpers/**' --glob '!**/test_fixtures/**'

Scope: Production code only (crates/ excluding test helpers)
Requires: EnvGuard + #[serial(bitnet_env)] pattern in tests
```

**Behavior**:
- Requires ripgrep to be installed locally
- Exit code 1 if violations found
- Helpful error messages with fix suggestions
- Color-coded output (red errors, green success)

---

## 4. RIPGREP USAGE SUMMARY

| File | Pattern | Purpose | Tool |
|------|---------|---------|------|
| `.githooks/pre-commit` | Negative lookahead `(?!\s*(=\|//))` | Bare ignore detection | Local |
| `scripts/check-serial-annotations.sh` | `EnvGuard\|\|temp_env` + backtrack 5 lines | Verify annotation pattern | CI guard |
| `scripts/check-feature-gates.sh` | Feature definition + `#\[cfg(...feature...]` | Cross-check feature usage | CI guard |
| `scripts/check-ignore-annotations.sh` | `#\[ignore\]` + context extraction | Verify issue references | CI guard |
| `ci.yml:env-mutation-guard` (line 570) | Inline ripgrep, no script | Check env mutations | CI job |

**Installation**:
- CI: Installed per-guard via `apt-get install ripgrep`
- Local: Manual installation required for pre-commit hook

---

## 5. CI JOB ORDERING AND EXECUTION

### 5.1 Execution Tiers

**Tier 0: Primary Gate (blocks everything else)**
- `test` (6 configs × multi-platform)
- Duration: 15-20 minutes
- Status: Must pass for all downstream

**Tier 1: Primary Features & Validation (depends on test)**
- `feature-matrix` (6 configs)
- `doctest-matrix` (3 configs)
- `doctest`
- Duration: 10-15 minutes each (parallel)

**Tier 2: Build & FFI Validation (depends on test)**
- `ffi-smoke` (2 compilers)
- `ffi-zero-warning-*` (3 variants)
- Duration: 5-10 minutes each

**Tier 3: Cross-Validation (depends on test)**
- `crossval-cpu-smoke`
- `crossval-cpu` (main/dispatch only)
- `build-test-cuda` (GPU runners only)
- `crossval-cuda` (GPU runners only)
- Duration: 10-30 minutes (GPU jobs longer)

**Tier 4: Independent Guards (no dependencies, immediate start)**
- `guard-fixture-integrity`
- `guard-serial-annotations`
- `guard-feature-consistency`
- `guard-ignore-annotations` (non-blocking)
- `env-mutation-guard`
- Duration: 1-3 minutes each (parallel)

**Tier 5: Always-Run Jobs (no dependencies)**
- `api-compat` (PR only)
- `security`
- `quality`
- `benchmark` (main only)

### 5.2 Critical Path Analysis

**Shortest Path to Merge**:
```
test (20 min)
  ├─ feature-matrix (15 min) ──┐
  ├─ doctest-matrix (10 min)   │
  └─ doctest (5 min)           ├─ ~15 min (parallel end)
                               
Total: 20 min (test) + 15 min (parallel downstream) = ~35 minutes
```

**Longest Individual Path**:
```
test (20 min) → crossval-cuda (30 min) = 50 minutes
(but only runs on main/dispatch/schedule)
```

**Total Parallel Wall Time** (with all guards):
- ~30-50 minutes including guards running independently
- Guards do NOT add to critical path (independent execution)

### 5.3 Job Matrix Expansion

**Test Job** (Lines 38-135):
```yaml
Matrix:
  os: [ubuntu-latest, windows-latest, macos-latest]
  rust: [stable]
  targets:
    - ubuntu + x86_64 (default)
    - ubuntu + aarch64 (cross-compile)
    - windows + x86_64 (default)
    - macos + x86_64 (default)
    - macos + aarch64 (native M1/M2)

Total: 6 configuration combinations
```

**Feature Matrix** (Lines 187-254):
```yaml
Matrix:
  features:
    - cpu
    - cpu,avx2
    - cpu,fixtures
    - cpu,avx2,fixtures
    - ffi
    - gpu (compile-only)

Total: 6 feature combinations
Execution: Sequential builds, parallel tests (--profile ci: 4 threads)
```

---

## 6. FIXTURE FILE TYPES AND QUANTIZATION FORMATS

### 6.1 QK256 Format Fixture

**Files**:
- `qk256_4x256.gguf` (10,816 bytes)
- `qk256_3x300.gguf` (10,696 bytes)

**Format Details**:
- Block size: 256 elements
- Packed bytes: 64 bytes per block
- Row stride: `ceil(cols/256) × 64` bytes
- Size formula: `rows × ceil(cols/256) × 64` bytes

**Test Coverage**:
- Single-block tensor (256 cols)
- Multi-block with tail (300 cols = 256 + 44)
- Block boundary detection
- Tail handling

### 6.2 BitNet32-F16 Format Fixture

**File**:
- `bitnet32_2x64.gguf` (8,832 bytes)

**Format Details**:
- Block size: 32 elements
- Packed bytes: 10 bytes per block (8 data + 2 F16 scale)
- Row stride: `ceil(cols/32) × 10` bytes
- Size formula: `rows × ceil(cols/32) × 10` bytes

**Test Coverage**:
- BitNet32-F16 vs QK256 discrimination
- Scale factor handling
- 2-block tensor validation

---

## 7. TEST EXECUTION AND ISOLATION

### 7.1 EnvGuard Pattern for Environment Tests

**Required Pattern**:
```rust
use serial_test::serial;
use tests::helpers::env_guard::EnvGuard;

#[test]
#[serial(bitnet_env)]  // Ensures serial execution
fn test_determinism_with_env_flags() {
    let _guard = EnvGuard::new("BITNET_DETERMINISTIC", "1");
    // Test code here - env automatically restored on drop
}
```

**Why Serial?**:
- Prevents race conditions when tests run in parallel
- CI uses `--test-threads=4` in `[profile.ci]`
- Guards ensure tests don't interfere with each other

**Validation**:
- Guard script: `scripts/check-serial-annotations.sh`
- CI job: `guard-serial-annotations`
- Pre-commit hook: Inline check

### 7.2 Fixture Loading with Temporal Files

**Pattern** (from fixture_integrity_tests.rs):
```rust
use std::fs;
use std::path::PathBuf;

fn fixture_path(filename: &str) -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest_dir)
        .join("../../ci/fixtures/qk256")
        .join(filename)
}

#[test]
fn test_qk256_4x256_header_integrity() {
    let path = fixture_path("qk256_4x256.gguf");
    assert!(path.exists(), "Fixture should exist at {:?}", path);
    
    let bytes = std::fs::read(&path)
        .expect("Should read qk256_4x256.gguf");
    
    validate_gguf_header(&bytes, "qk256_4x256.gguf");
}
```

**Fixture Location Resolution**:
- Uses `env!("CARGO_MANIFEST_DIR")` (compile-time)
- Relative path: `../../ci/fixtures/qk256/` (from crate)
- Allows tests to run from any working directory

---

## 8. VALIDATION GATES AND ERROR HANDLING

### 8.1 Gate Classification

| Gate | Type | Fails Build | Triggers | Purpose |
|------|------|-------------|----------|---------|
| guard-fixture-integrity | Binary validation | YES | Checksum/GGUF structure | Prevent corrupted fixtures |
| guard-serial-annotations | Pattern matching | YES | Ripgrep search | Ensure test isolation |
| guard-feature-consistency | Feature cross-check | YES | Feature audit | Prevent undefined features |
| guard-ignore-annotations | Documentation check | NO | Pattern matching | Track blocked tests |
| env-mutation-guard | Code pattern | YES | Ripgrep | Prevent env pollution |
| feature-matrix | Build test | YES | Cargo test | Feature combination validation |
| doctest-matrix | Example validation | YES | Cargo test | Documentation examples work |
| test | Full suite | YES | Cargo nextest | Comprehensive testing |

### 8.2 Error Message Flow

**Fixture Integrity Example**:
```
❌ Fixture checksum verification failed
  Expected: a41cc62c893bcf1d4c03c30ed3da12da03c339847c4d564e9e5794b5d4c6932a
  Got:      [actual checksum]
  
  To regenerate checksums:
    cd ci/fixtures/qk256
    sha256sum *.gguf > SHA256SUMS
```

**Serial Annotation Example**:
```
::error::Found env-mutating tests without #[serial(bitnet_env)]:
  crates/bitnet-inference/tests/determinism_tests.rs:42
  
  Example pattern:
    #[test]
    #[serial(bitnet_env)]
    fn test_with_env_mutation() {
        let _guard = EnvGuard::new("VAR_NAME", "value");
```

---

## 9. KEY METRICS AND STATISTICS

**Repository Size**:
- Fixture files: 4 files, ~30 KB total
- Fixture documentation: 4 files, ~11 KB
- CI/exploration docs: 200+ files (comprehensive analysis)

**Test Coverage**:
- Fixture integrity tests: 5 tests in fixture_integrity_tests.rs
- Total enabled tests: 1935+ (per PR #475 report)
- Ignored tests: ~70 (scaffolding/blocking issues)
- Fixture-capable tests: Multiple in qk256_dual_flavor_tests.rs

**Performance**:
- Guard execution: 1-3 minutes each (parallel)
- Full CI pipeline: 30-50 minutes wall time
- Critical path: test → feature-matrix → merge ready
- Fixture profile timeout: 600 seconds (10 minutes)

**Code Quality**:
- Feature definitions: CPU, GPU, CUDA, fixtures, ffi, crossval
- Feature combinations tested: 6 in matrix
- Ripgrep patterns: 4 guard scripts + 1 pre-commit hook
- Undefined feature check: 100% coverage

---

## 10. QUICK REFERENCE - RUNNING TESTS LOCALLY

### Run Fixture Integrity Tests
```bash
# With fixture feature
cargo test -p bitnet-models --test fixture_integrity_tests \
  --no-default-features --features fixtures

# Validation script
bash scripts/validate-fixtures.sh
```

### Run with Fixtures Feature Matrix
```bash
# QK256 dual-flavor tests
cargo test -p bitnet-models --test qk256_dual_flavor_tests \
  --no-default-features --features cpu,fixtures

# Using nextest (recommended)
cargo nextest run --no-default-features --features cpu,fixtures \
  --profile fixtures
```

### Enable Pre-commit Hook Locally
```bash
# One-time setup
git config core.hooksPath .githooks

# Now runs automatically before commits
git commit -m "Your message"
```

### Regenerate Fixtures
```bash
# Generate new fixtures
cargo test -p bitnet-models --test qk256_dual_flavor_tests \
  --no-default-features --features fixtures test_dump_fixture_for_debug \
  -- --nocapture

# Update checksums
cd ci/fixtures/qk256 && sha256sum *.gguf > SHA256SUMS
```

### Run Individual Guards
```bash
bash scripts/validate-fixtures.sh       # Fixture integrity
bash scripts/check-serial-annotations.sh  # Serial annotations
bash scripts/check-feature-gates.sh     # Feature consistency
bash scripts/check-ignore-annotations.sh  # Ignore justification
```

---

## 11. INTEGRATION POINTS

### 11.1 Where Fixtures Are Loaded

**Test Files Using Fixtures**:
- `crates/bitnet-models/tests/fixture_integrity_tests.rs` - Header validation
- `crates/bitnet-models/tests/qk256_dual_flavor_tests.rs` - Quantization testing
- `crates/bitnet-models/tests/helpers/fixture_loader.rs` - Loading utilities
- `crates/bitnet-quantization/tests/fixture_integration_test.rs` - Format detection

**CI Jobs Using Fixtures**:
- `feature-matrix` - When feature is "fixtures"
- `doctest-matrix` - Documentation examples with fixtures
- `guard-fixture-integrity` - Validation gate

### 11.2 CI Workflow Integration

**Fixture Validation Entry Points**:
1. **Pre-commit**: Local enforcement via `.githooks/pre-commit`
2. **CI Gate**: `guard-fixture-integrity` runs independently
3. **Feature Tests**: `feature-matrix` with `cpu,fixtures` config
4. **Nextest Profile**: Special 2-thread, 10min timeout profile

**Dependency Flow**:
```
Guard jobs (immediate)
    ↓ (independent, no blocking)
test (primary gate)
    ├─→ feature-matrix (with fixtures config)
    ├─→ doctest-matrix
    └─→ ffi-smoke
        ↓
    downstream jobs (crossval, etc.)
```

---

## 12. SPEC AND DOCUMENTATION REFERENCES

**Key Specs**:
- `SPEC-2025-006` - Feature gate validation, guards, fixture contract
- `docs/explanation/i2s-dual-flavor.md` - QK256 vs BitNet32 formats
- `docs/development/test-suite.md` - Testing framework guide
- `ci/CI_DAG_HYGIENE_AND_HOOKS_ANALYSIS.md` - Detailed DAG analysis

**Related Docs**:
- `ci/fixtures/qk256/README.md` - Comprehensive fixture documentation
- `ci/fixtures/qk256/QUICK_REFERENCE.md` - Fixture quick lookup
- `.github/workflows/ci.yml` - Full CI job definitions
- `.config/nextest.toml` - Test execution profiles

**Exploration Docs**:
- `ci/CI_EXPLORATION_SUMMARY.md` - Discovery findings
- `ci/CI_EXPLORATION_INDEX.md` - Navigation guide
- `ci/exploration/fixture_patterns.md` - Fixture usage patterns

---

## Summary Table

| Component | Type | Count | Purpose | Blocking |
|-----------|------|-------|---------|----------|
| GGUF Fixtures | Files | 3 + checksums | Quantization format testing | YES |
| Guard Jobs | CI jobs | 5 | Independent quality checks | 4 YES, 1 observe |
| Feature Configs | Matrix | 6 | Feature combination validation | YES |
| Test Profiles | nextest | 5 | Specialized test execution | YES |
| Ripgrep Patterns | Scripts | 5 | Code pattern validation | Local + CI |
| Pre-commit Checks | Hooks | 2 | Local enforcement | YES |

**Total Blocking Gates**: 9 jobs (test + 8 validators)
**Total Guard Scripts**: 4 (+ 1 inline in CI)
**Total Fixture Files**: 3 GGUF + 1 SHA256SUMS
**Total Test Platforms**: 5 (Linux, Windows, macOS × 2 architectures)

