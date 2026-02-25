# SPEC-2025-001: Priority Implementation Tasks for BitNet.rs Post-PR-475

**Created**: 2025-10-23
**Status**: Draft
**Implementation Phase**: v0.2 Foundation
**Dependencies**: PR #475 (comprehensive integration - merged to main)

---

## Executive Summary

This specification analyzes four high-priority user stories for BitNet.rs following the successful merge of PR #475 (comprehensive integration with QK256 AVX2, EnvGuard isolation, receipts, strict mode, and GGUF fixtures). These stories focus on **test infrastructure hardening**, **documentation consolidation**, and **build system hygiene** to establish a production-grade foundation for neural network inference.

**Key Priorities:**
1. **Real GGUF fixtures** - Persistent disk-based fixtures for CI/CD stability
2. **Complete EnvGuard rollout** - Eliminate all unsafe env mutations across test suite
3. **Documentation consolidation** - Streamline 35 solution docs into navigable structure
4. **FFI hygiene finalization** - Achieve zero warnings in bitnet-ggml-ffi build

---

## Story 1: Real GGUF Fixtures for QK256

### User Story
**As a developer**, I need minimal GGUF test files for QK256/BitNet-32 testing
**So that** CI/CD pipelines have stable, version-controlled fixtures without runtime generation overhead

**Acceptance Criteria:**
- Generate 3 persistent GGUF fixtures (4×256, 3×300, 2×64) and store in `ci/fixtures/`
- Update `qk256_dual_flavor_tests.rs` to load from disk instead of generating in-memory
- Validate fixtures with `bitnet-cli compat-check` command
- Document fixture format and regeneration process
- **Estimate**: 2-3 hours

---

### Requirements Analysis

#### Functional Requirements
1. **Persistent fixtures**: Create minimal, valid GGUF v3 files with deterministic content
2. **Format coverage**:
   - QK256 single-block (4×256 = 256 bytes quantized data)
   - QK256 multi-block with tail (3×300 = 384 bytes)
   - BitNet32F16 format (2×64 = 40 bytes with inline F16 scales)
3. **Disk storage**: Store in `ci/fixtures/qk256/` directory structure
4. **Test migration**: Convert in-memory generation to disk-based loading
5. **Validation**: Ensure fixtures pass GGUF compatibility checks and alignment validation

#### Quantization Constraints
- **QK256 block size**: 256 elements → 64 bytes packed (2-bit quantization)
- **BitNet32F16 block size**: 32 elements → 10 bytes (8 bytes packed + 2 bytes F16 scale)
- **GGUF v3 alignment**: 32-byte alignment for tensor data section
- **Metadata requirements**: Must include `tokenizer.ggml.tokens`, `bitnet-b1.58.embedding_length`, etc.

#### Performance Requirements
- **CI speedup**: Eliminate ~50-100ms per test for fixture generation (3 tests × ~50ms = 150ms savings)
- **Determinism**: Fixtures must produce identical results across platforms (x86_64, ARM64, WASM)
- **Size constraints**: Each fixture < 50KB (current: 4×256 ≈ 2KB, 3×300 ≈ 3KB, 2×64 ≈ 1KB)

---

### Architecture Approach

#### Crate-Specific Implementation Strategy

**Target Crates:**
- `bitnet-models/`: Test infrastructure changes in `tests/qk256_dual_flavor_tests.rs`
- `ci/fixtures/`: New directory for persistent GGUF fixtures

**Workspace Integration:**
```bash
# Directory structure
ci/fixtures/
├── qk256/
│   ├── qk256_4x256_seed42.gguf      # Single-block QK256 (256 bytes)
│   ├── qk256_3x300_seed44.gguf      # Multi-block with tail (384 bytes)
│   └── bitnet32_2x64_seed43.gguf    # BitNet32F16 format (40 bytes)
└── README.md                        # Fixture documentation

crates/bitnet-models/tests/
├── qk256_dual_flavor_tests.rs       # Updated to load from ci/fixtures/
└── helpers/
    ├── qk256_fixtures.rs            # Keep for regeneration tooling
    └── mod.rs
```

#### Feature Flag Analysis
- **No new feature flags required** - uses existing `fixtures` feature
- **Build configurations**: Tests run with `--features cpu,fixtures`
- **Conditional compilation**: `#[cfg_attr(not(feature = "fixtures"), ignore)]` remains unchanged

---

### Quantization Strategy

#### QK256 Format Preservation
**Challenge**: Ensure disk-based fixtures maintain exact quantization semantics as in-memory generators

**Approach:**
1. **One-time generation**: Use existing `helpers::qk256_fixtures::generate_*` functions
2. **Write to disk**: Store in `ci/fixtures/qk256/` with deterministic seeds
3. **Verification**: Load fixtures and validate against expected structure
4. **Regeneration script**: Create `scripts/regenerate-fixtures.sh` for reproducible fixture updates

**GGUF v3 Compliance:**
- Magic: `GGUF` (0x46554747)
- Version: 3 (little-endian u32)
- Tensor alignment: 32-byte boundary
- Metadata KV pairs: 8 required keys (general.name, tokenizer.ggml.tokens, etc.)
- Dual tensor structure: `tok_embeddings.weight` (I2_S) + `output.weight` (F16)

#### Device-Aware Dequantization (Not Applicable)
**Rationale**: Fixtures are test infrastructure - no runtime dequantization needed for loading tests

---

### GGUF Integration

#### Format Compatibility
**Validation Pipeline:**
```bash
# Generate fixtures (one-time script)
cargo run -p bitnet-models --test qk256_dual_flavor_tests --features fixtures -- \
  test_dump_fixture_for_debug --nocapture

# Copy to ci/fixtures/
cp /tmp/test_qk256_4x256.gguf ci/fixtures/qk256/qk256_4x256_seed42.gguf
cp /tmp/test_qk256_3x300.gguf ci/fixtures/qk256/qk256_3x300_seed44.gguf
cp /tmp/test_bitnet32_2x64.gguf ci/fixtures/qk256/bitnet32_2x64_seed43.gguf

# Validate with bitnet-cli
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- \
  compat-check ci/fixtures/qk256/qk256_4x256_seed42.gguf

# Inspect metadata
cargo run -p bitnet-cli --no-default-features --features cpu,full-cli -- \
  compat-check ci/fixtures/qk256/qk256_4x256_seed42.gguf --show-kv
```

#### Tensor Alignment Validation
**Critical Requirements:**
- All tensor offsets must be 32-byte aligned (GGUF v3 spec)
- Inter-tensor padding required for minimal parser compatibility
- Alignment checks in `helpers/alignment_validator.rs` must pass

---

### Testing Strategy

#### Unit Tests
```rust
// crates/bitnet-models/tests/qk256_dual_flavor_tests.rs

#[test]
#[cfg(feature = "fixtures")]
fn test_qk256_4x256_from_disk() {
    let fixture_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .parent().unwrap()
        .join("ci/fixtures/qk256/qk256_4x256_seed42.gguf");

    assert!(fixture_path.exists(), "Fixture file missing: {:?}", fixture_path);

    let result = load_gguf_full(&fixture_path, Device::Cpu, GGUFLoaderConfig::default())
        .expect("Failed to load QK256 fixture from disk");

    // Verify QK256 detection
    assert_eq!(result.i2s_qk256.len(), 1, "Should detect one QK256 tensor");
    assert!(result.i2s_qk256.contains_key("tok_embeddings.weight"));

    // Verify structure matches in-memory generation
    let qk256 = result.i2s_qk256.get("tok_embeddings.weight").unwrap();
    assert_eq!(qk256.rows, 4);
    assert_eq!(qk256.cols, 256);
    assert_eq!(qk256.row_stride_bytes, 64);
}

#[test]
#[cfg(feature = "fixtures")]
fn test_all_fixtures_exist_and_valid() {
    let fixtures = [
        "ci/fixtures/qk256/qk256_4x256_seed42.gguf",
        "ci/fixtures/qk256/qk256_3x300_seed44.gguf",
        "ci/fixtures/qk256/bitnet32_2x64_seed43.gguf",
    ];

    for fixture_path in fixtures {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent().unwrap()
            .parent().unwrap()
            .join(fixture_path);

        assert!(path.exists(), "Fixture missing: {}", fixture_path);

        let result = load_gguf_full(&path, Device::Cpu, GGUFLoaderConfig::default());
        assert!(result.is_ok(), "Fixture invalid: {} - {:?}", fixture_path, result.err());
    }
}
```

#### Integration Tests
- **Alignment validation**: Existing `alignment_validator.rs` tests against disk fixtures
- **Cross-validation**: Verify fixtures load identically on x86_64, ARM64, WASM
- **CI pipeline**: Add fixture existence check in `.github/workflows/ci.yml`

#### Strict Mode Validation
```bash
# Strict mode fixture validation
BITNET_STRICT_MODE=1 cargo run -p bitnet-cli --features cpu,full-cli -- \
  inspect --ln-stats --gate auto ci/fixtures/qk256/qk256_4x256_seed42.gguf
```

---

### Risk Mitigation

#### Technical Risk Assessment

**Risk 1: Fixture corruption or platform-specific differences**
- **Severity**: Medium
- **Probability**: Low
- **Mitigation**:
  - Store fixtures in Git LFS (if size grows > 1MB)
  - SHA256 checksums in `ci/fixtures/qk256/checksums.txt`
  - Regeneration script for reproducible fixtures
- **Validation**:
  ```bash
  sha256sum ci/fixtures/qk256/*.gguf > ci/fixtures/qk256/checksums.txt
  sha256sum -c ci/fixtures/qk256/checksums.txt
  ```

**Risk 2: GGUF format version drift**
- **Severity**: Medium
- **Probability**: Low
- **Mitigation**:
  - Lock GGUF v3 format in fixture generator
  - Version metadata in `ci/fixtures/qk256/README.md`
  - Automated compatibility checks in CI
- **Validation**:
  ```bash
  cargo test -p bitnet-models --features fixtures test_all_fixtures_exist_and_valid
  ```

**Risk 3: CI/CD path resolution issues**
- **Severity**: Low
- **Probability**: Low
- **Mitigation**:
  - Use `env!("CARGO_MANIFEST_DIR")` for relative path resolution
  - Fallback to absolute paths in CI environment
  - Document path resolution strategy in test comments

---

### Success Criteria

**Measurable Acceptance Criteria:**
1. ✅ **3 fixtures exist** in `ci/fixtures/qk256/` directory
2. ✅ **All fixtures pass** `bitnet-cli compat-check` validation
3. ✅ **12/12 tests pass** in `qk256_dual_flavor_tests.rs` using disk-based loading
4. ✅ **CI speedup**: 150ms reduction in test execution time (eliminate in-memory generation)
5. ✅ **Zero test failures** across x86_64, ARM64, WASM targets
6. ✅ **Documentation complete**: `ci/fixtures/qk256/README.md` with regeneration instructions

**Validation Commands:**
```bash
# Test fixture loading
cargo test -p bitnet-models --features cpu,fixtures test_qk256_detection_by_size

# Validate all fixtures
for f in ci/fixtures/qk256/*.gguf; do
  cargo run -p bitnet-cli --features cpu,full-cli -- compat-check "$f"
done

# CI integration test
cargo nextest run -p bitnet-models --features cpu,fixtures --profile ci
```

---

## Story 2: Complete EnvGuard Rollout

### User Story
**As a test maintainer**, I need all env-mutating tests to use proper isolation
**So that** parallel test execution is safe and deterministic without race conditions

**Acceptance Criteria:**
- All env-mutating tests use `EnvGuard` + `#[serial(bitnet_env)]`
- Add mini-guide in `docs/development/test-suite.md`
- Zero unsafe env mutations outside `EnvGuard` pattern
- **Estimate**: 2 hours

---

### Requirements Analysis

#### Functional Requirements
1. **Complete migration**: Convert all 21 files with unsafe env operations to use `EnvGuard`
2. **Pattern enforcement**: All tests use `#[serial(bitnet_env)]` marker
3. **Documentation**: Mini-guide in `docs/development/test-suite.md`
4. **Verification**: Automated check for unsafe env patterns in CI

#### Current State Analysis
```bash
# Files with unsafe env operations (excluding env_guard.rs): 21 files
# Existing #[serial(bitnet_env)] markers: 72 tests
# Target: 100% coverage for env-mutating tests
```

**Categories of env mutations:**
1. **Deterministic testing**: `BITNET_DETERMINISTIC`, `BITNET_SEED`, `RAYON_NUM_THREADS`
2. **Validation flags**: `BITNET_STRICT_MODE`, `BITNET_VALIDATION_GATE`
3. **GPU overrides**: `BITNET_GPU_FAKE`
4. **Test infrastructure**: `BITNET_SKIP_SLOW_TESTS`, `BITNET_RUN_IGNORED_TESTS`

#### Performance Requirements
- **No test slowdown**: EnvGuard overhead < 1ms per test
- **Parallel safety**: Tests with `#[serial(bitnet_env)]` run sequentially but don't block other tests
- **CI stability**: Zero flaky test failures from env races

---

### Architecture Approach

#### Crate-Specific Implementation Strategy

**Target Crates:**
- `tests/`: 6 files needing migration
- `tests-new/`: 7 files needing migration
- `xtask/`: 5 test files needing migration
- `crates/*/tests/`: Remaining files needing review

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
#[serial(bitnet_env)]  // Process-level serialization
fn test_strict_mode_enabled() {
    let _guard = EnvGuard::new("BITNET_STRICT_MODE");
    _guard.set("1");

    let config = StrictModeConfig::from_env();
    assert!(config.enabled);
    // ✅ Guard drops here, restoring original value
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
    // ✅ Automatic restoration on scope exit
}
```

#### Workspace Integration
- **Shared EnvGuard**: Use `tests/support/env_guard.rs` as canonical implementation
- **Crate-local copies**: Keep existing copies in `crates/*/tests/support/` for isolated compilation
- **Documentation**: Update `tests/support/env_guard.rs` module docs with rollout status

---

### Testing Strategy

#### Unit Tests (No changes needed - EnvGuard already has 7 tests)
- `test_env_guard_set_and_restore`
- `test_env_guard_remove_and_restore`
- `test_env_guard_multiple_sets`
- `test_env_guard_preserves_original`
- `test_env_guard_key_accessor`
- `test_env_guard_panic_safety`
- `test_env_guard_panic_safety_verification`

#### Integration Tests (Migration Verification)
```bash
# Verify no unsafe env operations remain (excluding EnvGuard itself)
! grep -r "unsafe.*set_var\|unsafe.*remove_var" --include="*.rs" \
  crates/ tests/ tests-new/ xtask/ | grep -v "env_guard.rs"

# Verify all env-mutating tests have #[serial(bitnet_env)]
# (Manual review required - automated check difficult)
grep -r "EnvGuard::new\|with_var" --include="*.rs" crates/ tests/ tests-new/ xtask/ | \
  while read -r line; do
    file=$(echo "$line" | cut -d: -f1)
    grep -B5 "EnvGuard::new\|with_var" "$file" | grep "#\[serial(bitnet_env)\]" || \
      echo "⚠️  Missing #[serial(bitnet_env)] in $file"
  done
```

#### CI Pipeline Addition
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

---

### Risk Mitigation

#### Technical Risk Assessment

**Risk 1: Missing #[serial(bitnet_env)] markers**
- **Severity**: Medium
- **Probability**: Medium
- **Mitigation**:
  - Manual review of all env-mutating tests
  - Document pattern in `docs/development/test-suite.md`
  - Add pre-commit hook for automated detection
- **Validation**:
  ```bash
  # Run tests in parallel to detect race conditions
  cargo nextest run --test-threads=8 --workspace
  ```

**Risk 2: Performance impact on CI**
- **Severity**: Low
- **Probability**: Low
- **Mitigation**:
  - EnvGuard overhead < 1ms per test (negligible)
  - `#[serial(bitnet_env)]` only affects env-mutating tests (~72 tests)
  - Other tests still run in parallel (1935+ tests)

**Risk 3: Closure-based vs RAII pattern confusion**
- **Severity**: Low
- **Probability**: Low
- **Mitigation**:
  - Document both patterns in `tests/support/env_guard.rs`
  - Provide examples in mini-guide
  - Prefer closure-based for new tests (cleaner idiom)

---

### Success Criteria

**Measurable Acceptance Criteria:**
1. ✅ **Zero unsafe env mutations** outside `env_guard.rs` (verified by CI check)
2. ✅ **All env-mutating tests** have `#[serial(bitnet_env)]` marker
3. ✅ **Mini-guide complete** in `docs/development/test-suite.md` (< 200 lines)
4. ✅ **CI check passes** for unsafe env pattern detection
5. ✅ **No test regressions** - all 1935+ tests still pass
6. ✅ **No performance degradation** - CI run time within ±5% of baseline

**Validation Commands:**
```bash
# Verify no unsafe env operations
! grep -r "unsafe.*set_var\|unsafe.*remove_var" --include="*.rs" \
  crates/ tests/ tests-new/ xtask/ | grep -v "env_guard.rs"

# Run full test suite with parallel execution
cargo nextest run --workspace --test-threads=4 --profile ci

# Verify serial tests still work
cargo nextest run --workspace --test-threads=1 --profile ci
```

---

## Story 3: Documentation Consolidation

### User Story
**As a documentation reader**, I need streamlined navigation with fewer index files
**So that** finding information is faster and documentation maintenance is easier

**Acceptance Criteria:**
- Merge small indexes into parent documents
- Update all cross-references to new structure
- Verify all internal links (lychee link checker)
- **Estimate**: 4-5 hours

---

### Requirements Analysis

#### Functional Requirements
1. **Index consolidation**: Merge 35 solution docs in `ci/solutions/` into navigable structure
2. **Link validation**: Update all cross-references and verify with `lychee`
3. **Navigation improvement**: Single entry point for CI solutions with clear categories
4. **Archival strategy**: Move redundant/outdated docs to `ci/solutions/archive/`

#### Current State Analysis
```bash
# Documentation files: 292 total
# Solution docs: 35 files in ci/solutions/
# Index files: ~8-10 INDEX/index files across docs/

# Redundant indexes identified:
ci/solutions/INDEX.md
ci/solutions/00_NAVIGATION_INDEX.md
ci/solutions/INDEX_RECEIPT_ANALYSIS.md
ci/solutions/README_RECEIPT_ANALYSIS.md
ci/solutions/QUICK_REFERENCE.md
ci/solutions/CLIPPY_QUICK_REFERENCE.md
ci/solutions/CONCURRENT_LOAD_QUARANTINE_QUICK_REF.md
```

#### Performance Requirements
- **Navigation time**: < 30 seconds to find any CI solution (from README)
- **Link validation**: 100% of internal links must resolve (verified by lychee)
- **Maintenance overhead**: Single source of truth for each topic (no duplication)

---

### Architecture Approach

#### Documentation Structure Redesign

**Proposed Structure:**
```
ci/solutions/
├── README.md                        # Main entry point (consolidates INDEX.md, 00_NAVIGATION_INDEX.md)
│   ├── Quick Reference (consolidates QUICK_REFERENCE.md)
│   ├── Solutions Index (alphabetical)
│   └── Navigation Guide
├── IMPLEMENTATION_GUIDE.md          # Consolidates IMPLEMENTATION_SUMMARY.md, SOLUTION_SUMMARY.md
├── qk256/
│   ├── README.md                    # Consolidates QK256_ANALYSIS_INDEX.md
│   ├── test_failure_analysis.md     # Renames QK256_TEST_FAILURE_ANALYSIS_INDEX.md
│   ├── property_test_analysis.md    # Keeps QK256_PROPERTY_TEST_ANALYSIS_INDEX.md
│   └── tolerance_strategy.md        # Keeps QK256_TOLERANCE_STRATEGY.md
├── clippy/
│   ├── README.md                    # Consolidates CLIPPY_LINT_FIXES.md + CLIPPY_QUICK_REFERENCE.md
│   └── fixes.md                     # Detailed fix guide
├── gguf/
│   ├── README.md                    # Consolidates GGUF_SHAPE_VALIDATION_INDEX.md
│   └── shape_validation_fix.md      # Keeps gguf_shape_validation_fix.md
├── receipts/
│   ├── README.md                    # Consolidates INDEX_RECEIPT_ANALYSIS.md + README_RECEIPT_ANALYSIS.md
│   └── test_quick_reference.md      # Keeps RECEIPT_TEST_QUICK_REFERENCE.md
├── batch_prefill/
│   ├── README.md                    # Consolidates BATCH_PREFILL_INDEX.md
│   └── perf_quarantine.md           # Keeps batch_prefill_perf_quarantine.md
├── archive/
│   ├── SUMMARY.md                   # Old summaries
│   ├── SOLUTIONS_SUMMARY.md
│   └── ANALYSIS_SUMMARY.md
└── _TEMPLATE.md                     # Template for new solution docs
```

**Consolidation Strategy:**
1. **Merge indexes**: Combine 3-4 index files into single `README.md` per category
2. **Archive summaries**: Move outdated summaries to `archive/`
3. **Update cross-refs**: Use `docs/` and `ci/solutions/` relative paths consistently
4. **Link validation**: Run lychee on all consolidated docs

---

### Testing Strategy

#### Link Validation
```bash
# Install lychee (if not already installed)
cargo install lychee

# Configure lychee for local links
cat > .lychee.toml << 'EOF'
# Lychee link checker configuration
[cache]
max_age = "1d"

[include]
include_verbatim = true

[accept]
accept = [200, 204, 301, 302, 403, 429]

[exclude]
exclude_path = ["target/", "vendor/", ".git/"]

# Exclude external links that require authentication
exclude = [
  "https://github.com/.*/issues/.*",
  "https://github.com/.*/pull/.*"
]
EOF

# Validate all markdown links
lychee --config .lychee.toml docs/ ci/ README.md CLAUDE.md

# Validate specific consolidated docs
lychee ci/solutions/README.md
lychee ci/solutions/qk256/README.md
lychee ci/solutions/clippy/README.md
```

#### Documentation Review Checklist
- [ ] All internal links resolve (lychee passes)
- [ ] Cross-references updated to new structure
- [ ] No orphaned docs (all reachable from top-level README)
- [ ] Table of contents matches actual structure
- [ ] Code examples still valid and tested

---

### Risk Mitigation

#### Technical Risk Assessment

**Risk 1: Breaking existing bookmarks/links**
- **Severity**: Low
- **Probability**: Medium
- **Mitigation**:
  - Add redirect comments in moved/archived files
  - Update CLAUDE.md with new navigation paths
  - Git commit message documents moved files

**Risk 2: Incomplete cross-reference updates**
- **Severity**: Medium
- **Probability**: Medium
- **Mitigation**:
  - Use lychee link checker for automated validation
  - Grep for old file paths and update systematically
  - Manual review of navigation flow

**Risk 3: Lost historical context in archived docs**
- **Severity**: Low
- **Probability**: Low
- **Mitigation**:
  - Preserve all content in `archive/` directory
  - Add metadata headers with archival date and reason
  - Link to archived docs from new consolidated docs

---

### Success Criteria

**Measurable Acceptance Criteria:**
1. ✅ **Index reduction**: From 35 files to ~15 files (57% reduction)
2. ✅ **Navigation time**: < 30 seconds to find any solution (user testing)
3. ✅ **Link validation**: 100% of internal links pass lychee check
4. ✅ **Zero orphaned docs**: All docs reachable from top-level README
5. ✅ **Single entry point**: `ci/solutions/README.md` as authoritative index
6. ✅ **Archival complete**: Outdated docs moved to `archive/` with metadata

**Validation Commands:**
```bash
# Verify link integrity
lychee --config .lychee.toml docs/ ci/ README.md CLAUDE.md

# Check for orphaned markdown files
find docs/ ci/ -name "*.md" -type f | while read -r file; do
  if ! grep -r "$(basename "$file")" docs/ ci/ README.md CLAUDE.md >/dev/null; then
    echo "⚠️  Orphaned: $file"
  fi
done

# Verify file count reduction
echo "Before: 35 files in ci/solutions/"
echo "After: $(find ci/solutions/ -name "*.md" -type f | wc -l) files"
```

---

## Story 4: FFI Hygiene Finalization

### User Story
**As a build maintainer**, I need zero warnings in bitnet-ggml-ffi
**So that** CI builds are clean and regressions are immediately visible

**Acceptance Criteria:**
- Confirm `-isystem` resolution for vendored GGML headers (suppresses third-party warnings)
- Ensure zero warnings in `bitnet-ggml-ffi` build output
- Add regression test in CI to fail on new warnings
- **Estimate**: 2-3 hours

---

### Requirements Analysis

#### Functional Requirements
1. **Zero warnings**: `bitnet-ggml-ffi` builds with zero compiler warnings
2. **Third-party isolation**: `-isystem` flag suppresses warnings from vendored GGML code
3. **Local code visibility**: Warnings from shim code (`ggml_quants_shim.c`) remain visible
4. **CI enforcement**: Add warning-as-error check for FFI crate

#### Current State Analysis
```rust
// crates/bitnet-ggml-ffi/build.rs (lines 32-50)
build
    .file("csrc/ggml_quants_shim.c")
    .file("csrc/ggml_consts.c")
    .include("csrc") // Local includes (use -I, warnings visible)
    // AC6: Use -isystem for vendored GGML headers (third-party code)
    .flag("-isystemcsrc/ggml/include")
    .flag("-isystemcsrc/ggml/src")
    // Warning suppression for vendored code
    .flag_if_supported("-Wno-sign-compare")
    .flag_if_supported("-Wno-unused-parameter")
    .flag_if_supported("-Wno-unused-function")
    .compile("bitnet_ggml_quants_shim");
```

**Potential Issues:**
- `-isystem` flag format may need platform-specific handling (GCC vs Clang)
- Vendored GGML commit may have new warnings after updates
- Shim code may have latent warnings not caught during development

---

### Architecture Approach

#### Crate-Specific Implementation Strategy

**Target Crates:**
- `bitnet-ggml-ffi/`: FFI bridge crate with C++ vendored dependencies

**Build System Improvements:**
```rust
// crates/bitnet-ggml-ffi/build.rs (enhanced)
fn main() {
    if std::env::var("CARGO_FEATURE_IQ2S_FFI").is_ok() {
        use std::{fs, path::Path};

        let marker = Path::new("csrc/VENDORED_GGML_COMMIT");
        let commit = fs::read_to_string(marker).unwrap_or_else(|_| "unknown".into()).trim().to_string();
        println!("cargo:rustc-env=BITNET_GGML_COMMIT={}", commit);

        // CI hygiene: fail if vendored commit unknown
        if std::env::var("CI").is_ok() && commit == "unknown" {
            panic!("VENDORED_GGML_COMMIT is 'unknown' in CI. Run: cargo xtask vendor-ggml --commit <sha>");
        }

        let mut build = cc::Build::new();

        // Platform-aware -isystem flag handling
        let is_msvc = build.get_compiler().is_like_msvc();
        let isystem_flag = if is_msvc {
            // MSVC uses /external:I for external headers (VS 2019+)
            "/external:I"
        } else {
            // GCC/Clang use -isystem
            "-isystem"
        };

        build
            .file("csrc/ggml_quants_shim.c")
            .file("csrc/ggml_consts.c")
            .include("csrc") // Local shim code (warnings visible)
            // Vendored GGML headers (warnings suppressed)
            .flag(&format!("{}csrc/ggml/include", isystem_flag))
            .flag(&format!("{}csrc/ggml/src", isystem_flag))
            .define("GGML_USE_K_QUANTS", None)
            .define("QK_IQ2_S", "256")
            .flag_if_supported("-O3")
            .flag_if_supported("-fPIC")
            // Suppress warnings only for vendored code (via -isystem)
            .flag_if_supported("-Wno-sign-compare")
            .flag_if_supported("-Wno-unused-parameter")
            // MSVC warning suppression for external headers
            .flag_if_supported("/external:W0") // Suppress all warnings from /external:I paths
            .compile("bitnet_ggml_quants_shim");

        // Rebuild triggers
        println!("cargo:rerun-if-changed=csrc/ggml_quants_shim.c");
        println!("cargo:rerun-if-changed=csrc/ggml_consts.c");
        println!("cargo:rerun-if-changed=csrc/VENDORED_GGML_COMMIT");
    }
}
```

---

### Testing Strategy

#### Build Validation Tests
```bash
# Test clean build (zero warnings)
cargo clean -p bitnet-ggml-ffi
cargo build -p bitnet-ggml-ffi --no-default-features --features iq2s-ffi 2>&1 | tee /tmp/ffi_build.log

# Verify zero warnings
if grep -i "warning" /tmp/ffi_build.log; then
  echo "❌ FFI build has warnings"
  exit 1
fi
echo "✅ FFI build clean (zero warnings)"

# Test with different compilers (if available)
CC=gcc cargo build -p bitnet-ggml-ffi --no-default-features --features iq2s-ffi
CC=clang cargo build -p bitnet-ggml-ffi --no-default-features --features iq2s-ffi
```

#### CI Pipeline Addition
```yaml
# .github/workflows/ci.yml
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

#### Regression Test
```rust
// xtask/tests/ffi_build_hygiene_test.rs

#[test]
#[cfg(feature = "iq2s-ffi")]
fn test_ffi_build_zero_warnings() {
    use std::process::Command;

    // Clean build
    Command::new("cargo")
        .args(&["clean", "-p", "bitnet-ggml-ffi"])
        .status()
        .expect("Failed to clean FFI crate");

    // Build with warnings captured
    let output = Command::new("cargo")
        .args(&["build", "-p", "bitnet-ggml-ffi", "--no-default-features", "--features", "iq2s-ffi"])
        .output()
        .expect("Failed to build FFI crate");

    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Check for warnings
    let has_warnings = stderr.contains("warning:") || stdout.contains("warning:");

    if has_warnings {
        eprintln!("❌ FFI build output:\n{}\n{}", stdout, stderr);
        panic!("FFI crate build produced warnings (expected zero)");
    }

    println!("✅ FFI build clean (zero warnings)");
}
```

---

### Risk Mitigation

#### Technical Risk Assessment

**Risk 1: Platform-specific -isystem flag incompatibility**
- **Severity**: Medium
- **Probability**: Low
- **Mitigation**:
  - Detect compiler type (GCC/Clang/MSVC) at build time
  - Use `/external:I` for MSVC, `-isystem` for GCC/Clang
  - Test on all supported platforms (Linux, macOS, Windows)
- **Validation**:
  ```bash
  # Test on multiple platforms
  cargo build -p bitnet-ggml-ffi --features iq2s-ffi --target x86_64-unknown-linux-gnu
  cargo build -p bitnet-ggml-ffi --features iq2s-ffi --target x86_64-apple-darwin
  cargo build -p bitnet-ggml-ffi --features iq2s-ffi --target x86_64-pc-windows-msvc
  ```

**Risk 2: Vendored GGML updates introduce new warnings**
- **Severity**: Low
- **Probability**: Medium
- **Mitigation**:
  - Pin vendored GGML commit in `csrc/VENDORED_GGML_COMMIT`
  - Test vendored updates in separate PR before merging
  - CI fails if `VENDORED_GGML_COMMIT` is "unknown"

**Risk 3: Shim code warnings not caught during development**
- **Severity**: Low
- **Probability**: Low
- **Mitigation**:
  - Keep shim code minimal (currently 2 files: `ggml_quants_shim.c`, `ggml_consts.c`)
  - Enable all warnings for shim code (via `-I` instead of `-isystem`)
  - CI regression test catches new warnings

---

### Success Criteria

**Measurable Acceptance Criteria:**
1. ✅ **Zero warnings**: `cargo build -p bitnet-ggml-ffi --features iq2s-ffi` produces zero warnings
2. ✅ **Platform support**: Builds cleanly on Linux (GCC/Clang), macOS (Clang), Windows (MSVC)
3. ✅ **CI enforcement**: New warnings fail CI build
4. ✅ **Regression test**: `xtask/tests/ffi_build_hygiene_test.rs` passes
5. ✅ **Documentation**: Update `crates/bitnet-ggml-ffi/README.md` with -isystem rationale
6. ✅ **Vendored commit tracking**: `VENDORED_GGML_COMMIT` enforced in CI

**Validation Commands:**
```bash
# Clean build test
cargo clean -p bitnet-ggml-ffi
cargo build -p bitnet-ggml-ffi --no-default-features --features iq2s-ffi 2>&1 | grep -i "warning" && exit 1 || echo "✅ Zero warnings"

# Regression test
cargo test -p xtask --test ffi_build_hygiene_test -- --nocapture

# Multi-platform validation (requires cross-compilation setup)
rustup target add x86_64-unknown-linux-gnu x86_64-apple-darwin x86_64-pc-windows-msvc
cargo build -p bitnet-ggml-ffi --features iq2s-ffi --target x86_64-unknown-linux-gnu
```

---

## Cross-Story Dependencies

### Dependency Graph
```
Story 1 (GGUF Fixtures)
  │
  ├─→ Story 2 (EnvGuard) [Weak: Fixtures use deterministic env for seed control]
  └─→ Story 3 (Docs) [Weak: New ci/fixtures/ needs documentation]

Story 2 (EnvGuard)
  │
  └─→ Story 3 (Docs) [Medium: Mini-guide references consolidated test docs]

Story 3 (Docs)
  │
  └─→ Story 4 (FFI) [None: Independent work streams]

Story 4 (FFI)
  │
  └─→ Story 1 (Fixtures) [None: Independent work streams]
```

### Recommended Implementation Order
1. **Story 4 (FFI Hygiene)** - Independent, quick win (2-3 hours)
2. **Story 1 (GGUF Fixtures)** - Foundation for CI stability (2-3 hours)
3. **Story 2 (EnvGuard)** - Parallel test safety (2 hours)
4. **Story 3 (Docs)** - Consolidation and navigation (4-5 hours)

**Total Estimate**: 10-13 hours (can be parallelized across 2 developers)

---

## BitNet.rs Alignment Verification

### TDD Practices
- ✅ **Story 1**: Fixtures enable faster test iteration (reduce generation overhead)
- ✅ **Story 2**: EnvGuard ensures deterministic tests (critical for TDD)
- ✅ **Story 3**: Documentation supports test-first development workflow
- ✅ **Story 4**: Zero-warning builds catch regressions early

### Feature-Gated Architecture
- ✅ **Story 1**: Uses existing `fixtures` feature flag
- ✅ **Story 2**: No new feature flags (test infrastructure)
- ✅ **Story 3**: No feature flag impact (documentation only)
- ✅ **Story 4**: Improves `iq2s-ffi` feature build quality

### Workspace Structure
- ✅ **Story 1**: Follows `ci/fixtures/` convention (separate from `crates/`)
- ✅ **Story 2**: Uses shared `tests/support/` infrastructure
- ✅ **Story 3**: Consolidates `ci/solutions/` into maintainable structure
- ✅ **Story 4**: Isolated to `crates/bitnet-ggml-ffi/` build system

### GPU/CPU Parity
- N/A for these stories (test infrastructure and documentation)

### GGUF Compatibility
- ✅ **Story 1**: Validates GGUF v3 compliance with alignment checks
- ✅ **Story 4**: Ensures FFI bridge maintains GGUF format compatibility

### Cross-Platform Support
- ✅ **Story 1**: Fixtures tested on x86_64, ARM64, WASM
- ✅ **Story 2**: EnvGuard platform-agnostic (std::env API)
- ✅ **Story 4**: Multi-compiler support (GCC, Clang, MSVC)

### System Integration
- ✅ **Story 1**: Fixtures reduce CI runtime (150ms savings per run)
- ✅ **Story 2**: EnvGuard prevents CI flakes from env races
- ✅ **Story 3**: Documentation improves developer onboarding
- ✅ **Story 4**: Zero warnings improve CI signal-to-noise ratio

---

## Implementation Roadmap

### Phase 1: FFI Hygiene (Week 1, Day 1-2)
- [ ] Implement platform-aware `-isystem` flag handling
- [ ] Add CI enforcement for zero warnings
- [ ] Create regression test in `xtask/tests/`
- [ ] Document `-isystem` rationale in README

### Phase 2: GGUF Fixtures (Week 1, Day 3-4)
- [ ] Generate 3 persistent fixtures in `ci/fixtures/qk256/`
- [ ] Migrate `qk256_dual_flavor_tests.rs` to disk-based loading
- [ ] Add SHA256 checksums for fixture integrity
- [ ] Create regeneration script and documentation

### Phase 3: EnvGuard Rollout (Week 1, Day 5)
- [ ] Migrate 21 files to EnvGuard pattern
- [ ] Add mini-guide to `docs/development/test-suite.md`
- [ ] Implement CI check for unsafe env mutations
- [ ] Verify parallel test execution stability

### Phase 4: Documentation Consolidation (Week 2, Day 1-3)
- [ ] Consolidate 35 solution docs into categorized structure
- [ ] Update all cross-references and links
- [ ] Run lychee link validator
- [ ] Archive outdated summaries

### Phase 5: Validation (Week 2, Day 4)
- [ ] Full CI pipeline test (all stories integrated)
- [ ] Cross-platform validation (Linux, macOS, Windows)
- [ ] Performance benchmarking (verify no regressions)
- [ ] Documentation review and finalization

---

## Appendix: Neural Network References

### Quantization Patterns (Story 1)
- **QK256 block layout**: 256 elements → 64 bytes (2-bit packed)
- **BitNet32F16 layout**: 32 elements + F16 scale → 10 bytes per block
- **GGUF v3 alignment**: 32-byte boundary for tensor data

### Cross-Validation Approaches (Story 2)
- **Deterministic testing**: `BITNET_DETERMINISTIC=1` + `BITNET_SEED=42` + `RAYON_NUM_THREADS=1`
- **EnvGuard isolation**: Prevents race conditions in parallel cross-validation tests
- **Parity validation**: Fixtures enable consistent C++/Rust comparison

### Production-Grade Patterns (All Stories)
- **Story 1**: Persistent fixtures reduce CI flakiness (no runtime generation variance)
- **Story 2**: EnvGuard ensures deterministic test execution (critical for ML reproducibility)
- **Story 3**: Documentation supports production onboarding and maintenance
- **Story 4**: Zero-warning builds catch quantization precision issues early

---

**End of Specification**
