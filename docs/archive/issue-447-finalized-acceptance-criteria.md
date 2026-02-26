# Issue #447: Finalized Acceptance Criteria

**Issue**: https://github.com/EffortlessMetrics/BitNet-rs/issues/447
**Date**: 2025-10-11
**Status**: Issue Ledger Validated - Ready for Spec Creation
**Finalized By**: issue-finalizer (generative flow microloop 1.3/8)

---

## Validation Summary

**Result**: ✅ PASS - All acceptance criteria validated, corrected, and ready for implementation

**Key Findings**:
1. **AC7 Critical Correction**: Removed invalid `reporting.fail_fast` reference (field does not exist in new TestConfig API)
2. **AC6 Status Update**: Changed from implementation to verification-only (module already declared)
3. **AC2 Environment Variable Clarification**: Specified default localhost fallback for OTLP endpoint

---

## Corrected Acceptance Criteria

### P0: OpenTelemetry Migration (bitnet-server)

#### AC1: Remove deprecated Prometheus exporter dependency and migrate to OTLP
**Requirements**:
- Remove `opentelemetry-prometheus@0.29.1` from workspace dependencies
- Update `opentelemetry-otlp` to include `metrics` feature in workspace config
- Delete incompatible PrometheusExporter integration code

**Validation Command**:
```bash
cargo check -p bitnet-server --no-default-features --features opentelemetry
```

**Expected Output**: Compilation succeeds with no errors

**Test Tag**: `// AC1: OTLP dependency migration`

**Evidence**: Workspace `Cargo.toml` line 233 shows `opentelemetry-prometheus@0.29.1` incompatible with `opentelemetry-sdk@0.31.0`

---

#### AC2: Implement OTLP metrics initialization with localhost fallback
**Requirements**:
- Support `OTEL_EXPORTER_OTLP_ENDPOINT` env var with default `http://127.0.0.1:4317`
- Implement `MetricsExporter` with `PeriodicReader` (60s interval)
- Preserve existing metric instrumentation points:
  - `record_inference_metrics()` (request counts, latencies)
  - `record_model_load_metrics()` (model loading operations)
  - `record_quantization_metrics()` (quantization operations)

**Validation Command**:
```bash
cargo build -p bitnet-server --no-default-features --features opentelemetry --release
```

**Expected Output**: Release build succeeds with optimized OTLP integration

**Test Tag**: `// AC2: OTLP metrics with env config`

**Code Pattern**:
```rust
let endpoint = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT")
    .unwrap_or_else(|_| "http://127.0.0.1:4317".to_string());
let exporter = MetricsExporter::builder()
    .with_tonic()
    .with_endpoint(endpoint)
    .with_timeout(Duration::from_secs(3))
    .build()?;
let reader = opentelemetry_sdk::metrics::PeriodicReader::builder(exporter)
    .with_interval(Duration::from_secs(60))
    .build();
```

**Evidence**: Spec lines 91-111 show OTLP migration pattern with environment variable support

---

#### AC3: Remove Prometheus code paths and verify clean compilation
**Requirements**:
- Delete `PrometheusExporter` initialization and type conversions
- Remove deprecated `opentelemetry_prometheus::exporter()` usage
- Ensure 0 clippy warnings in observability code

**Validation Command**:
```bash
cargo clippy -p bitnet-server --no-default-features --features opentelemetry -- -D warnings
```

**Expected Output**: No warnings, clean compilation with OTLP-only code

**Test Tag**: `// AC3: Clean OTLP-only compilation`

**Evidence**: Error output shows `PrometheusExporter` trait bound failure at `src/monitoring/opentelemetry.rs:92`

---

### P2: Inference Full-Engine Tests (bitnet-inference)

#### AC4: Export production engine types for full-engine feature tests
**Requirements**:
- Export `EngineConfig` and `ProductionInferenceEngine` in `bitnet-inference/src/lib.rs` public API
- Add missing imports to test modules:
  - `use std::env;`
  - `use anyhow::Context;`

**Validation Command**:
```bash
cargo check -p bitnet-inference --all-features
```

**Expected Output**: All features compile with exported types accessible

**Test Tag**: `// AC4: Export engine types for tests`

**Code Pattern**:
```rust
// In bitnet-inference/src/lib.rs
pub use crate::config::EngineConfig;
pub use crate::engine::ProductionInferenceEngine;
```

**Evidence**: Spec lines 139-142 show required public API exports

---

#### AC5: Ensure full-engine feature compiles with stubs for WIP functionality
**Requirements**:
- Implement minimal compile-only stubs with `#[ignore = "WIP: full-engine in progress"]` attribute
- Preserve test structure for future implementation completion
- Ensure no runtime test failures (all WIP tests ignored)

**Validation Command**:
```bash
cargo test -p bitnet-inference --no-default-features --features full-engine --no-run
```

**Expected Output**: Compilation succeeds, no tests executed (--no-run)

**Test Tag**: `// AC5: Full-engine compile stubs`

**Code Pattern**:
```rust
#[cfg(feature = "full-engine")]
#[ignore = "WIP: full-engine implementation in progress"]
#[test]
fn test_production_inference_engine_initialization() {
    // Stub test for compilation validation
}
```

**Evidence**: Spec lines 150-158 show stub pattern for WIP functionality

---

### P3: Test Infrastructure Fixes

#### AC6: Verify fixtures module compilation under fixtures feature
**Requirements**:
- **Status**: Module already declared in `tests/lib.rs:31` and `tests/common/mod.rs:35`
- Validate existing `FixtureManager` accessibility to dependent tests
- **Action**: Verification-only (no implementation changes needed)

**Validation Command**:
```bash
cargo test -p bitnet-tests --no-default-features --features fixtures --no-run
```

**Expected Output**: Compilation succeeds with fixtures module accessible

**Test Tag**: `// AC6: Verify fixtures module declaration`

**Evidence**:
- `tests/lib.rs:31`: `pub mod fixtures { pub use crate::common::fixtures::*; }`
- `tests/common/mod.rs:35`: `pub mod fixtures;`

**Correction Applied**: Changed from "Declare fixtures module" to "Verify fixtures module" based on existing declarations

---

#### AC7: Update tests crate to match current TestConfig API
**Requirements**:
- **CORRECTED**: Replace `timeout_seconds` → `test_timeout: Duration`
- **CRITICAL**: Remove invalid `fail_fast` references
  - Old `TestConfig` had `fail_fast` at top level (deprecated)
  - New `TestConfig` does NOT have `fail_fast` field (neither at top level nor in `ReportingConfig`)
  - `TimeConstraints.fail_fast` exists but is NOT part of `TestConfig` structure
- Update `tests/run_configuration_tests.rs` and related test files

**Validation Command**:
```bash
cargo test -p tests --no-run
```

**Expected Output**: Compilation succeeds with correct TestConfig API usage

**Test Tag**: `// AC7: TestConfig API migration (Duration, remove fail_fast)`

**Current API Structure** (`tests/common/config.rs:14-32`):
```rust
pub struct TestConfig {
    pub max_parallel_tests: usize,
    pub test_timeout: Duration,  // ← NEW: Duration instead of u64 seconds
    pub cache_dir: PathBuf,
    pub log_level: String,
    pub coverage_threshold: f64,
    #[cfg(feature = "fixtures")]
    pub fixtures: FixtureConfig,
    pub crossval: CrossValidationConfig,
    pub reporting: ReportingConfig,  // ← Does NOT contain fail_fast
}
```

**ReportingConfig Structure** (`tests/common/config.rs:148-175`):
```rust
pub struct ReportingConfig {
    pub output_dir: PathBuf,
    pub formats: Vec<ReportFormat>,
    pub include_artifacts: bool,
    pub generate_coverage: bool,
    pub generate_performance: bool,
    pub upload_reports: bool,
    // Note: fail_fast is NOT present in this structure
}
```

**Evidence**:
- `tests/common/config.rs:18`: `pub test_timeout: Duration` (correct field)
- `tests/common/config.rs:148-175`: ReportingConfig has 6 fields, `fail_fast` NOT present
- `tests/test_configuration_scenarios.rs:85`: `TimeConstraints.fail_fast` exists but is separate from TestConfig

**Correction Applied**: Removed invalid `reporting.fail_fast` reference from spec; clarified that `fail_fast` field does not exist in new TestConfig API

---

### CI Gate Improvements

#### AC8: Add feature-aware exploratory CI gates
**Requirements**:
- Maintain strict required gates:
  - `cargo clippy --workspace --no-default-features --features cpu -- -D warnings`
  - `cargo test --workspace --no-default-features --features cpu`
- Add exploratory jobs (allowed to fail until AC1-AC7 complete):
  - `cargo clippy --workspace --all-features -- -D warnings`
  - `cargo test --workspace --all-features`
- Promote exploratory to required after all fixes validated

**Validation Command**:
```bash
# Required gates (must pass)
cargo clippy --workspace --no-default-features --features cpu -- -D warnings
cargo test --workspace --no-default-features --features cpu

# Exploratory gates (allowed to fail initially)
cargo clippy --workspace --all-features -- -D warnings
cargo test --workspace --all-features
```

**Expected Output**: Required gates pass; exploratory gates may fail until AC1-AC7 complete

**Test Tag**: `// AC8: Feature-aware CI gates`

**Evidence**: Spec lines 41-44 show CI gate strategy with strict required baseline and exploratory all-features validation

---

## Validation Command Summary

```bash
# AC1-AC3: OpenTelemetry OTLP migration
cargo check -p bitnet-server --no-default-features --features opentelemetry
cargo build -p bitnet-server --no-default-features --features opentelemetry --release
cargo clippy -p bitnet-server --no-default-features --features opentelemetry -- -D warnings

# AC4-AC5: Inference engine types and stubs
cargo check -p bitnet-inference --all-features
cargo test -p bitnet-inference --no-default-features --features full-engine --no-run

# AC6: Fixtures module verification
cargo test -p bitnet-tests --no-default-features --features fixtures --no-run

# AC7: TestConfig API migration
cargo test -p tests --no-run

# AC8: All-features exploratory validation
cargo clippy --workspace --all-features -- -D warnings
cargo test --workspace --all-features
```

---

## BitNet-rs Standards Compliance

### Feature Flag Discipline
- ✅ All commands specify `--no-default-features` where appropriate
- ✅ Explicit feature selection: `cpu`, `gpu`, `opentelemetry`, `full-engine`, `fixtures`
- ✅ Default features are EMPTY to prevent unwanted dependencies

### Workspace Structure Alignment
- ✅ Multi-crate coordination: bitnet-server, bitnet-inference, bitnet-tests, tests
- ✅ Crate-specific validation commands for isolation
- ✅ Workspace-wide validation for integration

### Neural Network Development Patterns
- ✅ Observability isolated to bitnet-server (no inference performance impact)
- ✅ Test infrastructure changes do not affect production inference
- ✅ Compilation fixes only (no model format or algorithm changes)

### TDD and Test Naming
- ✅ All ACs have `// AC:N` test tag comments specified
- ✅ Validation commands map directly to AC requirements
- ✅ Story → Schema → Tests → Code traceability clear

### GGUF Compatibility
- ✅ No impact (compilation fixes only, no model format changes)

---

## Evidence Files Referenced

1. **TestConfig API Structure**:
   - `/home/steven/code/Rust/BitNet-rs/tests/common/config.rs:14-32` (TestConfig definition)
   - `/home/steven/code/Rust/BitNet-rs/tests/common/config.rs:148-175` (ReportingConfig definition)

2. **Fixtures Module Declaration**:
   - `/home/steven/code/Rust/BitNet-rs/tests/lib.rs:31` (module re-export)
   - `/home/steven/code/Rust/BitNet-rs/tests/common/mod.rs:35` (module declaration)

3. **OTLP Migration Specification**:
   - `/home/steven/code/Rust/BitNet-rs/docs/explanation/specs/issue-447-compilation-fixes-technical-spec.md:91-111` (migration pattern)

4. **fail_fast Field Investigation**:
   - `/home/steven/code/Rust/BitNet-rs/tests/test_configuration_scenarios.rs:85` (TimeConstraints.fail_fast)
   - `/home/steven/code/Rust/BitNet-rs/docs/research/pr-445-compilation-errors-research.md:289-300` (error context)

---

## Critical Corrections Applied

### 1. AC7: fail_fast Field Removal
**Original Spec Statement**: "Replace `fail_fast` → `reporting.fail_fast`"

**Evidence Against**:
- `ReportingConfig` structure (lines 148-175 of `tests/common/config.rs`) contains 6 fields:
  - `output_dir: PathBuf`
  - `formats: Vec<ReportFormat>`
  - `include_artifacts: bool`
  - `generate_coverage: bool`
  - `generate_performance: bool`
  - `upload_reports: bool`
- **NO `fail_fast` field present**

**Corrected Requirement**:
- Remove invalid `fail_fast` references from test code
- Field does not exist in new TestConfig API (neither top-level nor in ReportingConfig)

**Rationale**: Spec assumed `fail_fast` migrated to nested field, but actual API investigation shows field was removed entirely

---

### 2. AC6: Fixtures Module Status
**Original Spec Statement**: "Declare fixtures module in bitnet-tests"

**Evidence Against**:
- `tests/lib.rs:31` already declares: `pub mod fixtures { pub use crate::common::fixtures::*; }`
- `tests/common/mod.rs:35` already declares: `pub mod fixtures;`

**Corrected Requirement**:
- Verify existing module declaration and accessibility
- Validation-only (no implementation changes needed)

**Rationale**: Module already properly declared; AC should verify compilation, not implement

---

### 3. AC2: OTLP Environment Variable
**Original Spec Statement**: "Support `OTEL_EXPORTER_OTLP_ENDPOINT`"

**Clarification Added**:
- Default fallback: `http://127.0.0.1:4317` (standard OTLP localhost)
- Metrics work without requiring env var to be set
- Environment variable is optional override, not required configuration

**Rationale**: Standard OTLP practice uses localhost default; production deployments override with collector endpoint

---

## Routing Decision

**Status**: ✅ Ready for Spec Creation

**Decision**: NEXT → spec-creator

**Reason**:
- All 8 acceptance criteria validated and corrected
- Story → Schema → Tests → Code traceability clear
- Validation commands specified for each AC with BitNet-rs toolchain
- Issue Ledger complete with proper anchors and sections
- Requirements align with BitNet-rs neural network development standards
- No fundamental AC issues requiring return to issue-creator

**Quality Gates Passed**:
- ✅ All required Ledger sections present (Gates, Hoplog, Decision)
- ✅ All ACs atomic, observable, and testable
- ✅ BitNet-rs workspace crates and feature flags correctly referenced
- ✅ Compilation validation commands specified with proper feature discipline
- ✅ No-default-features baseline preserved
- ✅ TDD test tags assigned to all acceptance criteria

---

## Next Steps for spec-creator

1. **Create Technical Schema**:
   - OpenTelemetry OTLP metrics integration schema
   - bitnet-inference public API exports schema
   - TestConfig API migration schema

2. **Define Test Structure**:
   - AC-tagged test names (`test_ac1_*`, `test_ac2_*`, etc.)
   - Compilation validation test harness
   - Feature-aware CI gate structure

3. **Specify Implementation Order**:
   - P0 (AC1-AC3): OpenTelemetry OTLP migration
   - P2 (AC4-AC5): Inference engine type exports
   - P3 (AC6-AC7): Test infrastructure verification and API updates
   - CI (AC8): Feature-aware gate configuration

4. **Document Validation Evidence**:
   - Cargo check/build/clippy output expectations
   - Feature flag combination testing matrix
   - Dependency tree validation for OTLP compatibility
