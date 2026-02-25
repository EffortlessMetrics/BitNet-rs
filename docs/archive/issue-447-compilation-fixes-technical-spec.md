# Technical Specification: Issue #447 - Compilation Fixes Across Workspace

**Issue**: https://github.com/EffortlessMetrics/BitNet-rs/issues/447
**Spec File**: `/home/steven/code/Rust/BitNet-rs/docs/explanation/issue-compilation-fixes-spec.md`
**Date**: 2025-10-11
**Status**: Specification Complete - Ready for Implementation

---

## Executive Summary

Issue #447 addresses critical compilation failures across four BitNet.rs workspace crates that prevent successful builds with various feature flag combinations. The primary failure is an **OpenTelemetry Prometheus version incompatibility** (v0.29.1 incompatible with opentelemetry-sdk v0.31.0) in `bitnet-server`, blocking production monitoring. Secondary issues include missing type exports in `bitnet-inference`, fixture module visibility in `bitnet-tests`, and deprecated API usage in the `tests` crate.

**Impact Assessment**: The OpenTelemetry incompatibility affects all server observability builds (`--features opentelemetry`), while inference and test infrastructure issues block `--all-features` comprehensive testing workflows.

---

## 1. Requirements Analysis

### 1.1 Functional Requirements

#### FR1: OpenTelemetry OTLP Migration (P0 - Critical)
- **Current State**: `opentelemetry-prometheus@0.29.1` incompatible with `opentelemetry-sdk@0.31.0`
- **Target State**: OTLP-based metrics exporter using `opentelemetry-otlp@0.31.0` with `grpc-tonic` feature
- **Constraint**: Workspace already specifies `opentelemetry-otlp@0.31.0` with features `["grpc-tonic", "trace"]`
- **Environment Configuration**: Support `OTEL_EXPORTER_OTLP_ENDPOINT` (default: `http://127.0.0.1:4317`)
- **Backward Compatibility**: Preserve existing metric instrumentation points (request counts, latencies, resource usage)

#### FR2: Inference Engine Type Visibility (P2 - High)
- **Current State**: `EngineConfig` and `ProductionInferenceEngine` not exported for `full-engine` feature tests
- **Target State**: Export types in crate public API for test visibility
- **Missing Imports**: Add `std::env`, `anyhow::Context` to test modules
- **Compilation Stub**: Implement minimal compile-only stubs for WIP `full-engine` functionality with `#[ignore]` attribute

#### FR3: Test Infrastructure Fixes (P3 - Medium)
- **Fixtures Module**: Declare `#[cfg(feature = "fixtures")] pub mod fixtures;` in `bitnet-tests` module root
- **TestConfig API Migration**: Replace deprecated field access patterns:
  - `timeout_seconds` → `test_timeout` (Duration)
  - `fail_fast` → `reporting.fail_fast` (nested field)

#### FR4: CI Feature-Aware Gates (CI Improvements)
- **Required Gates (Strict)**: `cargo clippy --workspace --no-default-features --features cpu`
- **Exploratory Gates (Allowed to Fail)**: `cargo clippy --workspace --all-features`
- **Promotion Strategy**: Move exploratory to required after AC1-AC7 completion

### 1.2 Non-Functional Requirements

- **Zero Impact on Inference Performance**: OpenTelemetry changes isolated to server-only code paths
- **Backward Compatibility**: Feature flag behavior preserved for existing consumers
- **Observability Continuity**: OTLP metrics maintain same instrumentation points as Prometheus
- **Error Handling**: Proper error context for OTLP endpoint connection failures using `anyhow::Context`

---

## 2. Architecture Approach

### 2.1 Crate-Specific Implementation Strategy

#### 2.1.1 bitnet-server (OpenTelemetry Migration)

**Affected Files**:
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-server/Cargo.toml`
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-server/src/monitoring/opentelemetry.rs`
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-server/src/monitoring/prometheus.rs`

**Current Dependency Conflict**:
```toml
# Workspace Cargo.toml (line 233)
opentelemetry-prometheus = "0.29.1"  # ← Incompatible with opentelemetry-sdk@0.31.0

# bitnet-server depends on:
opentelemetry = { workspace = true }          # v0.31.0
opentelemetry-otlp = { workspace = true }     # v0.31.0
opentelemetry_sdk = { workspace = true }      # v0.31.0
opentelemetry-prometheus = { workspace = true }  # v0.29.1 ← CONFLICT
```

**Error Evidence**:
```
error[E0277]: the trait bound `PrometheusExporter: opentelemetry_sdk::metrics::reader::MetricReader` is not satisfied
   --> crates/bitnet-server/src/monitoring/opentelemetry.rs:92:84
    |
 92 |     let provider = SdkMeterProvider::builder().with_resource(resource).with_reader(reader).build();
    |                                                                        ----------- ^^^^^^ the trait `opentelemetry_sdk::metrics::reader::MetricReader` is not implemented for `PrometheusExporter`
```

**Migration Strategy**:
1. **Remove Prometheus Dependency**: Delete `opentelemetry-prometheus@0.29.1` from workspace `Cargo.toml`
2. **Update OTLP Configuration**: Add `metrics` feature to `opentelemetry-otlp` workspace dependency
3. **Implement OTLP Metrics Reader**: Replace `opentelemetry_prometheus::exporter()` with `opentelemetry_otlp::MetricsExporter`
4. **Environment Variable Support**: Read `OTEL_EXPORTER_OTLP_ENDPOINT` with fallback to `http://127.0.0.1:4317`

**Code Pattern**:
```rust
// Before (Prometheus v0.29.1 - BROKEN):
use opentelemetry_prometheus::exporter;
let reader = exporter().build()?;

// After (OTLP v0.31.0 - FIXED):
use opentelemetry_otlp::MetricsExporter;
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

**Observability Preservation**:
- Existing instrumentation points in `/home/steven/code/Rust/BitNet-rs/crates/bitnet-server/src/monitoring/opentelemetry.rs:109-220`:
  - `record_inference_metrics()` (lines 109-128)
  - `record_model_load_metrics()` (lines 131-146)
  - `record_quantization_metrics()` (lines 149-166)
  - All metrics preserved with OTLP exporter

#### 2.1.2 bitnet-inference (Type Exports & Stubs)

**Affected Files**:
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/src/lib.rs` (public API)
- `/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/tests/comprehensive_tests.rs`

**Current State**:
```rust
// comprehensive_tests.rs (line 1-8)
#![cfg(feature = "integration-tests")]  // ← WRONG: Should be "full-engine"
//! Placeholder: the comprehensive test suite was removed/moved.
#[test]
fn placeholder_comprehensive_tests() {
    // Intentionally empty.
}
```

**Implementation Strategy**:
1. **Export Missing Types**: Add to `bitnet-inference/src/lib.rs`:
   ```rust
   pub use crate::config::EngineConfig;
   pub use crate::engine::ProductionInferenceEngine;
   ```

2. **Add Missing Imports**: Update test modules requiring `full-engine` feature:
   ```rust
   use std::env;
   use anyhow::Context;
   ```

3. **Minimal Compile Stubs**: If `full-engine` implementation incomplete, add:
   ```rust
   #[cfg(feature = "full-engine")]
   #[ignore = "WIP: full-engine implementation in progress"]
   #[test]
   fn test_production_inference_engine_initialization() {
       // Stub test for compilation validation
   }
   ```

**Validation Commands**:
```bash
# AC4: Type exports
cargo check -p bitnet-inference --all-features

# AC5: Compile-only stubs
cargo test -p bitnet-inference --no-default-features --features full-engine --no-run
```

#### 2.1.3 bitnet-tests (Fixtures Module Declaration)

**Affected Files**:
- `/home/steven/code/Rust/BitNet-rs/tests/lib.rs`

**Current State**:
```rust
// tests/lib.rs (lines 30-32)
#[cfg(feature = "fixtures")]
pub mod fixtures {
    pub use crate::common::fixtures::*;
}
```

**Issue**: Module declaration exists, but `common::fixtures` module may not be properly declared in `tests/common/mod.rs`.

**Implementation Strategy**:
1. **Verify Module Structure**: Ensure `tests/common/mod.rs` declares:
   ```rust
   #[cfg(feature = "fixtures")]
   pub mod fixtures;
   ```

2. **Check Fixture Manager Exports**: Verify `FixtureManager` accessible from `common::fixtures`:
   ```rust
   // tests/common/fixtures.rs or tests/common/fixtures/mod.rs
   pub struct FixtureManager { /* ... */ }
   ```

**Validation Command**:
```bash
# AC6: Fixtures module compilation
cargo test -p bitnet-tests --no-default-features --features fixtures --no-run
```

#### 2.1.4 tests Crate (TestConfig API Migration)

**Affected Files**:
- `/home/steven/code/Rust/BitNet-rs/tests/run_configuration_tests.rs` (line 74, 320)

**Current API Usage** (DEPRECATED):
```rust
// run_configuration_tests.rs:74
assert_eq!(config.timeout_seconds, 300);  // ← Old field name

// run_configuration_tests.rs:320
let invalid_config = TestConfig { timeout_seconds: 0, ..Default::default() };  // ← Deprecated
```

**New API Structure** (`tests/common/config.rs:14-32`):
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
    pub reporting: ReportingConfig,  // ← NEW: Nested struct for fail_fast
}

pub struct ReportingConfig {
    pub output_dir: PathBuf,
    pub formats: Vec<ReportFormat>,
    pub include_artifacts: bool,
    pub generate_coverage: bool,
    pub generate_performance: bool,
    pub upload_reports: bool,
    // Note: fail_fast is NOT in ReportingConfig based on current code
}
```

**Migration Pattern**:
```rust
// Before (DEPRECATED):
config.timeout_seconds  // u64 in seconds
config.fail_fast        // bool at top level

// After (CORRECT):
config.test_timeout     // Duration
config.reporting.fail_fast  // ERROR: fail_fast not found in ReportingConfig
```

**CRITICAL FINDING**: The spec mentions `reporting.fail_fast`, but `ReportingConfig` in `/home/steven/code/Rust/BitNet-rs/tests/common/config.rs:148-175` does NOT contain a `fail_fast` field. This suggests:
1. **Option A**: The old `TestConfig` had `fail_fast` at top level, which needs removal (not migration)
2. **Option B**: `ReportingConfig` needs a new `fail_fast` field added

**Recommended Strategy**:
1. Search for all `fail_fast` usage in tests crate
2. If old `TestConfig` had `fail_fast` at top level, remove those references
3. If needed, add `fail_fast` to `ReportingConfig` structure

**Validation Command**:
```bash
# AC7: TestConfig API updates
cargo test -p tests --no-run
```

---

## 3. Risk Assessment & Mitigation

### 3.1 OpenTelemetry Migration (P0 - Medium Risk)

**Risk Factors**:
- **API Surface Change**: PrometheusExporter → OTLP MetricsExporter different initialization pattern
- **Dependency Compatibility**: tonic version alignment with `opentelemetry-otlp` gRPC feature
- **Runtime Behavior**: OTLP push model vs Prometheus pull model may require collector setup

**Mitigation Strategies**:
1. **Feature Flag Isolation**: Changes gated behind `opentelemetry` feature (line 71 of `bitnet-server/Cargo.toml`)
2. **Graceful Degradation**: Fallback to stdout exporter if OTLP endpoint unreachable
3. **Validation Sequence**:
   ```bash
   # Step 1: Check dependency tree for conflicts
   cargo tree -p bitnet-server --no-default-features --features opentelemetry | grep opentelemetry

   # Step 2: Compilation check
   cargo check -p bitnet-server --no-default-features --features opentelemetry

   # Step 3: Clippy validation
   cargo clippy -p bitnet-server --no-default-features --features opentelemetry -- -D warnings

   # Step 4: Integration test (if observability tests exist)
   cargo test -p bitnet-server --no-default-features --features opentelemetry
   ```

**Rollback Strategy**: Disable `opentelemetry` feature flag if production issues occur; server continues operation without observability.

### 3.2 Inference Engine Types (P2 - Low Risk)

**Risk Factors**:
- **API Stability**: Exporting internal types increases public API surface
- **Breaking Changes**: Future refactoring of `EngineConfig` or `ProductionInferenceEngine` becomes breaking

**Mitigation Strategies**:
1. **Documentation**: Mark exported types with `#[doc(hidden)]` if internal-only
2. **SemVer Consideration**: Document these types as unstable in next release notes
3. **Validation**:
   ```bash
   # Check compilation with all features
   cargo check -p bitnet-inference --all-features

   # Verify test compilation without execution
   cargo test -p bitnet-inference --no-default-features --features full-engine --no-run
   ```

### 3.3 Test Infrastructure (P3 - Trivial Risk)

**Risk Factors**: Mechanical API updates with clear mapping

**Mitigation**: Automated grep validation:
```bash
# Find all deprecated API usage
grep -r "timeout_seconds" /home/steven/code/Rust/BitNet-rs/tests --include="*.rs"
grep -r "fail_fast" /home/steven/code/Rust/BitNet-rs/tests --include="*.rs" | grep -v "reporting.fail_fast"
```

---

## 4. Implementation Sequence

### Phase 1: Dependency Resolution (AC1)
1. **Update Workspace Cargo.toml**:
   ```bash
   # Remove line 233:
   # opentelemetry-prometheus = "0.29.1"

   # Update line 232 to add metrics feature:
   opentelemetry-otlp = { version = "0.31.0", default-features = false, features = ["grpc-tonic", "trace", "metrics"] }
   ```

2. **Update bitnet-server Cargo.toml** (line 33):
   ```toml
   # Remove:
   opentelemetry-prometheus = { workspace = true, optional = true }

   # Keep (already correct):
   opentelemetry-otlp = { workspace = true, optional = true }
   ```

3. **Validation**:
   ```bash
   cargo check -p bitnet-server --no-default-features --features opentelemetry
   ```

### Phase 2: OTLP Implementation (AC2)
1. **Implement init_metrics() in opentelemetry.rs** (line 84-97):
   ```rust
   async fn init_metrics(config: &MonitoringConfig) -> Result<()> {
       let endpoint = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT")
           .unwrap_or_else(|_| "http://127.0.0.1:4317".to_string());

       let resource = Resource::builder()
           .with_service_name("bitnet-server")
           .with_attributes(vec![KeyValue::new("service.version", env!("CARGO_PKG_VERSION"))])
           .build();

       let exporter = opentelemetry_otlp::MetricsExporter::builder()
           .with_tonic()
           .with_endpoint(endpoint)
           .with_timeout(Duration::from_secs(3))
           .build()
           .context("Failed to build OTLP metrics exporter")?;

       let reader = opentelemetry_sdk::metrics::PeriodicReader::builder(exporter)
           .with_interval(Duration::from_secs(60))
           .build();

       let provider = SdkMeterProvider::builder()
           .with_resource(resource)
           .with_reader(reader)
           .build();

       global::set_meter_provider(provider);

       Ok(())
   }
   ```

2. **Validation**:
   ```bash
   cargo build -p bitnet-server --no-default-features --features opentelemetry --release
   ```

### Phase 3: Prometheus Code Removal (AC3)
1. **Delete Prometheus Module** (if exists):
   ```bash
   # Check if prometheus.rs is still used
   grep -r "prometheus::" /home/steven/code/Rust/BitNet-rs/crates/bitnet-server/src

   # If unused, remove file:
   rm /home/steven/code/Rust/BitNet-rs/crates/bitnet-server/src/monitoring/prometheus.rs
   ```

2. **Update Module Exports** in `monitoring/mod.rs`:
   ```rust
   #[cfg(feature = "opentelemetry")]
   pub mod opentelemetry;

   // Remove if Prometheus module deleted:
   // #[cfg(feature = "prometheus")]
   // pub mod prometheus;
   ```

3. **Validation**:
   ```bash
   cargo clippy -p bitnet-server --no-default-features --features opentelemetry -- -D warnings
   ```

### Phase 4: Inference Type Exports (AC4)
1. **Update bitnet-inference/src/lib.rs**:
   ```rust
   // Add to public API section
   pub use crate::config::EngineConfig;
   pub use crate::engine::ProductionInferenceEngine;
   ```

2. **Add Missing Imports to Test Modules**:
   ```rust
   // In tests requiring full-engine feature
   use std::env;
   use anyhow::Context;
   ```

3. **Validation**:
   ```bash
   cargo check -p bitnet-inference --all-features
   ```

### Phase 5: Full-Engine Stubs (AC5)
1. **Update comprehensive_tests.rs**:
   ```rust
   #![cfg(feature = "full-engine")]  // Fix: was "integration-tests"

   #[ignore = "WIP: full-engine implementation in progress"]
   #[test]
   fn placeholder_comprehensive_tests() {
       // Minimal stub for compilation validation
   }
   ```

2. **Validation**:
   ```bash
   cargo test -p bitnet-inference --no-default-features --features full-engine --no-run
   ```

### Phase 6: Fixtures Module (AC6)
1. **Verify Module Declaration** in `tests/common/mod.rs`:
   ```rust
   #[cfg(feature = "fixtures")]
   pub mod fixtures;
   ```

2. **Validation**:
   ```bash
   cargo test -p bitnet-tests --no-default-features --features fixtures --no-run
   ```

### Phase 7: TestConfig API Migration (AC7)
1. **Find Deprecated Usage**:
   ```bash
   grep -rn "timeout_seconds" /home/steven/code/Rust/BitNet-rs/tests/run_configuration_tests.rs
   ```

2. **Apply Mechanical Replacements**:
   ```rust
   // Line 74: BEFORE
   assert_eq!(config.timeout_seconds, 300);

   // Line 74: AFTER
   assert_eq!(config.test_timeout.as_secs(), 300);

   // Line 320: BEFORE
   let invalid_config = TestConfig { timeout_seconds: 0, ..Default::default() };

   // Line 320: AFTER
   let invalid_config = TestConfig { test_timeout: Duration::from_secs(0), ..Default::default() };
   ```

3. **Handle fail_fast Migration**:
   - **Option A (Recommended)**: Remove `fail_fast` references if no longer in API
   - **Option B**: Add `fail_fast` field to `ReportingConfig` if needed

4. **Validation**:
   ```bash
   cargo test -p tests --no-run
   ```

### Phase 8: CI Gate Updates (AC8)
1. **Update GitHub Actions Workflow** (`.github/workflows/*.yml`):
   ```yaml
   # Required gate (existing - keep strict)
   - name: Clippy Workspace CPU
     run: cargo clippy --workspace --no-default-features --features cpu -- -D warnings

   # New exploratory gate (allowed to fail initially)
   - name: Clippy All Features (Exploratory)
     run: cargo clippy --workspace --all-features -- -D warnings
     continue-on-error: true  # Allow failures until AC1-AC7 complete
   ```

2. **Promotion Plan**: Remove `continue-on-error: true` after all ACs verified

---

## 5. Test Strategy & Validation Plan

### 5.1 TDD Scaffolding Tags
```rust
// AC1: Remove Prometheus dependency
// File: Cargo.toml:233
// Remove: opentelemetry-prometheus = "0.29.1"

// AC2: OTLP metrics init
// File: crates/bitnet-server/src/monitoring/opentelemetry.rs:84
async fn init_metrics(config: &MonitoringConfig) -> Result<()> { /* ... */ }

// AC3: Clean observability compilation
// Validation: cargo clippy -p bitnet-server --no-default-features --features opentelemetry

// AC4: Export engine types
// File: crates/bitnet-inference/src/lib.rs
pub use crate::config::EngineConfig;

// AC5: Full-engine stubs
// File: crates/bitnet-inference/tests/comprehensive_tests.rs:1
#![cfg(feature = "full-engine")]

// AC6: Fixtures module declaration
// File: tests/lib.rs:30
#[cfg(feature = "fixtures")] pub mod fixtures;

// AC7: TestConfig API update
// File: tests/run_configuration_tests.rs:74
assert_eq!(config.test_timeout.as_secs(), 300);

// AC8: CI feature gates
// File: .github/workflows/*.yml
continue-on-error: true  # Exploratory gate
```

### 5.2 Comprehensive Validation Commands

#### Baseline Validation (Required Gates)
```bash
# Must pass: CPU feature gate
cargo check --workspace --no-default-features --features cpu
cargo clippy --workspace --no-default-features --features cpu -- -D warnings
cargo test --workspace --no-default-features --features cpu
```

#### AC-Specific Validation
```bash
# AC1-AC3: OpenTelemetry Migration
cargo tree -p bitnet-server --no-default-features --features opentelemetry | grep -E "(opentelemetry|prometheus)"
cargo check -p bitnet-server --no-default-features --features opentelemetry
cargo build -p bitnet-server --no-default-features --features opentelemetry --release
cargo clippy -p bitnet-server --no-default-features --features opentelemetry -- -D warnings

# AC4-AC5: Inference Engine Types
cargo check -p bitnet-inference --all-features
cargo test -p bitnet-inference --no-default-features --features full-engine --no-run

# AC6: Fixtures Module
cargo test -p bitnet-tests --no-default-features --features fixtures --no-run

# AC7: TestConfig API
cargo test -p tests --no-run
cargo test -p tests run_configuration_tests

# AC8: Exploratory All-Features (Post-Fixes)
cargo clippy --workspace --all-features -- -D warnings
cargo test --workspace --all-features --no-run
```

#### Feature Matrix Validation
```bash
# Validate feature flag isolation
cargo check --workspace --no-default-features --features cpu
cargo check --workspace --no-default-features --features gpu
cargo check --workspace --no-default-features --features opentelemetry

# Validate crate-specific feature combinations
cargo check -p bitnet-server --no-default-features --features cpu,opentelemetry
cargo check -p bitnet-inference --no-default-features --features cpu,full-engine
cargo check -p bitnet-tests --no-default-features --features fixtures,reporting
```

### 5.3 Cross-Validation Requirements
**Not Required**: No inference algorithm changes; observability and test infrastructure only.

### 5.4 Performance Validation
**Not Required**: OpenTelemetry migration should have minimal overhead (~1-2% for metrics collection), but no performance regression testing needed as server observability is orthogonal to inference performance.

---

## 6. Documentation Updates

### 6.1 Observability Documentation
- **File**: `docs/health-endpoints.md` (or create `docs/observability.md`)
- **Updates Required**:
  - Document OTLP endpoint configuration (`OTEL_EXPORTER_OTLP_ENDPOINT`)
  - Update metrics export examples from Prometheus to OTLP
  - Add collector setup instructions (Jaeger, OpenTelemetry Collector)

### 6.2 Development Workflow Documentation
- **File**: `docs/development/build-commands.md`
- **Updates Required**:
  - Add exploratory `--all-features` build commands
  - Document feature flag combinations that must compile

### 6.3 Testing Framework Documentation
- **File**: `docs/development/test-suite.md`
- **Updates Required**:
  - Document new `TestConfig` API structure
  - Update test configuration examples using `test_timeout: Duration`

---

## 7. BitNet.rs Alignment Verification

### 7.1 Feature-Gated Architecture ✅
- **Compliance**: All changes respect `--no-default-features` baseline
- **Validation**:
  ```bash
  cargo build --workspace --no-default-features --features cpu
  cargo build --workspace --no-default-features --features gpu
  ```

### 7.2 Workspace Structure ✅
- **Compliance**: Changes span multiple crates with coordinated dependency updates
- **Isolation**: OpenTelemetry changes confined to `bitnet-server`, no inference crates affected

### 7.3 TDD Practices ✅
- **Compliance**: Each AC maps to specific compilation validation command
- **Evidence**: Test infrastructure fixes (AC6-AC7) improve test reliability

### 7.4 Cross-Platform Compatibility ✅
- **Compliance**: OpenTelemetry OTLP works on Linux, macOS, Windows, WebAssembly
- **GPU/CPU Parity**: No inference backend changes

### 7.5 Zero Inference Impact ✅
- **Evidence**: Changes isolated to:
  - `bitnet-server` (observability layer)
  - `bitnet-inference` (type exports, no logic changes)
  - `bitnet-tests` (test utilities)
  - No changes to `bitnet-kernels`, `bitnet-quantization`, `bitnet-models`

---

## 8. Success Criteria & Acceptance Validation

### AC1: Prometheus Dependency Removal ✅
```bash
# Success: No opentelemetry-prometheus in dependency tree
cargo tree -p bitnet-server --no-default-features --features opentelemetry | grep -q "opentelemetry-prometheus" && echo "FAIL" || echo "PASS"

# Success: Compilation succeeds
cargo check -p bitnet-server --no-default-features --features opentelemetry
```

### AC2: OTLP Metrics Initialization ✅
```bash
# Success: Release build succeeds
cargo build -p bitnet-server --no-default-features --features opentelemetry --release

# Manual verification: Start server with OTEL_EXPORTER_OTLP_ENDPOINT set
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 cargo run -p bitnet-server --no-default-features --features opentelemetry
```

### AC3: Clean Observability Compilation ✅
```bash
# Success: Clippy passes with -D warnings
cargo clippy -p bitnet-server --no-default-features --features opentelemetry -- -D warnings
```

### AC4: Engine Type Exports ✅
```bash
# Success: All features compile
cargo check -p bitnet-inference --all-features
```

### AC5: Full-Engine Stubs ✅
```bash
# Success: Test compilation succeeds (no execution required)
cargo test -p bitnet-inference --no-default-features --features full-engine --no-run
```

### AC6: Fixtures Module Declaration ✅
```bash
# Success: Fixtures feature compiles
cargo test -p bitnet-tests --no-default-features --features fixtures --no-run
```

### AC7: TestConfig API Updates ✅
```bash
# Success: Tests crate compiles
cargo test -p tests --no-run

# Success: Specific test file compiles
cargo test -p tests run_configuration_tests --no-run
```

### AC8: CI Feature Gates ✅
```bash
# Success: Required gates pass (CPU feature)
cargo clippy --workspace --no-default-features --features cpu -- -D warnings

# Success: Exploratory gates run (allowed to fail before fixes)
cargo clippy --workspace --all-features -- -D warnings || echo "Expected failure before AC1-AC7 complete"
```

---

## 9. References & Neural Network Context

### 9.1 BitNet.rs Architecture Patterns
- **Observability Isolation**: Server-only concerns (health endpoints, metrics) separate from inference
- **Feature Flag Discipline**: Empty default features enforce explicit capability selection
- **Workspace Coordination**: Cross-crate dependency updates require careful sequencing

### 9.2 OpenTelemetry Migration Precedents
- **Rust Ecosystem Trend**: OpenTelemetry v0.31+ standardizes on OTLP for unified telemetry
- **Collector Architecture**: OTLP push model enables centralized observability (Jaeger, Prometheus, Datadog)
- **Production Pattern**: Environment variable configuration (`OTEL_EXPORTER_OTLP_ENDPOINT`) standard practice

### 9.3 Test Infrastructure Evolution
- **TestConfig Modernization**: Migration from scalar `timeout_seconds` to typed `Duration` improves API safety
- **Fixture Management**: Feature-gated module structure supports optional test utilities
- **CI Gate Strategy**: Exploratory gates enable proactive quality assurance without blocking development

---

## 10. Implementation Timeline & Risk Matrix

| Phase | AC | Time Estimate | Risk Level | Blocker Dependencies |
|-------|----|--------------:|------------|----------------------|
| 1 | AC1 | 0.5 days | Low | None |
| 2 | AC2 | 1.0 days | Medium | Phase 1 complete |
| 3 | AC3 | 0.5 days | Low | Phase 2 complete |
| 4 | AC4 | 0.25 days | Trivial | None (parallel with Phase 1-3) |
| 5 | AC5 | 0.25 days | Trivial | Phase 4 complete |
| 6 | AC6 | 0.25 days | Trivial | None (parallel) |
| 7 | AC7 | 0.25 days | Trivial | None (parallel) |
| 8 | AC8 | 0.25 days | Low | Phases 1-7 complete |

**Total Estimated Time**: 2.5-3.5 days (critical path through Phases 1-3)

**Risk Mitigation**:
- **Parallel Execution**: AC4-AC7 can proceed independently during AC1-AC3 blocking period
- **Rollback Plan**: Feature flag disable for OpenTelemetry if production issues occur
- **Incremental Validation**: Each phase has isolated validation command for early failure detection

---

## 11. Routing Decision

**Status**: Specification Complete → **FINALIZE → spec-finalizer**

**Evidence**:
1. ✅ Neural network requirements fully analyzed (zero inference impact confirmed)
2. ✅ Technical specification created in `docs/explanation/specs/` with comprehensive validation commands
3. ✅ Architecture approach aligns with BitNet.rs workspace structure and feature flags
4. ✅ Risk assessment includes specific validation commands and mitigation strategies
5. ✅ All ACs mapped to testable validation commands with expected outcomes

**Next Agent**: `spec-finalizer` for review and approval

---

**Specification Author**: BitNet.rs Neural Network Systems Architect
**Review Required**: Spec-finalizer (confirm OpenTelemetry migration strategy and TestConfig API assumptions)
**Implementation Ready**: Yes (after spec approval)
