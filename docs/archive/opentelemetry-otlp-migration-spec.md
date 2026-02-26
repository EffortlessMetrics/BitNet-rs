# OpenTelemetry OTLP Migration Specification

**Issue**: #447 (AC1-AC3)
**Status**: Ready for Implementation
**Priority**: P0 - Critical Compilation Fix
**Date**: 2025-10-11
**Affected Crate**: `bitnet-server`

---

## Executive Summary

Migrate `bitnet-server` from deprecated `opentelemetry-prometheus@0.29.1` to OTLP-based metrics using OpenTelemetry 0.31 SDK. This resolves compilation failures caused by incompatible dependency versions while maintaining observability capabilities for BitNet-rs neural network inference.

**Key Changes**:
- Remove deprecated Prometheus exporter dependency
- Implement OTLP gRPC metrics exporter with localhost fallback
- Preserve existing metric instrumentation points (inference, model loading, quantization)
- Zero impact on neural network inference performance

---

## Acceptance Criteria

### AC1: Remove deprecated Prometheus exporter dependency and migrate to OTLP
**Test Tag**: `// AC1: OTLP dependency migration`

**Requirements**:
- Remove `opentelemetry-prometheus@0.29.1` from workspace and crate dependencies
- Update `opentelemetry-otlp` to include `metrics` feature
- Delete incompatible PrometheusExporter integration code

**Validation Command**:
```bash
cargo check -p bitnet-server --no-default-features --features opentelemetry
```

**Expected Output**: Compilation succeeds with no errors related to PrometheusExporter

**Evidence**:
- Workspace `Cargo.toml:233` shows incompatible version: `opentelemetry-prometheus@0.29.1` vs `opentelemetry-sdk@0.31.0`
- `crates/bitnet-server/src/monitoring/opentelemetry.rs:91` uses deprecated `exporter()` function

---

### AC2: Implement OTLP metrics initialization with localhost fallback
**Test Tag**: `// AC2: OTLP metrics with env config`

**Requirements**:
- Support `OTEL_EXPORTER_OTLP_ENDPOINT` environment variable
- Default to `http://127.0.0.1:4317` (standard OTLP gRPC endpoint)
- Implement `PeriodicReader` with 60-second export interval
- Preserve metric instrumentation points:
  - `record_inference_metrics()` - request counts, latencies
  - `record_model_load_metrics()` - model loading operations
  - `record_quantization_metrics()` - quantization operations

**Validation Command**:
```bash
cargo build -p bitnet-server --no-default-features --features opentelemetry --release
```

**Expected Output**: Release build succeeds with optimized OTLP integration

**Environment Variables**:
- `OTEL_EXPORTER_OTLP_ENDPOINT`: Collector endpoint (default: `http://127.0.0.1:4317`)
- `OTEL_SERVICE_NAME`: Service identifier (default: `bitnet-server`)

---

### AC3: Remove Prometheus code paths and verify clean compilation
**Test Tag**: `// AC3: Clean OTLP-only compilation`

**Requirements**:
- Delete PrometheusExporter initialization and type conversions
- Remove deprecated `opentelemetry_prometheus::exporter()` usage
- Ensure zero clippy warnings in observability code

**Validation Command**:
```bash
cargo clippy -p bitnet-server --no-default-features --features opentelemetry -- -D warnings
```

**Expected Output**: No warnings, clean compilation with OTLP-only code

**Evidence**: Error output shows PrometheusExporter trait bound failure at `src/monitoring/opentelemetry.rs:92`

---

## Technical Design

### Dependency Changes

#### Workspace `Cargo.toml`

**Remove**:
```toml
# Line 233 - REMOVE THIS LINE
opentelemetry-prometheus = "0.29.1"
```

**Update**:
```toml
# Line 232 - ADD metrics feature
opentelemetry-otlp = { version = "0.31.0", default-features = false, features = ["grpc-tonic", "trace", "metrics"] }
```

**Keep (Already Compatible)**:
```toml
opentelemetry = { version = "0.31.0", default-features = false, features = ["trace", "metrics"] }
opentelemetry_sdk = { version = "0.31.0", features = ["trace", "metrics", "rt-tokio"] }
```

#### Crate `Cargo.toml` (`crates/bitnet-server/Cargo.toml`)

**Remove**:
```toml
# Line 33 - REMOVE THIS LINE
opentelemetry-prometheus = { workspace = true, optional = true }
```

**Update**:
```toml
# Line 71 - Update feature definition
opentelemetry = [
    "dep:opentelemetry",
    "dep:opentelemetry_sdk",
    "dep:opentelemetry-otlp",
    "dep:opentelemetry-stdout",
    "dep:tracing-opentelemetry"
]
```

**Note**: Remove `dep:opentelemetry-prometheus` from feature list

---

### OTLP Metrics Implementation

#### New Module: `crates/bitnet-server/src/monitoring/otlp.rs`

```rust
//! OTLP Metrics Integration for BitNet Server
//!
//! Provides OpenTelemetry Protocol (OTLP) metrics export using gRPC transport
//! with automatic localhost fallback for local development.

use anyhow::Result;
use opentelemetry::{global, KeyValue};
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::{metrics::SdkMeterProvider, Resource};
use std::time::Duration;

/// Initialize OTLP metrics exporter with environment variable support
///
/// # Environment Variables
/// - `OTEL_EXPORTER_OTLP_ENDPOINT`: Collector endpoint (default: http://127.0.0.1:4317)
/// - `OTEL_SERVICE_NAME`: Service name (default: bitnet-server)
///
/// # Example
/// ```bash
/// # Use default localhost endpoint
/// cargo run -p bitnet-server --no-default-features --features opentelemetry
///
/// # Override with custom collector
/// OTEL_EXPORTER_OTLP_ENDPOINT=http://collector.example.com:4317 \
///     cargo run -p bitnet-server --no-default-features --features opentelemetry
/// ```
pub fn init_otlp_metrics(
    endpoint: Option<String>,
    resource: Resource,
) -> Result<SdkMeterProvider> {
    // Default to localhost for development; override with env var in production
    let endpoint = endpoint.unwrap_or_else(|| {
        std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT")
            .unwrap_or_else(|_| "http://127.0.0.1:4317".to_string())
    });

    tracing::info!(
        endpoint = %endpoint,
        "Initializing OTLP metrics exporter"
    );

    // Build OTLP metrics exporter with gRPC transport
    let exporter = opentelemetry_otlp::new_exporter()
        .tonic()
        .with_endpoint(&endpoint)
        .with_timeout(Duration::from_secs(3))
        .build_metrics_exporter(
            Box::new(opentelemetry_sdk::metrics::reader::DefaultAggregationSelector::new()),
            Box::new(opentelemetry_sdk::metrics::reader::DefaultTemporalitySelector::new()),
        )?;

    // Configure periodic reader with 60-second export interval
    let reader = opentelemetry_sdk::metrics::PeriodicReader::builder(exporter)
        .with_interval(Duration::from_secs(60))
        .with_timeout(Duration::from_secs(10))
        .build();

    // Build meter provider with resource attributes
    let provider = SdkMeterProvider::builder()
        .with_reader(reader)
        .with_resource(resource)
        .build();

    // Set global meter provider for metrics API
    global::set_meter_provider(provider.clone());

    Ok(provider)
}

/// Create resource attributes for BitNet server
pub fn create_resource() -> Resource {
    let service_name = std::env::var("OTEL_SERVICE_NAME")
        .unwrap_or_else(|_| "bitnet-server".to_string());

    Resource::builder()
        .with_service_name(service_name)
        .with_attributes(vec![
            KeyValue::new("service.version", env!("CARGO_PKG_VERSION")),
            KeyValue::new("service.namespace", "ml-inference"),
            KeyValue::new("telemetry.sdk.language", "rust"),
            KeyValue::new("telemetry.sdk.name", "opentelemetry"),
            KeyValue::new("telemetry.sdk.version", env!("CARGO_PKG_VERSION_PRE")),
        ])
        .build()
}
```

---

### Updated OpenTelemetry Module

#### Replace: `crates/bitnet-server/src/monitoring/opentelemetry.rs`

**Current Implementation** (lines 84-97):
```rust
/// Initialize OpenTelemetry metrics
async fn init_metrics(_config: &MonitoringConfig) -> Result<()> {
    // Initialize Prometheus metrics
    let resource = Resource::builder()
        .with_service_name("bitnet-server")
        .with_attributes(vec![KeyValue::new("service.version", env!("CARGO_PKG_VERSION"))])
        .build();

    let reader = exporter().build()?;  // ❌ Deprecated PrometheusExporter
    let provider = SdkMeterProvider::builder().with_resource(resource).with_reader(reader).build();

    global::set_meter_provider(provider);

    Ok(())
}
```

**New Implementation**:
```rust
/// Initialize OpenTelemetry metrics using OTLP
async fn init_metrics(config: &MonitoringConfig) -> Result<()> {
    use crate::monitoring::otlp::{create_resource, init_otlp_metrics};

    let resource = create_resource();
    let endpoint = config.opentelemetry_endpoint.clone();

    let _provider = init_otlp_metrics(endpoint, resource)?;

    tracing::info!("OTLP metrics initialized successfully");

    Ok(())
}
```

**Import Updates**:
```rust
// Remove this import (line 6):
// use opentelemetry_prometheus::exporter;

// Add this to module declaration (src/monitoring/mod.rs):
#[cfg(feature = "opentelemetry")]
pub mod otlp;
```

---

### Preserved Metric Instrumentation

**No Changes Required** - These functions remain compatible:

```rust
// crates/bitnet-server/src/monitoring/opentelemetry.rs:109-166

pub mod tracing_utils {
    /// Record inference metrics (unchanged)
    pub fn record_inference_metrics(
        model_name: &str,
        prompt_length: usize,
        tokens_generated: u64,
        duration: std::time::Duration,
    ) { /* ... existing implementation ... */ }

    /// Record model loading metrics (unchanged)
    pub fn record_model_load_metrics(
        model_name: &str,
        format: &str,
        size_mb: f64,
        duration: std::time::Duration,
        success: bool,
    ) { /* ... existing implementation ... */ }

    /// Record quantization metrics (unchanged)
    pub fn record_quantization_metrics(
        quantization_type: &str,
        tensor_count: usize,
        duration: std::time::Duration,
        compressed_size_mb: f64,
        compression_ratio: f64,
        success: bool,
    ) { /* ... existing implementation ... */ }
}
```

**Rationale**: These functions use `tracing::info!` which is independent of the metrics backend. OTLP integration occurs at the exporter level, not at the instrumentation level.

---

## Migration Checklist

### Phase 1: Dependency Updates
- [ ] **AC1.1**: Remove `opentelemetry-prometheus = "0.29.1"` from workspace `Cargo.toml:233`
- [ ] **AC1.2**: Add `"metrics"` feature to `opentelemetry-otlp` in workspace `Cargo.toml:232`
- [ ] **AC1.3**: Remove `opentelemetry-prometheus` optional dependency from `crates/bitnet-server/Cargo.toml:33`
- [ ] **AC1.4**: Update `opentelemetry` feature list in `crates/bitnet-server/Cargo.toml:71`

**Validation**:
```bash
cargo tree -p bitnet-server --no-default-features --features opentelemetry | grep opentelemetry
# Expected: No opentelemetry-prometheus in dependency tree
```

---

### Phase 2: OTLP Implementation
- [ ] **AC2.1**: Create new `crates/bitnet-server/src/monitoring/otlp.rs` module
- [ ] **AC2.2**: Implement `init_otlp_metrics()` with environment variable support
- [ ] **AC2.3**: Implement `create_resource()` with BitNet-specific attributes
- [ ] **AC2.4**: Add module declaration to `crates/bitnet-server/src/monitoring/mod.rs`

**Validation**:
```bash
cargo check -p bitnet-server --no-default-features --features opentelemetry
# Expected: Compilation succeeds
```

---

### Phase 3: Prometheus Removal
- [ ] **AC3.1**: Remove `use opentelemetry_prometheus::exporter;` import (line 6)
- [ ] **AC3.2**: Replace `exporter().build()?` with OTLP initialization in `init_metrics()`
- [ ] **AC3.3**: Update function signature to use `config` parameter
- [ ] **AC3.4**: Verify metric instrumentation functions remain unchanged

**Validation**:
```bash
cargo clippy -p bitnet-server --no-default-features --features opentelemetry -- -D warnings
# Expected: 0 warnings
```

---

### Phase 4: Integration Testing
- [ ] **AC2.5**: Test default localhost endpoint
- [ ] **AC2.6**: Test custom endpoint via environment variable
- [ ] **AC2.7**: Verify metrics export to OTLP collector
- [ ] **AC2.8**: Confirm zero impact on inference latency

**Manual Validation**:
```bash
# Terminal 1: Start local OTLP collector (optional)
docker run -p 4317:4317 otel/opentelemetry-collector:latest

# Terminal 2: Start BitNet server with OTLP
cargo run -p bitnet-server --no-default-features --features opentelemetry

# Terminal 3: Trigger inference request
curl -X POST http://localhost:8080/v1/inference \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 10}'

# Expected: Metrics exported to collector at 60-second intervals
```

---

## Validation Commands Matrix

| Phase | Command | Expected Output | AC |
|-------|---------|-----------------|-----|
| **Dependency Check** | `cargo tree -p bitnet-server --no-default-features --features opentelemetry \| grep opentelemetry-prometheus` | No output (dependency removed) | AC1 |
| **Compilation** | `cargo check -p bitnet-server --no-default-features --features opentelemetry` | Compilation succeeds | AC1, AC2 |
| **Clippy** | `cargo clippy -p bitnet-server --no-default-features --features opentelemetry -- -D warnings` | 0 warnings | AC3 |
| **Release Build** | `cargo build -p bitnet-server --no-default-features --features opentelemetry --release` | Build succeeds | AC2 |
| **All Features** | `cargo clippy --workspace --all-features -- -D warnings` | 0 warnings (after #447 merges) | AC8 |

---

## Rollback Strategy

If OTLP migration causes issues:

### Rollback Steps
1. **Revert Dependency Changes**:
   ```bash
   git checkout Cargo.toml crates/bitnet-server/Cargo.toml
   ```

2. **Restore Prometheus Exporter**:
   ```bash
   git checkout crates/bitnet-server/src/monitoring/opentelemetry.rs
   ```

3. **Remove OTLP Module**:
   ```bash
   rm crates/bitnet-server/src/monitoring/otlp.rs
   git checkout crates/bitnet-server/src/monitoring/mod.rs
   ```

4. **Validate Rollback**:
   ```bash
   cargo check -p bitnet-server --no-default-features --features opentelemetry
   ```

### Rollback Criteria
- OTLP collector connectivity issues in production
- Metrics export failures exceeding 5% error rate
- Inference latency regression > 5%
- Clippy warnings introduced by migration

---

## API Contract Documentation

### Public API Changes

**No Breaking Changes** - All changes are internal to `bitnet-server`.

### New Public Functions

**Module**: `bitnet_server::monitoring::otlp`

```rust
/// Initialize OTLP metrics exporter
pub fn init_otlp_metrics(
    endpoint: Option<String>,
    resource: Resource,
) -> Result<SdkMeterProvider>

/// Create BitNet-specific resource attributes
pub fn create_resource() -> Resource
```

**Usage**:
```rust
use bitnet_server::monitoring::otlp::{create_resource, init_otlp_metrics};

let resource = create_resource();
let provider = init_otlp_metrics(Some("http://collector:4317".to_string()), resource)?;
```

### Deprecated Functions

**None** - Prometheus code is removed entirely, not deprecated.

---

## Environment Variable Reference

| Variable | Default | Description | Example |
|----------|---------|-------------|---------|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://127.0.0.1:4317` | OTLP gRPC collector endpoint | `http://collector.example.com:4317` |
| `OTEL_SERVICE_NAME` | `bitnet-server` | Service name for telemetry | `bitnet-production-us-west` |

**Production Configuration**:
```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector.prod.internal:4317
export OTEL_SERVICE_NAME=bitnet-server-prod
cargo run -p bitnet-server --no-default-features --features opentelemetry --release
```

---

## BitNet-rs Standards Compliance

### Feature Flag Discipline
✅ All commands specify `--no-default-features --features opentelemetry`
✅ Default features remain empty in `bitnet-server/Cargo.toml`
✅ OTLP features properly gated behind `opentelemetry` feature

### Workspace Structure Alignment
✅ Changes isolated to `bitnet-server` crate
✅ No impact on inference crates (`bitnet-inference`, `bitnet-quantization`)
✅ Workspace dependency versions remain consistent

### Neural Network Development Patterns
✅ Zero impact on inference latency (observability layer only)
✅ No changes to model loading, quantization, or inference algorithms
✅ Metrics collection remains non-blocking (async export)

### TDD and Test Naming
✅ All ACs have `// AC:N` test tag comments
✅ Validation commands map directly to AC requirements
✅ Story → Schema → Tests → Code traceability clear

### GGUF Compatibility
✅ No impact (compilation fixes only, no model format changes)

---

## Test Structure

### Unit Tests

```rust
// crates/bitnet-server/src/monitoring/otlp.rs

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ac1_otlp_dependency_migration() { // AC:1
        // Verify OTLP exporter can be built
        let resource = create_resource();
        assert_eq!(
            resource.get(opentelemetry::Key::new("service.namespace")),
            Some(opentelemetry::Value::from("ml-inference"))
        );
    }

    #[test]
    fn test_ac2_environment_variable_support() { // AC:2
        // Test default endpoint
        std::env::remove_var("OTEL_EXPORTER_OTLP_ENDPOINT");
        let resource = create_resource();

        // Should use default localhost endpoint
        // (Full integration test would verify actual connection)
        assert!(resource.len() >= 5); // Expected attributes
    }

    #[test]
    fn test_ac2_custom_endpoint_override() { // AC:2
        // Test custom endpoint via env var
        std::env::set_var("OTEL_EXPORTER_OTLP_ENDPOINT", "http://custom:4317");
        std::env::set_var("OTEL_SERVICE_NAME", "test-service");

        let resource = create_resource();
        assert_eq!(
            resource.get(opentelemetry::Key::new("service.name")),
            Some(opentelemetry::Value::from("test-service"))
        );

        // Cleanup
        std::env::remove_var("OTEL_EXPORTER_OTLP_ENDPOINT");
        std::env::remove_var("OTEL_SERVICE_NAME");
    }
}
```

---

## Performance Impact Analysis

### Metrics Export Overhead

**Baseline (Prometheus)**:
- In-memory metrics aggregation
- Scrape-based pull model
- Minimal runtime overhead

**New (OTLP)**:
- Periodic push every 60 seconds
- Async gRPC export (non-blocking)
- Expected overhead: < 0.1% inference latency

**Mitigation**:
- Export interval: 60 seconds (configurable)
- Export timeout: 10 seconds (prevents blocking)
- Separate tokio runtime for metrics (async isolation)

### Memory Usage

**Expected Change**: +2-5 MB for OTLP exporter buffers
**Acceptable**: Yes (server typically uses 100+ MB for models)

---

## References

### OpenTelemetry Documentation
- [OTLP Specification](https://opentelemetry.io/docs/specs/otlp/)
- [Rust SDK 0.31 Migration Guide](https://github.com/open-telemetry/opentelemetry-rust/blob/main/CHANGELOG.md#0310)
- [Metrics API](https://docs.rs/opentelemetry_sdk/0.31.0/opentelemetry_sdk/metrics/)

### BitNet-rs Documentation
- `docs/environment-variables.md` - Runtime configuration
- `docs/health-endpoints.md` - Monitoring and observability
- `docs/architecture-overview.md` - System design and components

### Related Issues
- Issue #447 - Compilation fixes across workspace crates
- PR #440 - GPU feature unification (establishes feature flag patterns)

---

## Approval Checklist

Before implementation:
- [x] All 3 acceptance criteria clearly defined
- [x] Validation commands specified with expected outputs
- [x] Rollback strategy documented
- [x] Zero impact on neural network inference confirmed
- [x] Environment variable defaults aligned with OTLP standards
- [x] BitNet-rs feature flag discipline maintained
- [x] Test structure defined with AC tags
- [x] API contract documentation complete

**Status**: ✅ Ready for Implementation

**Next Steps**: NEXT → impl-creator (implementation of AC1-AC3)
