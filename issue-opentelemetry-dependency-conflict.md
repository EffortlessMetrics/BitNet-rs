# [Dependency Conflict] opentelemetry-prometheus is discontinued and conflicts with opentelemetry 0.30.0

## Problem Description

The `opentelemetry-prometheus` crate used in `crates/bitnet-server/src/monitoring/opentelemetry.rs` is discontinued and no longer maintained. It depends on an incompatible version of the OpenTelemetry SDK (0.29.1) while other OpenTelemetry crates in the workspace use version 0.30.0, causing build failures and trait bound conflicts.

## Environment

- **File**: `crates/bitnet-server/src/monitoring/opentelemetry.rs`
- **Dependency**: `opentelemetry-prometheus = "0.29.1"`
- **Conflict**: `opentelemetry = "0.30.0"`, `opentelemetry-sdk = "0.30.0"`
- **Crate**: `bitnet-server`
- **Feature Flag**: `opentelemetry`

## Build Error Analysis

The build fails with trait bound incompatibility:

```
error[E0277]: the trait bound `PrometheusExporter: MetricReader` is not satisfied
   --> crates/bitnet-server/src/monitoring/opentelemetry.rs:85:84
    |
 85 |     let provider = SdkMeterProvider::builder().with_resource(resource).with_reader(reader).build();
    |                                                                        ----------- ^^^^^^ the trait `opentelemetry_sdk::metrics::reader::MetricReader` is not implemented for `opentelemetry_prometheus::PrometheusExporter`
    |                                                                        |
    |                                                                        required by a bound introduced by this call
```

## Root Cause Analysis

1. **Discontinued Crate**: `opentelemetry-prometheus` is no longer maintained by the OpenTelemetry team
2. **Version Mismatch**: The crate depends on OpenTelemetry SDK 0.29.1, while workspace uses 0.30.0
3. **Breaking API Changes**: OpenTelemetry 0.30.0 introduced breaking changes to the `MetricReader` trait
4. **Ecosystem Migration**: OpenTelemetry community has moved to OTLP-based metrics export
5. **Dependency Resolution Conflicts**: Cargo cannot resolve the conflicting OpenTelemetry versions

## Impact Assessment

**Severity**: High - Build Blocking
**Affected Components**:
- BitNet server monitoring and observability
- Prometheus metrics export functionality
- Production deployment monitoring stack
- OpenTelemetry integration

**Deployment Impact**:
- Cannot build with `opentelemetry` feature enabled
- Prometheus metrics collection non-functional
- Monitoring stack integration broken
- Production observability compromised

## Proposed Solution

### Primary Solution: Migrate to OTLP Exporter

Replace the deprecated Prometheus exporter with the standard OTLP exporter:

```toml
# In crates/bitnet-server/Cargo.toml
[dependencies]
# Remove discontinued dependency
# opentelemetry-prometheus = "0.29.1"

# Add OTLP exporter (maintained by OpenTelemetry team)
opentelemetry-otlp = { version = "0.30.0", features = ["metrics"], optional = true }
opentelemetry = { version = "0.30.0", features = ["metrics"], optional = true }
opentelemetry-sdk = { version = "0.30.0", features = ["metrics"], optional = true }

[features]
opentelemetry = ["dep:opentelemetry", "dep:opentelemetry-sdk", "dep:opentelemetry-otlp"]
```

```rust
// In crates/bitnet-server/src/monitoring/opentelemetry.rs
use opentelemetry_otlp::{WithExportConfig, MetricExporter};
use opentelemetry_sdk::metrics::{SdkMeterProvider, PeriodicReader};
use opentelemetry::global;

pub fn init_opentelemetry_metrics() -> Result<()> {
    // Configure OTLP endpoint (configurable via environment)
    let otlp_endpoint = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT")
        .unwrap_or_else(|_| "http://localhost:4317".to_string());

    // Create OTLP metrics exporter
    let exporter = opentelemetry_otlp::new_exporter()
        .tonic()
        .with_endpoint(&otlp_endpoint)
        .build_metrics_exporter()?;

    // Create periodic reader with OTLP exporter
    let reader = PeriodicReader::builder(exporter, opentelemetry_sdk::runtime::Tokio)
        .with_interval(std::time::Duration::from_secs(30))
        .build();

    // Create meter provider with resource information
    let resource = opentelemetry_sdk::Resource::new(vec![
        opentelemetry::KeyValue::new("service.name", "bitnet-server"),
        opentelemetry::KeyValue::new("service.version", env!("CARGO_PKG_VERSION")),
    ]);

    let provider = SdkMeterProvider::builder()
        .with_resource(resource)
        .with_reader(reader)
        .build();

    // Set global meter provider
    global::set_meter_provider(provider);

    info!("OpenTelemetry metrics initialized with OTLP endpoint: {}", otlp_endpoint);
    Ok(())
}

pub fn shutdown_opentelemetry() {
    if let Err(e) = global::shutdown_meter_provider() {
        warn!("Failed to shutdown OpenTelemetry meter provider: {}", e);
    }
}
```

### Alternative Solution: Use OpenTelemetry-Prometheus Bridge

If direct Prometheus format is required, use the official bridge:

```toml
# In crates/bitnet-server/Cargo.toml
[dependencies]
opentelemetry = { version = "0.30.0", features = ["metrics"] }
opentelemetry-sdk = { version = "0.30.0", features = ["metrics"] }
prometheus = "0.13"
```

```rust
// Custom Prometheus bridge implementation
use prometheus::{Encoder, TextEncoder, Registry, Counter, Histogram, Gauge};
use opentelemetry_sdk::metrics::{SdkMeterProvider, ManualReader};

pub struct PrometheusMetricsExporter {
    registry: Registry,
    reader: ManualReader,
}

impl PrometheusMetricsExporter {
    pub fn new() -> Result<Self> {
        let registry = Registry::new();
        let reader = ManualReader::builder().build();

        Ok(Self { registry, reader })
    }

    pub fn export_metrics(&self) -> Result<String> {
        // Collect metrics from OpenTelemetry
        let metrics = self.reader.collect()?;

        // Convert to Prometheus format
        let mut buffer = Vec::new();
        let encoder = TextEncoder::new();

        // Convert OpenTelemetry metrics to Prometheus format
        for metric in metrics {
            self.convert_metric_to_prometheus(&metric)?;
        }

        encoder.encode(&self.registry.gather(), &mut buffer)?;
        Ok(String::from_utf8(buffer)?)
    }

    fn convert_metric_to_prometheus(&self, metric: &Metric) -> Result<()> {
        // Implementation to convert OpenTelemetry metrics to Prometheus format
        // This is more complex but provides native Prometheus compatibility
        todo!("Implement metric conversion")
    }
}
```

### Migration Strategy for Production

Configure Prometheus to scrape OTLP collector:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'bitnet-server-otlp'
    static_configs:
      - targets: ['otel-collector:8889']  # OTLP collector Prometheus endpoint
    scrape_interval: 30s
    metrics_path: /metrics
```

```yaml
# otel-collector.yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

exporters:
  prometheus:
    endpoint: "0.0.0.0:8889"
    namespace: bitnet

processors:
  batch:

service:
  pipelines:
    metrics:
      receivers: [otlp]
      processors: [batch]
      exporters: [prometheus]
```

## Implementation Plan

### Phase 1: Dependency Update
- [ ] Remove `opentelemetry-prometheus` dependency from Cargo.toml
- [ ] Add `opentelemetry-otlp` dependency with metrics features
- [ ] Update OpenTelemetry SDK to consistent 0.30.0 version
- [ ] Resolve any remaining dependency conflicts

### Phase 2: Code Migration
- [ ] Replace Prometheus exporter with OTLP exporter in monitoring code
- [ ] Update configuration to use OTLP endpoint settings
- [ ] Add environment variable configuration for OTLP endpoint
- [ ] Update error handling for new exporter type

### Phase 3: Infrastructure Setup
- [ ] Deploy OpenTelemetry Collector with Prometheus exporter
- [ ] Update Prometheus configuration to scrape OTLP collector
- [ ] Test end-to-end metrics pipeline
- [ ] Validate metric names and labels compatibility

### Phase 4: Testing & Documentation
- [ ] Add integration tests for OTLP metrics export
- [ ] Update documentation for new monitoring setup
- [ ] Create migration guide for existing deployments
- [ ] Add troubleshooting guide for OTLP configuration

## Testing Strategy

### Build Testing
```bash
# Test build with opentelemetry feature
cargo build --no-default-features --features opentelemetry

# Test without feature (should still build)
cargo build --no-default-features

# Run tests with feature enabled
cargo test --features opentelemetry
```

### Integration Testing
```rust
#[test]
#[cfg(feature = "opentelemetry")]
async fn test_otlp_metrics_export() {
    // Start test OTLP collector
    let collector = start_test_otlp_collector().await;

    // Initialize OpenTelemetry with test endpoint
    std::env::set_var("OTEL_EXPORTER_OTLP_ENDPOINT", collector.endpoint());
    init_opentelemetry_metrics().unwrap();

    // Create and record some metrics
    let meter = global::meter("test");
    let counter = meter.u64_counter("test_counter").init();
    counter.add(1, &[]);

    // Force export and verify metrics received
    tokio::time::sleep(Duration::from_secs(1)).await;
    let metrics = collector.received_metrics().await;
    assert!(!metrics.is_empty());
}
```

### Performance Testing
```rust
#[test]
fn test_metrics_export_performance() {
    let start = Instant::now();

    // Record many metrics
    for i in 0..1000 {
        record_test_metric(i);
    }

    let duration = start.elapsed();
    assert!(duration < Duration::from_millis(100), "Metrics export too slow");
}
```

## Related Issues/PRs

- OpenTelemetry ecosystem migration to version 0.30.0
- BitNet server monitoring and observability improvements
- Production deployment monitoring stack updates
- Dependency management and version consistency

## Acceptance Criteria

- [ ] Build succeeds with `opentelemetry` feature enabled
- [ ] OTLP metrics export functional and tested
- [ ] No version conflicts in OpenTelemetry dependencies
- [ ] Prometheus metrics available via OTLP collector
- [ ] Configuration via environment variables works
- [ ] Performance impact is negligible (<5% overhead)
- [ ] Integration tests cover OTLP export functionality
- [ ] Documentation updated for new monitoring setup
- [ ] Migration path provided for existing deployments
- [ ] Backward compatibility maintained where possible

## Notes

The migration to OTLP is aligned with OpenTelemetry ecosystem direction and provides better long-term maintainability. The OTLP approach is more flexible and allows integration with multiple monitoring backends beyond Prometheus.

Consider implementing feature flags to support both OTLP and direct Prometheus export during transition period if needed for specific deployment requirements.
