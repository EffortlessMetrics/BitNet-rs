# [Dependencies] Migrate from discontinued opentelemetry-prometheus to OTLP exporter

## Problem Description

The `opentelemetry-prometheus` crate used in BitNet.rs is discontinued and causes build failures due to version incompatibilities with other OpenTelemetry crates. The workspace uses `opentelemetry` version 0.30.0, while `opentelemetry-prometheus` is stuck at 0.29.1, creating a dependency conflict that prevents successful compilation.

## Environment
- **Affected Files**:
  - `crates/bitnet-server/Cargo.toml`
  - `crates/bitnet-server/src/monitoring/opentelemetry.rs`
  - `Cargo.toml` (workspace dependencies)
- **OpenTelemetry Version**: 0.30.0 (workspace)
- **Prometheus Exporter Version**: 0.29.1 (incompatible)
- **Feature Flags**: `opentelemetry` feature
- **MSRV**: Rust 1.90.0

## Reproduction Steps

1. Enable OpenTelemetry feature and attempt to build:
   ```bash
   cargo build --features opentelemetry
   ```

2. Observe the compilation error:
   ```
   error[E0277]: the trait bound `PrometheusExporter: MetricReader` is not satisfied
     --> crates/bitnet-server/src/monitoring/opentelemetry.rs:85:84
      |
   85 |     let provider = SdkMeterProvider::builder().with_resource(resource).with_reader(reader).build();
      |                                                                        ----------- ^^^^^^ the trait `opentelemetry_sdk::metrics::reader::MetricReader` is not implemented for `opentelemetry_prometheus::PrometheusExporter`
      |                                                                        |
      |                                                                        required by a bound introduced by this call
   ```

3. Check dependency resolution:
   ```bash
   cargo tree | grep opentelemetry
   ```

**Expected Results**:
- Build should succeed with OpenTelemetry monitoring enabled
- Metrics should be exported to Prometheus-compatible endpoint
- All OpenTelemetry dependencies should be compatible

**Actual Results**:
- Build fails due to trait incompatibility
- Cannot use OpenTelemetry monitoring features
- Dependency version conflict prevents compilation

## Root Cause Analysis

### Dependency Conflict Details

The issue stems from incompatible versions of OpenTelemetry crates:

```toml
# Current workspace dependencies (Cargo.toml)
opentelemetry = "0.30.0"
opentelemetry-sdk = "0.30.0"
opentelemetry-prometheus = "0.29.1"  # ← PROBLEM: Discontinued, incompatible version
```

### OpenTelemetry Ecosystem Changes

1. **Prometheus Exporter Discontinuation**: The `opentelemetry-prometheus` crate was discontinued in favor of OTLP export with Prometheus scraping
2. **API Breaking Changes**: OpenTelemetry 0.30.0 introduced breaking changes that `opentelemetry-prometheus` 0.29.1 doesn't support
3. **Trait Evolution**: The `MetricReader` trait interface changed, making the old Prometheus exporter incompatible

### Current Implementation

```rust
// crates/bitnet-server/src/monitoring/opentelemetry.rs
use opentelemetry_prometheus::PrometheusExporter;
use opentelemetry_sdk::metrics::SdkMeterProvider;

pub fn init_metrics_exporter(endpoint: &str) -> Result<SdkMeterProvider, Box<dyn std::error::Error>> {
    let exporter = PrometheusExporter::builder()
        .with_default_histogram_boundaries(vec![
            0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
        ])
        .build()?;

    let reader = exporter; // PrometheusExporter no longer implements MetricReader

    let provider = SdkMeterProvider::builder()
        .with_resource(resource)
        .with_reader(reader)  // ← Fails here
        .build();

    Ok(provider)
}
```

## Impact Assessment

- **Severity**: Medium-High (blocks monitoring features)
- **Build Impact**:
  - Cannot compile with `opentelemetry` feature enabled
  - Dependency resolution failures
  - Prevents monitoring in production deployments

- **Monitoring Impact**:
  - No metrics export capability when OpenTelemetry is enabled
  - Cannot integrate with Prometheus monitoring stack
  - Loss of observability in production environments

- **Development Impact**:
  - Blocks development of monitoring features
  - Prevents testing of telemetry integration
  - Forces workarounds or feature disabling

## Proposed Solution

Migrate from the discontinued `opentelemetry-prometheus` to the OTLP (OpenTelemetry Protocol) exporter with Prometheus scraping support. This follows the current OpenTelemetry ecosystem direction and provides better compatibility.

### Technical Implementation

#### 1. Update Dependencies

```toml
# crates/bitnet-server/Cargo.toml
[dependencies]
# Remove discontinued prometheus exporter
# opentelemetry-prometheus = "0.29.1"  # ← Remove this

# Add OTLP exporter with HTTP transport
opentelemetry-otlp = { version = "0.30.0", features = ["http-proto", "reqwest-client"] }
opentelemetry = "0.30.0"
opentelemetry-sdk = "0.30.0"
opentelemetry-semantic-conventions = "0.30.0"
```

#### 2. OTLP HTTP Metrics Implementation

```rust
// crates/bitnet-server/src/monitoring/opentelemetry.rs
use opentelemetry::global;
use opentelemetry::KeyValue;
use opentelemetry_otlp::{WithExportConfig, ExportConfig};
use opentelemetry_sdk::{
    metrics::{
        reader::{DefaultAggregationSelector, DefaultTemporalitySelector},
        PeriodicReader, SdkMeterProvider,
    },
    Resource,
};
use std::time::Duration;

/// Initialize OTLP metrics exporter for Prometheus scraping
pub fn init_otlp_metrics_exporter(
    otlp_endpoint: &str,
    collection_interval: Duration,
) -> Result<SdkMeterProvider, Box<dyn std::error::Error + Send + Sync>> {

    // Configure OTLP HTTP exporter
    let export_config = ExportConfig {
        endpoint: otlp_endpoint.to_string(),
        timeout: Duration::from_secs(10),
        protocol: opentelemetry_otlp::Protocol::HttpBinary,
    };

    let exporter = opentelemetry_otlp::new_exporter()
        .http()
        .with_export_config(export_config)
        .build_metrics_exporter(
            Box::new(DefaultAggregationSelector::new()),
            Box::new(DefaultTemporalitySelector::new()),
        )?;

    // Create periodic reader for metrics collection
    let reader = PeriodicReader::builder(exporter, tokio::runtime::Handle::current())
        .with_interval(collection_interval)
        .build();

    // Set up resource attributes
    let resource = Resource::new(vec![
        KeyValue::new("service.name", "bitnet-inference-server"),
        KeyValue::new("service.version", env!("CARGO_PKG_VERSION")),
        KeyValue::new("service.namespace", "bitnet"),
    ]);

    // Build meter provider
    let provider = SdkMeterProvider::builder()
        .with_resource(resource)
        .with_reader(reader)
        .build();

    // Set global meter provider
    global::set_meter_provider(provider.clone());

    Ok(provider)
}

/// Alternative: Direct Prometheus metrics endpoint (without OTLP)
pub fn init_prometheus_direct_exporter(
    bind_address: &str,
) -> Result<SdkMeterProvider, Box<dyn std::error::Error + Send + Sync>> {
    use opentelemetry_sdk::metrics::reader::MetricReader;
    use prometheus::{Encoder, TextEncoder, Registry};
    use std::collections::HashMap;

    // Custom Prometheus metrics reader implementation
    let prometheus_reader = PrometheusMetricReader::new(bind_address)?;

    let resource = Resource::new(vec![
        KeyValue::new("service.name", "bitnet-inference-server"),
        KeyValue::new("service.version", env!("CARGO_PKG_VERSION")),
    ]);

    let provider = SdkMeterProvider::builder()
        .with_resource(resource)
        .with_reader(prometheus_reader)
        .build();

    global::set_meter_provider(provider.clone());

    Ok(provider)
}
```

#### 3. Custom Prometheus Metrics Reader (Alternative Approach)

```rust
// crates/bitnet-server/src/monitoring/prometheus_reader.rs
use opentelemetry_sdk::metrics::{
    data::{ResourceMetrics, Metric},
    reader::{AggregationSelector, MetricReader, TemporalitySelector},
    InstrumentKind, Pipeline, Reader,
};
use prometheus::{Registry, Encoder, TextEncoder};
use std::sync::{Arc, Mutex};
use tokio::sync::RwLock;

pub struct PrometheusMetricReader {
    registry: Arc<Registry>,
    server_handle: Option<tokio::task::JoinHandle<()>>,
    collected_metrics: Arc<RwLock<Vec<ResourceMetrics>>>,
}

impl PrometheusMetricReader {
    pub fn new(bind_address: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let registry = Arc::new(Registry::new());
        let collected_metrics = Arc::new(RwLock::new(Vec::new()));

        // Start HTTP server for Prometheus scraping
        let server_handle = Self::start_prometheus_server(
            bind_address.to_string(),
            registry.clone(),
            collected_metrics.clone(),
        )?;

        Ok(Self {
            registry,
            server_handle: Some(server_handle),
            collected_metrics,
        })
    }

    fn start_prometheus_server(
        bind_address: String,
        registry: Arc<Registry>,
        metrics: Arc<RwLock<Vec<ResourceMetrics>>>,
    ) -> Result<tokio::task::JoinHandle<()>, Box<dyn std::error::Error + Send + Sync>> {

        let handle = tokio::spawn(async move {
            let app = axum::Router::new()
                .route("/metrics", axum::routing::get({
                    let metrics = metrics.clone();
                    move || Self::handle_metrics_request(metrics)
                }));

            let listener = tokio::net::TcpListener::bind(&bind_address)
                .await
                .expect("Failed to bind metrics server");

            axum::serve(listener, app)
                .await
                .expect("Metrics server failed");
        });

        Ok(handle)
    }

    async fn handle_metrics_request(
        metrics: Arc<RwLock<Vec<ResourceMetrics>>>,
    ) -> Result<String, axum::http::StatusCode> {
        let metrics_data = metrics.read().await;
        let prometheus_text = Self::convert_to_prometheus_format(&metrics_data);
        Ok(prometheus_text)
    }

    fn convert_to_prometheus_format(metrics: &[ResourceMetrics]) -> String {
        let mut output = String::new();

        for resource_metric in metrics {
            for scope_metric in &resource_metric.scope_metrics {
                for metric in &scope_metric.metrics {
                    match metric {
                        Metric::Gauge(gauge_data) => {
                            output.push_str(&format!("# TYPE {} gauge\n", metric.name));
                            for data_point in &gauge_data.data_points {
                                output.push_str(&format!(
                                    "{} {}\n",
                                    metric.name,
                                    data_point.value
                                ));
                            }
                        }
                        Metric::Sum(sum_data) => {
                            output.push_str(&format!("# TYPE {} counter\n", metric.name));
                            for data_point in &sum_data.data_points {
                                output.push_str(&format!(
                                    "{} {}\n",
                                    metric.name,
                                    data_point.value
                                ));
                            }
                        }
                        Metric::Histogram(histogram_data) => {
                            output.push_str(&format!("# TYPE {} histogram\n", metric.name));
                            for data_point in &histogram_data.data_points {
                                // Convert OpenTelemetry histogram to Prometheus format
                                output.push_str(&Self::format_histogram_for_prometheus(
                                    &metric.name,
                                    data_point,
                                ));
                            }
                        }
                    }
                }
            }
        }

        output
    }
}

impl MetricReader for PrometheusMetricReader {
    fn collect(&self, rm: &mut ResourceMetrics) -> Result<(), opentelemetry_sdk::metrics::MetricError> {
        // Store collected metrics for Prometheus scraping
        let metrics = rm.clone();
        tokio::spawn({
            let collected_metrics = self.collected_metrics.clone();
            async move {
                let mut data = collected_metrics.write().await;
                data.clear();
                data.push(metrics);
            }
        });

        Ok(())
    }

    fn force_flush(&self, _timeout: Duration) -> Result<(), opentelemetry_sdk::metrics::MetricError> {
        // Prometheus reader doesn't need explicit flushing
        Ok(())
    }

    fn shutdown(&self, _timeout: Duration) -> Result<(), opentelemetry_sdk::metrics::MetricError> {
        if let Some(handle) = &self.server_handle {
            handle.abort();
        }
        Ok(())
    }
}
```

#### 4. Configuration and Integration

```rust
// crates/bitnet-server/src/config.rs
#[derive(Debug, Clone, serde::Deserialize)]
pub struct MonitoringConfig {
    pub enabled: bool,
    pub metrics_export: MetricsExportConfig,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub enum MetricsExportConfig {
    /// Export via OTLP to collector that forwards to Prometheus
    Otlp {
        endpoint: String,
        collection_interval_seconds: u64,
    },
    /// Direct Prometheus metrics endpoint
    Prometheus {
        bind_address: String,
    },
    /// Disabled
    None,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            metrics_export: MetricsExportConfig::Prometheus {
                bind_address: "0.0.0.0:9090".to_string(),
            },
        }
    }
}

// crates/bitnet-server/src/main.rs
async fn setup_monitoring(config: &MonitoringConfig) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if !config.enabled {
        return Ok(());
    }

    match &config.metrics_export {
        MetricsExportConfig::Otlp { endpoint, collection_interval_seconds } => {
            let collection_interval = Duration::from_secs(*collection_interval_seconds);
            monitoring::init_otlp_metrics_exporter(endpoint, collection_interval)?;
            log::info!("Initialized OTLP metrics export to {}", endpoint);
        }
        MetricsExportConfig::Prometheus { bind_address } => {
            monitoring::init_prometheus_direct_exporter(bind_address)?;
            log::info!("Started Prometheus metrics server on {}", bind_address);
        }
        MetricsExportConfig::None => {
            log::info!("Metrics export disabled");
        }
    }

    Ok(())
}
```

#### 5. Docker and Prometheus Configuration

```yaml
# docker/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  # Direct Prometheus scraping (Option 1)
  - job_name: 'bitnet-server'
    static_configs:
      - targets: ['bitnet-server:9090']
    scrape_interval: 10s

  # OTLP collector scraping (Option 2)
  - job_name: 'otel-collector'
    static_configs:
      - targets: ['otel-collector:8889']
    scrape_interval: 10s
```

```yaml
# docker/otel-collector.yml (for OTLP approach)
receivers:
  otlp:
    protocols:
      http:
        endpoint: 0.0.0.0:4318

exporters:
  prometheus:
    endpoint: "0.0.0.0:8889"

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

### Phase 1: Dependencies Update (Week 1)
- [ ] Remove `opentelemetry-prometheus` dependency
- [ ] Add `opentelemetry-otlp` dependency
- [ ] Update all OpenTelemetry crates to compatible versions
- [ ] Fix compilation errors

### Phase 2: OTLP Implementation (Week 2)
- [ ] Implement OTLP HTTP metrics exporter
- [ ] Add configuration for OTLP endpoint
- [ ] Test OTLP collector integration
- [ ] Add comprehensive error handling

### Phase 3: Alternative Prometheus Implementation (Week 3)
- [ ] Implement direct Prometheus metrics reader
- [ ] Add HTTP server for metrics scraping
- [ ] Convert OpenTelemetry metrics to Prometheus format
- [ ] Add configuration options

### Phase 4: Testing and Documentation (Week 4)
- [ ] Comprehensive testing of both approaches
- [ ] Update documentation for new configuration
- [ ] Add Docker compose examples
- [ ] Performance testing and optimization

## Testing Strategy

### Unit Tests
```rust
#[cfg(feature = "opentelemetry")]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_otlp_exporter_initialization() {
        let provider = init_otlp_metrics_exporter(
            "http://localhost:4318/v1/metrics",
            Duration::from_secs(10),
        ).await.unwrap();

        // Verify provider is functional
        let meter = provider.meter("test");
        let counter = meter.u64_counter("test_counter").init();
        counter.add(1, &[]);

        // Provider should be set globally
        let global_meter = global::meter("test");
        assert_eq!(meter.library().name, global_meter.library().name);
    }

    #[tokio::test]
    async fn test_prometheus_direct_exporter() {
        let provider = init_prometheus_direct_exporter("127.0.0.1:0").await.unwrap();

        let meter = provider.meter("test");
        let gauge = meter.f64_gauge("test_gauge").init();
        gauge.record(42.0, &[]);

        // Test metrics endpoint
        let client = reqwest::Client::new();
        let response = client
            .get("http://127.0.0.1:9090/metrics")
            .send()
            .await
            .unwrap();

        let body = response.text().await.unwrap();
        assert!(body.contains("test_gauge"));
        assert!(body.contains("42"));
    }
}
```

### Integration Tests
```rust
#[cfg(feature = "opentelemetry")]
mod integration_tests {
    #[tokio::test]
    async fn test_full_monitoring_pipeline() {
        // Start OTLP collector
        let collector = start_test_otel_collector().await;

        // Initialize BitNet server with OTLP monitoring
        let config = MonitoringConfig {
            enabled: true,
            metrics_export: MetricsExportConfig::Otlp {
                endpoint: collector.otlp_endpoint(),
                collection_interval_seconds: 1,
            },
        };

        setup_monitoring(&config).await.unwrap();

        // Generate some metrics
        emit_test_metrics().await;

        // Verify metrics reach Prometheus via collector
        tokio::time::sleep(Duration::from_secs(2)).await;
        let metrics = collector.get_prometheus_metrics().await;

        assert!(metrics.contains("bitnet_inference_requests_total"));
        assert!(metrics.contains("bitnet_model_load_duration_seconds"));
    }
}
```

## Migration Guide

### For Users Currently Using OpenTelemetry

1. **Update Configuration**:
   ```toml
   # Old configuration (remove)
   # [monitoring.prometheus]
   # endpoint = "0.0.0.0:9090"

   # New configuration (add)
   [monitoring.metrics_export]
   type = "otlp"
   endpoint = "http://otel-collector:4318/v1/metrics"
   collection_interval_seconds = 30
   ```

2. **Update Docker Compose**:
   ```yaml
   # Add OpenTelemetry collector service
   services:
     otel-collector:
       image: otel/opentelemetry-collector-contrib:latest
       ports:
         - "4318:4318"   # OTLP HTTP
         - "8889:8889"   # Prometheus metrics
       volumes:
         - ./otel-collector.yml:/etc/otel-collector-config.yml
       command: ["--config=/etc/otel-collector-config.yml"]
   ```

3. **Update Prometheus Configuration**:
   ```yaml
   scrape_configs:
     - job_name: 'bitnet-via-otel'
       static_configs:
         - targets: ['otel-collector:8889']
   ```

## Performance Impact

- **OTLP Overhead**: ~1-3% additional CPU usage for metrics export
- **Memory Usage**: Minimal increase (~10-50MB) for metrics buffering
- **Network**: Periodic HTTP requests to OTLP collector
- **Latency**: No impact on inference latency (async export)

## Acceptance Criteria

- [ ] Build succeeds with `opentelemetry` feature enabled
- [ ] OTLP metrics export works with standard collectors
- [ ] Direct Prometheus export works without collectors
- [ ] All existing metrics continue to be exported
- [ ] Configuration is backward-compatible where possible
- [ ] Performance overhead remains minimal
- [ ] Documentation updated for new configuration
- [ ] Migration guide provided for existing users
- [ ] Docker compose examples work out of the box

## Dependencies

- `opentelemetry-otlp` 0.30.0+
- `tokio` runtime for async HTTP export
- OpenTelemetry collector (for OTLP approach)
- Prometheus server for metrics scraping

## Related Issues

- Monitoring and observability
- Production deployment readiness
- Performance monitoring
- Dependency management and updates

## Labels
- `dependencies`
- `monitoring`
- `opentelemetry`
- `prometheus`
- `priority-medium`
- `breaking-change`