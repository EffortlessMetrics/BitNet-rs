# [Dependencies] Replace discontinued opentelemetry-prometheus with OTLP exporter

## Problem Description

The `opentelemetry-prometheus` crate is discontinued and causes build failures due to version conflicts with other OpenTelemetry crates. The workspace uses `opentelemetry` version 0.30.0 while `opentelemetry-prometheus` is stuck at 0.29.1, causing trait incompatibilities.

## Environment

- **File**: `crates/bitnet-server/src/monitoring/opentelemetry.rs`
- **Dependencies**: `opentelemetry-prometheus` 0.29.1 vs `opentelemetry` 0.30.0
- **Error**: `PrometheusExporter: MetricReader` trait not implemented
- **Component**: Monitoring and telemetry infrastructure
- **MSRV**: Rust 1.90.0

## Reproduction Steps

1. Enable `opentelemetry` feature in the build
2. Attempt to build the project
3. Observe trait bound error for `PrometheusExporter`
4. Note version conflict between OpenTelemetry dependencies

**Expected**: Successful build with working Prometheus metrics export
**Actual**: Build failure due to discontinued dependency incompatibility

## Root Cause Analysis

The error occurs due to incompatible trait implementations:

```rust
error[E0277]: the trait bound `PrometheusExporter: MetricReader` is not satisfied
  --> crates/bitnet-server/src/monitoring/opentelemetry.rs:85:84
   |
85 |     let provider = SdkMeterProvider::builder().with_resource(resource).with_reader(reader).build();
   |                                                                        ----------- ^^^^^^ the trait `opentelemetry_sdk::metrics::reader::MetricReader` is not implemented for `opentelemetry_prometheus::PrometheusExporter`
```

**Technical Issues:**
1. **Version Incompatibility**: Different OpenTelemetry crate versions
2. **Discontinued Maintenance**: `opentelemetry-prometheus` no longer updated
3. **Trait Evolution**: `MetricReader` trait changed between versions
4. **Ecosystem Shift**: OpenTelemetry ecosystem moved to OTLP-first approach

## Impact Assessment

**Severity**: Medium - Blocks monitoring functionality when enabled
**Type**: Dependency upgrade and architecture modernization

**Affected Components**:
- Prometheus metrics export
- Monitoring and observability infrastructure
- Production deployment telemetry
- Performance monitoring dashboards

**Business Impact**:
- Monitoring functionality completely broken
- Cannot track production performance metrics
- Missing observability for debugging issues
- Blocked adoption of modern telemetry practices

## Proposed Solution

### Primary Solution: Migrate to OTLP Exporter

Replace the discontinued Prometheus exporter with the standard OTLP exporter:

```rust
// Updated dependencies in Cargo.toml
[dependencies]
opentelemetry = "0.30.0"
opentelemetry_sdk = "0.30.0"
opentelemetry-otlp = "0.23.0"
# Remove: opentelemetry-prometheus = "0.29.1"

// Updated monitoring implementation
use opentelemetry::{global, metrics::*, KeyValue};
use opentelemetry_sdk::{
    metrics::{
        reader::{DefaultAggregationSelector, DefaultTemporalitySelector},
        PeriodicReader, SdkMeterProvider,
    },
    Resource,
};
use opentelemetry_otlp::{WithExportConfig, Protocol};
use std::time::Duration;

pub struct OpenTelemetryMonitoring {
    meter_provider: SdkMeterProvider,
    meter: Meter,
    // Metrics
    request_duration: Histogram<f64>,
    request_count: Counter<u64>,
    active_connections: UpDownCounter<i64>,
    memory_usage: Gauge<u64>,
}

impl OpenTelemetryMonitoring {
    pub fn new(config: &MonitoringConfig) -> Result<Self> {
        // Create OTLP exporter
        let otlp_exporter = opentelemetry_otlp::new_exporter()
            .tonic()
            .with_endpoint(&config.otlp_endpoint)
            .with_timeout(Duration::from_secs(10))
            .with_protocol(Protocol::Grpc)
            .build_metrics_exporter(
                Box::new(DefaultTemporalitySelector::new()),
                Box::new(DefaultAggregationSelector::new()),
            )?;

        // Create periodic reader
        let reader = PeriodicReader::builder(otlp_exporter, tokio::runtime::Handle::current())
            .with_interval(Duration::from_secs(config.export_interval_seconds))
            .build();

        // Create resource with service information
        let resource = Resource::new(vec![
            KeyValue::new("service.name", "bitnet-inference-server"),
            KeyValue::new("service.version", env!("CARGO_PKG_VERSION")),
            KeyValue::new("service.instance.id", uuid::Uuid::new_v4().to_string()),
        ]);

        // Create meter provider
        let meter_provider = SdkMeterProvider::builder()
            .with_reader(reader)
            .with_resource(resource)
            .build();

        // Set global meter provider
        global::set_meter_provider(meter_provider.clone());

        // Create meter for this service
        let meter = global::meter("bitnet-inference-server");

        // Create metrics
        let request_duration = meter
            .f64_histogram("request_duration_seconds")
            .with_description("Duration of inference requests")
            .with_unit("s")
            .init();

        let request_count = meter
            .u64_counter("requests_total")
            .with_description("Total number of inference requests")
            .init();

        let active_connections = meter
            .i64_up_down_counter("active_connections")
            .with_description("Number of active client connections")
            .init();

        let memory_usage = meter
            .u64_gauge("memory_usage_bytes")
            .with_description("Current memory usage")
            .with_unit("bytes")
            .init();

        Ok(Self {
            meter_provider,
            meter,
            request_duration,
            request_count,
            active_connections,
            memory_usage,
        })
    }

    pub fn record_request(&self, duration: Duration, method: &str, status: &str) {
        let labels = &[
            KeyValue::new("method", method.to_string()),
            KeyValue::new("status", status.to_string()),
        ];

        self.request_duration.record(duration.as_secs_f64(), labels);
        self.request_count.add(1, labels);
    }

    pub fn set_active_connections(&self, count: i64) {
        self.active_connections.add(count, &[]);
    }

    pub fn set_memory_usage(&self, bytes: u64) {
        self.memory_usage.record(bytes, &[]);
    }

    pub async fn shutdown(&self) -> Result<()> {
        self.meter_provider.shutdown()?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    pub otlp_endpoint: String,
    pub export_interval_seconds: u64,
    pub enable_trace_export: bool,
    pub service_name: String,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            otlp_endpoint: std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT")
                .unwrap_or_else(|_| "http://localhost:4317".to_string()),
            export_interval_seconds: 30,
            enable_trace_export: true,
            service_name: "bitnet-inference-server".to_string(),
        }
    }
}
```

### Integration with Inference Server

```rust
// Integration in server startup
impl InferenceServer {
    pub async fn new_with_monitoring(config: ServerConfig) -> Result<Self> {
        // Initialize monitoring if enabled
        let monitoring = if config.monitoring.enabled {
            Some(OpenTelemetryMonitoring::new(&config.monitoring.opentelemetry)?)
        } else {
            None
        };

        let mut server = Self::new(config).await?;
        server.monitoring = monitoring;
        Ok(server)
    }

    async fn handle_inference_request(&self, request: InferenceRequest) -> Result<InferenceResponse> {
        let start_time = std::time::Instant::now();

        // Process the request
        let result = self.process_inference(request).await;

        // Record metrics
        if let Some(monitoring) = &self.monitoring {
            let duration = start_time.elapsed();
            let status = if result.is_ok() { "success" } else { "error" };
            monitoring.record_request(duration, "inference", status);
        }

        result
    }
}
```

### Prometheus Integration via OTLP Collector

Create configuration for Prometheus to scrape OTLP collector:

```yaml
# otel-collector-config.yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  batch:

exporters:
  prometheus:
    endpoint: "0.0.0.0:8889"
    const_labels:
      service: bitnet-inference-server

service:
  pipelines:
    metrics:
      receivers: [otlp]
      processors: [batch]
      exporters: [prometheus]
```

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'bitnet-inference-server'
    static_configs:
      - targets: ['localhost:8889']
    scrape_interval: 10s
    metrics_path: /metrics
```

## Implementation Plan

### Phase 1: Dependency Migration (0.5 days)
- [ ] Remove `opentelemetry-prometheus` dependency
- [ ] Add `opentelemetry-otlp` dependency
- [ ] Update OpenTelemetry crates to consistent versions
- [ ] Fix import statements and dependency references

### Phase 2: OTLP Implementation (1 day)
- [ ] Implement OTLP metrics exporter
- [ ] Create monitoring configuration structure
- [ ] Add metrics initialization and lifecycle management
- [ ] Integrate with existing server infrastructure

### Phase 3: Collector Setup (0.5 days)
- [ ] Create OpenTelemetry Collector configuration
- [ ] Set up Prometheus scraping from collector
- [ ] Test end-to-end metrics pipeline
- [ ] Document deployment configuration

### Phase 4: Testing and Validation (0.5 days)
- [ ] Test metrics export functionality
- [ ] Validate Prometheus metric collection
- [ ] Verify monitoring dashboard compatibility
- [ ] Add integration tests for telemetry

## Testing Strategy

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_otlp_monitoring_initialization() {
        let config = MonitoringConfig {
            otlp_endpoint: "http://localhost:4317".to_string(),
            export_interval_seconds: 5,
            enable_trace_export: false,
            service_name: "test-service".to_string(),
        };

        let monitoring = OpenTelemetryMonitoring::new(&config).unwrap();

        // Test metric recording
        monitoring.record_request(Duration::from_millis(100), "test", "success");
        monitoring.set_active_connections(5);
        monitoring.set_memory_usage(1024 * 1024);

        // Cleanup
        monitoring.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_metrics_export() {
        // This test requires a running OTLP collector
        if std::env::var("OTEL_TEST_ENDPOINT").is_err() {
            return;
        }

        let config = MonitoringConfig::default();
        let monitoring = OpenTelemetryMonitoring::new(&config).unwrap();

        // Record some metrics
        for i in 0..10 {
            monitoring.record_request(
                Duration::from_millis(50 + i * 10),
                "inference",
                if i % 3 == 0 { "error" } else { "success" }
            );
        }

        // Wait for export
        tokio::time::sleep(Duration::from_secs(6)).await;
        monitoring.shutdown().await.unwrap();
    }
}
```

## Docker Compose for Development

```yaml
# docker-compose.monitoring.yml
version: '3.8'
services:
  otel-collector:
    image: otel/opentelemetry-collector-contrib:0.89.0
    command: ["--config=/etc/otel-collector-config.yaml"]
    volumes:
      - ./otel-collector-config.yaml:/etc/otel-collector-config.yaml
    ports:
      - "4317:4317"   # OTLP gRPC receiver
      - "4318:4318"   # OTLP HTTP receiver
      - "8889:8889"   # Prometheus metrics endpoint

  prometheus:
    image: prom/prometheus:v2.45.0
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'

  grafana:
    image: grafana/grafana:10.0.0
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana

volumes:
  grafana-storage:
```

## Configuration Examples

### Environment Variables
```bash
# OpenTelemetry configuration
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export OTEL_SERVICE_NAME=bitnet-inference-server
export OTEL_SERVICE_VERSION=0.1.0
export OTEL_RESOURCE_ATTRIBUTES=service.instance.id=$(hostname)

# BitNet server configuration
export BITNET_MONITORING_ENABLED=true
export BITNET_OTLP_ENDPOINT=http://localhost:4317
export BITNET_METRICS_INTERVAL=30
```

### Configuration File
```toml
# bitnet-server.toml
[monitoring]
enabled = true

[monitoring.opentelemetry]
otlp_endpoint = "http://localhost:4317"
export_interval_seconds = 30
enable_trace_export = true
service_name = "bitnet-inference-server"
```

## Acceptance Criteria

### Functional Requirements
- [ ] OpenTelemetry-Prometheus dependency removed successfully
- [ ] OTLP exporter functional with metrics export
- [ ] Prometheus metrics available via OTLP Collector
- [ ] All monitoring functionality preserved

### Quality Requirements
- [ ] Build succeeds without dependency conflicts
- [ ] Metrics export performance acceptable (<1% overhead)
- [ ] Error handling for collector unavailability
- [ ] Graceful shutdown with proper resource cleanup

### Documentation Requirements
- [ ] Deployment guide for OTLP Collector setup
- [ ] Configuration examples for different environments
- [ ] Migration guide from old Prometheus exporter
- [ ] Troubleshooting guide for common issues

## Related Issues/PRs

- Monitoring infrastructure modernization (#TBD)
- OpenTelemetry integration improvements (#TBD)
- Production deployment observability (#TBD)

## Labels

`dependencies`, `monitoring`, `opentelemetry`, `prometheus`, `medium-priority`, `infrastructure`

## Definition of Done

- [ ] Discontinued `opentelemetry-prometheus` dependency removed
- [ ] OTLP exporter successfully exports metrics
- [ ] Prometheus can scrape metrics via OTLP Collector
- [ ] All tests pass with new telemetry implementation
- [ ] Documentation updated with new setup instructions
- [ ] No build conflicts with OpenTelemetry dependencies
