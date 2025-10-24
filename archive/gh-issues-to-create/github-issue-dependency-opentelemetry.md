# [Critical] Replace discontinued opentelemetry-prometheus with OTLP exporter

## Problem Description

The `opentelemetry-prometheus` crate is discontinued and conflicts with other OpenTelemetry crates in the workspace, causing build failures when the `opentelemetry` feature is enabled.

## Environment
- **Affected File**: `crates/bitnet-server/src/monitoring/opentelemetry.rs:85`
- **Dependency Version Conflict**:
  - `opentelemetry-prometheus`: 0.29.1 (discontinued)
  - Other `opentelemetry` crates: 0.30.0
- **Build Configuration**: `--features opentelemetry`

## Reproduction Steps
1. Enable the opentelemetry feature: `cargo build --features opentelemetry`
2. Observe build failure with trait bound error

## Error Details
```
error[E0277]: the trait bound `PrometheusExporter: MetricReader` is not satisfied
   --> crates/bitnet-server/src/monitoring/opentelemetry.rs:85:84
    |
 85 |     let provider = SdkMeterProvider::builder().with_resource(resource).with_reader(reader).build();
    |                                                                        ----------- ^^^^^^ the trait `opentelemetry_sdk::metrics::reader::MetricReader` is not implemented for `opentelemetry_prometheus::PrometheusExporter`
```

## Root Cause Analysis

1. **Discontinued Crate**: `opentelemetry-prometheus` is no longer maintained
2. **Version Incompatibility**: The discontinued crate depends on older OpenTelemetry SDK versions
3. **Trait Incompatibility**: `PrometheusExporter` doesn't implement the updated `MetricReader` trait

## Impact Assessment
- **Severity**: Critical
- **Impact**: Complete build failure when monitoring features are enabled
- **Affected Components**:
  - `bitnet-server` monitoring system
  - Prometheus metrics export
  - Production observability
- **Business Impact**: No metrics collection in production deployments

## Proposed Solution

Replace `opentelemetry-prometheus` with the OTLP (OpenTelemetry Protocol) exporter, which has native Prometheus support.

### Implementation Plan

1. **Update Dependencies** in `Cargo.toml`:
```toml
# Remove
opentelemetry-prometheus = "0.29.1"

# Add
opentelemetry-otlp = { version = "0.30.0", features = ["metrics"] }
```

2. **Update Monitoring Code** in `crates/bitnet-server/src/monitoring/opentelemetry.rs`:
```rust
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::metrics::reader::PeriodicReader;

// Replace Prometheus exporter with OTLP
let exporter = opentelemetry_otlp::new_exporter()
    .tonic()
    .with_endpoint(endpoint);

let reader = PeriodicReader::builder(exporter, TemporalitySelector::default())
    .with_interval(std::time::Duration::from_secs(30))
    .build();

let provider = SdkMeterProvider::builder()
    .with_resource(resource)
    .with_reader(reader)
    .build();
```

3. **Update Infrastructure Configuration**:
   - Deploy OTLP collector to receive metrics
   - Configure Prometheus to scrape from OTLP collector
   - Update monitoring pipeline documentation

4. **Environment Variables**:
   - Add `OTLP_EXPORTER_ENDPOINT` configuration
   - Maintain backward compatibility where possible

### Alternative Solutions Considered

1. **Fork and maintain opentelemetry-prometheus**: Too much maintenance overhead
2. **Direct Prometheus HTTP endpoint**: Loses OpenTelemetry ecosystem benefits
3. **Different metrics library**: Would require larger refactoring

## Testing Strategy
- **Unit Tests**: Test OTLP exporter configuration and initialization
- **Integration Tests**: Verify metrics are exported correctly to OTLP collector
- **End-to-End Tests**: Confirm Prometheus can scrape metrics from OTLP collector
- **Compatibility Tests**: Ensure existing monitoring dashboards still work
- **Performance Tests**: Verify no performance regression in metrics collection

## Implementation Dependencies
- [ ] OTLP collector deployment/configuration
- [ ] Prometheus configuration update
- [ ] Documentation update for new monitoring setup
- [ ] Migration guide for existing deployments

## Acceptance Criteria
- [ ] Build succeeds with `--features opentelemetry`
- [ ] Metrics are successfully exported via OTLP
- [ ] Prometheus can scrape metrics from OTLP collector
- [ ] All existing monitoring functionality preserved
- [ ] Documentation updated with new configuration
- [ ] No performance regression in metrics collection
- [ ] CI/CD pipeline updated for new monitoring stack

## Migration Guide Required
Document the transition process for:
- Local development setup
- Production deployment changes
- Monitoring infrastructure updates
- Configuration parameter changes

## Labels
- `critical`
- `bug`
- `dependencies`
- `monitoring`
- `breaking-change`
- `priority-high`

## Related Issues
- Monitoring system architecture
- Production deployment configurations
- OpenTelemetry ecosystem alignment
