# Dependency issue: `opentelemetry-prometheus` is discontinued and conflicts with other `opentelemetry` crates

The `opentelemetry-prometheus` crate is discontinued and no longer maintained. It also conflicts with the other `opentelemetry` crates in the workspace, which are at version `0.30.0` while `opentelemetry-prometheus` is at `0.29.1`.

This causes the build to fail with the following error:

```
error[E0277]: the trait bound `PrometheusExporter: MetricReader` is not satisfied
   --> crates/bitnet-server/src/monitoring/opentelemetry.rs:85:84
    |
 85 |     let provider = SdkMeterProvider::builder().with_resource(resource).with_reader(reader).build();
    |                                                                        ----------- ^^^^^^ the trait `opentelemetry_sdk::metrics::reader::MetricReader` is not implemented for `opentelemetry_prometheus::PrometheusExporter`
    |                                                                        |
    |                                                                        required by a bound introduced by this call
```

## Description

The `opentelemetry-prometheus` crate is used to export metrics to Prometheus. However, this crate is discontinued and no longer maintained. It also has a dependency on an older version of the `opentelemetry_sdk` crate, which conflicts with the version used by the other `opentelemetry` crates in the workspace.

This conflict causes the build to fail when the `opentelemetry` feature is enabled.

## Proposed Fix

The recommended solution is to replace the `opentelemetry-prometheus` crate with the OTLP (OpenTelemetry Protocol) exporter. Prometheus has native support for OTLP, so this change should be relatively straightforward.

This will involve the following steps:

1.  **Remove the `opentelemetry-prometheus` dependency:** Remove the `opentelemetry-prometheus` dependency from the `Cargo.toml` files.

2.  **Add the `opentelemetry-otlp` dependency:** Add the `opentelemetry-otlp` dependency to the `Cargo.toml` files.

3.  **Update the monitoring code:** Update the monitoring code in `crates/bitnet-server/src/monitoring/opentelemetry.rs` to use the OTLP exporter instead of the Prometheus exporter.

### Example Implementation

```rust
// In crates/bitnet-server/src/monitoring/opentelemetry.rs

use opentelemetry_otlp::WithExportConfig;

// ...

    let exporter = opentelemetry_otlp::new_exporter().tonic().with_endpoint(endpoint);

    let reader = opentelemetry_sdk::metrics::reader::DefaultAggregationSelector::default();
    let provider = SdkMeterProvider::builder()
        .with_reader(
            opentelemetry_sdk::metrics::reader::PeriodicReader::builder(exporter, reader)
                .with_interval(std::time::Duration::from_secs(30))
                .build(),
        )
        .build();

// ...
```

This change will require a running OTLP collector to receive the metrics. The Prometheus configuration will also need to be updated to scrape the OTLP collector.