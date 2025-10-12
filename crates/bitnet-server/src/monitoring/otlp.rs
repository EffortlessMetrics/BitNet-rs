//! OTLP metrics initialization for BitNet.rs server observability
//!
//! This module provides OpenTelemetry Protocol (OTLP) metrics export
//! functionality with gRPC transport, replacing the deprecated Prometheus exporter.

use anyhow::Result;
use opentelemetry::{KeyValue, global};
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::{Resource, metrics::SdkMeterProvider};
use std::time::Duration;

/// Initialize OTLP metrics provider with gRPC exporter
///
/// Creates a PeriodicReader with 60s export interval and configures
/// the OTLP exporter to send metrics to the specified endpoint.
///
/// # Arguments
///
/// * `endpoint` - Optional OTLP collector endpoint (defaults to http://127.0.0.1:4317)
/// * `resource` - OpenTelemetry Resource with service attributes
///
/// # Errors
///
/// Returns error if OTLP exporter initialization fails or endpoint is unreachable.
pub fn init_otlp_metrics(endpoint: Option<String>, resource: Resource) -> Result<SdkMeterProvider> {
    let endpoint = endpoint.unwrap_or_else(|| {
        std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT")
            .unwrap_or_else(|_| "http://127.0.0.1:4317".to_string())
    });

    // Create OTLP metrics exporter with tonic gRPC transport
    let exporter = opentelemetry_otlp::MetricExporter::builder()
        .with_tonic()
        .with_endpoint(endpoint)
        .with_timeout(Duration::from_secs(3))
        .build()?;

    let reader = opentelemetry_sdk::metrics::PeriodicReader::builder(exporter)
        .with_interval(Duration::from_secs(60))
        .build();

    let provider = SdkMeterProvider::builder().with_reader(reader).with_resource(resource).build();

    global::set_meter_provider(provider.clone());
    Ok(provider)
}

/// Create OpenTelemetry Resource with BitNet-specific attributes
///
/// Sets resource attributes including service name, version, namespace,
/// and SDK information for telemetry correlation.
pub fn create_resource() -> Resource {
    let service_name =
        std::env::var("OTEL_SERVICE_NAME").unwrap_or_else(|_| "bitnet-server".to_string());

    Resource::builder()
        .with_attributes(vec![
            KeyValue::new("service.name", service_name),
            KeyValue::new("service.namespace", "ml-inference"),
            KeyValue::new("service.version", env!("CARGO_PKG_VERSION")),
            KeyValue::new("telemetry.sdk.language", "rust"),
            KeyValue::new("telemetry.sdk.name", "opentelemetry"),
        ])
        .build()
}
