//! OpenTelemetry integration for distributed tracing

use anyhow::Result;
use opentelemetry::{
    global,
    trace::{TraceError, Tracer},
    KeyValue,
};
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_prometheus::PrometheusExporter;
use std::time::Duration;

use super::MonitoringConfig;

/// Initialize OpenTelemetry tracing and metrics
pub async fn init_opentelemetry(config: &MonitoringConfig) -> Result<()> {
    // Initialize tracing
    init_tracing(config).await?;
    
    // Initialize metrics
    init_metrics(config).await?;

    tracing::info!(
        endpoint = config.opentelemetry_endpoint.as_deref().unwrap_or("default"),
        "OpenTelemetry initialized"
    );

    Ok(())
}

/// Initialize OpenTelemetry tracing
async fn init_tracing(config: &MonitoringConfig) -> Result<()> {
    let tracer = if let Some(endpoint) = &config.opentelemetry_endpoint {
        // OTLP exporter for remote collection
        opentelemetry_otlp::new_pipeline()
            .tracing()
            .with_exporter(
                opentelemetry_otlp::new_exporter()
                    .tonic()
                    .with_endpoint(endpoint)
                    .with_timeout(Duration::from_secs(3))
            )
            .with_trace_config(
                opentelemetry::sdk::trace::config()
                    .with_resource(opentelemetry::sdk::Resource::new(vec![
                        KeyValue::new("service.name", "bitnet-server"),
                        KeyValue::new("service.version", env!("CARGO_PKG_VERSION")),
                        KeyValue::new("service.namespace", "ml-inference"),
                    ]))
                    .with_sampler(opentelemetry::sdk::trace::Sampler::TraceIdRatioBased(1.0))
            )
            .install_batch(opentelemetry::runtime::Tokio)?
    } else {
        // Stdout exporter for development
        opentelemetry::sdk::trace::TracerProvider::builder()
            .with_simple_exporter(opentelemetry_stdout::SpanExporter::default())
            .with_config(
                opentelemetry::sdk::trace::config()
                    .with_resource(opentelemetry::sdk::Resource::new(vec![
                        KeyValue::new("service.name", "bitnet-server"),
                        KeyValue::new("service.version", env!("CARGO_PKG_VERSION")),
                    ]))
            )
            .build()
            .tracer("bitnet-server")
    };

    global::set_tracer_provider(tracer.provider().unwrap());

    Ok(())
}

/// Initialize OpenTelemetry metrics
async fn init_metrics(config: &MonitoringConfig) -> Result<()> {
    let exporter = PrometheusExporter::builder()
        .with_resource(opentelemetry::sdk::Resource::new(vec![
            KeyValue::new("service.name", "bitnet-server"),
            KeyValue::new("service.version", env!("CARGO_PKG_VERSION")),
        ]))
        .build()?;

    global::set_meter_provider(exporter.meter_provider().unwrap());

    Ok(())
}

/// Shutdown OpenTelemetry gracefully
pub async fn shutdown() -> Result<()> {
    global::shutdown_tracer_provider();
    tracing::info!("OpenTelemetry shutdown complete");
    Ok(())
}

/// OpenTelemetry tracing utilities
pub mod tracing_utils {
    use opentelemetry::{
        global,
        trace::{Span, SpanKind, Status, Tracer},
        Context, KeyValue,
    };
    use std::time::Instant;

    // Note: OpenTelemetry 0.29 has breaking changes with Span trait
    // For now, we'll provide simplified tracing utilities that work with the current API
    
    /// Record inference metrics (simplified version)
    pub fn record_inference_metrics(
        model_name: &str,
        prompt_length: usize,
        tokens_generated: u64,
        duration: std::time::Duration,
    ) {
        tracing::info!(
            model_name = model_name,
            prompt_length = prompt_length,
            tokens_generated = tokens_generated,
            duration_ms = duration.as_millis(),
            tokens_per_second = if duration.as_millis() > 0 {
                (tokens_generated as f64 * 1000.0) / duration.as_millis() as f64
            } else { 0.0 },
            "Inference completed"
        );
    }

    /// Record model loading metrics
    pub fn record_model_load_metrics(
        model_name: &str,
        format: &str,
        size_mb: f64,
        duration: std::time::Duration,
        success: bool,
    ) {
        tracing::info!(
            model_name = model_name,
            format = format,
            size_mb = size_mb,
            duration_ms = duration.as_millis(),
            success = success,
            "Model load completed"
        );
    }

    /// Record quantization metrics
    pub fn record_quantization_metrics(
        quantization_type: &str,
        tensor_count: usize,
        duration: std::time::Duration,
        compressed_size_mb: f64,
        compression_ratio: f64,
        success: bool,
    ) {
        tracing::info!(
            quantization_type = quantization_type,
            tensor_count = tensor_count,
            duration_ms = duration.as_millis(),
            compressed_size_mb = compressed_size_mb,
            compression_ratio = compression_ratio,
            success = success,
            "Quantization completed"
        );
    }
}

/// OpenTelemetry metrics utilities
pub mod metrics_utils {
    use opentelemetry::{
        global,
        metrics::{Counter, Histogram, Meter, UpDownCounter},
        KeyValue,
    };
    use std::sync::OnceLock;

    // Note: Static metrics removed due to OpenTelemetry 0.29 API changes
    // Using tracing-based metrics instead

    /// Initialize OpenTelemetry metrics (simplified for OpenTelemetry 0.29)
    pub fn init_metrics() {
        // Note: OpenTelemetry 0.29 has API changes
        // For now, we'll use tracing for metrics until the API stabilizes
        tracing::info!("OpenTelemetry metrics initialized (using tracing backend)");
    }

    /// Record inference request (simplified)
    pub fn record_inference_request(model: &str, quantization: &str) {
        tracing::debug!(
            model = model,
            quantization = quantization,
            "Inference request recorded"
        );
    }

    /// Record inference duration (simplified)
    pub fn record_inference_duration(duration: std::time::Duration, model: &str) {
        tracing::debug!(
            model = model,
            duration_seconds = duration.as_secs_f64(),
            "Inference duration recorded"
        );
    }

    /// Increment active requests (simplified)
    pub fn increment_active_requests() {
        tracing::debug!("Active requests incremented");
    }

    /// Decrement active requests (simplified)
    pub fn decrement_active_requests() {
        tracing::debug!("Active requests decremented");
    }

    /// Record tokens generated (simplified)
    pub fn record_tokens_generated(count: u64, model: &str) {
        tracing::debug!(
            model = model,
            tokens_generated = count,
            "Tokens generated recorded"
        );
    }

    /// Record model load (simplified)
    pub fn record_model_load(model: &str, format: &str) {
        tracing::debug!(
            model = model,
            format = format,
            "Model load recorded"
        );
    }

    /// Record model load duration (simplified)
    pub fn record_model_load_duration(duration: std::time::Duration, model: &str) {
        tracing::debug!(
            model = model,
            duration_seconds = duration.as_secs_f64(),
            "Model load duration recorded"
        );
    }
}