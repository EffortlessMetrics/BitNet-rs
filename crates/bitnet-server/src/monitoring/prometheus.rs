//! Prometheus metrics integration

use anyhow::Result;
use axum::{
    extract::State,
    http::{header, StatusCode},
    response::Response,
    routing::get,
    Router,
};
// Note: PrometheusBuilder removed due to API compatibility issues
use prometheus::{Encoder, TextEncoder};
use std::sync::Arc;

use super::MonitoringConfig;

/// Prometheus metrics exporter
pub struct PrometheusExporter {
    registry: prometheus::Registry,
    encoder: TextEncoder,
}

impl PrometheusExporter {
    /// Initialize Prometheus metrics exporter
    pub fn new(config: &MonitoringConfig) -> Result<Self> {
        let registry = prometheus::Registry::new();
        let encoder = TextEncoder::new();

        tracing::info!(
            endpoint = %config.prometheus_path,
            "Prometheus metrics exporter initialized"
        );

        Ok(Self { registry, encoder })
    }

    /// Export metrics in Prometheus format
    pub fn export_metrics(&self) -> Result<String> {
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        self.encoder.encode(&metric_families, &mut buffer)?;
        Ok(String::from_utf8(buffer)?)
    }

    /// Register custom metrics collectors
    pub fn register_custom_collectors(&self) -> Result<()> {
        // Register BitNet-specific metrics collectors
        let bitnet_collector = BitNetMetricsCollector::new();
        self.registry.register(Box::new(bitnet_collector))?;

        Ok(())
    }
}

/// Custom Prometheus collector for BitNet-specific metrics
struct BitNetMetricsCollector {
    // Custom metrics that aren't covered by the standard metrics crate
}

impl BitNetMetricsCollector {
    fn new() -> Self {
        Self {}
    }
}

impl prometheus::core::Collector for BitNetMetricsCollector {
    fn desc(&self) -> Vec<&prometheus::core::Desc> {
        // Return descriptions of custom metrics
        vec![]
    }

    fn collect(&self) -> Vec<prometheus::proto::MetricFamily> {
        // Collect custom metrics
        vec![]
    }
}

/// Create Prometheus metrics route
pub fn create_prometheus_routes(exporter: Arc<PrometheusExporter>) -> Router {
    Router::new().route("/metrics", get(metrics_handler)).with_state(exporter)
}

/// Prometheus metrics endpoint handler
async fn metrics_handler(
    State(exporter): State<Arc<PrometheusExporter>>,
) -> Result<Response<String>, StatusCode> {
    match exporter.export_metrics() {
        Ok(metrics) => {
            let response = Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")
                .body(metrics)
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
            Ok(response)
        }
        Err(e) => {
            tracing::error!("Failed to export Prometheus metrics: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Standard ML inference metrics for Prometheus
pub mod standard_metrics {
    use prometheus::{
        register_counter, register_counter_vec, register_gauge, register_gauge_vec, register_histogram_vec, Counter, CounterVec, Gauge, GaugeVec, HistogramVec,
    };
    use std::sync::OnceLock;

    // Request metrics
    static REQUESTS_TOTAL: OnceLock<CounterVec> = OnceLock::new();
    static REQUESTS_DURATION: OnceLock<HistogramVec> = OnceLock::new();
    static REQUESTS_ACTIVE: OnceLock<Gauge> = OnceLock::new();

    // Inference metrics
    static TOKENS_GENERATED_TOTAL: OnceLock<Counter> = OnceLock::new();
    static TOKENS_PER_SECOND: OnceLock<Gauge> = OnceLock::new();
    static INFERENCE_DURATION: OnceLock<HistogramVec> = OnceLock::new();

    // Model metrics
    static MODEL_LOAD_DURATION: OnceLock<HistogramVec> = OnceLock::new();
    static MODEL_MEMORY_USAGE: OnceLock<GaugeVec> = OnceLock::new();

    // System metrics
    static MEMORY_USAGE_BYTES: OnceLock<Gauge> = OnceLock::new();
    static CPU_USAGE_PERCENT: OnceLock<Gauge> = OnceLock::new();
    static GPU_MEMORY_USAGE_BYTES: OnceLock<Gauge> = OnceLock::new();

    // Error metrics
    static ERRORS_TOTAL: OnceLock<CounterVec> = OnceLock::new();

    /// Initialize all Prometheus metrics
    pub fn init_metrics() -> Result<(), prometheus::Error> {
        // Request metrics
        REQUESTS_TOTAL
            .set(register_counter_vec!(
                "bitnet_requests_total",
                "Total number of inference requests",
                &["method", "endpoint", "status"]
            )?)
            .map_err(|_| prometheus::Error::AlreadyReg)?;

        REQUESTS_DURATION
            .set(register_histogram_vec!(
                "bitnet_request_duration_seconds",
                "Request duration in seconds",
                &["method", "endpoint"],
                vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
            )?)
            .map_err(|_| prometheus::Error::AlreadyReg)?;

        REQUESTS_ACTIVE
            .set(register_gauge!("bitnet_requests_active", "Number of active inference requests")?)
            .map_err(|_| prometheus::Error::AlreadyReg)?;

        // Inference metrics
        TOKENS_GENERATED_TOTAL
            .set(register_counter!(
                "bitnet_tokens_generated_total",
                "Total number of tokens generated"
            )?)
            .map_err(|_| prometheus::Error::AlreadyReg)?;

        TOKENS_PER_SECOND
            .set(register_gauge!(
                "bitnet_tokens_per_second",
                "Current tokens generated per second"
            )?)
            .map_err(|_| prometheus::Error::AlreadyReg)?;

        INFERENCE_DURATION
            .set(register_histogram_vec!(
                "bitnet_inference_duration_seconds",
                "Inference duration in seconds",
                &["model", "quantization"],
                vec![0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 60.0]
            )?)
            .map_err(|_| prometheus::Error::AlreadyReg)?;

        // Model metrics
        MODEL_LOAD_DURATION
            .set(register_histogram_vec!(
                "bitnet_model_load_duration_seconds",
                "Model loading duration in seconds",
                &["model", "format"],
                vec![0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0]
            )?)
            .map_err(|_| prometheus::Error::AlreadyReg)?;

        MODEL_MEMORY_USAGE
            .set(register_gauge_vec!(
                "bitnet_model_memory_usage_bytes",
                "Model memory usage in bytes",
                &["model", "component"]
            )?)
            .map_err(|_| prometheus::Error::AlreadyReg)?;

        // System metrics
        MEMORY_USAGE_BYTES
            .set(register_gauge!("bitnet_memory_usage_bytes", "Process memory usage in bytes")?)
            .map_err(|_| prometheus::Error::AlreadyReg)?;

        CPU_USAGE_PERCENT
            .set(register_gauge!("bitnet_cpu_usage_percent", "CPU usage percentage")?)
            .map_err(|_| prometheus::Error::AlreadyReg)?;

        GPU_MEMORY_USAGE_BYTES
            .set(register_gauge!("bitnet_gpu_memory_usage_bytes", "GPU memory usage in bytes")?)
            .map_err(|_| prometheus::Error::AlreadyReg)?;

        // Error metrics
        ERRORS_TOTAL
            .set(register_counter_vec!(
                "bitnet_errors_total",
                "Total number of errors",
                &["type", "component"]
            )?)
            .map_err(|_| prometheus::Error::AlreadyReg)?;

        Ok(())
    }

    /// Get requests total counter
    pub fn requests_total() -> &'static CounterVec {
        REQUESTS_TOTAL.get().expect("Metrics not initialized")
    }

    /// Get request duration histogram
    pub fn requests_duration() -> &'static HistogramVec {
        REQUESTS_DURATION.get().expect("Metrics not initialized")
    }

    /// Get active requests gauge
    pub fn requests_active() -> &'static Gauge {
        REQUESTS_ACTIVE.get().expect("Metrics not initialized")
    }

    /// Get tokens generated total counter
    pub fn tokens_generated_total() -> &'static Counter {
        TOKENS_GENERATED_TOTAL.get().expect("Metrics not initialized")
    }

    /// Get tokens per second gauge
    pub fn tokens_per_second() -> &'static Gauge {
        TOKENS_PER_SECOND.get().expect("Metrics not initialized")
    }

    /// Get inference duration histogram
    pub fn inference_duration() -> &'static HistogramVec {
        INFERENCE_DURATION.get().expect("Metrics not initialized")
    }

    /// Get model load duration histogram
    pub fn model_load_duration() -> &'static HistogramVec {
        MODEL_LOAD_DURATION.get().expect("Metrics not initialized")
    }

    /// Get model memory usage gauge
    pub fn model_memory_usage() -> &'static GaugeVec {
        MODEL_MEMORY_USAGE.get().expect("Metrics not initialized")
    }

    /// Get memory usage gauge
    pub fn memory_usage_bytes() -> &'static Gauge {
        MEMORY_USAGE_BYTES.get().expect("Metrics not initialized")
    }

    /// Get CPU usage gauge
    pub fn cpu_usage_percent() -> &'static Gauge {
        CPU_USAGE_PERCENT.get().expect("Metrics not initialized")
    }

    /// Get GPU memory usage gauge
    pub fn gpu_memory_usage_bytes() -> &'static Gauge {
        GPU_MEMORY_USAGE_BYTES.get().expect("Metrics not initialized")
    }

    /// Get errors total counter
    pub fn errors_total() -> &'static CounterVec {
        ERRORS_TOTAL.get().expect("Metrics not initialized")
    }
}
