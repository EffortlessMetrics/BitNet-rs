//! # Tracing and Observability Integration Example
//!
//! This example demonstrates how to integrate BitNet.rs with tracing, metrics collection,
//! and observability tools for production monitoring and debugging.

use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{info, warn, error, debug, trace, instrument, Span};
use tracing_subscriber::{
    layer::SubscriberExt,
    util::SubscriberInitExt,
    fmt,
    EnvFilter,
};
use opentelemetry::{
    global,
    sdk::{
        trace::{self, TracerProvider},
        Resource,
    },
    KeyValue,
};
use opentelemetry_jaeger::new_agent_pipeline;
use prometheus::{
    Counter, Histogram, Gauge, Registry, Encoder, TextEncoder,
    register_counter, register_histogram, register_gauge,
    opts, histogram_opts,
};

// Import BitNet components
use bitnet_inference::{InferenceEngine, GenerationConfig};
use bitnet_models::{Model, ModelLoader};
use bitnet_tokenizers::TokenizerBuilder;
use bitnet_common::{BitNetConfig, Device};

/// Metrics collector for BitNet operations
#[derive(Clone)]
struct BitNetMetrics {
    // Request metrics
    requests_total: Counter,
    request_duration: Histogram,
    active_requests: Gauge,

    // Model metrics
    tokens_generated_total: Counter,
    model_load_duration: Histogram,
    model_memory_usage: Gauge,

    // Error metrics
    errors_total: Counter,

    // Performance metrics
    inference_latency: Histogram,
    throughput_tokens_per_second: Gauge,

    // System metrics
    cpu_usage: Gauge,
    memory_usage: Gauge,
    gpu_utilization: Gauge,
}

impl BitNetMetrics {
    fn new() -> Result<Self> {
        Ok(Self {
            requests_total: register_counter!(
                "bitnet_requests_total",
                "Total number of inference requests"
            )?,
            request_duration: register_histogram!(
                histogram_opts!(
                    "bitnet_request_duration_seconds",
                    "Request duration in seconds",
                    vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
                )
            )?,
            active_requests: register_gauge!(
                "bitnet_active_requests",
                "Number of currently active requests"
            )?,
            tokens_generated_total: register_counter!(
                "bitnet_tokens_generated_total",
                "Total number of tokens generated"
            )?,
            model_load_duration: register_histogram!(
                histogram_opts!(
                    "bitnet_model_load_duration_seconds",
                    "Model loading duration in seconds"
                )
            )?,
            model_memory_usage: register_gauge!(
                "bitnet_model_memory_bytes",
                "Model memory usage in bytes"
            )?,
            errors_total: register_counter!(
                "bitnet_errors_total",
                "Total number of errors"
            )?,
            inference_latency: register_histogram!(
                histogram_opts!(
                    "bitnet_inference_latency_seconds",
                    "Inference latency in seconds",
                    vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
                )
            )?,
            throughput_tokens_per_second: register_gauge!(
                "bitnet_throughput_tokens_per_second",
                "Current throughput in tokens per second"
            )?,
            cpu_usage: register_gauge!(
                "bitnet_cpu_usage_percent",
                "CPU usage percentage"
            )?,
            memory_usage: register_gauge!(
                "bitnet_memory_usage_bytes",
                "Memory usage in bytes"
            )?,
            gpu_utilization: register_gauge!(
                "bitnet_gpu_utilization_percent",
                "GPU utilization percentage"
            )?,
        })
    }
}

/// Instrumented BitNet service with comprehensive observability
struct ObservableBitNetService {
    engine: Arc<RwLock<InferenceEngine>>,
    metrics: BitNetMetrics,
    config: BitNetConfig,
}

impl ObservableBitNetService {
    async fn new() -> Result<Self> {
        let metrics = BitNetMetrics::new()?;

        // Load model with timing
        let load_start = Instant::now();
        let engine = Self::load_model_with_tracing().await?;
        let load_duration = load_start.elapsed();

        metrics.model_load_duration.observe(load_duration.as_secs_f64());
        metrics.model_memory_usage.set(estimate_model_memory_usage());

        info!(
            duration_ms = load_duration.as_millis(),
            "Model loaded successfully"
        );

        Ok(Self {
            engine: Arc::new(RwLock::new(engine)),
            metrics,
            config: BitNetConfig::default(),
        })
    }

    #[instrument(skip(self))]
    async fn generate_text(&self, prompt: &str, config: &GenerationConfig) -> Result<String> {
        let _timer = self.metrics.request_duration.start_timer();
        self.metrics.requests_total.inc();
        self.metrics.active_requests.inc();

        // Create a span for this request
        let span = tracing::info_span!(
            "generate_text",
            prompt_length = prompt.len(),
            max_tokens = config.max_new_tokens,
            temperature = config.temperature,
        );

        let result = async move {
            let start_time = Instant::now();

            debug!("Starting text generation");
            trace!(prompt = %prompt, "Input prompt");

            // Perform inference with detailed tracing
            let generated_text = {
                let mut engine = self.engine.write().await;

                // Add custom attributes to the span
                span.record("model_device", format!("{:?}", Device::Cpu));

                engine.generate_with_config(prompt, config)
                    .map_err(|e| {
                        self.metrics.errors_total.inc();
                        error!(error = %e, "Inference failed");
                        e
                    })?
            };

            let duration = start_time.elapsed();
            let tokens_generated = generated_text.split_whitespace().count() as u64;

            // Update metrics
            self.metrics.inference_latency.observe(duration.as_secs_f64());
            self.metrics.tokens_generated_total.inc_by(tokens_generated);

            // Calculate throughput
            let tokens_per_second = tokens_generated as f64 / duration.as_secs_f64();
            self.metrics.throughput_tokens_per_second.set(tokens_per_second);

            // Add more span attributes
            span.record("tokens_generated", tokens_generated);
            span.record("duration_ms", duration.as_millis());
            span.record("tokens_per_second", tokens_per_second);

            info!(
                tokens_generated = tokens_generated,
                duration_ms = duration.as_millis(),
                tokens_per_second = tokens_per_second,
                "Text generation completed"
            );

            trace!(generated_text = %generated_text, "Generated text");

            Ok(generated_text)
        }.instrument(span).await;

        self.metrics.active_requests.dec();
        result
    }

    #[instrument]
    async fn load_model_with_tracing() -> Result<InferenceEngine> {
        info!("Loading BitNet model with tracing");

        // Simulate model loading with detailed tracing
        let model_path = std::env::var("BITNET_MODEL_PATH")
            .unwrap_or_else(|_| "models/bitnet-1.58b.gguf".to_string());

        debug!(model_path = %model_path, "Loading model from path");

        // Load model (mock implementation)
        let model = load_mock_model().await?;

        // Load tokenizer
        let tokenizer_name = "gpt2";
        debug!(tokenizer = %tokenizer_name, "Loading tokenizer");
        let tokenizer = TokenizerBuilder::from_pretrained(tokenizer_name)?;

        // Create inference engine
        let device = Device::Cpu;
        debug!(device = ?device, "Creating inference engine");

        let engine = InferenceEngine::new(model, tokenizer, device)?;

        info!("Model and inference engine loaded successfully");
        Ok(engine)
    }

    /// Get metrics in Prometheus format
    fn get_metrics(&self) -> Result<String> {
        let encoder = TextEncoder::new();
        let metric_families = prometheus::gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer)?;
        Ok(String::from_utf8(buffer)?)
    }

    /// Update system metrics
    async fn update_system_metrics(&self) {
        // In a real implementation, these would collect actual system metrics
        self.metrics.cpu_usage.set(get_cpu_usage());
        self.metrics.memory_usage.set(get_memory_usage());
        self.metrics.gpu_utilization.set(get_gpu_utilization());
    }
}

/// Initialize comprehensive tracing and observability
async fn initialize_observability() -> Result<()> {
    // Initialize OpenTelemetry tracer
    let tracer = new_agent_pipeline()
        .with_service_name("bitnet-rs")
        .with_tags(vec![
            KeyValue::new("version", env!("CARGO_PKG_VERSION")),
            KeyValue::new("environment", "development"),
        ])
        .install_simple()?;

    // Initialize tracing subscriber with multiple layers
    tracing_subscriber::registry()
        .with(
            tracing_opentelemetry::layer().with_tracer(tracer)
        )
        .with(
            fmt::layer()
                .with_target(false)
                .with_thread_ids(true)
                .with_level(true)
                .with_file(true)
                .with_line_number(true)
        )
        .with(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,bitnet=debug"))
        )
        .init();

    info!("Observability initialized with OpenTelemetry and Prometheus");
    Ok(())
}

/// Comprehensive observability example
#[tokio::main]
async fn main() -> Result<()> {
    // Initialize observability stack
    initialize_observability().await?;

    info!("Starting BitNet observability example");

    // Create observable service
    let service = ObservableBitNetService::new().await?;

    // Start system metrics collection
    let metrics_service = service.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(10));
        loop {
            interval.tick().await;
            metrics_service.update_system_metrics().await;
        }
    });

    // Start metrics server
    let metrics_service = service.clone();
    tokio::spawn(async move {
        start_metrics_server(metrics_service).await;
    });

    // Run example inference requests with different scenarios
    run_inference_examples(&service).await?;

    // Keep the application running to observe metrics
    info!("Observability example running. Check metrics at http://localhost:9090/metrics");
    tokio::time::sleep(Duration::from_secs(60)).await;

    // Shutdown tracing
    global::shutdown_tracer_provider();

    Ok(())
}

/// Run various inference examples to generate observability data
#[instrument(skip(service))]
async fn run_inference_examples(service: &ObservableBitNetService) -> Result<()> {
    info!("Running inference examples to generate observability data");

    let examples = vec![
        ("Hello, world!", GenerationConfig { max_new_tokens: 20, temperature: 0.7, ..Default::default() }),
        ("The future of AI is", GenerationConfig { max_new_tokens: 50, temperature: 0.8, ..Default::default() }),
        ("Once upon a time", GenerationConfig { max_new_tokens: 100, temperature: 0.9, ..Default::default() }),
        ("Explain quantum computing", GenerationConfig { max_new_tokens: 200, temperature: 0.6, ..Default::default() }),
    ];

    for (i, (prompt, config)) in examples.iter().enumerate() {
        let span = tracing::info_span!("example_request", request_id = i);

        let result = async {
            info!(prompt = %prompt, "Running example request");

            match service.generate_text(prompt, config).await {
                Ok(generated) => {
                    info!(
                        generated_length = generated.len(),
                        "Example request completed successfully"
                    );
                    debug!(generated_text = %generated, "Generated text");
                }
                Err(e) => {
                    error!(error = %e, "Example request failed");
                }
            }

            // Add some delay between requests
            tokio::time::sleep(Duration::from_millis(500)).await;
        }.instrument(span).await;
    }

    // Simulate some error scenarios
    simulate_error_scenarios(service).await?;

    info!("Inference examples completed");
    Ok(())
}

/// Simulate error scenarios for observability testing
#[instrument(skip(service))]
async fn simulate_error_scenarios(service: &ObservableBitNetService) -> Result<()> {
    info!("Simulating error scenarios");

    // Simulate various error conditions
    let error_scenarios = vec![
        ("", "Empty prompt"),
        ("x".repeat(20000).as_str(), "Extremely long prompt"),
    ];

    for (prompt, scenario) in error_scenarios {
        let span = tracing::warn_span!("error_scenario", scenario = scenario);

        let _result = async {
            warn!(scenario = scenario, "Testing error scenario");

            let config = GenerationConfig::default();
            if let Err(e) = service.generate_text(prompt, &config).await {
                warn!(error = %e, scenario = scenario, "Expected error occurred");
            }
        }.instrument(span).await;
    }

    Ok(())
}

/// Start Prometheus metrics server
async fn start_metrics_server(service: ObservableBitNetService) {
    use warp::Filter;

    let metrics_route = warp::path("metrics")
        .map(move || {
            match service.get_metrics() {
                Ok(metrics) => warp::reply::with_header(
                    metrics,
                    "content-type",
                    "text/plain; version=0.0.4; charset=utf-8"
                ),
                Err(e) => {
                    error!(error = %e, "Failed to generate metrics");
                    warp::reply::with_header(
                        format!("Error generating metrics: {}", e),
                        "content-type",
                        "text/plain"
                    )
                }
            }
        });

    let health_route = warp::path("health")
        .map(|| {
            warp::reply::json(&serde_json::json!({
                "status": "healthy",
                "timestamp": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
            }))
        });

    let routes = metrics_route.or(health_route);

    info!("Starting metrics server on http://0.0.0.0:9090");
    warp::serve(routes)
        .run(([0, 0, 0, 0], 9090))
        .await;
}

/// Mock model loading for demonstration
async fn load_mock_model() -> Result<Box<dyn Model>> {
    // Simulate loading time
    tokio::time::sleep(Duration::from_millis(100)).await;
    Ok(Box::new(MockModel::new()))
}

/// Estimate model memory usage (mock implementation)
fn estimate_model_memory_usage() -> f64 {
    // In practice, this would calculate actual memory usage
    1024.0 * 1024.0 * 512.0 // 512 MB
}

/// Get CPU usage (mock implementation)
fn get_cpu_usage() -> f64 {
    // In practice, this would get actual CPU usage
    use rand::Rng;
    let mut rng = rand::thread_rng();
    rng.gen_range(10.0..80.0)
}

/// Get memory usage (mock implementation)
fn get_memory_usage() -> f64 {
    // In practice, this would get actual memory usage
    use rand::Rng;
    let mut rng = rand::thread_rng();
    rng.gen_range(1024.0 * 1024.0 * 100.0..1024.0 * 1024.0 * 1000.0) // 100MB - 1GB
}

/// Get GPU utilization (mock implementation)
fn get_gpu_utilization() -> f64 {
    // In practice, this would get actual GPU utilization
    use rand::Rng;
    let mut rng = rand::thread_rng();
    rng.gen_range(0.0..100.0)
}

/// Mock model for demonstration
struct MockModel {
    config: BitNetConfig,
}

impl MockModel {
    fn new() -> Self {
        Self {
            config: BitNetConfig::default(),
        }
    }
}

impl Model for MockModel {
    type Config = BitNetConfig;

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn forward(
        &self,
        _input: &dyn bitnet_common::Tensor,
        _cache: &mut bitnet_inference::KVCache,
    ) -> Result<Box<dyn bitnet_common::Tensor>> {
        // Mock implementation with some processing time
        std::thread::sleep(Duration::from_millis(10));
        Ok(Box::new(MockTensor::new(vec![1, 50257])))
    }

    fn generate(&self, _tokens: &[u32], _config: &bitnet_inference::GenerationConfig) -> Result<Vec<u32>> {
        // Mock generation with some processing time
        std::thread::sleep(Duration::from_millis(50));
        Ok(vec![1234, 5678, 9012])
    }
}

/// Mock tensor for demonstration
struct MockTensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl MockTensor {
    fn new(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Self {
            shape,
            data: vec![0.1; size],
        }
    }
}

impl bitnet_common::Tensor for MockTensor {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn dtype(&self) -> bitnet_common::DType {
        bitnet_common::DType::F32
    }

    fn device(&self) -> &bitnet_common::Device {
        &bitnet_common::Device::Cpu
    }

    fn as_slice<T>(&self) -> Result<&[T]> {
        unsafe {
            let ptr = self.data.as_ptr() as *const T;
            let slice = std::slice::from_raw_parts(ptr, self.data.len());
            Ok(slice)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_metrics_creation() {
        let metrics = BitNetMetrics::new().unwrap();

        // Test that metrics can be updated
        metrics.requests_total.inc();
        metrics.tokens_generated_total.inc_by(10);
        metrics.active_requests.set(5.0);

        // Verify metrics are working
        assert!(metrics.requests_total.get() > 0.0);
        assert!(metrics.tokens_generated_total.get() > 0.0);
        assert_eq!(metrics.active_requests.get(), 5.0);
    }

    #[tokio::test]
    async fn test_observable_service_creation() {
        // This test would require proper initialization in a real scenario
        // For now, we'll test the mock components
        let model = MockModel::new();
        assert_eq!(model.config().vocab_size, 50257);
    }
}
