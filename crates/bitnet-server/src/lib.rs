//! HTTP server for BitNet inference with comprehensive monitoring

pub mod monitoring;
pub mod streaming;

use anyhow::Result;
use axum::{
    Router,
    extract::State,
    http::StatusCode,
    middleware,
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tower_http::{cors::CorsLayer, trace::TraceLayer};

#[cfg(feature = "prometheus")]
use monitoring::prometheus::{PrometheusExporter, create_prometheus_routes};
use monitoring::{
    MonitoringConfig, MonitoringSystem,
    health::{HealthChecker, create_health_routes},
    metrics::MetricsCollector,
};

#[derive(Deserialize)]
pub struct InferenceRequest {
    pub prompt: String,
    pub max_tokens: Option<usize>,
    pub model: Option<String>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub repetition_penalty: Option<f32>,
}

#[derive(Serialize)]
pub struct InferenceResponse {
    pub text: String,
    pub tokens_generated: u64,
    pub inference_time_ms: u64,
    pub tokens_per_second: f64,
}

/// Server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub monitoring: MonitoringConfig,
    pub model_path: Option<String>,
    pub tokenizer_path: Option<String>,
    pub device: String,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            monitoring: MonitoringConfig::default(),
            model_path: None,
            tokenizer_path: None,
            device: "cpu".to_string(),
        }
    }
}

/// BitNet server with monitoring
pub struct BitNetServer {
    config: ServerConfig,
    monitoring: Arc<MonitoringSystem>,
    health_checker: Arc<HealthChecker>,
    #[cfg(feature = "prometheus")]
    prometheus_exporter: Option<Arc<PrometheusExporter>>,
    engine: Option<Arc<RwLock<bitnet_inference::InferenceEngine>>>,
}

impl BitNetServer {
    /// Create a new BitNet server
    pub async fn new(config: ServerConfig) -> Result<Self> {
        // Initialize monitoring system
        let monitoring = Arc::new(MonitoringSystem::new(config.monitoring.clone()).await?);

        // Initialize health checker
        let health_checker = Arc::new(HealthChecker::new(monitoring.metrics()));

        // Initialize Prometheus exporter if enabled
        #[cfg(feature = "prometheus")]
        let prometheus_exporter = if config.monitoring.prometheus_enabled {
            Some(Arc::new(PrometheusExporter::new(&config.monitoring)?))
        } else {
            None
        };
        #[cfg(not(feature = "prometheus"))]
        let _unused = ();

        // Try to load inference engine if model path provided
        let engine = if let Some(model_path) = &config.model_path {
            match Self::load_engine(model_path, config.tokenizer_path.as_deref(), &config.device)
                .await
            {
                Ok(eng) => {
                    tracing::info!("Loaded model from {}", model_path);
                    Some(Arc::new(RwLock::new(eng)))
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to load model: {}. Server will start without inference.",
                        e
                    );
                    None
                }
            }
        } else {
            tracing::info!("No model path provided. Server will start in mock mode.");
            None
        };

        Ok(Self {
            config,
            monitoring,
            health_checker,
            #[cfg(feature = "prometheus")]
            prometheus_exporter,
            engine,
        })
    }

    /// Load the inference engine
    async fn load_engine(
        model_path: &str,
        tokenizer_path: Option<&str>,
        device: &str,
    ) -> Result<bitnet_inference::InferenceEngine> {
        use bitnet_common::Device;
        use bitnet_models::formats::gguf::GgufLoader;
        use bitnet_models::loader::FormatLoader;
        use std::path::Path;

        // Parse device
        let device = match device {
            "cuda" | "gpu" => Device::Cuda(0),
            _ => Device::Cpu,
        };

        // Load model from GGUF
        let model_path = Path::new(model_path);
        if !model_path.exists() {
            anyhow::bail!("Model file not found: {}", model_path.display());
        }

        let loader = GgufLoader;
        let load_config = bitnet_models::loader::LoadConfig::default();
        let model = loader.load(model_path, &device, &load_config)?;

        // Load tokenizer
        let tokenizer: Arc<dyn bitnet_tokenizers::Tokenizer> =
            if let Some(tok_path) = tokenizer_path {
                let tok_path = Path::new(tok_path);
                if !tok_path.exists() {
                    anyhow::bail!("Tokenizer file not found: {}", tok_path.display());
                }
                Arc::from(bitnet_tokenizers::load_tokenizer(tok_path)?)
            } else {
                Arc::new(bitnet_tokenizers::BasicTokenizer::default())
            };

        // Create inference engine - model is already a Box<dyn Model>, convert to Arc
        let model: Arc<dyn bitnet_models::Model> = model.into();
        let engine = bitnet_inference::InferenceEngine::new(model, tokenizer, device)?;

        Ok(engine)
    }

    /// Create the application router with all routes and middleware
    pub fn create_app(&self) -> Router {
        let mut app = Router::new()
            .route("/inference", post(inference_handler))
            .route("/stream", post(streaming::streaming_handler))
            .route("/", get(root_handler))
            .with_state(AppState {
                metrics: self.monitoring.metrics(),
                engine: self.engine.clone(),
            });

        // Add health check routes
        app = app.merge(create_health_routes(self.health_checker.clone()));

        // Add Prometheus routes if enabled
        #[cfg(feature = "prometheus")]
        if let Some(prometheus) = &self.prometheus_exporter {
            app = app.merge(create_prometheus_routes(prometheus.clone()));
        }

        // Add middleware
        app = app
            .layer(middleware::from_fn(metrics_middleware))
            .layer(TraceLayer::new_for_http())
            .layer(CorsLayer::permissive());

        app
    }

    /// Start the server
    pub async fn start(&self) -> Result<()> {
        // Start background monitoring tasks
        self.monitoring.start_background_tasks().await?;

        let app = self.create_app();
        let addr = format!("{}:{}", self.config.host, self.config.port);

        tracing::info!(
            addr = %addr,
            prometheus_enabled = self.config.monitoring.prometheus_enabled,
            opentelemetry_enabled = self.config.monitoring.opentelemetry_enabled,
            "Starting BitNet server"
        );

        let listener = tokio::net::TcpListener::bind(&addr).await?;
        axum::serve(listener, app).await?;

        Ok(())
    }

    /// Shutdown the server gracefully
    pub async fn shutdown(&self) -> Result<()> {
        tracing::info!("Shutting down BitNet server");
        self.monitoring.shutdown().await?;
        Ok(())
    }
}

/// Application state shared across handlers
#[derive(Clone)]
pub(crate) struct AppState {
    metrics: Arc<MetricsCollector>,
    engine: Option<Arc<RwLock<bitnet_inference::InferenceEngine>>>,
}

/// Root handler
async fn root_handler() -> &'static str {
    "BitNet Inference Server"
}

/// Main inference handler with monitoring
async fn inference_handler(
    State(state): State<AppState>,
    axum::Json(request): axum::Json<InferenceRequest>,
) -> Result<axum::Json<InferenceResponse>, StatusCode> {
    let start_time = Instant::now();
    let request_id = uuid::Uuid::new_v4().to_string();

    // Create request tracker for metrics
    let tracker = state.metrics.track_request(request_id.clone());

    tracing::info!(
        request_id = %request_id,
        prompt_length = request.prompt.len(),
        max_tokens = request.max_tokens,
        model = request.model.as_deref().unwrap_or("default"),
        "Processing inference request"
    );

    // Use real inference if engine is available, otherwise simulate
    let inference_result = if let Some(engine) = &state.engine {
        real_inference(engine, &request).await
    } else {
        simulate_inference(&request).await
    };

    match inference_result {
        Ok(response) => {
            let duration = start_time.elapsed();
            tracker.record_tokens(response.tokens_generated);

            tracing::info!(
                request_id = %request_id,
                tokens_generated = response.tokens_generated,
                duration_ms = duration.as_millis(),
                tokens_per_second = response.tokens_per_second,
                "Inference completed successfully"
            );

            Ok(axum::Json(response))
        }
        Err(e) => {
            tracker.record_error("inference_failed");
            tracing::error!(
                request_id = %request_id,
                error = %e,
                "Inference failed"
            );
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Real inference using the BitNet engine
async fn real_inference(
    engine: &Arc<RwLock<bitnet_inference::InferenceEngine>>,
    request: &InferenceRequest,
) -> Result<InferenceResponse> {
    let start = Instant::now();

    // Build generation config
    let config = bitnet_inference::GenerationConfig {
        max_new_tokens: request.max_tokens.unwrap_or(64) as u32,
        temperature: request.temperature.unwrap_or(1.0),
        top_p: request.top_p.unwrap_or(0.9),
        top_k: request.top_k.unwrap_or(50) as u32,
        repetition_penalty: request.repetition_penalty.unwrap_or(1.0),
        ..Default::default()
    };

    // Run inference
    let engine = engine.read().await;
    let generated_text = engine.generate_with_config(&request.prompt, &config).await?;

    let duration = start.elapsed();
    // Estimate tokens (roughly 4 chars per token)
    let tokens_generated = (generated_text.len() / 4).max(1) as u64;
    let tokens_per_second = if duration.as_millis() > 0 {
        (tokens_generated as f64 * 1000.0) / duration.as_millis() as f64
    } else {
        0.0
    };

    Ok(InferenceResponse {
        text: generated_text,
        tokens_generated,
        inference_time_ms: duration.as_millis() as u64,
        tokens_per_second,
    })
}

/// Simulate inference (placeholder implementation)
async fn simulate_inference(request: &InferenceRequest) -> Result<InferenceResponse> {
    let start = Instant::now();

    // Simulate processing time
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    let tokens_generated = request.max_tokens.unwrap_or(50) as u64;
    let duration = start.elapsed();
    let tokens_per_second = if duration.as_millis() > 0 {
        (tokens_generated as f64 * 1000.0) / duration.as_millis() as f64
    } else {
        0.0
    };

    Ok(InferenceResponse {
        text: format!("Generated response to: {}", request.prompt),
        tokens_generated,
        inference_time_ms: duration.as_millis() as u64,
        tokens_per_second,
    })
}

/// Middleware for request metrics collection
async fn metrics_middleware(
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let start = Instant::now();
    let method = request.method().to_string();
    let path = request.uri().path().to_string();

    let response = next.run(request).await;

    let duration = start.elapsed();
    let status = response.status().as_u16();

    // Record metrics (in production, extract from state)
    tracing::debug!(
        method = %method,
        path = %path,
        status = status,
        duration_ms = duration.as_millis(),
        "Request completed"
    );

    response
}
