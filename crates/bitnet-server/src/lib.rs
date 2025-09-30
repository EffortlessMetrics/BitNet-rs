//! Production-ready HTTP server for BitNet inference with comprehensive features

pub mod batch_engine;
pub mod concurrency;
pub mod config;
pub mod execution_router;
pub mod model_manager;
pub mod monitoring;
pub mod security;
pub mod streaming;

use anyhow::Result;
use axum::{
    Router,
    extract::{Request, State},
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    response::{Json, Response},
    routing::{get, post},
};
use bitnet_common::Device;
use serde::{Deserialize, Serialize};
use std::net::IpAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tower_http::trace::TraceLayer;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use batch_engine::{BatchEngine, BatchRequest, RequestPriority};
use concurrency::{ConcurrencyManager, RequestMetadata};
pub use config::ServerConfig;
use execution_router::ExecutionRouter;
use model_manager::ModelManager;
use security::{SecurityValidator, configure_cors, security_headers_middleware};

#[cfg(feature = "prometheus")]
use monitoring::prometheus::{PrometheusExporter, create_prometheus_routes};
use monitoring::{
    MonitoringSystem,
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

/// Standardized error response for all API endpoints
#[derive(Serialize)]
pub struct ErrorResponse {
    pub error: String,
    pub error_code: String,
    pub request_id: Option<String>,
    pub details: Option<serde_json::Value>,
}

/// Enhanced inference request with additional metadata
#[derive(Deserialize)]
pub struct EnhancedInferenceRequest {
    #[serde(flatten)]
    pub base: InferenceRequest,
    pub priority: Option<String>,
    pub device_preference: Option<String>,
    pub quantization_hint: Option<String>,
    pub timeout_ms: Option<u64>,
}

/// Enhanced inference response with metadata
#[derive(Serialize)]
pub struct EnhancedInferenceResponse {
    #[serde(flatten)]
    pub base: InferenceResponse,
    pub device_used: String,
    pub quantization_type: String,
    pub batch_id: Option<String>,
    pub batch_size: Option<usize>,
    pub queue_time_ms: u64,
}

/// Model loading request
#[derive(Deserialize)]
pub struct ModelLoadRequest {
    pub model_path: String,
    pub tokenizer_path: Option<String>,
    pub device: Option<String>,
    pub model_id: Option<String>,
}

/// Model loading response
#[derive(Serialize)]
pub struct ModelLoadResponse {
    pub model_id: String,
    pub status: String,
    pub message: String,
}

/// Server statistics
#[derive(Serialize)]
pub struct ServerStats {
    pub uptime_seconds: u64,
    pub total_requests: u64,
    pub active_requests: usize,
    pub models_loaded: usize,
    pub device_statuses: Vec<execution_router::DeviceStatus>,
    pub batch_engine_stats: batch_engine::BatchEngineStats,
    pub concurrency_stats: concurrency::ConcurrencyStats,
}

/// Production-ready BitNet server with comprehensive features
pub struct BitNetServer {
    config: ServerConfig,
    model_manager: Arc<ModelManager>,
    execution_router: Arc<ExecutionRouter>,
    batch_engine: Arc<BatchEngine>,
    concurrency_manager: Arc<ConcurrencyManager>,
    security_validator: Arc<SecurityValidator>,
    monitoring: Arc<MonitoringSystem>,
    health_checker: Arc<HealthChecker>,
    #[cfg(feature = "prometheus")]
    prometheus_exporter: Option<Arc<PrometheusExporter>>,
    start_time: Instant,
}

impl BitNetServer {
    /// Create a new production-ready BitNet server
    pub async fn new(config: ServerConfig) -> Result<Self> {
        let start_time = Instant::now();

        info!("Initializing BitNet production server...");

        // Initialize monitoring system
        let monitoring = Arc::new(MonitoringSystem::new(config.monitoring.clone()).await?);
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

        // Initialize model manager
        let model_manager = Arc::new(ModelManager::new(config.model_manager.clone()));

        // Initialize execution router with available devices
        let devices = Self::detect_available_devices().await;
        let execution_router =
            Arc::new(ExecutionRouter::new(config.execution_router.clone(), devices).await?);

        // Initialize batch engine
        let batch_engine = Arc::new(BatchEngine::new(config.batch_engine.clone()));

        // Initialize concurrency manager
        let concurrency_manager = Arc::new(ConcurrencyManager::new(config.concurrency.clone()));

        // Initialize security validator
        let security_validator = Arc::new(SecurityValidator::new(config.security.clone())?);

        // Load default model if specified
        if let Some(model_path) = &config.server.default_model_path {
            let device = Device::Cpu; // TODO: Make configurable
            match model_manager
                .load_and_activate_model(
                    model_path,
                    config.server.default_tokenizer_path.as_deref(),
                    &device,
                )
                .await
            {
                Ok(model_id) => {
                    info!(model_id = %model_id, "Default model loaded successfully");
                }
                Err(e) => {
                    warn!(error = %e, "Failed to load default model, continuing without it");
                }
            }
        }

        info!("BitNet production server initialized successfully");

        Ok(Self {
            config,
            model_manager,
            execution_router,
            batch_engine,
            concurrency_manager,
            security_validator,
            monitoring,
            health_checker,
            #[cfg(feature = "prometheus")]
            prometheus_exporter,
            start_time,
        })
    }

    /// Detect available devices for execution
    async fn detect_available_devices() -> Vec<Device> {
        #[cfg(feature = "gpu")]
        let devices = {
            let mut devices = vec![Device::Cpu]; // CPU always available
            // Try to detect CUDA devices
            for i in 0..8 {
                // TODO: Implement actual CUDA device detection
                // For now, assume device 0 is available if GPU feature is enabled
                if i == 0 {
                    devices.push(Device::Cuda(i));
                    break;
                }
            }
            devices
        };

        #[cfg(not(feature = "gpu"))]
        let devices = vec![Device::Cpu]; // CPU always available

        info!("Detected devices: {:?}", devices);
        devices
    }

    /// Create the production application router with comprehensive routes and middleware
    pub fn create_app(&self) -> Router {
        let app_state = ProductionAppState {
            config: self.config.clone(),
            model_manager: Arc::clone(&self.model_manager),
            execution_router: Arc::clone(&self.execution_router),
            batch_engine: Arc::clone(&self.batch_engine),
            concurrency_manager: Arc::clone(&self.concurrency_manager),
            security_validator: Arc::clone(&self.security_validator),
            metrics: self.monitoring.metrics(),
            start_time: self.start_time,
        };

        let mut app = Router::new()
            // Core inference endpoints
            .route("/v1/inference", post(enhanced_inference_handler))
            .route("/v1/inference/stream", post(streaming::streaming_handler))
            .route("/inference", post(legacy_inference_handler)) // Legacy compatibility
            // Model management endpoints
            .route("/v1/models/load", post(load_model_handler))
            .route("/v1/models", get(list_models_handler))
            .route("/v1/models/:model_id", get(get_model_handler))
            .route("/v1/models/:model_id", axum::routing::delete(unload_model_handler))
            // Server statistics and management
            .route("/v1/stats", get(server_stats_handler))
            .route("/v1/devices", get(device_status_handler))
            // Root endpoint
            .route("/", get(root_handler))
            .with_state(app_state);

        // Add health check routes
        app = app.merge(create_health_routes(self.health_checker.clone()));

        // Add Prometheus routes if enabled
        #[cfg(feature = "prometheus")]
        if let Some(prometheus) = &self.prometheus_exporter {
            app = app.merge(create_prometheus_routes(prometheus.clone()));
        }

        // Add comprehensive middleware stack
        app = app
            .layer(middleware::from_fn(security_headers_middleware))
            .layer(middleware::from_fn_with_state(
                self.security_validator.clone(),
                request_validation_middleware,
            ))
            .layer(middleware::from_fn_with_state(
                self.config.security.clone(),
                security::ip_blocking_middleware,
            ))
            .layer(middleware::from_fn(enhanced_metrics_middleware))
            .layer(TraceLayer::new_for_http())
            .layer(configure_cors());

        app
    }

    /// Start the production server with all subsystems
    pub async fn start(&self) -> Result<()> {
        // Start background monitoring tasks
        self.monitoring.start_background_tasks().await?;

        // Start periodic device health updates
        let execution_router = Arc::clone(&self.execution_router);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));
            loop {
                interval.tick().await;
                execution_router.update_device_health().await;
            }
        });

        // Start rate limiter cleanup
        let concurrency_manager = Arc::clone(&self.concurrency_manager);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(300)); // 5 minutes
            loop {
                interval.tick().await;
                concurrency_manager.cleanup_rate_limiters().await;
            }
        });

        let app = self.create_app();
        let addr = format!("{}:{}", self.config.server.host, self.config.server.port);

        info!(
            addr = %addr,
            max_concurrent_requests = self.config.concurrency.max_concurrent_requests,
            max_batch_size = self.config.batch_engine.max_batch_size,
            prometheus_enabled = self.config.monitoring.prometheus_enabled,
            opentelemetry_enabled = self.config.monitoring.opentelemetry_enabled,
            authentication_enabled = self.config.security.require_authentication,
            "Starting BitNet production server"
        );

        let listener = tokio::net::TcpListener::bind(&addr).await?;
        axum::serve(listener, app).await?;

        Ok(())
    }

    /// Shutdown the server gracefully
    pub async fn shutdown(&self) -> Result<()> {
        info!("Starting graceful shutdown of BitNet production server");

        // TODO: Stop accepting new requests
        // TODO: Wait for active requests to complete (with timeout)
        // TODO: Shutdown subsystems in order

        self.monitoring.shutdown().await?;

        info!("BitNet production server shutdown complete");
        Ok(())
    }
}

/// Production application state shared across handlers
#[derive(Clone)]
pub struct ProductionAppState {
    pub config: ServerConfig,
    pub model_manager: Arc<ModelManager>,
    pub execution_router: Arc<ExecutionRouter>,
    pub batch_engine: Arc<BatchEngine>,
    pub concurrency_manager: Arc<ConcurrencyManager>,
    pub security_validator: Arc<SecurityValidator>,
    pub metrics: Arc<MetricsCollector>,
    pub start_time: Instant,
}

/// Root handler
async fn root_handler() -> &'static str {
    "BitNet Production Inference Server v1.0"
}

/// Enhanced inference handler with production features
async fn enhanced_inference_handler(
    State(state): State<ProductionAppState>,
    headers: HeaderMap,
    Json(request): Json<EnhancedInferenceRequest>,
) -> Result<Json<EnhancedInferenceResponse>, StatusCode> {
    let start_time = Instant::now();
    let request_id = Uuid::new_v4().to_string();

    // Extract client IP with localhost fallback
    let client_ip =
        extract_client_ip_from_headers(&headers).unwrap_or_else(|| IpAddr::from([127, 0, 0, 1]));

    // Create request metadata
    let metadata = RequestMetadata {
        id: request_id.clone(),
        client_ip,
        user_agent: headers.get("user-agent").and_then(|h| h.to_str().ok().map(String::from)),
        start_time,
        priority: parse_priority(request.priority.as_deref()),
    };

    // Validate request with standardized error handling
    if let Err(e) = state.security_validator.validate_inference_request(&request.base) {
        warn!(error = %e, "Request validation failed");
        let (status, _error_response) = handle_validation_error(&e, Some(request_id.clone()));
        return Err(status);
    }

    // Acquire concurrency slot with proper error handling
    let _slot = state.concurrency_manager.acquire_request_slot(metadata).await.map_err(|e| {
        warn!(error = %e, "Request rejected by concurrency manager");
        StatusCode::TOO_MANY_REQUESTS
    })?;

    // Create batch request
    let mut batch_request = BatchRequest::new(
        request.base.prompt.clone(),
        bitnet_inference::GenerationConfig {
            max_new_tokens: request.base.max_tokens.unwrap_or(64) as u32,
            temperature: request.base.temperature.unwrap_or(1.0),
            top_p: request.base.top_p.unwrap_or(0.9),
            top_k: request.base.top_k.unwrap_or(50) as u32,
            repetition_penalty: request.base.repetition_penalty.unwrap_or(1.0),
            ..Default::default()
        },
    );

    // Set request options
    batch_request = batch_request.with_priority(parse_priority(request.priority.as_deref()));

    if let Some(device_pref) = request.device_preference
        && let Ok(device) = parse_device(&device_pref)
    {
        batch_request = batch_request.with_device_preference(device);
    }

    if let Some(hint) = request.quantization_hint {
        batch_request = batch_request.with_quantization_hint(hint);
    }

    if let Some(timeout_ms) = request.timeout_ms {
        batch_request = batch_request.with_timeout(Duration::from_millis(timeout_ms));
    }

    let queue_time = start_time.elapsed();

    // Submit to batch engine and build response
    let result = state.batch_engine.submit_request(batch_request).await.map_err(|e| {
        error!(error = %e, "Batch processing failed");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    // Calculate tokens per second efficiently
    let tokens_per_second =
        calculate_tokens_per_second(result.tokens_generated, result.execution_time);

    let response = EnhancedInferenceResponse {
        base: InferenceResponse {
            text: result.generated_text,
            tokens_generated: result.tokens_generated,
            inference_time_ms: result.execution_time.as_millis() as u64,
            tokens_per_second,
        },
        device_used: format!("{:?}", result.device_used),
        quantization_type: result.quantization_type,
        batch_id: Some(result.batch_id),
        batch_size: Some(result.batch_size),
        queue_time_ms: queue_time.as_millis() as u64,
    };

    Ok(Json(response))
}

/// Legacy inference handler for backwards compatibility
async fn legacy_inference_handler(
    State(state): State<ProductionAppState>,
    headers: HeaderMap,
    Json(request): Json<InferenceRequest>,
) -> Result<Json<InferenceResponse>, StatusCode> {
    let enhanced_request = EnhancedInferenceRequest {
        base: request,
        priority: None,
        device_preference: None,
        quantization_hint: None,
        timeout_ms: None,
    };

    match enhanced_inference_handler(State(state), headers, Json(enhanced_request)).await {
        Ok(Json(enhanced_response)) => Ok(Json(enhanced_response.base)),
        Err(status) => Err(status),
    }
}

/// Load model handler
async fn load_model_handler(
    State(state): State<ProductionAppState>,
    Json(request): Json<ModelLoadRequest>,
) -> Result<Json<ModelLoadResponse>, StatusCode> {
    // Validate model path with standardized error handling
    if let Err(e) = state.security_validator.validate_model_request(&request.model_path) {
        warn!(error = %e, "Model load request validation failed");
        let (status, _error_response) = handle_validation_error(&e, None);
        return Err(status);
    }

    let device = parse_device(request.device.as_deref().unwrap_or("cpu")).unwrap_or(Device::Cpu);

    match state
        .model_manager
        .load_and_activate_model(&request.model_path, request.tokenizer_path.as_deref(), &device)
        .await
    {
        Ok(model_id) => {
            info!(model_id = %model_id, "Model loaded successfully");
            Ok(Json(ModelLoadResponse {
                model_id,
                status: "success".to_string(),
                message: "Model loaded and activated successfully".to_string(),
            }))
        }
        Err(e) => {
            error!(error = %e, "Failed to load model");
            Ok(Json(ModelLoadResponse {
                model_id: "none".to_string(),
                status: "error".to_string(),
                message: format!("Failed to load model: {}", e),
            }))
        }
    }
}

/// List models handler
async fn list_models_handler(
    State(state): State<ProductionAppState>,
) -> Json<Vec<model_manager::ModelMetadata>> {
    let models = state.model_manager.list_models().await;
    Json(models)
}

/// Get specific model handler
async fn get_model_handler(
    State(state): State<ProductionAppState>,
    axum::extract::Path(model_id): axum::extract::Path<String>,
) -> Result<Json<model_manager::ModelMetadata>, StatusCode> {
    match state.model_manager.get_model_metadata(&model_id).await {
        Some(metadata) => Ok(Json(metadata)),
        None => Err(StatusCode::NOT_FOUND),
    }
}

/// Unload model handler
async fn unload_model_handler(
    State(state): State<ProductionAppState>,
    axum::extract::Path(model_id): axum::extract::Path<String>,
) -> Result<StatusCode, StatusCode> {
    match state.model_manager.unload_model(&model_id).await {
        Ok(_) => {
            info!(model_id = %model_id, "Model unloaded successfully");
            Ok(StatusCode::NO_CONTENT)
        }
        Err(e) => {
            error!(model_id = %model_id, error = %e, "Failed to unload model");
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// Server statistics handler
async fn server_stats_handler(State(state): State<ProductionAppState>) -> Json<ServerStats> {
    let uptime = state.start_time.elapsed();
    let device_statuses = state.execution_router.get_device_statuses().await;
    let batch_stats = state.batch_engine.get_stats().await;
    let concurrency_stats = state.concurrency_manager.get_stats().await;
    let models = state.model_manager.list_models().await;

    let stats = ServerStats {
        uptime_seconds: uptime.as_secs(),
        total_requests: concurrency_stats.total_requests,
        active_requests: concurrency_stats.active_requests,
        models_loaded: models.len(),
        device_statuses,
        batch_engine_stats: batch_stats,
        concurrency_stats,
    };

    Json(stats)
}

/// Device status handler
async fn device_status_handler(
    State(state): State<ProductionAppState>,
) -> Json<Vec<execution_router::DeviceStatus>> {
    let statuses = state.execution_router.get_device_statuses().await;
    Json(statuses)
}

/// Enhanced middleware for comprehensive request metrics collection
async fn enhanced_metrics_middleware(request: Request, next: Next) -> Response {
    let start = Instant::now();
    let method = request.method().clone();
    let path = request.uri().path().to_string();
    let user_agent = request
        .headers()
        .get("user-agent")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("unknown")
        .to_string();

    let response = next.run(request).await;

    let duration = start.elapsed();
    let status = response.status();

    // Enhanced logging with more context
    if status.is_server_error() {
        error!(
            method = %method,
            path = %path,
            status = %status,
            duration_ms = duration.as_millis(),
            user_agent = %user_agent,
            "Request failed with server error"
        );
    } else if status.is_client_error() {
        warn!(
            method = %method,
            path = %path,
            status = %status,
            duration_ms = duration.as_millis(),
            "Request failed with client error"
        );
    } else {
        debug!(
            method = %method,
            path = %path,
            status = %status,
            duration_ms = duration.as_millis(),
            "Request completed successfully"
        );
    }

    response
}

/// Request validation middleware
async fn request_validation_middleware(
    State(validator): State<Arc<SecurityValidator>>,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // Check request size limits
    if let Some(content_length) = request.headers().get("content-length")
        && let Ok(length_str) = content_length.to_str()
        && let Ok(length) = length_str.parse::<usize>()
        && length > validator.config().max_prompt_length * 2
    {
        warn!(content_length = length, "Request payload too large");
        return Err(StatusCode::PAYLOAD_TOO_LARGE);
    }

    Ok(next.run(request).await)
}

/// Utility functions
/// Calculate tokens per second from token count and duration
fn calculate_tokens_per_second(tokens: u64, duration: Duration) -> f64 {
    let duration_ms = duration.as_millis();
    if duration_ms > 0 && tokens > 0 { (tokens as f64 * 1000.0) / duration_ms as f64 } else { 0.0 }
}

/// Create standardized error response
fn create_error_response(
    error: &str,
    error_code: &str,
    request_id: Option<String>,
    details: Option<serde_json::Value>,
) -> Json<ErrorResponse> {
    Json(ErrorResponse {
        error: error.to_string(),
        error_code: error_code.to_string(),
        request_id,
        details,
    })
}

/// Handle validation errors with consistent response format
fn handle_validation_error(
    error: &security::ValidationError,
    request_id: Option<String>,
) -> (StatusCode, Json<ErrorResponse>) {
    let (status, error_code) = match error {
        security::ValidationError::PromptTooLong(_, _) => {
            (StatusCode::BAD_REQUEST, "PROMPT_TOO_LONG")
        }
        security::ValidationError::TooManyTokens(_, _) => {
            (StatusCode::BAD_REQUEST, "TOO_MANY_TOKENS")
        }
        security::ValidationError::InvalidCharacters => {
            (StatusCode::BAD_REQUEST, "INVALID_CHARACTERS")
        }
        security::ValidationError::BlockedContent(_) => {
            (StatusCode::BAD_REQUEST, "BLOCKED_CONTENT")
        }
        security::ValidationError::MissingField(_) => (StatusCode::BAD_REQUEST, "MISSING_FIELD"),
        security::ValidationError::InvalidFieldValue(_) => {
            (StatusCode::BAD_REQUEST, "INVALID_FIELD_VALUE")
        }
    };

    let response = create_error_response(&error.to_string(), error_code, request_id, None);
    (status, response)
}

/// Parse request priority from string
fn parse_priority(priority: Option<&str>) -> RequestPriority {
    match priority {
        Some("low") => RequestPriority::Low,
        Some("normal") => RequestPriority::Normal,
        Some("high") => RequestPriority::High,
        Some("critical") => RequestPriority::Critical,
        _ => RequestPriority::Normal,
    }
}

/// Parse device from string
fn parse_device(device: &str) -> Result<Device> {
    match device.to_lowercase().as_str() {
        "cpu" => Ok(Device::Cpu),
        "gpu" | "cuda" => Ok(Device::Cuda(0)),
        _ if device.starts_with("cuda:") => {
            let id_str = &device[5..];
            let id = id_str.parse::<usize>()?;
            Ok(Device::Cuda(id))
        }
        _ => anyhow::bail!("Unknown device: {}", device),
    }
}

/// Extract client IP from headers using security module's implementation
fn extract_client_ip_from_headers(headers: &HeaderMap) -> Option<IpAddr> {
    security::extract_client_ip_from_headers(headers)
}
