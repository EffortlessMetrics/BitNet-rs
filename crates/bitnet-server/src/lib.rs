//! HTTP server for BitNet inference (pre-alpha; inference endpoints incomplete)
#![cfg_attr(doc, allow(dead_code, unused_imports, unused_variables))]

pub mod api_versioning;
pub mod auth;
pub mod batch_engine;
pub mod batch_request;
pub mod canary;
// Expose `caching` only when generating docs to avoid -Dwarnings dead_code in scaffolding.
#[cfg(doc)]
pub mod caching;
#[cfg(not(doc))]
mod caching;
pub mod concurrency;
pub mod config;
pub mod cors_config;
pub mod endpoint_registry;
pub mod execution_router;
pub mod gpu_streaming;
pub mod health;
pub mod health_monitor;
pub mod hf_model_service;
pub mod middleware_chain;
pub mod middleware_config;
pub mod model_manager;
pub mod model_registry;
pub mod monitoring;
pub mod rate_limiter;
pub mod request_context;
pub mod request_router;
pub mod request_utils;
pub mod runtime_model_registry;
pub mod security;
pub mod sse;
pub mod stream_handler;
pub mod streaming;
pub mod streaming_response;
pub mod websocket;
pub mod ws_messages;

use anyhow::Result;
use axum::{
    Router,
    extract::{Path, Request, State},
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Json, Response},
    routing::{get, post},
};
use bitnet_common::Device;
use bitnet_inference::{
    GenerationConfig,
    prompt_formatter::{
        Message as PromptMessage, Role as PromptRole, detect_template, format_prompt,
    },
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tower_http::trace::TraceLayer;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use batch_engine::{BatchEngine, BatchRequest};
use concurrency::{ConcurrencyManager, RequestMetadata};
pub use config::{DeviceConfig, ServerConfig};
use execution_router::ExecutionRouter;
use model_manager::ModelManager;
use request_utils::{
    calculate_tokens_per_second, create_error_response, extract_client_ip_from_headers,
    handle_validation_error, parse_device, parse_priority,
};
use security::{SecurityValidator, configure_cors, security_headers_middleware};

#[cfg(feature = "prometheus")]
use monitoring::prometheus::{PrometheusExporter, create_prometheus_routes};
use monitoring::{
    MonitoringSystem,
    health::{HealthChecker, create_health_routes},
    metrics::MetricsCollector,
};

const DENSE_QWEN25_Q8_MODEL_ID: &str = "qwen2.5-0.5b-instruct-q8_0";
const DENSE_QWEN25_Q8_MODEL_SHA256: &str =
    "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e";
const DENSE_QWEN3_Q8_MODEL_ID: &str = "qwen3-0.6b-instruct-q8_0";
const DENSE_QWEN3_Q8_MODEL_SHA256: &str =
    "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031";
const BITNET_QK256_MODEL_ID: &str = "microsoft-bitnet-b1.58-2B-4T-i2s";
const BITNET_QK256_MODEL_SHA256: &str =
    "4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162";
const BITNET_QK256_ROUTE: &str = "bitnet_qk256_cuda";
const DENSE_QWEN_ROUTE: &str = "dense_regular_llm_cuda";
const SHARED_ENGINE_ROUTE: &str = "shared_validated_local_inference_engine";
const MAX_SERVER_RECEIPTS: usize = 128;

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

/// OpenAI-compatible chat completions request.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatCompletionMessage>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stream: Option<bool>,
}

/// OpenAI-compatible chat message.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatCompletionMessage {
    pub role: String,
    pub content: String,
}

/// OpenAI-compatible chat completions response.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: ChatCompletionUsage,
    pub metadata: ChatCompletionResponseMetadata,
    pub receipt: ServerSharedEngineReceipt,
}

/// Trace metadata attached to an OpenAI-compatible chat response.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponseMetadata {
    pub receipt_id: String,
    pub receipt_path: String,
    pub latest_receipt_path: String,
    pub readiness_path: String,
    pub model_coverage_row: Option<String>,
    pub model_coverage_tier: Option<String>,
    pub selected_backend: String,
    pub selected_route: String,
    pub fallback_used: bool,
}

/// OpenAI-compatible chat completion choice.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChoice {
    pub index: usize,
    pub message: ChatCompletionMessage,
    pub finish_reason: String,
}

/// OpenAI-compatible token usage summary.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Per-request receipt summary for server shared-engine inference.
#[derive(Debug, Clone, Serialize)]
pub struct ServerSharedEngineReceipt {
    pub receipt_kind: String,
    pub request_id: String,
    pub runtime_path: String,
    pub runtime_api: String,
    pub model_identity: ServerSharedEngineModelIdentity,
    pub endpoint_profile: ServerSharedEngineEndpointProfile,
    pub generation_policy: ServerSharedEngineGenerationPolicy,
    pub requested_model: String,
    pub active_model_id: String,
    pub active_model_path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_sha256: Option<String>,
    pub model_coverage_row: Option<String>,
    pub model_coverage_tier: Option<String>,
    pub selected_backend: String,
    pub requested_backend: String,
    pub selected_route: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution_plan: Option<ServerSharedEngineExecutionPlan>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution_coverage: Option<ServerSharedEngineExecutionCoverage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kernel_stats: Option<Vec<ServerSharedEngineKernelStats>>,
    pub prompt_template: String,
    pub tokenizer_authority: String,
    pub prompt_authority: String,
    pub fallback_used: bool,
    pub simulated_inference: bool,
    pub streaming: bool,
    pub generated_text_non_empty: bool,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_ms: u64,
    pub quality_gate: ServerSharedEngineQualityGate,
    pub server_smoke_response_claimed: bool,
    pub server_ready_claimed: bool,
    pub speedup_claim: bool,
    pub full_cuda_residency_claimed: bool,
    pub dense_regular_llm_cuda_inference_claimed: bool,
    pub bitnet_packed_i2s_qk256_proof: bool,
}

impl ChatCompletionResponseMetadata {
    fn from_receipt(receipt: &ServerSharedEngineReceipt) -> Self {
        Self {
            receipt_id: receipt.request_id.clone(),
            receipt_path: format!("/receipts/{}", receipt.request_id),
            latest_receipt_path: "/receipts/latest".to_string(),
            readiness_path: "/readiness".to_string(),
            model_coverage_row: receipt.model_coverage_row.clone(),
            model_coverage_tier: receipt.model_coverage_tier.clone(),
            selected_backend: receipt.selected_backend.clone(),
            selected_route: receipt.selected_route.clone(),
            fallback_used: receipt.fallback_used,
        }
    }
}

/// Bounded in-memory receipt export store for server request receipts.
#[derive(Debug, Default)]
pub struct ServerReceiptStore {
    inner: RwLock<ServerReceiptStoreInner>,
}

#[derive(Debug, Default)]
struct ServerReceiptStoreInner {
    receipts: BTreeMap<String, ServerSharedEngineReceipt>,
    order: VecDeque<String>,
    latest: Option<String>,
}

impl ServerReceiptStore {
    pub async fn insert(&self, receipt: ServerSharedEngineReceipt) {
        let receipt_id = receipt.request_id.clone();
        let mut inner = self.inner.write().await;

        if inner.receipts.contains_key(&receipt_id) {
            inner.order.retain(|known| known != &receipt_id);
        }

        inner.receipts.insert(receipt_id.clone(), receipt);
        inner.order.push_back(receipt_id.clone());
        inner.latest = Some(receipt_id);

        while inner.order.len() > MAX_SERVER_RECEIPTS {
            if let Some(evicted) = inner.order.pop_front() {
                inner.receipts.remove(&evicted);
            }
        }
    }

    pub async fn latest(&self) -> Option<ServerSharedEngineReceipt> {
        let inner = self.inner.read().await;
        inner.latest.as_ref().and_then(|receipt_id| inner.receipts.get(receipt_id)).cloned()
    }

    pub async fn get(&self, receipt_id: &str) -> Option<ServerSharedEngineReceipt> {
        let inner = self.inner.read().await;
        inner.receipts.get(receipt_id).cloned()
    }
}

/// Stable error body for receipt lookup endpoints.
#[derive(Debug, Clone, Serialize)]
pub struct ServerReceiptLookupError {
    pub error_code: String,
    pub error: String,
    pub receipt_id: Option<String>,
}

/// Stable model identity attached to a server shared-engine receipt.
#[derive(Debug, Clone, Serialize)]
pub struct ServerSharedEngineModelIdentity {
    pub model_id: String,
    pub requested_model: String,
    pub active_model_id: String,
    pub active_model_path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_sha256: Option<String>,
}

/// Endpoint and request-shape profile attached to a server shared-engine receipt.
#[derive(Debug, Clone, Serialize)]
pub struct ServerSharedEngineEndpointProfile {
    pub endpoint: String,
    pub method: String,
    pub request_profile: String,
    pub streaming: bool,
    pub message_count: usize,
}

/// Generation policy attached to a server shared-engine receipt.
#[derive(Debug, Clone, Serialize)]
pub struct ServerSharedEngineGenerationPolicy {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub decoding: String,
}

/// Dispatch plan summary attached to BitNet QK256 server-smoke receipts.
#[derive(Debug, Clone, Serialize)]
pub struct ServerSharedEngineExecutionPlan {
    pub planner_version: String,
    pub model_family: String,
    pub quantization: String,
    pub selected_route: String,
    pub requested_backend: String,
    pub selected_backend: String,
    pub runtime_api: String,
    pub strict_fallback_policy: String,
    pub dense_regular_llm_cuda: bool,
    pub bitnet_packed_qk256_cuda: bool,
    pub cuda_bitnet_qk256_ops: u64,
    pub cuda_dense_regular_llm_ops: u64,
    pub cpu_fallback_ops: u64,
    pub unsupported_ops: u64,
    pub total_ops: u64,
    pub cuda_ops: u64,
    pub mixed_cuda_routes: bool,
    pub fallback_used: bool,
    pub strict_cuda_ready: bool,
    pub speedup_claim: bool,
    pub full_cuda_residency_claimed: bool,
}

/// QK256 dispatch coverage attached to BitNet server-smoke receipts.
#[derive(Debug, Clone, Serialize)]
pub struct ServerSharedEngineExecutionCoverage {
    pub execution_claim: String,
    pub bitnet_linear_layers_total: u64,
    pub bitnet_linear_layers_on_cuda: u64,
    pub bitnet_linear_layers_cpu_fallback: u64,
    pub unsupported_ops: Vec<String>,
    pub fallback_used: bool,
}

/// Kernel statistics attached to BitNet QK256 server-smoke receipts.
#[derive(Debug, Clone, Serialize)]
pub struct ServerSharedEngineKernelStats {
    pub kernel_id: String,
    pub invocations: u64,
    pub fallback_invocations: u64,
    pub cpu_fallback_invocations: u64,
    pub host_to_device_bytes: Option<u64>,
    pub device_to_host_bytes: Option<u64>,
    pub kernel_launches: u64,
    pub kernel_time_ms: Option<f64>,
    pub kernel_time_samples: Option<u64>,
}

/// Bounded quality gate attached to a server shared-engine receipt.
#[derive(Debug, Clone, Serialize)]
pub struct ServerSharedEngineQualityGate {
    pub gate: String,
    pub passed: bool,
    pub generated_text_non_empty: bool,
    pub utf8_valid: bool,
    pub broad_chat_quality_claimed: bool,
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

/// Server readiness and certification response.
#[derive(Debug, Clone, Serialize)]
pub struct ServerReadinessResponse {
    pub ready: bool,
    pub status: String,
    pub reason: Option<String>,
    pub model: ServerReadinessModelState,
    pub backend: ServerReadinessBackendState,
    pub inference: ServerReadinessInferenceState,
    pub claim_boundary: ServerReadinessClaimBoundary,
}

/// Model state used by the server readiness endpoint.
#[derive(Debug, Clone, Serialize)]
pub struct ServerReadinessModelState {
    pub default_model_configured: bool,
    pub loaded_models: usize,
    pub total_size_mb: u64,
    pub cache_size_limit: usize,
    pub memory_limit_gb: Option<f64>,
    pub active_model_id: Option<String>,
    pub active_model: Option<ServerReadinessActiveModel>,
}

/// Active model summary used by the server readiness endpoint.
#[derive(Debug, Clone, Serialize)]
pub struct ServerReadinessActiveModel {
    pub model_id: String,
    pub model_path: String,
    pub model_sha256: Option<String>,
    pub model_coverage_row: Option<String>,
    pub model_coverage_tier: Option<String>,
    pub selected_route: Option<String>,
    pub device: String,
    pub quantization_type: String,
    pub size_mb: u64,
    pub parameters: u64,
    pub context_length: u32,
}

/// Backend state used by the server readiness endpoint.
#[derive(Debug, Clone, Serialize)]
pub struct ServerReadinessBackendState {
    pub requested_default_device: String,
    pub selected_backend: Option<String>,
    pub configured_fallback_enabled: bool,
    pub server_fallback_policy: String,
    pub device_statuses: Vec<execution_router::DeviceStatus>,
}

/// Server inference readiness state.
#[derive(Debug, Clone, Serialize)]
pub struct ServerReadinessInferenceState {
    pub real_server_inference_ready: bool,
    pub batch_inference_ready: bool,
    pub simulated_inference_enabled: bool,
    pub runtime_path: String,
    pub unavailable_reason: String,
}

/// Claim boundaries reported by the server readiness endpoint.
#[derive(Debug, Clone, Serialize)]
pub struct ServerReadinessClaimBoundary {
    pub server_ready_claimed: bool,
    pub dense_regular_llm_cuda_inference_claimed: bool,
    pub bitnet_packed_i2s_qk256_proof: bool,
    pub speedup_claim: bool,
    pub full_cuda_residency_claimed: bool,
}

/// BitNet inference server (pre-alpha; inference endpoints incomplete)
pub struct BitNetServer {
    config: ServerConfig,
    model_manager: Arc<ModelManager>,
    execution_router: Arc<ExecutionRouter>,
    batch_engine: Arc<BatchEngine>,
    concurrency_manager: Arc<ConcurrencyManager>,
    security_validator: Arc<SecurityValidator>,
    receipt_store: Arc<ServerReceiptStore>,
    monitoring: Arc<MonitoringSystem>,
    health_checker: Arc<HealthChecker>,
    #[cfg(feature = "prometheus")]
    prometheus_exporter: Option<Arc<PrometheusExporter>>,
    start_time: Instant,
}

impl BitNetServer {
    /// Create a new BitNet server instance
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

        // Initialize bounded per-request receipt export store
        let receipt_store = Arc::new(ServerReceiptStore::default());

        // Load default model if specified
        if let Some(model_path) = &config.server.default_model_path {
            let device = config.server.default_device.resolve();
            info!(device = ?device, "Loading default model on configured device");
            match model_manager
                .load_and_activate_model(
                    model_path,
                    config.server.default_tokenizer_path.as_deref(),
                    &device,
                )
                .await
            {
                Ok(model_id) => {
                    info!(model_id = %model_id, device = ?device, "Default model loaded successfully");
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
            receipt_store,
            monitoring,
            health_checker,
            #[cfg(feature = "prometheus")]
            prometheus_exporter,
            start_time,
        })
    }

    /// Detect available devices for execution
    async fn detect_available_devices() -> Vec<Device> {
        #[cfg(any(feature = "gpu", feature = "cuda"))]
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

        #[cfg(not(any(feature = "gpu", feature = "cuda")))]
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
            receipt_store: Arc::clone(&self.receipt_store),
            metrics: self.monitoring.metrics(),
            start_time: self.start_time,
        };

        let mut app = Router::new()
            // Core inference endpoints
            .route("/v1/inference", post(enhanced_inference_handler))
            .route("/v1/inference/stream", post(streaming::streaming_handler))
            .route("/v1/chat/completions", post(chat_completions_handler))
            .route("/inference", post(legacy_inference_handler)) // Legacy compatibility
            // Model management endpoints
            .route("/v1/models/load", post(load_model_handler))
            .route("/v1/models", get(list_models_handler))
            .route("/v1/models/{model_id}", get(get_model_handler))
            .route("/v1/models/{model_id}", axum::routing::delete(unload_model_handler))
            // Server statistics and management
            .route("/v1/stats", get(server_stats_handler))
            .route("/v1/devices", get(device_status_handler))
            .route("/readiness", get(server_readiness_handler))
            .route("/v1/readiness", get(server_readiness_handler))
            .route("/receipts/latest", get(latest_receipt_handler))
            .route("/receipts/{receipt_id}", get(receipt_by_id_handler))
            // GPU streaming endpoint
            .route("/api/v1/generate/stream", post(gpu_streaming::gpu_stream_handler))
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
            .layer(configure_cors(&self.config.security));

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
                // Cleanup rate limiters inactive for 1 hour
                concurrency_manager.cleanup_rate_limiters(Duration::from_secs(3600)).await;
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
        axum::serve(listener, app.into_make_service_with_connect_info::<std::net::SocketAddr>())
            .await?;

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
    pub receipt_store: Arc<ServerReceiptStore>,
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
    connect_info: axum::extract::ConnectInfo<std::net::SocketAddr>,
    headers: HeaderMap,
    Json(request): Json<EnhancedInferenceRequest>,
) -> Result<Json<EnhancedInferenceResponse>, StatusCode> {
    let start_time = Instant::now();
    let request_id = Uuid::new_v4().to_string();

    // Extract client IP with localhost fallback
    let client_ip = if state.config.security.trust_forwarded_headers {
        extract_client_ip_from_headers(&headers).unwrap_or_else(|| connect_info.ip())
    } else {
        connect_info.ip()
    };

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
    let mut batch_request = BatchRequest::new(request.base.prompt.clone(), {
        let mut config = bitnet_inference::GenerationConfig::default()
            .with_max_tokens(request.base.max_tokens.unwrap_or(64) as u32)
            .with_temperature(request.base.temperature.unwrap_or(1.0))
            .with_top_p(request.base.top_p.unwrap_or(0.9))
            .with_top_k(request.base.top_k.unwrap_or(50) as u32);
        config.repetition_penalty = request.base.repetition_penalty.unwrap_or(1.0);
        config
    });

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
        StatusCode::SERVICE_UNAVAILABLE
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
    connect_info: axum::extract::ConnectInfo<std::net::SocketAddr>,
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

    match enhanced_inference_handler(State(state), connect_info, headers, Json(enhanced_request))
        .await
    {
        Ok(Json(enhanced_response)) => Ok(Json(enhanced_response.base)),
        Err(status) => Err(status),
    }
}

/// OpenAI-compatible chat completions endpoint.
async fn chat_completions_handler(
    State(state): State<ProductionAppState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    let request_id = Uuid::new_v4().to_string();
    let readiness = collect_server_readiness_response(&state).await;

    if request.stream.unwrap_or(false) {
        let details = serde_json::json!({
            "requested_model": request.model,
            "message_count": request.messages.len(),
            "stream": true,
            "required_stream": false,
            "readiness": readiness,
        });

        return (
            StatusCode::BAD_REQUEST,
            create_error_response(
                "Streaming chat completions are not wired to the shared inference engine yet",
                "SERVER_STREAMING_UNAVAILABLE",
                Some(request_id),
                Some(details),
            ),
        )
            .into_response();
    }

    let Some(active_model) = state.model_manager.get_active_model().await else {
        let details = serde_json::json!({
            "requested_model": request.model,
            "message_count": request.messages.len(),
            "stream": false,
            "readiness": readiness,
        });

        return (
            StatusCode::SERVICE_UNAVAILABLE,
            create_error_response(
                "OpenAI-compatible chat completions require an active verified model loaded through the server ModelManager",
                "SERVER_INFERENCE_UNAVAILABLE",
                Some(request_id),
                Some(details),
            ),
        )
            .into_response();
    };

    let (rendered_prompt, prompt_template) = match render_chat_completion_prompt(&request) {
        Ok(prompt) => prompt,
        Err(error) => {
            let details = serde_json::json!({
                "requested_model": request.model,
                "message_count": request.messages.len(),
                "stream": false,
                "readiness": readiness,
            });

            return (
                StatusCode::BAD_REQUEST,
                create_error_response(
                    &error,
                    "INVALID_CHAT_COMPLETION_REQUEST",
                    Some(request_id),
                    Some(details),
                ),
            )
                .into_response();
        }
    };

    let generation_config = match generation_config_from_chat_request(&request) {
        Ok(config) => config,
        Err(error) => {
            let details = serde_json::json!({
                "requested_model": request.model,
                "message_count": request.messages.len(),
                "stream": false,
                "readiness": readiness,
            });

            return (
                StatusCode::BAD_REQUEST,
                create_error_response(
                    &error,
                    "INVALID_GENERATION_CONFIG",
                    Some(request_id),
                    Some(details),
                ),
            )
                .into_response();
        }
    };

    let prompt_tokens = token_count_for_text(&active_model.engine, &rendered_prompt)
        .unwrap_or_else(|| {
            bitnet_inference::prompt_formatter::estimate_token_count(&rendered_prompt)
        });

    let qk256_coverage_before = bitnet_qk256_dispatch::qk256_dispatch_coverage();
    let qk256_runtime_before = bitnet_qk256_dispatch::qk256_cuda_runtime_stats();
    let start = Instant::now();
    let generated = match active_model
        .engine
        .generate_with_config(&rendered_prompt, &generation_config)
        .await
    {
        Ok(text) => text,
        Err(error) => {
            let details = serde_json::json!({
                "requested_model": request.model,
                "message_count": request.messages.len(),
                "stream": false,
                "active_model_id": active_model.metadata.model_id.clone(),
                "selected_backend": active_model.metadata.device.clone(),
                "readiness": readiness,
                "engine_error": error.to_string(),
            });

            return (
                StatusCode::SERVICE_UNAVAILABLE,
                create_error_response(
                    "Shared local inference engine failed to generate a chat completion",
                    "SERVER_SHARED_ENGINE_FAILED",
                    Some(request_id),
                    Some(details),
                ),
            )
                .into_response();
        }
    };
    let total_ms = start.elapsed().as_millis() as u64;
    let qk256_coverage_after = bitnet_qk256_dispatch::qk256_dispatch_coverage();
    let qk256_runtime_after = bitnet_qk256_dispatch::qk256_cuda_runtime_stats();

    active_model.update_usage();

    let completion_tokens = token_count_for_text(&active_model.engine, &generated)
        .unwrap_or_else(|| bitnet_inference::prompt_formatter::estimate_token_count(&generated));

    let usage = ChatCompletionUsage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens.saturating_add(completion_tokens),
    };
    let receipt = build_server_shared_engine_receipt(
        &request_id,
        &request,
        &active_model.metadata,
        &state.config.server.default_device,
        &prompt_template,
        &usage,
        &generated,
        total_ms,
        bitnet_server_qk256_evidence(
            &request,
            &active_model.metadata,
            &qk256_coverage_before,
            &qk256_coverage_after,
            &qk256_runtime_before,
            &qk256_runtime_after,
        )
        .as_ref(),
    );
    let metadata = ChatCompletionResponseMetadata::from_receipt(&receipt);
    state.receipt_store.insert(receipt.clone()).await;
    let response = ChatCompletionResponse {
        id: format!("chatcmpl-{request_id}"),
        object: "chat.completion".to_string(),
        created: current_unix_timestamp(),
        model: request.model.clone(),
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ChatCompletionMessage { role: "assistant".to_string(), content: generated },
            finish_reason: "stop".to_string(),
        }],
        usage,
        metadata,
        receipt,
    };

    (StatusCode::OK, Json(response)).into_response()
}

fn render_chat_completion_prompt(
    request: &ChatCompletionRequest,
) -> std::result::Result<(String, String), String> {
    if request.messages.is_empty() {
        return Err("messages must contain at least one chat message".to_string());
    }

    let mut messages = Vec::with_capacity(request.messages.len());
    for message in &request.messages {
        let role = match message.role.as_str() {
            "system" => PromptRole::System,
            "user" => PromptRole::User,
            "assistant" => PromptRole::Assistant,
            other => {
                return Err(format!(
                    "unsupported chat role `{other}`; supported roles are system, user, assistant"
                ));
            }
        };
        messages.push(PromptMessage { role, content: message.content.clone() });
    }

    let template = detect_template(&request.model);
    let rendered = format_prompt(&messages, &template);
    Ok((rendered, template.as_str().to_string()))
}

fn generation_config_from_chat_request(
    request: &ChatCompletionRequest,
) -> std::result::Result<GenerationConfig, String> {
    let max_tokens = request.max_tokens.unwrap_or(16);
    if max_tokens == 0 {
        return Err("max_tokens must be greater than zero".to_string());
    }
    let max_tokens =
        u32::try_from(max_tokens).map_err(|_| "max_tokens exceeds u32::MAX".to_string())?;

    let temperature = request.temperature.unwrap_or(0.0);
    let top_p = request.top_p.unwrap_or(1.0);
    let config = GenerationConfig::greedy()
        .with_max_tokens(max_tokens)
        .with_temperature(temperature)
        .with_top_p(top_p);

    config.validate()?;
    Ok(config)
}

fn token_count_for_text(engine: &bitnet_inference::InferenceEngine, text: &str) -> Option<usize> {
    engine.tokenizer().encode(text, false, true).ok().map(|tokens| tokens.len())
}

#[derive(Debug, Clone)]
struct ServerQk256Evidence {
    coverage: bitnet_qk256_dispatch::Qk256DispatchCoverageCounters,
    runtime_stats: bitnet_qk256_dispatch::Qk256CudaRuntimeStats,
}

fn bitnet_server_qk256_evidence(
    request: &ChatCompletionRequest,
    active_model: &model_manager::ModelMetadata,
    coverage_before: &bitnet_qk256_dispatch::Qk256DispatchCoverageCounters,
    coverage_after: &bitnet_qk256_dispatch::Qk256DispatchCoverageCounters,
    runtime_before: &bitnet_qk256_dispatch::Qk256CudaRuntimeStats,
    runtime_after: &bitnet_qk256_dispatch::Qk256CudaRuntimeStats,
) -> Option<ServerQk256Evidence> {
    is_official_bitnet_qk256_model(request, active_model).then(|| ServerQk256Evidence {
        coverage: qk256_dispatch_coverage_delta(coverage_before, coverage_after),
        runtime_stats: qk256_cuda_runtime_stats_delta(runtime_before, runtime_after),
    })
}

fn qk256_dispatch_coverage_delta(
    before: &bitnet_qk256_dispatch::Qk256DispatchCoverageCounters,
    after: &bitnet_qk256_dispatch::Qk256DispatchCoverageCounters,
) -> bitnet_qk256_dispatch::Qk256DispatchCoverageCounters {
    let bitnet_linear_layers_on_cuda =
        after.bitnet_linear_layers_on_cuda.saturating_sub(before.bitnet_linear_layers_on_cuda);
    let bitnet_linear_layers_cpu_fallback = after
        .bitnet_linear_layers_cpu_fallback
        .saturating_sub(before.bitnet_linear_layers_cpu_fallback);
    let bitnet_linear_layers_total =
        after.bitnet_linear_layers_total.saturating_sub(before.bitnet_linear_layers_total);
    let unsupported_delta = after
        .unsupported_ops
        .iter()
        .filter(|candidate| !before.unsupported_ops.iter().any(|prior| prior == *candidate))
        .cloned()
        .collect::<Vec<_>>();

    bitnet_qk256_dispatch::Qk256DispatchCoverageCounters {
        bitnet_linear_layers_total,
        bitnet_linear_layers_on_cuda,
        bitnet_linear_layers_cpu_fallback,
        unsupported_ops: unsupported_delta,
        execution_claim: if bitnet_linear_layers_on_cuda > 0 {
            "cuda_inference_contribution"
        } else {
            after.execution_claim
        },
    }
}

fn qk256_cuda_runtime_stats_delta(
    before: &bitnet_qk256_dispatch::Qk256CudaRuntimeStats,
    after: &bitnet_qk256_dispatch::Qk256CudaRuntimeStats,
) -> bitnet_qk256_dispatch::Qk256CudaRuntimeStats {
    let before_ms = before.kernel_time_ms.unwrap_or(0.0);
    let after_ms = after.kernel_time_ms.unwrap_or(0.0);
    let kernel_time_samples = after.kernel_time_samples.saturating_sub(before.kernel_time_samples);
    bitnet_qk256_dispatch::Qk256CudaRuntimeStats {
        host_to_device_bytes: after
            .host_to_device_bytes
            .saturating_sub(before.host_to_device_bytes),
        device_to_host_bytes: after
            .device_to_host_bytes
            .saturating_sub(before.device_to_host_bytes),
        kernel_time_ms: (kernel_time_samples > 0).then_some((after_ms - before_ms).max(0.0)),
        kernel_time_samples,
    }
}

#[allow(clippy::too_many_arguments)]
fn build_server_shared_engine_receipt(
    request_id: &str,
    request: &ChatCompletionRequest,
    active_model: &model_manager::ModelMetadata,
    configured_device: &DeviceConfig,
    prompt_template: &str,
    usage: &ChatCompletionUsage,
    generated_text: &str,
    total_ms: u64,
    qk256_evidence: Option<&ServerQk256Evidence>,
) -> ServerSharedEngineReceipt {
    let requested_backend = configured_device.backend_label();
    let selected_backend = selected_backend_label(configured_device, active_model);
    let route = server_receipt_route(configured_device, request, active_model);
    let coverage = server_receipt_model_coverage(&route, request, active_model);
    let fallback_used = matches!(configured_device, DeviceConfig::NvidiaRtx5070TiCuda)
        && selected_backend != requested_backend;
    let generated_text_non_empty = !generated_text.trim().is_empty();
    let dense_cuda_smoke_claimed = generated_text_non_empty
        && selected_backend == "nvidia-rtx-5070-ti-cuda"
        && !fallback_used
        && route == DENSE_QWEN_ROUTE;
    let bitnet_qk256_proof_claimed = generated_text_non_empty
        && selected_backend == "nvidia-rtx-5070-ti-cuda"
        && !fallback_used
        && route == BITNET_QK256_ROUTE
        && qk256_evidence.is_some_and(bitnet_qk256_evidence_is_strict_cuda);
    let server_smoke_response_claimed = dense_cuda_smoke_claimed || bitnet_qk256_proof_claimed;
    let streaming = request.stream.unwrap_or(false);
    let generation_policy = server_receipt_generation_policy(request);
    let execution_plan = qk256_evidence.map(|evidence| {
        server_receipt_qk256_execution_plan(
            &route,
            &requested_backend,
            &selected_backend,
            &evidence.coverage,
            fallback_used,
        )
    });
    let execution_coverage =
        qk256_evidence.map(|evidence| server_receipt_qk256_execution_coverage(&evidence.coverage));
    let kernel_stats = qk256_evidence.map(server_receipt_qk256_kernel_stats);

    ServerSharedEngineReceipt {
        receipt_kind: "server_shared_engine_chat_completion".to_string(),
        request_id: request_id.to_string(),
        runtime_path: "shared_local_inference_engine".to_string(),
        runtime_api: runtime_api_label(configured_device, active_model),
        model_identity: ServerSharedEngineModelIdentity {
            model_id: request.model.clone(),
            requested_model: request.model.clone(),
            active_model_id: active_model.model_id.clone(),
            active_model_path: active_model.model_path.clone(),
            model_sha256: active_model.model_sha256.clone(),
        },
        endpoint_profile: ServerSharedEngineEndpointProfile {
            endpoint: "/v1/chat/completions".to_string(),
            method: "POST".to_string(),
            request_profile: if streaming {
                "streaming_chat_completion".to_string()
            } else {
                "non_streaming_chat_completion".to_string()
            },
            streaming,
            message_count: request.messages.len(),
        },
        generation_policy,
        requested_model: request.model.clone(),
        active_model_id: active_model.model_id.clone(),
        active_model_path: active_model.model_path.clone(),
        model_sha256: active_model.model_sha256.clone(),
        model_coverage_row: coverage.as_ref().map(|coverage| coverage.row.to_string()),
        model_coverage_tier: coverage.as_ref().map(|coverage| coverage.tier.to_string()),
        requested_backend,
        selected_backend,
        selected_route: route,
        execution_plan,
        execution_coverage,
        kernel_stats,
        prompt_template: prompt_template.to_string(),
        tokenizer_authority: "active_model_tokenizer".to_string(),
        prompt_authority: "server_chat_template".to_string(),
        fallback_used,
        simulated_inference: false,
        streaming,
        generated_text_non_empty,
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
        total_ms,
        quality_gate: ServerSharedEngineQualityGate {
            gate: "server_non_empty_utf8_response".to_string(),
            passed: generated_text_non_empty,
            generated_text_non_empty,
            utf8_valid: true,
            broad_chat_quality_claimed: false,
        },
        server_smoke_response_claimed,
        server_ready_claimed: false,
        speedup_claim: false,
        full_cuda_residency_claimed: false,
        dense_regular_llm_cuda_inference_claimed: dense_cuda_smoke_claimed,
        bitnet_packed_i2s_qk256_proof: bitnet_qk256_proof_claimed,
    }
}

fn bitnet_qk256_evidence_is_strict_cuda(evidence: &ServerQk256Evidence) -> bool {
    evidence.coverage.bitnet_linear_layers_on_cuda > 0
        && evidence.coverage.bitnet_linear_layers_cpu_fallback == 0
        && evidence.coverage.unsupported_ops.is_empty()
}

fn server_receipt_qk256_execution_plan(
    route: &str,
    requested_backend: &str,
    selected_backend: &str,
    coverage: &bitnet_qk256_dispatch::Qk256DispatchCoverageCounters,
    fallback_used: bool,
) -> ServerSharedEngineExecutionPlan {
    let bitnet_ready = route == BITNET_QK256_ROUTE
        && coverage.bitnet_linear_layers_on_cuda > 0
        && coverage.bitnet_linear_layers_cpu_fallback == 0
        && coverage.unsupported_ops.is_empty()
        && !fallback_used;
    ServerSharedEngineExecutionPlan {
        planner_version: "cuda-planner-004".to_string(),
        model_family: "bitnet_b1_58".to_string(),
        quantization: "i2_s_qk256".to_string(),
        selected_route: route.to_string(),
        requested_backend: requested_backend.to_string(),
        selected_backend: selected_backend.to_string(),
        runtime_api: "cuda".to_string(),
        strict_fallback_policy: "reject".to_string(),
        dense_regular_llm_cuda: false,
        bitnet_packed_qk256_cuda: bitnet_ready,
        cuda_bitnet_qk256_ops: coverage.bitnet_linear_layers_on_cuda,
        cuda_dense_regular_llm_ops: 0,
        cpu_fallback_ops: coverage.bitnet_linear_layers_cpu_fallback,
        unsupported_ops: coverage.unsupported_ops.len() as u64,
        total_ops: coverage.bitnet_linear_layers_total,
        cuda_ops: coverage.bitnet_linear_layers_on_cuda,
        mixed_cuda_routes: false,
        fallback_used,
        strict_cuda_ready: bitnet_ready,
        speedup_claim: false,
        full_cuda_residency_claimed: false,
    }
}

fn server_receipt_qk256_execution_coverage(
    coverage: &bitnet_qk256_dispatch::Qk256DispatchCoverageCounters,
) -> ServerSharedEngineExecutionCoverage {
    ServerSharedEngineExecutionCoverage {
        execution_claim: coverage.execution_claim.to_string(),
        bitnet_linear_layers_total: coverage.bitnet_linear_layers_total,
        bitnet_linear_layers_on_cuda: coverage.bitnet_linear_layers_on_cuda,
        bitnet_linear_layers_cpu_fallback: coverage.bitnet_linear_layers_cpu_fallback,
        unsupported_ops: coverage.unsupported_ops.clone(),
        fallback_used: coverage.bitnet_linear_layers_cpu_fallback > 0,
    }
}

fn server_receipt_qk256_kernel_stats(
    evidence: &ServerQk256Evidence,
) -> Vec<ServerSharedEngineKernelStats> {
    vec![ServerSharedEngineKernelStats {
        kernel_id: "qk256_gemv_cuda".to_string(),
        invocations: evidence.coverage.bitnet_linear_layers_on_cuda,
        fallback_invocations: evidence.coverage.bitnet_linear_layers_cpu_fallback,
        cpu_fallback_invocations: evidence.coverage.bitnet_linear_layers_cpu_fallback,
        host_to_device_bytes: (evidence.runtime_stats.host_to_device_bytes > 0)
            .then_some(evidence.runtime_stats.host_to_device_bytes),
        device_to_host_bytes: (evidence.runtime_stats.device_to_host_bytes > 0)
            .then_some(evidence.runtime_stats.device_to_host_bytes),
        kernel_launches: evidence.coverage.bitnet_linear_layers_on_cuda,
        kernel_time_ms: evidence.runtime_stats.kernel_time_ms,
        kernel_time_samples: Some(evidence.runtime_stats.kernel_time_samples),
    }]
}

fn server_receipt_generation_policy(
    request: &ChatCompletionRequest,
) -> ServerSharedEngineGenerationPolicy {
    let temperature = request.temperature.unwrap_or(0.0);
    let top_p = request.top_p.unwrap_or(1.0);

    ServerSharedEngineGenerationPolicy {
        max_tokens: request.max_tokens.unwrap_or(16),
        temperature,
        top_p,
        decoding: if temperature == 0.0 { "greedy".to_string() } else { "sampling".to_string() },
    }
}

struct ServerReceiptModelCoverage {
    row: &'static str,
    tier: &'static str,
    route: &'static str,
}

fn server_receipt_route(
    configured_device: &DeviceConfig,
    request: &ChatCompletionRequest,
    active_model: &model_manager::ModelMetadata,
) -> String {
    if matches!(configured_device, DeviceConfig::NvidiaRtx5070TiCuda)
        && dense_qwen_server_coverage_for_request(request, active_model).is_some()
    {
        DENSE_QWEN_ROUTE.to_string()
    } else if matches!(configured_device, DeviceConfig::NvidiaRtx5070TiCuda)
        && is_official_bitnet_qk256_model(request, active_model)
    {
        BITNET_QK256_ROUTE.to_string()
    } else {
        SHARED_ENGINE_ROUTE.to_string()
    }
}

fn server_receipt_model_coverage(
    route: &str,
    request: &ChatCompletionRequest,
    active_model: &model_manager::ModelMetadata,
) -> Option<ServerReceiptModelCoverage> {
    if route == DENSE_QWEN_ROUTE
        && let Some(coverage) = dense_qwen_server_coverage_for_request(request, active_model)
    {
        return Some(coverage);
    }
    if route == BITNET_QK256_ROUTE && is_official_bitnet_qk256_model(request, active_model) {
        return Some(ServerReceiptModelCoverage {
            row: "bitnet_official_2b_i2s_qk256",
            tier: "product_cli_ready",
            route: BITNET_QK256_ROUTE,
        });
    }
    None
}

fn server_model_coverage_for_active_model(
    configured_device: &DeviceConfig,
    active_model: &model_manager::ModelMetadata,
) -> Option<ServerReceiptModelCoverage> {
    if !matches!(configured_device, DeviceConfig::NvidiaRtx5070TiCuda) {
        return None;
    }

    if active_model_device_is_cuda(active_model)
        && active_model
            .model_sha256
            .as_deref()
            .is_some_and(|sha256| sha256.eq_ignore_ascii_case(DENSE_QWEN25_Q8_MODEL_SHA256))
    {
        return Some(ServerReceiptModelCoverage {
            row: "dense_qwen25_05b_q8_cuda",
            tier: "product_cli_ready",
            route: DENSE_QWEN_ROUTE,
        });
    }

    if active_model_device_is_cuda(active_model)
        && active_model
            .model_sha256
            .as_deref()
            .is_some_and(|sha256| sha256.eq_ignore_ascii_case(DENSE_QWEN3_Q8_MODEL_SHA256))
    {
        return Some(ServerReceiptModelCoverage {
            row: "dense_qwen3_06b_q8_candidate",
            tier: "product_cli_ready",
            route: DENSE_QWEN_ROUTE,
        });
    }

    if active_model_device_is_cuda(active_model)
        && active_model
            .model_sha256
            .as_deref()
            .is_some_and(|sha256| sha256.eq_ignore_ascii_case(BITNET_QK256_MODEL_SHA256))
    {
        return Some(ServerReceiptModelCoverage {
            row: "bitnet_official_2b_i2s_qk256",
            tier: "product_cli_ready",
            route: BITNET_QK256_ROUTE,
        });
    }

    None
}

fn dense_qwen_server_coverage_for_request(
    request: &ChatCompletionRequest,
    active_model: &model_manager::ModelMetadata,
) -> Option<ServerReceiptModelCoverage> {
    if !active_model_device_is_cuda(active_model) {
        return None;
    }
    let sha256 = active_model.model_sha256.as_deref()?;

    if request.model == DENSE_QWEN25_Q8_MODEL_ID
        && sha256.eq_ignore_ascii_case(DENSE_QWEN25_Q8_MODEL_SHA256)
    {
        return Some(ServerReceiptModelCoverage {
            row: "dense_qwen25_05b_q8_cuda",
            tier: "product_cli_ready",
            route: DENSE_QWEN_ROUTE,
        });
    }

    if request.model == DENSE_QWEN3_Q8_MODEL_ID
        && sha256.eq_ignore_ascii_case(DENSE_QWEN3_Q8_MODEL_SHA256)
    {
        return Some(ServerReceiptModelCoverage {
            row: "dense_qwen3_06b_q8_candidate",
            tier: "product_cli_ready",
            route: DENSE_QWEN_ROUTE,
        });
    }

    None
}

fn is_official_bitnet_qk256_model(
    request: &ChatCompletionRequest,
    active_model: &model_manager::ModelMetadata,
) -> bool {
    request.model == BITNET_QK256_MODEL_ID
        && active_model_device_is_cuda(active_model)
        && active_model
            .model_sha256
            .as_deref()
            .is_some_and(|sha256| sha256.eq_ignore_ascii_case(BITNET_QK256_MODEL_SHA256))
}

fn active_model_device_is_cuda(active_model: &model_manager::ModelMetadata) -> bool {
    active_model.device.to_ascii_lowercase().contains("cuda")
}

fn runtime_api_label(
    configured_device: &DeviceConfig,
    active_model: &model_manager::ModelMetadata,
) -> String {
    if preserves_configured_backend_label(configured_device)
        && active_model_device_is_cuda(active_model)
    {
        "cuda".to_string()
    } else {
        active_model.device.clone()
    }
}

fn selected_backend_label(
    configured_device: &DeviceConfig,
    active_model: &model_manager::ModelMetadata,
) -> String {
    if matches!(configured_device, DeviceConfig::NvidiaRtx5070TiCuda)
        && !active_model_device_is_cuda(active_model)
    {
        return active_model.device.clone();
    }

    if preserves_configured_backend_label(configured_device) {
        configured_device.backend_label()
    } else {
        active_model.device.clone()
    }
}

fn preserves_configured_backend_label(configured_device: &DeviceConfig) -> bool {
    matches!(
        configured_device,
        DeviceConfig::NvidiaRtx5070TiCuda
            | DeviceConfig::NvidiaRtx5070TiWgpu
            | DeviceConfig::IntelNpu(_)
            | DeviceConfig::OpenVinoNpu
            | DeviceConfig::AppleM4Metal
            | DeviceConfig::AppleM4MpsGraph
            | DeviceConfig::AppleM4CpuNeon
            | DeviceConfig::AppleM3AirMetal
            | DeviceConfig::AppleM3AirMpsGraph
            | DeviceConfig::AppleM3AirCpuNeon
    )
}

fn current_unix_timestamp() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|duration| duration.as_secs()).unwrap_or(0)
}

fn active_model_supports_shared_inference(
    active_model: Option<&model_manager::ModelMetadata>,
) -> bool {
    active_model.is_some()
}

fn shared_engine_readiness_reason(
    active_model_id: Option<&str>,
    real_server_inference_ready: bool,
) -> Option<String> {
    if active_model_id.is_none() {
        Some("no_active_model".to_string())
    } else if !real_server_inference_ready {
        Some("server_shared_engine_not_available".to_string())
    } else {
        None
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

/// Latest per-request receipt handler.
async fn latest_receipt_handler(State(state): State<ProductionAppState>) -> Response {
    match state.receipt_store.latest().await {
        Some(receipt) => Json(receipt).into_response(),
        None => receipt_lookup_error(
            StatusCode::NOT_FOUND,
            "NO_RECEIPTS",
            "no server request receipts have been captured yet",
            None,
        ),
    }
}

/// Per-request receipt lookup handler.
async fn receipt_by_id_handler(
    State(state): State<ProductionAppState>,
    Path(receipt_id): Path<String>,
) -> Response {
    if !valid_server_receipt_id(&receipt_id) {
        return receipt_lookup_error(
            StatusCode::BAD_REQUEST,
            "INVALID_RECEIPT_ID",
            "receipt id must be 1-128 ASCII letters, digits, '-' or '_' characters",
            Some(receipt_id),
        );
    }

    match state.receipt_store.get(&receipt_id).await {
        Some(receipt) => Json(receipt).into_response(),
        None => receipt_lookup_error(
            StatusCode::NOT_FOUND,
            "RECEIPT_NOT_FOUND",
            "server request receipt was not found",
            Some(receipt_id),
        ),
    }
}

fn receipt_lookup_error(
    status: StatusCode,
    error_code: &str,
    error: &str,
    receipt_id: Option<String>,
) -> Response {
    (
        status,
        Json(ServerReceiptLookupError {
            error_code: error_code.to_string(),
            error: error.to_string(),
            receipt_id,
        }),
    )
        .into_response()
}

fn valid_server_receipt_id(receipt_id: &str) -> bool {
    !receipt_id.is_empty()
        && receipt_id.len() <= 128
        && receipt_id.chars().all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_'))
}

/// Readiness and certification handler.
async fn server_readiness_handler(
    State(state): State<ProductionAppState>,
) -> (StatusCode, Json<ServerReadinessResponse>) {
    let response = collect_server_readiness_response(&state).await;
    let status = if response.ready { StatusCode::OK } else { StatusCode::SERVICE_UNAVAILABLE };

    (status, Json(response))
}

async fn collect_server_readiness_response(state: &ProductionAppState) -> ServerReadinessResponse {
    let model_memory = state.model_manager.get_memory_stats().await;
    let active_model = if let Some(model_id) = &model_memory.active_model_id {
        state.model_manager.get_model_metadata(model_id).await
    } else {
        None
    };
    let device_statuses = state.execution_router.get_device_statuses().await;
    let selected_backend = active_model
        .as_ref()
        .map(|metadata| selected_backend_label(&state.config.server.default_device, metadata));

    build_server_readiness_response(
        model_memory,
        active_model,
        state.config.server.default_model_path.is_some(),
        state.config.server.default_device.backend_label(),
        &state.config.server.default_device,
        state.config.execution_router.fallback_enabled,
        device_statuses,
        selected_backend,
    )
}

fn build_server_readiness_response(
    model_memory: model_manager::ModelMemoryStats,
    active_model: Option<model_manager::ModelMetadata>,
    default_model_configured: bool,
    requested_default_device: String,
    configured_device: &DeviceConfig,
    configured_fallback_enabled: bool,
    device_statuses: Vec<execution_router::DeviceStatus>,
    selected_backend_label: Option<String>,
) -> ServerReadinessResponse {
    let real_server_inference_ready = active_model_supports_shared_inference(active_model.as_ref());
    let active_model_summary = active_model.as_ref().map(|metadata| {
        let coverage = server_model_coverage_for_active_model(configured_device, metadata);
        ServerReadinessActiveModel {
            model_id: metadata.model_id.clone(),
            model_path: metadata.model_path.clone(),
            model_sha256: metadata.model_sha256.clone(),
            model_coverage_row: coverage.as_ref().map(|coverage| coverage.row.to_string()),
            model_coverage_tier: coverage.as_ref().map(|coverage| coverage.tier.to_string()),
            selected_route: coverage.as_ref().map(|coverage| coverage.route.to_string()),
            device: metadata.device.clone(),
            quantization_type: metadata.quantization_type.clone(),
            size_mb: metadata.size_mb,
            parameters: metadata.parameters,
            context_length: metadata.context_length,
        }
    });
    let selected_backend = active_model.as_ref().and(selected_backend_label);

    let batch_inference_ready = false;
    let simulated_inference_enabled = false;
    let reason = shared_engine_readiness_reason(
        model_memory.active_model_id.as_deref(),
        real_server_inference_ready,
    );
    let ready = model_memory.active_model_id.is_some()
        && real_server_inference_ready
        && !simulated_inference_enabled;

    ServerReadinessResponse {
        ready,
        status: if ready { "ready".to_string() } else { "not_ready".to_string() },
        reason,
        model: ServerReadinessModelState {
            default_model_configured,
            loaded_models: model_memory.total_models,
            total_size_mb: model_memory.total_size_mb,
            cache_size_limit: model_memory.cache_size_limit,
            memory_limit_gb: model_memory.memory_limit_gb,
            active_model_id: model_memory.active_model_id,
            active_model: active_model_summary,
        },
        backend: ServerReadinessBackendState {
            requested_default_device,
            selected_backend,
            configured_fallback_enabled,
            server_fallback_policy: "fail_closed_until_real_engine".to_string(),
            device_statuses,
        },
        inference: ServerReadinessInferenceState {
            real_server_inference_ready,
            batch_inference_ready,
            simulated_inference_enabled,
            runtime_path: if real_server_inference_ready {
                "shared_local_inference_engine".to_string()
            } else {
                "unavailable".to_string()
            },
            unavailable_reason: if real_server_inference_ready {
                "available".to_string()
            } else {
                "no active verified model is loaded for the shared local inference engine"
                    .to_string()
            },
        },
        claim_boundary: ServerReadinessClaimBoundary {
            server_ready_claimed: false,
            dense_regular_llm_cuda_inference_claimed: false,
            bitnet_packed_i2s_qk256_proof: false,
            speedup_claim: false,
            full_cuda_residency_claimed: false,
        },
    }
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

#[cfg(test)]
mod tests {
    use super::{
        BitNetServer, ChatCompletionMessage, ChatCompletionRequest, ChatCompletionResponseMetadata,
        ChatCompletionUsage, DeviceConfig, ServerConfig, ServerReceiptStore,
        build_server_readiness_response, build_server_shared_engine_receipt,
        generation_config_from_chat_request, parse_device, render_chat_completion_prompt,
    };
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use serde_json::Value;
    use std::time::SystemTime;
    use tower::ServiceExt;

    use crate::model_manager::{ModelMemoryStats, ModelMetadata};

    fn qwen25_server_receipt(request_id: &str) -> super::ServerSharedEngineReceipt {
        let request = ChatCompletionRequest {
            model: "qwen2.5-0.5b-instruct-q8_0".to_string(),
            messages: vec![ChatCompletionMessage {
                role: "user".to_string(),
                content: "What is working capital?".to_string(),
            }],
            max_tokens: Some(16),
            temperature: Some(0.0),
            top_p: Some(1.0),
            stream: Some(false),
        };
        let usage =
            ChatCompletionUsage { prompt_tokens: 12, completion_tokens: 4, total_tokens: 16 };
        let metadata = ModelMetadata {
            model_id: "qwen2.5-0.5b-instruct-q8_0".to_string(),
            model_path: "models/qwen2.5-0.5b-q8_0.gguf".to_string(),
            model_sha256: Some(
                "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e".to_string(),
            ),
            device: "Cuda(0)".to_string(),
            quantization_type: "Q8_0".to_string(),
            loaded_at: SystemTime::UNIX_EPOCH,
            size_mb: 512,
            parameters: 500_000_000,
            context_length: 32_768,
            inference_count: 0,
            avg_tokens_per_second: 0.0,
        };

        build_server_shared_engine_receipt(
            request_id,
            &request,
            &metadata,
            &DeviceConfig::NvidiaRtx5070TiCuda,
            "chatml",
            &usage,
            "Working capital is current assets minus current liabilities.",
            25,
            None,
        )
    }

    fn qwen3_server_receipt(request_id: &str) -> super::ServerSharedEngineReceipt {
        let request = ChatCompletionRequest {
            model: "qwen3-0.6b-instruct-q8_0".to_string(),
            messages: vec![ChatCompletionMessage {
                role: "user".to_string(),
                content: "Say OK.".to_string(),
            }],
            max_tokens: Some(2),
            temperature: Some(0.0),
            top_p: Some(1.0),
            stream: Some(false),
        };
        let usage =
            ChatCompletionUsage { prompt_tokens: 15, completion_tokens: 1, total_tokens: 16 };
        let metadata = ModelMetadata {
            model_id: "qwen3-0.6b-instruct-q8_0".to_string(),
            model_path: "models/Qwen3-0.6B-Q8_0.gguf".to_string(),
            model_sha256: Some(
                "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031".to_string(),
            ),
            device: "Cuda(0)".to_string(),
            quantization_type: "Q8_0".to_string(),
            loaded_at: SystemTime::UNIX_EPOCH,
            size_mb: 620,
            parameters: 600_000_000,
            context_length: 32_768,
            inference_count: 0,
            avg_tokens_per_second: 0.0,
        };

        build_server_shared_engine_receipt(
            request_id,
            &request,
            &metadata,
            &DeviceConfig::NvidiaRtx5070TiCuda,
            "chatml",
            &usage,
            "OK",
            31,
            None,
        )
    }

    #[tokio::test]
    async fn server_receipt_store_exports_latest_and_by_id() {
        let store = ServerReceiptStore::default();
        let first = qwen25_server_receipt("receipt-1");
        let second = qwen25_server_receipt("receipt-2");

        store.insert(first).await;
        store.insert(second).await;

        let latest = store.latest().await.expect("latest receipt");
        assert_eq!(latest.request_id, "receipt-2");
        assert_eq!(latest.model_coverage_row.as_deref(), Some("dense_qwen25_05b_q8_cuda"));

        let by_id = store.get("receipt-1").await.expect("receipt by id");
        assert_eq!(by_id.request_id, "receipt-1");
        assert_eq!(by_id.selected_route, "dense_regular_llm_cuda");
    }

    #[tokio::test]
    async fn server_receipt_endpoints_fail_closed_without_receipts() {
        let server = BitNetServer::new(ServerConfig::default()).await.expect("server init");
        let app = server.create_app();

        let resp = app
            .clone()
            .oneshot(Request::builder().uri("/receipts/latest").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX).await.expect("body read");
        let json: Value = serde_json::from_slice(&body).expect("json body");
        assert_eq!(json["error_code"], "NO_RECEIPTS");

        let resp = app
            .oneshot(Request::builder().uri("/receipts/bad.id").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX).await.expect("body read");
        let json: Value = serde_json::from_slice(&body).expect("json body");
        assert_eq!(json["error_code"], "INVALID_RECEIPT_ID");
    }

    #[tokio::test]
    async fn server_receipt_endpoints_export_seeded_receipt() {
        let server = BitNetServer::new(ServerConfig::default()).await.expect("server init");
        server.receipt_store.insert(qwen25_server_receipt("receipt-1")).await;
        let app = server.create_app();

        let latest = app
            .clone()
            .oneshot(Request::builder().uri("/receipts/latest").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(latest.status(), StatusCode::OK);
        let body = axum::body::to_bytes(latest.into_body(), usize::MAX).await.expect("body read");
        let json: Value = serde_json::from_slice(&body).expect("json body");
        assert_eq!(json["request_id"], "receipt-1");
        assert_eq!(json["model_coverage_row"], "dense_qwen25_05b_q8_cuda");
        assert_eq!(json["server_ready_claimed"], false);

        let by_id = app
            .oneshot(Request::builder().uri("/receipts/receipt-1").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(by_id.status(), StatusCode::OK);
        let body = axum::body::to_bytes(by_id.into_body(), usize::MAX).await.expect("body read");
        let json: Value = serde_json::from_slice(&body).expect("json body");
        assert_eq!(json["request_id"], "receipt-1");
        assert_eq!(json["selected_backend"], "nvidia-rtx-5070-ti-cuda");
    }

    #[test]
    fn parse_device_supports_vulkan_and_opencl_aliases() {
        assert_eq!(parse_device("vulkan").unwrap(), Device::Cuda(0));
        assert_eq!(parse_device("opencl").unwrap(), Device::Cuda(0));
        assert_eq!(parse_device("ocl").unwrap(), Device::Cuda(0));
    }

    #[test]
    fn parse_device_supports_indexed_vulkan_and_opencl_aliases() {
        assert_eq!(parse_device("vulkan:2").unwrap(), Device::Cuda(2));
        assert_eq!(parse_device("opencl:3").unwrap(), Device::Cuda(3));
        assert_eq!(parse_device("ocl:4").unwrap(), Device::Cuda(4));
    }

    #[test]
    fn server_readiness_without_active_model_fails_closed() {
        let response = build_server_readiness_response(
            ModelMemoryStats {
                total_models: 0,
                total_size_mb: 0,
                active_model_id: None,
                cache_size_limit: 3,
                memory_limit_gb: Some(16.0),
            },
            None,
            false,
            "Auto".to_string(),
            &DeviceConfig::Auto,
            true,
            Vec::new(),
            None,
        );

        assert!(!response.ready);
        assert_eq!(response.status, "not_ready");
        assert_eq!(response.reason.as_deref(), Some("no_active_model"));
        assert!(!response.model.default_model_configured);
        assert_eq!(response.model.loaded_models, 0);
        assert!(response.model.active_model.is_none());
        assert_eq!(response.backend.server_fallback_policy, "fail_closed_until_real_engine");
        assert!(!response.inference.real_server_inference_ready);
        assert!(!response.inference.batch_inference_ready);
        assert!(!response.inference.simulated_inference_enabled);
        assert!(!response.claim_boundary.server_ready_claimed);
        assert!(!response.claim_boundary.speedup_claim);
        assert!(!response.claim_boundary.full_cuda_residency_claimed);
    }

    #[test]
    fn server_shared_engine_readiness_with_active_model_is_ready_for_non_streaming_chat() {
        let response = build_server_readiness_response(
            ModelMemoryStats {
                total_models: 1,
                total_size_mb: 512,
                active_model_id: Some("model-1".to_string()),
                cache_size_limit: 3,
                memory_limit_gb: Some(16.0),
            },
            Some(ModelMetadata {
                model_id: "model-1".to_string(),
                model_path: "models/qwen2.5-0.5b-q8_0.gguf".to_string(),
                model_sha256: None,
                device: "Cuda(0)".to_string(),
                quantization_type: "Q8_0".to_string(),
                loaded_at: SystemTime::UNIX_EPOCH,
                size_mb: 512,
                parameters: 500_000_000,
                context_length: 32_768,
                inference_count: 0,
                avg_tokens_per_second: 0.0,
            }),
            true,
            "gpu".to_string(),
            &DeviceConfig::Gpu(0),
            false,
            Vec::new(),
            Some("Cuda(0)".to_string()),
        );

        assert!(response.ready);
        assert_eq!(response.status, "ready");
        assert_eq!(response.reason, None);
        assert!(response.model.default_model_configured);
        assert_eq!(response.model.active_model_id.as_deref(), Some("model-1"));
        let active_model = response.model.active_model.as_ref().expect("active model");
        assert!(active_model.model_coverage_row.is_none());
        assert!(active_model.model_coverage_tier.is_none());
        assert!(active_model.selected_route.is_none());
        assert_eq!(response.backend.selected_backend.as_deref(), Some("Cuda(0)"));
        assert!(!response.backend.configured_fallback_enabled);
        assert!(response.inference.real_server_inference_ready);
        assert!(!response.inference.batch_inference_ready);
        assert_eq!(response.inference.runtime_path, "shared_local_inference_engine");
        assert_eq!(response.inference.unavailable_reason, "available");
        assert!(!response.claim_boundary.server_ready_claimed);
        assert!(!response.claim_boundary.dense_regular_llm_cuda_inference_claimed);
        assert!(!response.claim_boundary.bitnet_packed_i2s_qk256_proof);
        assert!(!response.claim_boundary.speedup_claim);
        assert!(!response.claim_boundary.full_cuda_residency_claimed);
    }

    #[test]
    fn server_readiness_links_exact_profile_model_coverage_row() {
        let response = build_server_readiness_response(
            ModelMemoryStats {
                total_models: 1,
                total_size_mb: 512,
                active_model_id: Some("model-1".to_string()),
                cache_size_limit: 3,
                memory_limit_gb: Some(16.0),
            },
            Some(ModelMetadata {
                model_id: "model-1".to_string(),
                model_path: "models/qwen2.5-0.5b-q8_0.gguf".to_string(),
                model_sha256: Some(
                    "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e".to_string(),
                ),
                device: "Cuda(0)".to_string(),
                quantization_type: "Q8_0".to_string(),
                loaded_at: SystemTime::UNIX_EPOCH,
                size_mb: 512,
                parameters: 500_000_000,
                context_length: 32_768,
                inference_count: 0,
                avg_tokens_per_second: 0.0,
            }),
            true,
            "nvidia-rtx-5070-ti-cuda".to_string(),
            &DeviceConfig::NvidiaRtx5070TiCuda,
            false,
            Vec::new(),
            Some("nvidia-rtx-5070-ti-cuda".to_string()),
        );

        let active_model = response.model.active_model.as_ref().expect("active model");
        assert_eq!(active_model.model_coverage_row.as_deref(), Some("dense_qwen25_05b_q8_cuda"));
        assert_eq!(active_model.model_coverage_tier.as_deref(), Some("product_cli_ready"));
        assert_eq!(active_model.selected_route.as_deref(), Some("dense_regular_llm_cuda"));
        assert!(!response.claim_boundary.server_ready_claimed);
        assert!(!response.claim_boundary.speedup_claim);
        assert!(!response.claim_boundary.full_cuda_residency_claimed);
    }

    #[test]
    fn server_readiness_links_qwen3_model_coverage_without_server_claims()
    -> Result<(), &'static str> {
        let response = build_server_readiness_response(
            ModelMemoryStats {
                total_models: 1,
                total_size_mb: 620,
                active_model_id: Some("model-1".to_string()),
                cache_size_limit: 3,
                memory_limit_gb: Some(16.0),
            },
            Some(ModelMetadata {
                model_id: "model-1".to_string(),
                model_path: "models/Qwen3-0.6B-Q8_0.gguf".to_string(),
                model_sha256: Some(
                    "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031".to_string(),
                ),
                device: "Cuda(0)".to_string(),
                quantization_type: "Q8_0".to_string(),
                loaded_at: SystemTime::UNIX_EPOCH,
                size_mb: 620,
                parameters: 600_000_000,
                context_length: 32_768,
                inference_count: 0,
                avg_tokens_per_second: 0.0,
            }),
            true,
            "nvidia-rtx-5070-ti-cuda".to_string(),
            &DeviceConfig::NvidiaRtx5070TiCuda,
            false,
            Vec::new(),
            Some("nvidia-rtx-5070-ti-cuda".to_string()),
        );

        let active_model = response.model.active_model.as_ref().ok_or("active model")?;
        assert_eq!(
            active_model.model_coverage_row.as_deref(),
            Some("dense_qwen3_06b_q8_candidate")
        );
        assert_eq!(active_model.model_coverage_tier.as_deref(), Some("product_cli_ready"));
        assert_eq!(active_model.selected_route.as_deref(), Some("dense_regular_llm_cuda"));
        assert!(!response.claim_boundary.server_ready_claimed);
        assert!(!response.claim_boundary.speedup_claim);
        assert!(!response.claim_boundary.full_cuda_residency_claimed);
        Ok(())
    }

    #[test]
    fn server_shared_engine_renders_qwen_chatml_prompt() {
        let request = ChatCompletionRequest {
            model: "qwen2.5-0.5b-instruct-q8_0".to_string(),
            messages: vec![ChatCompletionMessage {
                role: "user".to_string(),
                content: "Explain deferred revenue.".to_string(),
            }],
            max_tokens: Some(16),
            temperature: Some(0.0),
            top_p: Some(1.0),
            stream: Some(false),
        };

        let (rendered, template) = render_chat_completion_prompt(&request).unwrap();

        assert_eq!(template, "chatml");
        assert!(rendered.contains("<|im_start|>user"));
        assert!(rendered.contains("Explain deferred revenue."));
        assert!(rendered.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn server_shared_engine_rejects_invalid_generation_config() {
        let request = ChatCompletionRequest {
            model: "qwen2.5-0.5b-instruct-q8_0".to_string(),
            messages: vec![ChatCompletionMessage {
                role: "user".to_string(),
                content: "Hi".to_string(),
            }],
            max_tokens: Some(0),
            temperature: Some(0.0),
            top_p: Some(1.0),
            stream: Some(false),
        };

        let error = generation_config_from_chat_request(&request).unwrap_err();

        assert_eq!(error, "max_tokens must be greater than zero");
    }

    #[test]
    fn server_shared_engine_receipt_preserves_claim_boundaries() {
        let request = ChatCompletionRequest {
            model: "qwen2.5-0.5b-instruct-q8_0".to_string(),
            messages: vec![ChatCompletionMessage {
                role: "user".to_string(),
                content: "What is working capital?".to_string(),
            }],
            max_tokens: Some(16),
            temperature: Some(0.0),
            top_p: Some(1.0),
            stream: Some(false),
        };
        let usage =
            ChatCompletionUsage { prompt_tokens: 12, completion_tokens: 4, total_tokens: 16 };
        let metadata = ModelMetadata {
            model_id: "model-1".to_string(),
            model_path: "models/qwen2.5-0.5b-q8_0.gguf".to_string(),
            model_sha256: None,
            device: "Cuda(0)".to_string(),
            quantization_type: "Q8_0".to_string(),
            loaded_at: SystemTime::UNIX_EPOCH,
            size_mb: 512,
            parameters: 500_000_000,
            context_length: 32_768,
            inference_count: 0,
            avg_tokens_per_second: 0.0,
        };

        let receipt = build_server_shared_engine_receipt(
            "request-1",
            &request,
            &metadata,
            &DeviceConfig::Gpu(0),
            "chatml",
            &usage,
            "4",
            25,
            None,
        );

        assert_eq!(receipt.receipt_kind, "server_shared_engine_chat_completion");
        assert_eq!(receipt.runtime_path, "shared_local_inference_engine");
        assert_eq!(receipt.runtime_api, "Cuda(0)");
        assert_eq!(receipt.selected_route, "shared_validated_local_inference_engine");
        assert_eq!(receipt.requested_backend, "gpu");
        assert_eq!(receipt.selected_backend, "Cuda(0)");
        assert!(receipt.model_coverage_row.is_none());
        assert!(!receipt.fallback_used);
        assert!(!receipt.simulated_inference);
        assert!(!receipt.streaming);
        assert!(receipt.generated_text_non_empty);
        assert_eq!(receipt.prompt_template.as_str(), "chatml");
        assert_eq!(receipt.tokenizer_authority.as_str(), "active_model_tokenizer");
        assert_eq!(receipt.prompt_authority.as_str(), "server_chat_template");
        assert_eq!(receipt.quality_gate.gate.as_str(), "server_non_empty_utf8_response");
        assert!(receipt.quality_gate.passed);
        assert!(receipt.quality_gate.utf8_valid);
        assert!(!receipt.quality_gate.broad_chat_quality_claimed);
        assert_eq!(receipt.prompt_tokens, 12);
        assert_eq!(receipt.completion_tokens, 4);
        assert_eq!(receipt.total_ms, 25);
        assert!(!receipt.server_smoke_response_claimed);
        assert!(!receipt.server_ready_claimed);
        assert!(!receipt.speedup_claim);
        assert!(!receipt.full_cuda_residency_claimed);
        assert!(!receipt.dense_regular_llm_cuda_inference_claimed);
        assert!(!receipt.bitnet_packed_i2s_qk256_proof);
    }

    #[test]
    fn server_shared_engine_receipt_preserves_strict_configured_backend_label() {
        let request = ChatCompletionRequest {
            model: "qwen2.5-0.5b-instruct-q8_0".to_string(),
            messages: vec![ChatCompletionMessage {
                role: "user".to_string(),
                content: "What is working capital?".to_string(),
            }],
            max_tokens: Some(16),
            temperature: Some(0.0),
            top_p: Some(1.0),
            stream: Some(false),
        };
        let usage =
            ChatCompletionUsage { prompt_tokens: 12, completion_tokens: 4, total_tokens: 16 };
        let metadata = ModelMetadata {
            model_id: "model-1".to_string(),
            model_path: "models/qwen2.5-0.5b-q8_0.gguf".to_string(),
            model_sha256: Some(
                "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e".to_string(),
            ),
            device: "Cuda(0)".to_string(),
            quantization_type: "Q8_0".to_string(),
            loaded_at: SystemTime::UNIX_EPOCH,
            size_mb: 512,
            parameters: 500_000_000,
            context_length: 32_768,
            inference_count: 0,
            avg_tokens_per_second: 0.0,
        };

        let receipt = build_server_shared_engine_receipt(
            "request-1",
            &request,
            &metadata,
            &DeviceConfig::NvidiaRtx5070TiCuda,
            "chatml",
            &usage,
            "Working capital is current assets minus current liabilities.",
            25,
            None,
        );

        assert_eq!(receipt.selected_backend, "nvidia-rtx-5070-ti-cuda");
        assert_eq!(receipt.runtime_api, "cuda");
        assert_eq!(receipt.model_identity.model_id, request.model.as_str());
        assert_eq!(receipt.model_identity.requested_model, request.model.as_str());
        assert_eq!(receipt.model_identity.active_model_id.as_str(), "model-1");
        assert_eq!(
            receipt.model_identity.model_sha256.as_deref(),
            Some("ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e")
        );
        assert_eq!(receipt.model_sha256.as_deref(), receipt.model_identity.model_sha256.as_deref());
        assert_eq!(receipt.endpoint_profile.endpoint.as_str(), "/v1/chat/completions");
        assert_eq!(receipt.endpoint_profile.method.as_str(), "POST");
        assert_eq!(
            receipt.endpoint_profile.request_profile.as_str(),
            "non_streaming_chat_completion"
        );
        assert!(!receipt.endpoint_profile.streaming);
        assert_eq!(receipt.endpoint_profile.message_count, 1);
        assert_eq!(receipt.generation_policy.max_tokens, 16);
        assert_eq!(receipt.generation_policy.temperature, 0.0);
        assert_eq!(receipt.generation_policy.top_p, 1.0);
        assert_eq!(receipt.generation_policy.decoding.as_str(), "greedy");
        assert_eq!(receipt.prompt_template.as_str(), "chatml");
        assert_eq!(receipt.tokenizer_authority.as_str(), "active_model_tokenizer");
        assert_eq!(receipt.prompt_authority.as_str(), "server_chat_template");
        assert_eq!(receipt.quality_gate.gate.as_str(), "server_non_empty_utf8_response");
        assert_eq!(receipt.requested_backend, "nvidia-rtx-5070-ti-cuda");
        assert_eq!(receipt.selected_route, "dense_regular_llm_cuda");
        assert_eq!(receipt.model_coverage_row.as_deref(), Some("dense_qwen25_05b_q8_cuda"));
        assert_eq!(receipt.model_coverage_tier.as_deref(), Some("product_cli_ready"));
        assert!(!receipt.fallback_used);
        assert!(receipt.generated_text_non_empty);
        assert!(receipt.quality_gate.passed);
        assert!(receipt.server_smoke_response_claimed);
        assert!(!receipt.server_ready_claimed);
        assert!(!receipt.speedup_claim);
        assert!(receipt.dense_regular_llm_cuda_inference_claimed);
        assert!(!receipt.bitnet_packed_i2s_qk256_proof);

        let metadata = ChatCompletionResponseMetadata::from_receipt(&receipt);
        assert_eq!(metadata.receipt_id, "request-1");
        assert_eq!(metadata.receipt_path, "/receipts/request-1");
        assert_eq!(metadata.latest_receipt_path, "/receipts/latest");
        assert_eq!(metadata.readiness_path, "/readiness");
        assert_eq!(metadata.model_coverage_row.as_deref(), Some("dense_qwen25_05b_q8_cuda"));
        assert_eq!(metadata.model_coverage_tier.as_deref(), Some("product_cli_ready"));
        assert_eq!(metadata.selected_backend, "nvidia-rtx-5070-ti-cuda");
        assert_eq!(metadata.selected_route, "dense_regular_llm_cuda");
        assert!(!metadata.fallback_used);
    }

    #[test]
    fn server_shared_engine_receipt_records_qwen3_dense_smoke_boundary() {
        let receipt = qwen3_server_receipt("qwen3-request-1");

        assert_eq!(receipt.selected_backend, "nvidia-rtx-5070-ti-cuda");
        assert_eq!(receipt.runtime_api, "cuda");
        assert_eq!(receipt.model_identity.model_id, "qwen3-0.6b-instruct-q8_0");
        assert_eq!(receipt.requested_backend, "nvidia-rtx-5070-ti-cuda");
        assert_eq!(receipt.selected_route, "dense_regular_llm_cuda");
        assert_eq!(receipt.model_coverage_row.as_deref(), Some("dense_qwen3_06b_q8_candidate"));
        assert_eq!(receipt.model_coverage_tier.as_deref(), Some("product_cli_ready"));
        assert!(!receipt.fallback_used);
        assert!(receipt.generated_text_non_empty);
        assert!(receipt.quality_gate.passed);
        assert!(receipt.server_smoke_response_claimed);
        assert!(!receipt.server_ready_claimed);
        assert!(!receipt.speedup_claim);
        assert!(!receipt.full_cuda_residency_claimed);
        assert!(receipt.dense_regular_llm_cuda_inference_claimed);
        assert!(!receipt.bitnet_packed_i2s_qk256_proof);

        let metadata = ChatCompletionResponseMetadata::from_receipt(&receipt);
        assert_eq!(metadata.receipt_id, "qwen3-request-1");
        assert_eq!(metadata.model_coverage_row.as_deref(), Some("dense_qwen3_06b_q8_candidate"));
        assert_eq!(metadata.model_coverage_tier.as_deref(), Some("product_cli_ready"));
        assert_eq!(metadata.selected_backend, "nvidia-rtx-5070-ti-cuda");
        assert_eq!(metadata.selected_route, "dense_regular_llm_cuda");
        assert!(!metadata.fallback_used);
    }

    #[test]
    fn server_shared_engine_receipt_preserves_m3_air_configured_backend_labels() {
        let request = ChatCompletionRequest {
            model: "qwen2.5-0.5b-instruct-q8_0".to_string(),
            messages: vec![ChatCompletionMessage {
                role: "user".to_string(),
                content: "What is working capital?".to_string(),
            }],
            max_tokens: Some(16),
            temperature: Some(0.0),
            top_p: Some(1.0),
            stream: Some(false),
        };
        let usage =
            ChatCompletionUsage { prompt_tokens: 12, completion_tokens: 4, total_tokens: 16 };
        let metadata = ModelMetadata {
            model_id: "model-1".to_string(),
            model_path: "models/qwen2.5-0.5b-q8_0.gguf".to_string(),
            model_sha256: Some(
                "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e".to_string(),
            ),
            device: "Cuda(0)".to_string(),
            quantization_type: "Q8_0".to_string(),
            loaded_at: SystemTime::UNIX_EPOCH,
            size_mb: 512,
            parameters: 500_000_000,
            context_length: 32_768,
            inference_count: 0,
            avg_tokens_per_second: 0.0,
        };

        for (configured_device, expected_label) in [
            (DeviceConfig::AppleM3AirMetal, "apple-m3-air-metal"),
            (DeviceConfig::AppleM3AirMpsGraph, "apple-m3-air-mpsgraph"),
            (DeviceConfig::AppleM3AirCpuNeon, "apple-m3-air-cpu-neon"),
        ] {
            let receipt = build_server_shared_engine_receipt(
                "request-1",
                &request,
                &metadata,
                &configured_device,
                "chatml",
                &usage,
                "Working capital is current assets minus current liabilities.",
                25,
                None,
            );

            assert_eq!(receipt.requested_backend, expected_label);
            assert_eq!(receipt.selected_backend, expected_label);
            assert_eq!(receipt.runtime_api, "cuda");
            assert!(!receipt.server_smoke_response_claimed);
            assert!(!receipt.dense_regular_llm_cuda_inference_claimed);
            assert!(!receipt.bitnet_packed_i2s_qk256_proof);
        }
    }

    #[test]
    fn server_shared_engine_receipt_requires_exact_qwen_artifact_for_dense_claims() {
        let request = ChatCompletionRequest {
            model: "qwen2.5-0.5b-instruct-q8_0".to_string(),
            messages: vec![ChatCompletionMessage {
                role: "user".to_string(),
                content: "What is working capital?".to_string(),
            }],
            max_tokens: Some(16),
            temperature: Some(0.0),
            top_p: Some(1.0),
            stream: Some(false),
        };
        let usage =
            ChatCompletionUsage { prompt_tokens: 12, completion_tokens: 4, total_tokens: 16 };
        let metadata = ModelMetadata {
            model_id: "model-1".to_string(),
            model_path: "models/qwen2.5-0.5b-q8_0.gguf".to_string(),
            model_sha256: None,
            device: "Cuda(0)".to_string(),
            quantization_type: "Q8_0".to_string(),
            loaded_at: SystemTime::UNIX_EPOCH,
            size_mb: 512,
            parameters: 500_000_000,
            context_length: 32_768,
            inference_count: 0,
            avg_tokens_per_second: 0.0,
        };

        let receipt = build_server_shared_engine_receipt(
            "request-1",
            &request,
            &metadata,
            &DeviceConfig::NvidiaRtx5070TiCuda,
            "chatml",
            &usage,
            "Working capital is current assets minus current liabilities.",
            25,
            None,
        );

        assert_eq!(receipt.selected_backend, "nvidia-rtx-5070-ti-cuda");
        assert_eq!(receipt.runtime_api, "cuda");
        assert_eq!(receipt.selected_route, "shared_validated_local_inference_engine");
        assert!(receipt.model_coverage_row.is_none());
        assert!(receipt.generated_text_non_empty);
        assert!(!receipt.server_smoke_response_claimed);
        assert!(!receipt.server_ready_claimed);
        assert!(!receipt.dense_regular_llm_cuda_inference_claimed);
        assert!(!receipt.bitnet_packed_i2s_qk256_proof);
    }

    #[test]
    fn server_shared_engine_receipt_requires_cuda_loaded_artifact_for_dense_claims() {
        let request = ChatCompletionRequest {
            model: "qwen2.5-0.5b-instruct-q8_0".to_string(),
            messages: vec![ChatCompletionMessage {
                role: "user".to_string(),
                content: "What is working capital?".to_string(),
            }],
            max_tokens: Some(16),
            temperature: Some(0.0),
            top_p: Some(1.0),
            stream: Some(false),
        };
        let usage =
            ChatCompletionUsage { prompt_tokens: 12, completion_tokens: 4, total_tokens: 16 };
        let metadata = ModelMetadata {
            model_id: "model-1".to_string(),
            model_path: "models/qwen2.5-0.5b-q8_0.gguf".to_string(),
            model_sha256: Some(
                "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e".to_string(),
            ),
            device: "Cpu".to_string(),
            quantization_type: "Q8_0".to_string(),
            loaded_at: SystemTime::UNIX_EPOCH,
            size_mb: 512,
            parameters: 500_000_000,
            context_length: 32_768,
            inference_count: 0,
            avg_tokens_per_second: 0.0,
        };

        let receipt = build_server_shared_engine_receipt(
            "request-1",
            &request,
            &metadata,
            &DeviceConfig::NvidiaRtx5070TiCuda,
            "chatml",
            &usage,
            "Working capital is current assets minus current liabilities.",
            25,
            None,
        );

        assert_eq!(receipt.selected_backend, "Cpu");
        assert_eq!(receipt.runtime_api, "Cpu");
        assert_eq!(receipt.selected_route, "shared_validated_local_inference_engine");
        assert!(receipt.model_coverage_row.is_none());
        assert!(receipt.fallback_used);
        assert!(!receipt.server_smoke_response_claimed);
        assert!(!receipt.server_ready_claimed);
        assert!(!receipt.dense_regular_llm_cuda_inference_claimed);
        assert!(!receipt.bitnet_packed_i2s_qk256_proof);
    }

    #[test]
    fn server_shared_engine_receipt_preserves_m3_air_backend_labels() {
        let request = ChatCompletionRequest {
            model: "qwen2.5-0.5b-instruct-q8_0".to_string(),
            messages: vec![ChatCompletionMessage {
                role: "user".to_string(),
                content: "What is working capital?".to_string(),
            }],
            max_tokens: Some(16),
            temperature: Some(0.0),
            top_p: Some(1.0),
            stream: Some(false),
        };
        let usage =
            ChatCompletionUsage { prompt_tokens: 12, completion_tokens: 4, total_tokens: 16 };
        let metadata = ModelMetadata {
            model_id: "model-1".to_string(),
            model_path: "models/qwen2.5-0.5b-q8_0.gguf".to_string(),
            model_sha256: None,
            device: "Cpu".to_string(),
            quantization_type: "Q8_0".to_string(),
            loaded_at: SystemTime::UNIX_EPOCH,
            size_mb: 512,
            parameters: 500_000_000,
            context_length: 32_768,
            inference_count: 0,
            avg_tokens_per_second: 0.0,
        };

        for (configured_device, expected_label) in [
            (DeviceConfig::AppleM3AirMetal, "apple-m3-air-metal"),
            (DeviceConfig::AppleM3AirMpsGraph, "apple-m3-air-mpsgraph"),
            (DeviceConfig::AppleM3AirCpuNeon, "apple-m3-air-cpu-neon"),
        ] {
            let receipt = build_server_shared_engine_receipt(
                "request-1",
                &request,
                &metadata,
                &configured_device,
                "chatml",
                &usage,
                "Working capital is current assets minus current liabilities.",
                25,
                None,
            );

            assert_eq!(receipt.selected_backend, expected_label);
            assert_eq!(receipt.requested_backend, expected_label);
            assert_eq!(receipt.runtime_api, "Cpu");
        }
    }

    #[test]
    fn bitnet_qk256_server_smoke_receipt_records_qk256_claim_boundary() -> Result<(), &'static str>
    {
        let request = ChatCompletionRequest {
            model: "microsoft-bitnet-b1.58-2B-4T-i2s".to_string(),
            messages: vec![ChatCompletionMessage {
                role: "user".to_string(),
                content: "Say OK.".to_string(),
            }],
            max_tokens: Some(2),
            temperature: Some(0.0),
            top_p: Some(1.0),
            stream: Some(false),
        };
        let usage =
            ChatCompletionUsage { prompt_tokens: 14, completion_tokens: 1, total_tokens: 15 };
        let metadata = ModelMetadata {
            model_id: "model-1".to_string(),
            model_path: "models/microsoft-bitnet-b1.58-2B-4T-i2s/ggml-model-i2_s.gguf".to_string(),
            model_sha256: Some(
                "4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162".to_string(),
            ),
            device: "Cuda(0)".to_string(),
            quantization_type: "I2_S/QK256".to_string(),
            loaded_at: SystemTime::UNIX_EPOCH,
            size_mb: 1132,
            parameters: 2_000_000_000,
            context_length: 4096,
            inference_count: 0,
            avg_tokens_per_second: 0.0,
        };
        let evidence = super::ServerQk256Evidence {
            coverage: bitnet_qk256_dispatch::Qk256DispatchCoverageCounters {
                bitnet_linear_layers_total: 420,
                bitnet_linear_layers_on_cuda: 420,
                bitnet_linear_layers_cpu_fallback: 0,
                unsupported_ops: Vec::new(),
                execution_claim: "cuda_inference_contribution",
            },
            runtime_stats: bitnet_qk256_dispatch::Qk256CudaRuntimeStats {
                host_to_device_bytes: 1024,
                device_to_host_bytes: 2048,
                kernel_time_ms: Some(12.5),
                kernel_time_samples: 420,
            },
        };

        let receipt = build_server_shared_engine_receipt(
            "request-bitnet",
            &request,
            &metadata,
            &DeviceConfig::NvidiaRtx5070TiCuda,
            "bitnetcpp-answer",
            &usage,
            "OK",
            83,
            Some(&evidence),
        );

        assert_eq!(receipt.selected_backend, "nvidia-rtx-5070-ti-cuda");
        assert_eq!(receipt.runtime_api, "cuda");
        assert_eq!(receipt.selected_route, "bitnet_qk256_cuda");
        assert_eq!(receipt.model_coverage_row.as_deref(), Some("bitnet_official_2b_i2s_qk256"));
        assert_eq!(receipt.model_coverage_tier.as_deref(), Some("product_cli_ready"));
        assert!(!receipt.fallback_used);
        assert!(receipt.server_smoke_response_claimed);
        assert!(!receipt.server_ready_claimed);
        assert!(!receipt.speedup_claim);
        assert!(!receipt.full_cuda_residency_claimed);
        assert!(!receipt.dense_regular_llm_cuda_inference_claimed);
        assert!(receipt.bitnet_packed_i2s_qk256_proof);
        assert!(receipt.execution_plan.is_some(), "execution plan missing");
        if let Some(plan) = receipt.execution_plan.as_ref() {
            assert_eq!(plan.selected_route, "bitnet_qk256_cuda");
            assert!(plan.bitnet_packed_qk256_cuda);
            assert_eq!(plan.cuda_bitnet_qk256_ops, 420);
            assert_eq!(plan.cpu_fallback_ops, 0);
            assert_eq!(plan.cuda_dense_regular_llm_ops, 0);
            assert!(plan.strict_cuda_ready);
        }
        assert!(receipt.execution_coverage.is_some(), "execution coverage missing");
        if let Some(coverage) = receipt.execution_coverage.as_ref() {
            assert_eq!(coverage.bitnet_linear_layers_on_cuda, 420);
            assert_eq!(coverage.bitnet_linear_layers_cpu_fallback, 0);
        }
        assert!(receipt.kernel_stats.is_some(), "kernel stats missing");
        if let Some(stats) = receipt.kernel_stats.as_ref() {
            assert_eq!(stats[0].kernel_id, "qk256_gemv_cuda");
            assert_eq!(stats[0].invocations, 420);
            assert_eq!(stats[0].fallback_invocations, 0);
            assert_eq!(stats[0].cpu_fallback_invocations, 0);
        }
        Ok(())
    }

    #[test]
    fn server_shared_engine_receipt_records_streaming_sampling_profile() {
        let request = ChatCompletionRequest {
            model: "qwen2.5-0.5b-instruct-q8_0".to_string(),
            messages: vec![
                ChatCompletionMessage {
                    role: "system".to_string(),
                    content: "Answer concisely.".to_string(),
                },
                ChatCompletionMessage {
                    role: "user".to_string(),
                    content: "Summarize liquidity risk.".to_string(),
                },
            ],
            max_tokens: Some(24),
            temperature: Some(0.7),
            top_p: Some(0.9),
            stream: Some(true),
        };
        let usage =
            ChatCompletionUsage { prompt_tokens: 18, completion_tokens: 6, total_tokens: 24 };
        let metadata = ModelMetadata {
            model_id: "model-1".to_string(),
            model_path: "models/qwen2.5-0.5b-q8_0.gguf".to_string(),
            model_sha256: Some(
                "ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e".to_string(),
            ),
            device: "Cuda(0)".to_string(),
            quantization_type: "Q8_0".to_string(),
            loaded_at: SystemTime::UNIX_EPOCH,
            size_mb: 512,
            parameters: 500_000_000,
            context_length: 32_768,
            inference_count: 0,
            avg_tokens_per_second: 0.0,
        };

        let receipt = build_server_shared_engine_receipt(
            "request-2",
            &request,
            &metadata,
            &DeviceConfig::NvidiaRtx5070TiCuda,
            "chatml",
            &usage,
            "Liquidity risk is the chance cash is unavailable when needed.",
            31,
            None,
        );

        assert!(receipt.streaming);
        assert_eq!(receipt.endpoint_profile.request_profile.as_str(), "streaming_chat_completion");
        assert!(receipt.endpoint_profile.streaming);
        assert_eq!(receipt.endpoint_profile.message_count, 2);
        assert_eq!(receipt.generation_policy.max_tokens, 24);
        assert_eq!(receipt.generation_policy.temperature, 0.7);
        assert_eq!(receipt.generation_policy.top_p, 0.9);
        assert_eq!(receipt.generation_policy.decoding.as_str(), "sampling");
        assert_eq!(receipt.selected_backend, "nvidia-rtx-5070-ti-cuda");
        assert_eq!(receipt.selected_route, "dense_regular_llm_cuda");
        assert!(receipt.server_smoke_response_claimed);
        assert!(!receipt.server_ready_claimed);
        assert!(!receipt.speedup_claim);
        assert!(!receipt.full_cuda_residency_claimed);
        assert!(receipt.dense_regular_llm_cuda_inference_claimed);
        assert!(!receipt.bitnet_packed_i2s_qk256_proof);
    }
}
