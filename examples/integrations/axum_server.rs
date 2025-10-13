//! # Axum Web Server Integration Example
//!
//! This example demonstrates how to integrate BitNet.rs with the Axum web framework
//! to create a high-performance inference API server.

use axum::{
    extract::{Query, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response, Sse},
    routing::{get, post},
    Json, Router,
};
use axum::response::sse::{Event, KeepAlive};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tower::ServiceBuilder;
use tower_http::{
    cors::CorsLayer,
    trace::TraceLayer,
    timeout::TimeoutLayer,
};
use tracing::{info, warn, error, instrument};
use anyhow::Result;

// Import BitNet components
use bitnet_inference::{InferenceEngine, GenerationConfig};
use bitnet_models::{Model, ModelLoader};
use bitnet_tokenizers::TokenizerBuilder;
use bitnet_common::{BitNetConfig, Device};

/// Application state shared across handlers
#[derive(Clone)]
struct AppState {
    inference_engine: Arc<RwLock<InferenceEngine>>,
    config: BitNetConfig,
    stats: Arc<RwLock<ServerStats>>,
}

/// Server statistics for monitoring
#[derive(Debug, Default)]
struct ServerStats {
    total_requests: u64,
    total_tokens_generated: u64,
    average_latency_ms: f64,
    active_requests: u64,
}

/// Request payload for text generation
#[derive(Debug, Deserialize)]
struct GenerateRequest {
    prompt: String,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    stream: Option<bool>,
}

/// Response payload for text generation
#[derive(Debug, Serialize)]
struct GenerateResponse {
    generated_text: String,
    tokens_generated: u32,
    latency_ms: u64,
    model_info: ModelInfo,
}

/// Model information included in responses
#[derive(Debug, Serialize)]
struct ModelInfo {
    name: String,
    quantization: String,
    device: String,
}

/// Health check response
#[derive(Debug, Serialize)]
struct HealthResponse {
    status: String,
    model_loaded: bool,
    device: String,
    uptime_seconds: u64,
    stats: ServerStats,
}

/// Error response format
#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
    code: String,
    details: Option<String>,
}

/// Main server setup and startup
#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    info!("Starting BitNet Axum server...");

    // Load model and create inference engine
    let app_state = initialize_app_state().await?;

    // Build the router with all endpoints
    let app = create_router(app_state);

    // Start the server
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    info!("Server listening on http://0.0.0.0:3000");

    axum::serve(listener, app).await?;

    Ok(())
}

/// Initialize application state with model loading
async fn initialize_app_state() -> Result<AppState> {
    info!("Initializing application state...");

    // Load configuration
    let config = BitNetConfig::default();

    // Determine device
    let device = if cfg!(feature = "cuda") && is_cuda_available() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };

    info!("Using device: {:?}", device);

    // Load model (in practice, load from file)
    let model_path = std::env::var("BITNET_MODEL_PATH")
        .unwrap_or_else(|_| "models/bitnet-1.58b.gguf".to_string());

    info!("Loading model from: {}", model_path);
    let model = load_model(&model_path, &device).await?;

    // Load tokenizer
    let tokenizer_name = std::env::var("BITNET_TOKENIZER")
        .unwrap_or_else(|_| "gpt2".to_string());

    info!("Loading tokenizer: {}", tokenizer_name);
    let tokenizer = TokenizerBuilder::from_pretrained(&tokenizer_name)?;

    // Create inference engine
    let inference_engine = InferenceEngine::new(model, tokenizer, device)?;

    info!("Application state initialized successfully");

    Ok(AppState {
        inference_engine: Arc::new(RwLock::new(inference_engine)),
        config,
        stats: Arc::new(RwLock::new(ServerStats::default())),
    })
}

/// Create the main application router
fn create_router(state: AppState) -> Router {
    Router::new()
        // API routes
        .route("/generate", post(generate_handler))
        .route("/generate/stream", post(generate_stream_handler))
        .route("/health", get(health_handler))
        .route("/stats", get(stats_handler))
        .route("/model/info", get(model_info_handler))

        // Utility routes
        .route("/", get(root_handler))
        .route("/docs", get(docs_handler))

        // Add state
        .with_state(state)

        // Add middleware
        .layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(CorsLayer::permissive())
                .layer(TimeoutLayer::new(Duration::from_secs(300))) // 5 minute timeout
        )
}

/// Root endpoint with API information
async fn root_handler() -> impl IntoResponse {
    Json(serde_json::json!({
        "name": "BitNet.rs Inference Server",
        "version": env!("CARGO_PKG_VERSION"),
        "endpoints": {
            "generate": "POST /generate - Generate text from prompt",
            "stream": "POST /generate/stream - Stream generated text",
            "health": "GET /health - Health check",
            "stats": "GET /stats - Server statistics",
            "model": "GET /model/info - Model information"
        }
    }))
}

/// Generate text from prompt
#[instrument(skip(state))]
async fn generate_handler(
    State(state): State<AppState>,
    Json(request): Json<GenerateRequest>,
) -> Result<impl IntoResponse, AppError> {
    let start_time = Instant::now();

    // Update stats
    {
        let mut stats = state.stats.write().await;
        stats.total_requests += 1;
        stats.active_requests += 1;
    }

    info!("Generating text for prompt: {:?}", &request.prompt[..50.min(request.prompt.len())]);

    // Create generation config
    let gen_config = GenerationConfig {
        max_new_tokens: request.max_tokens.unwrap_or(100),
        temperature: request.temperature.unwrap_or(0.7),
        top_p: request.top_p.unwrap_or(0.9),
        top_k: request.top_k.unwrap_or(50),
        ..Default::default()
    };

    // Generate text
    let generated_text = {
        let mut engine = state.inference_engine.write().await;
        engine.generate_with_config(&request.prompt, &gen_config)
            .map_err(|e| AppError::InferenceError(e.to_string()))?
    };

    let latency = start_time.elapsed();
    let tokens_generated = generated_text.split_whitespace().count() as u32;

    // Update stats
    {
        let mut stats = state.stats.write().await;
        stats.active_requests -= 1;
        stats.total_tokens_generated += tokens_generated as u64;
        stats.average_latency_ms = (stats.average_latency_ms + latency.as_millis() as f64) / 2.0;
    }

    info!("Generated {} tokens in {:?}", tokens_generated, latency);

    let response = GenerateResponse {
        generated_text,
        tokens_generated,
        latency_ms: latency.as_millis() as u64,
        model_info: ModelInfo {
            name: "BitNet-1.58B".to_string(),
            quantization: "I2S".to_string(),
            device: format!("{:?}", Device::Cpu),
        },
    };

    Ok(Json(response))
}

/// Stream generated text
#[instrument(skip(state))]
async fn generate_stream_handler(
    State(state): State<AppState>,
    Json(request): Json<GenerateRequest>,
) -> Result<impl IntoResponse, AppError> {
    info!("Starting streaming generation for prompt");

    // Create generation config
    let gen_config = GenerationConfig {
        max_new_tokens: request.max_tokens.unwrap_or(100),
        temperature: request.temperature.unwrap_or(0.7),
        top_p: request.top_p.unwrap_or(0.9),
        top_k: request.top_k.unwrap_or(50),
        ..Default::default()
    };

    // Create the stream
    let stream = async_stream::stream! {
        let mut engine = state.inference_engine.write().await;
        let mut token_stream = engine.generate_stream_with_config(&request.prompt, &gen_config);

        let mut token_count = 0;
        while let Some(token_result) = token_stream.next().await {
            match token_result {
                Ok(token_text) => {
                    token_count += 1;
                    let event = Event::default()
                        .json_data(serde_json::json!({
                            "token": token_text,
                            "token_count": token_count,
                            "done": false
                        }))
                        .unwrap();
                    yield Ok(event);
                }
                Err(e) => {
                    let error_event = Event::default()
                        .json_data(serde_json::json!({
                            "error": e.to_string(),
                            "done": true
                        }))
                        .unwrap();
                    yield Ok(error_event);
                    break;
                }
            }
        }

        // Send completion event
        let done_event = Event::default()
            .json_data(serde_json::json!({
                "done": true,
                "total_tokens": token_count
            }))
            .unwrap();
        yield Ok(done_event);
    };

    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

/// Health check endpoint
async fn health_handler(State(state): State<AppState>) -> impl IntoResponse {
    let stats = state.stats.read().await;
    let uptime = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let health = HealthResponse {
        status: "healthy".to_string(),
        model_loaded: true,
        device: format!("{:?}", Device::Cpu),
        uptime_seconds: uptime,
        stats: stats.clone(),
    };

    Json(health)
}

/// Server statistics endpoint
async fn stats_handler(State(state): State<AppState>) -> impl IntoResponse {
    let stats = state.stats.read().await;
    Json(&*stats)
}

/// Model information endpoint
async fn model_info_handler(State(state): State<AppState>) -> impl IntoResponse {
    let engine = state.inference_engine.read().await;
    let config = engine.model_config();

    Json(serde_json::json!({
        "model_name": "BitNet-1.58B",
        "architecture": "BitNet",
        "quantization": "I2S",
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "num_layers": config.num_layers,
        "num_attention_heads": config.num_attention_heads,
        "device": format!("{:?}", Device::Cpu),
    }))
}

/// API documentation endpoint
async fn docs_handler() -> impl IntoResponse {
    let docs = r#"
# BitNet.rs Inference Server API

## Endpoints

### POST /generate
Generate text from a prompt.

**Request Body:**
```json
{
    "prompt": "Your prompt here",
    "max_tokens": 100,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50
}
```

### POST /generate/stream
Stream generated text token by token.

### GET /health
Health check endpoint.

### GET /stats
Server statistics.

### GET /model/info
Model information.

## Example Usage

```bash
curl -X POST http://localhost:3000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The future of AI is", "max_tokens": 50}'
```
"#;

    (
        [("content-type", "text/markdown")],
        docs
    )
}

/// Load model (mock implementation for example)
async fn load_model(path: &str, device: &Device) -> Result<Box<dyn Model>> {
    info!("Loading model from: {}", path);

    // In practice, this would load a real model
    tokio::time::sleep(Duration::from_millis(100)).await; // Simulate loading time

    Ok(Box::new(MockModel::new()))
}

/// Check if CUDA is available
fn is_cuda_available() -> bool {
    // In practice, this would check for CUDA availability
    false
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
        // Mock implementation
        Ok(Box::new(MockTensor::new(vec![1, 50257])))
    }

    fn generate(&self, _tokens: &[u32], _config: &bitnet_inference::GenerationConfig) -> Result<Vec<u32>> {
        // Mock generation
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

/// Application error types
#[derive(Debug)]
enum AppError {
    InferenceError(String),
    ModelError(String),
    ValidationError(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, error_message, code) = match self {
            AppError::InferenceError(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg, "INFERENCE_ERROR"),
            AppError::ModelError(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg, "MODEL_ERROR"),
            AppError::ValidationError(msg) => (StatusCode::BAD_REQUEST, msg, "VALIDATION_ERROR"),
        };

        let body = Json(ErrorResponse {
            error: error_message,
            code: code.to_string(),
            details: None,
        });

        (status, body).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum_test::TestServer;

    #[tokio::test]
    async fn test_health_endpoint() {
        let app_state = AppState {
            inference_engine: Arc::new(RwLock::new(
                InferenceEngine::new(
                    Box::new(MockModel::new()),
                    Arc::new(MockTokenizer::new()),
                    Device::Cpu,
                ).unwrap()
            )),
            config: BitNetConfig::default(),
            stats: Arc::new(RwLock::new(ServerStats::default())),
        };

        let app = create_router(app_state);
        let server = TestServer::new(app).unwrap();

        let response = server.get("/health").await;
        assert_eq!(response.status_code(), 200);
    }

    /// Mock tokenizer for testing
    struct MockTokenizer;

    impl MockTokenizer {
        fn new() -> Self {
            Self
        }
    }

    impl bitnet_tokenizers::Tokenizer for MockTokenizer {
        fn encode(&self, _text: &str, _add_bos: bool, _add_special: bool) -> Result<Vec<u32>> {
            Ok(vec![1, 2, 3])
        }

        fn decode(&self, _tokens: &[u32]) -> Result<String> {
            Ok("mock decoded text".to_string())
        }

        fn vocab_size(&self) -> usize {
            50257
        }

        fn eos_token_id(&self) -> Option<u32> {
            Some(50256)
        }

        fn pad_token_id(&self) -> Option<u32> {
            None
        }

        fn token_to_piece(&self, token: u32) -> Option<String> {
            Some(format!("<token_{}>", token))
        }
    }
}
