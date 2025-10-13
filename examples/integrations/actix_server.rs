//! # Actix-web Server Integration Example
//!
//! This example demonstrates how to integrate BitNet.rs with the Actix-web framework
//! to create a robust, production-ready inference API server with middleware support.

use actix_web::{
    web, App, HttpServer, HttpResponse, Result as ActixResult, Error,
    middleware::{Logger, DefaultHeaders, Compress},
    http::{header, StatusCode},
    dev::ServiceRequest,
    error::ResponseError,
};
use actix_web_httpauth::{extractors::bearer::BearerAuth, middleware::HttpAuthentication};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{info, warn, error, instrument};
use anyhow::Result;
use futures_util::{Stream, StreamExt};
use actix_web::web::Bytes;

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
    start_time: Instant,
    api_keys: Arc<Vec<String>>, // Simple API key authentication
}

/// Server statistics for monitoring
#[derive(Debug, Default, Clone, Serialize)]
struct ServerStats {
    total_requests: u64,
    total_tokens_generated: u64,
    average_latency_ms: f64,
    active_requests: u64,
    errors: u64,
    authenticated_requests: u64,
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
    user_id: Option<String>, // For usage tracking
}

/// Response payload for text generation
#[derive(Debug, Serialize)]
struct GenerateResponse {
    generated_text: String,
    tokens_generated: u32,
    latency_ms: u64,
    model_info: ModelInfo,
    request_id: String,
}

/// Model information included in responses
#[derive(Debug, Serialize)]
struct ModelInfo {
    name: String,
    quantization: String,
    device: String,
    version: String,
}

/// Health check response
#[derive(Debug, Serialize)]
struct HealthResponse {
    status: String,
    model_loaded: bool,
    device: String,
    uptime_seconds: u64,
    stats: ServerStats,
    version: String,
}

/// Streaming response chunk
#[derive(Debug, Serialize)]
struct StreamChunk {
    token: Option<String>,
    done: bool,
    token_count: Option<u32>,
    error: Option<String>,
    request_id: String,
}

/// Error response format
#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
    code: String,
    details: Option<String>,
    request_id: String,
}

/// Custom error types for Actix
#[derive(Debug, thiserror::Error)]
enum AppError {
    #[error("Inference error: {0}")]
    InferenceError(String),
    #[error("Model error: {0}")]
    ModelError(String),
    #[error("Validation error: {0}")]
    ValidationError(String),
    #[error("Authentication error: {0}")]
    AuthError(String),
    #[error("Rate limit exceeded")]
    RateLimitError,
}

impl ResponseError for AppError {
    fn error_response(&self) -> HttpResponse {
        let request_id = uuid::Uuid::new_v4().to_string();

        let (status, code) = match self {
            AppError::InferenceError(_) => (StatusCode::INTERNAL_SERVER_ERROR, "INFERENCE_ERROR"),
            AppError::ModelError(_) => (StatusCode::INTERNAL_SERVER_ERROR, "MODEL_ERROR"),
            AppError::ValidationError(_) => (StatusCode::BAD_REQUEST, "VALIDATION_ERROR"),
            AppError::AuthError(_) => (StatusCode::UNAUTHORIZED, "AUTH_ERROR"),
            AppError::RateLimitError => (StatusCode::TOO_MANY_REQUESTS, "RATE_LIMIT_ERROR"),
        };

        let error_response = ErrorResponse {
            error: self.to_string(),
            code: code.to_string(),
            details: None,
            request_id,
        };

        HttpResponse::build(status).json(error_response)
    }
}

/// Main server setup and startup
#[actix_web::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    info!("Starting BitNet Actix-web server...");

    // Initialize application state
    let app_state = initialize_app_state().await?;

    // Start the HTTP server
    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(app_state.clone()))
            .wrap(Logger::default())
            .wrap(Compress::default())
            .wrap(DefaultHeaders::new()
                .add(("X-Version", env!("CARGO_PKG_VERSION")))
                .add(("X-Powered-By", "BitNet.rs")))
            .service(
                web::scope("/api/v1")
                    .wrap(HttpAuthentication::bearer(auth_validator))
                    .service(generate_handler)
                    .service(generate_stream_handler)
                    .service(model_info_handler)
            )
            .service(
                web::scope("/api")
                    .service(health_handler)
                    .service(stats_handler)
                    .service(docs_handler)
            )
            .service(root_handler)
    })
    .bind("0.0.0.0:3000")?
    .run()
    .await?;

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

    // Load API keys from environment
    let api_keys = std::env::var("BITNET_API_KEYS")
        .unwrap_or_else(|_| "demo-key-123,test-key-456".to_string())
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();

    info!("Application state initialized successfully");

    Ok(AppState {
        inference_engine: Arc::new(RwLock::new(inference_engine)),
        config,
        stats: Arc::new(RwLock::new(ServerStats::default())),
        start_time: Instant::now(),
        api_keys: Arc::new(api_keys),
    })
}

/// Authentication validator for bearer tokens
async fn auth_validator(
    req: ServiceRequest,
    credentials: BearerAuth,
) -> Result<ServiceRequest, (Error, ServiceRequest)> {
    let app_state = req.app_data::<web::Data<AppState>>().unwrap();

    if app_state.api_keys.contains(&credentials.token().to_string()) {
        // Update authenticated request stats
        if let Ok(mut stats) = app_state.stats.try_write() {
            stats.authenticated_requests += 1;
        }
        Ok(req)
    } else {
        let error = AppError::AuthError("Invalid API key".to_string());
        Err((error.into(), req))
    }
}

/// Root endpoint with API information
#[actix_web::get("/")]
async fn root_handler() -> ActixResult<HttpResponse> {
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "name": "BitNet.rs Actix Server",
        "version": env!("CARGO_PKG_VERSION"),
        "endpoints": {
            "generate": "POST /api/v1/generate - Generate text from prompt (requires auth)",
            "stream": "POST /api/v1/generate/stream - Stream generated text (requires auth)",
            "health": "GET /api/health - Health check",
            "stats": "GET /api/stats - Server statistics",
            "model": "GET /api/v1/model/info - Model information (requires auth)",
            "docs": "GET /api/docs - API documentation"
        },
        "authentication": "Bearer token required for /api/v1/* endpoints"
    })))
}

/// Generate text from prompt (authenticated endpoint)
#[actix_web::post("/generate")]
#[instrument(skip(app_state))]
async fn generate_handler(
    request: web::Json<GenerateRequest>,
    app_state: web::Data<AppState>,
) -> ActixResult<HttpResponse, AppError> {
    let start_time = Instant::now();
    let request_id = uuid::Uuid::new_v4().to_string();

    // Update stats
    {
        let mut stats = app_state.stats.write().await;
        stats.total_requests += 1;
        stats.active_requests += 1;
    }

    info!("Generating text for prompt (request_id: {})", request_id);

    // Validate request
    if request.prompt.is_empty() {
        return Err(AppError::ValidationError("Prompt cannot be empty".to_string()));
    }

    if request.prompt.len() > 10000 {
        return Err(AppError::ValidationError("Prompt too long (max 10000 characters)".to_string()));
    }

    // Create generation config
    let gen_config = GenerationConfig {
        max_new_tokens: request.max_tokens.unwrap_or(100).min(1000), // Cap at 1000 tokens
        temperature: request.temperature.unwrap_or(0.7).clamp(0.1, 2.0),
        top_p: request.top_p.unwrap_or(0.9).clamp(0.1, 1.0),
        top_k: request.top_k.unwrap_or(50).clamp(1, 100),
        ..Default::default()
    };

    // Generate text
    let generated_text = {
        let mut engine = app_state.inference_engine.write().await;
        engine.generate_with_config(&request.prompt, &gen_config)
            .map_err(|e| AppError::InferenceError(e.to_string()))?
    };

    let latency = start_time.elapsed();
    let tokens_generated = generated_text.split_whitespace().count() as u32;

    // Update stats
    {
        let mut stats = app_state.stats.write().await;
        stats.active_requests -= 1;
        stats.total_tokens_generated += tokens_generated as u64;
        stats.average_latency_ms = (stats.average_latency_ms + latency.as_millis() as f64) / 2.0;
    }

    info!("Generated {} tokens in {:?} (request_id: {})", tokens_generated, latency, request_id);

    let response = GenerateResponse {
        generated_text,
        tokens_generated,
        latency_ms: latency.as_millis() as u64,
        model_info: ModelInfo {
            name: "BitNet-1.58B".to_string(),
            quantization: "I2S".to_string(),
            device: format!("{:?}", Device::Cpu),
            version: "1.0.0".to_string(),
        },
        request_id,
    };

    Ok(HttpResponse::Ok().json(response))
}

/// Stream generated text (authenticated endpoint)
#[actix_web::post("/generate/stream")]
#[instrument(skip(app_state))]
async fn generate_stream_handler(
    request: web::Json<GenerateRequest>,
    app_state: web::Data<AppState>,
) -> ActixResult<HttpResponse, AppError> {
    let request_id = uuid::Uuid::new_v4().to_string();
    info!("Starting streaming generation (request_id: {})", request_id);

    // Validate request
    if request.prompt.is_empty() {
        return Err(AppError::ValidationError("Prompt cannot be empty".to_string()));
    }

    // Create generation config
    let gen_config = GenerationConfig {
        max_new_tokens: request.max_tokens.unwrap_or(100).min(1000),
        temperature: request.temperature.unwrap_or(0.7).clamp(0.1, 2.0),
        top_p: request.top_p.unwrap_or(0.9).clamp(0.1, 1.0),
        top_k: request.top_k.unwrap_or(50).clamp(1, 100),
        ..Default::default()
    };

    // Create the stream
    let stream = async_stream::stream! {
        let mut engine = app_state.inference_engine.write().await;
        let mut token_stream = engine.generate_stream_with_config(&request.prompt, &gen_config);

        let mut token_count = 0;
        while let Some(token_result) = token_stream.next().await {
            match token_result {
                Ok(token_text) => {
                    token_count += 1;
                    let chunk = StreamChunk {
                        token: Some(token_text),
                        done: false,
                        token_count: Some(token_count),
                        error: None,
                        request_id: request_id.clone(),
                    };

                    let json_str = serde_json::to_string(&chunk).unwrap();
                    yield Ok::<_, AppError>(Bytes::from(format!("data: {}\n\n", json_str)));
                }
                Err(e) => {
                    let error_chunk = StreamChunk {
                        token: None,
                        done: true,
                        token_count: Some(token_count),
                        error: Some(e.to_string()),
                        request_id: request_id.clone(),
                    };

                    let json_str = serde_json::to_string(&error_chunk).unwrap();
                    yield Ok(Bytes::from(format!("data: {}\n\n", json_str)));
                    break;
                }
            }
        }

        // Send completion chunk
        let done_chunk = StreamChunk {
            token: None,
            done: true,
            token_count: Some(token_count),
            error: None,
            request_id: request_id.clone(),
        };

        let json_str = serde_json::to_string(&done_chunk).unwrap();
        yield Ok(Bytes::from(format!("data: {}\n\n", json_str)));
    };

    Ok(HttpResponse::Ok()
        .content_type("text/event-stream")
        .insert_header(("Cache-Control", "no-cache"))
        .insert_header(("Connection", "keep-alive"))
        .streaming(stream))
}

/// Health check endpoint (public)
#[actix_web::get("/health")]
async fn health_handler(app_state: web::Data<AppState>) -> ActixResult<HttpResponse> {
    let stats = app_state.stats.read().await;
    let uptime = app_state.start_time.elapsed().as_secs();

    let health = HealthResponse {
        status: "healthy".to_string(),
        model_loaded: true,
        device: format!("{:?}", Device::Cpu),
        uptime_seconds: uptime,
        stats: stats.clone(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    };

    Ok(HttpResponse::Ok().json(health))
}

/// Server statistics endpoint (public)
#[actix_web::get("/stats")]
async fn stats_handler(app_state: web::Data<AppState>) -> ActixResult<HttpResponse> {
    let stats = app_state.stats.read().await;
    Ok(HttpResponse::Ok().json(&*stats))
}

/// Model information endpoint (authenticated)
#[actix_web::get("/model/info")]
async fn model_info_handler(app_state: web::Data<AppState>) -> ActixResult<HttpResponse> {
    let engine = app_state.inference_engine.read().await;
    let config = engine.model_config();

    let info = serde_json::json!({
        "model_name": "BitNet-1.58B",
        "architecture": "BitNet",
        "quantization": "I2S",
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "num_layers": config.num_layers,
        "num_attention_heads": config.num_attention_heads,
        "device": format!("{:?}", Device::Cpu),
        "parameters": "1.58B",
        "memory_usage_mb": 512, // Mock value
    });

    Ok(HttpResponse::Ok().json(info))
}

/// API documentation endpoint (public)
#[actix_web::get("/docs")]
async fn docs_handler() -> ActixResult<HttpResponse> {
    let docs = r#"
# BitNet.rs Actix Server API

## Authentication

All `/api/v1/*` endpoints require Bearer token authentication:

```
Authorization: Bearer your-api-key-here
```

## Endpoints

### POST /api/v1/generate
Generate text from a prompt.

**Headers:**
- `Authorization: Bearer <api-key>`
- `Content-Type: application/json`

**Request Body:**
```json
{
    "prompt": "Your prompt here",
    "max_tokens": 100,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "user_id": "optional-user-id"
}
```

**Response:**
```json
{
    "generated_text": "Generated text here...",
    "tokens_generated": 42,
    "latency_ms": 150,
    "model_info": {
        "name": "BitNet-1.58B",
        "quantization": "I2S",
        "device": "Cpu",
        "version": "1.0.0"
    },
    "request_id": "uuid-here"
}
```

### POST /api/v1/generate/stream
Stream generated text token by token using Server-Sent Events.

### GET /api/health
Health check endpoint (no auth required).

### GET /api/stats
Server statistics (no auth required).

### GET /api/v1/model/info
Model information (requires auth).

## Example Usage

```bash
# Set your API key
export API_KEY="demo-key-123"

# Generate text
curl -X POST http://localhost:3000/api/v1/generate \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The future of AI is", "max_tokens": 50}'

# Stream generation
curl -X POST http://localhost:3000/api/v1/generate/stream \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time", "max_tokens": 100}'

# Health check (no auth)
curl http://localhost:3000/api/health
```

## Features

- **Authentication**: Bearer token authentication for API endpoints
- **Rate Limiting**: Built-in request validation and limits
- **Streaming**: Real-time token streaming via Server-Sent Events
- **Monitoring**: Comprehensive statistics and health checks
- **Middleware**: Compression, logging, and security headers
- **Error Handling**: Detailed error responses with request IDs
- **Validation**: Input validation and sanitization
"#;

    Ok(HttpResponse::Ok()
        .content_type("text/markdown")
        .body(docs))
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

#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::{test, App};

    #[actix_web::test]
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
            start_time: Instant::now(),
            api_keys: Arc::new(vec!["test-key".to_string()]),
        };

        let app = test::init_service(
            App::new()
                .app_data(web::Data::new(app_state))
                .service(health_handler)
        ).await;

        let req = test::TestRequest::get().uri("/health").to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());
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
