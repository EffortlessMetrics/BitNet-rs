//! # Warp Web Server Integration Example
//!
//! This example demonstrates how to integrate BitNet.rs with the Warp web framework
//! to create a lightweight, high-performance inference API server.

use warp::{Filter, Reply, Rejection, reject};
use warp::http::StatusCode;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{info, warn, error, instrument};
use anyhow::Result;
use futures_util::{Stream, StreamExt};

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
}

/// Server statistics for monitoring
#[derive(Debug, Default, Clone, Serialize)]
struct ServerStats {
    total_requests: u64,
    total_tokens_generated: u64,
    average_latency_ms: f64,
    active_requests: u64,
    errors: u64,
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

/// Streaming response chunk
#[derive(Debug, Serialize)]
struct StreamChunk {
    token: Option<String>,
    done: bool,
    token_count: Option<u32>,
    error: Option<String>,
}

/// Error response format
#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
    code: String,
    details: Option<String>,
}

/// Custom error types for Warp
#[derive(Debug)]
struct AppError {
    message: String,
    status: StatusCode,
}

impl reject::Reject for AppError {}

/// Main server setup and startup
#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    info!("Starting BitNet Warp server...");

    // Initialize application state
    let app_state = initialize_app_state().await?;
    
    // Create the API routes
    let api = create_api_routes(app_state);

    // Add CORS and logging
    let routes = api
        .with(warp::cors()
            .allow_any_origin()
            .allow_headers(vec!["content-type"])
            .allow_methods(vec!["GET", "POST", "OPTIONS"]))
        .with(warp::log("bitnet_server"))
        .recover(handle_rejection);

    info!("Server starting on http://0.0.0.0:3000");
    
    // Start the server
    warp::serve(routes)
        .run(([0, 0, 0, 0], 3000))
        .await;

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
        start_time: Instant::now(),
    })
}

/// Create all API routes
fn create_api_routes(
    state: AppState,
) -> impl Filter<Extract = impl Reply, Error = Rejection> + Clone {
    let state_filter = warp::any().map(move || state.clone());

    // Root endpoint
    let root = warp::path::end()
        .and(warp::get())
        .map(|| {
            warp::reply::json(&serde_json::json!({
                "name": "BitNet.rs Warp Server",
                "version": env!("CARGO_PKG_VERSION"),
                "endpoints": {
                    "generate": "POST /api/generate - Generate text from prompt",
                    "stream": "POST /api/generate/stream - Stream generated text",
                    "health": "GET /api/health - Health check",
                    "stats": "GET /api/stats - Server statistics",
                    "model": "GET /api/model/info - Model information"
                }
            }))
        });

    // API routes
    let api = warp::path("api");
    
    // Generate endpoint
    let generate = api
        .and(warp::path("generate"))
        .and(warp::path::end())
        .and(warp::post())
        .and(warp::body::json())
        .and(state_filter.clone())
        .and_then(generate_handler);

    // Stream endpoint
    let stream = api
        .and(warp::path("generate"))
        .and(warp::path("stream"))
        .and(warp::path::end())
        .and(warp::post())
        .and(warp::body::json())
        .and(state_filter.clone())
        .and_then(generate_stream_handler);

    // Health endpoint
    let health = api
        .and(warp::path("health"))
        .and(warp::path::end())
        .and(warp::get())
        .and(state_filter.clone())
        .and_then(health_handler);

    // Stats endpoint
    let stats = api
        .and(warp::path("stats"))
        .and(warp::path::end())
        .and(warp::get())
        .and(state_filter.clone())
        .and_then(stats_handler);

    // Model info endpoint
    let model_info = api
        .and(warp::path("model"))
        .and(warp::path("info"))
        .and(warp::path::end())
        .and(warp::get())
        .and(state_filter.clone())
        .and_then(model_info_handler);

    // Documentation endpoint
    let docs = warp::path("docs")
        .and(warp::path::end())
        .and(warp::get())
        .and_then(docs_handler);

    root
        .or(generate)
        .or(stream)
        .or(health)
        .or(stats)
        .or(model_info)
        .or(docs)
}

/// Generate text from prompt
#[instrument(skip(state))]
async fn generate_handler(
    request: GenerateRequest,
    state: AppState,
) -> Result<impl Reply, Rejection> {
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
            .map_err(|e| {
                error!("Inference error: {}", e);
                warp::reject::custom(AppError {
                    message: format!("Inference failed: {}", e),
                    status: StatusCode::INTERNAL_SERVER_ERROR,
                })
            })?
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

    Ok(warp::reply::json(&response))
}

/// Stream generated text
#[instrument(skip(state))]
async fn generate_stream_handler(
    request: GenerateRequest,
    state: AppState,
) -> Result<impl Reply, Rejection> {
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
                    let chunk = StreamChunk {
                        token: Some(token_text),
                        done: false,
                        token_count: Some(token_count),
                        error: None,
                    };
                    
                    let json_str = serde_json::to_string(&chunk).unwrap();
                    yield Ok::<_, warp::Error>(format!("data: {}\n\n", json_str));
                }
                Err(e) => {
                    let error_chunk = StreamChunk {
                        token: None,
                        done: true,
                        token_count: Some(token_count),
                        error: Some(e.to_string()),
                    };
                    
                    let json_str = serde_json::to_string(&error_chunk).unwrap();
                    yield Ok(format!("data: {}\n\n", json_str));
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
        };
        
        let json_str = serde_json::to_string(&done_chunk).unwrap();
        yield Ok(format!("data: {}\n\n", json_str));
    };

    let event_stream = warp::sse::reply(
        warp::sse::keep_alive().stream(stream.map(|item| {
            match item {
                Ok(data) => Ok(warp::sse::Event::default().data(data)),
                Err(e) => Err(warp::Error::from(e)),
            }
        }))
    );

    Ok(event_stream)
}

/// Health check endpoint
async fn health_handler(state: AppState) -> Result<impl Reply, Rejection> {
    let stats = state.stats.read().await;
    let uptime = state.start_time.elapsed().as_secs();

    let health = HealthResponse {
        status: "healthy".to_string(),
        model_loaded: true,
        device: format!("{:?}", Device::Cpu),
        uptime_seconds: uptime,
        stats: stats.clone(),
    };

    Ok(warp::reply::json(&health))
}

/// Server statistics endpoint
async fn stats_handler(state: AppState) -> Result<impl Reply, Rejection> {
    let stats = state.stats.read().await;
    Ok(warp::reply::json(&*stats))
}

/// Model information endpoint
async fn model_info_handler(state: AppState) -> Result<impl Reply, Rejection> {
    let engine = state.inference_engine.read().await;
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
    });

    Ok(warp::reply::json(&info))
}

/// API documentation endpoint
async fn docs_handler() -> Result<impl Reply, Rejection> {
    let docs = r#"
# BitNet.rs Warp Server API

## Endpoints

### POST /api/generate
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

**Response:**
```json
{
    "generated_text": "Generated text here...",
    "tokens_generated": 42,
    "latency_ms": 150,
    "model_info": {
        "name": "BitNet-1.58B",
        "quantization": "I2S",
        "device": "Cpu"
    }
}
```

### POST /api/generate/stream
Stream generated text token by token using Server-Sent Events.

### GET /api/health
Health check endpoint.

### GET /api/stats
Server statistics.

### GET /api/model/info
Model information.

## Example Usage

```bash
# Generate text
curl -X POST http://localhost:3000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The future of AI is", "max_tokens": 50}'

# Stream generation
curl -X POST http://localhost:3000/api/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time", "max_tokens": 100}'

# Health check
curl http://localhost:3000/api/health
```

## Features

- **Lightweight**: Built with Warp for minimal overhead
- **Streaming**: Real-time token streaming via Server-Sent Events
- **Monitoring**: Built-in statistics and health checks
- **CORS**: Cross-origin resource sharing enabled
- **Error Handling**: Comprehensive error responses
"#;

    Ok(warp::reply::with_header(
        docs,
        "content-type",
        "text/markdown",
    ))
}

/// Handle rejections and convert to proper error responses
async fn handle_rejection(err: Rejection) -> Result<impl Reply, std::convert::Infallible> {
    let (code, message, details) = if err.is_not_found() {
        (StatusCode::NOT_FOUND, "Not Found".to_string(), None)
    } else if let Some(app_error) = err.find::<AppError>() {
        (app_error.status, app_error.message.clone(), None)
    } else if err.find::<warp::reject::MethodNotAllowed>().is_some() {
        (StatusCode::METHOD_NOT_ALLOWED, "Method Not Allowed".to_string(), None)
    } else if err.find::<warp::reject::PayloadTooLarge>().is_some() {
        (StatusCode::PAYLOAD_TOO_LARGE, "Payload Too Large".to_string(), None)
    } else if err.find::<warp::reject::UnsupportedMediaType>().is_some() {
        (StatusCode::UNSUPPORTED_MEDIA_TYPE, "Unsupported Media Type".to_string(), None)
    } else {
        error!("Unhandled rejection: {:?}", err);
        (StatusCode::INTERNAL_SERVER_ERROR, "Internal Server Error".to_string(), Some(format!("{:?}", err)))
    };

    let error_response = ErrorResponse {
        error: message,
        code: code.as_u16().to_string(),
        details,
    };

    Ok(warp::reply::with_status(
        warp::reply::json(&error_response),
        code,
    ))
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
    use warp::test;

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
            start_time: Instant::now(),
        };

        let routes = create_api_routes(app_state);

        let resp = test::request()
            .method("GET")
            .path("/api/health")
            .reply(&routes)
            .await;

        assert_eq!(resp.status(), 200);
    }

    #[tokio::test]
    async fn test_root_endpoint() {
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
        };

        let routes = create_api_routes(app_state);

        let resp = test::request()
            .method("GET")
            .path("/")
            .reply(&routes)
            .await;

        assert_eq!(resp.status(), 200);
    }

    /// Mock tokenizer for testing
    struct MockTokenizer;

    impl MockTokenizer {
        fn new() -> Self {
            Self
        }
    }

    impl bitnet_tokenizers::Tokenizer for MockTokenizer {
        fn encode(&self, _text: &str, _add_special_tokens: bool) -> Result<Vec<u32>> {
            Ok(vec![1, 2, 3])
        }

        fn decode(&self, _tokens: &[u32], _skip_special_tokens: bool) -> Result<String> {
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
    }
}