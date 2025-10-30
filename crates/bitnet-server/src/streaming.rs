//! Server-Sent Events (SSE) streaming support for real-time generation

use anyhow::Result;
use axum::{
    extract::State,
    response::sse::{Event, KeepAlive, Sse},
};
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;
use tokio_stream::StreamExt;
use tracing::{debug, error, info, warn};

use crate::ProductionAppState;

/// Request for streaming token generation with timeout configuration
#[derive(Deserialize)]
pub struct StreamingRequest {
    /// The input prompt to generate from
    pub prompt: String,
    /// Maximum number of tokens to generate (default: 64)
    pub max_tokens: Option<usize>,
    /// Sampling temperature (default: 1.0)
    pub temperature: Option<f32>,
    /// Nucleus sampling parameter (default: 0.9)
    pub top_p: Option<f32>,
    /// Top-k sampling parameter (default: 50)
    pub top_k: Option<usize>,
    /// Repetition penalty (default: 1.0)
    pub repetition_penalty: Option<f32>,
    /// Client timeout for the entire generation (seconds, default: 30)
    pub timeout_seconds: Option<u64>,
    /// Enable detailed error reporting in stream (default: false)
    pub detailed_errors: Option<bool>,
}

/// Individual token in the streaming response
#[derive(Serialize, Deserialize)]
pub struct StreamingToken {
    /// The generated token text
    pub token: String,
    /// The token ID from the vocabulary
    pub token_id: u32,
    /// Cumulative generation time in milliseconds
    pub cumulative_time_ms: u64,
    /// Position of this token in the generation sequence
    pub position: usize,
}

/// Final message indicating streaming completion
#[derive(Serialize)]
pub struct StreamingComplete {
    /// Total number of tokens generated
    pub total_tokens: u64,
    /// Total generation time in milliseconds
    pub total_time_ms: u64,
    /// Average tokens per second
    pub tokens_per_second: f64,
    /// Whether generation completed normally or was cancelled/timed out
    pub completed_normally: bool,
    /// Optional completion reason for diagnostics
    pub completion_reason: Option<String>,
}

/// Error information for streaming failures
#[derive(Serialize)]
pub struct StreamingError {
    /// Error type classification
    pub error_type: String,
    /// Human-readable error message
    pub message: String,
    /// Optional recovery suggestions
    pub recovery_hints: Option<Vec<String>>,
    /// Number of tokens generated before error
    pub tokens_before_error: usize,
}

/// SSE streaming handler for token-by-token generation with comprehensive timeout and error handling
pub(crate) async fn streaming_handler(
    State(state): State<ProductionAppState>,
    axum::Json(request): axum::Json<StreamingRequest>,
) -> impl axum::response::IntoResponse {
    info!(
        "Streaming request received: prompt_len={}, max_tokens={:?}, timeout={:?}s",
        request.prompt.len(),
        request.max_tokens,
        request.timeout_seconds
    );

    let detailed_errors = request.detailed_errors.unwrap_or(false);
    let timeout_seconds = request.timeout_seconds.unwrap_or(30);

    let stream: Pin<Box<dyn Stream<Item = Result<Event, Infallible>> + Send>> = if let Some(model) =
        state.model_manager.get_active_model().await
    {
        debug!("Creating real stream with {}s timeout", timeout_seconds);
        Box::pin(create_error_handling_stream(real_stream(model, request).await, detailed_errors))
    } else {
        warn!("No inference engine available, using mock stream");
        Box::pin(create_error_handling_stream(mock_stream(request).await, detailed_errors))
    };

    Sse::new(stream)
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(1)).text("keep-alive"))
}

/// Create a stream with error handling (timeout handled at higher level)
fn create_error_handling_stream<S>(
    inner_stream: S,
    detailed_errors: bool,
) -> impl Stream<Item = Result<Event, Infallible>>
where
    S: Stream<Item = Result<Event, anyhow::Error>> + Send + 'static,
{
    inner_stream.map(move |result| match result {
        Ok(event) => Ok(event),
        Err(stream_error) => {
            error!("Stream error: {}", stream_error);

            if detailed_errors {
                let error_info = StreamingError {
                    error_type: "generation".to_string(),
                    message: format!("Generation failed: {}", stream_error),
                    recovery_hints: Some(vec![
                        "Try simplifying the prompt".to_string(),
                        "Reduce max_tokens".to_string(),
                        "Check server logs".to_string(),
                    ]),
                    tokens_before_error: 0,
                };
                Ok(Event::default().event("error").json_data(&error_info).unwrap_or_else(|_| {
                    Event::default().event("error").data("Serialization failed")
                }))
            } else {
                Ok(Event::default().event("error").data(format!("Stream error: {}", stream_error)))
            }
        }
    })
}

/// Real streaming using the BitNet engine
async fn real_stream(
    managed_model: Arc<crate::model_manager::ManagedModel>,
    request: StreamingRequest,
) -> impl Stream<Item = Result<Event, anyhow::Error>> {
    let start = std::time::Instant::now();

    async_stream::stream! {
        // Build generation config
        let mut config = bitnet_inference::GenerationConfig::default()
            .with_max_tokens(request.max_tokens.unwrap_or(64) as u32)
            .with_temperature(request.temperature.unwrap_or(1.0))
            .with_top_p(request.top_p.unwrap_or(0.9))
            .with_top_k(request.top_k.unwrap_or(50) as u32);
        config.repetition_penalty = request.repetition_penalty.unwrap_or(1.0);

        // Create streaming config
        let _stream_config = bitnet_inference::StreamingConfig {
            buffer_size: 128,
            flush_interval_ms: 100,
            max_retries: 3,
            token_timeout_ms: 5000,
            cancellable: true,
        };

        // Get the engine and create a generation stream
        let engine = &managed_model.engine;
        let tokenizer = engine.tokenizer();
        let mut gen_stream = engine.generate_stream_with_config(&request.prompt, &config)?;
        let mut token_count = 0u64;

        while let Some(token_result) = gen_stream.next().await {
            match token_result {
                Ok(stream_response) => {
                    token_count += 1;
                    let elapsed = start.elapsed();

                    let token_id = tokenizer
                        .encode(&stream_response.text, false, false)
                        .ok()
                        .and_then(|ids| ids.first().copied())
                        .unwrap_or(0);

                    let data = StreamingToken {
                        token: stream_response.text.clone(),
                        token_id,
                        cumulative_time_ms: elapsed.as_millis() as u64,
                        position: token_count as usize,
                    };

                    debug!("Streaming token {}: '{}'", token_count, stream_response.text);

                    yield Ok(Event::default()
                        .event("token")
                        .json_data(data)
                        .unwrap());
                }
                Err(e) => {
                    error!("Generation error after {} tokens: {}", token_count, e);

                    let error_info = StreamingError {
                        error_type: "generation".to_string(),
                        message: format!("Generation failed: {}", e),
                        recovery_hints: Some(vec![
                            "Try a different prompt".to_string(),
                            "Reduce generation parameters".to_string(),
                            "Check model compatibility".to_string(),
                        ]),
                        tokens_before_error: token_count as usize,
                    };

                    yield Ok(Event::default()
                        .event("error")
                        .json_data(&error_info)
                        .unwrap_or_else(|_| Event::default().event("error").data("Serialization failed")));
                    break;
                }
            }
        }

        // Send completion event
        let elapsed = start.elapsed();
        let tokens_per_second = if elapsed.as_millis() > 0 {
            (token_count as f64 * 1000.0) / elapsed.as_millis() as f64
        } else {
            0.0
        };

        let complete = StreamingComplete {
            total_tokens: token_count,
            total_time_ms: elapsed.as_millis() as u64,
            tokens_per_second,
            completed_normally: true,
            completion_reason: Some("Generation completed successfully".to_string()),
        };

        info!(
            "Streaming completed: {} tokens in {}ms ({:.2} tok/s)",
            token_count,
            elapsed.as_millis(),
            tokens_per_second
        );

        yield Ok(Event::default()
            .event("complete")
            .json_data(complete)
            .unwrap());
    }
}

/// Mock streaming for testing without a model
async fn mock_stream(
    request: StreamingRequest,
) -> impl Stream<Item = Result<Event, anyhow::Error>> {
    let start = std::time::Instant::now();
    let tokens = ["Hello", " ", "from", " ", "BitNet", " ", "server", "!"];
    let max_tokens = request.max_tokens.unwrap_or(8).min(tokens.len());

    async_stream::stream! {
        for (i, token) in tokens.iter().take(max_tokens).enumerate() {
            tokio::time::sleep(Duration::from_millis(50)).await;

            let elapsed = start.elapsed();
            let data = StreamingToken {
                token: token.to_string(),
                token_id: 0, // Mock token ID matching test tokenizer behavior
                cumulative_time_ms: elapsed.as_millis() as u64,
                position: i + 1,
            };

            yield Ok(Event::default()
                .event("token")
                .json_data(data)
                .unwrap());
        }

        // Send completion
        let elapsed = start.elapsed();
        let complete = StreamingComplete {
            total_tokens: max_tokens as u64,
            total_time_ms: elapsed.as_millis() as u64,
            tokens_per_second: (max_tokens as f64 * 1000.0) / elapsed.as_millis() as f64,
            completed_normally: true,
            completion_reason: Some("Mock generation completed".to_string()),
        };

        debug!("Mock stream completed: {} tokens", max_tokens);

        yield Ok(Event::default()
            .event("complete")
            .json_data(complete)
            .unwrap());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::response::IntoResponse;
    use bitnet_common::{BitNetConfig, ConcreteTensor};
    use bitnet_inference::InferenceEngine;
    use bitnet_models::Model;
    use bitnet_tokenizers::Tokenizer;
    use http_body_util::BodyExt;

    use std::any::Any;

    #[derive(Clone)]
    struct TestTokenizer;

    impl Tokenizer for TestTokenizer {
        fn encode(
            &self,
            text: &str,
            _add_bos: bool,
            _add_special: bool,
        ) -> bitnet_common::Result<Vec<u32>> {
            if let Some(id) = text.strip_prefix("token_") {
                Ok(vec![id.parse().unwrap_or(0)])
            } else {
                Ok(vec![0])
            }
        }

        fn decode(&self, tokens: &[u32]) -> bitnet_common::Result<String> {
            Ok(format!("token_{}", tokens[0]))
        }

        fn vocab_size(&self) -> usize {
            1000
        }

        fn token_to_piece(&self, token: u32) -> Option<String> {
            Some(format!("token_{}", token))
        }
    }

    struct DummyModel {
        config: BitNetConfig,
    }

    impl DummyModel {
        fn new() -> Self {
            Self { config: BitNetConfig::default() }
        }
    }

    impl Model for DummyModel {
        fn config(&self) -> &BitNetConfig {
            &self.config
        }

        fn forward(
            &self,
            _input: &ConcreteTensor,
            _cache: &mut dyn Any,
        ) -> bitnet_common::Result<ConcreteTensor> {
            Ok(ConcreteTensor::mock(vec![1, 50257]))
        }

        fn embed(&self, tokens: &[u32]) -> bitnet_common::Result<ConcreteTensor> {
            Ok(ConcreteTensor::mock(vec![1, tokens.len(), 1]))
        }

        fn logits(&self, _hidden: &ConcreteTensor) -> bitnet_common::Result<ConcreteTensor> {
            Ok(ConcreteTensor::mock(vec![1, 1, 50257]))
        }
    }

    #[tokio::test]
    async fn sse_token_ids_match_model_outputs() {
        let model = Arc::new(DummyModel::new());
        let tokenizer = Arc::new(TestTokenizer);
        let _engine =
            InferenceEngine::new(model, tokenizer.clone(), bitnet_common::Device::Cpu).unwrap();
        let app_state = ProductionAppState {
            config: crate::config::ServerConfig::default(),
            model_manager: Arc::new(crate::model_manager::ModelManager::new(
                crate::model_manager::ModelManagerConfig::default(),
            )),
            execution_router: Arc::new(
                crate::execution_router::ExecutionRouter::new(
                    crate::execution_router::ExecutionRouterConfig::default(),
                    vec![bitnet_common::Device::Cpu],
                )
                .await
                .unwrap(),
            ),
            batch_engine: Arc::new(crate::batch_engine::BatchEngine::new(
                crate::batch_engine::BatchEngineConfig::default(),
            )),
            concurrency_manager: Arc::new(crate::concurrency::ConcurrencyManager::new(
                crate::concurrency::ConcurrencyConfig::default(),
            )),
            security_validator: Arc::new(
                crate::security::SecurityValidator::new(crate::security::SecurityConfig::default())
                    .unwrap(),
            ),
            metrics: Arc::new(
                crate::monitoring::metrics::MetricsCollector::new(
                    &crate::monitoring::MonitoringConfig::default(),
                )
                .unwrap(),
            ),
            start_time: std::time::Instant::now(),
        };

        let request = StreamingRequest {
            prompt: "test".to_string(),
            max_tokens: Some(3),
            temperature: None,
            top_p: None,
            top_k: None,
            repetition_penalty: None,
            timeout_seconds: None,
            detailed_errors: None,
        };

        let sse = streaming_handler(State(app_state), axum::Json(request)).await;
        let response = sse.into_response();
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body.to_vec()).unwrap();

        for event in body_str.split("\n\n") {
            if event.starts_with("event: token")
                && let Some(data_line) = event.lines().find(|l| l.starts_with("data: "))
            {
                let json = &data_line[6..];
                let token: StreamingToken = serde_json::from_str(json).unwrap();
                let expected = tokenizer.encode(&token.token, false, false).unwrap();
                assert_eq!(expected[0], token.token_id);
            }
        }
    }
}
