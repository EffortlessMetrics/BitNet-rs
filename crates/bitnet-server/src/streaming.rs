//! Server-Sent Events (SSE) streaming support for real-time generation

use anyhow::Result;
use axum::{
    extract::State,
    response::{
        sse::{Event, KeepAlive, Sse},
        Response,
    },
};
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio_stream::StreamExt;

use crate::AppState;

#[derive(Deserialize)]
pub struct StreamingRequest {
    pub prompt: String,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub repetition_penalty: Option<f32>,
}

#[derive(Serialize)]
pub struct StreamingToken {
    pub token: String,
    pub token_id: u32,
    pub cumulative_time_ms: u64,
}

#[derive(Serialize)]
pub struct StreamingComplete {
    pub total_tokens: u64,
    pub total_time_ms: u64,
    pub tokens_per_second: f64,
}

/// SSE streaming handler for token-by-token generation
pub async fn streaming_handler(
    State(state): State<AppState>,
    axum::Json(request): axum::Json<StreamingRequest>,
) -> Response {
    let stream = if let Some(engine) = &state.engine {
        real_stream(engine, request).await
    } else {
        mock_stream(request).await
    };

    Sse::new(stream)
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(1)))
        .into_response()
}

/// Real streaming using the BitNet engine
async fn real_stream(
    engine: &Arc<RwLock<bitnet_inference::InferenceEngine>>,
    request: StreamingRequest,
) -> impl Stream<Item = Result<Event, Infallible>> {
    let start = std::time::Instant::now();
    let engine = engine.clone();

    async_stream::stream! {
        // Build generation config
        let config = bitnet_inference::GenerationConfig {
            max_new_tokens: request.max_tokens.unwrap_or(64),
            temperature: request.temperature.unwrap_or(1.0),
            top_p: request.top_p.unwrap_or(0.9),
            top_k: request.top_k.unwrap_or(50),
            repetition_penalty: request.repetition_penalty.unwrap_or(1.0),
            ..Default::default()
        };

        // Create streaming config
        let stream_config = bitnet_inference::StreamingConfig {
            stream_tokens: true,
            ..Default::default()
        };

        // Get the engine and create a generation stream
        let engine = engine.read().await;
        match engine.generate_stream(&request.prompt, &config, &stream_config).await {
            Ok(mut stream) => {
                let mut token_count = 0u64;
                
                while let Some(token_result) = stream.next().await {
                    match token_result {
                        Ok(token) => {
                            token_count += 1;
                            let elapsed = start.elapsed();
                            
                            let data = StreamingToken {
                                token: token.text,
                                token_id: token.id,
                                cumulative_time_ms: elapsed.as_millis() as u64,
                            };
                            
                            yield Ok(Event::default()
                                .event("token")
                                .json_data(data)
                                .unwrap());
                        }
                        Err(e) => {
                            yield Ok(Event::default()
                                .event("error")
                                .data(format!("Generation error: {}", e)));
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
                };
                
                yield Ok(Event::default()
                    .event("complete")
                    .json_data(complete)
                    .unwrap());
            }
            Err(e) => {
                yield Ok(Event::default()
                    .event("error")
                    .data(format!("Failed to create stream: {}", e)));
            }
        }
    }
}

/// Mock streaming for testing without a model
async fn mock_stream(
    request: StreamingRequest,
) -> impl Stream<Item = Result<Event, Infallible>> {
    let start = std::time::Instant::now();
    let tokens = vec!["Hello", " ", "from", " ", "BitNet", " ", "server", "!"];
    let max_tokens = request.max_tokens.unwrap_or(8).min(tokens.len());

    async_stream::stream! {
        for (i, token) in tokens.iter().take(max_tokens).enumerate() {
            tokio::time::sleep(Duration::from_millis(50)).await;
            
            let elapsed = start.elapsed();
            let data = StreamingToken {
                token: token.to_string(),
                token_id: i as u32,
                cumulative_time_ms: elapsed.as_millis() as u64,
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
        };
        
        yield Ok(Event::default()
            .event("complete")
            .json_data(complete)
            .unwrap());
    }
}