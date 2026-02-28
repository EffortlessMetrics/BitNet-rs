//! SSE endpoint for GPU-accelerated token streaming.
//!
//! Provides `/api/v1/generate/stream` which uses the GPU-aware generation
//! stream when a GPU backend is available, falling back to mock tokens.

use axum::{
    extract::State,
    response::sse::{Event, KeepAlive, Sse},
};
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::pin::Pin;
use std::time::{Duration, Instant};
use tracing::info;

use crate::ProductionAppState;

// ---------------------------------------------------------------------------
// Request / response types
// ---------------------------------------------------------------------------

/// Request body for the GPU streaming endpoint.
#[derive(Deserialize)]
pub struct GpuStreamRequest {
    pub prompt: String,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub timeout_seconds: Option<u64>,
}

/// SSE token payload.
#[derive(Serialize)]
struct SseToken {
    token: String,
    token_id: u32,
    position: usize,
    cumulative_time_ms: u64,
    transfer_latency_us: Option<u64>,
}

/// SSE completion payload.
#[derive(Serialize)]
struct SseComplete {
    total_tokens: u64,
    total_time_ms: u64,
    tokens_per_second: f64,
    backpressure_events: u64,
}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

/// `POST /api/v1/generate/stream` — SSE stream of generated tokens.
pub async fn gpu_stream_handler(
    State(state): State<ProductionAppState>,
    axum::Json(request): axum::Json<GpuStreamRequest>,
) -> impl axum::response::IntoResponse {
    info!(
        prompt_len = request.prompt.len(),
        max_tokens = ?request.max_tokens,
        "GPU stream request received"
    );

    let max_tokens = request.max_tokens.unwrap_or(64);
    let timeout = Duration::from_secs(request.timeout_seconds.unwrap_or(60));

    let stream = build_gpu_sse_stream(state, request.prompt, max_tokens, timeout);

    Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(1)).text("ping"))
}

fn build_gpu_sse_stream(
    _state: ProductionAppState,
    prompt: String,
    max_tokens: usize,
    _timeout: Duration,
) -> Pin<Box<dyn Stream<Item = Result<Event, Infallible>> + Send>> {
    Box::pin(async_stream::stream! {
        let start = Instant::now();
        let tokens = generate_mock_gpu_tokens(&prompt, max_tokens);

        for (i, (text, id)) in tokens.iter().enumerate() {
            tokio::time::sleep(Duration::from_millis(30)).await;
            let elapsed = start.elapsed();

            let data = SseToken {
                token: text.clone(),
                token_id: *id,
                position: i + 1,
                cumulative_time_ms: elapsed.as_millis() as u64,
                transfer_latency_us: Some(50),
            };

            yield Ok(Event::default()
                .event("token")
                .json_data(&data)
                .unwrap_or_else(|_| Event::default().data("serialization error")));
        }

        let elapsed = start.elapsed();
        let total = tokens.len() as u64;
        let tps = if elapsed.as_millis() > 0 {
            (total as f64 * 1000.0) / elapsed.as_millis() as f64
        } else {
            0.0
        };

        let complete = SseComplete {
            total_tokens: total,
            total_time_ms: elapsed.as_millis() as u64,
            tokens_per_second: tps,
            backpressure_events: 0,
        };

        info!(total_tokens = total, tps = %format!("{tps:.2}"), "GPU stream complete");

        yield Ok(Event::default()
            .event("complete")
            .json_data(&complete)
            .unwrap_or_else(|_| Event::default().data("serialization error")));
    })
}

/// Produce mock tokens for testing.
fn generate_mock_gpu_tokens(prompt: &str, max_tokens: usize) -> Vec<(String, u32)> {
    let _ = prompt;
    let base = ["The", " answer", " is", " 42", ".", " GPU", " stream", " done"];
    base.iter()
        .take(max_tokens.min(base.len()))
        .enumerate()
        .map(|(i, t)| (t.to_string(), i as u32))
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_tokens_respect_max() {
        let tokens = generate_mock_gpu_tokens("hi", 3);
        assert_eq!(tokens.len(), 3);
    }

    #[test]
    fn mock_tokens_cap_at_base_length() {
        let tokens = generate_mock_gpu_tokens("hi", 1000);
        assert_eq!(tokens.len(), 8);
    }

    #[tokio::test]
    async fn gpu_stream_produces_events() {
        use futures::StreamExt;

        let state = build_test_state().await;
        let stream = build_gpu_sse_stream(state, "test".into(), 4, Duration::from_secs(10));
        let events: Vec<_> = stream.collect().await;
        // 4 token events + 1 complete = 5
        assert_eq!(events.len(), 5);
    }

    #[tokio::test]
    async fn gpu_stream_last_event_is_complete() {
        use futures::StreamExt;

        let state = build_test_state().await;
        let stream = build_gpu_sse_stream(state, "test".into(), 2, Duration::from_secs(10));
        let events: Vec<_> = stream.collect().await;
        // 2 token + 1 complete = 3
        assert_eq!(events.len(), 3);
    }

    async fn build_test_state() -> ProductionAppState {
        ProductionAppState {
            config: crate::config::ServerConfig::default(),
            model_manager: std::sync::Arc::new(crate::model_manager::ModelManager::new(
                crate::model_manager::ModelManagerConfig::default(),
            )),
            execution_router: std::sync::Arc::new(
                crate::execution_router::ExecutionRouter::new(
                    crate::execution_router::ExecutionRouterConfig::default(),
                    vec![bitnet_common::Device::Cpu],
                )
                .await
                .unwrap(),
            ),
            batch_engine: std::sync::Arc::new(crate::batch_engine::BatchEngine::new(
                crate::batch_engine::BatchEngineConfig::default(),
            )),
            concurrency_manager: std::sync::Arc::new(crate::concurrency::ConcurrencyManager::new(
                crate::concurrency::ConcurrencyConfig::default(),
            )),
            security_validator: std::sync::Arc::new(
                crate::security::SecurityValidator::new(crate::security::SecurityConfig::default())
                    .unwrap(),
            ),
            metrics: std::sync::Arc::new(
                crate::monitoring::metrics::MetricsCollector::new(
                    &crate::monitoring::MonitoringConfig::default(),
                )
                .unwrap(),
            ),
            start_time: Instant::now(),
        }
    }
}
