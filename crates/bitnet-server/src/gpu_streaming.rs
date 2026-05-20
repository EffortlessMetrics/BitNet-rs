//! SSE endpoint for GPU-accelerated token streaming.
//!
//! Provides `/api/v1/generate/stream` for GPU-aware generation.
//!
//! The server must not synthesize tokens when the real GPU stream is not wired.

use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};
use std::time::Duration;
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

#[derive(Serialize)]
struct GpuStreamUnavailable {
    error: &'static str,
    error_code: &'static str,
    fallback_used: bool,
    tokens_generated: u64,
}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

/// `POST /api/v1/generate/stream` — SSE stream of generated tokens.
pub async fn gpu_stream_handler(
    State(state): State<ProductionAppState>,
    axum::Json(request): axum::Json<GpuStreamRequest>,
) -> Response {
    let _ = state;
    info!(
        prompt_len = request.prompt.len(),
        max_tokens = ?request.max_tokens,
        "GPU stream request received"
    );

    let _timeout = Duration::from_secs(request.timeout_seconds.unwrap_or(60));

    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(GpuStreamUnavailable {
            error: "GPU streaming inference is unavailable until it is wired to a real engine",
            error_code: "SERVER_REAL_INFERENCE_UNAVAILABLE",
            fallback_used: false,
            tokens_generated: 0,
        }),
    )
        .into_response()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use axum::response::IntoResponse;
    use http_body_util::BodyExt;
    use std::time::Instant;

    #[tokio::test]
    async fn gpu_stream_without_real_engine_returns_503_instead_of_mock_tokens() {
        let state = build_test_state().await;
        let request = GpuStreamRequest {
            prompt: "test".into(),
            max_tokens: Some(4),
            temperature: None,
            top_p: None,
            top_k: None,
            timeout_seconds: Some(10),
        };
        let response = gpu_stream_handler(State(state), axum::Json(request)).await.into_response();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body_str = String::from_utf8(body.to_vec()).unwrap();
        assert!(body_str.contains("SERVER_REAL_INFERENCE_UNAVAILABLE"));
        assert!(body_str.contains("\"fallback_used\":false"));
        assert!(body_str.contains("\"tokens_generated\":0"));
        assert!(!body_str.contains(" answer"));
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
            receipt_store: std::sync::Arc::new(crate::ServerReceiptStore::default()),
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
