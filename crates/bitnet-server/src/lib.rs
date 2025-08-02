//! HTTP server for BitNet inference

use axum::{routing::post, Router};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
pub struct InferenceRequest {
    pub prompt: String,
    pub max_tokens: Option<usize>,
}

#[derive(Serialize)]
pub struct InferenceResponse {
    pub text: String,
}

pub fn create_app() -> Router {
    Router::new()
        .route("/inference", post(inference_handler))
}

async fn inference_handler(
    axum::Json(request): axum::Json<InferenceRequest>,
) -> axum::Json<InferenceResponse> {
    // Placeholder implementation
    axum::Json(InferenceResponse {
        text: format!("Response to: {}", request.prompt),
    })
}