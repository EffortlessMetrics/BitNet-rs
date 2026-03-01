//! Integration tests for the OpenAI-compatible `/v1/completions` and
//! `/v1/chat/completions` endpoints.
//!
//! These tests exercise the route registration, request deserialization,
//! and response schema without requiring a real loaded model (the server
//! returns 503 when no model is loaded, which we assert as the expected
//! "no-model" path).  Full inference tests live behind `#[ignore]` and
//! need a model file.

#![allow(unused)]

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use bitnet_server::{BitNetServer, config::ServerConfig};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tower::ServiceExt; // for `oneshot`

// ---------------------------------------------------------------------------
// Helper: build a minimal test server (no model loaded)
// ---------------------------------------------------------------------------

async fn make_app() -> axum::Router {
    let config = ServerConfig::default();
    let server = BitNetServer::new(config).await.expect("server init failed");
    server.create_app()
}

// ---------------------------------------------------------------------------
// /v1/completions — no model loaded → 503
// ---------------------------------------------------------------------------

#[tokio::test]
async fn completions_without_model_returns_503() {
    let app = make_app().await;

    let body = json!({
        "model": "bitnet",
        "prompt": "Hello, world!",
        "max_tokens": 4
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(
        response.status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "/v1/completions should return 503 when no model is loaded"
    );
}

// ---------------------------------------------------------------------------
// /v1/chat/completions — no model loaded → 503
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_completions_without_model_returns_503() {
    let app = make_app().await;

    let body = json!({
        "model": "bitnet",
        "messages": [
            {"role": "user", "content": "What is 2+2?"}
        ],
        "max_tokens": 4
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(
        response.status(),
        StatusCode::SERVICE_UNAVAILABLE,
        "/v1/chat/completions should return 503 when no model is loaded"
    );
}

// ---------------------------------------------------------------------------
// /v1/chat/completions — empty messages array → 400
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_completions_empty_messages_returns_400() {
    let app = make_app().await;

    let body = json!({
        "model": "bitnet",
        "messages": []
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(
        response.status(),
        StatusCode::BAD_REQUEST,
        "empty messages array should be rejected with 400"
    );
}

// ---------------------------------------------------------------------------
// /v1/completions — schema validation: response is valid JSON with expected keys
// (exercised only when a model is loaded; ignored in normal CI)
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "requires real model file - set BITNET_GGUF and BITNET_TOKENIZER environment variables"]
async fn completions_with_model_returns_openai_schema() {
    // This test is exercised manually or in model-available CI.
    // It verifies that the response matches the OpenAI completions schema.
    let model_path =
        std::env::var("BITNET_GGUF").expect("BITNET_GGUF must be set for this test");
    let tokenizer_path = std::env::var("BITNET_TOKENIZER").ok();

    let mut config = ServerConfig::default();
    config.server.default_model_path = Some(model_path);
    config.server.default_tokenizer_path = tokenizer_path;

    let server = BitNetServer::new(config).await.expect("server init failed");
    let app = server.create_app();

    let body = json!({
        "model": "bitnet",
        "prompt": "What is 2+2?",
        "max_tokens": 8,
        "temperature": 0.0
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK, "completions should return 200 with model loaded");

    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&bytes).expect("response should be valid JSON");

    // Verify OpenAI schema keys are present.
    assert!(json.get("id").is_some(), "response must have 'id'");
    assert!(json.get("object").is_some(), "response must have 'object'");
    assert!(json.get("choices").is_some(), "response must have 'choices'");
    assert!(json.get("usage").is_some(), "response must have 'usage'");

    let choices = json["choices"].as_array().expect("choices must be an array");
    assert!(!choices.is_empty(), "choices array must not be empty");
    assert!(choices[0].get("text").is_some(), "choice must have 'text'");
    assert!(choices[0].get("finish_reason").is_some(), "choice must have 'finish_reason'");
}

// ---------------------------------------------------------------------------
// /v1/chat/completions — schema validation (requires model)
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "requires real model file - set BITNET_GGUF and BITNET_TOKENIZER environment variables"]
async fn chat_completions_with_model_returns_openai_schema() {
    let model_path =
        std::env::var("BITNET_GGUF").expect("BITNET_GGUF must be set for this test");
    let tokenizer_path = std::env::var("BITNET_TOKENIZER").ok();

    let mut config = ServerConfig::default();
    config.server.default_model_path = Some(model_path);
    config.server.default_tokenizer_path = tokenizer_path;

    let server = BitNetServer::new(config).await.expect("server init failed");
    let app = server.create_app();

    let body = json!({
        "model": "bitnet",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"}
        ],
        "max_tokens": 8,
        "temperature": 0.0
    });

    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(
        response.status(),
        StatusCode::OK,
        "chat/completions should return 200 with model loaded"
    );

    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&bytes).expect("response should be valid JSON");

    assert!(json.get("id").is_some(), "response must have 'id'");
    assert!(json.get("object").is_some(), "response must have 'object'");
    assert!(json.get("choices").is_some(), "response must have 'choices'");
    assert!(json.get("usage").is_some(), "response must have 'usage'");

    let choices = json["choices"].as_array().expect("choices must be an array");
    assert!(!choices.is_empty(), "choices array must not be empty");

    let message = choices[0].get("message").expect("choice must have 'message'");
    assert_eq!(message["role"], "assistant", "response role must be 'assistant'");
    assert!(message.get("content").is_some(), "message must have 'content'");
}
