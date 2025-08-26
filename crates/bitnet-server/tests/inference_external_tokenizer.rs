use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use tempfile::NamedTempFile;
use tower::ServiceExt; // for `oneshot`

use bitnet_server::{BitNetServer, InferenceRequest, ServerConfig};

#[tokio::test]
async fn inference_with_external_tokenizer_file() {
    // Create a temporary tokenizer file (GGUF)
    let tok_file = NamedTempFile::new().unwrap();
    let path = tok_file.path().with_extension("gguf");
    tok_file.persist(&path).unwrap();
    let mut tok_file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    use std::io::Write;
    tok_file.write_all(b"GGUF").unwrap();

    // Ensure tokenizer file can be loaded
    let _ = bitnet_tokenizers::loader::load_tokenizer(&path).unwrap();

    // Build server using the tokenizer path (no model -> mock mode)
    let mut config = ServerConfig::default();
    config.tokenizer_path = Some(path.to_string_lossy().into_owned());
    let server = BitNetServer::new(config).await.unwrap();
    let app: Router = server.create_app();

    // Prepare request
    let req_body = serde_json::to_vec(&InferenceRequest {
        prompt: "Hello".to_string(),
        max_tokens: Some(4),
        model: None,
        temperature: None,
        top_p: None,
        top_k: None,
        repetition_penalty: None,
    })
    .unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .method(axum::http::Method::POST)
                .uri("/inference")
                .header("content-type", "application/json")
                .body(Body::from(req_body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let resp: bitnet_server::InferenceResponse = serde_json::from_slice(&bytes).unwrap();
    assert!(!resp.text.is_empty());
}
