use axum::{
    Router,
    body::Body,
    http::{Request, StatusCode},
    routing::get,
};
use bitnet_server::security::{SecurityConfig, configure_cors};
use tower::ServiceExt; // for oneshot

#[tokio::test]
async fn test_cors_configuration() {
    // 1. Setup SecurityConfig with restricted origins
    let config = SecurityConfig {
        allowed_origins: vec!["http://trusted.com".to_string()],
        ..SecurityConfig::default()
    };

    // 2. Configure CORS middleware
    let cors_layer = configure_cors(&config);

    // 3. Create a simple router
    let app = Router::new().route("/", get(|| async { "Hello, World!" })).layer(cors_layer);

    // 4. Test valid origin
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/")
                .header("Origin", "http://trusted.com")
                .header("Access-Control-Request-Method", "GET")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get("access-control-allow-origin").unwrap().to_str().unwrap(),
        "http://trusted.com"
    );

    // 5. Test invalid origin
    let response = app
        .oneshot(
            Request::builder()
                .uri("/")
                .header("Origin", "http://evil.com")
                .header("Access-Control-Request-Method", "GET")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    // CORS middleware should NOT add Access-Control-Allow-Origin for invalid origins
    assert!(response.headers().get("access-control-allow-origin").is_none());
}
