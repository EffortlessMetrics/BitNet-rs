#[cfg(test)]
mod tests {
    use axum::{
        Router,
        http::{Method, Request, StatusCode},
        routing::get,
    };
    use bitnet_server::security::{self, SecurityConfig};
    use tower::ServiceExt; // for oneshot

    #[tokio::test]
    async fn test_cors_configuration() {
        // Test 1: Allow all (default)
        let config =
            SecurityConfig { allowed_origins: vec!["*".to_string()], ..Default::default() };

        let cors = security::configure_cors(&config);

        let app = Router::new().route("/", get(|| async { "hello" })).layer(cors);

        let req = Request::builder()
            .method(Method::OPTIONS)
            .uri("/")
            .header("Origin", "http://any.com")
            .header("Access-Control-Request-Method", "GET")
            .body(axum::body::Body::empty())
            .unwrap();

        let response = app.clone().oneshot(req).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        // With AllowOrigin returning true, it typically reflects the origin
        let allow_origin =
            response.headers().get("access-control-allow-origin").unwrap().to_str().unwrap();
        assert_eq!(allow_origin, "http://any.com");

        // Test 2: Specific origin allowed
        let config = SecurityConfig {
            allowed_origins: vec!["http://trusted.com".to_string()],
            ..Default::default()
        };

        let cors = security::configure_cors(&config);
        let app = Router::new().route("/", get(|| async { "hello" })).layer(cors);

        // Allowed origin
        let req = Request::builder()
            .method(Method::OPTIONS)
            .uri("/")
            .header("Origin", "http://trusted.com")
            .header("Access-Control-Request-Method", "GET")
            .body(axum::body::Body::empty())
            .unwrap();

        let response = app.clone().oneshot(req).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get("access-control-allow-origin").unwrap(),
            "http://trusted.com"
        );

        // Test 3: Blocked origin
        let req = Request::builder()
            .method(Method::OPTIONS)
            .uri("/")
            .header("Origin", "http://evil.com")
            .header("Access-Control-Request-Method", "GET")
            .body(axum::body::Body::empty())
            .unwrap();

        let response = app.clone().oneshot(req).await.unwrap();
        // Should not have CORS headers
        assert!(response.headers().get("access-control-allow-origin").is_none());
    }
}
