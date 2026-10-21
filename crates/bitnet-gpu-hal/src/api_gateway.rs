//! API gateway for OpenAI-compatible inference endpoints.
//!
//! Provides endpoint routing, API key authentication, CORS handling,
//! request/response transformation, and middleware chain execution
//! for serving `BitNet` models via an OpenAI-compatible HTTP API.

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

mod auth;
mod config;
mod middleware;
mod protocol;
mod rate_limiter;
mod transformer;
mod util;

pub use auth::{ApiKeyEntry, AuthProvider};
pub use config::{CorsConfig, GatewayConfig};
pub use middleware::{
    AuthMiddleware, CorsMiddleware, LoggingMiddleware, MaxBodySizeMiddleware, Middleware,
    MiddlewareResult,
};
pub use protocol::{ApiEndpoint, ApiRequest, ApiResponse, ApiVersion, HttpMethod, StatusCode};
pub use rate_limiter::RateLimiter;
pub use transformer::{InternalInferenceRequest, InternalInferenceResponse, RequestTransformer};

use util::strip_bearer;
#[cfg(test)]
use util::{generate_api_key, generate_request_id};

// ── API gateway ─────────────────────────────────────────────────────────────

/// The main API gateway that ties together endpoint routing, middleware,
/// authentication, and request handling.
pub struct ApiGateway {
    pub config: GatewayConfig,
    endpoints: HashMap<String, ApiEndpoint>,
    middlewares: Vec<Box<dyn Middleware>>,
    transformer: RequestTransformer,
    rate_limiter: RateLimiter,
}

impl fmt::Debug for ApiGateway {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ApiGateway")
            .field("config", &self.config)
            .field("endpoints", &self.endpoints.len())
            .field("middlewares", &self.middlewares.len())
            .finish_non_exhaustive()
    }
}

impl ApiGateway {
    /// Create a gateway with the given config.
    pub fn new(config: GatewayConfig) -> Self {
        Self {
            config,
            endpoints: HashMap::new(),
            middlewares: Vec::new(),
            transformer: RequestTransformer::default(),
            rate_limiter: RateLimiter::new(100, Duration::from_mins(1)),
        }
    }

    /// Create a gateway with default config.
    pub fn with_defaults() -> Self {
        Self::new(GatewayConfig::default())
    }

    /// Override the request transformer.
    pub fn set_transformer(&mut self, t: RequestTransformer) {
        self.transformer = t;
    }

    /// Override the rate limiter.
    pub fn set_rate_limiter(&mut self, rl: RateLimiter) {
        self.rate_limiter = rl;
    }

    /// Register an API endpoint.
    pub fn register_endpoint(&mut self, ep: ApiEndpoint) {
        let key = ep.route_key();
        log::info!("registering endpoint: {key}");
        self.endpoints.insert(key, ep);
    }

    /// Push a middleware to the end of the chain.
    pub fn add_middleware(&mut self, mw: Box<dyn Middleware>) {
        log::debug!("adding middleware: {}", mw.name());
        self.middlewares.push(mw);
    }

    /// Number of registered endpoints.
    pub fn endpoint_count(&self) -> usize {
        self.endpoints.len()
    }

    /// Number of registered middlewares.
    pub fn middleware_count(&self) -> usize {
        self.middlewares.len()
    }

    /// List all registered route keys.
    pub fn routes(&self) -> Vec<String> {
        let mut r: Vec<String> = self.endpoints.keys().cloned().collect();
        r.sort();
        r
    }

    /// Look up an endpoint by method + path.
    pub fn find_endpoint(&self, method: HttpMethod, path: &str) -> Option<&ApiEndpoint> {
        let key = format!("{method} {path}");
        self.endpoints.get(&key)
    }

    /// Handle an inbound request through the middleware chain and route it.
    pub fn handle_request(&mut self, mut req: ApiRequest) -> ApiResponse {
        let start = Instant::now();

        // Extract API key from header if not already set.
        if req.api_key.is_none()
            && let Some(v) = req.headers.get(&self.config.api_key_header)
        {
            req.api_key = Some(strip_bearer(v));
        }

        // Run middleware chain.
        for mw in &self.middlewares {
            match mw.process(req, &self.config) {
                MiddlewareResult::Continue(r) => req = r,
                MiddlewareResult::ShortCircuit(resp) => {
                    return Self::finalize(resp, start);
                }
            }
        }

        // Route to endpoint.
        let route_key = format!("{} {}", req.method, req.endpoint);

        // Check if the path exists with a different method → 405.
        let path_exists = self.endpoints.values().any(|ep| ep.path == req.endpoint);

        let ep = match self.endpoints.get(&route_key) {
            Some(ep) => ep.clone(),
            None if path_exists => {
                return Self::finalize(
                    ApiResponse::error(
                        StatusCode::METHOD_NOT_ALLOWED,
                        "method not allowed",
                        &req.request_id,
                    ),
                    start,
                );
            }
            None => {
                return Self::finalize(
                    ApiResponse::error(
                        StatusCode::NOT_FOUND,
                        "endpoint not found",
                        &req.request_id,
                    ),
                    start,
                );
            }
        };

        // Per-endpoint rate limiting.
        if let Some(limit) = ep.rate_limit {
            let key = req.api_key.as_deref().unwrap_or("anonymous").to_string();
            let rl_key = format!("{}:{}", ep.route_key(), key);
            // Temporarily adjust limiter for per-endpoint limit.
            let mut ep_limiter = RateLimiter::new(limit, Duration::from_mins(1));
            if !ep_limiter.check(&rl_key) {
                return Self::finalize(
                    ApiResponse::error(
                        StatusCode::TOO_MANY_REQUESTS,
                        "rate limit exceeded",
                        &req.request_id,
                    ),
                    start,
                );
            }
        }

        // Global rate limiting.
        let global_key = req.api_key.as_deref().unwrap_or("anonymous").to_string();
        if !self.rate_limiter.check(&global_key) {
            return Self::finalize(
                ApiResponse::error(
                    StatusCode::TOO_MANY_REQUESTS,
                    "global rate limit exceeded",
                    &req.request_id,
                ),
                start,
            );
        }

        // Dispatch to handler (stub: echo the handler name).
        let resp_body = format!(
            r#"{{"handler":"{}","request_id":"{}","api_version":"{}"}}"#,
            ep.handler_name, req.request_id, self.config.api_version,
        );

        let mut resp = ApiResponse::ok(resp_body.into_bytes(), &req.request_id);

        // Attach CORS headers if enabled.
        if self.config.enable_cors {
            for (k, v) in self.config.cors.to_headers() {
                resp.headers.insert(k, v);
            }
        }

        Self::finalize(resp, start)
    }

    /// Compute latency and return the response.
    fn finalize(resp: ApiResponse, start: Instant) -> ApiResponse {
        #[allow(clippy::cast_possible_truncation)]
        let elapsed = start.elapsed().as_millis() as u64;
        resp.with_latency(elapsed)
    }

    /// Convenience: transform an OpenAI-format body into an internal request.
    pub fn transform_request(&self, body: &[u8]) -> Result<InternalInferenceRequest, String> {
        self.transformer.to_internal(body)
    }

    /// Convenience: transform an internal response to `OpenAI` format.
    pub fn transform_response(&self, resp: &InternalInferenceResponse) -> Vec<u8> {
        self.transformer.to_openai_response(resp)
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ─────────────────────────────────────────────────────────

    fn test_config() -> GatewayConfig {
        GatewayConfig {
            max_request_size_bytes: 1024,
            timeout_ms: 5_000,
            ..GatewayConfig::default()
        }
    }

    fn test_gateway() -> ApiGateway {
        let mut gw = ApiGateway::new(test_config());
        gw.register_endpoint(ApiEndpoint {
            path: "/chat/completions".to_string(),
            method: HttpMethod::Post,
            handler_name: "chat_completions".to_string(),
            auth_required: true,
            rate_limit: None,
        });
        gw.register_endpoint(ApiEndpoint {
            path: "/models".to_string(),
            method: HttpMethod::Get,
            handler_name: "list_models".to_string(),
            auth_required: false,
            rate_limit: None,
        });
        gw
    }

    fn make_request(endpoint: &str, method: HttpMethod) -> ApiRequest {
        ApiRequest::new(endpoint, method, HashMap::new(), Vec::new())
    }

    fn make_request_with_body(endpoint: &str, method: HttpMethod, body: &[u8]) -> ApiRequest {
        ApiRequest::new(endpoint, method, HashMap::new(), body.to_vec())
    }

    fn make_auth_request(endpoint: &str, method: HttpMethod, api_key: &str) -> ApiRequest {
        let mut headers = HashMap::new();
        headers.insert("Authorization".to_string(), api_key.to_string());
        ApiRequest::new(endpoint, method, headers, Vec::new())
    }

    // ── GatewayConfig tests ────────────────────────────────────────────

    #[test]
    fn test_gateway_config_defaults() {
        let c = GatewayConfig::default();
        assert_eq!(c.api_version, ApiVersion::V1);
        assert!(c.enable_cors);
        assert_eq!(c.max_request_size_bytes, 4 * 1024 * 1024);
        assert_eq!(c.timeout_ms, 30_000);
        assert_eq!(c.api_key_header, "Authorization");
    }

    #[test]
    fn test_gateway_config_custom() {
        let c = GatewayConfig {
            api_version: ApiVersion::V2,
            enable_cors: false,
            max_request_size_bytes: 512,
            timeout_ms: 1_000,
            api_key_header: "X-Api-Key".to_string(),
            cors: CorsConfig::default(),
        };
        assert_eq!(c.api_version, ApiVersion::V2);
        assert!(!c.enable_cors);
        assert_eq!(c.max_request_size_bytes, 512);
    }

    // ── ApiVersion tests ───────────────────────────────────────────────

    #[test]
    fn test_api_version_display() {
        assert_eq!(ApiVersion::V1.to_string(), "v1");
        assert_eq!(ApiVersion::V2.to_string(), "v2");
    }

    #[test]
    fn test_api_version_prefix() {
        assert_eq!(ApiVersion::V1.prefix(), "/v1");
        assert_eq!(ApiVersion::V2.prefix(), "/v2");
    }

    #[test]
    fn test_api_version_parse() {
        assert_eq!(ApiVersion::from_str_prefix("v1"), Some(ApiVersion::V1));
        assert_eq!(ApiVersion::from_str_prefix("V2"), Some(ApiVersion::V2));
        assert_eq!(ApiVersion::from_str_prefix("v3"), None);
        assert_eq!(ApiVersion::from_str_prefix(""), None);
    }

    #[test]
    fn test_api_version_roundtrip() {
        for v in [ApiVersion::V1, ApiVersion::V2] {
            let s = v.to_string();
            assert_eq!(ApiVersion::from_str_prefix(&s), Some(v));
        }
    }

    // ── HttpMethod tests ───────────────────────────────────────────────

    #[test]
    fn test_http_method_display() {
        assert_eq!(HttpMethod::Get.to_string(), "GET");
        assert_eq!(HttpMethod::Post.to_string(), "POST");
        assert_eq!(HttpMethod::Delete.to_string(), "DELETE");
        assert_eq!(HttpMethod::Patch.to_string(), "PATCH");
    }

    #[test]
    fn test_http_method_parse() {
        assert_eq!(HttpMethod::from_str_upper("GET"), Some(HttpMethod::Get));
        assert_eq!(HttpMethod::from_str_upper("post"), Some(HttpMethod::Post));
        assert_eq!(HttpMethod::from_str_upper("UNKNOWN"), None);
    }

    #[test]
    fn test_http_method_parse_all_variants() {
        let cases = [
            ("GET", HttpMethod::Get),
            ("POST", HttpMethod::Post),
            ("PUT", HttpMethod::Put),
            ("DELETE", HttpMethod::Delete),
            ("PATCH", HttpMethod::Patch),
            ("OPTIONS", HttpMethod::Options),
        ];
        for (s, expected) in cases {
            assert_eq!(HttpMethod::from_str_upper(s), Some(expected));
        }
    }

    // ── StatusCode tests ───────────────────────────────────────────────

    #[test]
    fn test_status_code_categories() {
        assert!(StatusCode::OK.is_success());
        assert!(StatusCode::CREATED.is_success());
        assert!(!StatusCode::OK.is_client_error());
        assert!(StatusCode::BAD_REQUEST.is_client_error());
        assert!(StatusCode::NOT_FOUND.is_client_error());
        assert!(!StatusCode::BAD_REQUEST.is_server_error());
        assert!(StatusCode::INTERNAL_SERVER_ERROR.is_server_error());
        assert!(StatusCode::SERVICE_UNAVAILABLE.is_server_error());
    }

    #[test]
    fn test_status_code_reason_phrase() {
        assert_eq!(StatusCode::OK.reason_phrase(), "OK");
        assert_eq!(StatusCode::NOT_FOUND.reason_phrase(), "Not Found");
        assert_eq!(StatusCode::UNAUTHORIZED.reason_phrase(), "Unauthorized");
        assert_eq!(StatusCode(999).reason_phrase(), "Unknown");
    }

    #[test]
    fn test_status_code_display() {
        assert_eq!(StatusCode::OK.to_string(), "200 OK");
        assert_eq!(StatusCode::NOT_FOUND.to_string(), "404 Not Found");
    }

    #[test]
    fn test_status_code_no_content() {
        assert!(StatusCode::NO_CONTENT.is_success());
        assert_eq!(StatusCode::NO_CONTENT.0, 204);
    }

    #[test]
    fn test_status_code_equality() {
        assert_eq!(StatusCode(200), StatusCode::OK);
        assert_ne!(StatusCode(200), StatusCode::CREATED);
    }

    // ── ApiEndpoint tests ──────────────────────────────────────────────

    #[test]
    fn test_endpoint_route_key() {
        let ep = ApiEndpoint {
            path: "/chat/completions".to_string(),
            method: HttpMethod::Post,
            handler_name: "chat".to_string(),
            auth_required: true,
            rate_limit: None,
        };
        assert_eq!(ep.route_key(), "POST /chat/completions");
    }

    #[test]
    fn test_endpoint_with_rate_limit() {
        let ep = ApiEndpoint {
            path: "/models".to_string(),
            method: HttpMethod::Get,
            handler_name: "models".to_string(),
            auth_required: false,
            rate_limit: Some(10),
        };
        assert_eq!(ep.rate_limit, Some(10));
        assert!(!ep.auth_required);
    }

    // ── ApiRequest tests ───────────────────────────────────────────────

    #[test]
    fn test_request_new_sets_id_and_timestamp() {
        let req = make_request("/test", HttpMethod::Get);
        assert!(req.request_id.starts_with("req-"));
        assert!(req.timestamp > 0);
    }

    #[test]
    fn test_request_body_len() {
        let req = make_request_with_body("/test", HttpMethod::Post, b"hello");
        assert_eq!(req.body_len(), 5);
    }

    #[test]
    fn test_request_empty_body() {
        let req = make_request("/test", HttpMethod::Get);
        assert_eq!(req.body_len(), 0);
    }

    #[test]
    fn test_request_preserves_headers() {
        let mut headers = HashMap::new();
        headers.insert("Content-Type".to_string(), "application/json".to_string());
        headers.insert("X-Custom".to_string(), "value".to_string());
        let req = ApiRequest::new("/test", HttpMethod::Post, headers, Vec::new());
        assert_eq!(req.headers.get("Content-Type").unwrap(), "application/json");
        assert_eq!(req.headers.get("X-Custom").unwrap(), "value");
    }

    // ── ApiResponse tests ──────────────────────────────────────────────

    #[test]
    fn test_response_ok() {
        let resp = ApiResponse::ok(b"body".to_vec(), "req-1");
        assert_eq!(resp.status_code, StatusCode::OK);
        assert_eq!(resp.body, b"body");
        assert_eq!(resp.request_id, "req-1");
    }

    #[test]
    fn test_response_error() {
        let resp = ApiResponse::error(StatusCode::NOT_FOUND, "not found", "req-2");
        assert_eq!(resp.status_code, StatusCode::NOT_FOUND);
        let body = String::from_utf8(resp.body).unwrap();
        assert!(body.contains("not found"));
        assert!(body.contains("404"));
    }

    #[test]
    fn test_response_with_latency() {
        let resp = ApiResponse::ok(Vec::new(), "req-3").with_latency(42);
        assert_eq!(resp.latency_ms, 42);
    }

    #[test]
    fn test_response_with_header() {
        let resp = ApiResponse::ok(Vec::new(), "req-4").with_header("X-Foo", "bar");
        assert_eq!(resp.headers.get("X-Foo").unwrap(), "bar");
    }

    #[test]
    fn test_response_error_body_is_json() {
        let resp = ApiResponse::error(StatusCode::BAD_REQUEST, "oops", "req-5");
        let body = String::from_utf8(resp.body).unwrap();
        assert!(body.starts_with('{'));
        assert!(body.contains("\"error\""));
        assert!(body.contains("\"message\""));
    }

    // ── CorsConfig tests ───────────────────────────────────────────────

    #[test]
    fn test_cors_default() {
        let c = CorsConfig::default();
        assert_eq!(c.allowed_origins, vec!["*"]);
        assert_eq!(c.max_age, 86_400);
    }

    #[test]
    fn test_cors_to_headers() {
        let c = CorsConfig::default();
        let h = c.to_headers();
        assert_eq!(h.get("Access-Control-Allow-Origin").unwrap(), "*");
        assert!(h.contains_key("Access-Control-Allow-Methods"));
        assert!(h.contains_key("Access-Control-Allow-Headers"));
        assert_eq!(h.get("Access-Control-Max-Age").unwrap(), "86400");
    }

    #[test]
    fn test_cors_custom_origins() {
        let c = CorsConfig {
            allowed_origins: vec![
                "https://example.com".to_string(),
                "https://app.example.com".to_string(),
            ],
            ..CorsConfig::default()
        };
        let h = c.to_headers();
        let origin = h.get("Access-Control-Allow-Origin").unwrap();
        assert!(origin.contains("https://example.com"));
        assert!(origin.contains("https://app.example.com"));
    }

    #[test]
    fn test_cors_custom_methods() {
        let c = CorsConfig {
            allowed_methods: vec![HttpMethod::Get, HttpMethod::Post, HttpMethod::Delete],
            ..CorsConfig::default()
        };
        let h = c.to_headers();
        let methods = h.get("Access-Control-Allow-Methods").unwrap();
        assert!(methods.contains("GET"));
        assert!(methods.contains("POST"));
        assert!(methods.contains("DELETE"));
    }

    // ── AuthProvider tests ─────────────────────────────────────────────

    #[test]
    fn test_auth_create_and_validate() {
        let mut auth = AuthProvider::new();
        let key = auth.create_key("test-user");
        assert!(auth.validate(&key));
    }

    #[test]
    fn test_auth_invalid_key() {
        let auth = AuthProvider::new();
        assert!(!auth.validate("nonexistent"));
    }

    #[test]
    fn test_auth_revoke() {
        let mut auth = AuthProvider::new();
        let key = auth.create_key("user1");
        assert!(auth.validate(&key));
        assert!(auth.revoke(&key));
        assert!(!auth.validate(&key));
    }

    #[test]
    fn test_auth_revoke_idempotent() {
        let mut auth = AuthProvider::new();
        let key = auth.create_key("user1");
        assert!(auth.revoke(&key));
        assert!(!auth.revoke(&key)); // already revoked
    }

    #[test]
    fn test_auth_revoke_nonexistent() {
        let mut auth = AuthProvider::new();
        assert!(!auth.revoke("ghost-key"));
    }

    #[test]
    fn test_auth_list_keys() {
        let mut auth = AuthProvider::new();
        auth.create_key("alice");
        auth.create_key("bob");
        let names = auth.list_keys();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"alice"));
        assert!(names.contains(&"bob"));
    }

    #[test]
    fn test_auth_list_excludes_revoked() {
        let mut auth = AuthProvider::new();
        let k1 = auth.create_key("alice");
        auth.create_key("bob");
        auth.revoke(&k1);
        let names = auth.list_keys();
        assert_eq!(names.len(), 1);
        assert!(names.contains(&"bob"));
    }

    #[test]
    fn test_auth_insert_specific_key() {
        let mut auth = AuthProvider::new();
        auth.insert_key("my-fixed-key", "test");
        assert!(auth.validate("my-fixed-key"));
    }

    // ── RequestTransformer tests ───────────────────────────────────────

    #[test]
    fn test_transform_minimal_request() {
        let t = RequestTransformer::default();
        let body = br#"{"prompt":"Hello"}"#;
        let req = t.to_internal(body).unwrap();
        assert_eq!(req.prompt, "Hello");
        assert_eq!(req.model, "bitnet-b1.58-2B-4T");
        assert_eq!(req.max_tokens, 256);
        assert!((req.temperature - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_transform_full_request() {
        let t = RequestTransformer::default();
        let body = br#"{"model":"custom","prompt":"Hi","max_tokens":64,"temperature":0.5,"top_p":0.9,"stream":true}"#;
        let req = t.to_internal(body).unwrap();
        assert_eq!(req.model, "custom");
        assert_eq!(req.prompt, "Hi");
        assert_eq!(req.max_tokens, 64);
        assert!((req.temperature - 0.5).abs() < f32::EPSILON);
        assert!((req.top_p - 0.9).abs() < f32::EPSILON);
        assert!(req.stream);
    }

    #[test]
    fn test_transform_messages_field() {
        let t = RequestTransformer::default();
        let body = br#"{"messages":"What is 2+2?"}"#;
        let req = t.to_internal(body).unwrap();
        assert_eq!(req.prompt, "What is 2+2?");
    }

    #[test]
    fn test_transform_missing_prompt() {
        let t = RequestTransformer::default();
        let body = br#"{"model":"test"}"#;
        assert!(t.to_internal(body).is_err());
    }

    #[test]
    fn test_transform_invalid_json() {
        let t = RequestTransformer::default();
        assert!(t.to_internal(b"not json").is_err());
    }

    #[test]
    fn test_transform_empty_body() {
        let t = RequestTransformer::default();
        assert!(t.to_internal(b"").is_err());
    }

    #[test]
    fn test_transform_max_tokens_capped() {
        let t = RequestTransformer { max_tokens_limit: 100, ..RequestTransformer::default() };
        let body = br#"{"prompt":"Hi","max_tokens":9999}"#;
        let req = t.to_internal(body).unwrap();
        assert_eq!(req.max_tokens, 100);
    }

    #[test]
    fn test_transform_response_roundtrip() {
        let t = RequestTransformer::default();
        let internal = InternalInferenceResponse {
            text: "Hello world".to_string(),
            tokens_used: 5,
            finish_reason: "stop".to_string(),
            model: "bitnet-b1.58-2B-4T".to_string(),
        };
        let body = t.to_openai_response(&internal);
        let s = String::from_utf8(body).unwrap();
        assert!(s.contains("chat.completion"));
        assert!(s.contains("Hello world"));
        assert!(s.contains("stop"));
        assert!(s.contains("\"completion_tokens\":5"));
    }

    #[test]
    fn test_transform_response_escapes_quotes() {
        let t = RequestTransformer::default();
        let internal = InternalInferenceResponse {
            text: r#"He said "hi""#.to_string(),
            tokens_used: 3,
            finish_reason: "stop".to_string(),
            model: "test".to_string(),
        };
        let body = t.to_openai_response(&internal);
        let s = String::from_utf8(body).unwrap();
        assert!(s.contains(r#"He said \"hi\""#));
    }

    #[test]
    fn test_transform_idempotent_internal_request() {
        let t = RequestTransformer::default();
        let body = br#"{"prompt":"Test","max_tokens":32,"temperature":0.7}"#;
        let r1 = t.to_internal(body).unwrap();
        let r2 = t.to_internal(body).unwrap();
        assert_eq!(r1, r2);
    }

    // ── Middleware tests ────────────────────────────────────────────────

    #[test]
    fn test_auth_middleware_valid_key() {
        let mut auth = AuthProvider::new();
        auth.insert_key("valid-key", "test");
        let mw = AuthMiddleware::new(auth);
        let config = test_config();
        let req = make_auth_request("/test", HttpMethod::Get, "valid-key");
        match mw.process(req, &config) {
            MiddlewareResult::Continue(_) => {} // expected
            MiddlewareResult::ShortCircuit(r) => {
                panic!("expected Continue, got ShortCircuit with {}", r.status_code)
            }
        }
    }

    #[test]
    fn test_auth_middleware_invalid_key() {
        let auth = AuthProvider::new();
        let mw = AuthMiddleware::new(auth);
        let config = test_config();
        let req = make_auth_request("/test", HttpMethod::Get, "bad-key");
        match mw.process(req, &config) {
            MiddlewareResult::ShortCircuit(r) => {
                assert_eq!(r.status_code, StatusCode::UNAUTHORIZED);
            }
            MiddlewareResult::Continue(_) => panic!("expected ShortCircuit"),
        }
    }

    #[test]
    fn test_auth_middleware_missing_key() {
        let auth = AuthProvider::new();
        let mw = AuthMiddleware::new(auth);
        let config = test_config();
        let req = make_request("/test", HttpMethod::Get);
        match mw.process(req, &config) {
            MiddlewareResult::ShortCircuit(r) => {
                assert_eq!(r.status_code, StatusCode::UNAUTHORIZED);
                let body = String::from_utf8(r.body).unwrap();
                assert!(body.contains("missing"));
            }
            MiddlewareResult::Continue(_) => panic!("expected ShortCircuit"),
        }
    }

    #[test]
    fn test_max_body_middleware_under_limit() {
        let mw = MaxBodySizeMiddleware;
        let config = test_config();
        let req = make_request_with_body("/test", HttpMethod::Post, &[0u8; 512]);
        match mw.process(req, &config) {
            MiddlewareResult::Continue(_) => {}
            MiddlewareResult::ShortCircuit(r) => {
                panic!("expected Continue, got {}", r.status_code)
            }
        }
    }

    #[test]
    fn test_max_body_middleware_over_limit() {
        let mw = MaxBodySizeMiddleware;
        let config = test_config(); // limit = 1024
        let req = make_request_with_body("/test", HttpMethod::Post, &[0u8; 2048]);
        match mw.process(req, &config) {
            MiddlewareResult::ShortCircuit(r) => {
                assert_eq!(r.status_code, StatusCode::PAYLOAD_TOO_LARGE);
            }
            MiddlewareResult::Continue(_) => panic!("expected ShortCircuit"),
        }
    }

    #[test]
    fn test_max_body_middleware_exact_limit() {
        let mw = MaxBodySizeMiddleware;
        let config = test_config(); // limit = 1024
        let req = make_request_with_body("/test", HttpMethod::Post, &[0u8; 1024]);
        match mw.process(req, &config) {
            MiddlewareResult::Continue(_) => {}
            MiddlewareResult::ShortCircuit(_) => panic!("expected Continue at exact limit"),
        }
    }

    #[test]
    fn test_cors_middleware_options_request() {
        let mw = CorsMiddleware;
        let config = test_config();
        let req = make_request("/test", HttpMethod::Options);
        match mw.process(req, &config) {
            MiddlewareResult::ShortCircuit(r) => {
                assert_eq!(r.status_code, StatusCode::NO_CONTENT);
                assert!(r.headers.contains_key("Access-Control-Allow-Origin"));
            }
            MiddlewareResult::Continue(_) => panic!("expected ShortCircuit for OPTIONS"),
        }
    }

    #[test]
    fn test_cors_middleware_non_options_passes_through() {
        let mw = CorsMiddleware;
        let config = test_config();
        let req = make_request("/test", HttpMethod::Get);
        match mw.process(req, &config) {
            MiddlewareResult::Continue(_) => {}
            MiddlewareResult::ShortCircuit(_) => panic!("expected Continue for GET"),
        }
    }

    #[test]
    fn test_cors_middleware_disabled() {
        let mw = CorsMiddleware;
        let mut config = test_config();
        config.enable_cors = false;
        let req = make_request("/test", HttpMethod::Options);
        match mw.process(req, &config) {
            MiddlewareResult::Continue(_) => {} // passes through when CORS disabled
            MiddlewareResult::ShortCircuit(_) => {
                panic!("expected Continue when CORS disabled")
            }
        }
    }

    #[test]
    fn test_logging_middleware_records_ids() {
        let mw = LoggingMiddleware::default();
        let config = test_config();
        let req = make_request("/test", HttpMethod::Get);
        let id = req.request_id.clone();
        let _ = mw.process(req, &config);
        let seen = mw.seen.lock().unwrap();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0], id);
        drop(seen);
    }

    // ── RateLimiter tests ──────────────────────────────────────────────

    #[test]
    fn test_rate_limiter_allows_under_limit() {
        let mut rl = RateLimiter::new(3, Duration::from_mins(1));
        assert!(rl.check("key1"));
        assert!(rl.check("key1"));
        assert!(rl.check("key1"));
    }

    #[test]
    fn test_rate_limiter_blocks_over_limit() {
        let mut rl = RateLimiter::new(2, Duration::from_mins(1));
        assert!(rl.check("key1"));
        assert!(rl.check("key1"));
        assert!(!rl.check("key1")); // 3rd should fail
    }

    #[test]
    fn test_rate_limiter_separate_keys() {
        let mut rl = RateLimiter::new(1, Duration::from_mins(1));
        assert!(rl.check("key1"));
        assert!(rl.check("key2"));
        assert!(!rl.check("key1"));
        assert!(!rl.check("key2"));
    }

    #[test]
    fn test_rate_limiter_reset() {
        let mut rl = RateLimiter::new(1, Duration::from_mins(1));
        assert!(rl.check("key1"));
        assert!(!rl.check("key1"));
        rl.reset();
        assert!(rl.check("key1"));
    }

    // ── ApiGateway endpoint registration ───────────────────────────────

    #[test]
    fn test_gateway_register_endpoint() {
        let mut gw = ApiGateway::with_defaults();
        assert_eq!(gw.endpoint_count(), 0);
        gw.register_endpoint(ApiEndpoint {
            path: "/models".to_string(),
            method: HttpMethod::Get,
            handler_name: "list_models".to_string(),
            auth_required: false,
            rate_limit: None,
        });
        assert_eq!(gw.endpoint_count(), 1);
    }

    #[test]
    fn test_gateway_multiple_endpoints() {
        let gw = test_gateway();
        assert_eq!(gw.endpoint_count(), 2);
    }

    #[test]
    fn test_gateway_routes() {
        let gw = test_gateway();
        let routes = gw.routes();
        assert!(routes.contains(&"GET /models".to_string()));
        assert!(routes.contains(&"POST /chat/completions".to_string()));
    }

    #[test]
    fn test_gateway_find_endpoint() {
        let gw = test_gateway();
        let ep = gw.find_endpoint(HttpMethod::Get, "/models");
        assert!(ep.is_some());
        assert_eq!(ep.unwrap().handler_name, "list_models");
    }

    #[test]
    fn test_gateway_find_endpoint_missing() {
        let gw = test_gateway();
        assert!(gw.find_endpoint(HttpMethod::Get, "/nonexistent").is_none());
    }

    #[test]
    fn test_gateway_overwrite_endpoint() {
        let mut gw = ApiGateway::with_defaults();
        gw.register_endpoint(ApiEndpoint {
            path: "/models".to_string(),
            method: HttpMethod::Get,
            handler_name: "v1_models".to_string(),
            auth_required: false,
            rate_limit: None,
        });
        gw.register_endpoint(ApiEndpoint {
            path: "/models".to_string(),
            method: HttpMethod::Get,
            handler_name: "v2_models".to_string(),
            auth_required: false,
            rate_limit: None,
        });
        assert_eq!(gw.endpoint_count(), 1);
        assert_eq!(gw.find_endpoint(HttpMethod::Get, "/models").unwrap().handler_name, "v2_models");
    }

    // ── ApiGateway request handling ────────────────────────────────────

    #[test]
    fn test_gateway_handle_known_endpoint() {
        let mut gw = test_gateway();
        let req = make_request("/models", HttpMethod::Get);
        let resp = gw.handle_request(req);
        assert_eq!(resp.status_code, StatusCode::OK);
        let body = String::from_utf8(resp.body).unwrap();
        assert!(body.contains("list_models"));
    }

    #[test]
    fn test_gateway_unknown_endpoint_404() {
        let mut gw = test_gateway();
        let req = make_request("/nonexistent", HttpMethod::Get);
        let resp = gw.handle_request(req);
        assert_eq!(resp.status_code, StatusCode::NOT_FOUND);
    }

    #[test]
    fn test_gateway_wrong_method_405() {
        let mut gw = test_gateway();
        // /models only accepts GET, try DELETE.
        let req = make_request("/models", HttpMethod::Delete);
        let resp = gw.handle_request(req);
        assert_eq!(resp.status_code, StatusCode::METHOD_NOT_ALLOWED);
    }

    #[test]
    fn test_gateway_cors_headers_on_response() {
        let mut gw = test_gateway();
        let req = make_request("/models", HttpMethod::Get);
        let resp = gw.handle_request(req);
        assert!(resp.headers.contains_key("Access-Control-Allow-Origin"));
    }

    #[test]
    fn test_gateway_cors_disabled() {
        let mut config = test_config();
        config.enable_cors = false;
        let mut gw = ApiGateway::new(config);
        gw.register_endpoint(ApiEndpoint {
            path: "/models".to_string(),
            method: HttpMethod::Get,
            handler_name: "list_models".to_string(),
            auth_required: false,
            rate_limit: None,
        });
        let req = make_request("/models", HttpMethod::Get);
        let resp = gw.handle_request(req);
        assert!(!resp.headers.contains_key("Access-Control-Allow-Origin"));
    }

    #[test]
    fn test_gateway_latency_recorded() {
        let mut gw = test_gateway();
        let req = make_request("/models", HttpMethod::Get);
        let resp = gw.handle_request(req);
        // Latency is non-negative (could be 0 if very fast).
        assert!(resp.latency_ms < 10_000);
    }

    #[test]
    fn test_gateway_request_id_preserved() {
        let mut gw = test_gateway();
        let req = make_request("/models", HttpMethod::Get);
        let id = req.request_id.clone();
        let resp = gw.handle_request(req);
        assert_eq!(resp.request_id, id);
    }

    #[test]
    fn test_gateway_api_version_in_response() {
        let mut gw = test_gateway();
        let req = make_request("/models", HttpMethod::Get);
        let resp = gw.handle_request(req);
        let body = String::from_utf8(resp.body).unwrap();
        assert!(body.contains("v1"));
    }

    // ── Middleware chain in gateway ─────────────────────────────────────

    #[test]
    fn test_gateway_middleware_chain_executes_in_order() {
        let mut gw = test_gateway();
        gw.add_middleware(Box::new(MaxBodySizeMiddleware));
        gw.add_middleware(Box::new(CorsMiddleware));
        assert_eq!(gw.middleware_count(), 2);
    }

    #[test]
    fn test_gateway_middleware_short_circuits() {
        let mut gw = test_gateway();
        gw.add_middleware(Box::new(MaxBodySizeMiddleware));
        // config limit = 1024
        let req = make_request_with_body("/models", HttpMethod::Get, &[0u8; 2048]);
        let resp = gw.handle_request(req);
        assert_eq!(resp.status_code, StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[test]
    fn test_gateway_auth_middleware_blocks_unauthenticated() {
        let mut gw = test_gateway();
        let auth = AuthProvider::new(); // no keys
        gw.add_middleware(Box::new(AuthMiddleware::new(auth)));
        let req = make_request("/models", HttpMethod::Get);
        let resp = gw.handle_request(req);
        assert_eq!(resp.status_code, StatusCode::UNAUTHORIZED);
    }

    #[test]
    fn test_gateway_auth_middleware_allows_authenticated() {
        let mut gw = test_gateway();
        let mut auth = AuthProvider::new();
        auth.insert_key("test-key", "tester");
        gw.add_middleware(Box::new(AuthMiddleware::new(auth)));
        let req = make_auth_request("/models", HttpMethod::Get, "test-key");
        let resp = gw.handle_request(req);
        assert_eq!(resp.status_code, StatusCode::OK);
    }

    #[test]
    fn test_gateway_bearer_token_extraction() {
        let mut gw = test_gateway();
        let mut auth = AuthProvider::new();
        auth.insert_key("my-secret", "user");
        gw.add_middleware(Box::new(AuthMiddleware::new(auth)));
        let req = make_auth_request("/models", HttpMethod::Get, "Bearer my-secret");
        let resp = gw.handle_request(req);
        assert_eq!(resp.status_code, StatusCode::OK);
    }

    // ── Gateway transform pass-through ─────────────────────────────────

    #[test]
    fn test_gateway_transform_request() {
        let gw = test_gateway();
        let body = br#"{"prompt":"Hello","max_tokens":10}"#;
        let req = gw.transform_request(body).unwrap();
        assert_eq!(req.prompt, "Hello");
        assert_eq!(req.max_tokens, 10);
    }

    #[test]
    fn test_gateway_transform_response() {
        let gw = test_gateway();
        let resp = InternalInferenceResponse {
            text: "world".to_string(),
            tokens_used: 1,
            finish_reason: "stop".to_string(),
            model: "test".to_string(),
        };
        let body = gw.transform_response(&resp);
        let s = String::from_utf8(body).unwrap();
        assert!(s.contains("world"));
    }

    // ── Helpers tests ──────────────────────────────────────────────────

    #[test]
    fn test_strip_bearer() {
        assert_eq!(strip_bearer("Bearer token123"), "token123");
        assert_eq!(strip_bearer("raw-key"), "raw-key");
        assert_eq!(strip_bearer(""), "");
    }

    #[test]
    fn test_generate_request_id_format() {
        let id = generate_request_id();
        assert!(id.starts_with("req-"));
        // 32 hex chars (UUID v4 simple form) + "req-" prefix
        assert_eq!(id.len(), 4 + 32);
        assert!(id.chars().skip(4).all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_generate_api_key_format() {
        let key = generate_api_key();
        assert!(key.starts_with("sk-bitnet-"));
        assert_eq!(key.len(), "sk-bitnet-".len() + 32);
        assert!(key.chars().skip("sk-bitnet-".len()).all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_generate_ids_are_unique() {
        // Two ids generated back-to-back must not collide. Under the previous
        // timestamp-based implementation this was technically possible if both
        // calls landed in the same nanosecond.
        let mut seen = std::collections::HashSet::new();
        for _ in 0..256 {
            assert!(seen.insert(generate_request_id()));
            assert!(seen.insert(generate_api_key()));
        }
    }

    // ── Debug impls ────────────────────────────────────────────────────

    #[test]
    fn test_gateway_debug() {
        let gw = test_gateway();
        let dbg = format!("{gw:?}");
        assert!(dbg.contains("ApiGateway"));
        assert!(dbg.contains("endpoints"));
    }
}
