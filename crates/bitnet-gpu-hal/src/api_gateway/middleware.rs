use std::fmt;

use super::{ApiRequest, ApiResponse, AuthProvider, GatewayConfig, HttpMethod, StatusCode};

// ── Middleware ───────────────────────────────────────────────────────────────

/// Result type returned by middleware. `Continue` passes the (possibly
/// modified) request to the next middleware; `ShortCircuit` returns a
/// response immediately.
#[derive(Debug)]
pub enum MiddlewareResult {
    Continue(ApiRequest),
    ShortCircuit(ApiResponse),
}

/// Trait for pluggable middleware components.
pub trait Middleware: fmt::Debug + Send + Sync {
    /// Process a request. Return `Continue` to pass through or
    /// `ShortCircuit` to abort with an immediate response.
    fn process(&self, req: ApiRequest, config: &GatewayConfig) -> MiddlewareResult;

    /// Human-readable name for logging.
    fn name(&self) -> &'static str;
}

// ── Built-in middleware: auth ───────────────────────────────────────────────

/// Middleware that checks API key authentication.
#[derive(Debug)]
pub struct AuthMiddleware {
    provider: AuthProvider,
}

impl AuthMiddleware {
    pub const fn new(provider: AuthProvider) -> Self {
        Self { provider }
    }
}

impl Middleware for AuthMiddleware {
    fn process(&self, req: ApiRequest, config: &GatewayConfig) -> MiddlewareResult {
        // Prefer the pre-stripped api_key (set by gateway) over the raw header.
        let key = req.api_key.as_ref().or_else(|| req.headers.get(&config.api_key_header));

        match key {
            Some(k) if self.provider.validate(k) => MiddlewareResult::Continue(req),
            Some(_) => MiddlewareResult::ShortCircuit(ApiResponse::error(
                StatusCode::UNAUTHORIZED,
                "invalid API key",
                &req.request_id,
            )),
            None => MiddlewareResult::ShortCircuit(ApiResponse::error(
                StatusCode::UNAUTHORIZED,
                "missing API key",
                &req.request_id,
            )),
        }
    }

    fn name(&self) -> &'static str {
        "auth"
    }
}

// ── Built-in middleware: max body size ──────────────────────────────────────

#[derive(Debug)]
pub struct MaxBodySizeMiddleware;

impl Middleware for MaxBodySizeMiddleware {
    fn process(&self, req: ApiRequest, config: &GatewayConfig) -> MiddlewareResult {
        if req.body_len() > config.max_request_size_bytes {
            MiddlewareResult::ShortCircuit(ApiResponse::error(
                StatusCode::PAYLOAD_TOO_LARGE,
                format!(
                    "request body {} bytes exceeds limit of {}",
                    req.body_len(),
                    config.max_request_size_bytes
                ),
                &req.request_id,
            ))
        } else {
            MiddlewareResult::Continue(req)
        }
    }

    fn name(&self) -> &'static str {
        "max_body_size"
    }
}

// ── Built-in middleware: CORS ───────────────────────────────────────────────

#[derive(Debug)]
pub struct CorsMiddleware;

impl Middleware for CorsMiddleware {
    fn process(&self, req: ApiRequest, config: &GatewayConfig) -> MiddlewareResult {
        if req.method == HttpMethod::Options && config.enable_cors {
            let mut resp = ApiResponse::ok(Vec::new(), &req.request_id);
            resp.status_code = StatusCode::NO_CONTENT;
            for (k, v) in config.cors.to_headers() {
                resp.headers.insert(k, v);
            }
            MiddlewareResult::ShortCircuit(resp)
        } else {
            MiddlewareResult::Continue(req)
        }
    }

    fn name(&self) -> &'static str {
        "cors"
    }
}

// ── Built-in middleware: request logging ────────────────────────────────────

/// Simple middleware that records request ids it has seen (for testing).
#[derive(Debug, Default)]
pub struct LoggingMiddleware {
    pub seen: std::sync::Mutex<Vec<String>>,
}

impl Middleware for LoggingMiddleware {
    fn process(&self, req: ApiRequest, _config: &GatewayConfig) -> MiddlewareResult {
        if let Ok(mut v) = self.seen.lock() {
            v.push(req.request_id.clone());
        }
        MiddlewareResult::Continue(req)
    }

    fn name(&self) -> &'static str {
        "logging"
    }
}
