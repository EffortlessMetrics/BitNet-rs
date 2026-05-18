use std::collections::HashMap;

use super::{ApiVersion, HttpMethod};

// ── Configuration ───────────────────────────────────────────────────────────

/// Top-level gateway configuration.
#[derive(Debug, Clone)]
pub struct GatewayConfig {
    /// API version to serve.
    pub api_version: ApiVersion,
    /// Whether to enable CORS headers on responses.
    pub enable_cors: bool,
    /// Maximum allowed request body size in bytes.
    pub max_request_size_bytes: usize,
    /// Per-request timeout in milliseconds.
    pub timeout_ms: u64,
    /// Header name used to carry the API key.
    pub api_key_header: String,
    /// CORS settings (used only when `enable_cors` is true).
    pub cors: CorsConfig,
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            api_version: ApiVersion::V1,
            enable_cors: true,
            max_request_size_bytes: 4 * 1024 * 1024, // 4 MiB
            timeout_ms: 30_000,
            api_key_header: "Authorization".to_string(),
            cors: CorsConfig::default(),
        }
    }
}

// ── CORS ────────────────────────────────────────────────────────────────────

/// Cross-Origin Resource Sharing (CORS) configuration.
#[derive(Debug, Clone)]
pub struct CorsConfig {
    pub allowed_origins: Vec<String>,
    pub allowed_methods: Vec<HttpMethod>,
    pub allowed_headers: Vec<String>,
    /// Max age for preflight caching, in seconds.
    pub max_age: u32,
}

impl Default for CorsConfig {
    fn default() -> Self {
        Self {
            allowed_origins: vec!["*".to_string()],
            allowed_methods: vec![HttpMethod::Get, HttpMethod::Post, HttpMethod::Options],
            allowed_headers: vec!["Content-Type".to_string(), "Authorization".to_string()],
            max_age: 86_400,
        }
    }
}

impl CorsConfig {
    /// Generate CORS headers as key-value pairs.
    pub fn to_headers(&self) -> HashMap<String, String> {
        let mut h = HashMap::new();
        h.insert("Access-Control-Allow-Origin".to_string(), self.allowed_origins.join(", "));
        h.insert(
            "Access-Control-Allow-Methods".to_string(),
            self.allowed_methods
                .iter()
                .map(std::string::ToString::to_string)
                .collect::<Vec<_>>()
                .join(", "),
        );
        h.insert("Access-Control-Allow-Headers".to_string(), self.allowed_headers.join(", "));
        h.insert("Access-Control-Max-Age".to_string(), self.max_age.to_string());
        h
    }
}
