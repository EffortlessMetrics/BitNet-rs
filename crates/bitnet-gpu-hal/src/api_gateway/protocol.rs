use std::collections::HashMap;
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

use super::util::generate_request_id;

// ── API version ─────────────────────────────────────────────────────────────

/// Supported API versions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ApiVersion {
    V1,
    V2,
}

impl fmt::Display for ApiVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::V1 => write!(f, "v1"),
            Self::V2 => write!(f, "v2"),
        }
    }
}

impl ApiVersion {
    /// Try to parse a version from a string like `"v1"` or `"v2"`.
    pub fn from_str_prefix(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "v1" => Some(Self::V1),
            "v2" => Some(Self::V2),
            _ => None,
        }
    }

    /// Return the URL prefix for this version (e.g. `"/v1"`).
    pub const fn prefix(self) -> &'static str {
        match self {
            Self::V1 => "/v1",
            Self::V2 => "/v2",
        }
    }
}

// ── HTTP method ─────────────────────────────────────────────────────────────

/// HTTP methods supported by the gateway.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HttpMethod {
    Get,
    Post,
    Put,
    Delete,
    Patch,
    Options,
}

impl fmt::Display for HttpMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Get => "GET",
            Self::Post => "POST",
            Self::Put => "PUT",
            Self::Delete => "DELETE",
            Self::Patch => "PATCH",
            Self::Options => "OPTIONS",
        };
        write!(f, "{s}")
    }
}

impl HttpMethod {
    pub fn from_str_upper(s: &str) -> Option<Self> {
        match s.trim().to_uppercase().as_str() {
            "GET" => Some(Self::Get),
            "POST" => Some(Self::Post),
            "PUT" => Some(Self::Put),
            "DELETE" => Some(Self::Delete),
            "PATCH" => Some(Self::Patch),
            "OPTIONS" => Some(Self::Options),
            _ => None,
        }
    }
}

// ── Status code ─────────────────────────────────────────────────────────────

/// Thin wrapper around an HTTP status code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StatusCode(pub u16);

impl StatusCode {
    pub const OK: Self = Self(200);
    pub const CREATED: Self = Self(201);
    pub const NO_CONTENT: Self = Self(204);
    pub const BAD_REQUEST: Self = Self(400);
    pub const UNAUTHORIZED: Self = Self(401);
    pub const FORBIDDEN: Self = Self(403);
    pub const NOT_FOUND: Self = Self(404);
    pub const METHOD_NOT_ALLOWED: Self = Self(405);
    pub const REQUEST_TIMEOUT: Self = Self(408);
    pub const PAYLOAD_TOO_LARGE: Self = Self(413);
    pub const TOO_MANY_REQUESTS: Self = Self(429);
    pub const INTERNAL_SERVER_ERROR: Self = Self(500);
    pub const SERVICE_UNAVAILABLE: Self = Self(503);

    pub fn is_success(self) -> bool {
        (200..300).contains(&self.0)
    }

    pub fn is_client_error(self) -> bool {
        (400..500).contains(&self.0)
    }

    pub fn is_server_error(self) -> bool {
        (500..600).contains(&self.0)
    }

    pub const fn reason_phrase(self) -> &'static str {
        match self.0 {
            200 => "OK",
            201 => "Created",
            204 => "No Content",
            400 => "Bad Request",
            401 => "Unauthorized",
            403 => "Forbidden",
            404 => "Not Found",
            405 => "Method Not Allowed",
            408 => "Request Timeout",
            413 => "Payload Too Large",
            429 => "Too Many Requests",
            500 => "Internal Server Error",
            503 => "Service Unavailable",
            _ => "Unknown",
        }
    }
}

impl fmt::Display for StatusCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.0, self.reason_phrase())
    }
}

// ── Endpoint definition ─────────────────────────────────────────────────────

/// A registered API endpoint.
#[derive(Debug, Clone)]
pub struct ApiEndpoint {
    /// URL path pattern (e.g. `"/chat/completions"`).
    pub path: String,
    /// Accepted HTTP method.
    pub method: HttpMethod,
    /// Logical handler name used for dispatch.
    pub handler_name: String,
    /// Whether this endpoint requires authentication.
    pub auth_required: bool,
    /// Optional per-endpoint rate limit (requests/sec). `None` = unlimited.
    pub rate_limit: Option<u32>,
}

impl ApiEndpoint {
    /// Canonical route key combining method + path, e.g. `"POST /chat/completions"`.
    pub fn route_key(&self) -> String {
        format!("{} {}", self.method, self.path)
    }
}

// ── Request / Response ──────────────────────────────────────────────────────

/// A parsed inbound API request.
#[derive(Debug, Clone)]
pub struct ApiRequest {
    /// Target endpoint path (without version prefix).
    pub endpoint: String,
    /// HTTP method.
    pub method: HttpMethod,
    /// Request headers.
    pub headers: HashMap<String, String>,
    /// Raw request body.
    pub body: Vec<u8>,
    /// Extracted API key (if present).
    pub api_key: Option<String>,
    /// Unique request identifier.
    pub request_id: String,
    /// Request receive timestamp (millis since UNIX epoch).
    pub timestamp: u64,
}

impl ApiRequest {
    /// Create a new request with auto-generated id and timestamp.
    pub fn new(
        endpoint: impl Into<String>,
        method: HttpMethod,
        headers: HashMap<String, String>,
        body: Vec<u8>,
    ) -> Self {
        #[allow(clippy::cast_possible_truncation)] // millis won't exceed u64 in practice
        let timestamp =
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64;
        Self {
            endpoint: endpoint.into(),
            method,
            headers,
            body,
            api_key: None,
            request_id: generate_request_id(),
            timestamp,
        }
    }

    /// Body length in bytes.
    pub const fn body_len(&self) -> usize {
        self.body.len()
    }
}

/// An outbound API response.
#[derive(Debug, Clone)]
pub struct ApiResponse {
    /// HTTP status code.
    pub status_code: StatusCode,
    /// Response headers.
    pub headers: HashMap<String, String>,
    /// Raw response body.
    pub body: Vec<u8>,
    /// Correlation request id.
    pub request_id: String,
    /// Time spent processing, in milliseconds.
    pub latency_ms: u64,
}

impl ApiResponse {
    pub fn ok(body: Vec<u8>, request_id: impl Into<String>) -> Self {
        Self {
            status_code: StatusCode::OK,
            headers: HashMap::new(),
            body,
            request_id: request_id.into(),
            latency_ms: 0,
        }
    }

    pub fn error(
        status: StatusCode,
        message: impl Into<String>,
        request_id: impl Into<String>,
    ) -> Self {
        let msg = message.into();
        let body = format!(
            r#"{{"error":{{"message":"{}","type":"api_error","code":{}}}}}"#,
            msg, status.0
        );
        Self {
            status_code: status,
            headers: HashMap::new(),
            body: body.into_bytes(),
            request_id: request_id.into(),
            latency_ms: 0,
        }
    }

    #[must_use]
    pub const fn with_latency(mut self, latency_ms: u64) -> Self {
        self.latency_ms = latency_ms;
        self
    }

    #[must_use]
    pub fn with_header(mut self, key: impl Into<String>, val: impl Into<String>) -> Self {
        self.headers.insert(key.into(), val.into());
        self
    }
}
