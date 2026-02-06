//! Security features including validation, authentication, and input sanitization

use anyhow::Result;
use axum::{
    extract::{Request, State},
    http::{HeaderMap, StatusCode},
    middleware::Next,
    response::Response,
};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::net::IpAddr;
use std::sync::Arc;
use tracing::{debug, warn};

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub jwt_secret: Option<String>,
    pub require_authentication: bool,
    pub max_prompt_length: usize,
    pub max_tokens_per_request: u32,
    pub allowed_origins: Vec<String>,
    pub blocked_ips: HashSet<IpAddr>,
    pub rate_limit_by_ip: bool,
    pub input_sanitization: bool,
    pub content_filtering: bool,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            jwt_secret: None,
            require_authentication: false,
            max_prompt_length: 8192, // 8KB max prompt
            max_tokens_per_request: 2048,
            allowed_origins: vec!["*".to_string()],
            blocked_ips: HashSet::new(),
            rate_limit_by_ip: true,
            input_sanitization: true,
            content_filtering: true,
        }
    }
}

/// JWT claims structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,             // Subject (user ID)
    pub exp: usize,              // Expiration time
    pub iat: usize,              // Issued at
    pub role: Option<String>,    // User role
    pub rate_limit: Option<u64>, // Custom rate limit for user
}

/// Authentication middleware state
#[derive(Clone)]
pub struct AuthState {
    pub config: SecurityConfig,
    pub jwt_secret: Option<String>,
}

/// Request validation errors
#[derive(Debug, thiserror::Error)]
pub enum ValidationError {
    #[error("Prompt too long: {0} characters (max: {1})")]
    PromptTooLong(usize, usize),

    #[error("Too many tokens requested: {0} (max: {1})")]
    TooManyTokens(u32, u32),

    #[error("Invalid characters in prompt")]
    InvalidCharacters,

    #[error("Blocked content detected: {0}")]
    BlockedContent(String),

    #[error("Missing required field: {0}")]
    MissingField(String),

    #[error("Invalid field value: {0}")]
    InvalidFieldValue(String),
}

/// Security validator
pub struct SecurityValidator {
    config: SecurityConfig,
    blocked_patterns: Vec<regex::Regex>,
}

impl SecurityValidator {
    pub fn new(config: SecurityConfig) -> Result<Self> {
        let blocked_patterns = if config.content_filtering {
            vec![
                regex::Regex::new(r"(?i)(hack|exploit|vulnerability)")?,
                regex::Regex::new(r"(?i)(malware|virus|trojan)")?,
                regex::Regex::new(r"(?i)(sql\s+injection|xss|csrf)")?,
                // Add more patterns as needed
            ]
        } else {
            Vec::new()
        };

        Ok(Self { config, blocked_patterns })
    }

    /// Get access to the security configuration
    pub fn config(&self) -> &SecurityConfig {
        &self.config
    }

    /// Validate inference request
    pub fn validate_inference_request(
        &self,
        request: &crate::InferenceRequest,
    ) -> Result<(), ValidationError> {
        // Check prompt length
        if request.prompt.len() > self.config.max_prompt_length {
            return Err(ValidationError::PromptTooLong(
                request.prompt.len(),
                self.config.max_prompt_length,
            ));
        }

        // Check max tokens
        if let Some(max_tokens) = request.max_tokens
            && max_tokens > self.config.max_tokens_per_request as usize
        {
            return Err(ValidationError::TooManyTokens(
                max_tokens as u32,
                self.config.max_tokens_per_request,
            ));
        }

        // Input sanitization
        if self.config.input_sanitization {
            self.sanitize_input(&request.prompt)?;
        }

        // Content filtering
        if self.config.content_filtering {
            self.check_content_filter(&request.prompt)?;
        }

        // Validate optional parameters
        if let Some(temp) = request.temperature
            && (!(0.0..=2.0).contains(&temp))
        {
            return Err(ValidationError::InvalidFieldValue(format!(
                "temperature must be between 0.0 and 2.0, got {}",
                temp
            )));
        }

        if let Some(top_p) = request.top_p
            && (!(0.0..=1.0).contains(&top_p))
        {
            return Err(ValidationError::InvalidFieldValue(format!(
                "top_p must be between 0.0 and 1.0, got {}",
                top_p
            )));
        }

        if let Some(top_k) = request.top_k
            && (top_k == 0 || top_k > 1000)
        {
            return Err(ValidationError::InvalidFieldValue(format!(
                "top_k must be between 1 and 1000, got {}",
                top_k
            )));
        }

        if let Some(rep_penalty) = request.repetition_penalty
            && (!(0.1..=10.0).contains(&rep_penalty))
        {
            return Err(ValidationError::InvalidFieldValue(format!(
                "repetition_penalty must be between 0.1 and 10.0, got {}",
                rep_penalty
            )));
        }

        Ok(())
    }

    /// Sanitize input text
    fn sanitize_input(&self, input: &str) -> Result<(), ValidationError> {
        // Check for null bytes and control characters (except newline and tab)
        if input.chars().any(|c| c.is_control() && c != '\n' && c != '\t') {
            return Err(ValidationError::InvalidCharacters);
        }

        // Check for excessively long lines (potential DoS)
        if input.lines().any(|line| line.len() > 1024) {
            return Err(ValidationError::InvalidCharacters);
        }

        Ok(())
    }

    /// Check content against filters
    fn check_content_filter(&self, content: &str) -> Result<(), ValidationError> {
        for pattern in &self.blocked_patterns {
            if let Some(matched) = pattern.find(content) {
                return Err(ValidationError::BlockedContent(matched.as_str().to_string()));
            }
        }

        Ok(())
    }

    /// Validate model loading request
    pub fn validate_model_request(&self, model_path: &str) -> Result<(), ValidationError> {
        // Basic path validation
        if model_path.is_empty() {
            return Err(ValidationError::MissingField("model_path".to_string()));
        }

        // Prevent path traversal attacks
        if model_path.contains("..") || model_path.contains("~") {
            return Err(ValidationError::InvalidFieldValue(
                "Invalid characters in model path".to_string(),
            ));
        }

        // Only allow specific file extensions
        if !model_path.ends_with(".gguf") && !model_path.ends_with(".safetensors") {
            return Err(ValidationError::InvalidFieldValue(
                "Only .gguf and .safetensors files are allowed".to_string(),
            ));
        }

        Ok(())
    }
}

/// Authentication middleware
pub async fn auth_middleware(
    State(auth_state): State<AuthState>,
    headers: HeaderMap,
    mut request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // Skip authentication if not required
    if !auth_state.config.require_authentication {
        return Ok(next.run(request).await);
    }

    // Extract authorization header
    let auth_header = headers
        .get("authorization")
        .and_then(|h| h.to_str().ok())
        .ok_or(StatusCode::UNAUTHORIZED)?;

    // Check if it's a Bearer token
    if !auth_header.starts_with("Bearer ") {
        return Err(StatusCode::UNAUTHORIZED);
    }

    let token = &auth_header[7..]; // Remove "Bearer " prefix

    // Validate JWT token
    if let Some(jwt_secret) = &auth_state.jwt_secret {
        match validate_jwt_token(token, jwt_secret) {
            Ok(claims) => {
                // Add claims to request extensions
                request.extensions_mut().insert(claims);
                debug!("Request authenticated successfully");
            }
            Err(e) => {
                warn!(error = %e, "JWT validation failed");
                return Err(StatusCode::UNAUTHORIZED);
            }
        }
    } else {
        warn!("JWT secret not configured but authentication required");
        return Err(StatusCode::INTERNAL_SERVER_ERROR);
    }

    Ok(next.run(request).await)
}

/// Validate JWT token
fn validate_jwt_token(token: &str, secret: &str) -> Result<Claims> {
    use jsonwebtoken::{Algorithm, DecodingKey, Validation, decode};

    let decoding_key = DecodingKey::from_secret(secret.as_bytes());
    let validation = Validation::new(Algorithm::HS256);

    let token_data = decode::<Claims>(token, &decoding_key, &validation)?;

    Ok(token_data.claims)
}

/// IP blocking middleware
pub async fn ip_blocking_middleware(
    State(config): State<SecurityConfig>,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // Extract client IP
    let client_ip = extract_client_ip(&request);

    // Check if IP is blocked
    if let Some(ip) = client_ip
        && config.blocked_ips.contains(&ip)
    {
        warn!(ip = %ip, "Blocked IP attempted access");
        return Err(StatusCode::FORBIDDEN);
    }

    Ok(next.run(request).await)
}

/// Extract client IP from request
fn extract_client_ip(request: &Request) -> Option<IpAddr> {
    extract_client_ip_from_headers(request.headers())
}

/// Extract client IP from headers (shared utility)
pub fn extract_client_ip_from_headers(headers: &HeaderMap) -> Option<IpAddr> {
    // Try X-Forwarded-For header first (for reverse proxies)
    if let Some(forwarded) = headers.get("x-forwarded-for")
        && let Ok(forwarded_str) = forwarded.to_str()
        && let Some(first_ip) = forwarded_str.split(',').next()
        && let Ok(ip) = first_ip.trim().parse::<IpAddr>()
    {
        return Some(ip);
    }

    // Try X-Real-IP header
    if let Some(real_ip) = headers.get("x-real-ip")
        && let Ok(ip_str) = real_ip.to_str()
        && let Ok(ip) = ip_str.parse::<IpAddr>()
    {
        return Some(ip);
    }

    // Fall back to connection info (would need to be passed from the server)
    None
}

/// CORS middleware configuration
pub fn configure_cors(config: &SecurityConfig) -> tower_http::cors::CorsLayer {
    use axum::http::HeaderValue;
    use tower_http::cors::{Any, CorsLayer};

    let mut cors = CorsLayer::new()
        .allow_methods(Any)
        .allow_headers(Any)
        .max_age(std::time::Duration::from_secs(3600));

    if config.allowed_origins.contains(&"*".to_string()) {
        cors = cors.allow_origin(Any);
    } else {
        let origins: Vec<HeaderValue> = config
            .allowed_origins
            .iter()
            .filter_map(|origin| match origin.parse::<HeaderValue>() {
                Ok(val) => Some(val),
                Err(e) => {
                    warn!("Invalid CORS origin '{}': {}", origin, e);
                    None
                }
            })
            .collect();

        if !origins.is_empty() {
            cors = cors.allow_origin(origins);
        }
    }

    cors
}

/// Input validation helper for JSON payloads
pub fn validate_json_payload<T>(payload: &str, max_size: usize) -> Result<T>
where
    T: serde::de::DeserializeOwned,
{
    // Check payload size
    if payload.len() > max_size {
        anyhow::bail!("Payload too large: {} bytes (max: {})", payload.len(), max_size);
    }

    // Parse JSON
    let parsed: T =
        serde_json::from_str(payload).map_err(|e| anyhow::anyhow!("Invalid JSON: {}", e))?;

    Ok(parsed)
}

/// Security headers middleware
pub async fn security_headers_middleware(request: Request, next: Next) -> Response {
    let mut response = next.run(request).await;

    let headers = response.headers_mut();

    // Add security headers
    headers.insert("X-Content-Type-Options", "nosniff".parse().unwrap());
    headers.insert("X-Frame-Options", "DENY".parse().unwrap());
    headers.insert("X-XSS-Protection", "1; mode=block".parse().unwrap());
    headers.insert("Referrer-Policy", "strict-origin-when-cross-origin".parse().unwrap());
    headers.insert(
        "Content-Security-Policy",
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'".parse().unwrap(),
    );

    response
}

/// Request sanitization middleware
pub async fn request_sanitization_middleware(
    State(validator): State<Arc<SecurityValidator>>,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // For inference requests, we'll validate in the handler
    // This middleware focuses on general request sanitization

    // Check request size
    if let Some(content_length) = request.headers().get("content-length")
        && let Ok(length_str) = content_length.to_str()
        && let Ok(length) = length_str.parse::<usize>()
        && length > validator.config.max_prompt_length * 2
    {
        warn!(content_length = length, "Request too large");
        return Err(StatusCode::PAYLOAD_TOO_LARGE);
    }

    // Check Content-Type for JSON endpoints
    let content_type = request.headers().get("content-type").and_then(|ct| ct.to_str().ok());

    if let Some(ct) = content_type {
        if ct.contains("application/json") {
            // JSON request - will be validated in handlers
        } else if !ct.contains("multipart/form-data") {
            // Only allow JSON and form data
            return Err(StatusCode::UNSUPPORTED_MEDIA_TYPE);
        }
    }

    Ok(next.run(request).await)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prompt_validation() {
        let config = SecurityConfig {
            max_prompt_length: 100,
            max_tokens_per_request: 50,
            input_sanitization: true,
            content_filtering: false,
            ..Default::default()
        };

        let validator = SecurityValidator::new(config).unwrap();

        let request = crate::InferenceRequest {
            prompt: "Hello, world!".to_string(),
            max_tokens: Some(25),
            model: None,
            temperature: Some(1.0),
            top_p: Some(0.9),
            top_k: Some(50),
            repetition_penalty: Some(1.0),
        };

        assert!(validator.validate_inference_request(&request).is_ok());
    }

    #[test]
    fn test_prompt_too_long() {
        let config = SecurityConfig { max_prompt_length: 10, ..Default::default() };

        let validator = SecurityValidator::new(config).unwrap();

        let request = crate::InferenceRequest {
            prompt: "This is a very long prompt that exceeds the limit".to_string(),
            max_tokens: Some(25),
            model: None,
            temperature: None,
            top_p: None,
            top_k: None,
            repetition_penalty: None,
        };

        assert!(matches!(
            validator.validate_inference_request(&request),
            Err(ValidationError::PromptTooLong(_, _))
        ));
    }

    #[test]
    fn test_invalid_temperature() {
        let config = SecurityConfig::default();
        let validator = SecurityValidator::new(config).unwrap();

        let request = crate::InferenceRequest {
            prompt: "Hello".to_string(),
            max_tokens: Some(25),
            model: None,
            temperature: Some(5.0), // Invalid: too high
            top_p: None,
            top_k: None,
            repetition_penalty: None,
        };

        assert!(matches!(
            validator.validate_inference_request(&request),
            Err(ValidationError::InvalidFieldValue(_))
        ));
    }

    #[tokio::test]
    async fn test_cors_wildcard() {
        use axum::{Router, routing::get};
        use tower::ServiceExt;

        let config =
            SecurityConfig { allowed_origins: vec!["*".to_string()], ..Default::default() };

        let app =
            Router::new().route("/", get(|| async { "hello" })).layer(configure_cors(&config));

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/")
                    .header("Origin", "http://example.com")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.headers().get("access-control-allow-origin").unwrap(), "*");
    }

    #[tokio::test]
    async fn test_cors_specific_origin() {
        use axum::{Router, routing::get};
        use tower::ServiceExt;

        let config = SecurityConfig {
            allowed_origins: vec!["http://trusted.com".to_string()],
            ..Default::default()
        };

        let app =
            Router::new().route("/", get(|| async { "hello" })).layer(configure_cors(&config));

        // Allowed origin
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/")
                    .header("Origin", "http://trusted.com")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(
            response.headers().get("access-control-allow-origin").unwrap(),
            "http://trusted.com"
        );

        // Blocked origin
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/")
                    .header("Origin", "http://evil.com")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        // When origin is blocked, the header is usually missing or doesn't match
        assert!(response.headers().get("access-control-allow-origin").is_none());
    }
}
