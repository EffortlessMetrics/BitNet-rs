//! Server middleware configuration.
//!
//! Rate limiting, CORS, logging, and request validation middleware.

use std::time::Duration;

/// Rate limit configuration.
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    pub requests_per_second: u32,
    pub burst_size: u32,
    pub per_ip: bool,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self { requests_per_second: 100, burst_size: 200, per_ip: true }
    }
}

/// CORS configuration.
#[derive(Debug, Clone)]
pub struct CorsConfig {
    pub allowed_origins: Vec<String>,
    pub allow_credentials: bool,
    pub max_age: Duration,
}

impl Default for CorsConfig {
    fn default() -> Self {
        Self {
            allowed_origins: vec!["*".into()],
            allow_credentials: false,
            max_age: Duration::from_hours(1),
        }
    }
}

impl CorsConfig {
    pub fn restrictive(origins: Vec<String>) -> Self {
        Self { allowed_origins: origins, allow_credentials: true, max_age: Duration::from_mins(10) }
    }

    pub fn is_wildcard(&self) -> bool {
        self.allowed_origins.iter().any(|o| o == "*")
    }
}

/// Request validation config.
#[derive(Debug, Clone)]
pub struct RequestValidation {
    pub max_body_bytes: usize,
    pub max_prompt_tokens: usize,
    pub max_output_tokens: usize,
    pub require_model_field: bool,
}

impl Default for RequestValidation {
    fn default() -> Self {
        Self {
            max_body_bytes: 10 * 1024 * 1024, // 10 MB
            max_prompt_tokens: 16384,
            max_output_tokens: 4096,
            require_model_field: false,
        }
    }
}

/// Logging configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Off,
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl LogLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Error => "error",
            Self::Warn => "warn",
            Self::Info => "info",
            Self::Debug => "debug",
            Self::Trace => "trace",
        }
    }
}

/// Full middleware stack configuration.
#[derive(Debug, Clone)]
pub struct MiddlewareConfig {
    pub rate_limit: Option<RateLimitConfig>,
    pub cors: CorsConfig,
    pub validation: RequestValidation,
    pub log_level: LogLevel,
    pub request_id: bool,
    pub compression: bool,
    pub timeout: Duration,
}

impl Default for MiddlewareConfig {
    fn default() -> Self {
        Self {
            rate_limit: Some(RateLimitConfig::default()),
            cors: CorsConfig::default(),
            validation: RequestValidation::default(),
            log_level: LogLevel::Info,
            request_id: true,
            compression: true,
            timeout: Duration::from_mins(5),
        }
    }
}

impl MiddlewareConfig {
    pub fn development() -> Self {
        Self {
            rate_limit: None,
            cors: CorsConfig::default(),
            validation: RequestValidation::default(),
            log_level: LogLevel::Debug,
            request_id: true,
            compression: false,
            timeout: Duration::from_mins(10),
        }
    }

    pub fn production() -> Self {
        Self {
            rate_limit: Some(RateLimitConfig {
                requests_per_second: 50,
                burst_size: 100,
                per_ip: true,
            }),
            cors: CorsConfig::restrictive(vec![]),
            validation: RequestValidation {
                max_body_bytes: 5 * 1024 * 1024,
                max_prompt_tokens: 8192,
                max_output_tokens: 2048,
                require_model_field: true,
            },
            log_level: LogLevel::Warn,
            request_id: true,
            compression: true,
            timeout: Duration::from_mins(2),
        }
    }

    pub fn has_rate_limit(&self) -> bool {
        self.rate_limit.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_middleware() {
        let m = MiddlewareConfig::default();
        assert!(m.has_rate_limit());
        assert!(m.request_id);
        assert!(m.compression);
    }

    #[test]
    fn test_development() {
        let m = MiddlewareConfig::development();
        assert!(!m.has_rate_limit());
        assert_eq!(m.log_level, LogLevel::Debug);
    }

    #[test]
    fn test_production() {
        let m = MiddlewareConfig::production();
        assert!(m.has_rate_limit());
        assert_eq!(m.log_level, LogLevel::Warn);
        assert!(m.validation.require_model_field);
    }

    #[test]
    fn test_cors_default() {
        let c = CorsConfig::default();
        assert!(c.is_wildcard());
        assert!(!c.allow_credentials);
    }

    #[test]
    fn test_cors_restrictive() {
        let c = CorsConfig::restrictive(vec!["https://example.com".into()]);
        assert!(!c.is_wildcard());
        assert!(c.allow_credentials);
    }

    #[test]
    fn test_rate_limit_default() {
        let r = RateLimitConfig::default();
        assert_eq!(r.requests_per_second, 100);
        assert!(r.per_ip);
    }

    #[test]
    fn test_validation_default() {
        let v = RequestValidation::default();
        assert_eq!(v.max_prompt_tokens, 16384);
        assert!(!v.require_model_field);
    }

    #[test]
    fn test_log_level_str() {
        assert_eq!(LogLevel::Info.as_str(), "info");
        assert_eq!(LogLevel::Trace.as_str(), "trace");
    }

    #[test]
    fn test_timeout() {
        let m = MiddlewareConfig::default();
        assert_eq!(m.timeout, Duration::from_mins(5));
    }

    #[test]
    fn test_production_limits() {
        let m = MiddlewareConfig::production();
        assert!(m.validation.max_body_bytes < 10 * 1024 * 1024);
    }
}
