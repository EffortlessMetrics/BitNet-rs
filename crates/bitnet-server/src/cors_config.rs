//! CORS (Cross-Origin Resource Sharing) configuration.
//!
//! Configure allowed origins, methods, headers, and preflight
//! handling for the inference server API.

/// CORS configuration.
#[derive(Debug, Clone)]
pub struct CorsConfig {
    pub allowed_origins: Vec<String>,
    pub allowed_methods: Vec<String>,
    pub allowed_headers: Vec<String>,
    pub expose_headers: Vec<String>,
    pub allow_credentials: bool,
    pub max_age_secs: u64,
}

impl Default for CorsConfig {
    fn default() -> Self {
        Self {
            allowed_origins: vec!["*".to_string()],
            allowed_methods: vec!["GET".to_string(), "POST".to_string(), "OPTIONS".to_string()],
            allowed_headers: vec!["Content-Type".to_string(), "Authorization".to_string()],
            expose_headers: vec![],
            allow_credentials: false,
            max_age_secs: 3600,
        }
    }
}

impl CorsConfig {
    pub fn new() -> Self {
        Self::default()
    }

    /// Restrictive preset: no wildcards.
    pub fn restrictive() -> Self {
        Self {
            allowed_origins: vec![],
            allowed_methods: vec!["GET".to_string(), "POST".to_string()],
            allowed_headers: vec!["Content-Type".to_string()],
            expose_headers: vec![],
            allow_credentials: false,
            max_age_secs: 600,
        }
    }

    /// Permissive preset: allow everything.
    pub fn permissive() -> Self {
        Self {
            allowed_origins: vec!["*".to_string()],
            allowed_methods: vec![
                "GET".to_string(),
                "POST".to_string(),
                "PUT".to_string(),
                "DELETE".to_string(),
                "PATCH".to_string(),
                "OPTIONS".to_string(),
            ],
            allowed_headers: vec!["*".to_string()],
            expose_headers: vec!["*".to_string()],
            allow_credentials: false,
            max_age_secs: 86400,
        }
    }

    pub fn add_origin(mut self, origin: impl Into<String>) -> Self {
        self.allowed_origins.push(origin.into());
        self
    }

    pub fn add_method(mut self, method: impl Into<String>) -> Self {
        self.allowed_methods.push(method.into());
        self
    }

    pub fn add_header(mut self, header: impl Into<String>) -> Self {
        self.allowed_headers.push(header.into());
        self
    }

    pub fn with_credentials(mut self) -> Self {
        self.allow_credentials = true;
        self
    }

    pub fn with_max_age(mut self, secs: u64) -> Self {
        self.max_age_secs = secs;
        self
    }

    /// Check if a given origin is allowed.
    pub fn is_origin_allowed(&self, origin: &str) -> bool {
        self.allowed_origins.iter().any(|o| o == "*" || o == origin)
    }

    /// Check if a given method is allowed.
    pub fn is_method_allowed(&self, method: &str) -> bool {
        self.allowed_methods.iter().any(|m| m.eq_ignore_ascii_case(method))
    }

    /// Generate CORS headers for a response.
    pub fn response_headers(&self, request_origin: Option<&str>) -> Vec<(String, String)> {
        let mut headers = Vec::new();

        // Access-Control-Allow-Origin
        if let Some(origin) = request_origin
            && self.is_origin_allowed(origin)
        {
            if self.allowed_origins.iter().any(|o| o == "*") && !self.allow_credentials {
                headers.push(("Access-Control-Allow-Origin".into(), "*".into()));
            } else {
                headers.push(("Access-Control-Allow-Origin".into(), origin.into()));
            }
        }

        // Methods
        if !self.allowed_methods.is_empty() {
            headers.push(("Access-Control-Allow-Methods".into(), self.allowed_methods.join(", ")));
        }

        // Headers
        if !self.allowed_headers.is_empty() {
            headers.push(("Access-Control-Allow-Headers".into(), self.allowed_headers.join(", ")));
        }

        // Credentials
        if self.allow_credentials {
            headers.push(("Access-Control-Allow-Credentials".into(), "true".into()));
        }

        // Max-Age
        headers.push(("Access-Control-Max-Age".into(), self.max_age_secs.to_string()));

        headers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = CorsConfig::default();
        assert!(config.is_origin_allowed("http://example.com"));
        assert!(config.is_method_allowed("GET"));
        assert!(config.is_method_allowed("POST"));
    }

    #[test]
    fn test_restrictive() {
        let config = CorsConfig::restrictive();
        assert!(!config.is_origin_allowed("http://example.com"));
        assert!(!config.allow_credentials);
    }

    #[test]
    fn test_permissive() {
        let config = CorsConfig::permissive();
        assert!(config.is_origin_allowed("anything"));
        assert!(config.is_method_allowed("DELETE"));
    }

    #[test]
    fn test_specific_origin() {
        let config = CorsConfig::restrictive().add_origin("http://localhost:3000");
        assert!(config.is_origin_allowed("http://localhost:3000"));
        assert!(!config.is_origin_allowed("http://evil.com"));
    }

    #[test]
    fn test_method_check() {
        let config = CorsConfig::default();
        assert!(config.is_method_allowed("get")); // case insensitive
        assert!(!config.is_method_allowed("DELETE"));
    }

    #[test]
    fn test_builder_chain() {
        let config = CorsConfig::new()
            .add_origin("https://app.example.com")
            .add_method("PUT")
            .add_header("X-Custom")
            .with_credentials()
            .with_max_age(7200);
        assert!(config.allow_credentials);
        assert_eq!(config.max_age_secs, 7200);
    }

    #[test]
    fn test_response_headers_wildcard() {
        let config = CorsConfig::default();
        let headers = config.response_headers(Some("http://example.com"));
        let origin_header = headers.iter().find(|(k, _)| k == "Access-Control-Allow-Origin");
        assert_eq!(origin_header.unwrap().1, "*");
    }

    #[test]
    fn test_response_headers_specific() {
        let config =
            CorsConfig::restrictive().add_origin("http://localhost:3000").with_credentials();
        let headers = config.response_headers(Some("http://localhost:3000"));
        let origin = headers.iter().find(|(k, _)| k == "Access-Control-Allow-Origin");
        assert_eq!(origin.unwrap().1, "http://localhost:3000");
        let creds = headers.iter().find(|(k, _)| k == "Access-Control-Allow-Credentials");
        assert_eq!(creds.unwrap().1, "true");
    }

    #[test]
    fn test_response_headers_blocked() {
        let config = CorsConfig::restrictive();
        let headers = config.response_headers(Some("http://evil.com"));
        let origin = headers.iter().find(|(k, _)| k == "Access-Control-Allow-Origin");
        assert!(origin.is_none());
    }

    #[test]
    fn test_max_age_header() {
        let config = CorsConfig::default();
        let headers = config.response_headers(None);
        let max_age = headers.iter().find(|(k, _)| k == "Access-Control-Max-Age");
        assert_eq!(max_age.unwrap().1, "3600");
    }

    #[test]
    fn test_expose_headers() {
        let config = CorsConfig::permissive();
        assert!(!config.expose_headers.is_empty());
    }

    #[test]
    fn test_empty_methods() {
        let config = CorsConfig { allowed_methods: vec![], ..CorsConfig::default() };
        assert!(!config.is_method_allowed("GET"));
    }
}
