//! SRP CORS policy primitives shared across server-facing crates.

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
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Restrictive preset: no wildcards.
    #[must_use]
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
    #[must_use]
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

    #[must_use]
    pub fn add_origin(mut self, origin: impl Into<String>) -> Self {
        self.allowed_origins.push(origin.into());
        self
    }

    #[must_use]
    pub fn add_method(mut self, method: impl Into<String>) -> Self {
        self.allowed_methods.push(method.into());
        self
    }

    #[must_use]
    pub fn add_header(mut self, header: impl Into<String>) -> Self {
        self.allowed_headers.push(header.into());
        self
    }

    #[must_use]
    pub fn with_credentials(mut self) -> Self {
        self.allow_credentials = true;
        self
    }

    #[must_use]
    pub fn with_max_age(mut self, secs: u64) -> Self {
        self.max_age_secs = secs;
        self
    }

    /// Check if a given origin is allowed.
    #[must_use]
    pub fn is_origin_allowed(&self, origin: &str) -> bool {
        self.allowed_origins.iter().any(|o| o == "*" || o == origin)
    }

    /// Check if a given method is allowed.
    #[must_use]
    pub fn is_method_allowed(&self, method: &str) -> bool {
        self.allowed_methods.iter().any(|m| m.eq_ignore_ascii_case(method))
    }

    /// Generate CORS headers for a response.
    #[must_use]
    pub fn response_headers(&self, request_origin: Option<&str>) -> Vec<(String, String)> {
        let mut headers = Vec::new();

        if let Some(origin) = request_origin
            && self.is_origin_allowed(origin)
        {
            if self.allowed_origins.iter().any(|o| o == "*") && !self.allow_credentials {
                headers.push(("Access-Control-Allow-Origin".into(), "*".into()));
            } else {
                headers.push(("Access-Control-Allow-Origin".into(), origin.into()));
            }
        }

        if !self.allowed_methods.is_empty() {
            headers.push(("Access-Control-Allow-Methods".into(), self.allowed_methods.join(", ")));
        }

        if !self.allowed_headers.is_empty() {
            headers.push(("Access-Control-Allow-Headers".into(), self.allowed_headers.join(", ")));
        }

        if self.allow_credentials {
            headers.push(("Access-Control-Allow-Credentials".into(), "true".into()));
        }

        headers.push(("Access-Control-Max-Age".into(), self.max_age_secs.to_string()));

        headers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_allows_default_paths() {
        let config = CorsConfig::default();
        assert!(config.is_origin_allowed("http://example.com"));
        assert!(config.is_method_allowed("GET"));
        assert!(config.is_method_allowed("POST"));
    }

    #[test]
    fn restrictive_blocks_unknown_origins() {
        let config = CorsConfig::restrictive();
        assert!(!config.is_origin_allowed("http://example.com"));
        assert!(!config.allow_credentials);
    }

    #[test]
    fn response_headers_include_expected_values() {
        let config =
            CorsConfig::restrictive().add_origin("http://localhost:3000").with_credentials();
        let headers = config.response_headers(Some("http://localhost:3000"));
        assert!(
            headers
                .iter()
                .any(|(k, v)| k == "Access-Control-Allow-Origin" && v == "http://localhost:3000")
        );
        assert!(
            headers.iter().any(|(k, v)| k == "Access-Control-Allow-Credentials" && v == "true")
        );
    }
}
