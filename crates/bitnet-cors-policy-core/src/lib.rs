//! SRP CORS policy primitives for middleware configuration.

use std::time::Duration;

/// Shared CORS policy settings for middleware stacks.
#[derive(Debug, Clone)]
pub struct CorsPolicy {
    pub allowed_origins: Vec<String>,
    pub allow_credentials: bool,
    pub max_age: Duration,
}

impl Default for CorsPolicy {
    fn default() -> Self {
        Self {
            allowed_origins: vec!["*".into()],
            allow_credentials: false,
            max_age: Duration::from_secs(3600),
        }
    }
}

impl CorsPolicy {
    /// Restrictive profile allowing only explicit origins.
    #[must_use]
    pub const fn restrictive(origins: Vec<String>) -> Self {
        Self {
            allowed_origins: origins,
            allow_credentials: true,
            max_age: Duration::from_secs(600),
        }
    }

    /// Returns true when wildcard origin is configured.
    #[must_use]
    pub fn is_wildcard(&self) -> bool {
        self.allowed_origins.iter().any(|o| o == "*")
    }
}

#[cfg(test)]
mod tests {
    use super::CorsPolicy;

    #[test]
    fn default_policy_uses_wildcard_without_credentials() {
        let policy = CorsPolicy::default();
        assert!(policy.is_wildcard());
        assert!(!policy.allow_credentials);
    }

    #[test]
    fn restrictive_policy_disables_wildcard_and_enables_credentials() {
        let policy = CorsPolicy::restrictive(vec!["https://example.com".to_string()]);
        assert!(!policy.is_wildcard());
        assert!(policy.allow_credentials);
    }
}
