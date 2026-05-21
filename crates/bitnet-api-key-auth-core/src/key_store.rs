use crate::api_key::ApiKey;
use crate::auth_result::AuthResult;
use bitnet_http_auth_core::bearer_token;
use std::collections::HashMap;

/// Authentication mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthMode {
    /// No authentication required.
    Disabled,
    /// Require a valid API key.
    Required,
    /// Allow requests without key, but track authenticated ones.
    Optional,
}

/// Key store for managing API keys.
#[derive(Debug)]
pub struct KeyStore {
    keys: HashMap<String, ApiKey>,
    mode: AuthMode,
}

impl KeyStore {
    /// Creates a new key store for the selected auth mode.
    #[must_use]
    pub fn new(mode: AuthMode) -> Self {
        Self { keys: HashMap::new(), mode }
    }

    /// Creates a key store with authentication disabled.
    #[must_use]
    pub fn disabled() -> Self {
        Self::new(AuthMode::Disabled)
    }

    /// Adds or replaces a named API key.
    pub fn add_key(&mut self, name: impl Into<String>, key: &str) {
        let name = name.into();
        let api_key = ApiKey::new(&name, key);
        self.keys.insert(name, api_key);
    }

    /// Removes a named API key.
    pub fn remove_key(&mut self, name: &str) -> bool {
        self.keys.remove(name).is_some()
    }

    /// Disables a named API key.
    pub fn disable_key(&mut self, name: &str) -> bool {
        if let Some(key) = self.keys.get_mut(name) {
            key.enabled = false;
            true
        } else {
            false
        }
    }

    /// Enables a named API key.
    pub fn enable_key(&mut self, name: &str) -> bool {
        if let Some(key) = self.keys.get_mut(name) {
            key.enabled = true;
            true
        } else {
            false
        }
    }

    /// Total number of registered keys.
    #[must_use]
    pub fn key_count(&self) -> usize {
        self.keys.len()
    }

    /// Number of currently enabled keys.
    #[must_use]
    pub fn active_key_count(&self) -> usize {
        self.keys.values().filter(|k| k.enabled).count()
    }

    /// Current authentication mode.
    #[must_use]
    pub fn mode(&self) -> AuthMode {
        self.mode
    }

    /// Authenticates a token according to configured mode.
    #[must_use]
    pub fn authenticate(&mut self, token: Option<&str>) -> AuthResult {
        match self.mode {
            AuthMode::Disabled => AuthResult::Allowed,
            AuthMode::Optional => token.map_or(AuthResult::Allowed, |tok| self.check_token(tok)),
            AuthMode::Required => {
                token.map_or(AuthResult::NoCredentials, |tok| self.check_token(tok))
            }
        }
    }

    fn check_token(&mut self, token: &str) -> AuthResult {
        for key in self.keys.values_mut() {
            if key.matches(token) {
                key.usage_count += 1;
                return AuthResult::Allowed;
            }
        }
        AuthResult::Denied("invalid API key".into())
    }

    /// Extract bearer token from Authorization header value.
    #[must_use]
    pub fn extract_bearer(header_value: &str) -> Option<&str> {
        bearer_token(header_value)
    }

    /// Usage stats by key name.
    #[must_use]
    pub fn usage_stats(&self) -> Vec<(&str, u64)> {
        self.keys.iter().map(|(name, key)| (name.as_str(), key.usage_count)).collect()
    }
}
