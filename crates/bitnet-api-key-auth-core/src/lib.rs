//! SRP core primitives for API-key authentication flows.
//!
//! This crate owns key-store policy and token authentication semantics while
//! remaining independent from HTTP framework types.

use bitnet_http_auth_core::bearer_token;
use std::collections::HashMap;
use std::time::Instant;

/// Authentication result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthResult {
    /// Request is allowed.
    Allowed,
    /// Request was denied with a reason.
    Denied(String),
    /// No credential was supplied.
    NoCredentials,
}

impl AuthResult {
    /// Returns true if this result allows request processing.
    #[must_use]
    pub fn is_allowed(&self) -> bool {
        matches!(self, Self::Allowed)
    }
}

/// API key entry with metadata.
#[derive(Debug, Clone)]
pub struct ApiKey {
    /// Hash of the API key (never stores cleartext key).
    pub key_hash: u64,
    /// Human-friendly key name.
    pub name: String,
    /// Key creation instant.
    pub created_at: Instant,
    /// Number of successful authentications.
    pub usage_count: u64,
    /// Whether this key is enabled.
    pub enabled: bool,
}

impl ApiKey {
    /// Creates a new API key entry.
    #[must_use]
    pub fn new(name: impl Into<String>, key: &str) -> Self {
        Self {
            key_hash: hash_key(key),
            name: name.into(),
            created_at: Instant::now(),
            usage_count: 0,
            enabled: true,
        }
    }

    /// Returns true if this key matches provided cleartext token and is enabled.
    #[must_use]
    pub fn matches(&self, key: &str) -> bool {
        self.enabled && hash_key(key) == self.key_hash
    }
}

fn hash_key(key: &str) -> u64 {
    // Simple FNV-1a hash for key comparison.
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in key.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn required_mode_enforces_credentials() {
        let mut store = KeyStore::new(AuthMode::Required);
        store.add_key("test", "secret123");
        assert_eq!(store.authenticate(None), AuthResult::NoCredentials);
        assert!(store.authenticate(Some("secret123")).is_allowed());
    }

    #[test]
    fn optional_mode_allows_missing_but_rejects_invalid() {
        let mut store = KeyStore::new(AuthMode::Optional);
        store.add_key("test", "secret123");
        assert!(store.authenticate(None).is_allowed());
        assert_eq!(store.authenticate(Some("wrong")), AuthResult::Denied("invalid API key".into()));
    }

    #[test]
    fn key_enable_disable_roundtrip() {
        let mut store = KeyStore::new(AuthMode::Required);
        store.add_key("test", "secret123");
        assert!(store.disable_key("test"));
        assert!(!store.authenticate(Some("secret123")).is_allowed());
        assert!(store.enable_key("test"));
        assert!(store.authenticate(Some("secret123")).is_allowed());
    }

    #[test]
    fn usage_tracking_counts_successes() {
        let mut store = KeyStore::new(AuthMode::Required);
        store.add_key("test", "key1");
        let _ = store.authenticate(Some("key1"));
        let _ = store.authenticate(Some("key1"));
        let stats = store.usage_stats();
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].1, 2);
    }

    #[test]
    fn bearer_extraction_uses_shared_http_auth_core() {
        assert_eq!(KeyStore::extract_bearer("Bearer abc123"), Some("abc123"));
        assert_eq!(KeyStore::extract_bearer("Token abc123"), None);
    }
}
