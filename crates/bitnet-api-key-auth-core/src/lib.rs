//! SRP helpers for API-key authentication policy and bearer-token extraction.
//!
//! This crate keeps key management and authentication mode decisions independent
//! from any web framework so they can be reused across server and gateway layers.

use bitnet_http_auth_core::bearer_token;
use std::collections::HashMap;
use std::time::Instant;

/// Authentication result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthResult {
    Allowed,
    Denied(String),
    NoCredentials,
}

impl AuthResult {
    #[must_use]
    pub fn is_allowed(&self) -> bool {
        matches!(self, Self::Allowed)
    }
}

/// API key entry with metadata.
#[derive(Debug, Clone)]
pub struct ApiKey {
    pub key_hash: u64,
    pub name: String,
    pub created_at: Instant,
    pub usage_count: u64,
    pub enabled: bool,
}

impl ApiKey {
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

    #[must_use]
    pub fn matches(&self, key: &str) -> bool {
        self.enabled && hash_key(key) == self.key_hash
    }
}

fn hash_key(key: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in key.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Authentication configuration.
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
    #[must_use]
    pub fn new(mode: AuthMode) -> Self {
        Self { keys: HashMap::new(), mode }
    }

    #[must_use]
    pub fn disabled() -> Self {
        Self::new(AuthMode::Disabled)
    }

    pub fn add_key(&mut self, name: impl Into<String>, key: &str) {
        let name = name.into();
        let api_key = ApiKey::new(&name, key);
        self.keys.insert(name, api_key);
    }

    pub fn remove_key(&mut self, name: &str) -> bool {
        self.keys.remove(name).is_some()
    }

    pub fn disable_key(&mut self, name: &str) -> bool {
        if let Some(key) = self.keys.get_mut(name) {
            key.enabled = false;
            true
        } else {
            false
        }
    }

    pub fn enable_key(&mut self, name: &str) -> bool {
        if let Some(key) = self.keys.get_mut(name) {
            key.enabled = true;
            true
        } else {
            false
        }
    }

    #[must_use]
    pub fn key_count(&self) -> usize {
        self.keys.len()
    }

    #[must_use]
    pub fn active_key_count(&self) -> usize {
        self.keys.values().filter(|k| k.enabled).count()
    }

    #[must_use]
    pub fn mode(&self) -> AuthMode {
        self.mode
    }

    /// Authenticate a request.
    #[must_use]
    pub fn authenticate(&mut self, token: Option<&str>) -> AuthResult {
        match self.mode {
            AuthMode::Disabled => AuthResult::Allowed,
            AuthMode::Optional => {
                if let Some(tok) = token {
                    self.check_token(tok)
                } else {
                    AuthResult::Allowed
                }
            }
            AuthMode::Required => {
                if let Some(tok) = token {
                    self.check_token(tok)
                } else {
                    AuthResult::NoCredentials
                }
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

    /// Get usage stats.
    #[must_use]
    pub fn usage_stats(&self) -> Vec<(&str, u64)> {
        self.keys.iter().map(|(name, key)| (name.as_str(), key.usage_count)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disabled_allows_all() {
        let mut store = KeyStore::disabled();
        assert!(store.authenticate(None).is_allowed());
        assert!(store.authenticate(Some("anything")).is_allowed());
    }

    #[test]
    fn test_required_no_token() {
        let mut store = KeyStore::new(AuthMode::Required);
        store.add_key("test", "secret123");
        assert_eq!(store.authenticate(None), AuthResult::NoCredentials);
    }

    #[test]
    fn test_required_valid_token() {
        let mut store = KeyStore::new(AuthMode::Required);
        store.add_key("test", "secret123");
        assert!(store.authenticate(Some("secret123")).is_allowed());
    }

    #[test]
    fn test_required_invalid_token() {
        let mut store = KeyStore::new(AuthMode::Required);
        store.add_key("test", "secret123");
        let result = store.authenticate(Some("wrong"));
        assert_eq!(result, AuthResult::Denied("invalid API key".into()));
    }

    #[test]
    fn test_optional_no_token() {
        let mut store = KeyStore::new(AuthMode::Optional);
        store.add_key("test", "secret123");
        assert!(store.authenticate(None).is_allowed());
    }

    #[test]
    fn test_optional_invalid_token() {
        let mut store = KeyStore::new(AuthMode::Optional);
        store.add_key("test", "secret123");
        let result = store.authenticate(Some("wrong"));
        assert!(!result.is_allowed());
    }

    #[test]
    fn test_disable_key() {
        let mut store = KeyStore::new(AuthMode::Required);
        store.add_key("test", "secret123");
        store.disable_key("test");
        assert!(!store.authenticate(Some("secret123")).is_allowed());
        assert_eq!(store.active_key_count(), 0);
    }

    #[test]
    fn test_enable_key() {
        let mut store = KeyStore::new(AuthMode::Required);
        store.add_key("test", "secret123");
        store.disable_key("test");
        store.enable_key("test");
        assert!(store.authenticate(Some("secret123")).is_allowed());
    }

    #[test]
    fn test_usage_tracking() {
        let mut store = KeyStore::new(AuthMode::Required);
        store.add_key("test", "key1");
        let _ = store.authenticate(Some("key1"));
        let _ = store.authenticate(Some("key1"));
        let stats = store.usage_stats();
        assert_eq!(stats[0].1, 2);
    }

    #[test]
    fn test_extract_bearer() {
        assert_eq!(KeyStore::extract_bearer("Bearer abc123"), Some("abc123"));
        assert_eq!(KeyStore::extract_bearer("Token abc123"), None);
    }

    #[test]
    fn test_remove_key() {
        let mut store = KeyStore::new(AuthMode::Required);
        store.add_key("test", "key1");
        assert_eq!(store.key_count(), 1);
        store.remove_key("test");
        assert_eq!(store.key_count(), 0);
    }

    #[test]
    fn test_mode_preserved() {
        assert_eq!(KeyStore::new(AuthMode::Disabled).mode(), AuthMode::Disabled);
        assert_eq!(KeyStore::new(AuthMode::Optional).mode(), AuthMode::Optional);
        assert_eq!(KeyStore::new(AuthMode::Required).mode(), AuthMode::Required);
    }
}
