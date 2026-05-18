use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use super::util::generate_api_key;

// ── Auth provider ───────────────────────────────────────────────────────────

/// Simple in-memory API key authentication provider.
#[derive(Debug, Clone)]
pub struct AuthProvider {
    /// Set of valid API keys.
    keys: HashMap<String, ApiKeyEntry>,
}

/// Metadata for a single API key.
#[derive(Debug, Clone)]
pub struct ApiKeyEntry {
    pub key: String,
    pub name: String,
    pub created_at: u64,
    pub revoked: bool,
}

impl AuthProvider {
    pub fn new() -> Self {
        Self { keys: HashMap::new() }
    }

    /// Create and register a new API key, returning the key string.
    pub fn create_key(&mut self, name: impl Into<String>) -> String {
        let key = generate_api_key();
        #[allow(clippy::cast_possible_truncation)]
        let now =
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64;
        self.keys.insert(
            key.clone(),
            ApiKeyEntry { key: key.clone(), name: name.into(), created_at: now, revoked: false },
        );
        key
    }

    /// Insert a specific key (useful for tests).
    pub fn insert_key(&mut self, key: impl Into<String>, name: impl Into<String>) {
        let k = key.into();
        #[allow(clippy::cast_possible_truncation)]
        let now =
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64;
        self.keys.insert(
            k.clone(),
            ApiKeyEntry { key: k, name: name.into(), created_at: now, revoked: false },
        );
    }

    /// Validate that a key exists and has not been revoked.
    pub fn validate(&self, key: &str) -> bool {
        self.keys.get(key).is_some_and(|entry| !entry.revoked)
    }

    /// Revoke a key. Returns `true` if the key existed and was not already revoked.
    pub fn revoke(&mut self, key: &str) -> bool {
        if let Some(entry) = self.keys.get_mut(key)
            && !entry.revoked
        {
            entry.revoked = true;
            return true;
        }
        false
    }

    /// List all non-revoked key names.
    pub fn list_keys(&self) -> Vec<&str> {
        self.keys.values().filter(|e| !e.revoked).map(|e| e.name.as_str()).collect()
    }
}

impl Default for AuthProvider {
    fn default() -> Self {
        Self::new()
    }
}
