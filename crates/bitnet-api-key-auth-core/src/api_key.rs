use std::time::Instant;

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
