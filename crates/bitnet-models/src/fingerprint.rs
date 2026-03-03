//! Model fingerprinting.
//!
//! Generate unique fingerprints for model identification and caching.

use sha2::{Digest, Sha256};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Model fingerprint components.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct FingerprintInput {
    pub architecture: String,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub quant_type: String,
}

/// Computed fingerprint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelFingerprint {
    pub hash: u64,
    pub hex: String,
    pub short: String,
}

impl ModelFingerprint {
    pub fn matches(&self, other: &ModelFingerprint) -> bool {
        self.hash == other.hash
    }
}

/// Compute a fingerprint from model properties.
pub fn compute_fingerprint(input: &FingerprintInput) -> ModelFingerprint {
    let mut hasher = DefaultHasher::new();
    input.hash(&mut hasher);
    let hash = hasher.finish();
    let hex = format!("{hash:016x}");
    let short = hex[..8].to_string();
    ModelFingerprint { hash, hex, short }
}

/// Compute a fingerprint from raw bytes (e.g., first N bytes of a file).
pub fn fingerprint_bytes(data: &[u8]) -> ModelFingerprint {
    let mut hasher = DefaultHasher::new();
    data.hash(&mut hasher);
    let hash = hasher.finish();
    let hex = format!("{hash:016x}");
    let short = hex[..8].to_string();
    ModelFingerprint { hash, hex, short }
}

/// Fingerprint cache entry.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub fingerprint: ModelFingerprint,
    pub model_path: String,
    pub model_name: String,
    pub file_size: u64,
}

/// Fingerprint cache for quick model lookup.
#[derive(Debug, Clone, Default)]
pub struct FingerprintCache {
    entries: Vec<CacheEntry>,
}

impl FingerprintCache {
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    pub fn add(&mut self, entry: CacheEntry) {
        self.entries.push(entry);
    }

    pub fn find(&self, fp: &ModelFingerprint) -> Option<&CacheEntry> {
        self.entries.iter().find(|e| e.fingerprint.matches(fp))
    }

    pub fn find_by_name(&self, name: &str) -> Option<&CacheEntry> {
        self.entries.iter().find(|e| e.model_name == name)
    }

    pub fn count(&self) -> usize {
        self.entries.len()
    }

    pub fn remove(&mut self, fp: &ModelFingerprint) -> bool {
        let before = self.entries.len();
        self.entries.retain(|e| !e.fingerprint.matches(fp));
        self.entries.len() < before
    }

    pub fn all(&self) -> &[CacheEntry] {
        &self.entries
    }
}

/// Quick fingerprint from model config values.
pub fn quick_fingerprint(
    arch: &str,
    layers: usize,
    hidden: usize,
    vocab: usize,
) -> ModelFingerprint {
    compute_fingerprint(&FingerprintInput {
        architecture: arch.to_string(),
        num_layers: layers,
        hidden_size: hidden,
        vocab_size: vocab,
        quant_type: String::new(),
    })
}

// Legacy SHA256 fingerprinting used by GGUF loader for policy matching.

/// Compute SHA256 fingerprint of a GGUF file (returns `"sha256-<hex>"`).
pub fn compute_gguf_fingerprint(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    format!("sha256-{:x}", result)
}

/// Format raw hash bytes as a `"sha256-<hex>"` string.
pub fn format_fingerprint(hash_bytes: &[u8]) -> String {
    format!("sha256-{}", hex::encode(hash_bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_input() -> FingerprintInput {
        FingerprintInput {
            architecture: "PhiForCausalLM".into(),
            num_layers: 40,
            hidden_size: 5120,
            vocab_size: 100352,
            quant_type: "bf16".into(),
        }
    }

    #[test]
    fn test_deterministic() {
        let a = compute_fingerprint(&sample_input());
        let b = compute_fingerprint(&sample_input());
        assert_eq!(a, b);
    }

    #[test]
    fn test_different_inputs() {
        let mut input = sample_input();
        let a = compute_fingerprint(&input);
        input.num_layers = 32;
        let b = compute_fingerprint(&input);
        assert_ne!(a, b);
    }

    #[test]
    fn test_hex_length() {
        let fp = compute_fingerprint(&sample_input());
        assert_eq!(fp.hex.len(), 16);
        assert_eq!(fp.short.len(), 8);
    }

    #[test]
    fn test_matches() {
        let a = compute_fingerprint(&sample_input());
        let b = compute_fingerprint(&sample_input());
        assert!(a.matches(&b));
    }

    #[test]
    fn test_fingerprint_bytes() {
        let a = fingerprint_bytes(b"hello world");
        let b = fingerprint_bytes(b"hello world");
        assert_eq!(a, b);
    }

    #[test]
    fn test_fingerprint_bytes_different() {
        let a = fingerprint_bytes(b"hello");
        let b = fingerprint_bytes(b"world");
        assert_ne!(a, b);
    }

    #[test]
    fn test_cache_add_find() {
        let mut cache = FingerprintCache::new();
        let fp = compute_fingerprint(&sample_input());
        cache.add(CacheEntry {
            fingerprint: fp.clone(),
            model_path: "/models/phi4.gguf".into(),
            model_name: "phi-4".into(),
            file_size: 28_000_000_000,
        });
        assert_eq!(cache.count(), 1);
        assert!(cache.find(&fp).is_some());
    }

    #[test]
    fn test_cache_find_by_name() {
        let mut cache = FingerprintCache::new();
        let fp = compute_fingerprint(&sample_input());
        cache.add(CacheEntry {
            fingerprint: fp,
            model_path: "/test".into(),
            model_name: "phi-4".into(),
            file_size: 100,
        });
        assert!(cache.find_by_name("phi-4").is_some());
        assert!(cache.find_by_name("llama").is_none());
    }

    #[test]
    fn test_cache_remove() {
        let mut cache = FingerprintCache::new();
        let fp = compute_fingerprint(&sample_input());
        cache.add(CacheEntry {
            fingerprint: fp.clone(),
            model_path: "/test".into(),
            model_name: "test".into(),
            file_size: 100,
        });
        assert!(cache.remove(&fp));
        assert_eq!(cache.count(), 0);
    }

    #[test]
    fn test_quick_fingerprint() {
        let fp = quick_fingerprint("Phi", 40, 5120, 100352);
        assert!(!fp.hex.is_empty());
    }

    #[test]
    fn test_empty_cache() {
        let cache = FingerprintCache::new();
        assert_eq!(cache.count(), 0);
        let fp = compute_fingerprint(&sample_input());
        assert!(cache.find(&fp).is_none());
    }

    #[test]
    fn test_cache_all() {
        let mut cache = FingerprintCache::new();
        cache.add(CacheEntry {
            fingerprint: compute_fingerprint(&sample_input()),
            model_path: "/a".into(),
            model_name: "a".into(),
            file_size: 1,
        });
        assert_eq!(cache.all().len(), 1);
    }
}
