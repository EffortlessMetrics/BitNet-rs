//! Test scaffolding for TokenizerAuthority schema and validation
//!
//! Specification: docs/specs/parity-both-preflight-tokenizer.md
//! Acceptance Criteria: AC4-AC7
//!
//! Test Categories:
//! 1. TokenizerAuthority struct construction (TC1)
//! 2. TokenizerSource variants (TC2)
//! 3. SHA256 hash computation deterministic (TC3)
//! 4. Hash consistency across platforms (TC4)
//! 5. Tokenizer config serialization (TC5)
//! 6. Parity validation token sequence (TC6)
//! 7. Receipt schema v2 backward compatibility (TC7)
//! 8. Builder API patterns (TC8)
//! 9. Error handling (TC9)
//! 10. Integration with ParityReceipt (TC10)

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::Path;

// ========================================
// Data Structures (AC4)
// ========================================

/// Tokenizer authority metadata for receipt reproducibility
///
/// Tests: TC1, TC5, TC7, TC10
/// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC4
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TokenizerAuthority {
    /// Tokenizer source: "gguf_embedded" | "external" | "auto_discovered"
    pub source: TokenizerSource,

    /// Path to tokenizer (GGUF path or tokenizer.json path)
    pub path: String,

    /// SHA256 hash of tokenizer.json file (if external)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_hash: Option<String>,

    /// SHA256 hash of effective tokenizer config (canonical JSON)
    pub config_hash: String,

    /// Token count (for quick validation)
    pub token_count: usize,
}

/// Tokenizer source enum (AC5)
///
/// Tests: TC2
/// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC5
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TokenizerSource {
    /// GGUF file contains embedded tokenizer metadata
    GgufEmbedded,
    /// External tokenizer.json file explicitly provided
    External,
    /// Tokenizer auto-discovered from model directory
    AutoDiscovered,
}

/// ParityReceipt v2 with tokenizer authority (AC4, AC7)
///
/// Tests: TC7, TC10
/// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC4
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParityReceipt {
    pub version: u32,
    pub timestamp: String,
    pub model: String,
    pub backend: String,
    pub prompt: String,

    // NEW: v2 fields (optional for backward compatibility)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokenizer_authority: Option<TokenizerAuthority>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_template: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub determinism_seed: Option<u64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_sha256: Option<String>,
}

// ========================================
// Helper Functions (AC6)
// ========================================

/// Compute SHA256 hash of tokenizer.json file (AC6)
///
/// Tests: TC3, TC4
/// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC6
pub fn compute_tokenizer_file_hash(_tokenizer_path: &Path) -> Result<String> {
    // TODO: Implement file reading and SHA256 hash computation
    // This is TDD scaffolding - test will compile but fail until implemented
    unimplemented!("AC6: compute_tokenizer_file_hash - blocked by missing std::fs integration")
}

/// Compute SHA256 hash of tokenizer config (canonical JSON) (AC6)
///
/// Tests: TC3, TC4
/// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC6
pub fn compute_tokenizer_config_hash(_vocab: &serde_json::Value) -> Result<String> {
    // TODO: Implement canonical JSON serialization and SHA256 hash
    // Strategy: Sort keys, serialize to canonical JSON, hash bytes
    unimplemented!(
        "AC6: compute_tokenizer_config_hash - blocked by missing canonical JSON serialization"
    )
}

/// Validate tokenizer parity between Rust and C++ implementations (AC7)
///
/// Tests: TC6, TC9
/// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC7
pub fn validate_tokenizer_parity(
    rust_tokens: &[u32],
    cpp_tokens: &[u32],
    backend_name: &str,
) -> Result<()> {
    // Check 1: Length parity
    if rust_tokens.len() != cpp_tokens.len() {
        anyhow::bail!(
            "Tokenizer parity mismatch for {}: Rust {} tokens vs C++ {} tokens",
            backend_name,
            rust_tokens.len(),
            cpp_tokens.len()
        );
    }

    // Check 2: Token-by-token comparison
    for (i, (r_token, c_token)) in rust_tokens.iter().zip(cpp_tokens.iter()).enumerate() {
        if r_token != c_token {
            anyhow::bail!(
                "Tokenizer divergence for {} at position {}: Rust token={}, C++ token={}",
                backend_name,
                i,
                r_token,
                c_token
            );
        }
    }

    Ok(())
}

/// Validate tokenizer authority consistency between two lanes (AC7)
///
/// Tests: TC6, TC10
/// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC7
pub fn validate_tokenizer_consistency(
    lane_a: &TokenizerAuthority,
    lane_b: &TokenizerAuthority,
) -> Result<()> {
    // Config hash must match (effective tokenizer is identical)
    if lane_a.config_hash != lane_b.config_hash {
        anyhow::bail!(
            "Tokenizer config mismatch: Lane A hash={}, Lane B hash={}",
            lane_a.config_hash,
            lane_b.config_hash
        );
    }

    // Token count should match (sanity check)
    if lane_a.token_count != lane_b.token_count {
        anyhow::bail!(
            "Token count mismatch: Lane A={}, Lane B={}",
            lane_a.token_count,
            lane_b.token_count
        );
    }

    Ok(())
}

// ========================================
// Builder API (AC4, AC8)
// ========================================

impl ParityReceipt {
    /// Create new ParityReceipt with minimal fields
    ///
    /// Tests: TC8, TC10
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC8
    pub fn new(model: &str, backend: &str, prompt: &str) -> Self {
        Self {
            version: 1,
            timestamp: chrono::Utc::now().to_rfc3339(),
            model: model.to_string(),
            backend: backend.to_string(),
            prompt: prompt.to_string(),
            tokenizer_authority: None,
            prompt_template: None,
            determinism_seed: None,
            model_sha256: None,
        }
    }

    /// Set tokenizer authority metadata (AC4)
    ///
    /// Tests: TC8, TC10
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC4
    pub fn set_tokenizer_authority(&mut self, authority: TokenizerAuthority) {
        self.tokenizer_authority = Some(authority);
    }

    /// Set prompt template used (AC4)
    ///
    /// Tests: TC8, TC10
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC4
    pub fn set_prompt_template(&mut self, template: String) {
        self.prompt_template = Some(template);
    }

    /// Infer schema version based on fields present (AC7)
    ///
    /// Tests: TC7
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC7
    pub fn infer_version(&self) -> &str {
        match (&self.tokenizer_authority, &self.prompt_template) {
            (Some(_), _) | (_, Some(_)) => "2.0.0",
            _ => "1.0.0",
        }
    }
}

// ========================================
// TC1: TokenizerAuthority Struct Construction
// ========================================

#[cfg(test)]
mod tc1_tokenizer_authority_construction {
    use super::*;

    /// Test: TokenizerAuthority struct creation with all fields
    ///
    /// AC: AC4
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC4
    #[test]
    fn test_tokenizer_authority_full_construction() {
        let authority = TokenizerAuthority {
            source: TokenizerSource::External,
            path: "models/tokenizer.json".to_string(),
            file_hash: Some("abc123def456".to_string()),
            config_hash: "789ghi012jkl".to_string(),
            token_count: 128000,
        };

        assert_eq!(authority.source, TokenizerSource::External);
        assert_eq!(authority.path, "models/tokenizer.json");
        assert_eq!(authority.file_hash, Some("abc123def456".to_string()));
        assert_eq!(authority.config_hash, "789ghi012jkl");
        assert_eq!(authority.token_count, 128000);
    }

    /// Test: TokenizerAuthority with GGUF embedded source (no file hash)
    ///
    /// AC: AC4, AC5
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC5
    #[test]
    fn test_tokenizer_authority_gguf_embedded_no_file_hash() {
        let authority = TokenizerAuthority {
            source: TokenizerSource::GgufEmbedded,
            path: "models/model.gguf".to_string(),
            file_hash: None, // GGUF embedded has no separate file
            config_hash: "embedded_config_hash".to_string(),
            token_count: 32000,
        };

        assert_eq!(authority.source, TokenizerSource::GgufEmbedded);
        assert!(authority.file_hash.is_none());
        assert_eq!(authority.config_hash, "embedded_config_hash");
    }

    /// Test: TokenizerAuthority with auto-discovered source
    ///
    /// AC: AC4, AC5
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC5
    #[test]
    fn test_tokenizer_authority_auto_discovered() {
        let authority = TokenizerAuthority {
            source: TokenizerSource::AutoDiscovered,
            path: "models/auto/tokenizer.json".to_string(),
            file_hash: Some("auto_file_hash".to_string()),
            config_hash: "auto_config_hash".to_string(),
            token_count: 50000,
        };

        assert_eq!(authority.source, TokenizerSource::AutoDiscovered);
        assert_eq!(authority.file_hash, Some("auto_file_hash".to_string()));
    }

    /// Test: TokenizerAuthority minimal construction (external source)
    ///
    /// AC: AC4
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC4
    #[test]
    fn test_tokenizer_authority_minimal_external() {
        let authority = TokenizerAuthority {
            source: TokenizerSource::External,
            path: "tokenizer.json".to_string(),
            file_hash: Some("minimal_hash".to_string()),
            config_hash: "config_hash".to_string(),
            token_count: 1000,
        };

        assert_eq!(authority.token_count, 1000);
        assert_eq!(authority.config_hash, "config_hash");
    }
}

// ========================================
// TC2: TokenizerSource Enum Variants
// ========================================

#[cfg(test)]
mod tc2_tokenizer_source_variants {
    use super::*;

    /// Test: TokenizerSource GgufEmbedded variant
    ///
    /// AC: AC5
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC5
    #[test]
    fn test_tokenizer_source_gguf_embedded() {
        let source = TokenizerSource::GgufEmbedded;
        assert_eq!(source, TokenizerSource::GgufEmbedded);
    }

    /// Test: TokenizerSource External variant
    ///
    /// AC: AC5
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC5
    #[test]
    fn test_tokenizer_source_external() {
        let source = TokenizerSource::External;
        assert_eq!(source, TokenizerSource::External);
    }

    /// Test: TokenizerSource AutoDiscovered variant
    ///
    /// AC: AC5
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC5
    #[test]
    fn test_tokenizer_source_auto_discovered() {
        let source = TokenizerSource::AutoDiscovered;
        assert_eq!(source, TokenizerSource::AutoDiscovered);
    }

    /// Test: TokenizerSource serialization to JSON
    ///
    /// AC: AC5
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC5
    #[test]
    fn test_tokenizer_source_serialization() {
        let sources = vec![
            (TokenizerSource::GgufEmbedded, "\"gguf_embedded\""),
            (TokenizerSource::External, "\"external\""),
            (TokenizerSource::AutoDiscovered, "\"auto_discovered\""),
        ];

        for (source, expected_json) in sources {
            let json = serde_json::to_string(&source).unwrap();
            assert_eq!(json, expected_json);
        }
    }

    /// Test: TokenizerSource deserialization from JSON
    ///
    /// AC: AC5
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC5
    #[test]
    fn test_tokenizer_source_deserialization() {
        let cases = vec![
            ("\"gguf_embedded\"", TokenizerSource::GgufEmbedded),
            ("\"external\"", TokenizerSource::External),
            ("\"auto_discovered\"", TokenizerSource::AutoDiscovered),
        ];

        for (json, expected_source) in cases {
            let deserialized: TokenizerSource = serde_json::from_str(json).unwrap();
            assert_eq!(deserialized, expected_source);
        }
    }
}

// ========================================
// TC3: SHA256 Hash Computation Deterministic
// ========================================

#[cfg(test)]
mod tc3_sha256_hash_deterministic {
    use super::*;

    /// Test: SHA256 hash determinism (same input → same output)
    ///
    /// AC: AC6
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC6
    #[test]
    #[ignore] // Blocked by AC6 implementation: compute_tokenizer_file_hash
    fn test_file_hash_determinism() {
        let path = Path::new("tests/fixtures/tokenizer.json");

        // Compute hash twice
        let hash1 = compute_tokenizer_file_hash(path).unwrap();
        let hash2 = compute_tokenizer_file_hash(path).unwrap();

        assert_eq!(hash1, hash2, "File hash should be deterministic");
    }

    /// Test: Config hash determinism (same vocab → same hash)
    ///
    /// AC: AC6
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC6
    #[test]
    #[ignore] // Blocked by AC6 implementation: compute_tokenizer_config_hash
    fn test_config_hash_determinism() {
        let vocab = serde_json::json!({
            "token_0": 0,
            "token_1": 1,
            "token_2": 2,
        });

        let hash1 = compute_tokenizer_config_hash(&vocab).unwrap();
        let hash2 = compute_tokenizer_config_hash(&vocab).unwrap();

        assert_eq!(hash1, hash2, "Config hash should be deterministic");
    }

    /// Test: SHA256 hash format (64 hex characters)
    ///
    /// AC: AC6
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC6
    #[test]
    #[ignore] // Blocked by AC6 implementation: compute_tokenizer_file_hash
    fn test_hash_format_64_hex_chars() {
        let path = Path::new("tests/fixtures/tokenizer.json");
        let hash = compute_tokenizer_file_hash(path).unwrap();

        assert_eq!(hash.len(), 64, "SHA256 hash should be 64 hex characters");
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()), "Hash should only contain hex digits");
    }

    /// Test: Hash computation with empty vocab
    ///
    /// AC: AC6
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC6
    #[test]
    #[ignore] // Blocked by AC6 implementation: compute_tokenizer_config_hash
    fn test_config_hash_empty_vocab() {
        let empty_vocab = serde_json::json!({});
        let hash = compute_tokenizer_config_hash(&empty_vocab).unwrap();

        assert!(!hash.is_empty(), "Empty vocab should still produce hash");
        assert_eq!(hash.len(), 64);
    }

    /// Test: Hash computation with large vocab
    ///
    /// AC: AC6
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC6
    #[test]
    #[ignore] // Blocked by AC6 implementation: compute_tokenizer_config_hash
    fn test_config_hash_large_vocab() {
        // Simulate large vocab (128k tokens)
        let mut vocab = serde_json::Map::new();
        for i in 0..128000 {
            vocab.insert(format!("token_{}", i), serde_json::json!(i));
        }
        let large_vocab = serde_json::Value::Object(vocab);

        let hash = compute_tokenizer_config_hash(&large_vocab).unwrap();
        assert_eq!(hash.len(), 64);
    }
}

// ========================================
// TC4: Hash Consistency Across Platforms
// ========================================

#[cfg(test)]
mod tc4_hash_consistency_platforms {
    use super::*;

    /// Test: Config hash invariant to key order (canonical JSON)
    ///
    /// AC: AC6
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC6
    #[test]
    #[ignore] // Blocked by AC6 implementation: compute_tokenizer_config_hash
    fn test_config_hash_invariant_to_key_order() {
        let vocab1 = serde_json::json!({
            "a": 0,
            "b": 1,
            "c": 2,
        });

        let vocab2 = serde_json::json!({
            "c": 2,
            "a": 0,
            "b": 1,
        });

        let hash1 = compute_tokenizer_config_hash(&vocab1).unwrap();
        let hash2 = compute_tokenizer_config_hash(&vocab2).unwrap();

        assert_eq!(hash1, hash2, "Config hash should be invariant to key order (canonical JSON)");
    }

    /// Test: File hash binary consistency
    ///
    /// AC: AC6
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC6
    #[test]
    #[ignore] // Blocked by AC6 implementation: compute_tokenizer_file_hash
    fn test_file_hash_binary_consistency() {
        let path = Path::new("tests/fixtures/tokenizer.json");

        // Compute hash multiple times in loop (simulate different runtime conditions)
        let hashes: Vec<String> =
            (0..10).map(|_| compute_tokenizer_file_hash(path).unwrap()).collect();

        // All hashes should be identical
        assert!(hashes.windows(2).all(|w| w[0] == w[1]));
    }

    /// Property-based test: Config hash determinism with random vocabs
    ///
    /// AC: AC6
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC6
    #[test]
    #[ignore] // Blocked by AC6 implementation: compute_tokenizer_config_hash
    fn test_config_hash_property_based_determinism() {
        // TODO: Implement property-based test with proptest
        // Strategy: Generate random vocabs, verify hash consistency
        unimplemented!("Property-based test for config hash determinism - requires proptest setup")
    }

    /// Parametric test: Hash length invariant across different vocab sizes
    ///
    /// AC: AC6
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC6
    #[test]
    #[ignore] // Blocked by AC6 implementation: compute_tokenizer_config_hash
    fn test_config_hash_length_invariant_parametric() {
        // Test with multiple vocab sizes
        let sizes = vec![1, 10, 100, 1000, 10000, 128000];

        for token_count in sizes {
            let mut vocab = serde_json::Map::new();
            for i in 0..token_count {
                vocab.insert(format!("token_{}", i), serde_json::json!(i));
            }
            let vocab_json = serde_json::Value::Object(vocab);

            let hash = compute_tokenizer_config_hash(&vocab_json).unwrap();
            assert_eq!(hash.len(), 64, "Hash should be 64 chars for vocab size {}", token_count);
        }
    }
}

// ========================================
// TC5: Tokenizer Config Serialization
// ========================================

#[cfg(test)]
mod tc5_tokenizer_config_serialization {
    use super::*;

    /// Test: TokenizerAuthority serialization to JSON
    ///
    /// AC: AC4
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC4
    #[test]
    fn test_tokenizer_authority_serialization() {
        let authority = TokenizerAuthority {
            source: TokenizerSource::External,
            path: "tokenizer.json".to_string(),
            file_hash: Some("abc123".to_string()),
            config_hash: "def456".to_string(),
            token_count: 5,
        };

        let json = serde_json::to_string(&authority).unwrap();
        assert!(json.contains("\"source\":\"external\""));
        assert!(json.contains("\"path\":\"tokenizer.json\""));
        assert!(json.contains("\"file_hash\":\"abc123\""));
        assert!(json.contains("\"config_hash\":\"def456\""));
        assert!(json.contains("\"token_count\":5"));
    }

    /// Test: TokenizerAuthority deserialization from JSON
    ///
    /// AC: AC4
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC4
    #[test]
    fn test_tokenizer_authority_deserialization() {
        let json = r#"{
            "source": "external",
            "path": "tokenizer.json",
            "file_hash": "abc123",
            "config_hash": "def456",
            "token_count": 5
        }"#;

        let authority: TokenizerAuthority = serde_json::from_str(json).unwrap();
        assert_eq!(authority.source, TokenizerSource::External);
        assert_eq!(authority.path, "tokenizer.json");
        assert_eq!(authority.file_hash, Some("abc123".to_string()));
        assert_eq!(authority.config_hash, "def456");
        assert_eq!(authority.token_count, 5);
    }

    /// Test: TokenizerAuthority round-trip serialization
    ///
    /// AC: AC4
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC4
    #[test]
    fn test_tokenizer_authority_round_trip() {
        let original = TokenizerAuthority {
            source: TokenizerSource::GgufEmbedded,
            path: "model.gguf".to_string(),
            file_hash: None,
            config_hash: "embedded_hash".to_string(),
            token_count: 32000,
        };

        let json = serde_json::to_string(&original).unwrap();
        let deserialized: TokenizerAuthority = serde_json::from_str(&json).unwrap();

        assert_eq!(original, deserialized);
    }

    /// Test: TokenizerAuthority skip serializing None file_hash
    ///
    /// AC: AC4
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC4
    #[test]
    fn test_tokenizer_authority_skip_none_file_hash() {
        let authority = TokenizerAuthority {
            source: TokenizerSource::GgufEmbedded,
            path: "model.gguf".to_string(),
            file_hash: None,
            config_hash: "hash".to_string(),
            token_count: 1000,
        };

        let json = serde_json::to_string(&authority).unwrap();
        assert!(!json.contains("\"file_hash\""), "None file_hash should be skipped");
    }

    /// Test: TokenizerAuthority pretty-print JSON
    ///
    /// AC: AC4
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC4
    #[test]
    fn test_tokenizer_authority_pretty_print() {
        let authority = TokenizerAuthority {
            source: TokenizerSource::External,
            path: "tokenizer.json".to_string(),
            file_hash: Some("hash123".to_string()),
            config_hash: "config_hash".to_string(),
            token_count: 128000,
        };

        let json = serde_json::to_string_pretty(&authority).unwrap();
        assert!(json.contains("  \"source\": \"external\""));
        assert!(json.contains("\n"));
    }
}

// ========================================
// TC6: Parity Validation Token Sequence
// ========================================

#[cfg(test)]
mod tc6_parity_validation_token_sequence {
    use super::*;

    /// Test: Tokenizer parity with identical tokens
    ///
    /// AC: AC7
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC7
    #[test]
    fn test_tokenizer_parity_identical_tokens() {
        let rust = vec![1, 2, 3, 4, 5];
        let cpp = vec![1, 2, 3, 4, 5];

        let result = validate_tokenizer_parity(&rust, &cpp, "bitnet");
        assert!(result.is_ok());
    }

    /// Test: Tokenizer parity length mismatch
    ///
    /// AC: AC7
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC7
    #[test]
    fn test_tokenizer_parity_length_mismatch() {
        let rust = vec![1, 2, 3, 4, 5];
        let cpp = vec![1, 2, 3, 4];

        let result = validate_tokenizer_parity(&rust, &cpp, "bitnet");
        assert!(result.is_err());

        let err = result.unwrap_err();
        let err_msg = err.to_string();
        assert!(err_msg.contains("5 tokens vs"));
        assert!(err_msg.contains("4 tokens"));
        assert!(err_msg.contains("bitnet"));
    }

    /// Test: Tokenizer parity token divergence at position
    ///
    /// AC: AC7
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC7
    #[test]
    fn test_tokenizer_parity_token_divergence() {
        let rust = vec![1, 2, 3, 4, 5];
        let cpp = vec![1, 2, 99, 4, 5]; // Divergence at position 2

        let result = validate_tokenizer_parity(&rust, &cpp, "bitnet");
        assert!(result.is_err());

        let err = result.unwrap_err();
        let err_msg = err.to_string();
        assert!(err_msg.contains("position 2"));
        assert!(err_msg.contains("Rust token=3"));
        assert!(err_msg.contains("C++ token=99"));
    }

    /// Test: Tokenizer parity empty token sequences
    ///
    /// AC: AC7
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC7
    #[test]
    fn test_tokenizer_parity_empty_sequences() {
        let rust: Vec<u32> = vec![];
        let cpp: Vec<u32> = vec![];

        let result = validate_tokenizer_parity(&rust, &cpp, "llama");
        assert!(result.is_ok());
    }

    /// Test: Tokenizer parity large token sequences
    ///
    /// AC: AC7
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC7
    #[test]
    fn test_tokenizer_parity_large_sequences() {
        let rust: Vec<u32> = (0..10000).collect();
        let cpp: Vec<u32> = (0..10000).collect();

        let result = validate_tokenizer_parity(&rust, &cpp, "llama");
        assert!(result.is_ok());
    }

    /// Test: Tokenizer parity divergence at first position
    ///
    /// AC: AC7
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC7
    #[test]
    fn test_tokenizer_parity_divergence_first_position() {
        let rust = vec![1, 2, 3];
        let cpp = vec![99, 2, 3];

        let result = validate_tokenizer_parity(&rust, &cpp, "bitnet");
        assert!(result.is_err());

        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("position 0"));
    }

    /// Test: Tokenizer parity divergence at last position
    ///
    /// AC: AC7
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC7
    #[test]
    fn test_tokenizer_parity_divergence_last_position() {
        let rust = vec![1, 2, 3, 4, 5];
        let cpp = vec![1, 2, 3, 4, 99];

        let result = validate_tokenizer_parity(&rust, &cpp, "llama");
        assert!(result.is_err());

        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("position 4"));
    }
}

// ========================================
// TC7: Receipt Schema v2 Backward Compatibility
// ========================================

#[cfg(test)]
mod tc7_receipt_schema_v2_backward_compat {
    use super::*;

    /// Test: ParityReceipt v1 deserialization (no tokenizer_authority)
    ///
    /// AC: AC4, AC7
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC4
    #[test]
    fn test_parity_receipt_v1_deserialization() {
        let json_v1 = r#"{
            "version": 1,
            "timestamp": "2025-10-26T14:30:00Z",
            "model": "model.gguf",
            "backend": "bitnet",
            "prompt": "test"
        }"#;

        let receipt: ParityReceipt = serde_json::from_str(json_v1).unwrap();
        assert!(receipt.tokenizer_authority.is_none());
        assert!(receipt.prompt_template.is_none());
        assert!(receipt.determinism_seed.is_none());
        assert!(receipt.model_sha256.is_none());
    }

    /// Test: ParityReceipt v2 serialization with tokenizer_authority
    ///
    /// AC: AC4, AC7
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC4
    #[test]
    fn test_parity_receipt_v2_serialization() {
        let mut receipt = ParityReceipt::new("model.gguf", "bitnet", "What is 2+2?");

        receipt.set_tokenizer_authority(TokenizerAuthority {
            source: TokenizerSource::External,
            path: "tokenizer.json".to_string(),
            file_hash: Some("hash123".to_string()),
            config_hash: "hash456".to_string(),
            token_count: 5,
        });

        let json = serde_json::to_string_pretty(&receipt).unwrap();
        assert!(json.contains("tokenizer_authority"));
        assert!(json.contains("hash123"));
    }

    /// Test: ParityReceipt infer_version v1
    ///
    /// AC: AC7
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC7
    #[test]
    fn test_parity_receipt_infer_version_v1() {
        let receipt = ParityReceipt::new("model.gguf", "bitnet", "test");
        assert_eq!(receipt.infer_version(), "1.0.0");
    }

    /// Test: ParityReceipt infer_version v2 (with tokenizer_authority)
    ///
    /// AC: AC7
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC7
    #[test]
    fn test_parity_receipt_infer_version_v2_tokenizer() {
        let mut receipt = ParityReceipt::new("model.gguf", "bitnet", "test");
        receipt.set_tokenizer_authority(TokenizerAuthority {
            source: TokenizerSource::External,
            path: "tokenizer.json".to_string(),
            file_hash: None,
            config_hash: "hash".to_string(),
            token_count: 1,
        });

        assert_eq!(receipt.infer_version(), "2.0.0");
    }

    /// Test: ParityReceipt infer_version v2 (with prompt_template)
    ///
    /// AC: AC7
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC7
    #[test]
    fn test_parity_receipt_infer_version_v2_template() {
        let mut receipt = ParityReceipt::new("model.gguf", "bitnet", "test");
        receipt.set_prompt_template("instruct".to_string());

        assert_eq!(receipt.infer_version(), "2.0.0");
    }

    /// Test: ParityReceipt skip_serializing_if for optional fields
    ///
    /// AC: AC4, AC7
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC4
    #[test]
    fn test_parity_receipt_skip_none_fields() {
        let receipt = ParityReceipt::new("model.gguf", "bitnet", "test");
        let json = serde_json::to_string(&receipt).unwrap();

        assert!(!json.contains("\"tokenizer_authority\""));
        assert!(!json.contains("\"prompt_template\""));
        assert!(!json.contains("\"determinism_seed\""));
        assert!(!json.contains("\"model_sha256\""));
    }
}

// ========================================
// TC8: Builder API Patterns
// ========================================

#[cfg(test)]
mod tc8_builder_api_patterns {
    use super::*;

    /// Test: ParityReceipt builder with tokenizer authority
    ///
    /// AC: AC4, AC8
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC8
    #[test]
    fn test_parity_receipt_builder_with_tokenizer_authority() {
        let mut receipt = ParityReceipt::new("model.gguf", "bitnet", "What is 2+2?");

        receipt.set_tokenizer_authority(TokenizerAuthority {
            source: TokenizerSource::External,
            path: "tokenizer.json".to_string(),
            file_hash: Some("file_hash".to_string()),
            config_hash: "config_hash".to_string(),
            token_count: 5,
        });

        assert!(receipt.tokenizer_authority.is_some());
        let authority = receipt.tokenizer_authority.unwrap();
        assert_eq!(authority.source, TokenizerSource::External);
        assert_eq!(authority.token_count, 5);
    }

    /// Test: ParityReceipt builder with prompt template
    ///
    /// AC: AC4, AC8
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC8
    #[test]
    fn test_parity_receipt_builder_with_prompt_template() {
        let mut receipt = ParityReceipt::new("model.gguf", "bitnet", "test");
        receipt.set_prompt_template("instruct".to_string());

        assert_eq!(receipt.prompt_template, Some("instruct".to_string()));
    }

    /// Test: ParityReceipt builder chaining
    ///
    /// AC: AC8
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC8
    #[test]
    fn test_parity_receipt_builder_chaining() {
        let mut receipt = ParityReceipt::new("model.gguf", "bitnet", "test");

        receipt.set_tokenizer_authority(TokenizerAuthority {
            source: TokenizerSource::External,
            path: "tokenizer.json".to_string(),
            file_hash: Some("hash".to_string()),
            config_hash: "config".to_string(),
            token_count: 1,
        });
        receipt.set_prompt_template("llama3-chat".to_string());
        receipt.determinism_seed = Some(42);
        receipt.model_sha256 = Some("model_hash".to_string());

        assert!(receipt.tokenizer_authority.is_some());
        assert_eq!(receipt.prompt_template, Some("llama3-chat".to_string()));
        assert_eq!(receipt.determinism_seed, Some(42));
        assert_eq!(receipt.model_sha256, Some("model_hash".to_string()));
    }

    /// Test: ParityReceipt new creates valid timestamp
    ///
    /// AC: AC8
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC8
    #[test]
    fn test_parity_receipt_new_creates_valid_timestamp() {
        let receipt = ParityReceipt::new("model.gguf", "bitnet", "test");

        // Verify timestamp is RFC3339 format (ISO 8601)
        assert!(!receipt.timestamp.is_empty(), "Timestamp should not be empty");
        assert!(receipt.timestamp.contains("T"), "Timestamp should contain 'T' separator");
        // Check for either Z (UTC) or +/- offset
        assert!(
            receipt.timestamp.contains("Z")
                || receipt.timestamp.contains("+")
                || receipt.timestamp.contains("-"),
            "Timestamp should have timezone indicator (Z, +, or -)"
        );
    }
}

// ========================================
// TC9: Error Handling
// ========================================

#[cfg(test)]
mod tc9_error_handling {
    use super::*;

    /// Test: compute_tokenizer_file_hash with missing file
    ///
    /// AC: AC6
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC6
    #[test]
    #[ignore] // Blocked by AC6 implementation: compute_tokenizer_file_hash
    fn test_file_hash_missing_file_error() {
        let path = Path::new("nonexistent/tokenizer.json");
        let result = compute_tokenizer_file_hash(path);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Failed to read"));
    }

    /// Test: validate_tokenizer_consistency with hash mismatch
    ///
    /// AC: AC7
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC7
    #[test]
    fn test_tokenizer_consistency_hash_mismatch() {
        let lane_a = TokenizerAuthority {
            source: TokenizerSource::External,
            path: "tokenizer.json".to_string(),
            file_hash: Some("hash_a".to_string()),
            config_hash: "config_hash_a".to_string(),
            token_count: 5,
        };

        let lane_b = TokenizerAuthority {
            source: TokenizerSource::External,
            path: "tokenizer.json".to_string(),
            file_hash: Some("hash_b".to_string()),
            config_hash: "config_hash_b".to_string(), // Different config hash
            token_count: 5,
        };

        let result = validate_tokenizer_consistency(&lane_a, &lane_b);
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(err.to_string().contains("Tokenizer config mismatch"));
        assert!(err.to_string().contains("config_hash_a"));
        assert!(err.to_string().contains("config_hash_b"));
    }

    /// Test: validate_tokenizer_consistency with token count mismatch
    ///
    /// AC: AC7
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC7
    #[test]
    fn test_tokenizer_consistency_token_count_mismatch() {
        let lane_a = TokenizerAuthority {
            source: TokenizerSource::External,
            path: "tokenizer.json".to_string(),
            file_hash: Some("hash".to_string()),
            config_hash: "config_hash".to_string(),
            token_count: 5,
        };

        let lane_b = TokenizerAuthority {
            source: TokenizerSource::External,
            path: "tokenizer.json".to_string(),
            file_hash: Some("hash".to_string()),
            config_hash: "config_hash".to_string(),
            token_count: 10, // Different token count
        };

        let result = validate_tokenizer_consistency(&lane_a, &lane_b);
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(err.to_string().contains("Token count mismatch"));
        assert!(err.to_string().contains("Lane A=5"));
        assert!(err.to_string().contains("Lane B=10"));
    }

    /// Test: validate_tokenizer_consistency with identical authorities
    ///
    /// AC: AC7
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC7
    #[test]
    fn test_tokenizer_consistency_identical_authorities() {
        let lane_a = TokenizerAuthority {
            source: TokenizerSource::External,
            path: "tokenizer.json".to_string(),
            file_hash: Some("hash".to_string()),
            config_hash: "config_hash".to_string(),
            token_count: 5,
        };

        let lane_b = lane_a.clone();

        let result = validate_tokenizer_consistency(&lane_a, &lane_b);
        assert!(result.is_ok());
    }

    /// Test: validate_tokenizer_parity with backend name in error
    ///
    /// AC: AC7
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC7
    #[test]
    fn test_tokenizer_parity_backend_name_in_error() {
        let rust = vec![1, 2, 3];
        let cpp = vec![1, 2];

        let result = validate_tokenizer_parity(&rust, &cpp, "llama");
        assert!(result.is_err());

        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("llama"));
    }
}

// ========================================
// TC10: Integration with ParityReceipt
// ========================================

#[cfg(test)]
mod tc10_integration_parity_receipt {
    use super::*;

    /// Test: ParityReceipt full integration with all v2 fields
    ///
    /// AC: AC4, AC7, AC8
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC8
    #[test]
    fn test_parity_receipt_full_v2_integration() {
        let mut receipt = ParityReceipt::new(
            "models/microsoft-bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf",
            "bitnet",
            "[INST] What is 2+2? [/INST]",
        );

        receipt.set_tokenizer_authority(TokenizerAuthority {
            source: TokenizerSource::External,
            path: "models/tokenizer.json".to_string(),
            file_hash: Some("6f3ef9d7a3c2b1e0".to_string()),
            config_hash: "a1b2c3d4e5f6789".to_string(),
            token_count: 4,
        });

        receipt.set_prompt_template("instruct".to_string());
        receipt.determinism_seed = Some(42);
        receipt.model_sha256 =
            Some("fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210".to_string());

        // Verify all v2 fields populated
        assert!(receipt.tokenizer_authority.is_some());
        assert_eq!(receipt.prompt_template, Some("instruct".to_string()));
        assert_eq!(receipt.determinism_seed, Some(42));
        assert!(receipt.model_sha256.is_some());
        assert_eq!(receipt.infer_version(), "2.0.0");
    }

    /// Test: ParityReceipt serialization produces valid JSON
    ///
    /// AC: AC4, AC8
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC8
    #[test]
    fn test_parity_receipt_serialization_valid_json() {
        let mut receipt = ParityReceipt::new("model.gguf", "bitnet", "test");

        receipt.set_tokenizer_authority(TokenizerAuthority {
            source: TokenizerSource::External,
            path: "tokenizer.json".to_string(),
            file_hash: Some("hash".to_string()),
            config_hash: "config".to_string(),
            token_count: 1,
        });

        let json = serde_json::to_string_pretty(&receipt).unwrap();

        // Verify valid JSON
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed["tokenizer_authority"].is_object());
        assert_eq!(parsed["tokenizer_authority"]["source"], "external");
    }

    /// Test: ParityReceipt round-trip with tokenizer authority
    ///
    /// AC: AC4, AC8
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC8
    #[test]
    fn test_parity_receipt_round_trip_with_tokenizer() {
        let mut original = ParityReceipt::new("model.gguf", "bitnet", "test");

        original.set_tokenizer_authority(TokenizerAuthority {
            source: TokenizerSource::AutoDiscovered,
            path: "auto/tokenizer.json".to_string(),
            file_hash: Some("auto_hash".to_string()),
            config_hash: "auto_config".to_string(),
            token_count: 128000,
        });

        let json = serde_json::to_string(&original).unwrap();
        let deserialized: ParityReceipt = serde_json::from_str(&json).unwrap();

        assert_eq!(
            original.tokenizer_authority.unwrap(),
            deserialized.tokenizer_authority.unwrap()
        );
    }

    /// Test: ParityReceipt with GGUF embedded tokenizer (no file_hash)
    ///
    /// AC: AC4, AC5
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC5
    #[test]
    fn test_parity_receipt_gguf_embedded_no_file_hash() {
        let mut receipt = ParityReceipt::new("model.gguf", "bitnet", "test");

        receipt.set_tokenizer_authority(TokenizerAuthority {
            source: TokenizerSource::GgufEmbedded,
            path: "model.gguf".to_string(),
            file_hash: None, // GGUF embedded has no separate file
            config_hash: "embedded_config".to_string(),
            token_count: 32000,
        });

        let json = serde_json::to_string(&receipt).unwrap();
        assert!(!json.contains("\"file_hash\""));
    }

    /// Test: Multiple receipts with identical tokenizer authority
    ///
    /// AC: AC7
    /// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC7
    #[test]
    fn test_multiple_receipts_identical_tokenizer_authority() {
        let tokenizer_authority = TokenizerAuthority {
            source: TokenizerSource::External,
            path: "tokenizer.json".to_string(),
            file_hash: Some("hash".to_string()),
            config_hash: "config_hash".to_string(),
            token_count: 5,
        };

        let mut receipt_bitnet = ParityReceipt::new("model.gguf", "bitnet", "test");
        receipt_bitnet.set_tokenizer_authority(tokenizer_authority.clone());

        let mut receipt_llama = ParityReceipt::new("model.gguf", "llama", "test");
        receipt_llama.set_tokenizer_authority(tokenizer_authority.clone());

        // Verify consistency
        let result = validate_tokenizer_consistency(
            &receipt_bitnet.tokenizer_authority.unwrap(),
            &receipt_llama.tokenizer_authority.unwrap(),
        );
        assert!(result.is_ok());
    }
}
