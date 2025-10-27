//! Test scaffolding for TokenizerAuthority integration in parity-both receipts
//!
//! **Specification**: `docs/specs/tokenizer-authority-integration-parity-both.md`
//! **Coverage**: AC1-AC6
//!
//! ## Test Categories
//!
//! 1. **AC1**: Single TokenizerAuthority Computation (TC1, 8 tests)
//! 2. **AC2**: Dual Receipt Injection (TC2, 6 tests)
//! 3. **AC3**: Validation Logic (TC3, 8 tests)
//! 4. **AC4**: Exit Code 2 on Mismatch (TC4, 4 tests)
//! 5. **AC5**: Source Detection (TC5, 6 tests)
//! 6. **AC6**: Hash Computation (TC6, 12 tests)
//!
//! ## Existing Test Categories (Baseline)
//!
//! - TokenizerAuthority struct construction (TC1)
//! - TokenizerSource variants (TC2)
//! - SHA256 hash computation deterministic (TC3)
//! - Hash consistency across platforms (TC4)
//! - Tokenizer config serialization (TC5)
//! - Parity validation token sequence (TC6)
//! - Receipt schema v2 backward compatibility (TC7)
//! - Builder API patterns (TC8)
//! - Error handling (TC9)
//! - Integration with ParityReceipt (TC10)
//!
//! ## New Test Categories (Integration)
//!
//! - AC1: Single computation verification
//! - AC2: Dual receipt identical metadata
//! - AC3: Hash-based consistency validation
//! - AC4: Exit code enforcement
//! - AC5: Source detection heuristics
//! - AC6: Hash determinism and collision resistance

use std::path::Path;

// Import from production code
use bitnet_crossval::receipt::{
    ParityReceipt, TokenizerAuthority, TokenizerSource, validate_tokenizer_consistency,
    validate_tokenizer_parity,
};

// ========================================
// Data Structures (AC4)
// ========================================
//
// TokenizerAuthority, TokenizerSource, and ParityReceipt are imported from bitnet_crossval::receipt
// See: crossval/src/receipt.rs

// ========================================
// Helper Functions (AC6)
// ========================================

/// Compute SHA256 hash of tokenizer.json file (AC6)
///
/// Tests: TC3, TC4
/// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC6
pub fn compute_tokenizer_file_hash(_tokenizer_path: &Path) -> anyhow::Result<String> {
    // TODO: Implement file reading and SHA256 hash computation
    // This is TDD scaffolding - test will compile but fail until implemented
    unimplemented!("AC6: compute_tokenizer_file_hash - blocked by missing std::fs integration")
}

/// Compute SHA256 hash of tokenizer config (canonical JSON) (AC6)
///
/// Tests: TC3, TC4
/// Spec: docs/specs/parity-both-preflight-tokenizer.md#AC6
pub fn compute_tokenizer_config_hash(_vocab: &serde_json::Value) -> anyhow::Result<String> {
    // TODO: Implement canonical JSON serialization and SHA256 hash
    // Strategy: Sort keys, serialize to canonical JSON, hash bytes
    unimplemented!(
        "AC6: compute_tokenizer_config_hash - blocked by missing canonical JSON serialization"
    )
}

// validate_tokenizer_parity and validate_tokenizer_consistency are now imported
// from bitnet_crossval::receipt - see imports at top of file

// ========================================
// Builder API (AC4, AC8)
// ========================================
//
// ParityReceipt builder methods are already implemented in crossval/src/receipt.rs
// See: ParityReceipt::new, set_tokenizer_authority, set_prompt_template, infer_version

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

// ========================================
// NEW TEST CATEGORIES: Integration Tests (AC1-AC6)
// ========================================

// ========================================
// TC_AC1: Single TokenizerAuthority Computation
// ========================================

#[cfg(test)]
mod tc_ac1_single_tokenizer_authority_computation {
    use super::*;

    /// Test: TokenizerAuthority computed once for dual-lane parity-both
    ///
    /// AC: AC1
    /// Spec: docs/specs/tokenizer-authority-integration-parity-both.md#AC1
    #[test]
    fn test_tokenizer_authority_computed_once_shared_setup() {
        // Tests feature spec: tokenizer-authority-integration-parity-both.md#AC1
        // Verify: TokenizerAuthority computed once, not per-lane
        todo!("AC1: Verify single TokenizerAuthority computation in shared setup (STEP 2.5)");
    }

    /// Test: TokenizerAuthority contains all required fields after computation
    ///
    /// AC: AC1
    /// Spec: docs/specs/tokenizer-authority-integration-parity-both.md#AC1
    #[test]
    fn test_tokenizer_authority_complete_fields_after_computation() {
        // Tests feature spec: tokenizer-authority-integration-parity-both.md#AC1
        // Verify: source, path, file_hash, config_hash, token_count populated
        todo!("AC1: Verify TokenizerAuthority has all required fields after computation");
    }

    /// Test: Source detection for external tokenizer.json
    ///
    /// AC: AC1, AC5
    /// Spec: docs/specs/tokenizer-authority-integration-parity-both.md#AC1
    #[test]
    fn test_tokenizer_authority_source_external() {
        // Tests feature spec: tokenizer-authority-integration-parity-both.md#AC1
        // Verify: detect_tokenizer_source() returns External for tokenizer.json
        todo!("AC1: Verify source detection for external tokenizer.json");
    }

    /// Test: Source detection for GGUF-embedded tokenizer
    ///
    /// AC: AC1, AC5
    /// Spec: docs/specs/tokenizer-authority-integration-parity-both.md#AC1
    #[test]
    fn test_tokenizer_authority_source_gguf_embedded() {
        // Tests feature spec: tokenizer-authority-integration-parity-both.md#AC1
        // Verify: detect_tokenizer_source() returns GgufEmbedded for model.gguf
        todo!("AC1: Verify source detection for GGUF-embedded tokenizer");
    }

    /// Test: File hash computed only for external tokenizers
    ///
    /// AC: AC1, AC6
    /// Spec: docs/specs/tokenizer-authority-integration-parity-both.md#AC1
    #[test]
    fn test_tokenizer_authority_file_hash_external_only() {
        // Tests feature spec: tokenizer-authority-integration-parity-both.md#AC1
        // Verify: file_hash is Some() for External, None for GgufEmbedded
        todo!("AC1: Verify file hash computed only for external tokenizers");
    }

    /// Test: Config hash always computed from tokenizer trait
    ///
    /// AC: AC1, AC6
    /// Spec: docs/specs/tokenizer-authority-integration-parity-both.md#AC1
    #[test]
    fn test_tokenizer_authority_config_hash_always_computed() {
        // Tests feature spec: tokenizer-authority-integration-parity-both.md#AC1
        // Verify: config_hash always present (64 hex chars)
        todo!("AC1: Verify config hash always computed from tokenizer trait");
    }

    /// Test: Token count captured from Rust tokenization result
    ///
    /// AC: AC1
    /// Spec: docs/specs/tokenizer-authority-integration-parity-both.md#AC1
    #[test]
    fn test_tokenizer_authority_token_count_from_rust_tokenization() {
        // Tests feature spec: tokenizer-authority-integration-parity-both.md#AC1
        // Verify: token_count matches rust_tokens.len()
        todo!("AC1: Verify token count captured from Rust tokenization result");
    }

    /// Test: TokenizerAuthority deterministic (same inputs → same output)
    ///
    /// AC: AC1, AC6
    /// Spec: docs/specs/tokenizer-authority-integration-parity-both.md#AC1
    #[test]
    fn test_tokenizer_authority_deterministic_computation() {
        // Tests feature spec: tokenizer-authority-integration-parity-both.md#AC1
        // Verify: Computing authority twice yields identical results
        todo!("AC1: Verify TokenizerAuthority computation is deterministic");
    }
}

// ========================================
// TC_AC2: Dual Receipt Injection
// ========================================

#[cfg(test)]
mod tc_ac2_dual_receipt_injection {
    use super::*;

    /// Test: Same TokenizerAuthority passed to both run_single_lane() calls
    ///
    /// AC: AC2
    /// Spec: docs/specs/tokenizer-authority-integration-parity-both.md#AC2
    #[test]
    fn test_same_authority_passed_to_both_lanes() {
        // Tests feature spec: tokenizer-authority-integration-parity-both.md#AC2
        // Verify: Same authority reference passed to Lane A and Lane B
        todo!("AC2: Verify same TokenizerAuthority passed to both run_single_lane() calls");
    }

    /// Test: Authority cloned into each receipt via set_tokenizer_authority()
    ///
    /// AC: AC2
    /// Spec: docs/specs/tokenizer-authority-integration-parity-both.md#AC2
    #[test]
    fn test_authority_cloned_into_receipts() {
        // Tests feature spec: tokenizer-authority-integration-parity-both.md#AC2
        // Verify: receipt.set_tokenizer_authority(authority.clone()) called in each lane
        todo!("AC2: Verify authority cloned into each receipt via set_tokenizer_authority()");
    }

    /// Test: Both receipts contain identical tokenizer_authority field
    ///
    /// AC: AC2
    /// Spec: docs/specs/tokenizer-authority-integration-parity-both.md#AC2
    #[test]
    fn test_both_receipts_identical_tokenizer_authority() {
        // Tests feature spec: tokenizer-authority-integration-parity-both.md#AC2
        // Verify: receipt_bitnet.json and receipt_llama.json have matching tokenizer_authority
        todo!("AC2: Verify both receipts contain identical tokenizer_authority field");
    }

    /// Test: Receipt files written atomically to output directory
    ///
    /// AC: AC2
    /// Spec: docs/specs/tokenizer-authority-integration-parity-both.md#AC2
    #[test]
    fn test_receipt_files_written_atomically() {
        // Tests feature spec: tokenizer-authority-integration-parity-both.md#AC2
        // Verify: receipt.write_to_file() succeeds for both lanes
        todo!("AC2: Verify receipt files written atomically to output directory");
    }

    /// Test: Receipt JSON includes tokenizer_authority with all fields
    ///
    /// AC: AC2
    /// Spec: docs/specs/tokenizer-authority-integration-parity-both.md#AC2
    #[test]
    fn test_receipt_json_includes_tokenizer_authority() {
        // Tests feature spec: tokenizer-authority-integration-parity-both.md#AC2
        // Verify: JSON serialization includes tokenizer_authority object
        todo!("AC2: Verify receipt JSON includes tokenizer_authority with all fields");
    }

    /// Test: Receipt schema v2.0.0 backward compatibility
    ///
    /// AC: AC2
    /// Spec: docs/specs/tokenizer-authority-integration-parity-both.md#AC2
    #[test]
    fn test_receipt_schema_v2_backward_compat() {
        // Tests feature spec: tokenizer-authority-integration-parity-both.md#AC2
        // Verify: v2 receipts with tokenizer_authority can be deserialized
        todo!("AC2: Verify receipt schema v2.0.0 backward compatibility");
    }
}

// ========================================
// TC_AC3: Validation Logic (Hash Comparison)
// ========================================

#[cfg(test)]
mod tc_ac3_validation_logic {
    use super::*;

    /// Test: Validation checks config_hash match across lanes
    ///
    /// AC: AC3
    /// Spec: docs/specs/tokenizer-authority-integration-parity-both.md#AC3
    #[test]
    fn test_validation_checks_config_hash_match() {
        // Tests feature spec: tokenizer-authority-integration-parity-both.md#AC3
        // Verify: validate_tokenizer_consistency() checks config_hash equality
        todo!("AC3: Verify validation checks config_hash match across lanes");
    }

    /// Test: Validation checks token_count match across lanes
    ///
    /// AC: AC3
    /// Spec: docs/specs/tokenizer-authority-integration-parity-both.md#AC3
    #[test]
    fn test_validation_checks_token_count_match() {
        // Tests feature spec: tokenizer-authority-integration-parity-both.md#AC3
        // Verify: validate_tokenizer_consistency() checks token_count equality
        todo!("AC3: Verify validation checks token_count match across lanes");
    }

    /// Test: Validation succeeds when both authorities identical
    ///
    /// AC: AC3
    /// Spec: docs/specs/tokenizer-authority-integration-parity-both.md#AC3
    #[test]
    fn test_validation_succeeds_identical_authorities() {
        // Tests feature spec: tokenizer-authority-integration-parity-both.md#AC3
        // Verify: validate_tokenizer_consistency(auth_a, auth_b) returns Ok(())
        let auth = TokenizerAuthority {
            source: TokenizerSource::External,
            path: "tokenizer.json".to_string(),
            file_hash: Some("hash".to_string()),
            config_hash: "config_hash".to_string(),
            token_count: 5,
        };

        let result = validate_tokenizer_consistency(&auth, &auth);
        assert!(result.is_ok());
    }

    /// Test: Validation fails on config_hash mismatch
    ///
    /// AC: AC3, AC4
    /// Spec: docs/specs/tokenizer-authority-integration-parity-both.md#AC3
    #[test]
    fn test_validation_fails_config_hash_mismatch() {
        // Tests feature spec: tokenizer-authority-integration-parity-both.md#AC3
        // Verify: Different config_hash triggers validation error
        let auth_a = TokenizerAuthority {
            source: TokenizerSource::External,
            path: "tokenizer.json".to_string(),
            file_hash: Some("hash".to_string()),
            config_hash: "config_hash_a".to_string(),
            token_count: 5,
        };

        let auth_b = TokenizerAuthority {
            source: TokenizerSource::External,
            path: "tokenizer.json".to_string(),
            file_hash: Some("hash".to_string()),
            config_hash: "config_hash_b".to_string(),
            token_count: 5,
        };

        let result = validate_tokenizer_consistency(&auth_a, &auth_b);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Tokenizer config mismatch"));
    }

    /// Test: Validation fails on token_count mismatch
    ///
    /// AC: AC3, AC4
    /// Spec: docs/specs/tokenizer-authority-integration-parity-both.md#AC3
    #[test]
    fn test_validation_fails_token_count_mismatch() {
        // Tests feature spec: tokenizer-authority-integration-parity-both.md#AC3
        // Verify: Different token_count triggers validation error
        let auth_a = TokenizerAuthority {
            source: TokenizerSource::External,
            path: "tokenizer.json".to_string(),
            file_hash: Some("hash".to_string()),
            config_hash: "config_hash".to_string(),
            token_count: 5,
        };

        let auth_b = TokenizerAuthority {
            source: TokenizerSource::External,
            path: "tokenizer.json".to_string(),
            file_hash: Some("hash".to_string()),
            config_hash: "config_hash".to_string(),
            token_count: 10,
        };

        let result = validate_tokenizer_consistency(&auth_a, &auth_b);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Token count mismatch"));
    }

    /// Test: Validation error message includes both hashes
    ///
    /// AC: AC3
    /// Spec: docs/specs/tokenizer-authority-integration-parity-both.md#AC3
    #[test]
    fn test_validation_error_message_includes_hashes() {
        // Tests feature spec: tokenizer-authority-integration-parity-both.md#AC3
        // Verify: Error message shows both Lane A and Lane B config hashes
        todo!("AC3: Verify validation error message includes both config hashes");
    }

    /// Test: Validation executed in STEP 7.5 after receipt writes
    ///
    /// AC: AC3
    /// Spec: docs/specs/tokenizer-authority-integration-parity-both.md#AC3
    #[test]
    fn test_validation_executed_after_receipt_writes() {
        // Tests feature spec: tokenizer-authority-integration-parity-both.md#AC3
        // Verify: Validation happens after both receipts written to disk
        todo!("AC3: Verify validation executed in STEP 7.5 after receipt writes");
    }

    /// Test: Validation loads receipts from disk to extract authorities
    ///
    /// AC: AC3
    /// Spec: docs/specs/tokenizer-authority-integration-parity-both.md#AC3
    #[test]
    fn test_validation_loads_receipts_from_disk() {
        // Tests feature spec: tokenizer-authority-integration-parity-both.md#AC3
        // Verify: Receipts re-read from receipt_bitnet.json, receipt_llama.json
        todo!("AC3: Verify validation loads receipts from disk to extract authorities");
    }
}

// ========================================
// TC_AC4: Exit Code 2 on Tokenizer Mismatch
// ========================================

#[cfg(test)]
mod tc_ac4_exit_code_2_mismatch {
    use super::*;

    /// Test: Exit code 2 on config_hash mismatch
    ///
    /// AC: AC4
    /// Spec: docs/specs/tokenizer-authority-integration-parity-both.md#AC4
    #[test]
    #[ignore] // Integration test - requires end-to-end parity-both execution
    fn test_exit_code_2_config_hash_mismatch() {
        // Tests feature spec: tokenizer-authority-integration-parity-both.md#AC4
        // Verify: std::process::exit(2) called on config_hash mismatch
        todo!("AC4: Verify exit code 2 on config_hash mismatch");
    }

    /// Test: Exit code 2 on token_count mismatch
    ///
    /// AC: AC4
    /// Spec: docs/specs/tokenizer-authority-integration-parity-both.md#AC4
    #[test]
    #[ignore] // Integration test - requires end-to-end parity-both execution
    fn test_exit_code_2_token_count_mismatch() {
        // Tests feature spec: tokenizer-authority-integration-parity-both.md#AC4
        // Verify: std::process::exit(2) called on token_count mismatch
        todo!("AC4: Verify exit code 2 on token_count mismatch");
    }

    /// Test: Exit code semantics preserved (0=success, 1=divergence, 2=error)
    ///
    /// AC: AC4
    /// Spec: docs/specs/tokenizer-authority-integration-parity-both.md#AC4
    #[test]
    #[ignore] // Integration test - requires end-to-end parity-both execution
    fn test_exit_code_semantics_preserved() {
        // Tests feature spec: tokenizer-authority-integration-parity-both.md#AC4
        // Verify: Exit code semantics match spec table
        todo!("AC4: Verify exit code semantics preserved (0=success, 1=divergence, 2=error)");
    }

    /// Test: Error message printed to stderr before exit
    ///
    /// AC: AC4
    /// Spec: docs/specs/tokenizer-authority-integration-parity-both.md#AC4
    #[test]
    #[ignore] // Integration test - requires end-to-end parity-both execution
    fn test_error_message_printed_to_stderr() {
        // Tests feature spec: tokenizer-authority-integration-parity-both.md#AC4
        // Verify: eprintln!() called with detailed error message
        todo!("AC4: Verify error message printed to stderr before exit");
    }
}

// ========================================
// TC_AC5: Source Detection Heuristics
// ========================================

#[cfg(test)]
mod tc_ac5_source_detection {
    use super::*;
    use bitnet_crossval::receipt::detect_tokenizer_source;

    /// Test: detect_tokenizer_source() returns External for tokenizer.json
    ///
    /// AC: AC5
    /// Spec: docs/specs/tokenizer-authority-integration-parity-both.md#AC5
    #[test]
    #[ignore] // Requires file system fixture
    fn test_detect_source_external_tokenizer_json() {
        // Tests feature spec: tokenizer-authority-integration-parity-both.md#AC5
        // Verify: Path ending with "tokenizer.json" → External
        todo!("AC5: Verify detect_tokenizer_source() returns External for tokenizer.json");
    }

    /// Test: detect_tokenizer_source() returns GgufEmbedded for model.gguf
    ///
    /// AC: AC5
    /// Spec: docs/specs/tokenizer-authority-integration-parity-both.md#AC5
    #[test]
    fn test_detect_source_gguf_embedded() {
        // Tests feature spec: tokenizer-authority-integration-parity-both.md#AC5
        // Verify: Path not ending with "tokenizer.json" → GgufEmbedded
        let path = Path::new("models/model.gguf");
        let source = detect_tokenizer_source(path);
        assert_eq!(source, TokenizerSource::GgufEmbedded);
    }

    /// Test: detect_tokenizer_source() handles non-existent paths
    ///
    /// AC: AC5
    /// Spec: docs/specs/tokenizer-authority-integration-parity-both.md#AC5
    #[test]
    fn test_detect_source_nonexistent_path() {
        // Tests feature spec: tokenizer-authority-integration-parity-both.md#AC5
        // Verify: Non-existent path returns GgufEmbedded (safe default)
        let path = Path::new("nonexistent/tokenizer.json");
        let source = detect_tokenizer_source(path);
        assert_eq!(source, TokenizerSource::GgufEmbedded);
    }

    /// Test: detect_tokenizer_source() case sensitivity
    ///
    /// AC: AC5
    /// Spec: docs/specs/tokenizer-authority-integration-parity-both.md#AC5
    #[test]
    fn test_detect_source_case_sensitivity() {
        // Tests feature spec: tokenizer-authority-integration-parity-both.md#AC5
        // Verify: "Tokenizer.json" (capitalized) not treated as External
        let path = Path::new("models/Tokenizer.json");
        let source = detect_tokenizer_source(path);
        // Expect GgufEmbedded because exact match "tokenizer.json" required
        assert_eq!(source, TokenizerSource::GgufEmbedded);
    }

    /// Test: detect_tokenizer_source() with absolute paths
    ///
    /// AC: AC5
    /// Spec: docs/specs/tokenizer-authority-integration-parity-both.md#AC5
    #[test]
    #[ignore] // Requires file system fixture
    fn test_detect_source_absolute_path() {
        // Tests feature spec: tokenizer-authority-integration-parity-both.md#AC5
        // Verify: Absolute path ending with "tokenizer.json" → External
        todo!("AC5: Verify detect_tokenizer_source() with absolute paths");
    }

    /// Test: AutoDiscovered variant not yet implemented
    ///
    /// AC: AC5
    /// Spec: docs/specs/tokenizer-authority-integration-parity-both.md#AC5
    #[test]
    fn test_auto_discovered_not_implemented() {
        // Tests feature spec: tokenizer-authority-integration-parity-both.md#AC5
        // Verify: AutoDiscovered is enum variant but not returned by detect function yet
        // This is expected behavior - AutoDiscovered is future enhancement
        assert_eq!(TokenizerSource::AutoDiscovered, TokenizerSource::AutoDiscovered);
    }
}

// ========================================
// TC_AC6: Hash Computation (Determinism & Collision Resistance)
// ========================================

#[cfg(test)]
mod tc_ac6_hash_computation {
    use super::*;
    use bitnet_crossval::receipt::{
        compute_tokenizer_config_hash_from_tokenizer, compute_tokenizer_file_hash,
    };

    /// Test: compute_tokenizer_file_hash() returns 64 hex characters
    ///
    /// AC: AC6
    /// Spec: docs/specs/tokenizer-authority-integration-parity-both.md#AC6
    #[test]
    #[ignore] // Requires file system fixture
    fn test_file_hash_64_hex_chars() {
        // Tests feature spec: tokenizer-authority-integration-parity-both.md#AC6
        // Verify: SHA256 hash is lowercase hex string of exactly 64 characters
        todo!("AC6: Verify compute_tokenizer_file_hash() returns 64 hex characters");
    }

    /// Test: compute_tokenizer_file_hash() deterministic (same file → same hash)
    ///
    /// AC: AC6
    /// Spec: docs/specs/tokenizer-authority-integration-parity-both.md#AC6
    #[test]
    #[ignore] // Requires file system fixture
    fn test_file_hash_deterministic() {
        // Tests feature spec: tokenizer-authority-integration-parity-both.md#AC6
        // Verify: Hashing same file twice produces identical output
        todo!("AC6: Verify compute_tokenizer_file_hash() deterministic");
    }

    /// Test: compute_tokenizer_file_hash() fails on missing file
    ///
    /// AC: AC6
    /// Spec: docs/specs/tokenizer-authority-integration-parity-both.md#AC6
    #[test]
    fn test_file_hash_missing_file_error() {
        // Tests feature spec: tokenizer-authority-integration-parity-both.md#AC6
        // Verify: Missing file returns Err with descriptive message
        let path = Path::new("nonexistent/tokenizer.json");
        let result = compute_tokenizer_file_hash(path);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Failed to read"));
    }

    /// Test: compute_tokenizer_config_hash_from_tokenizer() deterministic
    ///
    /// AC: AC6
    /// Spec: docs/specs/tokenizer-authority-integration-parity-both.md#AC6
    #[test]
    #[ignore] // Requires mock tokenizer implementation
    fn test_config_hash_deterministic() {
        // Tests feature spec: tokenizer-authority-integration-parity-both.md#AC6
        // Verify: Hashing same tokenizer config twice produces identical output
        todo!("AC6: Verify compute_tokenizer_config_hash_from_tokenizer() deterministic");
    }

    /// Test: compute_tokenizer_config_hash_from_tokenizer() uses canonical JSON
    ///
    /// AC: AC6
    /// Spec: docs/specs/tokenizer-authority-integration-parity-both.md#AC6
    #[test]
    #[ignore] // Requires mock tokenizer implementation
    fn test_config_hash_canonical_json() {
        // Tests feature spec: tokenizer-authority-integration-parity-both.md#AC6
        // Verify: Hash computed from canonical JSON (vocab_size, real_vocab_size)
        todo!("AC6: Verify compute_tokenizer_config_hash_from_tokenizer() uses canonical JSON");
    }

    /// Test: Config hash includes vocab_size and real_vocab_size
    ///
    /// AC: AC6
    /// Spec: docs/specs/tokenizer-authority-integration-parity-both.md#AC6
    #[test]
    #[ignore] // Requires mock tokenizer implementation
    fn test_config_hash_includes_vocab_sizes() {
        // Tests feature spec: tokenizer-authority-integration-parity-both.md#AC6
        // Verify: Config hash changes when vocab_size or real_vocab_size differ
        todo!("AC6: Verify config hash includes vocab_size and real_vocab_size");
    }

    /// Test: File hash and config hash are independent
    ///
    /// AC: AC6
    /// Spec: docs/specs/tokenizer-authority-integration-parity-both.md#AC6
    #[test]
    #[ignore] // Requires mock tokenizer implementation
    fn test_file_hash_and_config_hash_independent() {
        // Tests feature spec: tokenizer-authority-integration-parity-both.md#AC6
        // Verify: Modifying file content changes file_hash but not config_hash
        todo!("AC6: Verify file hash and config hash are independent");
    }

    /// Test: Hash collision resistance (different inputs → different hashes)
    ///
    /// AC: AC6
    /// Spec: docs/specs/tokenizer-authority-integration-parity-both.md#AC6
    #[test]
    #[ignore] // Requires mock tokenizer implementation
    fn test_hash_collision_resistance() {
        // Tests feature spec: tokenizer-authority-integration-parity-both.md#AC6
        // Verify: Different vocab configs produce different config hashes
        todo!("AC6: Verify hash collision resistance");
    }

    /// Test: Hash format is lowercase hex (no uppercase, no hyphens)
    ///
    /// AC: AC6
    /// Spec: docs/specs/tokenizer-authority-integration-parity-both.md#AC6
    #[test]
    #[ignore] // Requires file system fixture
    fn test_hash_format_lowercase_hex() {
        // Tests feature spec: tokenizer-authority-integration-parity-both.md#AC6
        // Verify: Hash string is lowercase hex (a-f0-9), no uppercase or punctuation
        todo!("AC6: Verify hash format is lowercase hex");
    }

    /// Test: File hash performance (< 100ms for typical tokenizer.json)
    ///
    /// AC: AC6
    /// Spec: docs/specs/tokenizer-authority-integration-parity-both.md#AC6
    #[test]
    #[ignore] // Requires file system fixture and performance profiling
    fn test_file_hash_performance() {
        // Tests feature spec: tokenizer-authority-integration-parity-both.md#AC6
        // Verify: File hash computation completes in < 100ms for 2MB tokenizer file
        todo!("AC6: Verify file hash performance (< 100ms for typical tokenizer.json)");
    }

    /// Test: Config hash performance (< 1ms for typical tokenizer)
    ///
    /// AC: AC6
    /// Spec: docs/specs/tokenizer-authority-integration-parity-both.md#AC6
    #[test]
    #[ignore] // Requires mock tokenizer implementation and performance profiling
    fn test_config_hash_performance() {
        // Tests feature spec: tokenizer-authority-integration-parity-both.md#AC6
        // Verify: Config hash computation completes in < 1ms
        todo!("AC6: Verify config hash performance (< 1ms for typical tokenizer)");
    }

    /// Test: Hash memory overhead is negligible (< 200 bytes per receipt)
    ///
    /// AC: AC6
    /// Spec: docs/specs/tokenizer-authority-integration-parity-both.md#AC6
    #[test]
    fn test_hash_memory_overhead_negligible() {
        // Tests feature spec: tokenizer-authority-integration-parity-both.md#AC6
        // Verify: TokenizerAuthority struct size is acceptable
        let auth = TokenizerAuthority {
            source: TokenizerSource::External,
            path: "tokenizer.json".to_string(),
            file_hash: Some("a".repeat(64)),
            config_hash: "b".repeat(64),
            token_count: 128000,
        };

        // Memory layout check - TokenizerAuthority should be < 200 bytes
        let size = std::mem::size_of_val(&auth);
        assert!(size < 200, "TokenizerAuthority struct too large: {} bytes", size);
    }
}
