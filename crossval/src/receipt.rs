//! Parity receipt generation for cross-validation debugging and CI validation
//!
//! This module provides structured JSON output for per-token parity comparisons,
//! enabling systematic debugging of divergence points and CI quality gates.
//!
//! ## Schema
//!
//! Receipt format follows semantic versioning:
//! - **v1**: Initial schema with position-level metrics (MSE, KL, TopK)
//! - Future versions may add additional metrics while maintaining backward compatibility
//!
//! ## Usage
//!
//! ```rust,ignore
//! use bitnet_crossval::receipt::{ParityReceipt, PositionMetrics, Thresholds};
//!
//! let mut receipt = ParityReceipt::new("model.gguf", "bitnet", "What is 2+2?");
//! receipt.set_thresholds(Thresholds { mse: 1e-4, kl: 0.1, topk: 0.8 });
//!
//! // Add position metrics
//! receipt.add_position(PositionMetrics {
//!     pos: 0,
//!     mse: 1e-6,
//!     max_abs: 1e-5,
//!     kl: Some(0.01),
//!     topk_agree: Some(1.0),
//!     top5_rust: vec![128000, 1229, 374, 220, 17],
//!     top5_cpp: vec![128000, 1229, 374, 220, 17],
//! });
//!
//! receipt.finalize();
//! receipt.write_to_file("parity-receipt.json")?;
//! ```

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Tokenizer source type (AC5)
///
/// Specification: docs/specs/parity-both-preflight-tokenizer.md#AC5
///
/// Represents where the tokenizer configuration originated from.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TokenizerSource {
    /// Tokenizer embedded in GGUF file
    GgufEmbedded,
    /// External tokenizer.json file (explicitly provided)
    External,
    /// Auto-discovered tokenizer from model directory
    AutoDiscovered,
}

/// Tokenizer authority metadata for receipt reproducibility (AC4-AC6)
///
/// Specification: docs/specs/parity-both-preflight-tokenizer.md#AC4
///
/// This structure captures complete tokenizer provenance to ensure
/// receipt reproducibility and enable tokenizer parity validation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TokenizerAuthority {
    /// Tokenizer source: GgufEmbedded, External, or AutoDiscovered
    pub source: TokenizerSource,

    /// Path to tokenizer (GGUF path or tokenizer.json path)
    pub path: String,

    /// SHA256 hash of tokenizer.json file (if external)
    ///
    /// This is None for GGUF-embedded tokenizers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_hash: Option<String>,

    /// SHA256 hash of effective tokenizer config (canonical representation)
    ///
    /// This hash is computed from the tokenizer's configuration fingerprint,
    /// ensuring consistency across different instances.
    pub config_hash: String,

    /// Token count (for quick validation)
    pub token_count: usize,
}

/// Parity receipt - structured output for cross-validation comparison
///
/// This receipt captures per-position metrics comparing Rust and C++ logits,
/// enabling systematic debugging and CI validation.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ParityReceipt {
    /// Schema version (semantic versioning)
    pub version: u32,

    /// RFC3339 timestamp when receipt was generated
    pub timestamp: String,

    /// Model path or identifier
    pub model: String,

    /// C++ backend used (bitnet or llama)
    pub backend: String,

    /// Input prompt (formatted)
    pub prompt: String,

    /// Number of token positions compared
    #[serde(default)]
    pub positions: usize,

    /// Quality thresholds for validation
    #[serde(default)]
    pub thresholds: Thresholds,

    /// Per-position metrics (one entry per token position)
    #[serde(default)]
    pub rows: Vec<PositionMetrics>,

    /// Aggregate summary metrics
    #[serde(default)]
    pub summary: Summary,

    // v2 fields (optional for backward compatibility)
    /// Tokenizer authority metadata (v2.0.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokenizer_authority: Option<TokenizerAuthority>,

    /// Prompt template used (v2.0.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_template: Option<String>,

    /// Determinism seed (v2.0.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub determinism_seed: Option<u64>,

    /// Model SHA256 hash (v2.0.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_sha256: Option<String>,
}

/// Quality thresholds for parity validation
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Thresholds {
    /// Mean squared error threshold (lower is better)
    pub mse: f32,

    /// Kullback-Leibler divergence threshold (optional, lower is better)
    pub kl: f32,

    /// Top-K agreement threshold (0.0-1.0, higher is better)
    pub topk: f32,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            mse: 1e-4, // Default MSE threshold
            kl: 0.1,   // Default KL threshold
            topk: 0.8, // Default top-K agreement (80%)
        }
    }
}

/// Per-position parity metrics
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PositionMetrics {
    /// Token position (0-indexed)
    pub pos: usize,

    /// Mean squared error between Rust and C++ logits
    pub mse: f32,

    /// Maximum absolute difference across all logits at this position
    pub max_abs: f32,

    /// Kullback-Leibler divergence (optional - requires softmax normalization)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kl: Option<f32>,

    /// Top-K agreement (fraction of top-K tokens that match, optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub topk_agree: Option<f32>,

    /// Top-5 token IDs from Rust logits (highest to lowest)
    pub top5_rust: Vec<usize>,

    /// Top-5 token IDs from C++ logits (highest to lowest)
    pub top5_cpp: Vec<usize>,
}

/// Aggregate summary metrics
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Summary {
    /// True if all positions passed quality thresholds
    pub all_passed: bool,

    /// First position where divergence was detected (None if all passed)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_divergence: Option<usize>,

    /// Mean MSE across all positions
    pub mean_mse: f32,

    /// Mean KL divergence across all positions (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean_kl: Option<f32>,
}

impl Default for Summary {
    fn default() -> Self {
        Self { all_passed: true, first_divergence: None, mean_mse: 0.0, mean_kl: None }
    }
}

impl ParityReceipt {
    /// Create a new parity receipt with metadata
    ///
    /// # Arguments
    ///
    /// * `model` - Model path or identifier
    /// * `backend` - C++ backend used (e.g., "bitnet", "llama")
    /// * `prompt` - Input prompt (formatted with template)
    pub fn new(model: &str, backend: &str, prompt: &str) -> Self {
        Self {
            version: 1, // Schema version v1
            timestamp: chrono::Utc::now().to_rfc3339(),
            model: model.to_string(),
            backend: backend.to_string(),
            prompt: prompt.to_string(),
            positions: 0,
            thresholds: Thresholds::default(),
            rows: Vec::new(),
            summary: Summary {
                all_passed: true,
                first_divergence: None,
                mean_mse: 0.0,
                mean_kl: None,
            },
            tokenizer_authority: None,
            prompt_template: None,
            determinism_seed: None,
            model_sha256: None,
        }
    }

    /// Set custom quality thresholds
    pub fn set_thresholds(&mut self, thresholds: Thresholds) {
        self.thresholds = thresholds;
    }

    /// Add a position's metrics to the receipt
    pub fn add_position(&mut self, metrics: PositionMetrics) {
        self.rows.push(metrics);
    }

    /// Set tokenizer authority metadata (AC4)
    ///
    /// Specification: docs/specs/parity-both-preflight-tokenizer.md#AC4
    pub fn set_tokenizer_authority(&mut self, authority: TokenizerAuthority) {
        self.tokenizer_authority = Some(authority);
    }

    /// Set prompt template used (AC4)
    ///
    /// Specification: docs/specs/parity-both-preflight-tokenizer.md#AC4
    pub fn set_prompt_template(&mut self, template: String) {
        self.prompt_template = Some(template);
    }

    /// Infer schema version based on fields present (AC7)
    ///
    /// Specification: docs/specs/parity-both-preflight-tokenizer.md#AC7
    ///
    /// Returns "1.0.0" if no v2 fields are present, "2.0.0" otherwise.
    pub fn infer_version(&self) -> &str {
        match (&self.tokenizer_authority, &self.prompt_template) {
            (Some(_), _) | (_, Some(_)) => "2.0.0",
            _ => "1.0.0",
        }
    }

    /// Finalize the receipt by computing summary statistics
    ///
    /// This should be called after all positions have been added and before
    /// serializing to JSON.
    pub fn finalize(&mut self) {
        self.positions = self.rows.len();

        if self.rows.is_empty() {
            return;
        }

        // Compute mean MSE
        let total_mse: f32 = self.rows.iter().map(|r| r.mse).sum();
        self.summary.mean_mse = total_mse / self.rows.len() as f32;

        // Compute mean KL (if available for all positions)
        let kl_values: Vec<f32> = self.rows.iter().filter_map(|r| r.kl).collect();
        if kl_values.len() == self.rows.len() {
            let total_kl: f32 = kl_values.iter().sum();
            self.summary.mean_kl = Some(total_kl / kl_values.len() as f32);
        }

        // Determine first divergence based on MSE threshold
        self.summary.first_divergence = self.rows.iter().position(|r| r.mse > self.thresholds.mse);

        // Update all_passed flag
        self.summary.all_passed = self.summary.first_divergence.is_none();
    }

    /// Serialize receipt to JSON string
    ///
    /// # Returns
    ///
    /// Pretty-printed JSON string
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Write receipt to file as JSON
    ///
    /// # Arguments
    ///
    /// * `path` - Output file path
    pub fn write_to_file(&self, path: &Path) -> anyhow::Result<()> {
        let json = self.to_json()?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

// ============================================================================
// Tokenizer Authority Helpers (AC6)
// ============================================================================

use anyhow::Context;

/// Compute SHA256 hash of tokenizer.json file (AC6)
///
/// Specification: docs/specs/parity-both-preflight-tokenizer.md#AC6
///
/// Returns lowercase hex-encoded SHA256 hash (64 characters)
pub fn compute_tokenizer_file_hash(tokenizer_path: &Path) -> anyhow::Result<String> {
    use sha2::{Digest, Sha256};

    let contents = std::fs::read(tokenizer_path)
        .with_context(|| format!("Failed to read tokenizer file: {}", tokenizer_path.display()))?;

    let mut hasher = Sha256::new();
    hasher.update(&contents);
    Ok(format!("{:x}", hasher.finalize()))
}

/// Compute SHA256 hash of tokenizer config from vocab sizes (AC6)
///
/// Specification: docs/specs/parity-both-preflight-tokenizer.md#AC6
///
/// This computes a hash from vocab size and special tokens.
/// Takes vocab_size and special_tokens list for canonical hashing.
pub fn compute_tokenizer_config_hash(
    vocab_size: usize,
    special_tokens: &[String],
) -> anyhow::Result<String> {
    use sha2::{Digest, Sha256};

    // Create canonical representation from vocab config
    let config_repr = serde_json::json!({
        "vocab_size": vocab_size,
        "special_tokens": special_tokens,
    });
    let canonical_json =
        serde_json::to_string(&config_repr).context("Failed to serialize tokenizer config")?;

    let mut hasher = Sha256::new();
    hasher.update(canonical_json.as_bytes());
    Ok(format!("{:x}", hasher.finalize()))
}

/// Compute SHA256 hash of tokenizer config from Tokenizer trait (AC6)
///
/// Specification: docs/specs/parity-both-preflight-tokenizer.md#AC6
///
/// This computes a hash from the tokenizer's vocab size configuration.
pub fn compute_tokenizer_config_hash_from_tokenizer(
    tokenizer: &dyn bitnet_tokenizers::Tokenizer,
) -> anyhow::Result<String> {
    use sha2::{Digest, Sha256};

    // Create canonical representation from vocab sizes
    let config_repr = serde_json::json!({
        "vocab_size": tokenizer.vocab_size(),
        "real_vocab_size": tokenizer.real_vocab_size(),
    });
    let canonical_json =
        serde_json::to_string(&config_repr).context("Failed to serialize tokenizer config")?;

    let mut hasher = Sha256::new();
    hasher.update(canonical_json.as_bytes());
    Ok(format!("{:x}", hasher.finalize()))
}

/// Detect tokenizer source (AC5)
///
/// Specification: docs/specs/parity-both-preflight-tokenizer.md#AC5
pub fn detect_tokenizer_source(tokenizer_path: &Path) -> TokenizerSource {
    // Check if file exists and is named tokenizer.json
    if tokenizer_path.exists()
        && tokenizer_path.file_name() == Some(std::ffi::OsStr::new("tokenizer.json"))
    {
        TokenizerSource::External
    } else {
        TokenizerSource::GgufEmbedded
    }
}

/// Validate tokenizer parity between Rust and C++ implementations (AC7)
///
/// Specification: docs/specs/parity-both-preflight-tokenizer.md#AC7
///
/// Compares token sequences from Rust and C++ tokenizers to ensure parity.
/// Returns an error if lengths differ or any token at corresponding positions differs.
///
/// # Arguments
///
/// * `rust_tokens` - Token IDs from Rust tokenizer
/// * `cpp_tokens` - Token IDs from C++ tokenizer
/// * `backend_name` - Name of the C++ backend (for error reporting)
///
/// # Errors
///
/// Returns an error if:
/// - Token sequence lengths differ
/// - Any token at corresponding positions differs
pub fn validate_tokenizer_parity(
    rust_tokens: &[u32],
    cpp_tokens: &[u32],
    backend_name: &str,
) -> anyhow::Result<()> {
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
/// Specification: docs/specs/parity-both-preflight-tokenizer.md#AC7
///
/// Ensures that two TokenizerAuthority instances represent the same effective tokenizer
/// by comparing config hashes and token counts.
///
/// # Arguments
///
/// * `lane_a` - First TokenizerAuthority to compare
/// * `lane_b` - Second TokenizerAuthority to compare
///
/// # Errors
///
/// Returns an error if:
/// - Config hashes differ (different effective tokenizers)
/// - Token counts differ (sanity check)
pub fn validate_tokenizer_consistency(
    lane_a: &TokenizerAuthority,
    lane_b: &TokenizerAuthority,
) -> anyhow::Result<()> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_receipt_builder_api() {
        let mut receipt = ParityReceipt::new("model.gguf", "bitnet", "What is 2+2?");

        assert_eq!(receipt.version, 1);
        assert_eq!(receipt.model, "model.gguf");
        assert_eq!(receipt.backend, "bitnet");
        assert_eq!(receipt.prompt, "What is 2+2?");
        assert_eq!(receipt.positions, 0);
        assert!(receipt.rows.is_empty());

        // Add a position
        receipt.add_position(PositionMetrics {
            pos: 0,
            mse: 1e-6,
            max_abs: 1e-5,
            kl: Some(0.01),
            topk_agree: Some(1.0),
            top5_rust: vec![128000, 1229, 374, 220, 17],
            top5_cpp: vec![128000, 1229, 374, 220, 17],
        });

        assert_eq!(receipt.rows.len(), 1);

        receipt.finalize();
        assert_eq!(receipt.positions, 1);
        assert!(receipt.summary.all_passed);
        assert!(receipt.summary.first_divergence.is_none());
    }

    #[test]
    fn test_receipt_serialization() {
        let mut receipt = ParityReceipt::new("model.gguf", "llama", "Test prompt");
        receipt.add_position(PositionMetrics {
            pos: 0,
            mse: 1e-5,
            max_abs: 1e-4,
            kl: Some(0.02),
            topk_agree: Some(0.9),
            top5_rust: vec![1, 2, 3, 4, 5],
            top5_cpp: vec![1, 2, 3, 4, 5],
        });

        receipt.finalize();

        let json = receipt.to_json().expect("Failed to serialize");
        assert!(json.contains("\"version\": 1"));
        assert!(json.contains("\"model\": \"model.gguf\""));
        assert!(json.contains("\"backend\": \"llama\""));
        assert!(json.contains("\"prompt\": \"Test prompt\""));
        assert!(json.contains("\"positions\": 1"));
    }

    #[test]
    fn test_receipt_deserialization() {
        let json = r#"{
            "version": 1,
            "timestamp": "2025-01-15T10:30:00Z",
            "model": "model.gguf",
            "backend": "bitnet",
            "prompt": "Hello",
            "positions": 1,
            "thresholds": {
                "mse": 0.0001,
                "kl": 0.1,
                "topk": 0.8
            },
            "rows": [
                {
                    "pos": 0,
                    "mse": 0.00001,
                    "max_abs": 0.0001,
                    "kl": 0.01,
                    "topk_agree": 1.0,
                    "top5_rust": [1, 2, 3, 4, 5],
                    "top5_cpp": [1, 2, 3, 4, 5]
                }
            ],
            "summary": {
                "all_passed": true,
                "mean_mse": 0.00001,
                "mean_kl": 0.01
            }
        }"#;

        let receipt: ParityReceipt = serde_json::from_str(json).expect("Failed to deserialize");
        assert_eq!(receipt.version, 1);
        assert_eq!(receipt.model, "model.gguf");
        assert_eq!(receipt.backend, "bitnet");
        assert_eq!(receipt.positions, 1);
        assert_eq!(receipt.rows.len(), 1);
        assert!(receipt.summary.all_passed);
    }

    #[test]
    fn test_summary_divergence_detection() {
        let mut receipt = ParityReceipt::new("model.gguf", "bitnet", "Test");
        receipt.set_thresholds(Thresholds { mse: 1e-4, kl: 0.1, topk: 0.8 });

        // Add two passing positions
        receipt.add_position(PositionMetrics {
            pos: 0,
            mse: 1e-6,
            max_abs: 1e-5,
            kl: None,
            topk_agree: None,
            top5_rust: vec![1, 2, 3, 4, 5],
            top5_cpp: vec![1, 2, 3, 4, 5],
        });

        receipt.add_position(PositionMetrics {
            pos: 1,
            mse: 1e-6,
            max_abs: 1e-5,
            kl: None,
            topk_agree: None,
            top5_rust: vec![6, 7, 8, 9, 10],
            top5_cpp: vec![6, 7, 8, 9, 10],
        });

        // Add one failing position (MSE exceeds threshold)
        receipt.add_position(PositionMetrics {
            pos: 2,
            mse: 1e-3, // Exceeds threshold of 1e-4
            max_abs: 1e-2,
            kl: None,
            topk_agree: None,
            top5_rust: vec![11, 12, 13, 14, 15],
            top5_cpp: vec![16, 17, 18, 19, 20],
        });

        receipt.finalize();

        assert_eq!(receipt.positions, 3);
        assert!(!receipt.summary.all_passed);
        assert_eq!(receipt.summary.first_divergence, Some(2));
    }

    #[test]
    fn test_thresholds_default() {
        let thresholds = Thresholds::default();
        assert_eq!(thresholds.mse, 1e-4);
        assert_eq!(thresholds.kl, 0.1);
        assert_eq!(thresholds.topk, 0.8);
    }

    #[test]
    fn test_empty_receipt() {
        let mut receipt = ParityReceipt::new("model.gguf", "bitnet", "Empty");
        receipt.finalize();

        assert_eq!(receipt.positions, 0);
        assert!(receipt.rows.is_empty());
        assert!(receipt.summary.all_passed);
        assert_eq!(receipt.summary.mean_mse, 0.0);
        assert!(receipt.summary.mean_kl.is_none());
    }

    #[test]
    fn test_file_io() {
        use tempfile::NamedTempFile;

        let mut receipt = ParityReceipt::new("model.gguf", "llama", "File test");
        receipt.add_position(PositionMetrics {
            pos: 0,
            mse: 1e-5,
            max_abs: 1e-4,
            kl: None,
            topk_agree: None,
            top5_rust: vec![1, 2, 3, 4, 5],
            top5_cpp: vec![1, 2, 3, 4, 5],
        });
        receipt.finalize();

        // Write to temp file
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let temp_path = temp_file.path();

        receipt.write_to_file(temp_path).expect("Failed to write receipt");

        // Read back and verify
        let content = std::fs::read_to_string(temp_path).expect("Failed to read file");
        assert!(content.contains("\"version\": 1"));
        assert!(content.contains("\"model\": \"model.gguf\""));
        assert!(content.contains("\"backend\": \"llama\""));

        // Verify deserialization works
        let loaded: ParityReceipt = serde_json::from_str(&content).expect("Failed to deserialize");
        assert_eq!(loaded.version, receipt.version);
        assert_eq!(loaded.model, receipt.model);
        assert_eq!(loaded.positions, receipt.positions);
    }

    #[test]
    fn test_compute_tokenizer_config_hash_determinism() {
        use super::compute_tokenizer_config_hash;

        let vocab_size = 128000;
        let special_tokens = vec!["<bos>".to_string(), "<eos>".to_string()];

        let hash1 = compute_tokenizer_config_hash(vocab_size, &special_tokens).unwrap();
        let hash2 = compute_tokenizer_config_hash(vocab_size, &special_tokens).unwrap();

        assert_eq!(hash1, hash2, "Config hash should be deterministic");
        assert_eq!(hash1.len(), 64, "SHA256 hash should be 64 hex characters");
        assert!(
            hash1.chars().all(|c| c.is_ascii_hexdigit()),
            "Hash should only contain hex digits"
        );
    }

    #[test]
    fn test_compute_tokenizer_file_hash_determinism() {
        use super::compute_tokenizer_file_hash;
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create temp file with known content
        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file.write_all(b"test tokenizer content").expect("Failed to write");
        let temp_path = temp_file.path();

        // Compute hash twice
        let hash1 = compute_tokenizer_file_hash(temp_path).unwrap();
        let hash2 = compute_tokenizer_file_hash(temp_path).unwrap();

        assert_eq!(hash1, hash2, "File hash should be deterministic");
        assert_eq!(hash1.len(), 64, "SHA256 hash should be 64 hex characters");
        assert!(
            hash1.chars().all(|c| c.is_ascii_hexdigit()),
            "Hash should only contain hex digits"
        );
    }
}
