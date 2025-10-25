//! Cross-validation framework for BitNet Rust vs C++ implementations
//!
//! This crate provides functionality to compare the BitNet Rust implementation
//! against the original C++ implementation for numerical accuracy and performance.
//!
//! # Features
//!
//! - `crossval`: Enables cross-validation functionality (requires C++ dependencies)
//!
//! # Usage
//!
//! ```bash
//! # Enable cross-validation features
//! cargo test --features crossval
//! cargo bench --features crossval
//! ```

#[cfg(any(feature = "crossval", feature = "ffi"))]
pub mod cpp_bindings;

#[cfg(feature = "crossval")]
pub mod comparison;

pub mod backend;
pub mod fixtures;
pub mod logits_compare;
pub mod metrics;
pub mod receipt;
pub mod score;
pub mod token_parity;
pub mod utils;
pub mod validation;

/// Error types for cross-validation operations
#[derive(thiserror::Error, Debug)]
pub enum CrossvalError {
    #[error("C++ implementation not available (compile with --features crossval)")]
    CppNotAvailable,

    #[error("Model loading failed: {0}")]
    ModelLoadError(String),

    #[error("Inference failed: {0}")]
    InferenceError(String),

    #[error("Numerical comparison failed: {0}")]
    ComparisonError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, CrossvalError>;

/// Configuration for cross-validation tests
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CrossvalConfig {
    /// Numerical tolerance for floating-point comparisons
    pub tolerance: f64,
    /// Maximum number of tokens to compare
    pub max_tokens: usize,
    /// Whether to run performance benchmarks
    pub benchmark: bool,
}

impl Default for CrossvalConfig {
    fn default() -> Self {
        Self { tolerance: 1e-6, max_tokens: 1000, benchmark: false }
    }
}

/// Assert that the first token logits match between Rust and C++ implementations
#[cfg(feature = "crossval")]
pub fn assert_first_logits_match(model_path: &str, prompt: &str) {
    use bitnet_inference::eval_logits_once;
    use bitnet_sys::wrapper::{self, Session as CppSession};

    wrapper::init_backend();
    let _guard = scopeguard::guard((), |_| wrapper::free_backend());

    let mut cpp_session =
        CppSession::load_deterministic(model_path).expect("failed to load C++ model");
    let tokens = cpp_session.tokenize(prompt).expect("tokenize failed");
    let cpp_logits = cpp_session.eval_and_get_logits(&tokens, 0).expect("C++ inference failed");
    let rust_logits = eval_logits_once(model_path, &tokens).expect("Rust inference failed");

    assert!(
        (rust_logits[0] - cpp_logits[0]).abs() < 1e-4,
        "First token logits diverged: rust={} cpp={}",
        rust_logits[0],
        cpp_logits[0]
    );
}

/// Stub when crossval feature is disabled
#[cfg(not(feature = "crossval"))]
pub fn assert_first_logits_match(_model_path: &str, _prompt: &str) {
    panic!("crossval feature required for assert_first_logits_match");
}

#[cfg(test)]
mod tests {
    #[test]
    #[cfg(feature = "ffi")]
    fn test_crossval_backend_detection_env_vars() {
        // Verify that build.rs exports CROSSVAL_HAS_* environment variables
        let has_bitnet = env!("CROSSVAL_HAS_BITNET");
        let has_llama = env!("CROSSVAL_HAS_LLAMA");

        println!("CROSSVAL_HAS_BITNET = {}", has_bitnet);
        println!("CROSSVAL_HAS_LLAMA = {}", has_llama);

        // Validate that the env vars are valid boolean strings
        assert!(has_bitnet == "true" || has_bitnet == "false");
        assert!(has_llama == "true" || has_llama == "false");
    }
}
