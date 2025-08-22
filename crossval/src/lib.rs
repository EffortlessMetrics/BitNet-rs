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

#[cfg(feature = "crossval")]
pub mod cpp_bindings;

#[cfg(feature = "crossval")]
pub mod comparison;

pub mod fixtures;
pub mod utils;
pub mod validation;
pub mod score;

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
