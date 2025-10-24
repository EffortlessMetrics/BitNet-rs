//! Test Helper Modules for BitNet.rs Model Testing
//!
//! This module provides shared test utilities for GGUF fixture generation,
//! quantization testing, model loading validation, and tensor alignment validation.

pub mod alignment_validator;
pub mod fixture_loader;
pub mod qk256_fixtures;
pub mod qk256_tolerance;

// Re-export commonly used fixture generators for qk256_fixture_validation tests
#[allow(unused_imports)]
pub use qk256_fixtures::{generate_bitnet32_2x64, generate_qk256_3x300, generate_qk256_4x256};

// Re-export alignment validation utilities
#[allow(unused_imports)]
pub use alignment_validator::{
    AlignmentConfig, ValidationResult, validate_all_tensors, validate_candle_tensor,
    validate_gguf_tensor_metadata,
};

// Re-export QK256 tolerance helpers for property tests
#[allow(unused_imports)]
pub use qk256_tolerance::{approx_eq, approx_eq_with_len};

// Re-export fixture loader utilities for disk-based fixtures
#[allow(unused_imports)]
pub use fixture_loader::{fixture_path, load_fixture_bytes};
