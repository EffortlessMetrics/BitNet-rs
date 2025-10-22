//! Test Helper Modules for BitNet.rs Model Testing
//!
//! This module provides shared test utilities for GGUF fixture generation,
//! quantization testing, and model loading validation.

pub mod qk256_fixtures;

// Re-export commonly used fixture generators for qk256_fixture_validation tests
#[allow(unused_imports)]
pub use qk256_fixtures::{generate_bitnet32_2x64, generate_qk256_3x300, generate_qk256_4x256};
