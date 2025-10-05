//! Issue #261 Cross-Validation Reference Data
//!
//! C++ reference implementation outputs for systematic comparison with Rust implementation.
//! Supports AC9 (cross-validation accuracy tests) with >99.5% correlation and <1e-5 MSE.
//!
//! Cross-validation targets:
//! - Correlation: >99.5%
//! - MSE: <1e-5
//! - Performance variance: <5%
//! - Numerical tolerance: <1e-6 for individual operations

#![allow(dead_code)]

/// Cross-validation test fixture with C++ reference outputs
#[derive(Debug, Clone)]
pub struct CrossValFixture {
    pub test_id: &'static str,
    pub input_prompt: &'static str,
    pub input_tokens: Vec<u32>,
    pub rust_expected_output: Vec<f32>,
    pub cpp_reference_output: Vec<f32>,
    pub quantization_type: QuantizationType,
    pub model_config: ModelConfig,
    pub validation_targets: ValidationTargets,
    pub description: &'static str,
}

/// Quantization type for cross-validation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationType {
    I2S,
    TL1,
    TL2,
}

/// Model configuration for cross-validation testing
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub architecture: &'static str,
    pub vocab_size: u32,
    pub embedding_dim: u32,
    pub num_layers: u32,
    pub num_heads: u32,
    pub max_seq_len: u32,
    pub quantization_block_size: usize,
}

/// Validation targets for cross-validation
#[derive(Debug, Clone, Copy)]
pub struct ValidationTargets {
    pub min_correlation: f32,          // >99.5%
    pub max_mse: f32,                  // <1e-5
    pub max_performance_variance: f32, // <5%
    pub numerical_tolerance: f32,      // <1e-6
}

impl Default for ValidationTargets {
    fn default() -> Self {
        Self {
            min_correlation: 0.995,
            max_mse: 1e-5,
            max_performance_variance: 0.05,
            numerical_tolerance: 1e-6,
        }
    }
}

/// Cross-validation report structure
#[derive(Debug, Clone)]
pub struct CrossValReport {
    pub test_id: String,
    pub passed: bool,
    pub correlation: f32,
    pub mse: f32,
    pub max_abs_error: f32,
    pub rust_performance_tokens_per_sec: f32,
    pub cpp_performance_tokens_per_sec: f32,
    pub performance_variance: f32,
    pub failure_reasons: Vec<String>,
}

/// Quantization accuracy comparison fixture
#[derive(Debug, Clone)]
pub struct QuantizationAccuracyFixture {
    pub test_id: &'static str,
    pub quantization_type: QuantizationType,
    pub input_fp32: Vec<f32>,
    pub rust_quantized: Vec<i8>,
    pub rust_scales: Vec<f32>,
    pub cpp_quantized: Vec<i8>,
    pub cpp_scales: Vec<f32>,
    pub target_accuracy: f32,
    pub tolerance: f32,
    pub description: &'static str,
}

// ============================================================================
// I2S Cross-Validation Fixtures (AC9)
// ============================================================================

/// Load I2S cross-validation fixtures with C++ reference data
#[cfg(feature = "crossval")]
pub fn load_i2s_crossval_fixtures() -> Vec<CrossValFixture> {
    vec![
        // Basic I2S cross-validation
        CrossValFixture {
            test_id: "i2s_crossval_basic",
            input_prompt: "The quick brown fox",
            input_tokens: vec![464, 2068, 17354, 22004], // Example token IDs
            rust_expected_output: generate_rust_i2s_output(4, 42),
            cpp_reference_output: generate_cpp_reference_output(4, 42),
            quantization_type: QuantizationType::I2S,
            model_config: ModelConfig {
                architecture: "bitnet",
                vocab_size: 32000,
                embedding_dim: 768,
                num_layers: 2,
                num_heads: 8,
                max_seq_len: 512,
                quantization_block_size: 32,
            },
            validation_targets: ValidationTargets::default(),
            description: "Basic I2S cross-validation with C++ reference",
        },
        // I2S cross-validation with longer sequence
        CrossValFixture {
            test_id: "i2s_crossval_long_sequence",
            input_prompt: "In a world where artificial intelligence",
            input_tokens: vec![512, 263, 3186, 988, 23116, 21082],
            rust_expected_output: generate_rust_i2s_output(6, 123),
            cpp_reference_output: generate_cpp_reference_output(6, 123),
            quantization_type: QuantizationType::I2S,
            model_config: ModelConfig {
                architecture: "bitnet",
                vocab_size: 32000,
                embedding_dim: 1024,
                num_layers: 4,
                num_heads: 16,
                max_seq_len: 1024,
                quantization_block_size: 64,
            },
            validation_targets: ValidationTargets {
                min_correlation: 0.998,
                max_mse: 5e-6,
                max_performance_variance: 0.03,
                numerical_tolerance: 5e-7,
            },
            description: "I2S cross-validation with longer input sequence",
        },
        // I2S cross-validation with deterministic seed
        CrossValFixture {
            test_id: "i2s_crossval_deterministic",
            input_prompt: "Test deterministic inference",
            input_tokens: vec![4321, 8765, 1234],
            rust_expected_output: generate_deterministic_output(3, 42),
            cpp_reference_output: generate_deterministic_output(3, 42), // Should be identical
            quantization_type: QuantizationType::I2S,
            model_config: ModelConfig {
                architecture: "bitnet",
                vocab_size: 32000,
                embedding_dim: 768,
                num_layers: 2,
                num_heads: 8,
                max_seq_len: 512,
                quantization_block_size: 32,
            },
            validation_targets: ValidationTargets {
                min_correlation: 1.0, // Should be exact with deterministic mode
                max_mse: 1e-8,
                max_performance_variance: 0.05,
                numerical_tolerance: 1e-9,
            },
            description: "I2S cross-validation with BITNET_DETERMINISTIC=1",
        },
    ]
}

/// Load TL cross-validation fixtures (TL1/TL2)
#[cfg(feature = "crossval")]
pub fn load_tl_crossval_fixtures() -> Vec<CrossValFixture> {
    vec![
        // TL1 cross-validation (ARM NEON)
        CrossValFixture {
            test_id: "tl1_crossval_basic",
            input_prompt: "ARM NEON test",
            input_tokens: vec![1234, 5678, 9012],
            rust_expected_output: generate_rust_tl_output(3, 42),
            cpp_reference_output: generate_cpp_reference_output(3, 42),
            quantization_type: QuantizationType::TL1,
            model_config: ModelConfig {
                architecture: "bitnet",
                vocab_size: 32000,
                embedding_dim: 768,
                num_layers: 2,
                num_heads: 8,
                max_seq_len: 512,
                quantization_block_size: 64,
            },
            validation_targets: ValidationTargets {
                min_correlation: 0.996, // TL1 target ≥99.6%
                max_mse: 1e-5,
                max_performance_variance: 0.05,
                numerical_tolerance: 1e-6,
            },
            description: "TL1 cross-validation for ARM NEON optimization",
        },
        // TL2 cross-validation (x86 AVX2/AVX-512)
        CrossValFixture {
            test_id: "tl2_crossval_basic",
            input_prompt: "x86 AVX test",
            input_tokens: vec![9876, 5432, 1098],
            rust_expected_output: generate_rust_tl_output(3, 123),
            cpp_reference_output: generate_cpp_reference_output(3, 123),
            quantization_type: QuantizationType::TL2,
            model_config: ModelConfig {
                architecture: "bitnet",
                vocab_size: 32000,
                embedding_dim: 768,
                num_layers: 2,
                num_heads: 8,
                max_seq_len: 512,
                quantization_block_size: 64,
            },
            validation_targets: ValidationTargets {
                min_correlation: 0.996, // TL2 target ≥99.6%
                max_mse: 1e-5,
                max_performance_variance: 0.05,
                numerical_tolerance: 1e-6,
            },
            description: "TL2 cross-validation for x86 AVX optimization",
        },
    ]
}

// ============================================================================
// Quantization Accuracy Cross-Validation Fixtures
// ============================================================================

/// Load quantization accuracy cross-validation fixtures
#[cfg(feature = "crossval")]
pub fn load_quantization_accuracy_fixtures() -> Vec<QuantizationAccuracyFixture> {
    vec![
        // I2S accuracy validation
        QuantizationAccuracyFixture {
            test_id: "i2s_accuracy_crossval",
            quantization_type: QuantizationType::I2S,
            input_fp32: generate_fp32_weights(512, 42),
            rust_quantized: generate_i2s_quantized(512, 42),
            rust_scales: generate_scales(512, 32, 42),
            cpp_quantized: generate_i2s_quantized(512, 42), // Should match
            cpp_scales: generate_scales(512, 32, 42),
            target_accuracy: 0.998, // ≥99.8% for I2S
            tolerance: 1e-3,
            description: "I2S quantization accuracy cross-validation vs C++ reference",
        },
        // TL1 accuracy validation
        QuantizationAccuracyFixture {
            test_id: "tl1_accuracy_crossval",
            quantization_type: QuantizationType::TL1,
            input_fp32: generate_fp32_weights(512, 123),
            rust_quantized: generate_tl_quantized(512, 123),
            rust_scales: generate_scales(512, 64, 123),
            cpp_quantized: generate_tl_quantized(512, 123),
            cpp_scales: generate_scales(512, 64, 123),
            target_accuracy: 0.996, // ≥99.6% for TL1
            tolerance: 1e-2,
            description: "TL1 quantization accuracy cross-validation vs C++ reference",
        },
        // TL2 accuracy validation
        QuantizationAccuracyFixture {
            test_id: "tl2_accuracy_crossval",
            quantization_type: QuantizationType::TL2,
            input_fp32: generate_fp32_weights(512, 456),
            rust_quantized: generate_tl_quantized(512, 456),
            rust_scales: generate_scales(512, 64, 456),
            cpp_quantized: generate_tl_quantized(512, 456),
            cpp_scales: generate_scales(512, 64, 456),
            target_accuracy: 0.996, // ≥99.6% for TL2
            tolerance: 1e-2,
            description: "TL2 quantization accuracy cross-validation vs C++ reference",
        },
    ]
}

// ============================================================================
// Helper Functions for Test Data Generation
// ============================================================================

/// Generate Rust I2S inference output (mock for testing)
fn generate_rust_i2s_output(length: usize, seed: u64) -> Vec<f32> {
    generate_deterministic_logits(length, seed)
}

/// Generate Rust TL inference output
fn generate_rust_tl_output(length: usize, seed: u64) -> Vec<f32> {
    generate_deterministic_logits(length, seed)
}

/// Generate C++ reference output (mock - should be identical with deterministic seed)
fn generate_cpp_reference_output(length: usize, seed: u64) -> Vec<f32> {
    // In real implementation, this would load from actual C++ reference
    // For testing, we generate slightly different values to simulate comparison
    let mut output = generate_deterministic_logits(length, seed);

    // Add small numerical differences to simulate cross-validation
    let mut rng_state = seed.wrapping_mul(7);
    for val in output.iter_mut() {
        let noise = (lcg_random(&mut rng_state) - 0.5) * 1e-6;
        *val += noise;
    }

    output
}

/// Generate deterministic output (should be identical for Rust and C++)
fn generate_deterministic_output(length: usize, seed: u64) -> Vec<f32> {
    generate_deterministic_logits(length, seed)
}

/// Generate deterministic logits
fn generate_deterministic_logits(length: usize, seed: u64) -> Vec<f32> {
    let mut logits = Vec::with_capacity(length);
    let mut rng_state = seed;

    for _ in 0..length {
        let logit = -5.0 + 10.0 * lcg_random(&mut rng_state);
        logits.push(logit);
    }

    logits
}

/// Generate FP32 weights for accuracy testing
fn generate_fp32_weights(size: usize, seed: u64) -> Vec<f32> {
    let mut weights = Vec::with_capacity(size);
    let mut rng_state = seed;

    for _ in 0..size {
        let weight = -1.0 + 2.0 * lcg_random(&mut rng_state);
        weights.push(weight * 0.1); // Scale to typical weight range
    }

    weights
}

/// Generate I2S quantized values
fn generate_i2s_quantized(size: usize, seed: u64) -> Vec<i8> {
    let mut quantized = Vec::with_capacity(size);
    let mut rng_state = seed.wrapping_mul(2);

    for _ in 0..size {
        let val = match (lcg_random(&mut rng_state) * 4.0) as u32 % 4 {
            0 => -2,
            1 => -1,
            2 => 0,
            _ => 1,
        };
        quantized.push(val);
    }

    quantized
}

/// Generate TL quantized values
fn generate_tl_quantized(size: usize, seed: u64) -> Vec<i8> {
    let mut quantized = Vec::with_capacity(size);
    let mut rng_state = seed.wrapping_mul(3);

    for _ in 0..size {
        let val = match (lcg_random(&mut rng_state) * 4.0) as u32 % 4 {
            0 => -2,
            1 => -1,
            2 => 0,
            _ => 1,
        };
        quantized.push(val);
    }

    quantized
}

/// Generate scale factors
fn generate_scales(size: usize, block_size: usize, seed: u64) -> Vec<f32> {
    let num_blocks = size.div_ceil(block_size);
    let mut scales = Vec::with_capacity(num_blocks);
    let mut rng_state = seed.wrapping_mul(5);

    for _ in 0..num_blocks {
        let scale = 0.01 + lcg_random(&mut rng_state) * 0.15;
        scales.push(scale);
    }

    scales
}

/// Linear congruential generator for deterministic testing
// Use LCG random from helpers to avoid duplication (now with proper clamping for Box-Muller)
use crate::helpers::issue_261_test_helpers::lcg_random;

// ============================================================================
// Cross-Validation Utilities
// ============================================================================

// Re-export accuracy calculation functions from helpers to avoid duplication
pub use crate::helpers::issue_261_test_helpers::{
    calculate_correlation, calculate_max_abs_error, calculate_mse,
};

/// Validate cross-validation results
pub fn validate_crossval_results(fixture: &CrossValFixture) -> CrossValReport {
    let correlation =
        calculate_correlation(&fixture.rust_expected_output, &fixture.cpp_reference_output);
    let mse = calculate_mse(&fixture.rust_expected_output, &fixture.cpp_reference_output);
    let max_abs_error =
        calculate_max_abs_error(&fixture.rust_expected_output, &fixture.cpp_reference_output);

    let mut failure_reasons = Vec::new();
    let mut passed = true;

    if correlation < fixture.validation_targets.min_correlation {
        failure_reasons.push(format!(
            "Correlation {:.4} below minimum {:.4}",
            correlation, fixture.validation_targets.min_correlation
        ));
        passed = false;
    }

    if mse > fixture.validation_targets.max_mse {
        failure_reasons.push(format!(
            "MSE {:.2e} exceeds maximum {:.2e}",
            mse, fixture.validation_targets.max_mse
        ));
        passed = false;
    }

    CrossValReport {
        test_id: fixture.test_id.to_string(),
        passed,
        correlation,
        mse,
        max_abs_error,
        rust_performance_tokens_per_sec: 0.0, // Set by actual test
        cpp_performance_tokens_per_sec: 0.0,  // Set by actual test
        performance_variance: 0.0,            // Set by actual test
        failure_reasons,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correlation_calculation() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let corr = calculate_correlation(&a, &b);
        assert!((corr - 1.0).abs() < 1e-6, "Perfect correlation should be 1.0");
    }

    #[test]
    fn test_mse_calculation() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let mse = calculate_mse(&a, &b);
        assert!(mse < 1e-10, "Identical vectors should have near-zero MSE");
    }

    #[test]
    #[cfg(feature = "crossval")]
    fn test_crossval_fixture_validation() {
        let fixtures = load_i2s_crossval_fixtures();
        for fixture in fixtures {
            let report = validate_crossval_results(&fixture);
            // With deterministic generation, correlation should be very high
            assert!(report.correlation > 0.99, "Correlation should be >99%");
        }
    }

    #[test]
    fn test_deterministic_output_generation() {
        let output1 = generate_deterministic_output(100, 42);
        let output2 = generate_deterministic_output(100, 42);
        assert_eq!(output1, output2, "Deterministic outputs should be identical");
    }
}
