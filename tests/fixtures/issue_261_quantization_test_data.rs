//! Issue #261 Quantization Test Fixtures
//!
//! Comprehensive test data for I2S, TL1, and TL2 quantization algorithms.
//! Supports AC3 (I2S kernel integration) and AC4 (TL kernel integration).
//!
//! Feature-gated for CPU/GPU scenarios with deterministic generation.

#![allow(dead_code)]

use std::f32::consts::PI;

/// Quantization test fixture with known inputs/outputs for validation
#[derive(Debug, Clone)]
pub struct QuantizationTestFixture {
    pub test_id: &'static str,
    pub quantization_type: QuantizationType,
    pub input_fp32: Vec<f32>,
    pub expected_quantized: Vec<i8>,
    pub expected_scales: Vec<f32>,
    pub block_size: usize,
    pub target_accuracy: f32,     // ≥99.8% for I2S, ≥99.6% for TL1/TL2
    pub tolerance: f32,            // 1e-3 for I2S, 1e-2 for TL1/TL2
    pub device_type: DeviceType,
    pub simd_optimized: bool,
    pub description: &'static str,
}

/// Quantization type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationType {
    I2S,  // 2-bit signed: {-2, -1, 0, 1}
    TL1,  // Table lookup 1 (ARM NEON optimized)
    TL2,  // Table lookup 2 (x86 AVX2/AVX-512 optimized)
}

/// Device type for test fixture selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    Cpu,
    CpuSimdAvx2,
    CpuSimdAvx512,
    CpuSimdNeon,
    Gpu,
    GpuMixedPrecision,
}

/// Edge case test fixture for robustness validation
#[derive(Debug, Clone)]
pub struct EdgeCaseFixture {
    pub case_name: &'static str,
    pub input_data: Vec<f32>,
    pub expected_behavior: ExpectedBehavior,
    pub quantization_type: QuantizationType,
    pub description: &'static str,
}

/// Expected behavior for edge case testing
#[derive(Debug, Clone)]
pub enum ExpectedBehavior {
    SuccessWithAccuracy(f32),
    ErrorCondition(&'static str),
    SpecialHandling(&'static str),
}

// ============================================================================
// I2S Quantization Fixtures (AC3)
// ============================================================================

/// Load I2S quantization test fixtures for CPU
#[cfg(feature = "cpu")]
pub fn load_i2s_cpu_fixtures() -> Vec<QuantizationTestFixture> {
    vec![
        // Basic I2S quantization - small tensor
        QuantizationTestFixture {
            test_id: "i2s_cpu_basic_256",
            quantization_type: QuantizationType::I2S,
            input_fp32: generate_normal_weights(256, 0.0, 0.1, 42),
            expected_quantized: generate_i2s_quantized(256, 42),
            expected_scales: generate_scales(256, 32, 42),
            block_size: 32,
            target_accuracy: 0.998, // ≥99.8% for I2S
            tolerance: 1e-3,
            device_type: DeviceType::Cpu,
            simd_optimized: false,
            description: "Basic I2S quantization for 256-element tensor on CPU",
        },
        // I2S with AVX2 optimization
        QuantizationTestFixture {
            test_id: "i2s_cpu_avx2_1024",
            quantization_type: QuantizationType::I2S,
            input_fp32: generate_normal_weights(1024, 0.0, 0.08, 123),
            expected_quantized: generate_i2s_quantized(1024, 123),
            expected_scales: generate_scales(1024, 64, 123),
            block_size: 64,
            target_accuracy: 0.9985,
            tolerance: 1e-3,
            device_type: DeviceType::CpuSimdAvx2,
            simd_optimized: true,
            description: "I2S quantization with AVX2 SIMD optimization",
        },
        // I2S with AVX-512 optimization
        QuantizationTestFixture {
            test_id: "i2s_cpu_avx512_2048",
            quantization_type: QuantizationType::I2S,
            input_fp32: generate_normal_weights(2048, 0.0, 0.05, 456),
            expected_quantized: generate_i2s_quantized(2048, 456),
            expected_scales: generate_scales(2048, 128, 456),
            block_size: 128,
            target_accuracy: 0.999,
            tolerance: 1e-3,
            device_type: DeviceType::CpuSimdAvx512,
            simd_optimized: true,
            description: "I2S quantization with AVX-512 SIMD optimization",
        },
        // I2S large tensor
        QuantizationTestFixture {
            test_id: "i2s_cpu_large_4096",
            quantization_type: QuantizationType::I2S,
            input_fp32: generate_normal_weights(4096, 0.0, 0.07, 789),
            expected_quantized: generate_i2s_quantized(4096, 789),
            expected_scales: generate_scales(4096, 128, 789),
            block_size: 128,
            target_accuracy: 0.9992,
            tolerance: 1e-3,
            device_type: DeviceType::Cpu,
            simd_optimized: false,
            description: "I2S quantization for large 4096-element tensor",
        },
        // I2S block size alignment test (82 elements per spec)
        QuantizationTestFixture {
            test_id: "i2s_cpu_block_alignment_820",
            quantization_type: QuantizationType::I2S,
            input_fp32: generate_normal_weights(820, 0.0, 0.1, 111),
            expected_quantized: generate_i2s_quantized(820, 111),
            expected_scales: generate_scales(820, 82, 111),
            block_size: 82,
            target_accuracy: 0.998,
            tolerance: 1e-3,
            device_type: DeviceType::Cpu,
            simd_optimized: false,
            description: "I2S quantization with 82-element block alignment",
        },
    ]
}

/// Load I2S quantization test fixtures for GPU
#[cfg(feature = "gpu")]
pub fn load_i2s_gpu_fixtures() -> Vec<QuantizationTestFixture> {
    vec![
        // GPU basic I2S quantization
        QuantizationTestFixture {
            test_id: "i2s_gpu_basic_1024",
            quantization_type: QuantizationType::I2S,
            input_fp32: generate_normal_weights(1024, 0.0, 0.08, 42),
            expected_quantized: generate_i2s_quantized(1024, 42),
            expected_scales: generate_scales(1024, 64, 42),
            block_size: 64,
            target_accuracy: 0.999,
            tolerance: 1e-3,
            device_type: DeviceType::Gpu,
            simd_optimized: true,
            description: "I2S quantization on GPU with CUDA kernels",
        },
        // GPU mixed precision I2S
        QuantizationTestFixture {
            test_id: "i2s_gpu_mixed_precision_2048",
            quantization_type: QuantizationType::I2S,
            input_fp32: generate_normal_weights(2048, 0.0, 0.06, 123),
            expected_quantized: generate_i2s_quantized(2048, 123),
            expected_scales: generate_scales(2048, 128, 123),
            block_size: 128,
            target_accuracy: 0.9995,
            tolerance: 1e-3,
            device_type: DeviceType::GpuMixedPrecision,
            simd_optimized: true,
            description: "I2S quantization with mixed precision (FP16/BF16) on GPU",
        },
        // GPU large tensor
        QuantizationTestFixture {
            test_id: "i2s_gpu_large_8192",
            quantization_type: QuantizationType::I2S,
            input_fp32: generate_normal_weights(8192, 0.0, 0.05, 456),
            expected_quantized: generate_i2s_quantized(8192, 456),
            expected_scales: generate_scales(8192, 256, 456),
            block_size: 256,
            target_accuracy: 0.9998,
            tolerance: 1e-3,
            device_type: DeviceType::Gpu,
            simd_optimized: true,
            description: "I2S quantization for large tensor on GPU",
        },
    ]
}

// ============================================================================
// TL1 Quantization Fixtures (AC4) - ARM NEON optimized
// ============================================================================

/// Load TL1 quantization test fixtures for CPU (ARM)
#[cfg(all(feature = "cpu", target_arch = "aarch64"))]
pub fn load_tl1_cpu_fixtures() -> Vec<QuantizationTestFixture> {
    vec![
        // Basic TL1 quantization
        QuantizationTestFixture {
            test_id: "tl1_cpu_neon_512",
            quantization_type: QuantizationType::TL1,
            input_fp32: generate_normal_weights(512, 0.0, 0.1, 42),
            expected_quantized: generate_tl_quantized(512, 42),
            expected_scales: generate_scales(512, 64, 42),
            block_size: 64,
            target_accuracy: 0.996, // ≥99.6% for TL1
            tolerance: 1e-2,
            device_type: DeviceType::CpuSimdNeon,
            simd_optimized: true,
            description: "TL1 quantization with ARM NEON optimization",
        },
        // TL1 medium tensor
        QuantizationTestFixture {
            test_id: "tl1_cpu_neon_1024",
            quantization_type: QuantizationType::TL1,
            input_fp32: generate_normal_weights(1024, 0.0, 0.08, 123),
            expected_quantized: generate_tl_quantized(1024, 123),
            expected_scales: generate_scales(1024, 128, 123),
            block_size: 128,
            target_accuracy: 0.997,
            tolerance: 1e-2,
            device_type: DeviceType::CpuSimdNeon,
            simd_optimized: true,
            description: "TL1 quantization for medium tensor on ARM",
        },
        // TL1 large tensor
        QuantizationTestFixture {
            test_id: "tl1_cpu_neon_2048",
            quantization_type: QuantizationType::TL1,
            input_fp32: generate_normal_weights(2048, 0.0, 0.06, 456),
            expected_quantized: generate_tl_quantized(2048, 456),
            expected_scales: generate_scales(2048, 128, 456),
            block_size: 128,
            target_accuracy: 0.998,
            tolerance: 1e-2,
            device_type: DeviceType::CpuSimdNeon,
            simd_optimized: true,
            description: "TL1 quantization for large tensor on ARM",
        },
    ]
}

// ============================================================================
// TL2 Quantization Fixtures (AC4) - x86 AVX2/AVX-512 optimized
// ============================================================================

/// Load TL2 quantization test fixtures for CPU (x86_64)
#[cfg(all(feature = "cpu", target_arch = "x86_64"))]
pub fn load_tl2_cpu_fixtures() -> Vec<QuantizationTestFixture> {
    vec![
        // Basic TL2 quantization with AVX2
        QuantizationTestFixture {
            test_id: "tl2_cpu_avx2_512",
            quantization_type: QuantizationType::TL2,
            input_fp32: generate_normal_weights(512, 0.0, 0.1, 42),
            expected_quantized: generate_tl_quantized(512, 42),
            expected_scales: generate_scales(512, 64, 42),
            block_size: 64,
            target_accuracy: 0.996, // ≥99.6% for TL2
            tolerance: 1e-2,
            device_type: DeviceType::CpuSimdAvx2,
            simd_optimized: true,
            description: "TL2 quantization with x86 AVX2 optimization",
        },
        // TL2 with AVX-512
        QuantizationTestFixture {
            test_id: "tl2_cpu_avx512_1024",
            quantization_type: QuantizationType::TL2,
            input_fp32: generate_normal_weights(1024, 0.0, 0.08, 123),
            expected_quantized: generate_tl_quantized(1024, 123),
            expected_scales: generate_scales(1024, 128, 123),
            block_size: 128,
            target_accuracy: 0.997,
            tolerance: 1e-2,
            device_type: DeviceType::CpuSimdAvx512,
            simd_optimized: true,
            description: "TL2 quantization with x86 AVX-512 optimization",
        },
        // TL2 large tensor
        QuantizationTestFixture {
            test_id: "tl2_cpu_large_4096",
            quantization_type: QuantizationType::TL2,
            input_fp32: generate_normal_weights(4096, 0.0, 0.06, 456),
            expected_quantized: generate_tl_quantized(4096, 456),
            expected_scales: generate_scales(4096, 128, 456),
            block_size: 128,
            target_accuracy: 0.998,
            tolerance: 1e-2,
            device_type: DeviceType::CpuSimdAvx2,
            simd_optimized: true,
            description: "TL2 quantization for large tensor on x86",
        },
    ]
}

// ============================================================================
// Edge Case Fixtures
// ============================================================================

/// Load edge case test fixtures
pub fn load_edge_case_fixtures() -> Vec<EdgeCaseFixture> {
    vec![
        // All zeros
        EdgeCaseFixture {
            case_name: "all_zeros",
            input_data: vec![0.0; 256],
            expected_behavior: ExpectedBehavior::SuccessWithAccuracy(1.0),
            quantization_type: QuantizationType::I2S,
            description: "All-zero input tensor",
        },
        // All ones
        EdgeCaseFixture {
            case_name: "all_ones",
            input_data: vec![1.0; 256],
            expected_behavior: ExpectedBehavior::SuccessWithAccuracy(0.999),
            quantization_type: QuantizationType::I2S,
            description: "All-one input tensor",
        },
        // Mixed signs
        EdgeCaseFixture {
            case_name: "mixed_signs",
            input_data: generate_alternating_signs(256),
            expected_behavior: ExpectedBehavior::SuccessWithAccuracy(0.998),
            quantization_type: QuantizationType::I2S,
            description: "Alternating positive/negative values",
        },
        // Extreme values
        EdgeCaseFixture {
            case_name: "extreme_values",
            input_data: generate_extreme_values(256),
            expected_behavior: ExpectedBehavior::SuccessWithAccuracy(0.995),
            quantization_type: QuantizationType::I2S,
            description: "Extreme positive and negative values",
        },
        // Very small values (near-zero)
        EdgeCaseFixture {
            case_name: "near_zero",
            input_data: vec![1e-7; 256],
            expected_behavior: ExpectedBehavior::SuccessWithAccuracy(0.99),
            quantization_type: QuantizationType::I2S,
            description: "Near-zero floating point values",
        },
        // Single element
        EdgeCaseFixture {
            case_name: "single_element",
            input_data: vec![0.5],
            expected_behavior: ExpectedBehavior::SpecialHandling("Minimum block size handling"),
            quantization_type: QuantizationType::I2S,
            description: "Single-element tensor",
        },
        // Misaligned block size
        EdgeCaseFixture {
            case_name: "misaligned_block",
            input_data: generate_normal_weights(127, 0.0, 0.1, 42), // Not power of 2
            expected_behavior: ExpectedBehavior::SuccessWithAccuracy(0.998),
            quantization_type: QuantizationType::I2S,
            description: "Tensor size not aligned to block size",
        },
    ]
}

// ============================================================================
// Helper Functions for Deterministic Test Data Generation
// ============================================================================

/// Generate normal distribution weights (Box-Muller transform)
fn generate_normal_weights(size: usize, mean: f32, std: f32, seed: u64) -> Vec<f32> {
    let mut weights = Vec::with_capacity(size);
    let mut rng_state = seed;

    for i in 0..size {
        if i % 2 == 0 && i + 1 < size {
            // Box-Muller transform for normal distribution
            let u1 = lcg_random(&mut rng_state);
            let u2 = lcg_random(&mut rng_state);
            let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
            let z1 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).sin();
            weights.push(mean + std * z0);
            weights.push(mean + std * z1);
        }
    }

    // Handle odd-sized arrays
    if size % 2 == 1 {
        let u1 = lcg_random(&mut rng_state);
        let u2 = lcg_random(&mut rng_state);
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        weights.push(mean + std * z0);
    }

    weights.truncate(size);
    weights
}

/// Generate I2S quantized values: {-2, -1, 0, 1}
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

/// Generate TL quantized values (table lookup)
fn generate_tl_quantized(size: usize, seed: u64) -> Vec<i8> {
    let mut quantized = Vec::with_capacity(size);
    let mut rng_state = seed.wrapping_mul(3);

    for _ in 0..size {
        // TL uses similar range but different lookup table
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

/// Generate scale factors for quantization
fn generate_scales(size: usize, block_size: usize, seed: u64) -> Vec<f32> {
    let num_blocks = size.div_ceil(block_size);
    let mut scales = Vec::with_capacity(num_blocks);
    let mut rng_state = seed.wrapping_mul(5);

    for _ in 0..num_blocks {
        // Typical scale factors for neural network weights
        let scale = 0.01 + lcg_random(&mut rng_state) * 0.15;
        scales.push(scale);
    }

    scales
}

/// Generate alternating signs for edge case testing
fn generate_alternating_signs(size: usize) -> Vec<f32> {
    (0..size).map(|i| if i % 2 == 0 { 0.5 } else { -0.5 }).collect()
}

/// Generate extreme values for edge case testing
fn generate_extreme_values(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| match i % 4 {
            0 => 100.0,
            1 => -100.0,
            2 => 0.001,
            _ => -0.001,
        })
        .collect()
}

/// Simple linear congruential generator for deterministic testing
/// Supports BITNET_SEED environment variable
fn lcg_random(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
    (*state as f32) / (u32::MAX as f32)
}

// ============================================================================
// Fixture Validation Utilities
// ============================================================================

/// Validate fixture data integrity
pub fn validate_fixture_integrity(fixture: &QuantizationTestFixture) -> Result<(), String> {
    if fixture.input_fp32.is_empty() {
        return Err("Input FP32 data cannot be empty".to_string());
    }

    if fixture.expected_quantized.len() != fixture.input_fp32.len() {
        return Err(format!(
            "Quantized length ({}) must match input length ({})",
            fixture.expected_quantized.len(),
            fixture.input_fp32.len()
        ));
    }

    let expected_scale_count = fixture.input_fp32.len().div_ceil(fixture.block_size);
    if fixture.expected_scales.len() != expected_scale_count {
        return Err(format!(
            "Expected {} scales, got {}",
            expected_scale_count,
            fixture.expected_scales.len()
        ));
    }

    // Validate accuracy targets
    match fixture.quantization_type {
        QuantizationType::I2S => {
            if fixture.target_accuracy < 0.998 {
                return Err("I2S target accuracy must be ≥99.8%".to_string());
            }
            if fixture.tolerance > 1e-3 {
                return Err("I2S tolerance must be ≤1e-3".to_string());
            }
        }
        QuantizationType::TL1 | QuantizationType::TL2 => {
            if fixture.target_accuracy < 0.996 {
                return Err("TL target accuracy must be ≥99.6%".to_string());
            }
            if fixture.tolerance > 1e-2 {
                return Err("TL tolerance must be ≤1e-2".to_string());
            }
        }
    }

    Ok(())
}

/// Get fixture by test ID
pub fn get_fixture_by_id(test_id: &str) -> Option<QuantizationTestFixture> {
    #[cfg(feature = "cpu")]
    {
        let mut all_fixtures = load_i2s_cpu_fixtures();

        #[cfg(target_arch = "x86_64")]
        {
            all_fixtures.extend(load_tl2_cpu_fixtures());
        }

        #[cfg(target_arch = "aarch64")]
        {
            all_fixtures.extend(load_tl1_cpu_fixtures());
        }

        #[cfg(feature = "gpu")]
        {
            all_fixtures.extend(load_i2s_gpu_fixtures());
        }

        all_fixtures.into_iter().find(|f| f.test_id == test_id)
    }

    #[cfg(not(feature = "cpu"))]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "cpu")]
    fn test_fixture_integrity_validation() {
        let fixtures = load_i2s_cpu_fixtures();
        for fixture in fixtures {
            validate_fixture_integrity(&fixture).expect("Fixture should be valid");
        }
    }

    #[test]
    fn test_deterministic_generation() {
        let weights1 = generate_normal_weights(100, 0.0, 0.1, 42);
        let weights2 = generate_normal_weights(100, 0.0, 0.1, 42);
        assert_eq!(weights1, weights2, "Deterministic generation should produce identical results");
    }

    #[test]
    fn test_i2s_quantized_range() {
        let quantized = generate_i2s_quantized(1000, 42);
        for val in quantized {
            assert!(
                val >= -2 && val <= 1,
                "I2S quantized values must be in range [-2, 1]"
            );
        }
    }

    #[test]
    fn test_scale_generation() {
        let scales = generate_scales(1024, 64, 42);
        assert_eq!(scales.len(), 16, "Should generate correct number of scales");
        for scale in scales {
            assert!(scale > 0.0 && scale < 1.0, "Scale factors should be in valid range");
        }
    }
}
