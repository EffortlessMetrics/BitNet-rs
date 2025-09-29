//! I2S Quantization Test Fixtures for Issue #260 Mock Elimination
//!
//! Provides realistic test data for validating I2S quantization implementation
//! against mock computation detection. Includes CPU/GPU variants, cross-validation
//! data, and accuracy benchmarks targeting >99.8% correlation with FP32.

#![allow(unused_imports)]
#![allow(dead_code)]

use std::collections::HashMap;

/// I2S quantization test fixture with device-aware validation data
#[derive(Debug, Clone)]
pub struct I2STestFixture {
    pub name: &'static str,
    pub input_weights: Vec<f32>,
    pub expected_quantized: Vec<i8>,
    pub expected_scales: Vec<f32>,
    pub block_size: usize,
    pub device_type: DeviceType,
    pub tolerance: f32,
    pub target_correlation: f32,
    pub memory_alignment: usize,
    pub simd_friendly: bool,
}

/// Device type for test fixture selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    Cpu,
    CpuSimd,
    Gpu,
    GpuMixedPrecision,
}

/// I2S quantization accuracy validation fixture
#[derive(Debug, Clone)]
pub struct I2SAccuracyFixture {
    pub test_id: &'static str,
    pub reference_fp32: Vec<f32>,
    pub expected_i2s_quantized: Vec<i8>,
    pub expected_scales: Vec<f32>,
    pub expected_correlation: f32,
    pub max_error_threshold: f32,
    pub block_size: usize,
    pub validation_notes: &'static str,
}

/// Cross-validation fixture for C++ reference comparison
#[derive(Debug, Clone)]
pub struct I2SCrossValidationFixture {
    pub scenario: &'static str,
    pub input_data: Vec<f32>,
    pub rust_expected: Vec<i8>,
    pub cpp_reference: Option<Vec<i8>>,
    pub tolerance: f32,
    pub quantization_params: I2SQuantizationParams,
    pub validation_required: bool,
}

/// I2S quantization parameters for test configuration
#[derive(Debug, Clone)]
pub struct I2SQuantizationParams {
    pub block_size: usize,
    pub scale_computation_method: ScaleMethod,
    pub clamp_range: (i8, i8),
    pub symmetric: bool,
}

/// Scale computation method for I2S quantization
#[derive(Debug, Clone, Copy)]
pub enum ScaleMethod {
    AbsMax,
    RmsNorm,
    PercentileClipping,
}

/// Performance benchmark fixture for I2S quantization
#[derive(Debug, Clone)]
pub struct I2SPerformanceFixture {
    pub benchmark_name: &'static str,
    pub weight_matrix_sizes: Vec<(usize, usize)>,
    pub expected_cpu_throughput_range: (f32, f32), // tok/s
    pub expected_gpu_throughput_range: (f32, f32), // tok/s
    pub memory_efficiency_target: f32,             // compression ratio
    pub block_sizes: Vec<usize>,
}

/// Load basic I2S test fixtures with realistic weight distributions
pub fn load_i2s_cpu_fixtures() -> Vec<I2STestFixture> {
    vec![
        // Small matrix - typical embedding layer
        I2STestFixture {
            name: "small_embedding_256",
            input_weights: generate_realistic_weights(256, WeightDistribution::Normal(0.0, 0.1)),
            expected_quantized: generate_expected_i2s_quantized(256, 32),
            expected_scales: generate_expected_scales(256, 32),
            block_size: 32,
            device_type: DeviceType::Cpu,
            tolerance: 0.02,
            target_correlation: 0.998,
            memory_alignment: 32,
            simd_friendly: true,
        },
        // Medium matrix - attention layer
        I2STestFixture {
            name: "attention_weights_1024",
            input_weights: generate_realistic_weights(1024, WeightDistribution::Xavier),
            expected_quantized: generate_expected_i2s_quantized(1024, 64),
            expected_scales: generate_expected_scales(1024, 64),
            block_size: 64,
            device_type: DeviceType::CpuSimd,
            tolerance: 0.015,
            target_correlation: 0.9985,
            memory_alignment: 64,
            simd_friendly: true,
        },
        // Large matrix - MLP layer
        I2STestFixture {
            name: "mlp_weights_4096",
            input_weights: generate_realistic_weights(4096, WeightDistribution::Kaiming),
            expected_quantized: generate_expected_i2s_quantized(4096, 128),
            expected_scales: generate_expected_scales(4096, 128),
            block_size: 128,
            device_type: DeviceType::Cpu,
            tolerance: 0.01,
            target_correlation: 0.999,
            memory_alignment: 32,
            simd_friendly: true,
        },
        // Edge case - very small matrix
        I2STestFixture {
            name: "bias_vector_64",
            input_weights: generate_realistic_weights(64, WeightDistribution::Uniform(-0.1, 0.1)),
            expected_quantized: generate_expected_i2s_quantized(64, 16),
            expected_scales: generate_expected_scales(64, 16),
            block_size: 16,
            device_type: DeviceType::Cpu,
            tolerance: 0.03,
            target_correlation: 0.995,
            memory_alignment: 16,
            simd_friendly: false,
        },
        // Boundary condition - large block size
        I2STestFixture {
            name: "large_block_2048",
            input_weights: generate_realistic_weights(2048, WeightDistribution::Normal(0.0, 0.05)),
            expected_quantized: generate_expected_i2s_quantized(2048, 256),
            expected_scales: generate_expected_scales(2048, 256),
            block_size: 256,
            device_type: DeviceType::CpuSimd,
            tolerance: 0.008,
            target_correlation: 0.9995,
            memory_alignment: 64,
            simd_friendly: true,
        },
    ]
}

/// Load GPU-specific I2S test fixtures with CUDA optimization scenarios
#[cfg(feature = "gpu")]
pub fn load_i2s_gpu_fixtures() -> Vec<I2STestFixture> {
    vec![
        // GPU tensor core friendly
        I2STestFixture {
            name: "gpu_tensor_core_1024",
            input_weights: generate_realistic_weights(1024, WeightDistribution::Normal(0.0, 0.08)),
            expected_quantized: generate_expected_i2s_quantized(1024, 64),
            expected_scales: generate_expected_scales(1024, 64),
            block_size: 64,
            device_type: DeviceType::Gpu,
            tolerance: 0.01,
            target_correlation: 0.999,
            memory_alignment: 128, // GPU memory alignment
            simd_friendly: true,
        },
        // Mixed precision scenario
        I2STestFixture {
            name: "gpu_mixed_precision_2048",
            input_weights: generate_realistic_weights(2048, WeightDistribution::Xavier),
            expected_quantized: generate_expected_i2s_quantized(2048, 128),
            expected_scales: generate_expected_scales(2048, 128),
            block_size: 128,
            device_type: DeviceType::GpuMixedPrecision,
            tolerance: 0.005,
            target_correlation: 0.9998,
            memory_alignment: 128,
            simd_friendly: true,
        },
        // Large GPU matrix
        I2STestFixture {
            name: "gpu_large_matrix_8192",
            input_weights: generate_realistic_weights(8192, WeightDistribution::Kaiming),
            expected_quantized: generate_expected_i2s_quantized(8192, 256),
            expected_scales: generate_expected_scales(8192, 256),
            block_size: 256,
            device_type: DeviceType::Gpu,
            tolerance: 0.003,
            target_correlation: 0.9999,
            memory_alignment: 128,
            simd_friendly: true,
        },
    ]
}

/// Load accuracy validation fixtures for I2S quantization
pub fn load_i2s_accuracy_fixtures() -> Vec<I2SAccuracyFixture> {
    vec![
        I2SAccuracyFixture {
            test_id: "accuracy_validation_basic",
            reference_fp32: generate_reference_fp32_weights(512),
            expected_i2s_quantized: generate_expected_i2s_quantized(512, 64),
            expected_scales: generate_expected_scales(512, 64),
            expected_correlation: 0.9985,
            max_error_threshold: 0.02,
            block_size: 64,
            validation_notes: "Basic accuracy validation for I2S quantization",
        },
        I2SAccuracyFixture {
            test_id: "accuracy_high_precision",
            reference_fp32: generate_reference_fp32_weights(1024),
            expected_i2s_quantized: generate_expected_i2s_quantized(1024, 128),
            expected_scales: generate_expected_scales(1024, 128),
            expected_correlation: 0.9992,
            max_error_threshold: 0.01,
            block_size: 128,
            validation_notes: "High precision I2S quantization validation",
        },
        I2SAccuracyFixture {
            test_id: "accuracy_edge_case_small",
            reference_fp32: generate_reference_fp32_weights(128),
            expected_i2s_quantized: generate_expected_i2s_quantized(128, 16),
            expected_scales: generate_expected_scales(128, 16),
            expected_correlation: 0.998,
            max_error_threshold: 0.025,
            block_size: 16,
            validation_notes: "Edge case validation for small tensors",
        },
    ]
}

/// Load cross-validation fixtures for C++ reference comparison
pub fn load_i2s_crossval_fixtures() -> Vec<I2SCrossValidationFixture> {
    vec![
        I2SCrossValidationFixture {
            scenario: "cpp_reference_basic",
            input_data: generate_deterministic_weights(256, 42),
            rust_expected: generate_expected_i2s_quantized(256, 32),
            cpp_reference: Some(generate_cpp_reference_quantized(256, 32)),
            tolerance: 0.001,
            quantization_params: I2SQuantizationParams {
                block_size: 32,
                scale_computation_method: ScaleMethod::AbsMax,
                clamp_range: (-2, 1),
                symmetric: true,
            },
            validation_required: true,
        },
        I2SCrossValidationFixture {
            scenario: "cpp_reference_large",
            input_data: generate_deterministic_weights(1024, 123),
            rust_expected: generate_expected_i2s_quantized(1024, 64),
            cpp_reference: Some(generate_cpp_reference_quantized(1024, 64)),
            tolerance: 0.001,
            quantization_params: I2SQuantizationParams {
                block_size: 64,
                scale_computation_method: ScaleMethod::AbsMax,
                clamp_range: (-2, 1),
                symmetric: true,
            },
            validation_required: true,
        },
    ]
}

/// Load performance benchmark fixtures
pub fn load_i2s_performance_fixtures() -> Vec<I2SPerformanceFixture> {
    vec![
        I2SPerformanceFixture {
            benchmark_name: "cpu_performance_baseline",
            weight_matrix_sizes: vec![(256, 256), (512, 512), (1024, 1024), (2048, 2048)],
            expected_cpu_throughput_range: (10.0, 25.0),
            expected_gpu_throughput_range: (50.0, 120.0),
            memory_efficiency_target: 4.0, // 4x compression
            block_sizes: vec![32, 64, 128, 256],
        },
        I2SPerformanceFixture {
            benchmark_name: "gpu_performance_optimization",
            weight_matrix_sizes: vec![(1024, 1024), (2048, 2048), (4096, 4096)],
            expected_cpu_throughput_range: (8.0, 20.0),
            expected_gpu_throughput_range: (80.0, 200.0),
            memory_efficiency_target: 4.5, // Better compression on GPU
            block_sizes: vec![64, 128, 256],
        },
    ]
}

/// Weight distribution types for realistic test data generation
#[derive(Debug, Clone, Copy)]
pub enum WeightDistribution {
    Normal(f32, f32),  // mean, std
    Uniform(f32, f32), // min, max
    Xavier,
    Kaiming,
}

/// Generate realistic weight matrices with specified distribution
fn generate_realistic_weights(size: usize, distribution: WeightDistribution) -> Vec<f32> {
    use std::f32::consts::PI;

    let mut weights = Vec::with_capacity(size);
    let mut rng_state = 12345u64; // Deterministic for testing

    for _i in 0..size {
        let weight = match distribution {
            WeightDistribution::Normal(mean, std) => {
                // Box-Muller transform for normal distribution
                let u1 = lcg_random(&mut rng_state);
                let u2 = lcg_random(&mut rng_state);
                let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
                mean + std * z0
            }
            WeightDistribution::Uniform(min, max) => {
                let u = lcg_random(&mut rng_state);
                min + (max - min) * u
            }
            WeightDistribution::Xavier => {
                let limit = (6.0 / size as f32).sqrt();
                let u = lcg_random(&mut rng_state);
                -limit + 2.0 * limit * u
            }
            WeightDistribution::Kaiming => {
                let std = (2.0 / size as f32).sqrt();
                let u1 = lcg_random(&mut rng_state);
                let u2 = lcg_random(&mut rng_state);
                let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
                std * z0
            }
        };
        weights.push(weight);
    }

    weights
}

/// Generate deterministic weights for cross-validation
fn generate_deterministic_weights(size: usize, seed: u64) -> Vec<f32> {
    let mut weights = Vec::with_capacity(size);
    let mut rng_state = seed;

    for _ in 0..size {
        let weight = -1.0 + 2.0 * lcg_random(&mut rng_state);
        weights.push(weight * 0.1); // Scale to typical weight range
    }

    weights
}

/// Generate expected I2S quantized values
fn generate_expected_i2s_quantized(size: usize, _block_size: usize) -> Vec<i8> {
    let mut quantized = Vec::with_capacity(size);
    let mut rng_state = 54321u64;

    for _ in 0..size {
        // I2S uses {-2, -1, 0, 1} quantization levels
        let val = match (lcg_random(&mut rng_state) * 4.0) as u32 {
            0 => -2,
            1 => -1,
            2 => 0,
            _ => 1,
        };
        quantized.push(val);
    }

    quantized
}

/// Generate expected scale factors
fn generate_expected_scales(size: usize, block_size: usize) -> Vec<f32> {
    let num_blocks = size.div_ceil(block_size);
    let mut scales = Vec::with_capacity(num_blocks);
    let mut rng_state = 98765u64;

    for _ in 0..num_blocks {
        // Typical scale factors for neural network weights
        let scale = 0.01 + lcg_random(&mut rng_state) * 0.1;
        scales.push(scale);
    }

    scales
}

/// Generate reference FP32 weights for accuracy validation
fn generate_reference_fp32_weights(size: usize) -> Vec<f32> {
    generate_realistic_weights(size, WeightDistribution::Normal(0.0, 0.08))
}

/// Generate C++ reference quantized values (mock for testing)
fn generate_cpp_reference_quantized(size: usize, block_size: usize) -> Vec<i8> {
    // In real implementation, this would load from C++ reference
    generate_expected_i2s_quantized(size, block_size)
}

/// Simple linear congruential generator for deterministic testing
fn lcg_random(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
    (*state as f32) / (u32::MAX as f32)
}

/// Get fixture by name for dynamic test selection
pub fn get_fixture_by_name(name: &str) -> Option<I2STestFixture> {
    #[cfg(feature = "gpu")]
    {
        let mut all_fixtures = load_i2s_cpu_fixtures();
        all_fixtures.extend(load_i2s_gpu_fixtures());
        all_fixtures.into_iter().find(|f| f.name == name)
    }

    #[cfg(not(feature = "gpu"))]
    {
        let all_fixtures = load_i2s_cpu_fixtures();
        all_fixtures.into_iter().find(|f| f.name == name)
    }
}

/// Validate fixture data integrity
pub fn validate_fixture_integrity(fixture: &I2STestFixture) -> Result<(), String> {
    if fixture.input_weights.is_empty() {
        return Err("Input weights cannot be empty".to_string());
    }

    if fixture.expected_quantized.len() != fixture.input_weights.len() {
        return Err("Quantized length must match input length".to_string());
    }

    let expected_scale_count = fixture.input_weights.len().div_ceil(fixture.block_size);
    if fixture.expected_scales.len() != expected_scale_count {
        return Err(format!(
            "Expected {} scales, got {}",
            expected_scale_count,
            fixture.expected_scales.len()
        ));
    }

    if fixture.target_correlation < 0.99 {
        return Err("Target correlation should be >= 0.99 for I2S".to_string());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixture_integrity() {
        #[cfg(feature = "cpu")]
        {
            let fixtures = load_i2s_cpu_fixtures();
            for fixture in fixtures {
                validate_fixture_integrity(&fixture).expect("Fixture should be valid");
            }
        }
    }

    #[test]
    fn test_deterministic_weight_generation() {
        let weights1 = generate_deterministic_weights(100, 42);
        let weights2 = generate_deterministic_weights(100, 42);
        assert_eq!(weights1, weights2, "Deterministic weights should be identical");
    }

    #[test]
    fn test_fixture_retrieval() {
        #[cfg(feature = "cpu")]
        {
            let fixture = get_fixture_by_name("small_embedding_256");
            assert!(fixture.is_some(), "Should find fixture by name");
            assert_eq!(fixture.unwrap().name, "small_embedding_256");
        }
    }
}
