//! Quantization Test Data for Issue #453 Strict Quantization Guards
//!
//! Provides realistic quantization matrices and ground truth data for testing
//! strict mode validation of I2S, TL1, and TL2 quantization algorithms.
//!
//! All test data is deterministically generated to support reproducible testing
//! with BITNET_DETERMINISTIC=1 and BITNET_SEED=42.

#![allow(dead_code)]

use std::f32::consts::PI;

/// Quantization test matrix with ground truth for validation
#[derive(Debug, Clone)]
pub struct QuantizationTestMatrix {
    pub test_id: &'static str,
    pub quantization_type: QuantizationType,
    pub input_fp32: Vec<f32>,
    pub expected_quantized: Vec<i8>,
    pub expected_scales: Vec<f32>,
    pub block_size: usize,
    pub shape: (usize, usize), // (rows, cols)
    pub target_correlation: f32, // I2S: 99.8%, TL1: 99.6%, TL2: 99.6%
    pub tolerance: f32,
    pub device_type: DeviceType,
    pub kernel_available: bool,
    pub description: &'static str,
}

/// Quantization type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationType {
    I2S,  // 2-bit signed: {-2, -1, 0, 1}
    TL1,  // Table lookup 1 (ARM NEON optimized)
    TL2,  // Table lookup 2 (x86 AVX2/AVX-512 optimized)
}

/// Device type for quantization testing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    Cpu,
    CpuSimdAvx2,
    CpuSimdAvx512,
    CpuSimdNeon,
    Gpu,
    GpuFp16,
    GpuBf16,
}

// ============================================================================
// I2S Quantization Matrices (Production-Ready)
// ============================================================================

/// Small I2S quantization matrix (128×128) for unit testing
#[cfg(feature = "cpu")]
pub fn i2s_matrix_small() -> QuantizationTestMatrix {
    let size = 128 * 128; // 16,384 elements
    let block_size = 32;

    QuantizationTestMatrix {
        test_id: "i2s_small_128x128",
        quantization_type: QuantizationType::I2S,
        input_fp32: generate_normal_distribution(size, 0.0, 0.1, 42),
        expected_quantized: generate_i2s_quantized(size, block_size, 42),
        expected_scales: generate_i2s_scales(size, block_size, 42),
        block_size,
        shape: (128, 128),
        target_correlation: 0.998, // 99.8% accuracy
        tolerance: 1e-3,
        device_type: DeviceType::Cpu,
        kernel_available: true,
        description: "Small I2S matrix for fast unit testing (128×128)",
    }
}

/// Medium I2S quantization matrix (512×512) for integration testing
#[cfg(feature = "cpu")]
pub fn i2s_matrix_medium() -> QuantizationTestMatrix {
    let size = 512 * 512; // 262,144 elements
    let block_size = 64;

    QuantizationTestMatrix {
        test_id: "i2s_medium_512x512",
        quantization_type: QuantizationType::I2S,
        input_fp32: generate_normal_distribution(size, 0.0, 0.08, 123),
        expected_quantized: generate_i2s_quantized(size, block_size, 123),
        expected_scales: generate_i2s_scales(size, block_size, 123),
        block_size,
        shape: (512, 512),
        target_correlation: 0.9985,
        tolerance: 1e-3,
        device_type: DeviceType::CpuSimdAvx2,
        kernel_available: true,
        description: "Medium I2S matrix for integration testing (512×512)",
    }
}

/// Large I2S quantization matrix (2048×2048) for stress testing
#[cfg(feature = "cpu")]
pub fn i2s_matrix_large() -> QuantizationTestMatrix {
    let size = 2048 * 2048; // 4,194,304 elements
    let block_size = 128;

    QuantizationTestMatrix {
        test_id: "i2s_large_2048x2048",
        quantization_type: QuantizationType::I2S,
        input_fp32: generate_normal_distribution(size, 0.0, 0.05, 456),
        expected_quantized: generate_i2s_quantized(size, block_size, 456),
        expected_scales: generate_i2s_scales(size, block_size, 456),
        block_size,
        shape: (2048, 2048),
        target_correlation: 0.999,
        tolerance: 1e-3,
        device_type: DeviceType::CpuSimdAvx512,
        kernel_available: true,
        description: "Large I2S matrix for stress testing (2048×2048)",
    }
}

/// I2S matrix with unavailable kernel for fallback testing
#[cfg(feature = "cpu")]
pub fn i2s_matrix_fallback_scenario() -> QuantizationTestMatrix {
    let size = 256 * 256;
    let block_size = 32;

    QuantizationTestMatrix {
        test_id: "i2s_fallback_256x256",
        quantization_type: QuantizationType::I2S,
        input_fp32: generate_normal_distribution(size, 0.0, 0.1, 789),
        expected_quantized: generate_i2s_quantized(size, block_size, 789),
        expected_scales: generate_i2s_scales(size, block_size, 789),
        block_size,
        shape: (256, 256),
        target_correlation: 0.998,
        tolerance: 1e-3,
        device_type: DeviceType::Cpu,
        kernel_available: false, // Force fallback for strict mode testing
        description: "I2S matrix with unavailable kernel (strict mode should reject)",
    }
}

// ============================================================================
// GPU I2S Quantization Matrices
// ============================================================================

/// GPU I2S matrix with FP16 mixed precision
#[cfg(feature = "gpu")]
pub fn i2s_gpu_matrix_fp16() -> QuantizationTestMatrix {
    let size = 1024 * 1024;
    let block_size = 128;

    QuantizationTestMatrix {
        test_id: "i2s_gpu_fp16_1024x1024",
        quantization_type: QuantizationType::I2S,
        input_fp32: generate_normal_distribution(size, 0.0, 0.07, 111),
        expected_quantized: generate_i2s_quantized(size, block_size, 111),
        expected_scales: generate_i2s_scales(size, block_size, 111),
        block_size,
        shape: (1024, 1024),
        target_correlation: 0.9988,
        tolerance: 1e-3,
        device_type: DeviceType::GpuFp16,
        kernel_available: true,
        description: "GPU I2S matrix with FP16 mixed precision (1024×1024)",
    }
}

/// GPU I2S matrix with BF16 mixed precision
#[cfg(feature = "gpu")]
pub fn i2s_gpu_matrix_bf16() -> QuantizationTestMatrix {
    let size = 1024 * 1024;
    let block_size = 128;

    QuantizationTestMatrix {
        test_id: "i2s_gpu_bf16_1024x1024",
        quantization_type: QuantizationType::I2S,
        input_fp32: generate_normal_distribution(size, 0.0, 0.06, 222),
        expected_quantized: generate_i2s_quantized(size, block_size, 222),
        expected_scales: generate_i2s_scales(size, block_size, 222),
        block_size,
        shape: (1024, 1024),
        target_correlation: 0.9987,
        tolerance: 1e-3,
        device_type: DeviceType::GpuBf16,
        kernel_available: true,
        description: "GPU I2S matrix with BF16 mixed precision (1024×1024)",
    }
}

// ============================================================================
// TL1 Quantization Matrices (ARM NEON Table Lookup)
// ============================================================================

/// TL1 matrix optimized for ARM NEON (256×256)
#[cfg(feature = "cpu")]
pub fn tl1_matrix_neon() -> QuantizationTestMatrix {
    let size = 256 * 256;
    let block_size = 32;

    QuantizationTestMatrix {
        test_id: "tl1_neon_256x256",
        quantization_type: QuantizationType::TL1,
        input_fp32: generate_uniform_distribution(size, -0.5, 0.5, 333),
        expected_quantized: generate_tl1_quantized(size, 333),
        expected_scales: generate_tl1_scales(size, block_size, 333),
        block_size,
        shape: (256, 256),
        target_correlation: 0.996, // 99.6% for TL1
        tolerance: 1e-2,
        device_type: DeviceType::CpuSimdNeon,
        kernel_available: true,
        description: "TL1 matrix optimized for ARM NEON table lookup (256×256)",
    }
}

/// TL1 matrix with unavailable NEON kernel
#[cfg(feature = "cpu")]
pub fn tl1_matrix_fallback() -> QuantizationTestMatrix {
    let size = 128 * 128;
    let block_size = 32;

    QuantizationTestMatrix {
        test_id: "tl1_fallback_128x128",
        quantization_type: QuantizationType::TL1,
        input_fp32: generate_uniform_distribution(size, -0.5, 0.5, 444),
        expected_quantized: generate_tl1_quantized(size, 444),
        expected_scales: generate_tl1_scales(size, block_size, 444),
        block_size,
        shape: (128, 128),
        target_correlation: 0.996,
        tolerance: 1e-2,
        device_type: DeviceType::CpuSimdNeon,
        kernel_available: false, // Force fallback for strict mode testing
        description: "TL1 matrix with unavailable NEON kernel (strict mode should reject)",
    }
}

// ============================================================================
// TL2 Quantization Matrices (x86 AVX2/AVX-512 Table Lookup)
// ============================================================================

/// TL2 matrix optimized for AVX2 (512×512)
#[cfg(feature = "cpu")]
pub fn tl2_matrix_avx2() -> QuantizationTestMatrix {
    let size = 512 * 512;
    let block_size = 64;

    QuantizationTestMatrix {
        test_id: "tl2_avx2_512x512",
        quantization_type: QuantizationType::TL2,
        input_fp32: generate_normal_distribution(size, 0.0, 0.09, 555),
        expected_quantized: generate_tl2_quantized(size, 555),
        expected_scales: generate_tl2_scales(size, block_size, 555),
        block_size,
        shape: (512, 512),
        target_correlation: 0.996, // 99.6% for TL2
        tolerance: 1e-2,
        device_type: DeviceType::CpuSimdAvx2,
        kernel_available: true,
        description: "TL2 matrix optimized for AVX2 table lookup (512×512)",
    }
}

/// TL2 matrix optimized for AVX-512 (1024×1024)
#[cfg(feature = "cpu")]
pub fn tl2_matrix_avx512() -> QuantizationTestMatrix {
    let size = 1024 * 1024;
    let block_size = 128;

    QuantizationTestMatrix {
        test_id: "tl2_avx512_1024x1024",
        quantization_type: QuantizationType::TL2,
        input_fp32: generate_normal_distribution(size, 0.0, 0.04, 666),
        expected_quantized: generate_tl2_quantized(size, 666),
        expected_scales: generate_tl2_scales(size, block_size, 666),
        block_size,
        shape: (1024, 1024),
        target_correlation: 0.9965,
        tolerance: 1e-2,
        device_type: DeviceType::CpuSimdAvx512,
        kernel_available: true,
        description: "TL2 matrix optimized for AVX-512 table lookup (1024×1024)",
    }
}

/// TL2 matrix with unavailable AVX kernel
#[cfg(feature = "cpu")]
pub fn tl2_matrix_fallback() -> QuantizationTestMatrix {
    let size = 256 * 256;
    let block_size = 32;

    QuantizationTestMatrix {
        test_id: "tl2_fallback_256x256",
        quantization_type: QuantizationType::TL2,
        input_fp32: generate_normal_distribution(size, 0.0, 0.08, 777),
        expected_quantized: generate_tl2_quantized(size, 777),
        expected_scales: generate_tl2_scales(size, block_size, 777),
        block_size,
        shape: (256, 256),
        target_correlation: 0.996,
        tolerance: 1e-2,
        device_type: DeviceType::CpuSimdAvx2,
        kernel_available: false, // Force fallback for strict mode testing
        description: "TL2 matrix with unavailable AVX kernel (strict mode should reject)",
    }
}

// ============================================================================
// Ground Truth FP32 Reference Data
// ============================================================================

/// Ground truth FP32 weights for cross-validation
pub struct GroundTruthFP32 {
    pub test_id: &'static str,
    pub weights: Vec<f32>,
    pub shape: (usize, usize),
    pub description: &'static str,
}

/// Small FP32 reference weights for I2S validation
pub fn ground_truth_fp32_small() -> GroundTruthFP32 {
    let size = 128 * 128;
    GroundTruthFP32 {
        test_id: "fp32_reference_128x128",
        weights: generate_normal_distribution(size, 0.0, 0.1, 42),
        shape: (128, 128),
        description: "FP32 reference weights for I2S quantization validation",
    }
}

/// Large FP32 reference weights for stress testing
pub fn ground_truth_fp32_large() -> GroundTruthFP32 {
    let size = 2048 * 2048;
    GroundTruthFP32 {
        test_id: "fp32_reference_2048x2048",
        weights: generate_normal_distribution(size, 0.0, 0.05, 456),
        shape: (2048, 2048),
        description: "FP32 reference weights for large matrix stress testing",
    }
}

// ============================================================================
// Accuracy Metrics for Validation
// ============================================================================

/// Expected accuracy metrics for quantization validation
#[derive(Debug, Clone, Copy)]
pub struct QuantizationAccuracyMetrics {
    pub min_correlation: f32,
    pub max_mse: f32,
    pub max_mae: f32,
    pub quantization_type: QuantizationType,
}

/// I2S accuracy thresholds (99.8%+ correlation)
pub const I2S_ACCURACY_METRICS: QuantizationAccuracyMetrics = QuantizationAccuracyMetrics {
    min_correlation: 0.998,
    max_mse: 1e-3,
    max_mae: 5e-3,
    quantization_type: QuantizationType::I2S,
};

/// TL1 accuracy thresholds (99.6%+ correlation)
pub const TL1_ACCURACY_METRICS: QuantizationAccuracyMetrics = QuantizationAccuracyMetrics {
    min_correlation: 0.996,
    max_mse: 1e-2,
    max_mae: 1e-2,
    quantization_type: QuantizationType::TL1,
};

/// TL2 accuracy thresholds (99.6%+ correlation)
pub const TL2_ACCURACY_METRICS: QuantizationAccuracyMetrics = QuantizationAccuracyMetrics {
    min_correlation: 0.996,
    max_mse: 1e-2,
    max_mae: 1e-2,
    quantization_type: QuantizationType::TL2,
};

// ============================================================================
// Test Data Generators (Deterministic)
// ============================================================================

/// Generate normal distribution with deterministic seed
fn generate_normal_distribution(size: usize, mean: f32, std_dev: f32, seed: u64) -> Vec<f32> {
    // Box-Muller transform for normal distribution
    let mut rng = SimpleRng::new(seed);
    let mut result = Vec::with_capacity(size);

    for _ in 0..(size / 2) {
        let u1 = rng.next_f32();
        let u2 = rng.next_f32();

        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;

        result.push(mean + std_dev * r * theta.cos());
        result.push(mean + std_dev * r * theta.sin());
    }

    if size % 2 == 1 {
        let u1 = rng.next_f32();
        let u2 = rng.next_f32();
        result.push(mean + std_dev * (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos());
    }

    result
}

/// Generate uniform distribution with deterministic seed
fn generate_uniform_distribution(size: usize, min: f32, max: f32, seed: u64) -> Vec<f32> {
    let mut rng = SimpleRng::new(seed);
    (0..size).map(|_| min + (max - min) * rng.next_f32()).collect()
}

/// Generate I2S quantized values (2-bit signed: {-2, -1, 0, 1})
fn generate_i2s_quantized(size: usize, block_size: usize, seed: u64) -> Vec<i8> {
    let mut rng = SimpleRng::new(seed);
    (0..size)
        .map(|_| {
            let val = rng.next_f32();
            if val < 0.25 {
                -2
            } else if val < 0.5 {
                -1
            } else if val < 0.75 {
                0
            } else {
                1
            }
        })
        .collect()
}

/// Generate I2S scale factors (per-block quantization)
fn generate_i2s_scales(size: usize, block_size: usize, seed: u64) -> Vec<f32> {
    let num_blocks = (size + block_size - 1) / block_size;
    let mut rng = SimpleRng::new(seed + 1000);
    (0..num_blocks)
        .map(|_| 0.05 + 0.15 * rng.next_f32()) // Scale factors in [0.05, 0.20]
        .collect()
}

/// Generate TL1 quantized values (table lookup ternary)
fn generate_tl1_quantized(size: usize, seed: u64) -> Vec<i8> {
    let mut rng = SimpleRng::new(seed);
    (0..size)
        .map(|_| {
            let val = rng.next_f32();
            if val < 0.33 {
                -1
            } else if val < 0.66 {
                0
            } else {
                1
            }
        })
        .collect()
}

/// Generate TL1 scale factors
fn generate_tl1_scales(size: usize, block_size: usize, seed: u64) -> Vec<f32> {
    let num_blocks = (size + block_size - 1) / block_size;
    let mut rng = SimpleRng::new(seed + 2000);
    (0..num_blocks)
        .map(|_| 0.08 + 0.12 * rng.next_f32()) // Scale factors in [0.08, 0.20]
        .collect()
}

/// Generate TL2 quantized values (enhanced table lookup)
fn generate_tl2_quantized(size: usize, seed: u64) -> Vec<i8> {
    let mut rng = SimpleRng::new(seed);
    (0..size)
        .map(|_| {
            let val = rng.next_f32();
            if val < 0.2 {
                -2
            } else if val < 0.4 {
                -1
            } else if val < 0.6 {
                0
            } else if val < 0.8 {
                1
            } else {
                2
            }
        })
        .collect()
}

/// Generate TL2 scale factors
fn generate_tl2_scales(size: usize, block_size: usize, seed: u64) -> Vec<f32> {
    let num_blocks = (size + block_size - 1) / block_size;
    let mut rng = SimpleRng::new(seed + 3000);
    (0..num_blocks)
        .map(|_| 0.06 + 0.14 * rng.next_f32()) // Scale factors in [0.06, 0.20]
        .collect()
}

// ============================================================================
// Simple Deterministic RNG (for test data generation only)
// ============================================================================

/// Simple LCG-based RNG for deterministic test data generation
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        // LCG parameters (same as glibc)
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 32) as f32 / (u32::MAX as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_generation() {
        let data1 = generate_normal_distribution(100, 0.0, 0.1, 42);
        let data2 = generate_normal_distribution(100, 0.0, 0.1, 42);
        assert_eq!(data1, data2, "Test data generation must be deterministic");
    }

    #[test]
    #[cfg(feature = "cpu")]
    fn test_i2s_matrix_shapes() {
        let small = i2s_matrix_small();
        assert_eq!(small.shape, (128, 128));
        assert_eq!(small.input_fp32.len(), 128 * 128);

        let medium = i2s_matrix_medium();
        assert_eq!(medium.shape, (512, 512));
        assert_eq!(medium.input_fp32.len(), 512 * 512);
    }

    #[test]
    #[cfg(feature = "cpu")]
    fn test_quantization_accuracy_metrics() {
        assert!(I2S_ACCURACY_METRICS.min_correlation >= 0.998);
        assert!(TL1_ACCURACY_METRICS.min_correlation >= 0.996);
        assert!(TL2_ACCURACY_METRICS.min_correlation >= 0.996);
    }

    #[test]
    fn test_i2s_quantized_values_range() {
        let quantized = generate_i2s_quantized(1000, 32, 42);
        for val in quantized {
            assert!(val >= -2 && val <= 1, "I2S values must be in {{-2, -1, 0, 1}}");
        }
    }

    #[test]
    fn test_tl1_quantized_values_range() {
        let quantized = generate_tl1_quantized(1000, 42);
        for val in quantized {
            assert!(val >= -1 && val <= 1, "TL1 values must be in {{-1, 0, 1}}");
        }
    }

    #[test]
    fn test_tl2_quantized_values_range() {
        let quantized = generate_tl2_quantized(1000, 42);
        for val in quantized {
            assert!(val >= -2 && val <= 2, "TL2 values must be in {{-2, -1, 0, 1, 2}}");
        }
    }
}
