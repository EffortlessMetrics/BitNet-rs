#![allow(unused)]
#![allow(dead_code)]
#![allow(clippy::upper_case_acronyms)]
#![allow(clippy::if_same_then_else)]

//! Quantization Test Data for BitNet-rs Neural Network Components
//!
//! This module provides comprehensive quantization test fixtures supporting
//! I2S, TL1, TL2, and cross-validation with C++ reference implementations.

use serde::{Deserialize, Serialize};
use std::sync::LazyLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationTestVector {
    pub name: &'static str,
    pub input_data: Vec<f32>,
    pub expected_quantized: Vec<i8>,
    pub expected_scales: Vec<f32>,
    pub block_size: usize,
    pub quantization_type: QuantizationType,
    pub device_type: DeviceType,
    pub tolerance: f32,
    pub accuracy_target: f32,
    pub test_description: &'static str,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossValidationFixture {
    pub name: &'static str,
    pub input_tokens: Vec<u32>,
    pub rust_output: Vec<f32>,
    pub cpp_reference: Vec<f32>,
    pub tolerance: f32,
    pub quantization_type: QuantizationType,
    pub model_config: ModelConfig,
    pub validation_notes: &'static str,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum QuantizationType {
    I2S,  // 2-bit signed quantization
    TL1,  // Table lookup 1
    TL2,  // Table lookup 2
    IQ2S, // GGML-compatible quantization
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(clippy::upper_case_acronyms)]
pub enum DeviceType {
    CPU,
    GPU,
    Mixed, // For cross-device validation
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub embedding_dim: u32,
    pub num_layers: u32,
    pub num_heads: u32,
    pub vocab_size: u32,
    pub context_length: u32,
}

/// I2S Quantization Test Vectors
/// Production-grade 2-bit signed quantization with 99%+ accuracy targets
pub static I2S_TEST_VECTORS: LazyLock<Vec<QuantizationTestVector>> = LazyLock::new(|| {
    vec![
        QuantizationTestVector {
            name: "i2s_basic_positive",
            input_data: vec![0.5, 1.0, 1.5, 2.0, 0.25, 0.75],
            expected_quantized: vec![1, 1, 1, 1, 0, 1],
            expected_scales: vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            block_size: 32,
            quantization_type: QuantizationType::I2S,
            device_type: DeviceType::CPU,
            tolerance: 0.1,
            accuracy_target: 0.99,
            test_description: "Basic I2S quantization with positive values",
        },
        QuantizationTestVector {
            name: "i2s_mixed_signs",
            input_data: vec![-1.5, -0.5, 0.0, 0.5, 1.5, -2.0],
            expected_quantized: vec![-1, -1, 0, 1, 1, -1],
            expected_scales: vec![1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
            block_size: 32,
            quantization_type: QuantizationType::I2S,
            device_type: DeviceType::CPU,
            tolerance: 0.1,
            accuracy_target: 0.99,
            test_description: "I2S quantization with mixed positive/negative values",
        },
        QuantizationTestVector {
            name: "i2s_extreme_values",
            input_data: vec![-10.0, -5.0, 0.0, 5.0, 10.0, 15.0],
            expected_quantized: vec![-1, -1, 0, 1, 1, 1],
            expected_scales: vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            block_size: 32,
            quantization_type: QuantizationType::I2S,
            device_type: DeviceType::CPU,
            tolerance: 0.2,
            accuracy_target: 0.95,
            test_description: "I2S quantization with extreme value ranges",
        },
        QuantizationTestVector {
            name: "i2s_gpu_large_tensor",
            input_data: (0..1024).map(|i| (i as f32 / 512.0) - 1.0).collect(),
            expected_quantized: (0..1024)
                .map(|i| {
                    if i < 256 {
                        -1
                    } else if i < 512 {
                        0
                    } else {
                        1
                    }
                })
                .collect(),
            expected_scales: vec![1.0; 1024],
            block_size: 64,
            quantization_type: QuantizationType::I2S,
            device_type: DeviceType::GPU,
            tolerance: 0.1,
            accuracy_target: 0.99,
            test_description: "Large tensor I2S quantization for GPU testing",
        },
    ]
});

/// TL1 Quantization Test Vectors
/// Table lookup quantization optimized for CPU SIMD operations
pub static TL1_TEST_VECTORS: LazyLock<Vec<QuantizationTestVector>> = LazyLock::new(|| {
    vec![
        QuantizationTestVector {
            name: "tl1_lookup_basic",
            input_data: vec![0.1, 0.3, 0.7, 0.9, 1.1, 1.3],
            expected_quantized: vec![0, 0, 1, 1, 1, 1],
            expected_scales: vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            block_size: 16,
            quantization_type: QuantizationType::TL1,
            device_type: DeviceType::CPU,
            tolerance: 0.15,
            accuracy_target: 0.98,
            test_description: "Basic TL1 table lookup quantization",
        },
        QuantizationTestVector {
            name: "tl1_simd_aligned",
            input_data: (0..256).map(|i| (i as f32 / 128.0) - 1.0).collect(),
            expected_quantized: (0..256)
                .map(|i| {
                    if i < 64 {
                        -1
                    } else if i < 128 {
                        0
                    } else {
                        1
                    }
                })
                .collect(),
            expected_scales: vec![1.0; 256],
            block_size: 32,
            quantization_type: QuantizationType::TL1,
            device_type: DeviceType::CPU,
            tolerance: 0.12,
            accuracy_target: 0.98,
            test_description: "SIMD-aligned TL1 quantization for CPU optimization",
        },
    ]
});

/// TL2 Quantization Test Vectors
/// Advanced table lookup quantization optimized for GPU tensor cores
pub static TL2_TEST_VECTORS: LazyLock<Vec<QuantizationTestVector>> = LazyLock::new(|| {
    vec![
        QuantizationTestVector {
            name: "tl2_tensor_core",
            input_data: (0..512).map(|i| ((i as f32 / 256.0) - 1.0) * 2.0).collect(),
            expected_quantized: (0..512)
                .map(|i| {
                    if i < 128 {
                        -1
                    } else if i < 256 {
                        0
                    } else {
                        1
                    }
                })
                .collect(),
            expected_scales: vec![2.0; 512],
            block_size: 64,
            quantization_type: QuantizationType::TL2,
            device_type: DeviceType::GPU,
            tolerance: 0.1,
            accuracy_target: 0.98,
            test_description: "TL2 quantization optimized for GPU tensor cores",
        },
        QuantizationTestVector {
            name: "tl2_mixed_precision",
            input_data: vec![0.125, 0.25, 0.5, 1.0, 2.0, 4.0],
            expected_quantized: vec![0, 0, 1, 1, 1, 1],
            expected_scales: vec![4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
            block_size: 16,
            quantization_type: QuantizationType::TL2,
            device_type: DeviceType::GPU,
            tolerance: 0.1,
            accuracy_target: 0.98,
            test_description: "TL2 quantization with mixed precision support",
        },
    ]
});

/// Cross-Validation Test Fixtures
/// Reference data for validating against C++ BitNet implementation
#[cfg(feature = "crossval")]
pub static CROSSVAL_FIXTURES: LazyLock<Vec<CrossValidationFixture>> = LazyLock::new(|| {
    vec![
        CrossValidationFixture {
            name: "crossval_i2s_basic",
            input_tokens: vec![1, 2, 3, 4, 5],
            rust_output: vec![0.1, 0.2, 0.3, 0.4, 0.5],
            cpp_reference: vec![0.101, 0.199, 0.301, 0.398, 0.502],
            tolerance: 0.01,
            quantization_type: QuantizationType::I2S,
            model_config: ModelConfig {
                embedding_dim: 768,
                num_layers: 12,
                num_heads: 12,
                vocab_size: 32000,
                context_length: 2048,
            },
            validation_notes: "Basic I2S quantization parity with C++ reference",
        },
        CrossValidationFixture {
            name: "crossval_tl1_inference",
            input_tokens: vec![10, 20, 30, 40, 50, 60, 70, 80],
            rust_output: vec![0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85],
            cpp_reference: vec![0.151, 0.249, 0.352, 0.448, 0.551, 0.648, 0.752, 0.849],
            tolerance: 0.005,
            quantization_type: QuantizationType::TL1,
            model_config: ModelConfig {
                embedding_dim: 1024,
                num_layers: 24,
                num_heads: 16,
                vocab_size: 32000,
                context_length: 4096,
            },
            validation_notes: "TL1 quantization inference parity validation",
        },
    ]
});

/// Edge Case Test Data
/// Boundary conditions and error scenarios for robust testing
pub static EDGE_CASE_VECTORS: LazyLock<Vec<QuantizationTestVector>> = LazyLock::new(|| {
    vec![
        QuantizationTestVector {
            name: "edge_all_zeros",
            input_data: vec![0.0; 64],
            expected_quantized: vec![0; 64],
            expected_scales: vec![1.0; 64],
            block_size: 32,
            quantization_type: QuantizationType::I2S,
            device_type: DeviceType::CPU,
            tolerance: 0.0,
            accuracy_target: 1.0,
            test_description: "Edge case: all zero input values",
        },
        QuantizationTestVector {
            name: "edge_tiny_values",
            input_data: vec![1e-6, -1e-6, 1e-7, -1e-7],
            expected_quantized: vec![0, 0, 0, 0],
            expected_scales: vec![1e-6, 1e-6, 1e-6, 1e-6],
            block_size: 32,
            quantization_type: QuantizationType::I2S,
            device_type: DeviceType::CPU,
            tolerance: 1e-6,
            accuracy_target: 0.95,
            test_description: "Edge case: extremely small input values",
        },
        QuantizationTestVector {
            name: "edge_huge_values",
            input_data: vec![1e6, -1e6, 1e7, -1e7],
            expected_quantized: vec![1, -1, 1, -1],
            expected_scales: vec![1e6, 1e6, 1e6, 1e6],
            block_size: 32,
            quantization_type: QuantizationType::I2S,
            device_type: DeviceType::CPU,
            tolerance: 1e5,
            accuracy_target: 0.90,
            test_description: "Edge case: extremely large input values",
        },
    ]
});

/// Get all quantization test vectors for a specific type
pub fn get_quantization_vectors(
    quant_type: QuantizationType,
) -> Vec<&'static QuantizationTestVector> {
    match quant_type {
        QuantizationType::I2S => I2S_TEST_VECTORS.iter().collect(),
        QuantizationType::TL1 => TL1_TEST_VECTORS.iter().collect(),
        QuantizationType::TL2 => TL2_TEST_VECTORS.iter().collect(),
        _ => vec![],
    }
}

/// Get test vectors for specific device type
#[cfg(feature = "cpu")]
pub fn get_cpu_test_vectors() -> Vec<&'static QuantizationTestVector> {
    let mut vectors = Vec::new();
    vectors.extend(I2S_TEST_VECTORS.iter().filter(|v| v.device_type == DeviceType::CPU));
    vectors.extend(TL1_TEST_VECTORS.iter().filter(|v| v.device_type == DeviceType::CPU));
    vectors.extend(EDGE_CASE_VECTORS.iter().filter(|v| v.device_type == DeviceType::CPU));
    vectors
}

/// Get test vectors for GPU testing
#[cfg(feature = "gpu")]
pub fn get_gpu_test_vectors() -> Vec<&'static QuantizationTestVector> {
    let mut vectors = Vec::new();
    vectors.extend(I2S_TEST_VECTORS.iter().filter(|v| v.device_type == DeviceType::GPU));
    vectors.extend(TL2_TEST_VECTORS.iter().filter(|v| v.device_type == DeviceType::GPU));
    vectors
}

/// Get cross-validation fixtures
#[cfg(feature = "crossval")]
pub fn get_crossval_fixtures() -> Vec<&'static CrossValidationFixture> {
    CROSSVAL_FIXTURES.iter().collect()
}

/// Get edge case test vectors
pub fn get_edge_case_vectors() -> Vec<&'static QuantizationTestVector> {
    EDGE_CASE_VECTORS.iter().collect()
}

/// Generate deterministic test data with seed
pub fn generate_deterministic_test_data(size: usize, seed: u64, range: (f32, f32)) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut data = Vec::with_capacity(size);
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);

    for i in 0..size {
        i.hash(&mut hasher);
        let hash_val = hasher.finish();
        let normalized = (hash_val as f64 / u64::MAX as f64) as f32;
        let value = range.0 + normalized * (range.1 - range.0);
        data.push(value);

        // Update hasher for next iteration
        hasher = DefaultHasher::new();
        (seed + i as u64).hash(&mut hasher);
    }

    data
}

/// Validate quantization accuracy
pub fn validate_quantization_accuracy(
    original: &[f32],
    quantized: &[i8],
    scales: &[f32],
    tolerance: f32,
) -> f32 {
    if original.len() != quantized.len() || original.len() != scales.len() {
        return 0.0;
    }

    let mut accurate_count = 0;
    for i in 0..original.len() {
        let reconstructed = quantized[i] as f32 * scales[i];
        let error = (original[i] - reconstructed).abs();
        if error <= tolerance {
            accurate_count += 1;
        }
    }

    accurate_count as f32 / original.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_i2s_quantization_vectors() {
        let vectors = get_quantization_vectors(QuantizationType::I2S);
        assert!(!vectors.is_empty());

        for vector in vectors {
            assert_eq!(vector.input_data.len(), vector.expected_quantized.len());
            assert_eq!(vector.input_data.len(), vector.expected_scales.len());
            assert!(vector.tolerance > 0.0);
            assert!(vector.accuracy_target > 0.0 && vector.accuracy_target <= 1.0);
        }
    }

    #[test]
    fn test_deterministic_data_generation() {
        let data1 = generate_deterministic_test_data(100, 42, (-1.0, 1.0));
        let data2 = generate_deterministic_test_data(100, 42, (-1.0, 1.0));

        assert_eq!(data1.len(), 100);
        assert_eq!(data1, data2); // Should be identical with same seed

        for value in &data1 {
            assert!(*value >= -1.0 && *value <= 1.0);
        }
    }

    #[test]
    fn test_quantization_accuracy_validation() {
        let original = vec![1.0, 2.0, 3.0, 4.0];
        let quantized = vec![1, 1, 1, 1];
        let scales = vec![1.0, 2.0, 3.0, 4.0];

        let accuracy = validate_quantization_accuracy(&original, &quantized, &scales, 0.1);
        assert_eq!(accuracy, 1.0); // Perfect reconstruction
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn test_cpu_vector_filtering() {
        let cpu_vectors = get_cpu_test_vectors();
        assert!(!cpu_vectors.is_empty());

        for vector in cpu_vectors {
            assert_eq!(vector.device_type, DeviceType::CPU);
        }
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_vector_filtering() {
        let gpu_vectors = get_gpu_test_vectors();
        assert!(!gpu_vectors.is_empty());

        for vector in gpu_vectors {
            assert_eq!(vector.device_type, DeviceType::GPU);
        }
    }

    #[test]
    fn test_edge_case_vectors() {
        let edge_vectors = get_edge_case_vectors();
        assert!(!edge_vectors.is_empty());

        // Verify edge case naming convention
        for vector in edge_vectors {
            assert!(vector.name.starts_with("edge_"));
        }
    }

    #[cfg(feature = "crossval")]
    #[test]
    fn test_crossval_fixtures() {
        let fixtures = get_crossval_fixtures();
        assert!(!fixtures.is_empty());

        for fixture in fixtures {
            assert_eq!(fixture.rust_output.len(), fixture.cpp_reference.len());
            assert!(fixture.tolerance > 0.0);
            assert!(!fixture.validation_notes.is_empty());
        }
    }
}
