//! Quantization test vectors for BitNet.rs neural network components
//!
//! Provides comprehensive test data for I2S, TL1, TL2, and IQ2_S quantization algorithms
//! with device-aware validation data for CPU/GPU parity testing.

use bitnet_common::{BitNetError, QuantizationType, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::LazyLock;

/// Quantization test vector for neural network validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationTestVector {
    pub quantization_type: QuantizationType,
    pub input_data: Vec<f32>,
    pub expected_quantized: Vec<i8>,
    pub expected_scales: Vec<f32>,
    pub expected_dequantized: Vec<f32>,
    pub block_size: usize,
    pub tolerance: f32,
    pub device_compatible: Vec<String>,
    pub vocab_range: Option<(u32, u32)>,
    pub test_scenario: String,
}

/// Mixed precision test data for GPU acceleration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixedPrecisionTestData {
    pub precision_mode: String,
    pub input_fp32: Vec<f32>,
    pub expected_fp16: Vec<u16>, // half::f16 as u16 for serialization
    pub expected_bf16: Vec<u16>, // half::bf16 as u16 for serialization
    pub quantization_compatible: Vec<QuantizationType>,
    pub tensor_core_eligible: bool,
    pub compute_capability: String,
}

/// Device-aware validation data for CPU/GPU parity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceValidationData {
    pub test_name: String,
    pub cpu_reference: Vec<f32>,
    pub gpu_expected: Vec<f32>,
    pub tolerance_absolute: f32,
    pub tolerance_relative: f32,
    pub quantization_type: QuantizationType,
    pub memory_alignment: usize,
    pub simd_compatible: bool,
    pub cuda_compatible: bool,
}

/// Cross-validation reference data for C++ implementation comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossValidationReference {
    pub cpp_implementation: String,
    pub rust_implementation: String,
    pub input_tokens: Vec<u32>,
    pub expected_cpp_output: Vec<f32>,
    pub expected_rust_output: Vec<f32>,
    pub tolerance: f32,
    pub quantization_params: HashMap<String, f32>,
    pub model_architecture: String,
}

// Static test vectors for different quantization algorithms

/// I2S quantization test vectors (optimal for large vocabularies 128k+)
static I2S_TEST_VECTORS: LazyLock<Vec<QuantizationTestVector>> = LazyLock::new(|| {
    vec![
        QuantizationTestVector {
            quantization_type: QuantizationType::I2S,
            input_data: vec![
                -2.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, -1.8, 0.7, 1.2, -0.3, 2.1,
                -2.0, 0.8, -1.5, 1.7, -0.9,
            ],
            expected_quantized: vec![
                -1, -1, 0, 0, 0, 1, 1, 1, 1, 1, -1, 0, 1, 0, 1, -1, 0, -1, 1, 0,
            ],
            expected_scales: vec![2.5, 2.1], // Per block scales
            expected_dequantized: vec![
                -2.5, -2.5, 0.0, 0.0, 0.0, 2.5, 2.5, 2.5, 2.5, 2.5, -2.1, 0.0, 2.1, 0.0, 2.1, -2.1,
                0.0, -2.1, 2.1, 0.0,
            ],
            block_size: 10,
            tolerance: 0.5,
            device_compatible: vec!["CPU".to_string(), "GPU".to_string()],
            vocab_range: Some((0, 128256)),
            test_scenario: "large_vocabulary_optimal".to_string(),
        },
        QuantizationTestVector {
            quantization_type: QuantizationType::I2S,
            input_data: vec![4.0, -4.0, 3.5, -3.5, 2.8, -2.8, 1.9, -1.9],
            expected_quantized: vec![1, -1, 1, -1, 1, -1, 0, -1],
            expected_scales: vec![4.0],
            expected_dequantized: vec![4.0, -4.0, 4.0, -4.0, 4.0, -4.0, 0.0, -4.0],
            block_size: 8,
            tolerance: 0.8,
            device_compatible: vec!["CPU".to_string(), "GPU".to_string()],
            vocab_range: Some((0, 128256)),
            test_scenario: "extreme_values".to_string(),
        },
        QuantizationTestVector {
            quantization_type: QuantizationType::I2S,
            input_data: vec![0.1, -0.1, 0.05, -0.05, 0.2, -0.15, 0.08, -0.12],
            expected_quantized: vec![0, 0, 0, 0, 1, 0, 0, 0],
            expected_scales: vec![0.2],
            expected_dequantized: vec![0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0],
            block_size: 8,
            tolerance: 0.15,
            device_compatible: vec!["CPU".to_string(), "GPU".to_string()],
            vocab_range: Some((0, 128256)),
            test_scenario: "small_values_precision".to_string(),
        },
    ]
});

/// TL1 quantization test vectors (efficient for smaller vocabularies 32k)
static TL1_TEST_VECTORS: LazyLock<Vec<QuantizationTestVector>> = LazyLock::new(|| {
    vec![
        QuantizationTestVector {
            quantization_type: QuantizationType::TL1,
            input_data: vec![
                -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5,
                4.5,
            ],
            expected_quantized: vec![0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 4, 5, 6, 7, 7],
            expected_scales: vec![1.0, 1.0], // Lookup table scales
            expected_dequantized: vec![
                -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0,
                4.0,
            ],
            block_size: 8,
            tolerance: 0.3,
            device_compatible: vec!["CPU".to_string()],
            vocab_range: Some((0, 32000)),
            test_scenario: "medium_vocabulary_efficient".to_string(),
        },
        QuantizationTestVector {
            quantization_type: QuantizationType::TL1,
            input_data: vec![1.2, -1.7, 0.8, -0.3, 2.1, -2.9, 1.6, -1.1],
            expected_quantized: vec![4, 1, 4, 3, 5, 0, 5, 2],
            expected_scales: vec![1.0],
            expected_dequantized: vec![1.0, -2.0, 1.0, 0.0, 2.0, -3.0, 2.0, -1.0],
            block_size: 8,
            tolerance: 0.5,
            device_compatible: vec!["CPU".to_string()],
            vocab_range: Some((0, 32000)),
            test_scenario: "lookup_table_rounding".to_string(),
        },
    ]
});

/// TL2 quantization test vectors (enhanced table lookup)
static TL2_TEST_VECTORS: LazyLock<Vec<QuantizationTestVector>> = LazyLock::new(|| {
    vec![QuantizationTestVector {
        quantization_type: QuantizationType::TL2,
        input_data: vec![
            -4.5, -3.2, -1.8, -0.5, 0.3, 1.2, 2.7, 3.9, -4.1, -2.9, -1.3, -0.1, 0.7, 1.8, 3.1, 4.2,
        ],
        expected_quantized: vec![0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7],
        expected_scales: vec![0.64, 0.6], // Enhanced precision scales
        expected_dequantized: vec![
            -4.48, -3.2, -1.92, -0.64, 0.0, 0.64, 1.92, 3.2, -4.2, -3.0, -1.8, -0.6, 0.0, 0.6, 1.8,
            3.0,
        ],
        block_size: 8,
        tolerance: 0.25,
        device_compatible: vec!["CPU".to_string(), "GPU".to_string()],
        vocab_range: Some((0, 50257)),
        test_scenario: "enhanced_precision_lookup".to_string(),
    }]
});

/// IQ2_S quantization test vectors (GGML-compatible with 82-byte blocks)
static IQ2S_TEST_VECTORS: LazyLock<Vec<QuantizationTestVector>> = LazyLock::new(|| {
    vec![QuantizationTestVector {
        quantization_type: QuantizationType::I2S,
        input_data: vec![
            // 82-byte block worth of data for GGML compatibility
            -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5,
            -1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2, 1.6, -1.8, -1.2, -0.6, 0.0, 0.6, 1.2, 1.8, 2.4,
        ],
        expected_quantized: vec![
            0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4,
            5, 6, 7,
        ],
        expected_scales: vec![0.5, 0.5, 0.4, 0.6],
        expected_dequantized: vec![
            -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5,
            -1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2, 1.6, -1.8, -1.2, -0.6, 0.0, 0.6, 1.2, 1.8, 2.4,
        ],
        block_size: 32, // GGML 82-byte alignment
        tolerance: 0.1,
        device_compatible: vec!["CPU".to_string(), "GPU".to_string()],
        vocab_range: Some((0, 128256)),
        test_scenario: "ggml_compatibility_82byte".to_string(),
    }]
});

/// Mixed precision test data for GPU acceleration
static MIXED_PRECISION_DATA: LazyLock<Vec<MixedPrecisionTestData>> = LazyLock::new(|| {
    vec![
        MixedPrecisionTestData {
            precision_mode: "FP16".to_string(),
            input_fp32: vec![1.5, -2.25, 0.75, -0.125, 3.0, -1.875],
            expected_fp16: vec![0x3E00, 0xC100, 0x3A00, 0xB000, 0x4200, 0xBF00], // FP16 bit patterns
            expected_bf16: vec![0x3FC0, 0xC010, 0x3F40, 0xBE00, 0x4040, 0xBFF0], // BF16 bit patterns
            quantization_compatible: vec![QuantizationType::I2S],
            tensor_core_eligible: true,
            compute_capability: "7.5".to_string(),
        },
        MixedPrecisionTestData {
            precision_mode: "BF16".to_string(),
            input_fp32: vec![0.5, -1.0, 2.5, -0.25, 1.75, -3.5],
            expected_fp16: vec![0x3800, 0xBC00, 0x4100, 0xB400, 0x3E00, 0xC600],
            expected_bf16: vec![0x3F00, 0xBF80, 0x4020, 0xBE80, 0x3FE0, 0xC060],
            quantization_compatible: vec![QuantizationType::I2S, QuantizationType::TL2],
            tensor_core_eligible: true,
            compute_capability: "8.0".to_string(),
        },
    ]
});

/// Device validation data for CPU/GPU parity testing
static DEVICE_VALIDATION_DATA: LazyLock<Vec<DeviceValidationData>> = LazyLock::new(|| {
    vec![
        DeviceValidationData {
            test_name: "i2s_cpu_gpu_parity".to_string(),
            cpu_reference: vec![1.0, -1.0, 0.5, -0.5, 2.0, -2.0],
            gpu_expected: vec![1.0, -1.0, 0.5, -0.5, 2.0, -2.0],
            tolerance_absolute: 1e-6,
            tolerance_relative: 1e-5,
            quantization_type: QuantizationType::I2S,
            memory_alignment: 32,
            simd_compatible: true,
            cuda_compatible: true,
        },
        DeviceValidationData {
            test_name: "tl1_cpu_specialized".to_string(),
            cpu_reference: vec![0.8, -1.2, 1.5, -0.7, 2.1, -1.9],
            gpu_expected: vec![1.0, -1.0, 2.0, -1.0, 2.0, -2.0], // Quantized approximation
            tolerance_absolute: 0.5,
            tolerance_relative: 0.2,
            quantization_type: QuantizationType::TL1,
            memory_alignment: 16,
            simd_compatible: true,
            cuda_compatible: false,
        },
    ]
});

/// Cross-validation reference data for C++ implementation comparison
static CROSS_VALIDATION_REFERENCE: LazyLock<Vec<CrossValidationReference>> = LazyLock::new(|| {
    vec![
        CrossValidationReference {
            cpp_implementation: "llama.cpp".to_string(),
            rust_implementation: "bitnet-rs".to_string(),
            input_tokens: vec![9906, 1917, 8989, 4632], // "Hello world Neural network"
            expected_cpp_output: vec![0.95, -0.32, 1.2, -0.67, 0.88, -1.1, 0.55, -0.23],
            expected_rust_output: vec![0.94, -0.33, 1.19, -0.66, 0.89, -1.09, 0.56, -0.24],
            tolerance: 0.05,
            quantization_params: {
                let mut params = HashMap::new();
                params.insert("scale".to_string(), 2.5);
                params.insert("block_size".to_string(), 8.0);
                params
            },
            model_architecture: "BitNet-b1.58".to_string(),
        },
        CrossValidationReference {
            cpp_implementation: "ggml".to_string(),
            rust_implementation: "bitnet-rs".to_string(),
            input_tokens: vec![15043, 3186], // "Hello world" in LLaMA-2
            expected_cpp_output: vec![1.12, -0.89, 0.67, -0.45],
            expected_rust_output: vec![1.11, -0.88, 0.68, -0.44],
            tolerance: 0.02,
            quantization_params: {
                let mut params = HashMap::new();
                params.insert("lookup_table_size".to_string(), 256.0);
                params.insert("precision_bits".to_string(), 4.0);
                params
            },
            model_architecture: "BitNet-TL1".to_string(),
        },
    ]
});

/// Quantization fixtures manager
pub struct QuantizationFixtures {
    pub test_vectors: HashMap<QuantizationType, Vec<QuantizationTestVector>>,
    pub mixed_precision_data: Vec<MixedPrecisionTestData>,
    pub device_validation_data: Vec<DeviceValidationData>,
    pub cross_validation_data: Vec<CrossValidationReference>,
}

impl QuantizationFixtures {
    /// Initialize all quantization test fixtures
    pub fn new() -> Self {
        let mut test_vectors = HashMap::new();
        test_vectors.insert(QuantizationType::I2S, I2S_TEST_VECTORS.clone());
        test_vectors.insert(QuantizationType::TL1, TL1_TEST_VECTORS.clone());
        test_vectors.insert(QuantizationType::TL2, TL2_TEST_VECTORS.clone());
        test_vectors.insert(QuantizationType::I2S, IQ2S_TEST_VECTORS.clone());

        Self {
            test_vectors,
            mixed_precision_data: MIXED_PRECISION_DATA.clone(),
            device_validation_data: DEVICE_VALIDATION_DATA.clone(),
            cross_validation_data: CROSS_VALIDATION_REFERENCE.clone(),
        }
    }

    /// Get test vectors for specific quantization type
    pub fn get_test_vectors(
        &self,
        quant_type: &QuantizationType,
    ) -> Option<&Vec<QuantizationTestVector>> {
        self.test_vectors.get(quant_type)
    }

    /// Get test vectors compatible with specific vocabulary size
    pub fn get_vocab_compatible_vectors(&self, vocab_size: u32) -> Vec<&QuantizationTestVector> {
        let mut compatible = Vec::new();
        for vectors in self.test_vectors.values() {
            for vector in vectors {
                if let Some((min, max)) = vector.vocab_range {
                    if vocab_size >= min && vocab_size <= max {
                        compatible.push(vector);
                    }
                }
            }
        }
        compatible
    }

    /// Get device-compatible test vectors
    pub fn get_device_compatible_vectors(&self, device: &str) -> Vec<&QuantizationTestVector> {
        let mut compatible = Vec::new();
        for vectors in self.test_vectors.values() {
            for vector in vectors {
                if vector.device_compatible.contains(&device.to_string()) {
                    compatible.push(vector);
                }
            }
        }
        compatible
    }

    /// Get mixed precision test data for specific precision mode
    pub fn get_mixed_precision_data(&self, precision: &str) -> Vec<&MixedPrecisionTestData> {
        self.mixed_precision_data.iter().filter(|data| data.precision_mode == precision).collect()
    }

    /// Get cross-validation data for specific C++ implementation
    pub fn get_cross_validation_data(&self, cpp_impl: &str) -> Vec<&CrossValidationReference> {
        self.cross_validation_data
            .iter()
            .filter(|data| data.cpp_implementation == cpp_impl)
            .collect()
    }

    /// Generate deterministic test data with seed
    pub fn generate_deterministic_vectors(
        &self,
        seed: u64,
        count: usize,
    ) -> Vec<QuantizationTestVector> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut vectors = Vec::new();
        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);

        for i in 0..count {
            (seed + i as u64).hash(&mut hasher);
            let hash = hasher.finish();

            // Generate deterministic values based on hash
            let input_size = 8 + (hash % 24) as usize; // 8-32 elements
            let mut input_data = Vec::new();

            for j in 0..input_size {
                let val = ((hash + j as u64) as f32 / u64::MAX as f32) * 8.0 - 4.0; // -4.0 to 4.0
                input_data.push(val);
            }

            vectors.push(QuantizationTestVector {
                quantization_type: QuantizationType::I2S,
                input_data: input_data.clone(),
                expected_quantized: input_data
                    .iter()
                    .map(|&x| {
                        if x > 1.0 {
                            1
                        } else if x < -1.0 {
                            -1
                        } else {
                            0
                        }
                    })
                    .collect(),
                expected_scales: vec![input_data.iter().map(|x| x.abs()).fold(0.0f32, f32::max)],
                expected_dequantized: input_data
                    .iter()
                    .map(|&x| {
                        let scale = input_data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                        if x > scale / 2.0 {
                            scale
                        } else if x < -scale / 2.0 {
                            -scale
                        } else {
                            0.0
                        }
                    })
                    .collect(),
                block_size: input_size,
                tolerance: 0.5,
                device_compatible: vec!["CPU".to_string(), "GPU".to_string()],
                vocab_range: Some((0, 128256)),
                test_scenario: format!("deterministic_seed_{}_iter_{}", seed, i),
            });
        }

        vectors
    }

    /// Write all quantization fixtures to binary files for performance testing
    pub async fn write_binary_fixtures(&self, fixtures_dir: &std::path::Path) -> Result<()> {
        use std::io::Write;
        use tokio::fs;

        let quant_dir = fixtures_dir.join("quantization");
        fs::create_dir_all(&quant_dir).await.map_err(BitNetError::Io)?;

        // Write I2S test vectors
        if let Some(vectors) = self.test_vectors.get(&QuantizationType::I2S) {
            let mut buffer = Vec::new();
            for vector in vectors {
                // Write header: quantization type, input size, block size
                buffer.extend_from_slice(&(0u32).to_le_bytes()); // I2S = 0
                buffer.extend_from_slice(&(vector.input_data.len() as u32).to_le_bytes());
                buffer.extend_from_slice(&(vector.block_size as u32).to_le_bytes());
                buffer.extend_from_slice(&vector.tolerance.to_le_bytes());

                // Write input data
                for &val in &vector.input_data {
                    buffer.extend_from_slice(&val.to_le_bytes());
                }

                // Write expected quantized data
                for &val in &vector.expected_quantized {
                    buffer.push(val as u8);
                }

                // Write scales
                buffer.extend_from_slice(&(vector.expected_scales.len() as u32).to_le_bytes());
                for &scale in &vector.expected_scales {
                    buffer.extend_from_slice(&scale.to_le_bytes());
                }
            }
            fs::write(quant_dir.join("i2s_test_vectors.bin"), buffer)
                .await
                .map_err(BitNetError::Io)?;
        }

        // Write mixed precision data
        let mut fp16_buffer = Vec::new();
        for data in &self.mixed_precision_data {
            if data.precision_mode == "FP16" {
                for &val in &data.input_fp32 {
                    fp16_buffer.extend_from_slice(&val.to_le_bytes());
                }
            }
        }
        fs::write(quant_dir.join("mixed_precision_fp16.bin"), fp16_buffer)
            .await
            .map_err(BitNetError::Io)?;

        // Write cross-validation reference data
        let mut crossval_buffer = Vec::new();
        for data in &self.cross_validation_data {
            // Write input tokens
            crossval_buffer.extend_from_slice(&(data.input_tokens.len() as u32).to_le_bytes());
            for &token in &data.input_tokens {
                crossval_buffer.extend_from_slice(&token.to_le_bytes());
            }

            // Write expected outputs
            crossval_buffer
                .extend_from_slice(&(data.expected_rust_output.len() as u32).to_le_bytes());
            for &val in &data.expected_rust_output {
                crossval_buffer.extend_from_slice(&val.to_le_bytes());
            }

            crossval_buffer.extend_from_slice(&data.tolerance.to_le_bytes());
        }
        fs::write(quant_dir.join("cross_validation_reference.bin"), crossval_buffer)
            .await
            .map_err(BitNetError::Io)?;

        Ok(())
    }
}

/// CPU-specific quantization test utilities
#[cfg(feature = "cpu")]
pub mod cpu_quantization {
    use super::*;

    pub fn get_simd_compatible_vectors() -> Vec<&'static QuantizationTestVector> {
        I2S_TEST_VECTORS
            .iter()
            .chain(TL1_TEST_VECTORS.iter())
            .chain(TL2_TEST_VECTORS.iter())
            .filter(|v| v.device_compatible.contains(&"CPU".to_string()))
            .collect()
    }

    pub fn get_avx2_test_data() -> Vec<([f32; 8], [i8; 8], f32)> {
        vec![
            ([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, -1.5, 0.5], [-1, -1, 0, 1, 1, 1, -1, 0], 2.0),
            ([1.2, -1.8, 2.5, -0.3, 0.7, -2.1, 1.9, -0.6], [1, -1, 1, 0, 0, -1, 1, 0], 2.5),
        ]
    }
}

/// GPU-specific quantization test utilities
#[cfg(feature = "gpu")]
pub mod gpu_quantization {
    use super::*;

    pub fn get_cuda_compatible_vectors() -> Vec<&'static QuantizationTestVector> {
        I2S_TEST_VECTORS
            .iter()
            .chain(TL2_TEST_VECTORS.iter())
            .chain(IQ2S_TEST_VECTORS.iter())
            .filter(|v| v.device_compatible.contains(&"GPU".to_string()))
            .collect()
    }

    pub fn get_tensor_core_data() -> Vec<&'static MixedPrecisionTestData> {
        MIXED_PRECISION_DATA.iter().filter(|data| data.tensor_core_eligible).collect()
    }

    pub fn get_memory_coalescing_patterns() -> Vec<(Vec<f32>, usize)> {
        vec![
            (vec![1.0; 32], 32),   // 32-byte aligned
            (vec![0.5; 64], 32),   // 64 elements, 32-byte alignment
            (vec![-1.5; 128], 32), // Large vector with alignment
        ]
    }
}

/// FFI bridge test data for C++ comparison
#[cfg(feature = "ffi")]
pub mod ffi_quantization {
    use super::*;

    pub fn get_ffi_test_vectors() -> Vec<&'static CrossValidationReference> {
        CROSS_VALIDATION_REFERENCE.iter().collect()
    }

    pub fn create_cpp_compatible_test_data() -> Vec<(Vec<f32>, Vec<i8>, f32)> {
        vec![
            (vec![2.0, -2.0, 1.0, -1.0], vec![1, -1, 1, -1], 2.0),
            (vec![0.5, -1.5, 2.5, -0.75], vec![0, -1, 1, 0], 2.5),
        ]
    }
}

/// Load quantization fixtures with proper error handling
#[cfg(test)]
pub fn load_quantization_fixtures() -> QuantizationFixtures {
    QuantizationFixtures::new()
}

/// Generate test data for specific quantization algorithm and vocab size
#[cfg(test)]
pub fn generate_vocab_specific_test_data(
    quant_type: QuantizationType,
    vocab_size: u32,
) -> Vec<QuantizationTestVector> {
    let fixtures = QuantizationFixtures::new();
    fixtures
        .get_vocab_compatible_vectors(vocab_size)
        .into_iter()
        .filter(|v| v.quantization_type == quant_type)
        .cloned()
        .collect()
}

/// Validate quantization accuracy against tolerance
#[cfg(test)]
pub fn validate_quantization_accuracy(
    input: &[f32],
    quantized: &[i8],
    dequantized: &[f32],
    tolerance: f32,
) -> bool {
    if input.len() != dequantized.len() {
        return false;
    }

    for (orig, deq) in input.iter().zip(dequantized.iter()) {
        let error = (orig - deq).abs();
        if error > tolerance {
            return false;
        }
    }

    true
}
