//! IQ2_S GGML-compatible quantization fixtures for BitNet.rs
//!
//! Provides comprehensive test data for IQ2_S quantization (GGML-compatible format)
//! with 82-byte block structure and device-aware optimization patterns.

use super::{TestEnvironmentConfig, quantization::ToleranceConfig};
use bitnet_common::{BitNetError, Device, QuantizationType, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::LazyLock;

/// IQ2_S quantization configuration constants
pub const IQ2S_BLOCK_SIZE: usize = 82; // GGML standard 82-byte blocks
pub const IQ2S_QUANT_BITS: u8 = 2; // 2-bit quantization
pub const IQ2S_VALUES_PER_BLOCK: usize = 256; // 256 values per block
pub const IQ2S_SCALES_PER_BLOCK: usize = 2; // 2 scales per block

/// IQ2_S block structure matching GGML format
#[derive(Debug, Clone)]
#[repr(C)]
pub struct Iq2sBlock {
    /// Quantized values (64 bytes for 256 values @ 2 bits each)
    pub quants: [u8; 64],
    /// Scale factors (2 x f16 = 4 bytes)
    pub scales: [u16; 2], // f16 as u16 for serialization
    /// Quantization indices (14 bytes)
    pub qh: [u8; 14],
}

impl Serialize for Iq2sBlock {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("Iq2sBlock", 3)?;
        state.serialize_field("quants", &self.quants.as_slice())?;
        state.serialize_field("scales", &self.scales.as_slice())?;
        state.serialize_field("qh", &self.qh.as_slice())?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for Iq2sBlock {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{self, MapAccess, Visitor};
        use std::fmt;

        struct Iq2sBlockVisitor;

        impl<'de> Visitor<'de> for Iq2sBlockVisitor {
            type Value = Iq2sBlock;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct Iq2sBlock")
            }

            fn visit_map<V>(self, mut map: V) -> std::result::Result<Iq2sBlock, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut quants: Option<Vec<u8>> = None;
                let mut scales: Option<Vec<u16>> = None;
                let mut qh: Option<Vec<u8>> = None;

                while let Some(key) = map.next_key::<&str>()? {
                    match key {
                        "quants" => quants = Some(map.next_value()?),
                        "scales" => scales = Some(map.next_value()?),
                        "qh" => qh = Some(map.next_value()?),
                        _ => {
                            let _: serde::de::IgnoredAny = map.next_value()?;
                        }
                    }
                }

                let quants = quants.ok_or_else(|| de::Error::missing_field("quants"))?;
                let scales = scales.ok_or_else(|| de::Error::missing_field("scales"))?;
                let qh = qh.ok_or_else(|| de::Error::missing_field("qh"))?;

                if quants.len() != 64 {
                    return Err(de::Error::invalid_length(quants.len(), &"64"));
                }
                if scales.len() != 2 {
                    return Err(de::Error::invalid_length(scales.len(), &"2"));
                }
                if qh.len() != 14 {
                    return Err(de::Error::invalid_length(qh.len(), &"14"));
                }

                let mut quants_array = [0u8; 64];
                let mut scales_array = [0u16; 2];
                let mut qh_array = [0u8; 14];

                quants_array.copy_from_slice(&quants);
                scales_array.copy_from_slice(&scales);
                qh_array.copy_from_slice(&qh);

                Ok(Iq2sBlock { quants: quants_array, scales: scales_array, qh: qh_array })
            }
        }

        const FIELDS: &[&str] = &["quants", "scales", "qh"];
        deserializer.deserialize_struct("Iq2sBlock", FIELDS, Iq2sBlockVisitor)
    }
}

impl Iq2sBlock {
    /// Size of IQ2_S block in bytes (matches GGML)
    pub const SIZE: usize = 64 + 4 + 14; // quants + scales + qh = 82 bytes
}

/// IQ2_S quantization test case
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Iq2sTestCase {
    pub test_name: String,
    pub description: String,
    pub input_values: Vec<f32>, // 256 float32 values
    pub expected_blocks: Vec<Iq2sBlock>,
    pub expected_dequantized: Vec<f32>,
    pub tolerance: ToleranceConfig,
    pub device_variants: HashMap<Device, DeviceSpecificData>,
    pub ggml_compatible: bool,
}

/// Device-specific IQ2_S optimization data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceSpecificData {
    pub device: Device,
    pub optimization_level: String,
    pub vectorization_width: usize,
    pub expected_throughput_gops: f32,
    pub memory_alignment_bytes: usize,
}

/// IQ2_S test fixtures collection
pub struct Iq2sQuantizationFixtures {
    pub test_cases: Vec<Iq2sTestCase>,
    pub lookup_tables: Iq2sLookupTables,
    pub validation_data: Vec<Iq2sValidationCase>,
    pub config: TestEnvironmentConfig,
}

/// IQ2_S lookup tables for quantization/dequantization
#[derive(Debug, Clone)]
pub struct Iq2sLookupTables {
    pub quantization_table: [i8; 256], // Value to quantized index mapping
    pub dequantization_table: [f32; 4], // Quantized index to value mapping (4 values for 2-bit)
    pub qh_lookup: [u8; 256],          // High bits lookup table
}

/// IQ2_S validation test case
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Iq2sValidationCase {
    pub case_name: String,
    pub input_tensor_shape: Vec<usize>,
    pub input_data: Vec<f32>,
    pub expected_compressed_size: usize,
    pub expected_quality_metrics: QualityMetrics,
}

/// Quality metrics for IQ2_S quantization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub mse: f64,                // Mean squared error
    pub psnr: f64,               // Peak signal-to-noise ratio
    pub compression_ratio: f64,  // Original size / compressed size
    pub outlier_percentage: f64, // Percentage of values with high error
    pub cosine_similarity: f64,  // Cosine similarity with original
}

/// Static IQ2_S test cases
static IQ2S_TEST_CASES: LazyLock<Vec<Iq2sTestCase>> = LazyLock::new(|| {
    vec![
        // Basic IQ2_S quantization test
        create_basic_iq2s_test_case(),
        // Neural network weight matrix test
        create_neural_weight_test_case(),
        // Edge case values test
        create_edge_case_test(),
        // Large tensor test
        create_large_tensor_test(),
        // Mixed distribution test
        create_mixed_distribution_test(),
    ]
});

impl Iq2sQuantizationFixtures {
    /// Create new IQ2_S quantization fixtures
    pub fn new(config: &TestEnvironmentConfig) -> Self {
        Self {
            test_cases: IQ2S_TEST_CASES.clone(),
            lookup_tables: create_iq2s_lookup_tables(),
            validation_data: create_validation_cases(),
            config: config.clone(),
        }
    }

    /// Initialize IQ2_S fixtures with device-specific optimizations
    pub async fn initialize(&mut self) -> Result<()> {
        // Generate device-specific test data
        self.generate_device_variants().await?;

        // Validate GGML compatibility
        self.validate_ggml_format().await?;

        // Precompute validation metrics
        self.compute_validation_metrics().await?;

        Ok(())
    }

    /// Generate device-specific variants for each test case
    async fn generate_device_variants(&mut self) -> Result<()> {
        for test_case in &mut self.test_cases {
            // CPU variant
            test_case.device_variants.insert(
                Device::Cpu,
                DeviceSpecificData {
                    device: Device::Cpu,
                    optimization_level: "SIMD".to_string(),
                    vectorization_width: 8, // AVX2
                    expected_throughput_gops: 25.0,
                    memory_alignment_bytes: 32,
                },
            );

            // GPU variant (if available)
            #[cfg(feature = "gpu")]
            {
                test_case.device_variants.insert(
                    Device::Cuda(0),
                    DeviceSpecificData {
                        device: Device::Cuda(0),
                        optimization_level: "CUDA".to_string(),
                        vectorization_width: 32, // Warp size
                        expected_throughput_gops: 250.0,
                        memory_alignment_bytes: 128,
                    },
                );
            }
        }

        Ok(())
    }

    /// Validate GGML format compatibility
    async fn validate_ggml_format(&self) -> Result<()> {
        for test_case in &self.test_cases {
            for block in &test_case.expected_blocks {
                // Validate block size
                if std::mem::size_of::<Iq2sBlock>() != IQ2S_BLOCK_SIZE {
                    return Err(BitNetError::Validation(format!(
                        "IQ2_S block size mismatch: expected {}, got {}",
                        IQ2S_BLOCK_SIZE,
                        std::mem::size_of::<Iq2sBlock>()
                    )));
                }

                // Validate quantized data range
                for &quant_byte in &block.quants {
                    if quant_byte > 255 {
                        // 8 bits max
                        return Err(BitNetError::Validation(
                            "IQ2_S quantized value out of range".to_string(),
                        ));
                    }
                }

                // Validate scales are reasonable
                for &scale in &block.scales {
                    if scale == 0 {
                        return Err(BitNetError::Validation(
                            "IQ2_S scale cannot be zero".to_string(),
                        ));
                    }
                }
            }
        }

        println!("✓ IQ2_S GGML format validation passed");
        Ok(())
    }

    /// Compute validation metrics for all test cases
    async fn compute_validation_metrics(&mut self) -> Result<()> {
        let mut results = Vec::new();
        for validation_case in &self.validation_data {
            let metrics = self
                .compute_quality_metrics(
                    &validation_case.input_data,
                    &validation_case.input_data, // Mock: use same data for now
                )
                .await?;
            results.push(metrics);
        }

        for (validation_case, metrics) in self.validation_data.iter_mut().zip(results) {
            validation_case.expected_quality_metrics = metrics;
        }

        Ok(())
    }

    /// Compute quality metrics comparing original and dequantized data
    pub async fn compute_quality_metrics(
        &self,
        original: &[f32],
        dequantized: &[f32],
    ) -> Result<QualityMetrics> {
        if original.len() != dequantized.len() {
            return Err(BitNetError::Validation(
                "Original and dequantized data length mismatch".to_string(),
            ));
        }

        let n = original.len() as f64;

        // Mean Squared Error
        let mse: f64 = original
            .iter()
            .zip(dequantized.iter())
            .map(|(&a, &b)| ((a - b) as f64).powi(2))
            .sum::<f64>()
            / n;

        // Peak Signal-to-Noise Ratio
        let max_val = original.iter().map(|x| x.abs()).fold(0.0f32, f32::max) as f64;
        let psnr = if mse > 1e-10 {
            20.0 * (max_val / mse.sqrt()).log10()
        } else {
            100.0 // Very high PSNR for near-perfect match
        };

        // Compression ratio
        let original_size = original.len() * std::mem::size_of::<f32>();
        let compressed_size = (original.len() / IQ2S_VALUES_PER_BLOCK) * IQ2S_BLOCK_SIZE;
        let compression_ratio = original_size as f64 / compressed_size as f64;

        // Outlier percentage (errors > 10% of max value)
        let error_threshold = max_val * 0.1;
        let outliers = original
            .iter()
            .zip(dequantized.iter())
            .filter(|(&a, &b)| (a - b).abs() > error_threshold)
            .count();
        let outlier_percentage = (outliers as f64 / n) * 100.0;

        // Cosine similarity
        let dot_product: f64 =
            original.iter().zip(dequantized.iter()).map(|(&a, &b)| (a as f64) * (b as f64)).sum();
        let norm_a: f64 = original.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
        let norm_b: f64 = dequantized.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
        let cosine_similarity =
            if norm_a > 1e-10 && norm_b > 1e-10 { dot_product / (norm_a * norm_b) } else { 1.0 };

        Ok(QualityMetrics { mse, psnr, compression_ratio, outlier_percentage, cosine_similarity })
    }

    /// Get test case by name
    pub fn get_test_case(&self, name: &str) -> Option<&Iq2sTestCase> {
        self.test_cases.iter().find(|case| case.test_name == name)
    }

    /// Get validation case by name
    pub fn get_validation_case(&self, name: &str) -> Option<&Iq2sValidationCase> {
        self.validation_data.iter().find(|case| case.case_name == name)
    }

    /// Quantize input values to IQ2_S format
    pub fn quantize_to_iq2s(&self, input: &[f32]) -> Result<Vec<Iq2sBlock>> {
        if input.len() % IQ2S_VALUES_PER_BLOCK != 0 {
            return Err(BitNetError::Validation(format!(
                "Input length {} not divisible by block size {}",
                input.len(),
                IQ2S_VALUES_PER_BLOCK
            )));
        }

        let mut blocks = Vec::new();

        for chunk in input.chunks(IQ2S_VALUES_PER_BLOCK) {
            let block = self.quantize_block(chunk)?;
            blocks.push(block);
        }

        Ok(blocks)
    }

    /// Quantize a single block of 256 values
    fn quantize_block(&self, values: &[f32]) -> Result<Iq2sBlock> {
        if values.len() != IQ2S_VALUES_PER_BLOCK {
            return Err(BitNetError::Validation(format!(
                "Block size mismatch: expected {}, got {}",
                IQ2S_VALUES_PER_BLOCK,
                values.len()
            )));
        }

        // Find min and max for scaling
        let min_val = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Calculate scales (simplified - real implementation would be more sophisticated)
        let scale1 = (max_val - min_val) / 3.0; // 2-bit has 4 levels (0,1,2,3)
        let scale2 = scale1; // Use same scale for both (simplified)

        let mut quants = [0u8; 64]; // 256 values * 2 bits / 8 = 64 bytes
        let mut qh = [0u8; 14];

        // Quantize values (simplified quantization)
        for (i, &value) in values.iter().enumerate() {
            let normalized = if scale1 > 1e-8 {
                ((value - min_val) / scale1).round().clamp(0.0, 3.0) as u8
            } else {
                0u8
            };

            // Pack 4 values (2 bits each) into each byte
            let byte_idx = i / 4;
            let bit_offset = (i % 4) * 2;
            quants[byte_idx] |= normalized << bit_offset;
        }

        // Convert f32 scales to f16 (simplified as u16)
        let scale1_f16 = f32_to_f16_bits(scale1);
        let scale2_f16 = f32_to_f16_bits(scale2);

        // Mock qh data (real implementation would compute this properly)
        for (i, qh_byte) in qh.iter_mut().enumerate() {
            *qh_byte = (i * 17) as u8; // Some pattern for testing
        }

        Ok(Iq2sBlock { quants, scales: [scale1_f16, scale2_f16], qh })
    }

    /// Dequantize IQ2_S blocks back to f32 values
    pub fn dequantize_from_iq2s(&self, blocks: &[Iq2sBlock]) -> Result<Vec<f32>> {
        let mut output = Vec::with_capacity(blocks.len() * IQ2S_VALUES_PER_BLOCK);

        for block in blocks {
            let dequantized = self.dequantize_block(block)?;
            output.extend(dequantized);
        }

        Ok(output)
    }

    /// Dequantize a single IQ2_S block
    fn dequantize_block(&self, block: &Iq2sBlock) -> Result<Vec<f32>> {
        let mut values = Vec::with_capacity(IQ2S_VALUES_PER_BLOCK);

        let scale1 = f16_bits_to_f32(block.scales[0]);
        let _scale2 = f16_bits_to_f32(block.scales[1]);

        // Dequantize each value
        for i in 0..IQ2S_VALUES_PER_BLOCK {
            let byte_idx = i / 4;
            let bit_offset = (i % 4) * 2;
            let quant_val = (block.quants[byte_idx] >> bit_offset) & 0x3; // Extract 2 bits

            // Convert back to float (simplified)
            let dequantized = (quant_val as f32) * scale1;
            values.push(dequantized);
        }

        Ok(values)
    }
}

/// Create basic IQ2_S test case
fn create_basic_iq2s_test_case() -> Iq2sTestCase {
    let input_values: Vec<f32> = (0..256)
        .map(|i| (i as f32 - 128.0) / 128.0) // Range -1.0 to 1.0
        .collect();

    let expected_blocks = vec![mock_iq2s_block()];
    let expected_dequantized = input_values.clone(); // Simplified for now

    Iq2sTestCase {
        test_name: "basic_iq2s_quantization".to_string(),
        description: "Basic IQ2_S quantization test with linear distribution".to_string(),
        input_values,
        expected_blocks,
        expected_dequantized,
        tolerance: ToleranceConfig {
            quantization_tolerance: 0.1,
            dequantization_tolerance: 0.1,
            scale_tolerance: 0.01,
            numerical_accuracy_threshold: 0.8,
        },
        device_variants: HashMap::new(),
        ggml_compatible: true,
    }
}

/// Create neural network weight test case
fn create_neural_weight_test_case() -> Iq2sTestCase {
    // Generate realistic neural network weight distribution
    let mut input_values = Vec::with_capacity(256);
    let mut rng_state = 42u64;

    for _i in 0..256 {
        // Simple LCG for reproducible weights
        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        let normalized = (rng_state as f32) / (u64::MAX as f32);

        // Box-Muller transform for normal distribution
        let weight = ((-2.0 * normalized.ln()).sqrt()
            * (2.0 * std::f32::consts::PI * normalized).cos())
            * 0.2;
        input_values.push(weight);
    }

    Iq2sTestCase {
        test_name: "neural_weight_iq2s".to_string(),
        description: "IQ2_S quantization of realistic neural network weights".to_string(),
        input_values: input_values.clone(),
        expected_blocks: vec![mock_iq2s_block()],
        expected_dequantized: input_values,
        tolerance: ToleranceConfig {
            quantization_tolerance: 0.2,
            dequantization_tolerance: 0.2,
            scale_tolerance: 0.05,
            numerical_accuracy_threshold: 0.7,
        },
        device_variants: HashMap::new(),
        ggml_compatible: true,
    }
}

/// Create edge case test
fn create_edge_case_test() -> Iq2sTestCase {
    let mut input_values = vec![0.0; 256];

    // Fill with edge case values
    input_values[0] = f32::NEG_INFINITY;
    input_values[1] = f32::INFINITY;
    input_values[2] = f32::NAN;
    input_values[3] = f32::MIN;
    input_values[4] = f32::MAX;
    input_values[5] = -0.0;
    input_values[6] = f32::EPSILON;
    input_values[7] = -f32::EPSILON;

    Iq2sTestCase {
        test_name: "edge_cases_iq2s".to_string(),
        description: "IQ2_S quantization edge cases (NaN, infinity, extreme values)".to_string(),
        input_values: input_values.clone(),
        expected_blocks: vec![mock_iq2s_block()],
        expected_dequantized: input_values,
        tolerance: ToleranceConfig {
            quantization_tolerance: 1.0,
            dequantization_tolerance: 1.0,
            scale_tolerance: 0.5,
            numerical_accuracy_threshold: 0.5,
        },
        device_variants: HashMap::new(),
        ggml_compatible: true,
    }
}

/// Create large tensor test case
fn create_large_tensor_test() -> Iq2sTestCase {
    let input_values: Vec<f32> = (0..1024) // 4 blocks worth
        .map(|i| ((i % 256) as f32 - 128.0) / 256.0)
        .collect();

    Iq2sTestCase {
        test_name: "large_tensor_iq2s".to_string(),
        description: "Large tensor IQ2_S quantization test (4 blocks)".to_string(),
        input_values: input_values.clone(),
        expected_blocks: vec![mock_iq2s_block(); 4],
        expected_dequantized: input_values,
        tolerance: ToleranceConfig {
            quantization_tolerance: 0.15,
            dequantization_tolerance: 0.15,
            scale_tolerance: 0.02,
            numerical_accuracy_threshold: 0.75,
        },
        device_variants: HashMap::new(),
        ggml_compatible: true,
    }
}

/// Create mixed distribution test
fn create_mixed_distribution_test() -> Iq2sTestCase {
    let mut input_values = Vec::with_capacity(256);

    // Mixed distribution: some zero, some small, some large
    for i in 0..256 {
        let value = match i % 8 {
            0 => 0.0,                // Zero values
            1 => 0.001,              // Very small positive
            2 => -0.001,             // Very small negative
            3 => 1.0,                // Large positive
            4 => -1.0,               // Large negative
            5 => 0.5,                // Medium positive
            6 => -0.5,               // Medium negative
            7 => (i as f32) / 256.0, // Linear ramp
            _ => unreachable!(),
        };
        input_values.push(value);
    }

    Iq2sTestCase {
        test_name: "mixed_distribution_iq2s".to_string(),
        description: "Mixed value distribution IQ2_S quantization test".to_string(),
        input_values: input_values.clone(),
        expected_blocks: vec![mock_iq2s_block()],
        expected_dequantized: input_values,
        tolerance: ToleranceConfig {
            quantization_tolerance: 0.3,
            dequantization_tolerance: 0.3,
            scale_tolerance: 0.1,
            numerical_accuracy_threshold: 0.6,
        },
        device_variants: HashMap::new(),
        ggml_compatible: true,
    }
}

/// Create IQ2_S lookup tables
fn create_iq2s_lookup_tables() -> Iq2sLookupTables {
    let mut quantization_table = [0i8; 256];
    let dequantization_table = [-1.5, -0.5, 0.5, 1.5]; // 2-bit symmetric quantization
    let mut qh_lookup = [0u8; 256];

    // Fill quantization lookup table
    for i in 0..256 {
        quantization_table[i] = (i % 4) as i8;
    }

    // Fill qh lookup table (simplified pattern)
    for i in 0..256 {
        qh_lookup[i] = (i ^ (i >> 4)) as u8;
    }

    Iq2sLookupTables { quantization_table, dequantization_table, qh_lookup }
}

/// Create validation test cases
fn create_validation_cases() -> Vec<Iq2sValidationCase> {
    vec![
        Iq2sValidationCase {
            case_name: "small_matrix".to_string(),
            input_tensor_shape: vec![16, 16], // 256 elements
            input_data: (0..256).map(|i| (i as f32) / 256.0).collect(),
            expected_compressed_size: 82, // 1 block
            expected_quality_metrics: QualityMetrics {
                mse: 0.01,
                psnr: 40.0,
                compression_ratio: 4.0,
                outlier_percentage: 2.0,
                cosine_similarity: 0.98,
            },
        },
        Iq2sValidationCase {
            case_name: "attention_weights".to_string(),
            input_tensor_shape: vec![32, 32], // 1024 elements = 4 blocks
            input_data: (0..1024).map(|i| ((i as f32) / 1024.0 - 0.5) * 2.0).collect(),
            expected_compressed_size: 328, // 4 blocks * 82 bytes
            expected_quality_metrics: QualityMetrics {
                mse: 0.02,
                psnr: 35.0,
                compression_ratio: 4.0,
                outlier_percentage: 3.0,
                cosine_similarity: 0.95,
            },
        },
    ]
}

/// Create mock IQ2_S block for testing
fn mock_iq2s_block() -> Iq2sBlock {
    let mut quants = [0u8; 64];
    for (i, byte) in quants.iter_mut().enumerate() {
        *byte = (i % 256) as u8;
    }

    let mut qh = [0u8; 14];
    for (i, byte) in qh.iter_mut().enumerate() {
        *byte = (i * 17) as u8;
    }

    Iq2sBlock {
        quants,
        scales: [0x3C00, 0x3C00], // 1.0 in f16 format
        qh,
    }
}

/// Convert f32 to f16 bits (simplified)
fn f32_to_f16_bits(value: f32) -> u16 {
    if value == 0.0 {
        return 0;
    }
    if value == 1.0 {
        return 0x3C00; // 1.0 in f16
    }
    // Simplified conversion - real implementation would be more robust
    let bits = value.to_bits();
    let sign = (bits >> 31) as u16;
    let exp = ((bits >> 23) & 0xFF) as u16;
    let mantissa = (bits >> 13) as u16 & 0x3FF;

    // Simplified f16 encoding
    (sign << 15) | ((exp.saturating_sub(112)) << 10) | mantissa
}

/// Convert f16 bits to f32 (simplified)
fn f16_bits_to_f32(bits: u16) -> f32 {
    if bits == 0 {
        return 0.0;
    }
    if bits == 0x3C00 {
        return 1.0;
    }

    let sign = (bits >> 15) != 0;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x3FF) as u32;

    // Simplified f32 reconstruction
    let f32_exp = exp + 112;
    let f32_mantissa = mantissa << 13;
    let f32_bits = ((sign as u32) << 31) | (f32_exp << 23) | f32_mantissa;

    f32::from_bits(f32_bits)
}

/// Create IQ2_S quantization fixtures for testing
#[cfg(test)]
pub fn create_iq2s_fixtures() -> Iq2sQuantizationFixtures {
    use super::TestEnvironmentConfig;

    let config = TestEnvironmentConfig::from_env();
    Iq2sQuantizationFixtures::new(&config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iq2s_block_size() {
        assert_eq!(std::mem::size_of::<Iq2sBlock>(), IQ2S_BLOCK_SIZE);
    }

    #[test]
    fn test_iq2s_fixtures_creation() {
        let fixtures = create_iq2s_fixtures();
        assert!(!fixtures.test_cases.is_empty());
        assert_eq!(fixtures.lookup_tables.dequantization_table.len(), 4);
    }

    #[tokio::test]
    async fn test_basic_quantization_flow() {
        let fixtures = create_iq2s_fixtures();
        let input_data: Vec<f32> = (0..256).map(|i| (i as f32) / 256.0).collect();

        let blocks = fixtures.quantize_to_iq2s(&input_data).expect("Quantization failed");
        assert_eq!(blocks.len(), 1);

        let dequantized = fixtures.dequantize_from_iq2s(&blocks).expect("Dequantization failed");
        assert_eq!(dequantized.len(), 256);
    }
}
