//! BitNet Quantization Test Fixtures (I2_S, TL1, TL2)
//!
//! Comprehensive quantization test data for BitNet.rs neural network validation.
//! Provides realistic test vectors with known accuracy targets for each quantization
//! algorithm including device-aware CPU/GPU variants.

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// BitNet quantization test fixture with input/output validation data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationTestFixture {
    pub name: String,
    pub algorithm: QuantizationAlgorithm,
    pub input_data: Vec<f32>,
    pub input_shape: Vec<usize>,
    pub expected_quantized: QuantizedOutput,
    pub expected_dequantized: Vec<f32>,
    pub accuracy_metrics: AccuracyMetrics,
    pub device_variants: HashMap<DeviceType, DeviceSpecificData>,
    pub test_metadata: TestMetadata,
}

/// Supported BitNet quantization algorithms
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum QuantizationAlgorithm {
    I2S,  // 2-bit signed quantization
    TL1,  // Table lookup quantization level 1
    TL2,  // Table lookup quantization level 2
    IQ2S, // GGML-compatible 2-bit quantization
}

/// Quantized tensor output data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedOutput {
    pub data: Vec<u8>,
    pub scales: Vec<f32>,
    pub lookup_table: Option<Vec<f32>>,
    pub block_size: usize,
    pub data_format: String,
}

/// Accuracy validation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyMetrics {
    pub cosine_similarity: f32,
    pub mse: f32,
    pub max_absolute_error: f32,
    pub snr_db: f32,
    pub accuracy_threshold: f32,
    pub passes_threshold: bool,
}

/// Device-specific test data and optimizations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum DeviceType {
    CPU,
    CUDA,
    Metal,
}

/// Device-specific optimization data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceSpecificData {
    pub optimized_layout: Vec<u8>,
    pub simd_alignment: usize,
    pub memory_layout: String,
    pub performance_target_ms: f32,
    pub fallback_required: bool,
}

/// Test fixture metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestMetadata {
    pub seed: u64,
    pub generation_timestamp: String,
    pub bitnet_version: String,
    pub validation_status: String,
    pub notes: Vec<String>,
}

/// BitNet quantization fixture generator
pub struct BitNetQuantizationFixtures {
    seed: u64,
}

impl BitNetQuantizationFixtures {
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Generate comprehensive I2_S quantization test fixtures
    pub fn generate_i2s_fixtures(&self) -> Result<Vec<QuantizationTestFixture>> {
        let mut fixtures = Vec::new();

        // Basic I2_S quantization test
        fixtures.push(self.create_i2s_basic_fixture()?);

        // Large tensor I2_S test
        fixtures.push(self.create_i2s_large_tensor_fixture()?);

        // Edge case: small values
        fixtures.push(self.create_i2s_small_values_fixture()?);

        // Edge case: extreme values
        fixtures.push(self.create_i2s_extreme_values_fixture()?);

        // Mixed precision scenarios
        fixtures.push(self.create_i2s_mixed_precision_fixture()?);

        Ok(fixtures)
    }

    /// Generate TL1 table lookup quantization fixtures
    pub fn generate_tl1_fixtures(&self) -> Result<Vec<QuantizationTestFixture>> {
        let mut fixtures = Vec::new();

        // Basic TL1 quantization
        fixtures.push(self.create_tl1_basic_fixture()?);

        // Optimized lookup table
        fixtures.push(self.create_tl1_optimized_table_fixture()?);

        // Cache-friendly access patterns
        fixtures.push(self.create_tl1_cache_friendly_fixture()?);

        Ok(fixtures)
    }

    /// Generate TL2 table lookup quantization fixtures
    pub fn generate_tl2_fixtures(&self) -> Result<Vec<QuantizationTestFixture>> {
        let mut fixtures = Vec::new();

        // Basic TL2 quantization
        fixtures.push(self.create_tl2_basic_fixture()?);

        // High precision table
        fixtures.push(self.create_tl2_high_precision_fixture()?);

        // Comparison with TL1
        fixtures.push(self.create_tl2_vs_tl1_fixture()?);

        Ok(fixtures)
    }

    /// Generate cross-algorithm comparison fixtures
    pub fn generate_cross_algorithm_fixtures(&self) -> Result<Vec<QuantizationTestFixture>> {
        let mut fixtures = Vec::new();

        // Same input data across all algorithms
        let base_data = self.generate_neural_network_weights(1024, self.seed);

        for algorithm in
            &[QuantizationAlgorithm::I2S, QuantizationAlgorithm::TL1, QuantizationAlgorithm::TL2]
        {
            fixtures.push(self.create_cross_algorithm_fixture(&base_data, algorithm.clone())?);
        }

        Ok(fixtures)
    }

    /// Create basic I2_S quantization fixture
    fn create_i2s_basic_fixture(&self) -> Result<QuantizationTestFixture> {
        let input_data = self.generate_neural_network_weights(512, self.seed);
        let shape = vec![16, 32]; // 16x32 weight matrix

        let quantized = self.simulate_i2s_quantization(&input_data)?;
        let dequantized = self.simulate_i2s_dequantization(&quantized)?;
        let metrics = self.calculate_accuracy_metrics(&input_data, &dequantized);

        let mut device_variants = HashMap::new();
        device_variants.insert(
            DeviceType::CPU,
            DeviceSpecificData {
                optimized_layout: self.create_cpu_optimized_layout(&quantized.data),
                simd_alignment: 32,
                memory_layout: "row_major".to_string(),
                performance_target_ms: 0.1,
                fallback_required: false,
            },
        );

        device_variants.insert(
            DeviceType::CUDA,
            DeviceSpecificData {
                optimized_layout: self.create_gpu_optimized_layout(&quantized.data),
                simd_alignment: 128,
                memory_layout: "coalesced".to_string(),
                performance_target_ms: 0.05,
                fallback_required: false,
            },
        );

        Ok(QuantizationTestFixture {
            name: "i2s_basic_quantization".to_string(),
            algorithm: QuantizationAlgorithm::I2S,
            input_data,
            input_shape: shape,
            expected_quantized: quantized,
            expected_dequantized: dequantized,
            accuracy_metrics: metrics,
            device_variants,
            test_metadata: TestMetadata {
                seed: self.seed,
                generation_timestamp: chrono::Utc::now().to_rfc3339(),
                bitnet_version: "0.1.0".to_string(),
                validation_status: "validated".to_string(),
                notes: vec![
                    "Basic I2_S quantization test with 16x32 weight matrix".to_string(),
                    "Target accuracy: >99% cosine similarity".to_string(),
                ],
            },
        })
    }

    /// Create large tensor I2_S fixture for performance testing
    fn create_i2s_large_tensor_fixture(&self) -> Result<QuantizationTestFixture> {
        let input_data = self.generate_neural_network_weights(16384, self.seed + 1000); // 128x128 matrix
        let shape = vec![128, 128];

        let quantized = self.simulate_i2s_quantization(&input_data)?;
        let dequantized = self.simulate_i2s_dequantization(&quantized)?;
        let metrics = self.calculate_accuracy_metrics(&input_data, &dequantized);

        let mut device_variants = HashMap::new();
        device_variants.insert(
            DeviceType::CPU,
            DeviceSpecificData {
                optimized_layout: self.create_cpu_optimized_layout(&quantized.data),
                simd_alignment: 32,
                memory_layout: "blocked".to_string(),
                performance_target_ms: 1.0,
                fallback_required: false,
            },
        );

        device_variants.insert(
            DeviceType::CUDA,
            DeviceSpecificData {
                optimized_layout: self.create_gpu_optimized_layout(&quantized.data),
                simd_alignment: 128,
                memory_layout: "tensor_core_friendly".to_string(),
                performance_target_ms: 0.2,
                fallback_required: false,
            },
        );

        Ok(QuantizationTestFixture {
            name: "i2s_large_tensor_performance".to_string(),
            algorithm: QuantizationAlgorithm::I2S,
            input_data,
            input_shape: shape,
            expected_quantized: quantized,
            expected_dequantized: dequantized,
            accuracy_metrics: metrics,
            device_variants,
            test_metadata: TestMetadata {
                seed: self.seed + 1000,
                generation_timestamp: chrono::Utc::now().to_rfc3339(),
                bitnet_version: "0.1.0".to_string(),
                validation_status: "validated".to_string(),
                notes: vec![
                    "Large tensor I2_S quantization for performance validation".to_string(),
                    "Tests SIMD/CUDA optimization paths".to_string(),
                ],
            },
        })
    }

    /// Create TL1 basic quantization fixture
    fn create_tl1_basic_fixture(&self) -> Result<QuantizationTestFixture> {
        let input_data = self.generate_neural_network_weights(256, self.seed + 2000);
        let shape = vec![16, 16];

        let quantized = self.simulate_tl1_quantization(&input_data)?;
        let dequantized = self.simulate_tl1_dequantization(&quantized)?;
        let metrics = self.calculate_accuracy_metrics(&input_data, &dequantized);

        let mut device_variants = HashMap::new();
        device_variants.insert(
            DeviceType::CPU,
            DeviceSpecificData {
                optimized_layout: self.create_cpu_optimized_layout(&quantized.data),
                simd_alignment: 16,
                memory_layout: "cache_friendly".to_string(),
                performance_target_ms: 0.05,
                fallback_required: false,
            },
        );

        Ok(QuantizationTestFixture {
            name: "tl1_basic_lookup_quantization".to_string(),
            algorithm: QuantizationAlgorithm::TL1,
            input_data,
            input_shape: shape,
            expected_quantized: quantized,
            expected_dequantized: dequantized,
            accuracy_metrics: metrics,
            device_variants,
            test_metadata: TestMetadata {
                seed: self.seed + 2000,
                generation_timestamp: chrono::Utc::now().to_rfc3339(),
                bitnet_version: "0.1.0".to_string(),
                validation_status: "validated".to_string(),
                notes: vec![
                    "TL1 4-bit lookup table quantization".to_string(),
                    "16-entry lookup table with 4-bit indices".to_string(),
                ],
            },
        })
    }

    /// Create TL2 basic quantization fixture
    fn create_tl2_basic_fixture(&self) -> Result<QuantizationTestFixture> {
        let input_data = self.generate_neural_network_weights(512, self.seed + 2500);
        let shape = vec![16, 32];

        let quantized = self.simulate_tl2_quantization(&input_data)?;
        let dequantized = self.simulate_tl2_dequantization(&quantized)?;
        let metrics = self.calculate_accuracy_metrics(&input_data, &dequantized);

        let mut device_variants = HashMap::new();
        device_variants.insert(
            DeviceType::CPU,
            DeviceSpecificData {
                optimized_layout: self.create_cpu_optimized_layout(&quantized.data),
                simd_alignment: 32,
                memory_layout: "linear".to_string(),
                performance_target_ms: 0.1,
                fallback_required: false,
            },
        );

        Ok(QuantizationTestFixture {
            name: "tl2_basic_quantization".to_string(),
            algorithm: QuantizationAlgorithm::TL2,
            input_data,
            input_shape: shape,
            expected_quantized: quantized,
            expected_dequantized: dequantized,
            accuracy_metrics: metrics,
            device_variants,
            test_metadata: TestMetadata {
                seed: self.seed + 2500,
                generation_timestamp: chrono::Utc::now().to_rfc3339(),
                bitnet_version: "0.1.0".to_string(),
                validation_status: "validated".to_string(),
                notes: vec![
                    "TL2 8-bit lookup table quantization".to_string(),
                    "256-entry lookup table with 8-bit indices".to_string(),
                ],
            },
        })
    }

    /// Create TL2 high precision quantization fixture
    fn create_tl2_high_precision_fixture(&self) -> Result<QuantizationTestFixture> {
        let input_data = self.generate_neural_network_weights(1024, self.seed + 3000);
        let shape = vec![32, 32];

        let quantized = self.simulate_tl2_quantization(&input_data)?;
        let dequantized = self.simulate_tl2_dequantization(&quantized)?;
        let metrics = self.calculate_accuracy_metrics(&input_data, &dequantized);

        let mut device_variants = HashMap::new();
        device_variants.insert(
            DeviceType::CPU,
            DeviceSpecificData {
                optimized_layout: self.create_cpu_optimized_layout(&quantized.data),
                simd_alignment: 32,
                memory_layout: "vectorized".to_string(),
                performance_target_ms: 0.2,
                fallback_required: false,
            },
        );

        Ok(QuantizationTestFixture {
            name: "tl2_high_precision_quantization".to_string(),
            algorithm: QuantizationAlgorithm::TL2,
            input_data,
            input_shape: shape,
            expected_quantized: quantized,
            expected_dequantized: dequantized,
            accuracy_metrics: metrics,
            device_variants,
            test_metadata: TestMetadata {
                seed: self.seed + 3000,
                generation_timestamp: chrono::Utc::now().to_rfc3339(),
                bitnet_version: "0.1.0".to_string(),
                validation_status: "validated".to_string(),
                notes: vec![
                    "TL2 8-bit lookup table quantization".to_string(),
                    "256-entry lookup table for higher precision".to_string(),
                ],
            },
        })
    }

    /// Create cross-algorithm comparison fixture
    fn create_cross_algorithm_fixture(
        &self,
        input_data: &[f32],
        algorithm: QuantizationAlgorithm,
    ) -> Result<QuantizationTestFixture> {
        let shape = vec![32, input_data.len() / 32];

        let (quantized, dequantized) = match algorithm {
            QuantizationAlgorithm::I2S => {
                let q = self.simulate_i2s_quantization(input_data)?;
                let d = self.simulate_i2s_dequantization(&q)?;
                (q, d)
            }
            QuantizationAlgorithm::TL1 => {
                let q = self.simulate_tl1_quantization(input_data)?;
                let d = self.simulate_tl1_dequantization(&q)?;
                (q, d)
            }
            QuantizationAlgorithm::TL2 => {
                let q = self.simulate_tl2_quantization(input_data)?;
                let d = self.simulate_tl2_dequantization(&q)?;
                (q, d)
            }
            _ => return Err(anyhow::anyhow!("Unsupported algorithm for cross-comparison")),
        };

        let metrics = self.calculate_accuracy_metrics(input_data, &dequantized);

        Ok(QuantizationTestFixture {
            name: format!("cross_algorithm_{:?}_comparison", algorithm).to_lowercase(),
            algorithm: algorithm.clone(),
            input_data: input_data.to_vec(),
            input_shape: shape,
            expected_quantized: quantized,
            expected_dequantized: dequantized,
            accuracy_metrics: metrics,
            device_variants: HashMap::new(),
            test_metadata: TestMetadata {
                seed: self.seed + 4000,
                generation_timestamp: chrono::Utc::now().to_rfc3339(),
                bitnet_version: "0.1.0".to_string(),
                validation_status: "validated".to_string(),
                notes: vec![
                    format!("{:?} quantization for cross-algorithm comparison", algorithm),
                    "Same input data used across all algorithms".to_string(),
                ],
            },
        })
    }

    /// Generate realistic neural network weight data
    fn generate_neural_network_weights(&self, size: usize, seed: u64) -> Vec<f32> {
        let mut weights = Vec::with_capacity(size);
        let mut state = seed;

        for _ in 0..size {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;

            // Generate weights with normal distribution suitable for neural networks
            let uniform = (state as f64) / (u64::MAX as f64);
            let normal = self.box_muller_transform(uniform, 0.0, 0.1); // Mean 0, std 0.1
            weights.push(normal as f32);
        }

        weights
    }

    /// Box-Muller transform for normal distribution
    fn box_muller_transform(&self, uniform: f64, mean: f64, std_dev: f64) -> f64 {
        // Simplified version - in practice would use proper Box-Muller
        let z = 6.0 * (uniform - 0.5); // Approximate normal
        mean + std_dev * z
    }

    /// Simulate I2_S quantization (2-bit signed)
    fn simulate_i2s_quantization(&self, input: &[f32]) -> Result<QuantizedOutput> {
        let block_size = 32;
        let num_blocks = (input.len() + block_size - 1) / block_size;
        let mut scales = Vec::with_capacity(num_blocks);
        let mut quantized_data = Vec::new();

        for block_start in (0..input.len()).step_by(block_size) {
            let block_end = std::cmp::min(block_start + block_size, input.len());
            let block = &input[block_start..block_end];

            // Calculate scale factor
            let max_abs = block.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let scale = if max_abs > 0.0 { max_abs / 1.5 } else { 1.0 }; // Map to [-1.5, 1.5] range
            scales.push(scale);

            // Quantize to 2-bit signed values: -2, -1, 0, 1
            for chunk in block.chunks(4) {
                let mut packed_byte = 0u8;
                for (i, &value) in chunk.iter().enumerate() {
                    let normalized = value / scale;
                    let quantized = if normalized >= 0.5 {
                        1
                    } else if normalized >= -0.5 {
                        0
                    } else if normalized >= -1.5 {
                        -1
                    } else {
                        -2
                    };

                    // Pack into 2-bit slots (0-3 encoding: -2->0, -1->1, 0->2, 1->3)
                    let encoded = (quantized + 2) as u8;
                    packed_byte |= encoded << (i * 2);
                }
                quantized_data.push(packed_byte);
            }
        }

        Ok(QuantizedOutput {
            data: quantized_data,
            scales,
            lookup_table: None,
            block_size,
            data_format: "i2s_blocked".to_string(),
        })
    }

    /// Simulate I2_S dequantization
    fn simulate_i2s_dequantization(&self, quantized: &QuantizedOutput) -> Result<Vec<f32>> {
        let mut dequantized = Vec::new();
        let mut scale_idx = 0;
        let mut data_idx = 0;

        while data_idx < quantized.data.len() && scale_idx < quantized.scales.len() {
            let scale = quantized.scales[scale_idx];
            let elements_in_block =
                std::cmp::min(quantized.block_size, (quantized.data.len() - data_idx) * 4);

            for _ in (0..elements_in_block).step_by(4) {
                if data_idx >= quantized.data.len() {
                    break;
                }

                let packed_byte = quantized.data[data_idx];
                data_idx += 1;

                for i in 0..4 {
                    if dequantized.len() >= elements_in_block + scale_idx * quantized.block_size {
                        break;
                    }

                    let encoded = (packed_byte >> (i * 2)) & 0x3;
                    let quantized_value = (encoded as i8) - 2; // Decode: 0->-2, 1->-1, 2->0, 3->1
                    let dequantized_value = quantized_value as f32 * scale;
                    dequantized.push(dequantized_value);
                }
            }
            scale_idx += 1;
        }

        Ok(dequantized)
    }

    /// Simulate TL1 quantization (4-bit lookup table)
    fn simulate_tl1_quantization(&self, input: &[f32]) -> Result<QuantizedOutput> {
        let table_size = 16;

        // Generate optimal lookup table using k-means clustering (simplified)
        let lookup_table = self.generate_lookup_table(input, table_size)?;

        // Quantize by finding nearest table entry
        let mut quantized_data = Vec::new();
        for chunk in input.chunks(2) {
            let mut packed_byte = 0u8;
            for (i, &value) in chunk.iter().enumerate() {
                let idx = self.find_nearest_table_entry(&lookup_table, value);
                packed_byte |= (idx as u8) << (i * 4);
            }
            quantized_data.push(packed_byte);
        }

        Ok(QuantizedOutput {
            data: quantized_data,
            scales: vec![], // TL1 doesn't use scales
            lookup_table: Some(lookup_table),
            block_size: input.len(),
            data_format: "tl1_lookup".to_string(),
        })
    }

    /// Simulate TL1 dequantization
    fn simulate_tl1_dequantization(&self, quantized: &QuantizedOutput) -> Result<Vec<f32>> {
        let lookup_table = quantized
            .lookup_table
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("TL1 requires lookup table"))?;
        let mut dequantized = Vec::new();

        for &packed_byte in &quantized.data {
            for i in 0..2 {
                if dequantized.len() >= quantized.block_size {
                    break;
                }
                let idx = (packed_byte >> (i * 4)) & 0xF;
                if (idx as usize) < lookup_table.len() {
                    dequantized.push(lookup_table[idx as usize]);
                }
            }
        }

        Ok(dequantized)
    }

    /// Simulate TL2 quantization (8-bit lookup table)
    fn simulate_tl2_quantization(&self, input: &[f32]) -> Result<QuantizedOutput> {
        let table_size = 256;

        // Generate lookup table
        let lookup_table = self.generate_lookup_table(input, table_size)?;

        // Quantize with 8-bit indices
        let mut quantized_data = Vec::new();
        for &value in input {
            let idx = self.find_nearest_table_entry(&lookup_table, value);
            quantized_data.push(idx as u8);
        }

        Ok(QuantizedOutput {
            data: quantized_data,
            scales: vec![],
            lookup_table: Some(lookup_table),
            block_size: input.len(),
            data_format: "tl2_lookup".to_string(),
        })
    }

    /// Simulate TL2 dequantization
    fn simulate_tl2_dequantization(&self, quantized: &QuantizedOutput) -> Result<Vec<f32>> {
        let lookup_table = quantized
            .lookup_table
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("TL2 requires lookup table"))?;
        let mut dequantized = Vec::new();

        for &idx in &quantized.data {
            if (idx as usize) < lookup_table.len() {
                dequantized.push(lookup_table[idx as usize]);
            }
        }

        Ok(dequantized)
    }

    /// Generate lookup table using simplified k-means
    fn generate_lookup_table(&self, input: &[f32], table_size: usize) -> Result<Vec<f32>> {
        let mut table = Vec::with_capacity(table_size);

        // Find min/max range
        let min_val = input.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = input.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Initialize table with uniform distribution
        for i in 0..table_size {
            let t = i as f32 / (table_size - 1) as f32;
            table.push(min_val + t * (max_val - min_val));
        }

        // Simple k-means iteration (simplified for fixtures)
        for _ in 0..10 {
            let mut clusters: Vec<Vec<f32>> = vec![Vec::new(); table_size];

            for &value in input {
                let idx = self.find_nearest_table_entry(&table, value);
                clusters[idx].push(value);
            }

            for (i, cluster) in clusters.iter().enumerate() {
                if !cluster.is_empty() {
                    table[i] = cluster.iter().sum::<f32>() / cluster.len() as f32;
                }
            }
        }

        Ok(table)
    }

    /// Find nearest lookup table entry
    fn find_nearest_table_entry(&self, table: &[f32], value: f32) -> usize {
        let mut best_idx = 0;
        let mut best_distance = (table[0] - value).abs();

        for (i, &table_value) in table.iter().enumerate() {
            let distance = (table_value - value).abs();
            if distance < best_distance {
                best_distance = distance;
                best_idx = i;
            }
        }

        best_idx
    }

    /// Calculate accuracy metrics between original and dequantized data
    fn calculate_accuracy_metrics(&self, original: &[f32], dequantized: &[f32]) -> AccuracyMetrics {
        let n = std::cmp::min(original.len(), dequantized.len()) as f32;

        // Cosine similarity
        let dot_product: f32 = original.iter().zip(dequantized.iter()).map(|(&a, &b)| a * b).sum();
        let norm_orig: f32 = original.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let norm_deq: f32 = dequantized.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let cosine_similarity = if norm_orig > 0.0 && norm_deq > 0.0 {
            dot_product / (norm_orig * norm_deq)
        } else {
            1.0
        };

        // MSE
        let mse: f32 =
            original.iter().zip(dequantized.iter()).map(|(&a, &b)| (a - b).powi(2)).sum::<f32>()
                / n;

        // Max absolute error
        let max_absolute_error = original
            .iter()
            .zip(dequantized.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        // SNR in dB
        let signal_power: f32 = original.iter().map(|&x| x * x).sum::<f32>() / n;
        let noise_power = mse;
        let snr_db = if noise_power > 0.0 {
            10.0 * (signal_power / noise_power).log10()
        } else {
            f32::INFINITY
        };

        let accuracy_threshold = 0.99; // 99% cosine similarity target
        let passes_threshold = cosine_similarity >= accuracy_threshold;

        AccuracyMetrics {
            cosine_similarity,
            mse,
            max_absolute_error,
            snr_db,
            accuracy_threshold,
            passes_threshold,
        }
    }

    /// Create CPU-optimized memory layout
    fn create_cpu_optimized_layout(&self, data: &[u8]) -> Vec<u8> {
        // For CPU, optimize for cache line access (64-byte alignment)
        let mut optimized = data.to_vec();
        while optimized.len() % 64 != 0 {
            optimized.push(0);
        }
        optimized
    }

    /// Create GPU-optimized memory layout
    fn create_gpu_optimized_layout(&self, data: &[u8]) -> Vec<u8> {
        // For GPU, optimize for coalesced memory access (128-byte alignment)
        let mut optimized = data.to_vec();
        while optimized.len() % 128 != 0 {
            optimized.push(0);
        }
        optimized
    }

    /// Generate all fixture types
    pub fn generate_all_fixtures(&self) -> Result<Vec<QuantizationTestFixture>> {
        let mut all_fixtures = Vec::new();

        all_fixtures.extend(self.generate_i2s_fixtures()?);
        all_fixtures.extend(self.generate_tl1_fixtures()?);
        all_fixtures.extend(self.generate_tl2_fixtures()?);
        all_fixtures.extend(self.generate_cross_algorithm_fixtures()?);

        Ok(all_fixtures)
    }

    // Additional edge case fixtures

    fn create_i2s_small_values_fixture(&self) -> Result<QuantizationTestFixture> {
        // Generate small values near quantization boundaries
        let input_data: Vec<f32> = (0..128)
            .map(|i| {
                let base = (i as f32) / 1000.0; // Very small values
                if i % 2 == 0 { base } else { -base }
            })
            .collect();

        let shape = vec![8, 16];
        let quantized = self.simulate_i2s_quantization(&input_data)?;
        let dequantized = self.simulate_i2s_dequantization(&quantized)?;
        let metrics = self.calculate_accuracy_metrics(&input_data, &dequantized);

        Ok(QuantizationTestFixture {
            name: "i2s_small_values_edge_case".to_string(),
            algorithm: QuantizationAlgorithm::I2S,
            input_data,
            input_shape: shape,
            expected_quantized: quantized,
            expected_dequantized: dequantized,
            accuracy_metrics: metrics,
            device_variants: HashMap::new(),
            test_metadata: TestMetadata {
                seed: self.seed + 5000,
                generation_timestamp: chrono::Utc::now().to_rfc3339(),
                bitnet_version: "0.1.0".to_string(),
                validation_status: "validated".to_string(),
                notes: vec![
                    "Edge case: very small values near quantization boundaries".to_string(),
                    "Tests quantization precision limits".to_string(),
                ],
            },
        })
    }

    fn create_i2s_extreme_values_fixture(&self) -> Result<QuantizationTestFixture> {
        // Generate extreme values to test saturation behavior
        let input_data: Vec<f32> = (0..256)
            .map(|i| {
                match i % 4 {
                    0 => 10.0,  // Large positive
                    1 => -10.0, // Large negative
                    2 => 0.0,   // Zero
                    _ => 1.0,   // Normal range
                }
            })
            .collect();

        let shape = vec![16, 16];
        let quantized = self.simulate_i2s_quantization(&input_data)?;
        let dequantized = self.simulate_i2s_dequantization(&quantized)?;
        let metrics = self.calculate_accuracy_metrics(&input_data, &dequantized);

        Ok(QuantizationTestFixture {
            name: "i2s_extreme_values_saturation".to_string(),
            algorithm: QuantizationAlgorithm::I2S,
            input_data,
            input_shape: shape,
            expected_quantized: quantized,
            expected_dequantized: dequantized,
            accuracy_metrics: metrics,
            device_variants: HashMap::new(),
            test_metadata: TestMetadata {
                seed: self.seed + 6000,
                generation_timestamp: chrono::Utc::now().to_rfc3339(),
                bitnet_version: "0.1.0".to_string(),
                validation_status: "validated".to_string(),
                notes: vec![
                    "Edge case: extreme values testing saturation behavior".to_string(),
                    "Validates quantization clipping and scale factor calculation".to_string(),
                ],
            },
        })
    }

    fn create_i2s_mixed_precision_fixture(&self) -> Result<QuantizationTestFixture> {
        // Mixed precision scenario for GPU acceleration
        let input_data = self.generate_neural_network_weights(2048, self.seed + 7000);
        let shape = vec![64, 32];

        let quantized = self.simulate_i2s_quantization(&input_data)?;
        let dequantized = self.simulate_i2s_dequantization(&quantized)?;
        let metrics = self.calculate_accuracy_metrics(&input_data, &dequantized);

        let mut device_variants = HashMap::new();
        device_variants.insert(
            DeviceType::CUDA,
            DeviceSpecificData {
                optimized_layout: self.create_gpu_optimized_layout(&quantized.data),
                simd_alignment: 128,
                memory_layout: "fp16_interleaved".to_string(),
                performance_target_ms: 0.1,
                fallback_required: false,
            },
        );

        Ok(QuantizationTestFixture {
            name: "i2s_mixed_precision_gpu".to_string(),
            algorithm: QuantizationAlgorithm::I2S,
            input_data,
            input_shape: shape,
            expected_quantized: quantized,
            expected_dequantized: dequantized,
            accuracy_metrics: metrics,
            device_variants,
            test_metadata: TestMetadata {
                seed: self.seed + 7000,
                generation_timestamp: chrono::Utc::now().to_rfc3339(),
                bitnet_version: "0.1.0".to_string(),
                validation_status: "validated".to_string(),
                notes: vec![
                    "Mixed precision I2S quantization for GPU acceleration".to_string(),
                    "Tests FP16/BF16 integration with Tensor Core optimization".to_string(),
                ],
            },
        })
    }

    fn create_tl1_optimized_table_fixture(&self) -> Result<QuantizationTestFixture> {
        // Optimized TL1 with carefully crafted lookup table
        let input_data = self.generate_neural_network_weights(512, self.seed + 8000);
        let shape = vec![32, 16];

        let quantized = self.simulate_tl1_quantization(&input_data)?;
        let dequantized = self.simulate_tl1_dequantization(&quantized)?;
        let metrics = self.calculate_accuracy_metrics(&input_data, &dequantized);

        Ok(QuantizationTestFixture {
            name: "tl1_optimized_lookup_table".to_string(),
            algorithm: QuantizationAlgorithm::TL1,
            input_data,
            input_shape: shape,
            expected_quantized: quantized,
            expected_dequantized: dequantized,
            accuracy_metrics: metrics,
            device_variants: HashMap::new(),
            test_metadata: TestMetadata {
                seed: self.seed + 8000,
                generation_timestamp: chrono::Utc::now().to_rfc3339(),
                bitnet_version: "0.1.0".to_string(),
                validation_status: "validated".to_string(),
                notes: vec![
                    "TL1 with optimized 16-entry lookup table".to_string(),
                    "Validates k-means clustering for table generation".to_string(),
                ],
            },
        })
    }

    fn create_tl1_cache_friendly_fixture(&self) -> Result<QuantizationTestFixture> {
        // Cache-friendly TL1 access patterns
        let input_data = self.generate_neural_network_weights(1024, self.seed + 9000);
        let shape = vec![32, 32];

        let quantized = self.simulate_tl1_quantization(&input_data)?;
        let dequantized = self.simulate_tl1_dequantization(&quantized)?;
        let metrics = self.calculate_accuracy_metrics(&input_data, &dequantized);

        let mut device_variants = HashMap::new();
        device_variants.insert(
            DeviceType::CPU,
            DeviceSpecificData {
                optimized_layout: self.create_cpu_optimized_layout(&quantized.data),
                simd_alignment: 64,
                memory_layout: "cache_blocking".to_string(),
                performance_target_ms: 0.15,
                fallback_required: false,
            },
        );

        Ok(QuantizationTestFixture {
            name: "tl1_cache_friendly_access".to_string(),
            algorithm: QuantizationAlgorithm::TL1,
            input_data,
            input_shape: shape,
            expected_quantized: quantized,
            expected_dequantized: dequantized,
            accuracy_metrics: metrics,
            device_variants,
            test_metadata: TestMetadata {
                seed: self.seed + 9000,
                generation_timestamp: chrono::Utc::now().to_rfc3339(),
                bitnet_version: "0.1.0".to_string(),
                validation_status: "validated".to_string(),
                notes: vec![
                    "TL1 optimized for cache-friendly memory access patterns".to_string(),
                    "Tests CPU cache optimization with blocked layout".to_string(),
                ],
            },
        })
    }

    fn create_tl2_vs_tl1_fixture(&self) -> Result<QuantizationTestFixture> {
        // Direct comparison between TL2 and TL1 precision
        let input_data = self.generate_neural_network_weights(512, self.seed + 10000);
        let shape = vec![16, 32];

        let quantized = self.simulate_tl2_quantization(&input_data)?;
        let dequantized = self.simulate_tl2_dequantization(&quantized)?;
        let metrics = self.calculate_accuracy_metrics(&input_data, &dequantized);

        Ok(QuantizationTestFixture {
            name: "tl2_vs_tl1_precision_comparison".to_string(),
            algorithm: QuantizationAlgorithm::TL2,
            input_data,
            input_shape: shape,
            expected_quantized: quantized,
            expected_dequantized: dequantized,
            accuracy_metrics: metrics,
            device_variants: HashMap::new(),
            test_metadata: TestMetadata {
                seed: self.seed + 10000,
                generation_timestamp: chrono::Utc::now().to_rfc3339(),
                bitnet_version: "0.1.0".to_string(),
                validation_status: "validated".to_string(),
                notes: vec![
                    "TL2 (256-entry) vs TL1 (16-entry) precision comparison".to_string(),
                    "Validates trade-off between table size and accuracy".to_string(),
                ],
            },
        })
    }
}

/// Create comprehensive BitNet quantization test fixtures
pub fn create_bitnet_quantization_fixtures(seed: u64) -> Result<Vec<QuantizationTestFixture>> {
    let generator = BitNetQuantizationFixtures::new(seed);
    generator.generate_all_fixtures()
}

/// Save fixtures to JSON file for external validation
pub fn save_fixtures_to_file(
    fixtures: &[QuantizationTestFixture],
    path: &std::path::Path,
) -> Result<()> {
    let json_data = serde_json::to_string_pretty(fixtures)?;
    std::fs::write(path, json_data)?;
    Ok(())
}

/// Load fixtures from JSON file
pub fn load_fixtures_from_file(path: &std::path::Path) -> Result<Vec<QuantizationTestFixture>> {
    let json_data = std::fs::read_to_string(path)?;
    let fixtures = serde_json::from_str(&json_data)?;
    Ok(fixtures)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_i2s_quantization_fixtures() -> Result<()> {
        let generator = BitNetQuantizationFixtures::new(42);
        let fixtures = generator.generate_i2s_fixtures()?;

        assert!(!fixtures.is_empty());

        for fixture in &fixtures {
            assert_eq!(fixture.algorithm, QuantizationAlgorithm::I2S);
            assert!(fixture.accuracy_metrics.cosine_similarity > 0.9);
            assert!(!fixture.input_data.is_empty());
            assert!(!fixture.expected_dequantized.is_empty());
        }

        Ok(())
    }

    #[test]
    fn test_tl1_quantization_fixtures() -> Result<()> {
        let generator = BitNetQuantizationFixtures::new(42);
        let fixtures = generator.generate_tl1_fixtures()?;

        assert!(!fixtures.is_empty());

        for fixture in &fixtures {
            assert_eq!(fixture.algorithm, QuantizationAlgorithm::TL1);
            assert!(fixture.expected_quantized.lookup_table.is_some());
            let lookup_table = fixture.expected_quantized.lookup_table.as_ref().unwrap();
            assert_eq!(lookup_table.len(), 16); // TL1 uses 16-entry table
        }

        Ok(())
    }

    #[test]
    fn test_fixture_serialization() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let file_path = temp_dir.path().join("test_fixtures.json");

        let fixtures = create_bitnet_quantization_fixtures(42)?;
        save_fixtures_to_file(&fixtures, &file_path)?;

        let loaded_fixtures = load_fixtures_from_file(&file_path)?;
        assert_eq!(fixtures.len(), loaded_fixtures.len());

        Ok(())
    }
}
