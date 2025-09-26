//! Multi-Head Attention test fixtures for BitNet.rs neural network components
//!
//! Provides comprehensive test data for multi-head attention mechanisms including
//! query, key, value projections, attention computation, and output projections
//! with realistic transformer architecture patterns.

use super::{TestEnvironmentConfig, quantization::ToleranceConfig};
use bitnet_common::{BitNetError, Device, QuantizationType, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::LazyLock;

/// Multi-head attention configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub sequence_length: usize,
    pub use_bias: bool,
    pub dropout_prob: f32,
    pub attention_scale: f32,
}

impl AttentionConfig {
    /// Create standard BitNet attention configuration
    pub fn bitnet_standard() -> Self {
        Self {
            hidden_size: 512,
            num_attention_heads: 8,
            head_dim: 64, // hidden_size / num_attention_heads
            sequence_length: 128,
            use_bias: false,   // BitNet typically doesn't use bias
            dropout_prob: 0.0, // Disable for testing
            attention_scale: 1.0 / (64.0_f32.sqrt()), // 1/sqrt(head_dim)
        }
    }

    /// Create large model configuration
    pub fn bitnet_large() -> Self {
        Self {
            hidden_size: 2048,
            num_attention_heads: 16,
            head_dim: 128,
            sequence_length: 512,
            use_bias: false,
            dropout_prob: 0.0,
            attention_scale: 1.0 / (128.0_f32.sqrt()),
        }
    }
}

/// Multi-head attention test case
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionTestCase {
    pub test_name: String,
    pub description: String,
    pub config: AttentionConfig,
    pub input_data: AttentionInputData,
    pub weight_matrices: AttentionWeights,
    pub expected_outputs: AttentionOutputs,
    pub quantization_data: HashMap<QuantizationType, QuantizedAttentionData>,
    pub device_variants: HashMap<Device, DeviceAttentionData>,
    pub tolerance: ToleranceConfig,
}

/// Input data for attention computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionInputData {
    pub input_sequence: Vec<Vec<f32>>,         // [seq_len, hidden_size]
    pub attention_mask: Option<Vec<Vec<f32>>>, // [seq_len, seq_len] optional
    pub position_ids: Vec<usize>,              // [seq_len] position indices
    pub causal_mask: bool,                     // Whether to use causal (autoregressive) masking
}

/// Attention weight matrices
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionWeights {
    pub q_proj_weight: Vec<Vec<f32>>, // [hidden_size, hidden_size] Query projection
    pub k_proj_weight: Vec<Vec<f32>>, // [hidden_size, hidden_size] Key projection
    pub v_proj_weight: Vec<Vec<f32>>, // [hidden_size, hidden_size] Value projection
    pub o_proj_weight: Vec<Vec<f32>>, // [hidden_size, hidden_size] Output projection
    pub q_proj_bias: Option<Vec<f32>>, // [hidden_size] Optional query bias
    pub k_proj_bias: Option<Vec<f32>>, // [hidden_size] Optional key bias
    pub v_proj_bias: Option<Vec<f32>>, // [hidden_size] Optional value bias
    pub o_proj_bias: Option<Vec<f32>>, // [hidden_size] Optional output bias
}

/// Expected attention computation outputs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionOutputs {
    pub query_states: Vec<Vec<Vec<f32>>>, // [num_heads, seq_len, head_dim]
    pub key_states: Vec<Vec<Vec<f32>>>,   // [num_heads, seq_len, head_dim]
    pub value_states: Vec<Vec<Vec<f32>>>, // [num_heads, seq_len, head_dim]
    pub attention_scores: Vec<Vec<Vec<f32>>>, // [num_heads, seq_len, seq_len]
    pub attention_probs: Vec<Vec<Vec<f32>>>, // [num_heads, seq_len, seq_len] (after softmax)
    pub context_layer: Vec<Vec<Vec<f32>>>, // [num_heads, seq_len, head_dim]
    pub output_sequence: Vec<Vec<f32>>,   // [seq_len, hidden_size] Final output
}

/// Quantized attention data for different quantization types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedAttentionData {
    pub quantization_type: QuantizationType,
    pub quantized_q_proj: Vec<i8>,
    pub quantized_k_proj: Vec<i8>,
    pub quantized_v_proj: Vec<i8>,
    pub quantized_o_proj: Vec<i8>,
    pub quantization_scales: Vec<f32>,
    pub quantization_error_metrics: QuantizationErrorMetrics,
}

/// Device-specific attention optimization data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceAttentionData {
    pub device: Device,
    pub optimization_strategy: String,
    pub memory_layout: String, // "row_major", "column_major", "blocked"
    pub kernel_configuration: KernelConfig,
    pub performance_metrics: AttentionPerformanceMetrics,
}

/// GPU/CPU kernel configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelConfig {
    pub block_size: usize,
    pub thread_block_size: usize,
    pub shared_memory_bytes: usize,
    pub vectorization_width: usize,
}

/// Performance metrics for attention computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionPerformanceMetrics {
    pub flops_per_second: f64,
    pub memory_bandwidth_gb_s: f64,
    pub latency_ms: f32,
    pub memory_usage_mb: f32,
    pub cache_efficiency: f32,
}

/// Quantization error metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationErrorMetrics {
    pub mean_squared_error: f64,
    pub max_absolute_error: f64,
    pub signal_to_noise_ratio: f64,
    pub cosine_similarity: f64,
}

/// Multi-head attention fixtures collection
pub struct AttentionFixtures {
    pub test_cases: Vec<AttentionTestCase>,
    pub rope_test_data: RoPETestData,
    pub kv_cache_test_data: KVCacheTestData,
    pub config: TestEnvironmentConfig,
}

/// RoPE (Rotary Position Embedding) test data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoPETestData {
    pub sequence_length: usize,
    pub head_dim: usize,
    pub rope_base: f64,
    pub position_indices: Vec<usize>,
    pub cos_cache: Vec<Vec<f32>>, // [seq_len, head_dim/2]
    pub sin_cache: Vec<Vec<f32>>, // [seq_len, head_dim/2]
    pub expected_rotated_queries: Vec<Vec<Vec<f32>>>, // [num_heads, seq_len, head_dim]
    pub expected_rotated_keys: Vec<Vec<Vec<f32>>>, // [num_heads, seq_len, head_dim]
}

/// KV-Cache test data for autoregressive generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KVCacheTestData {
    pub max_sequence_length: usize,
    pub cache_key_states: Vec<Vec<Vec<f32>>>, // [max_seq, num_heads, head_dim]
    pub cache_value_states: Vec<Vec<Vec<f32>>>, // [max_seq, num_heads, head_dim]
    pub cache_positions: Vec<usize>,          // Current cache positions
    pub incremental_keys: Vec<Vec<Vec<f32>>>, // New keys to append
    pub incremental_values: Vec<Vec<Vec<f32>>>, // New values to append
    pub expected_updated_cache: (Vec<Vec<Vec<f32>>>, Vec<Vec<Vec<f32>>>), // (keys, values)
}

/// Static attention test cases
static ATTENTION_TEST_CASES: LazyLock<Vec<AttentionTestCase>> = LazyLock::new(|| {
    vec![
        create_basic_attention_test(),
        create_causal_attention_test(),
        create_long_sequence_test(),
        create_multi_head_scaling_test(),
        create_quantized_attention_test(),
    ]
});

impl AttentionFixtures {
    /// Create new attention fixtures
    pub fn new(config: &TestEnvironmentConfig) -> Self {
        Self {
            test_cases: ATTENTION_TEST_CASES.clone(),
            rope_test_data: create_rope_test_data(),
            kv_cache_test_data: create_kv_cache_test_data(),
            config: config.clone(),
        }
    }

    /// Initialize attention fixtures with device optimizations
    pub async fn initialize(&mut self) -> Result<()> {
        // Generate quantized variants
        self.generate_quantized_variants().await?;

        // Create device-specific optimizations
        self.generate_device_variants().await?;

        // Precompute RoPE tables
        self.precompute_rope_tables().await?;

        // Initialize KV cache test data
        self.initialize_kv_cache_data().await?;

        Ok(())
    }

    /// Generate quantized variants for all test cases
    async fn generate_quantized_variants(&mut self) -> Result<()> {
        for test_case in &mut self.test_cases {
            // Generate I2S quantized data
            let i2s_data = self
                .quantize_attention_weights(&test_case.weight_matrices, QuantizationType::I2S)
                .await?;
            test_case.quantization_data.insert(QuantizationType::I2S, i2s_data);

            // Generate TL1 quantized data
            let tl1_data = self
                .quantize_attention_weights(&test_case.weight_matrices, QuantizationType::TL1)
                .await?;
            test_case.quantization_data.insert(QuantizationType::TL1, tl1_data);

            // Generate TL2 quantized data
            let tl2_data = self
                .quantize_attention_weights(&test_case.weight_matrices, QuantizationType::TL2)
                .await?;
            test_case.quantization_data.insert(QuantizationType::TL2, tl2_data);
        }

        Ok(())
    }

    /// Generate device-specific variants
    async fn generate_device_variants(&mut self) -> Result<()> {
        for test_case in &mut self.test_cases {
            // CPU variant with SIMD optimizations
            let cpu_data = DeviceAttentionData {
                device: Device::Cpu,
                optimization_strategy: "SIMD_Optimized".to_string(),
                memory_layout: "blocked".to_string(),
                kernel_configuration: KernelConfig {
                    block_size: 64,
                    thread_block_size: 1, // Single threaded for CPU
                    shared_memory_bytes: 0,
                    vectorization_width: 8, // AVX2
                },
                performance_metrics: AttentionPerformanceMetrics {
                    flops_per_second: 50e9, // 50 GFLOPS
                    memory_bandwidth_gb_s: 100.0,
                    latency_ms: 2.0,
                    memory_usage_mb: 128.0,
                    cache_efficiency: 0.8,
                },
            };
            test_case.device_variants.insert(Device::Cpu, cpu_data);

            // GPU variant (if available)
            #[cfg(feature = "gpu")]
            {
                let gpu_data = DeviceAttentionData {
                    device: Device::Cuda(0),
                    optimization_strategy: "Flash_Attention".to_string(),
                    memory_layout: "row_major".to_string(),
                    kernel_configuration: KernelConfig {
                        block_size: 256,
                        thread_block_size: 256,
                        shared_memory_bytes: 49152, // 48KB shared memory
                        vectorization_width: 32,    // Warp size
                    },
                    performance_metrics: AttentionPerformanceMetrics {
                        flops_per_second: 500e9, // 500 GFLOPS
                        memory_bandwidth_gb_s: 900.0,
                        latency_ms: 0.2,
                        memory_usage_mb: 512.0,
                        cache_efficiency: 0.95,
                    },
                };
                test_case.device_variants.insert(Device::Cuda(0), gpu_data);
            }
        }

        Ok(())
    }

    /// Quantize attention weights for specific quantization type
    async fn quantize_attention_weights(
        &self,
        weights: &AttentionWeights,
        qtype: QuantizationType,
    ) -> Result<QuantizedAttentionData> {
        // Flatten weight matrices for quantization
        let q_flat: Vec<f32> = weights.q_proj_weight.iter().flatten().copied().collect();
        let k_flat: Vec<f32> = weights.k_proj_weight.iter().flatten().copied().collect();
        let v_flat: Vec<f32> = weights.v_proj_weight.iter().flatten().copied().collect();
        let o_flat: Vec<f32> = weights.o_proj_weight.iter().flatten().copied().collect();

        // Mock quantization (replace with actual quantization kernels)
        let (quantized_q, scale_q) = self.mock_quantize(&q_flat, qtype).await?;
        let (quantized_k, scale_k) = self.mock_quantize(&k_flat, qtype).await?;
        let (quantized_v, scale_v) = self.mock_quantize(&v_flat, qtype).await?;
        let (quantized_o, scale_o) = self.mock_quantize(&o_flat, qtype).await?;

        // Compute error metrics
        let error_metrics = self
            .compute_quantization_error_metrics(
                &q_flat,
                &self.mock_dequantize(&quantized_q, scale_q, qtype).await?,
            )
            .await?;

        Ok(QuantizedAttentionData {
            quantization_type: qtype,
            quantized_q_proj: quantized_q,
            quantized_k_proj: quantized_k,
            quantized_v_proj: quantized_v,
            quantized_o_proj: quantized_o,
            quantization_scales: vec![scale_q, scale_k, scale_v, scale_o],
            quantization_error_metrics: error_metrics,
        })
    }

    /// Mock quantization function (replace with real implementation)
    async fn mock_quantize(&self, data: &[f32], qtype: QuantizationType) -> Result<(Vec<i8>, f32)> {
        let max_val = data.iter().map(|x| x.abs()).fold(0.0, f32::max);
        let scale = match qtype {
            QuantizationType::I2S => max_val / 1.0, // ±1 range
            QuantizationType::TL1 => max_val / 2.0, // ±2 range
            QuantizationType::TL2 => max_val / 3.0, // ±3 range
            _ => max_val,
        };

        let quantized: Vec<i8> = data
            .iter()
            .map(|&x| {
                let normalized = if scale > 1e-8 { x / scale } else { 0.0 };
                normalized.round().clamp(-127.0, 127.0) as i8
            })
            .collect();

        Ok((quantized, scale))
    }

    /// Mock dequantization function
    async fn mock_dequantize(
        &self,
        data: &[i8],
        scale: f32,
        _qtype: QuantizationType,
    ) -> Result<Vec<f32>> {
        let dequantized: Vec<f32> = data.iter().map(|&x| (x as f32) * scale).collect();

        Ok(dequantized)
    }

    /// Compute quantization error metrics
    async fn compute_quantization_error_metrics(
        &self,
        original: &[f32],
        quantized: &[f32],
    ) -> Result<QuantizationErrorMetrics> {
        if original.len() != quantized.len() {
            return Err(BitNetError::Validation("Length mismatch".to_string()));
        }

        let n = original.len() as f64;

        // Mean squared error
        let mse: f64 = original
            .iter()
            .zip(quantized.iter())
            .map(|(a, b)| ((a - b) as f64).powi(2))
            .sum::<f64>()
            / n;

        // Max absolute error
        let max_abs_error = original
            .iter()
            .zip(quantized.iter())
            .map(|(a, b)| (a - b).abs() as f64)
            .fold(0.0, f64::max);

        // Signal-to-noise ratio
        let signal_power: f64 = original.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / n;
        let snr = if mse > 1e-10 { 10.0 * (signal_power / mse).log10() } else { 100.0 };

        // Cosine similarity
        let dot_product: f64 =
            original.iter().zip(quantized.iter()).map(|(a, b)| (*a as f64) * (*b as f64)).sum();
        let norm_orig: f64 = original.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
        let norm_quant: f64 = quantized.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
        let cosine_sim = if norm_orig > 1e-10 && norm_quant > 1e-10 {
            dot_product / (norm_orig * norm_quant)
        } else {
            1.0
        };

        Ok(QuantizationErrorMetrics {
            mean_squared_error: mse,
            max_absolute_error: max_abs_error,
            signal_to_noise_ratio: snr,
            cosine_similarity: cosine_sim,
        })
    }

    /// Precompute RoPE tables
    async fn precompute_rope_tables(&mut self) -> Result<()> {
        let rope_data = &mut self.rope_test_data;
        let seq_len = rope_data.sequence_length;
        let head_dim = rope_data.head_dim;
        let rope_base = rope_data.rope_base;

        // Precompute cos and sin tables
        let mut cos_cache = vec![vec![0.0; head_dim / 2]; seq_len];
        let mut sin_cache = vec![vec![0.0; head_dim / 2]; seq_len];

        for pos in 0..seq_len {
            for i in 0..(head_dim / 2) {
                let inv_freq = 1.0 / rope_base.powf((2.0 * i as f64) / head_dim as f64);
                let angle = pos as f64 * inv_freq;
                cos_cache[pos][i] = angle.cos() as f32;
                sin_cache[pos][i] = angle.sin() as f32;
            }
        }

        rope_data.cos_cache = cos_cache;
        rope_data.sin_cache = sin_cache;

        Ok(())
    }

    /// Initialize KV cache test data
    async fn initialize_kv_cache_data(&mut self) -> Result<()> {
        // This would initialize realistic KV cache test patterns
        // Implementation details would depend on the specific caching strategy

        Ok(())
    }

    /// Get test case by name
    pub fn get_test_case(&self, name: &str) -> Option<&AttentionTestCase> {
        self.test_cases.iter().find(|case| case.test_name == name)
    }

    /// Get device-specific data for test case
    pub fn get_device_data(
        &self,
        test_name: &str,
        device: &Device,
    ) -> Option<&DeviceAttentionData> {
        self.get_test_case(test_name)?.device_variants.get(device)
    }
}

/// Create basic attention test case
fn create_basic_attention_test() -> AttentionTestCase {
    let config = AttentionConfig::bitnet_standard();
    let seq_len = config.sequence_length;
    let hidden_size = config.hidden_size;

    // Create input sequence (batch size = 1)
    let input_sequence: Vec<Vec<f32>> = (0..seq_len)
        .map(|i| {
            (0..hidden_size)
                .map(|j| ((i * hidden_size + j) as f32) / (seq_len * hidden_size) as f32)
                .collect()
        })
        .collect();

    // Create identity-like weight matrices for predictable behavior
    let q_proj_weight = create_identity_matrix(hidden_size, hidden_size);
    let k_proj_weight = create_identity_matrix(hidden_size, hidden_size);
    let v_proj_weight = create_identity_matrix(hidden_size, hidden_size);
    let o_proj_weight = create_identity_matrix(hidden_size, hidden_size);

    AttentionTestCase {
        test_name: "basic_attention".to_string(),
        description: "Basic multi-head attention computation".to_string(),
        config,
        input_data: AttentionInputData {
            input_sequence,
            attention_mask: None,
            position_ids: (0..seq_len).collect(),
            causal_mask: false,
        },
        weight_matrices: AttentionWeights {
            q_proj_weight,
            k_proj_weight,
            v_proj_weight,
            o_proj_weight,
            q_proj_bias: None,
            k_proj_bias: None,
            v_proj_bias: None,
            o_proj_bias: None,
        },
        expected_outputs: AttentionOutputs {
            query_states: vec![], // Will be computed during initialization
            key_states: vec![],
            value_states: vec![],
            attention_scores: vec![],
            attention_probs: vec![],
            context_layer: vec![],
            output_sequence: vec![],
        },
        quantization_data: HashMap::new(),
        device_variants: HashMap::new(),
        tolerance: ToleranceConfig {
            quantization_tolerance: 0.1,
            dequantization_tolerance: 0.1,
            scale_tolerance: 0.01,
            numerical_accuracy_threshold: 0.95,
        },
    }
}

/// Create causal attention test case
fn create_causal_attention_test() -> AttentionTestCase {
    let mut test_case = create_basic_attention_test();
    test_case.test_name = "causal_attention".to_string();
    test_case.description =
        "Causal (masked) multi-head attention for autoregressive generation".to_string();
    test_case.input_data.causal_mask = true;

    // Create lower triangular attention mask
    let seq_len = test_case.config.sequence_length;
    let mut attention_mask = vec![vec![f32::NEG_INFINITY; seq_len]; seq_len];
    for i in 0..seq_len {
        for j in 0..=i {
            attention_mask[i][j] = 0.0;
        }
    }
    test_case.input_data.attention_mask = Some(attention_mask);

    test_case
}

/// Create long sequence test case
fn create_long_sequence_test() -> AttentionTestCase {
    let mut config = AttentionConfig::bitnet_large();
    config.sequence_length = 1024; // Long sequence

    let mut test_case = create_basic_attention_test();
    test_case.test_name = "long_sequence_attention".to_string();
    test_case.description = "Long sequence multi-head attention (1024 tokens)".to_string();
    test_case.config = config.clone();

    // Update input data for longer sequence
    let seq_len = config.sequence_length;
    let hidden_size = config.hidden_size;
    test_case.input_data.input_sequence = (0..seq_len)
        .map(|i| {
            (0..hidden_size)
                .map(|j| ((i * hidden_size + j) as f32) / (seq_len * hidden_size) as f32)
                .collect()
        })
        .collect();
    test_case.input_data.position_ids = (0..seq_len).collect();

    test_case
}

/// Create multi-head scaling test case
fn create_multi_head_scaling_test() -> AttentionTestCase {
    let mut config = AttentionConfig::bitnet_standard();
    config.num_attention_heads = 16; // More heads
    config.head_dim = config.hidden_size / config.num_attention_heads;
    config.attention_scale = 1.0 / (config.head_dim as f32).sqrt();

    let mut test_case = create_basic_attention_test();
    test_case.test_name = "multi_head_scaling".to_string();
    test_case.description = "Multi-head attention with 16 heads and proper scaling".to_string();
    test_case.config = config;

    test_case
}

/// Create quantized attention test case
fn create_quantized_attention_test() -> AttentionTestCase {
    let mut test_case = create_basic_attention_test();
    test_case.test_name = "quantized_attention".to_string();
    test_case.description = "Multi-head attention with I2S quantized weights".to_string();

    // Relaxed tolerance for quantized computation
    test_case.tolerance = ToleranceConfig {
        quantization_tolerance: 0.3,
        dequantization_tolerance: 0.3,
        scale_tolerance: 0.1,
        numerical_accuracy_threshold: 0.8,
    };

    test_case
}

/// Create identity matrix for weight initialization
fn create_identity_matrix(rows: usize, cols: usize) -> Vec<Vec<f32>> {
    let mut matrix = vec![vec![0.0; cols]; rows];
    let min_dim = rows.min(cols);
    for i in 0..min_dim {
        matrix[i][i] = 1.0;
    }
    matrix
}

/// Create RoPE test data
fn create_rope_test_data() -> RoPETestData {
    RoPETestData {
        sequence_length: 128,
        head_dim: 64,
        rope_base: 10000.0,
        position_indices: (0..128).collect(),
        cos_cache: vec![], // Will be computed during initialization
        sin_cache: vec![], // Will be computed during initialization
        expected_rotated_queries: vec![],
        expected_rotated_keys: vec![],
    }
}

/// Create KV cache test data
fn create_kv_cache_test_data() -> KVCacheTestData {
    KVCacheTestData {
        max_sequence_length: 2048,
        cache_key_states: vec![],
        cache_value_states: vec![],
        cache_positions: vec![],
        incremental_keys: vec![],
        incremental_values: vec![],
        expected_updated_cache: (vec![], vec![]),
    }
}

/// Create attention fixtures for testing
#[cfg(test)]
pub fn create_attention_fixtures() -> AttentionFixtures {
    let config = TestEnvironmentConfig::from_env();
    AttentionFixtures::new(&config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_config_creation() {
        let config = AttentionConfig::bitnet_standard();
        assert_eq!(config.hidden_size, 512);
        assert_eq!(config.num_attention_heads, 8);
        assert_eq!(config.head_dim, 64);
    }

    #[test]
    fn test_identity_matrix_creation() {
        let matrix = create_identity_matrix(3, 3);
        assert_eq!(matrix[0][0], 1.0);
        assert_eq!(matrix[1][1], 1.0);
        assert_eq!(matrix[2][2], 1.0);
        assert_eq!(matrix[0][1], 0.0);
    }

    #[tokio::test]
    async fn test_attention_fixtures_initialization() {
        let mut fixtures = create_attention_fixtures();
        fixtures.initialize().await.expect("Initialization failed");

        assert!(!fixtures.test_cases.is_empty());

        // Check that quantization data was generated
        let basic_test = fixtures.get_test_case("basic_attention").unwrap();
        assert!(basic_test.quantization_data.contains_key(&QuantizationType::I2S));
    }
}
