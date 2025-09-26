//! Quantized Linear Layer Implementation
//!
//! This module provides the core quantized linear layer implementation for BitNet
//! neural networks, supporting I2S, TL1, and TL2 quantization formats with
//! device-aware kernel selection for optimal performance.

use anyhow::{Context, Result};
use bitnet_common::{BitNetTensor, Device, QuantizationType, Tensor};
use bitnet_kernels::{DeviceAwareQuantizer, KernelManager};
pub use bitnet_quantization::QuantizedTensor as QuantizedTensorType;
use bitnet_quantization::{Quantize, QuantizedTensor};
#[cfg(feature = "gpu")]
use candle_core::DType;
use std::sync::Arc;

/// Comprehensive error types for quantized linear layer operations
#[derive(Debug, thiserror::Error)]
pub enum QuantizedLinearError {
    #[error("Incompatible tensor shapes: input {input:?}, weight {weight:?}")]
    ShapeMismatch { input: Vec<usize>, weight: Vec<usize> },

    #[error("Unsupported quantization type: {qtype:?} on device {device}")]
    UnsupportedQuantization { qtype: QuantizationType, device: String },

    #[error("Kernel operation failed: {kernel} - {reason}")]
    KernelError { kernel: String, reason: String },

    #[error("Memory allocation failed: requested {size} bytes")]
    MemoryError { size: usize },

    #[error("Quantization accuracy too low: {accuracy:.4} < {threshold:.4}")]
    AccuracyError { accuracy: f32, threshold: f32 },

    #[error("Device mismatch: tensor on {tensor_device}, layer on {layer_device}")]
    DeviceMismatch { tensor_device: String, layer_device: String },
}

/// Performance metrics for quantized linear operations
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub average_lookup_cycles: f32,
    pub memory_efficiency: f32,
    pub throughput_gops: f32,
}

/// Accuracy validation report for quantization
#[derive(Debug, Clone)]
pub struct AccuracyReport {
    pub mean_correlation: f32,
    pub min_correlation: f32,
    pub mean_mse: f32,
    pub max_mse: f32,
    pub samples_tested: usize,
}

impl AccuracyReport {
    /// Check if accuracy meets BitNet requirements (>99% correlation)
    pub fn meets_requirements(&self) -> bool {
        self.mean_correlation > 0.99 && self.min_correlation > 0.95 && self.mean_mse < 1e-6
    }
}

/// Cross-validation report for C++ reference comparison
#[derive(Debug)]
pub struct CrossValidationReport {
    pub layer_name: String,
    pub total_tests: usize,
    pub passed_tests: usize,
    pub overall_correlation: f32,
    pub results: Vec<LayerValidationResult>,
}

#[derive(Debug)]
pub struct LayerValidationResult {
    pub test_case: usize,
    pub correlation: f32,
    pub mse: f32,
    pub passed: bool,
}

/// Consistency validation result for cross-device operations
#[derive(Debug)]
pub struct ConsistencyResult {
    pub max_difference: f32,
    pub max_variance: f32,
}

/// Tensor statistics for quantization optimization
#[derive(Debug)]
pub struct TensorStatistics {
    pub mean: f32,
    pub variance: f32,
    pub min: f32,
    pub max: f32,
    pub std_dev: f32,
}

/// Lookup table for TL1/TL2 quantization
#[derive(Debug, Clone)]
pub struct LookupTable {
    entries: Vec<f32>,
    size: usize,
}

impl LookupTable {
    pub fn new(entries: Vec<f32>) -> Self {
        let size = entries.len();
        Self { entries, size }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn entries(&self) -> &[f32] {
        &self.entries
    }

    pub fn cache_efficiency(&self) -> f32 {
        // Simple heuristic: smaller tables have better cache efficiency
        if self.size <= 16 {
            0.98
        } else if self.size <= 256 {
            0.95
        } else {
            0.90
        }
    }

    pub fn memory_footprint(&self) -> usize {
        self.size * std::mem::size_of::<f32>()
    }
}

/// High-performance quantized linear layer with device-aware optimization
pub struct QuantizedLinear {
    /// Quantized weight storage
    weights: QuantizedTensor,
    /// Optional bias (kept in FP32)
    bias: Option<BitNetTensor>,

    /// Quantization infrastructure
    #[allow(dead_code)]
    quantizer: Arc<DeviceAwareQuantizer>,
    qtype: QuantizationType,

    /// Layer metadata
    in_features: usize,
    out_features: usize,
    device: Device,

    /// Performance optimization
    workspace: Option<BitNetTensor>,
    #[allow(dead_code)]
    scale_cache: Option<BitNetTensor>,
    kernel_manager: Arc<KernelManager>,
}

impl QuantizedLinear {
    /// Create new quantized linear layer with I2S quantization
    pub fn new_i2s(weights: QuantizedTensor, device: Device) -> Result<Self> {
        let in_features = weights.shape[0];
        let out_features = weights.shape[1];

        let mut layer = Self {
            weights,
            bias: None,
            quantizer: Arc::new(DeviceAwareQuantizer::new(device)?),
            qtype: QuantizationType::I2S,
            in_features,
            out_features,
            device,
            workspace: None,
            scale_cache: None,
            kernel_manager: Arc::new(KernelManager::new()),
        };

        layer.optimize_memory_layout()?;
        Ok(layer)
    }

    /// Create new quantized linear layer with TL1 quantization
    pub fn new_tl1(
        weights: QuantizedTensor,
        _lookup_table: LookupTable,
        device: Device,
    ) -> Result<Self> {
        let in_features = weights.shape[0];
        let out_features = weights.shape[1];

        let mut layer = Self {
            weights,
            bias: None,
            quantizer: Arc::new(DeviceAwareQuantizer::new(device)?),
            qtype: QuantizationType::TL1,
            in_features,
            out_features,
            device,
            workspace: None,
            scale_cache: None,
            kernel_manager: Arc::new(KernelManager::new()),
        };

        layer.optimize_memory_layout()?;
        Ok(layer)
    }

    /// Create new quantized linear layer with TL2 quantization
    pub fn new_tl2(
        weights: QuantizedTensor,
        _lookup_table: LookupTable,
        device: Device,
    ) -> Result<Self> {
        let in_features = weights.shape[0];
        let out_features = weights.shape[1];

        let mut layer = Self {
            weights,
            bias: None,
            quantizer: Arc::new(DeviceAwareQuantizer::new(device)?),
            qtype: QuantizationType::TL2,
            in_features,
            out_features,
            device,
            workspace: None,
            scale_cache: None,
            kernel_manager: Arc::new(KernelManager::new()),
        };

        layer.optimize_memory_layout()?;
        Ok(layer)
    }

    /// Forward pass with quantized matrix multiplication
    /// Input: [batch_size, seq_len, in_features]
    /// Output: [batch_size, seq_len, out_features]
    pub async fn forward(&self, input: &BitNetTensor) -> Result<BitNetTensor> {
        self.validate_input(input)?;

        let input_shape = input.shape();
        if input_shape.len() < 2 {
            return Err(QuantizedLinearError::ShapeMismatch {
                input: input_shape.to_vec(),
                weight: vec![self.in_features, self.out_features],
            }
            .into());
        }

        // Handle both 2D and 3D inputs
        let (_batch_size, _seq_len, features) = if input_shape.len() == 3 {
            (input_shape[0], input_shape[1], input_shape[2])
        } else {
            (1, input_shape[0], input_shape[1])
        };

        if features != self.in_features {
            return Err(QuantizedLinearError::ShapeMismatch {
                input: input_shape.to_vec(),
                weight: vec![self.in_features, self.out_features],
            }
            .into());
        }

        // Perform quantized matrix multiplication based on type
        let output = match self.qtype {
            QuantizationType::I2S => self.forward_i2s(input).await?,
            QuantizationType::TL1 => self.forward_tl1(input).await?,
            QuantizationType::TL2 => self.forward_tl2(input).await?,
        };

        // Add bias if present
        let final_output = if let Some(ref bias) = self.bias {
            let bias_candle = bias.to_candle()?;
            let output_candle = output.to_candle()?;
            let biased = output_candle
                .broadcast_add(&bias_candle)
                .context("Failed to add bias to output")?;
            BitNetTensor::new(biased)
        } else {
            output
        };

        Ok(final_output)
    }

    /// I2S-specific forward pass implementation
    async fn forward_i2s(&self, input: &BitNetTensor) -> Result<BitNetTensor> {
        let _kernel = self.kernel_manager.select_best().context("Failed to select I2S kernel")?;

        // Convert input to candle tensor for processing
        let input_candle = input.to_candle()?;
        let input_shape = input_candle.shape();

        // Flatten input for matrix multiplication
        let input_2d = if input_shape.dims().len() > 2 {
            let total_batch =
                input_shape.dims()[..input_shape.dims().len() - 1].iter().product::<usize>();
            input_candle
                .reshape(&[total_batch, self.in_features])
                .context("Failed to reshape input for I2S")?
        } else {
            input_candle.clone()
        };

        // For now, we'll create a simplified I2S matrix multiplication
        // This should be replaced with actual quantized kernels
        let dequantized_weights =
            self.weights.dequantize().context("Failed to dequantize I2S weights")?;
        let weight_candle = dequantized_weights.to_candle()?;
        let weight_transposed = weight_candle.t().context("Failed to transpose weights")?;

        let output_2d = input_2d
            .matmul(&weight_transposed)
            .context("Failed to perform I2S matrix multiplication")?;

        // Reshape back to original batch dimensions
        let output = if input_shape.dims().len() > 2 {
            let mut output_shape = input_shape.dims().to_vec();
            let last_idx = output_shape.len() - 1;
            output_shape[last_idx] = self.out_features;
            output_2d.reshape(output_shape).context("Failed to reshape I2S output")?
        } else {
            output_2d
        };

        Ok(BitNetTensor::new(output))
    }

    /// TL1 forward pass optimized for ARM NEON
    async fn forward_tl1(&self, input: &BitNetTensor) -> Result<BitNetTensor> {
        // For now, fallback to generic implementation
        self.forward_generic(input).await
    }

    /// TL2 forward pass optimized for x86 AVX2/AVX-512
    async fn forward_tl2(&self, input: &BitNetTensor) -> Result<BitNetTensor> {
        // For now, fallback to generic implementation
        self.forward_generic(input).await
    }

    /// Generic fallback implementation
    async fn forward_generic(&self, input: &BitNetTensor) -> Result<BitNetTensor> {
        // Simple implementation using dequantized weights
        let dequantized_weights =
            self.weights.dequantize().context("Failed to dequantize weights")?;
        let weight_candle = dequantized_weights.to_candle()?;
        let input_candle = input.to_candle()?;

        let weight_transposed = weight_candle.t().context("Failed to transpose weights")?;
        let output = input_candle
            .matmul(&weight_transposed)
            .context("Failed to perform generic matrix multiplication")?;

        Ok(BitNetTensor::new(output))
    }

    /// Optimize memory layout for cache efficiency
    pub fn optimize_memory_layout(&mut self) -> Result<()> {
        match self.qtype {
            QuantizationType::I2S => self.optimize_i2s_layout()?,
            QuantizationType::TL1 => self.optimize_tl1_layout()?,
            QuantizationType::TL2 => self.optimize_tl2_layout()?,
        }

        // Pre-allocate workspace for GPU operations
        if matches!(self.device, Device::Cuda(_)) {
            self.allocate_gpu_workspace()?;
        }

        Ok(())
    }

    /// I2S memory layout: 82-byte blocks with SIMD alignment
    fn optimize_i2s_layout(&mut self) -> Result<()> {
        // I2S optimization is already handled by the quantization crate
        // This is a placeholder for future optimizations
        Ok(())
    }

    /// Optimize TL1 layout for ARM NEON
    fn optimize_tl1_layout(&mut self) -> Result<()> {
        // TL1 optimization placeholder
        Ok(())
    }

    /// Optimize TL2 layout for x86 AVX
    fn optimize_tl2_layout(&mut self) -> Result<()> {
        // TL2 optimization placeholder
        Ok(())
    }

    /// Pre-allocate GPU workspace to avoid runtime allocation
    #[cfg(feature = "gpu")]
    fn allocate_gpu_workspace(&mut self) -> Result<()> {
        let workspace_size = self.calculate_gpu_workspace_size()?;
        self.workspace = Some(BitNetTensor::zeros(&[workspace_size], DType::F32, &self.device)?);

        log::debug!("Allocated GPU workspace: {} MB", workspace_size * 4 / 1024 / 1024);
        Ok(())
    }

    #[cfg(not(feature = "gpu"))]
    fn allocate_gpu_workspace(&mut self) -> Result<()> {
        // No-op for non-GPU builds
        Ok(())
    }

    #[allow(dead_code)]
    fn calculate_gpu_workspace_size(&self) -> Result<usize> {
        // GPU kernels need temporary storage for:
        // - Dequantized weights (if needed): in_features * out_features * sizeof(f16)
        // - Intermediate results: max_batch_size * out_features * sizeof(f32)
        let max_batch_size = 64; // Conservative estimate
        let dequant_size = self.in_features * self.out_features * 2; // FP16
        let intermediate_size = max_batch_size * self.out_features * 4; // FP32

        Ok(dequant_size + intermediate_size)
    }

    /// Validate input tensor before forward pass
    fn validate_input(&self, input: &BitNetTensor) -> Result<(), QuantizedLinearError> {
        // Check device compatibility
        if !self.is_device_compatible(input) {
            return Err(QuantizedLinearError::DeviceMismatch {
                tensor_device: format!("{:?}", input.device()),
                layer_device: format!("{:?}", self.device),
            });
        }

        // Check shape compatibility
        let input_dims = input.shape();
        if input_dims.len() < 2 || input_dims[input_dims.len() - 1] != self.in_features {
            return Err(QuantizedLinearError::ShapeMismatch {
                input: input_dims.to_vec(),
                weight: vec![self.in_features, self.out_features],
            });
        }

        Ok(())
    }

    /// Check if tensor device is compatible with layer device
    fn is_device_compatible(&self, _tensor: &BitNetTensor) -> bool {
        // For now, allow any device combination
        // In a full implementation, this would enforce strict device matching
        true
    }

    /// Get memory usage in bytes (for optimization)
    pub fn memory_usage(&self) -> usize {
        let weight_bytes = self.weights.data.len();
        let scale_bytes = self.weights.scales.len() * std::mem::size_of::<f32>();
        let bias_bytes = self
            .bias
            .as_ref()
            .map(|b| b.shape().iter().product::<usize>() * std::mem::size_of::<f32>())
            .unwrap_or(0);
        let workspace_bytes = self
            .workspace
            .as_ref()
            .map(|w| w.shape().iter().product::<usize>() * std::mem::size_of::<f32>())
            .unwrap_or(0);

        weight_bytes + scale_bytes + bias_bytes + workspace_bytes
    }

    /// Get performance metrics for the layer
    pub fn get_performance_metrics(&self) -> PerformanceMetrics {
        PerformanceMetrics {
            average_lookup_cycles: match self.qtype {
                QuantizationType::I2S => 1.5, // Optimistic for I2S
                QuantizationType::TL1 => 2.0, // Table lookup overhead
                QuantizationType::TL2 => 3.0, // Larger table lookup overhead
            },
            memory_efficiency: self.weights.compression_ratio(),
            throughput_gops: 50.0, // Placeholder - should be measured
        }
    }

    /// Validate quantization accuracy against FP32 reference
    pub fn validate_accuracy(&self, fp32_weights: &BitNetTensor, tolerance: f32) -> Result<f32> {
        let dequantized = self
            .weights
            .dequantize()
            .context("Failed to dequantize weights for accuracy validation")?;

        // Compute correlation between original and quantized weights
        let original_vec =
            fp32_weights.to_vec().context("Failed to convert FP32 weights to vector")?;
        let dequantized_vec =
            dequantized.to_vec().context("Failed to convert dequantized weights to vector")?;

        if original_vec.len() != dequantized_vec.len() {
            return Err(QuantizedLinearError::ShapeMismatch {
                input: vec![original_vec.len()],
                weight: vec![dequantized_vec.len()],
            }
            .into());
        }

        let correlation = self.compute_correlation_vectors(&original_vec, &dequantized_vec)?;

        if correlation < tolerance {
            return Err(QuantizedLinearError::AccuracyError {
                accuracy: correlation,
                threshold: tolerance,
            }
            .into());
        }

        Ok(correlation)
    }

    /// Compute Pearson correlation coefficient between two vectors
    fn compute_correlation_vectors(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(QuantizedLinearError::ShapeMismatch {
                input: vec![a.len()],
                weight: vec![b.len()],
            }
            .into());
        }

        let n = a.len() as f32;
        let mean_a = a.iter().sum::<f32>() / n;
        let mean_b = b.iter().sum::<f32>() / n;

        let mut num = 0.0;
        let mut den_a = 0.0;
        let mut den_b = 0.0;

        for (&a_val, &b_val) in a.iter().zip(b.iter()) {
            let da = a_val - mean_a;
            let db = b_val - mean_b;
            num += da * db;
            den_a += da * da;
            den_b += db * db;
        }

        let denominator = (den_a * den_b).sqrt();
        if denominator == 0.0 {
            return Ok(0.0); // Handle degenerate case
        }

        Ok(num / denominator)
    }
}

// Helper functions for test scaffolding - simplified implementations

/// Create mock input tensor with specified dimensions
pub fn create_mock_tensor(
    batch_size: usize,
    seq_len: usize,
    hidden_size: usize,
) -> Result<BitNetTensor> {
    let shape = vec![batch_size, seq_len, hidden_size];
    let total_elements = shape.iter().product::<usize>();
    let data: Vec<f32> = (0..total_elements)
        .map(|i| (i as f32 % 1000.0) / 1000.0) // Simple pattern
        .collect();

    Ok(BitNetTensor::from_slice(&data, &shape, &Device::Cpu)?)
}

/// Create mock weight matrix for linear layer
pub fn create_mock_weight_matrix(input_size: usize, output_size: usize) -> Result<Vec<f32>> {
    let total_elements = input_size * output_size;
    let weights: Vec<f32> = (0..total_elements)
        .map(|i| ((i as f32).sin() * 0.1)) // Simple sinusoidal pattern
        .collect();
    Ok(weights)
}

/// Validate tensor contains no NaN or infinite values
pub fn validate_tensor_stability(tensor: &BitNetTensor) -> Result<()> {
    let data = tensor.to_vec().context("Failed to convert tensor to vector for stability check")?;

    if data.iter().any(|&x| !x.is_finite()) {
        return Err(QuantizedLinearError::KernelError {
            kernel: "stability_check".to_string(),
            reason: "Tensor contains NaN or Inf values".to_string(),
        }
        .into());
    }

    Ok(())
}

/// Check if GPU acceleration is available
pub fn is_gpu_available() -> bool {
    #[cfg(feature = "gpu")]
    {
        // Check for CUDA availability
        candle_core::Device::new_cuda(0).is_ok()
    }
    #[cfg(not(feature = "gpu"))]
    {
        false
    }
}

/// Check if FFI bridge is available
pub fn is_ffi_available() -> bool {
    // FFI feature not available in bitnet-inference crate
    false
}

/// Calculate tensor statistics for quantization
pub fn calculate_tensor_statistics(data: &[f32]) -> Result<TensorStatistics> {
    if data.is_empty() {
        return Ok(TensorStatistics { mean: 0.0, variance: 0.0, min: 0.0, max: 0.0, std_dev: 0.0 });
    }

    let mean = data.iter().sum::<f32>() / data.len() as f32;
    let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
    let std_dev = variance.sqrt();
    let min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    Ok(TensorStatistics { mean, variance, min, max, std_dev })
}

/// Validate quantization consistency between devices
pub fn validate_device_consistency(
    a: &QuantizedTensor,
    b: &QuantizedTensor,
    tolerance: f32,
) -> Result<ConsistencyResult> {
    if a.data.len() != b.data.len() {
        return Err(QuantizedLinearError::ShapeMismatch {
            input: vec![a.data.len()],
            weight: vec![b.data.len()],
        }
        .into());
    }

    let max_difference = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(&x, &y)| (x as f32 - y as f32).abs())
        .fold(0.0, f32::max);

    // Convert scales to check variance
    let scale_variance = if a.scales.len() == b.scales.len() {
        a.scales.iter().zip(b.scales.iter()).map(|(&x, &y)| (x - y).abs()).fold(0.0, f32::max)
    } else {
        tolerance + 1.0 // Force failure if scales don't match
    };

    Ok(ConsistencyResult { max_difference, max_variance: scale_variance })
}

/// Validate tensor consistency across multiple implementations
pub fn validate_tensor_consistency(
    tensors: &[&BitNetTensor],
    _tolerance: f32,
) -> Result<ConsistencyResult> {
    if tensors.len() < 2 {
        return Ok(ConsistencyResult { max_difference: 0.0, max_variance: 0.0 });
    }

    let mut max_difference: f32 = 0.0;
    let mut max_variance: f32 = 0.0;

    // Convert all tensors to vectors for comparison
    let tensor_vecs: Result<Vec<Vec<f32>>> = tensors
        .iter()
        .map(|t| t.to_vec().map_err(|e| anyhow::anyhow!("Failed to convert tensor: {}", e)))
        .collect();
    let tensor_vecs = tensor_vecs.context("Failed to convert tensors to vectors")?;

    // Compare each pair of tensors
    for i in 0..tensor_vecs.len() {
        for j in i + 1..tensor_vecs.len() {
            let a = &tensor_vecs[i];
            let b = &tensor_vecs[j];

            if a.len() != b.len() {
                return Err(QuantizedLinearError::ShapeMismatch {
                    input: vec![a.len()],
                    weight: vec![b.len()],
                }
                .into());
            }

            let pair_max_diff =
                a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).abs()).fold(0.0, f32::max);

            max_difference = max_difference.max(pair_max_diff);
        }
    }

    // Calculate variance across all tensors at each position
    if !tensor_vecs[0].is_empty() {
        for pos in 0..tensor_vecs[0].len() {
            let values: Vec<f32> = tensor_vecs.iter().map(|v| v[pos]).collect();
            let mean = values.iter().sum::<f32>() / values.len() as f32;
            let variance =
                values.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
            max_variance = max_variance.max(variance);
        }
    }

    Ok(ConsistencyResult { max_difference, max_variance })
}
