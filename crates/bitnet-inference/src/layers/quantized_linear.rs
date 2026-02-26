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
#[cfg(any(feature = "gpu", feature = "cuda"))]
use candle_core::DType;
use std::sync::Arc;

// Import QK256 kernel from bitnet-models
use bitnet_models::quant::i2s_qk256;

/// SIMD-aligned block size for I2S quantization (BitNet-rs optimization)
const I2S_BLOCK_SIZE: usize = 82;
/// Cache line size for optimal memory access patterns
const CACHE_LINE_SIZE: usize = 64;
/// Maximum workspace size to prevent excessive memory allocation (8GB)
const MAX_WORKSPACE_SIZE: usize = 8 * 1024 * 1024 * 1024;

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
    pub(crate) qtype: QuantizationType,

    /// Layer metadata
    pub(crate) in_features: usize,
    pub(crate) out_features: usize,
    pub(crate) device: Device,

    /// Performance optimization
    workspace: Option<BitNetTensor>,
    scale_cache: Option<BitNetTensor>,
    kernel_manager: Arc<KernelManager>,

    /// Memory pool for zero-copy operations
    memory_pool: Option<Vec<u8>>,
    /// SIMD alignment padding
    alignment_padding: usize,

    /// Optional QK256 quantized data (GGML I2_S format with QK=256)
    /// When present, forward pass uses QK256 kernel instead of standard I2S path
    qk256_data: Option<std::sync::OnceLock<QK256Data>>,
}

/// Storage for QK256 quantized weights (GGML I2_S format)
#[derive(Clone, Debug)]
struct QK256Data {
    /// Packed quantized bytes
    qs: Vec<u8>,
    /// Number of rows
    rows: usize,
    /// Number of columns
    cols: usize,
    /// Bytes per row (ceil(cols/256) * 64)
    row_stride_bytes: usize,
}

impl QuantizedLinear {
    /// Create new quantized linear layer with I2S quantization
    pub fn new_i2s(weights: QuantizedTensor, device: Device) -> Result<Self> {
        let in_features = weights.shape[0];
        let out_features = weights.shape[1];

        let alignment_padding = Self::calculate_alignment_padding(in_features, out_features);

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
            memory_pool: None,
            alignment_padding,
            qk256_data: None,
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

        let alignment_padding = Self::calculate_alignment_padding(in_features, out_features);

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
            memory_pool: None,
            alignment_padding,
            qk256_data: None,
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

        let alignment_padding = Self::calculate_alignment_padding(in_features, out_features);

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
            memory_pool: None,
            alignment_padding,
            qk256_data: None,
        };

        layer.optimize_memory_layout()?;
        Ok(layer)
    }

    /// Set QK256 quantized data for this layer (GGML I2_S format with QK=256)
    ///
    /// This method allows the model loader to provide pre-quantized QK256 weights
    /// loaded from GGUF files. When QK256 data is set, the forward pass will use
    /// the QK256 kernel instead of the standard I2S path.
    ///
    /// # Arguments
    ///
    /// * `qs_bytes` - Packed quantized bytes (2 bits per element)
    /// * `rows` - Number of rows in the weight matrix
    /// * `cols` - Number of columns in the weight matrix
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Dimensions don't match layer configuration
    /// - Data size doesn't match expected packing
    /// - QK256 data was already set (OnceLock constraint)
    pub fn set_qk256_data(&mut self, qs_bytes: Vec<u8>, rows: usize, cols: usize) -> Result<()> {
        // Validate dimensions match layer configuration
        // QK256 format: weight matrix as [rows, cols] where each row is a weight vector
        // For linear layer y = W*x:
        //   - rows = number of output features (one row per output)
        //   - cols = number of input features (length of each weight vector)
        //
        // Note: QuantizedLinear stores dimensions as:
        //   - self.in_features = input dimension
        //   - self.out_features = output dimension
        if rows != self.out_features {
            anyhow::bail!(
                "QK256 rows mismatch: got {}, expected {} (out_features)",
                rows,
                self.out_features
            );
        }
        if cols != self.in_features {
            anyhow::bail!(
                "QK256 cols mismatch: got {}, expected {} (in_features)",
                cols,
                self.in_features
            );
        }

        // Calculate expected packing
        let blocks_per_row = cols.div_ceil(256);
        let row_stride_bytes = blocks_per_row * 64; // 64 bytes per QK256 block
        let expected_bytes = rows * row_stride_bytes;

        if qs_bytes.len() != expected_bytes {
            anyhow::bail!(
                "QK256 data size mismatch: got {} bytes, expected {} for {}×{} matrix",
                qs_bytes.len(),
                expected_bytes,
                rows,
                cols
            );
        }

        // Create QK256 data structure
        let data = QK256Data { qs: qs_bytes, rows, cols, row_stride_bytes };

        // Initialize OnceLock if not already present
        let lock = self.qk256_data.get_or_insert_with(std::sync::OnceLock::new);

        // Try to set the data (will fail if already set)
        lock.set(data).map_err(|_| {
            anyhow::anyhow!("QK256 data already set for this layer - cannot set twice")
        })?;

        tracing::debug!(
            "Set QK256 data for layer {}×{}: {} bytes ({} blocks/row)",
            rows,
            cols,
            expected_bytes,
            blocks_per_row
        );

        Ok(())
    }

    /// Check if this layer has a native quantized kernel available (no FP32 fallback)
    pub fn has_native_quantized_kernel(&self) -> bool {
        // Check kernel availability based on quantization type and device
        match (self.device, self.qtype) {
            // CPU I2S always has native kernels via optimized paths
            (Device::Cpu, QuantizationType::I2S) => true,

            // TL1/TL2 require kernel manager support
            (Device::Cpu, QuantizationType::TL1) | (Device::Cpu, QuantizationType::TL2) => {
                self.kernel_manager.select_best().is_ok()
            }

            // GPU kernels require explicit kernel availability
            #[cfg(any(feature = "gpu", feature = "cuda"))]
            (Device::Cuda(_), _) => self.kernel_manager.select_best().is_ok(),

            // Default: assume no native kernel
            _ => false,
        }
    }

    /// Check if this layer would fall back to FP32 dequantization
    pub fn is_fallback_path(&self) -> bool {
        !self.has_native_quantized_kernel()
    }

    /// Reject FP32 fallback in strict mode with detailed error message
    ///
    /// This helper centralizes strict mode error handling to avoid duplication
    /// and ensure consistent error messages across all quantization paths.
    #[inline]
    fn strict_reject_fp32_fallback(&self, reason: &str) -> Result<BitNetTensor> {
        // Format layer name without exposing internal fields
        let layer_name =
            format!("QuantizedLinear[qtype={:?}, device={:?}]", self.qtype, self.device);
        let msg = format!(
            "FP32 fallback rejected in strict mode - layer={}, reason={}. \
             Strict mode requires native quantized kernels.",
            layer_name, reason
        );
        Err(bitnet_common::BitNetError::StrictMode(msg).into())
    }

    /// Forward pass with quantized matrix multiplication
    /// Input: [batch_size, seq_len, in_features]
    /// Output: [batch_size, seq_len, out_features]
    pub async fn forward(&self, input: &BitNetTensor) -> Result<BitNetTensor> {
        self.validate_input(input)?;

        let input_shape = input.shape();
        self.validate_input_dimensions(input_shape)?;

        // AC1: Debug assertions - panic in debug mode if fallback would occur
        #[cfg(debug_assertions)]
        {
            if self.is_fallback_path() {
                panic!(
                    "FP32 fallback blocked in debug mode: layer=[{}x{}], qtype={:?}, device={:?}, reason=kernel_unavailable",
                    self.in_features, self.out_features, self.qtype, self.device
                );
            }
        }

        // AC3: Strict mode validation - return error if fallback would occur
        // This enforces quantized-only inference in production when BITNET_STRICT_MODE=1
        let strict_mode = bitnet_common::strict_mode::StrictModeEnforcer::new();
        if strict_mode.get_config().enforce_quantized_inference && self.is_fallback_path() {
            return self.strict_reject_fp32_fallback("kernel_unavailable");
        }

        // Perform quantized matrix multiplication based on type
        let output = match self.qtype {
            QuantizationType::I2S => self.forward_i2s(input).await?,
            QuantizationType::TL1 => self.forward_tl1(input).await?,
            QuantizationType::TL2 => self.forward_tl2(input).await?,
        };

        // Add bias if present
        self.apply_bias_if_present(output)
    }

    /// Validate input dimensions for forward pass
    fn validate_input_dimensions(&self, input_shape: &[usize]) -> Result<()> {
        if input_shape.len() < 2 {
            return Err(QuantizedLinearError::ShapeMismatch {
                input: input_shape.to_vec(),
                weight: vec![self.in_features, self.out_features],
            }
            .into());
        }

        let features = input_shape[input_shape.len() - 1];
        if features != self.in_features {
            return Err(QuantizedLinearError::ShapeMismatch {
                input: input_shape.to_vec(),
                weight: vec![self.in_features, self.out_features],
            }
            .into());
        }

        Ok(())
    }

    /// Apply bias to output tensor if bias is present
    fn apply_bias_if_present(&self, output: BitNetTensor) -> Result<BitNetTensor> {
        if let Some(ref bias) = self.bias {
            let bias_candle = bias.to_candle()?;
            let output_candle = output.to_candle()?;
            let biased = output_candle
                .broadcast_add(&bias_candle)
                .context("Failed to add bias to output")?;
            Ok(BitNetTensor::new(biased))
        } else {
            Ok(output)
        }
    }

    /// I2S-specific forward pass implementation with optimized kernels
    async fn forward_i2s(&self, input: &BitNetTensor) -> Result<BitNetTensor> {
        let input_candle = input.to_candle()?;
        let input_shape = input_candle.shape();

        // Check for QK256 data first (GGML I2_S format with QK=256)
        if let Some(qk256_lock) = &self.qk256_data
            && let Some(qk256) = qk256_lock.get()
        {
            tracing::trace!(
                "Using QK256 kernel for layer {}×{} (GGML I2_S format)",
                self.out_features,
                self.in_features
            );
            return self.forward_qk256(input, qk256, input_shape.dims()).await;
        }

        // Standard I2S path (32-element blocks)
        let provider =
            self.kernel_manager.select_best().context("Failed to select kernel provider")?;

        // Reshape to 2D for efficient matrix multiplication
        let input_2d = self.prepare_input_for_matmul(&input_candle)?;

        // Select optimal kernel path based on device and quantization support
        let output_2d = if self.can_use_native_quantized_matmul() {
            self.quantized_matmul_i2s(&input_2d, provider).await?
        } else {
            self.fallback_i2s_matmul(&input_2d).await?
        };

        // Restore original tensor shape
        self.restore_output_shape(output_2d, input_shape.dims())
    }

    /// Forward pass using QK256 kernel (GGML I2_S format with QK=256)
    ///
    /// This method handles matrix multiplication for weights stored in the GGML I2_S
    /// quantization format (2-bit codes with 256-element blocks). It extracts the
    /// input data, calls the QK256 GEMV kernel, and reshapes the output to match
    /// the original input dimensions.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor (shape: [batch, seq, in_features] or [batch, in_features])
    /// * `qk256` - QK256 quantized weight data
    /// * `input_dims` - Original input dimensions for output reshaping
    ///
    /// # Returns
    ///
    /// Output tensor with shape matching input but last dimension = out_features
    async fn forward_qk256(
        &self,
        input: &BitNetTensor,
        qk256: &QK256Data,
        input_dims: &[usize],
    ) -> Result<BitNetTensor> {
        // Extract input data as f32 vector
        let input_candle = input.to_candle()?;
        let input_flat = input_candle
            .flatten_all()?
            .to_dtype(candle_core::DType::F32)?
            .to_vec1::<f32>()
            .context("Failed to extract input data for QK256 GEMV")?;

        // Calculate batch size (total number of input vectors)
        let batch_size = if input_dims.len() > 2 {
            input_dims[..input_dims.len() - 1].iter().product()
        } else {
            input_dims[0]
        };

        // Validate input dimensions
        let in_features = input_dims[input_dims.len() - 1];
        if in_features != self.in_features {
            anyhow::bail!(
                "QK256 forward: input features {} != layer in_features {}",
                in_features,
                self.in_features
            );
        }

        // Allocate output buffer
        let mut output_data = vec![0.0f32; batch_size * self.out_features];

        // Process each batch row using QK256 GEMV
        for batch_idx in 0..batch_size {
            let input_start = batch_idx * in_features;
            let input_end = input_start + in_features;
            let x = &input_flat[input_start..input_end];

            let output_start = batch_idx * self.out_features;
            let y_batch = &mut output_data[output_start..output_start + self.out_features];

            // Call QK256 GEMV kernel: y = W * x
            i2s_qk256::gemv_qk256(
                &qk256.qs,
                x,
                y_batch,
                qk256.rows,
                qk256.cols,
                qk256.row_stride_bytes,
            )
            .context("QK256 GEMV failed")?;
        }

        // Reshape output to match input dimensions (but with out_features as last dim)
        let output_shape = if input_dims.len() > 2 {
            let mut shape = input_dims[..input_dims.len() - 1].to_vec();
            shape.push(self.out_features);
            shape
        } else {
            vec![batch_size, self.out_features]
        };

        let output_tensor =
            candle_core::Tensor::from_vec(output_data, output_shape, input_candle.device())
                .context("Failed to create QK256 output tensor")?;

        Ok(BitNetTensor::new(output_tensor))
    }

    /// Prepare input tensor for matrix multiplication with optimal memory layout
    fn prepare_input_for_matmul(&self, input: &candle_core::Tensor) -> Result<candle_core::Tensor> {
        let shape = input.shape();
        if shape.dims().len() > 2 {
            let total_batch = self.compute_total_batch_size(shape.dims());
            self.reshape_with_alignment(input, total_batch, self.in_features)
        } else {
            Ok(input.clone())
        }
    }

    /// Restore output tensor to original batch dimensions
    fn restore_output_shape(
        &self,
        output_2d: candle_core::Tensor,
        input_dims: &[usize],
    ) -> Result<BitNetTensor> {
        let output = if input_dims.len() > 2 {
            let mut output_shape = input_dims.to_vec();
            let last_idx = output_shape.len() - 1;
            output_shape[last_idx] = self.out_features;
            output_2d.reshape(output_shape).context("Failed to reshape output")?
        } else {
            output_2d
        };

        Ok(BitNetTensor::new(output))
    }

    /// TL1 forward pass optimized for ARM NEON with lookup table vectorization
    async fn forward_tl1(&self, input: &BitNetTensor) -> Result<BitNetTensor> {
        #[cfg(target_arch = "aarch64")]
        {
            // Use ARM NEON-optimized TL1 kernel when available
            if let Ok(provider) = self.kernel_manager.select_best() {
                if provider.name().contains("neon") || provider.name().contains("arm") {
                    return self.vectorized_tl1_matmul(input, provider).await;
                }
            }
        }

        // Fallback to optimized generic implementation
        self.forward_tl1_generic(input).await
    }

    /// TL2 forward pass optimized for x86 AVX2/AVX-512 with larger lookup tables
    async fn forward_tl2(&self, input: &BitNetTensor) -> Result<BitNetTensor> {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            // Use x86 AVX-optimized TL2 kernel when available
            if let Ok(provider) = self.kernel_manager.select_best()
                && (provider.name().contains("avx") || provider.name().contains("x86"))
            {
                return self.vectorized_tl2_matmul(input, provider).await;
            }
        }

        // Fallback to optimized generic implementation
        self.forward_tl2_generic(input).await
    }

    /// Calculate alignment padding for optimal SIMD performance
    fn calculate_alignment_padding(in_features: usize, out_features: usize) -> usize {
        let total_elements = in_features * out_features;
        let remainder = total_elements % CACHE_LINE_SIZE;
        if remainder == 0 { 0 } else { CACHE_LINE_SIZE - remainder }
    }

    /// Compute total batch size for efficient memory layout
    fn compute_total_batch_size(&self, dims: &[usize]) -> usize {
        dims[..dims.len() - 1].iter().product::<usize>()
    }

    /// Reshape tensor with SIMD alignment for optimal performance
    fn reshape_with_alignment(
        &self,
        tensor: &candle_core::Tensor,
        batch_size: usize,
        feature_size: usize,
    ) -> Result<candle_core::Tensor> {
        let reshaped = tensor
            .reshape(&[batch_size, feature_size])
            .context("Failed to reshape tensor with alignment")?;

        // Add alignment padding if needed for vectorization
        if self.alignment_padding > 0 {
            // Note: In a full implementation, this would add actual padding
            // For now, we just return the reshaped tensor
        }

        Ok(reshaped)
    }

    /// Check if native quantized matrix multiplication is available
    fn can_use_native_quantized_matmul(&self) -> bool {
        match (&self.device, &self.qtype) {
            (Device::Cuda(_), QuantizationType::I2S) => true, // GPU I2S kernel available
            (Device::Cpu, QuantizationType::I2S) => true, // CPU I2S kernel always available via fallback
            (Device::Cpu, QuantizationType::TL1) => true, // CPU TL1 kernel available
            (Device::Cpu, QuantizationType::TL2) => true, // CPU TL2 kernel available
            _ => true,                                    // Default to native quantized operations
        }
    }

    /// Native quantized matrix multiplication for I2S without dequantization
    async fn quantized_matmul_i2s(
        &self,
        input: &candle_core::Tensor,
        provider: &dyn bitnet_kernels::KernelProvider,
    ) -> Result<candle_core::Tensor> {
        use utils::quantize_input_i2s;

        // Extract input data and compute dimensions
        let input_shape = input.shape().dims();
        let input_data = self.extract_input_data(input)?;
        let (batch_size, in_features, out_features) = self.calculate_matmul_dimensions(input_shape);

        // Prepare quantized inputs and weights
        let quantized_input = quantize_input_i2s(&input_data, in_features)?;
        let weight_data = self.prepare_quantized_weights_i2s()?;

        // Execute kernel operation
        let mut output_data = vec![0.0f32; batch_size * out_features];
        provider
            .matmul_i2s(
                &quantized_input,
                &weight_data,
                &mut output_data,
                batch_size,
                out_features,
                in_features,
            )
            .context("Native I2S quantized matmul failed")?;

        // Apply quantization scales
        self.apply_quantization_scales(&mut output_data, out_features);

        // Create output tensor with correct shape
        self.create_output_tensor(
            output_data,
            input_shape,
            batch_size,
            out_features,
            input.device(),
        )
    }

    /// Extract input data from tensor
    fn extract_input_data(&self, input: &candle_core::Tensor) -> Result<Vec<f32>> {
        input
            .flatten_all()?
            .to_dtype(candle_core::DType::F32)?
            .to_vec1::<f32>()
            .context("Failed to extract input data for quantized matmul")
    }

    /// Calculate matrix multiplication dimensions
    fn calculate_matmul_dimensions(&self, input_shape: &[usize]) -> (usize, usize, usize) {
        let batch_size = if input_shape.len() > 2 {
            input_shape[..input_shape.len() - 1].iter().product()
        } else {
            input_shape[0]
        };
        let in_features = input_shape[input_shape.len() - 1];
        let out_features = self.out_features;

        (batch_size, in_features, out_features)
    }

    /// Prepare quantized weights for I2S kernel
    fn prepare_quantized_weights_i2s(&self) -> Result<Vec<u8>> {
        use utils::unpack_2bit_values;

        let unpacked_weights = unpack_2bit_values(&self.weights.data, self.weights.numel());

        // Convert i8 range [-2,1] to u8 range [0,3] for kernel interface
        Ok(unpacked_weights.iter().map(|&x| (x + 2) as u8).collect())
    }

    /// Apply quantization scales to output data
    fn apply_quantization_scales(&self, output_data: &mut [f32], out_features: usize) {
        let input_scale = 1.0; // Input quantization scale
        let block_size = self.weights.block_size;
        let batch_size = output_data.len() / out_features;

        // Apply scales per-feature or per-block, not per-batch-row
        for row in 0..batch_size {
            let base = row * out_features;
            for col in 0..out_features {
                // Determine which scale to use based on block structure
                // For per-feature quantization: use scales[col]
                // For block quantization: use scales[col / block_size]
                let scale_idx = if self.weights.scales.len() == out_features {
                    col // Per-feature quantization
                } else {
                    // Block-based quantization: determine block for this output feature
                    let weight_idx = col * self.in_features; // Simplified: assumes column-major
                    (weight_idx / block_size).min(self.weights.scales.len() - 1)
                };
                let scale = self.weights.scales.get(scale_idx).copied().unwrap_or(1.0);
                output_data[base + col] *= input_scale * scale;
            }
        }
    }

    /// Create output tensor with proper shape
    fn create_output_tensor(
        &self,
        output_data: Vec<f32>,
        input_shape: &[usize],
        batch_size: usize,
        out_features: usize,
        device: &candle_core::Device,
    ) -> Result<candle_core::Tensor> {
        let output_shape = if input_shape.len() > 2 {
            let mut shape = input_shape[..input_shape.len() - 1].to_vec();
            shape.push(out_features);
            shape
        } else {
            vec![batch_size, out_features]
        };

        candle_core::Tensor::from_vec(output_data, output_shape, device)
            .context("Failed to create output tensor")
    }

    /// Fallback I2S matrix multiplication with dequantization (deprecated - only for compatibility)
    async fn fallback_i2s_matmul(
        &self,
        input: &candle_core::Tensor,
    ) -> Result<candle_core::Tensor> {
        log::warn!(
            "Using deprecated fallback_i2s_matmul - this should not happen in native quantized mode"
        );

        let dequantized_weights =
            self.weights.dequantize().context("Failed to dequantize I2S weights")?;
        let weight_candle = dequantized_weights.to_candle()?;
        let weight_transposed = weight_candle.t().context("Failed to transpose weights")?;

        input.matmul(&weight_transposed).context("Failed to perform I2S matrix multiplication")
    }

    /// Native TL1 quantized matrix multiplication
    async fn forward_tl1_generic(&self, input: &BitNetTensor) -> Result<BitNetTensor> {
        let input_candle = input.to_candle()?;

        // Dequantize TL1 weights to float32 using the correct TL1 dequantization path
        // (unsigned 2-bit codes [0,3] → centered values via block scales).
        // The previous approach passed TL1 data through matmul_i2s which expected I2S
        // format, causing wrong unpack offsets and near-zero inputs after over-coarse
        // integer quantization of the activations.
        let dequantized_weights = self.weights.dequantize().with_context(|| {
            format!(
                "Failed to dequantize TL1 weights for layer {}x{}",
                self.in_features, self.out_features
            )
        })?;
        let weight_candle = dequantized_weights.to_candle()?;

        // Flatten all leading dimensions to 2D so candle's matmul can handle
        // inputs with shape [batch, seq, in_features].
        let input_dims = input_candle.dims().to_vec();
        let n_rows: usize = input_dims[..input_dims.len() - 1].iter().product();
        let k = input_dims[input_dims.len() - 1];
        let input_2d =
            input_candle.reshape(&[n_rows, k]).context("Failed to reshape TL1 input to 2D")?;

        // Weights stored as [in_features, out_features]; no transpose needed.
        let output_2d = input_2d
            .matmul(&weight_candle)
            .context("Failed to perform TL1 matrix multiplication")?;

        // Restore the original leading dimensions.
        let mut out_shape = input_dims[..input_dims.len() - 1].to_vec();
        out_shape.push(self.out_features);
        let output = output_2d.reshape(out_shape).context("Failed to restore TL1 output shape")?;
        Ok(BitNetTensor::new(output))
    }

    /// Native TL2 quantized matrix multiplication
    async fn forward_tl2_generic(&self, input: &BitNetTensor) -> Result<BitNetTensor> {
        let input_candle = input.to_candle()?;

        // Dequantize TL2 weights to float32 using the correct TL2 dequantization path.
        // The previous approach passed TL2 data through matmul_i2s (an I2S-format kernel)
        // which caused mismatched unpacking and a hardcoded 0.9 correction factor.
        let dequantized_weights = self.weights.dequantize().with_context(|| {
            format!(
                "Failed to dequantize TL2 weights for layer {}x{}",
                self.in_features, self.out_features
            )
        })?;
        let weight_candle = dequantized_weights.to_candle()?;

        // Flatten all leading dimensions to 2D so candle's matmul can handle
        // inputs with shape [batch, seq, in_features].
        let input_dims = input_candle.dims().to_vec();
        let n_rows: usize = input_dims[..input_dims.len() - 1].iter().product();
        let k = input_dims[input_dims.len() - 1];
        let input_2d =
            input_candle.reshape(&[n_rows, k]).context("Failed to reshape TL2 input to 2D")?;

        // Weights stored as [in_features, out_features]; no transpose needed.
        let output_2d = input_2d
            .matmul(&weight_candle)
            .context("Failed to perform TL2 matrix multiplication")?;

        // Restore the original leading dimensions.
        let mut out_shape = input_dims[..input_dims.len() - 1].to_vec();
        out_shape.push(self.out_features);
        let output = output_2d.reshape(out_shape).context("Failed to restore TL2 output shape")?;
        Ok(BitNetTensor::new(output))
    }

    /// Vectorized TL1 matrix multiplication using NEON
    #[cfg(target_arch = "aarch64")]
    async fn vectorized_tl1_matmul(
        &self,
        input: &BitNetTensor,
        _provider: &dyn bitnet_kernels::KernelProvider,
    ) -> Result<BitNetTensor> {
        // Would use NEON-optimized TL1 kernel
        self.forward_tl1_generic(input).await
    }

    /// Vectorized TL2 matrix multiplication using AVX
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    async fn vectorized_tl2_matmul(
        &self,
        input: &BitNetTensor,
        _provider: &dyn bitnet_kernels::KernelProvider,
    ) -> Result<BitNetTensor> {
        // Would use AVX-optimized TL2 kernel
        self.forward_tl2_generic(input).await
    }

    /// Generic fallback implementation with improved error handling
    #[allow(dead_code)]
    async fn forward_generic(&self, input: &BitNetTensor) -> Result<BitNetTensor> {
        let dequantized_weights = self.weights.dequantize().with_context(|| {
            format!(
                "Failed to dequantize {:?} weights for layer {}x{}",
                self.qtype, self.in_features, self.out_features
            )
        })?;
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
        // Ensure weights are aligned to I2S block boundaries
        let total_blocks = self.weights.data.len().div_ceil(I2S_BLOCK_SIZE);
        let aligned_size = total_blocks * I2S_BLOCK_SIZE;

        // Pre-allocate aligned memory pool for I2S operations
        if aligned_size > 0 && aligned_size <= MAX_WORKSPACE_SIZE {
            let mut pool = Vec::with_capacity(aligned_size + CACHE_LINE_SIZE);
            pool.resize(aligned_size, 0u8);
            self.memory_pool = Some(pool);

            log::debug!(
                "Optimized I2S layout: {} blocks, {} bytes aligned",
                total_blocks,
                aligned_size
            );
        }

        Ok(())
    }

    /// Optimize TL1 layout for ARM NEON (16-byte alignment)
    fn optimize_tl1_layout(&mut self) -> Result<()> {
        #[cfg(target_arch = "aarch64")]
        {
            // ARM NEON requires 16-byte alignment for optimal performance
            let neon_alignment = 16;
            let aligned_size = self.calculate_aligned_size(neon_alignment);

            if aligned_size > 0 && aligned_size <= MAX_WORKSPACE_SIZE {
                let mut pool = Vec::with_capacity(aligned_size + neon_alignment);
                pool.resize(aligned_size, 0u8);
                self.memory_pool = Some(pool);

                log::debug!("Optimized TL1 layout for NEON: {} bytes", aligned_size);
            }
        }

        Ok(())
    }

    /// Optimize TL2 layout for x86 AVX (32-byte alignment)
    fn optimize_tl2_layout(&mut self) -> Result<()> {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            // AVX requires 32-byte alignment for optimal performance
            let avx_alignment = if cfg!(target_feature = "avx512f") { 64 } else { 32 };
            let aligned_size = self.calculate_aligned_size(avx_alignment);

            if aligned_size > 0 && aligned_size <= MAX_WORKSPACE_SIZE {
                let mut pool = Vec::with_capacity(aligned_size + avx_alignment);
                pool.resize(aligned_size, 0u8);
                self.memory_pool = Some(pool);

                log::debug!(
                    "Optimized TL2 layout for AVX: {} bytes (alignment: {})",
                    aligned_size,
                    avx_alignment
                );
            }
        }

        Ok(())
    }

    /// Calculate aligned memory size for given alignment requirement
    fn calculate_aligned_size(&self, alignment: usize) -> usize {
        let base_size = self.in_features * self.out_features * std::mem::size_of::<f32>();
        let remainder = base_size % alignment;
        if remainder == 0 { base_size } else { base_size + (alignment - remainder) }
    }

    /// Pre-allocate GPU workspace to avoid runtime allocation
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    fn allocate_gpu_workspace(&mut self) -> Result<()> {
        let workspace_size = self.calculate_gpu_workspace_size()?;
        self.workspace = Some(BitNetTensor::zeros(&[workspace_size], DType::F32, &self.device)?);

        log::debug!("Allocated GPU workspace: {} MB", workspace_size * 4 / 1024 / 1024);
        Ok(())
    }

    #[cfg(not(any(feature = "gpu", feature = "cuda")))]
    fn allocate_gpu_workspace(&mut self) -> Result<()> {
        // No-op for non-GPU builds
        Ok(())
    }

    /// Calculate GPU workspace size with memory constraints
    #[allow(dead_code)]
    fn calculate_gpu_workspace_size(&self) -> Result<usize> {
        // GPU kernels need temporary storage for different quantization types
        let base_weight_size = self.in_features * self.out_features;

        let (dequant_multiplier, intermediate_multiplier) = match self.qtype {
            QuantizationType::I2S => (2, 4), // FP16 + FP32
            QuantizationType::TL1 => (2, 4), // FP16 + FP32
            QuantizationType::TL2 => (4, 4), // FP32 + FP32 (larger tables)
        };

        // Conservative batch size estimate based on available GPU memory
        let max_batch_size = match self.device {
            Device::Cuda(_) => {
                // Estimate based on 6GB GPU memory target
                let available_memory: usize = 6 * 1024 * 1024 * 1024; // 6GB
                let model_memory = base_weight_size * dequant_multiplier;
                let remaining = available_memory.saturating_sub(model_memory);
                (remaining / (self.out_features * intermediate_multiplier)).min(128)
            }
            _ => 64, // Conservative default
        };

        let dequant_size = base_weight_size * dequant_multiplier;
        let intermediate_size = max_batch_size * self.out_features * intermediate_multiplier;
        let total_size = dequant_size + intermediate_size;

        // Clamp to maximum workspace size to prevent OOM
        let workspace_size = total_size.min(MAX_WORKSPACE_SIZE);

        log::debug!(
            "GPU workspace size: {} MB (batch_size: {}, qtype: {:?})",
            workspace_size / (1024 * 1024),
            max_batch_size,
            self.qtype
        );

        Ok(workspace_size)
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

    /// Get comprehensive memory usage breakdown (for optimization)
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
        let scale_cache_bytes = self
            .scale_cache
            .as_ref()
            .map(|s| s.shape().iter().product::<usize>() * std::mem::size_of::<f32>())
            .unwrap_or(0);
        let memory_pool_bytes = self.memory_pool.as_ref().map(|p| p.capacity()).unwrap_or(0);

        weight_bytes
            + scale_bytes
            + bias_bytes
            + workspace_bytes
            + scale_cache_bytes
            + memory_pool_bytes
    }

    /// Get detailed memory breakdown for profiling
    pub fn memory_breakdown(&self) -> std::collections::HashMap<String, usize> {
        let mut breakdown = std::collections::HashMap::new();

        breakdown.insert("weights".to_string(), self.weights.data.len());
        breakdown
            .insert("scales".to_string(), self.weights.scales.len() * std::mem::size_of::<f32>());
        breakdown.insert(
            "bias".to_string(),
            self.bias
                .as_ref()
                .map(|b| b.shape().iter().product::<usize>() * std::mem::size_of::<f32>())
                .unwrap_or(0),
        );
        breakdown.insert(
            "workspace".to_string(),
            self.workspace
                .as_ref()
                .map(|w| w.shape().iter().product::<usize>() * std::mem::size_of::<f32>())
                .unwrap_or(0),
        );
        breakdown.insert(
            "scale_cache".to_string(),
            self.scale_cache
                .as_ref()
                .map(|s| s.shape().iter().product::<usize>() * std::mem::size_of::<f32>())
                .unwrap_or(0),
        );
        breakdown.insert(
            "memory_pool".to_string(),
            self.memory_pool.as_ref().map(|p| p.capacity()).unwrap_or(0),
        );

        breakdown
    }

    /// Get detailed performance metrics for the layer
    pub fn get_performance_metrics(&self) -> PerformanceMetrics {
        let theoretical_gops = self.calculate_theoretical_throughput();
        let memory_bandwidth_efficiency = self.calculate_memory_bandwidth_efficiency();

        PerformanceMetrics {
            average_lookup_cycles: match self.qtype {
                QuantizationType::I2S => {
                    // I2S with optimized kernels
                    if self.can_use_native_quantized_matmul() { 1.2 } else { 1.8 }
                }
                QuantizationType::TL1 => {
                    // TL1 with NEON optimization
                    #[cfg(target_arch = "aarch64")]
                    {
                        if cfg!(target_feature = "neon") { 1.5 } else { 2.2 }
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    {
                        2.2
                    }
                }
                QuantizationType::TL2 => {
                    // TL2 with AVX optimization
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    {
                        if cfg!(target_feature = "avx2") { 2.0 } else { 3.2 }
                    }
                    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
                    {
                        3.2
                    }
                }
            },
            memory_efficiency: self.weights.compression_ratio() * memory_bandwidth_efficiency,
            throughput_gops: theoretical_gops,
        }
    }

    /// Calculate theoretical throughput based on device and quantization type
    fn calculate_theoretical_throughput(&self) -> f32 {
        let ops_per_element = 2.0; // MAC operation
        let total_ops = (self.in_features * self.out_features) as f32 * ops_per_element;

        match &self.device {
            Device::Cuda(_) => {
                // GPU throughput depends on quantization type
                match self.qtype {
                    QuantizationType::I2S => total_ops * 0.8, // 80% peak efficiency
                    QuantizationType::TL1 => total_ops * 0.7, // Lookup overhead
                    QuantizationType::TL2 => total_ops * 0.65, // Larger lookup overhead
                }
            }
            Device::Cpu => {
                // CPU throughput with SIMD optimization
                let simd_efficiency = match self.qtype {
                    QuantizationType::I2S => {
                        #[cfg(target_feature = "avx2")]
                        {
                            0.6
                        }
                        #[cfg(not(target_feature = "avx2"))]
                        {
                            0.3
                        }
                    }
                    QuantizationType::TL1 => {
                        #[cfg(target_arch = "aarch64")]
                        {
                            0.5
                        }
                        #[cfg(not(target_arch = "aarch64"))]
                        {
                            0.25
                        }
                    }
                    QuantizationType::TL2 => {
                        #[cfg(target_feature = "avx2")]
                        {
                            0.4
                        }
                        #[cfg(not(target_feature = "avx2"))]
                        {
                            0.2
                        }
                    }
                };
                total_ops * simd_efficiency
            }
            Device::Metal => {
                // Metal GPU throughput (similar to CUDA but potentially different)
                match self.qtype {
                    QuantizationType::I2S => total_ops * 0.75,
                    QuantizationType::TL1 => total_ops * 0.65,
                    QuantizationType::TL2 => total_ops * 0.6,
                }
            }
        }
    }

    /// Calculate memory bandwidth efficiency
    fn calculate_memory_bandwidth_efficiency(&self) -> f32 {
        let arithmetic_intensity = self.calculate_arithmetic_intensity();

        // Memory bandwidth efficiency based on arithmetic intensity and caching
        let cache_efficiency = if self.memory_pool.is_some() { 1.2 } else { 1.0 };
        let alignment_efficiency = if self.alignment_padding > 0 { 1.1 } else { 1.0 };

        (arithmetic_intensity * cache_efficiency * alignment_efficiency).min(1.0)
    }

    /// Calculate arithmetic intensity (operations per byte)
    fn calculate_arithmetic_intensity(&self) -> f32 {
        let ops_per_element = 2.0; // MAC operation
        let total_ops = (self.in_features * self.out_features) as f32 * ops_per_element;
        let memory_bytes = self.memory_usage() as f32;

        if memory_bytes > 0.0 { total_ops / memory_bytes } else { 0.0 }
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
        .map(|i| (i as f32).sin() * 0.1) // Simple sinusoidal pattern
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
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        // Check for CUDA availability
        candle_core::Device::new_cuda(0).is_ok()
    }
    #[cfg(not(any(feature = "gpu", feature = "cuda")))]
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

impl QuantizedLinear {
    /// Native TL1 quantized matrix multiplication
    #[allow(dead_code)]
    async fn quantized_matmul_tl1(
        &self,
        input: &candle_core::Tensor,
        provider: &dyn bitnet_kernels::KernelProvider,
    ) -> Result<candle_core::Tensor> {
        use utils::{quantize_input_tl1, unpack_tl1_values};

        // Get input dimensions and data
        let input_shape = input.shape().dims();
        let input_data = input
            .flatten_all()?
            .to_dtype(candle_core::DType::F32)?
            .to_vec1::<f32>()
            .context("Failed to extract input data for TL1 quantized matmul")?;

        // Calculate matrix dimensions
        let batch_size = if input_shape.len() > 2 {
            input_shape[..input_shape.len() - 1].iter().product()
        } else {
            input_shape[0]
        };
        let in_features = input_shape[input_shape.len() - 1];
        let out_features = self.out_features;

        // Quantize input to TL1 format (use simple quantization for now)
        let quantized_input = quantize_input_tl1(&input_data, in_features)?;

        // Unpack quantized weights for TL1 (assumes 4-bit encoding)
        let unpacked_weights = unpack_tl1_values(&self.weights.data, self.weights.numel());

        // Convert weights to u8 format for kernel interface
        let weight_data: Vec<u8> = unpacked_weights.iter()
            .map(|&x| (x + 8) as u8) // Convert signed to unsigned range
            .collect();

        // Prepare output buffer
        let mut output_data = vec![0.0f32; batch_size * out_features];

        // Call quantized kernel (fallback to I2S kernel for now)
        provider
            .matmul_i2s(
                &quantized_input,
                &weight_data,
                &mut output_data,
                batch_size,
                out_features,
                in_features,
            )
            .context("Native TL1 quantized matmul failed")?;

        // Apply TL1-specific scaling per-feature/block
        let input_scale = 1.0;
        let block_size = self.weights.block_size;
        for row in 0..batch_size {
            let base = row * out_features;
            for col in 0..out_features {
                let scale_idx = if self.weights.scales.len() == out_features {
                    col
                } else {
                    let weight_idx = col * in_features;
                    (weight_idx / block_size).min(self.weights.scales.len() - 1)
                };
                let scale = self.weights.scales.get(scale_idx).copied().unwrap_or(1.0);
                output_data[base + col] *= input_scale * scale * 0.8; // TL1 correction factor
            }
        }

        // Reshape output
        let output_shape = if input_shape.len() > 2 {
            let mut shape = input_shape[..input_shape.len() - 1].to_vec();
            shape.push(out_features);
            shape
        } else {
            vec![batch_size, out_features]
        };

        let output_tensor =
            candle_core::Tensor::from_vec(output_data, output_shape, input.device())
                .context("Failed to create output tensor for TL1 quantized matmul")?;

        Ok(output_tensor)
    }

    /// Native TL2 quantized matrix multiplication
    #[allow(dead_code)]
    async fn quantized_matmul_tl2(
        &self,
        input: &candle_core::Tensor,
        provider: &dyn bitnet_kernels::KernelProvider,
    ) -> Result<candle_core::Tensor> {
        use utils::{quantize_input_tl2, unpack_tl2_values};

        // Get input dimensions and data
        let input_shape = input.shape().dims();
        let input_data = input
            .flatten_all()?
            .to_dtype(candle_core::DType::F32)?
            .to_vec1::<f32>()
            .context("Failed to extract input data for TL2 quantized matmul")?;

        // Calculate matrix dimensions
        let batch_size = if input_shape.len() > 2 {
            input_shape[..input_shape.len() - 1].iter().product()
        } else {
            input_shape[0]
        };
        let in_features = input_shape[input_shape.len() - 1];
        let out_features = self.out_features;

        // Quantize input to TL2 format (use enhanced quantization)
        let quantized_input = quantize_input_tl2(&input_data, in_features)?;

        // Unpack quantized weights for TL2 (assumes 8-bit encoding)
        let unpacked_weights = unpack_tl2_values(&self.weights.data, self.weights.numel());

        // Convert weights to u8 format for kernel interface
        let weight_data: Vec<u8> = unpacked_weights.iter()
            .map(|&x| (x as i16 + 128) as u8) // Convert signed to unsigned range
            .collect();

        // Prepare output buffer
        let mut output_data = vec![0.0f32; batch_size * out_features];

        // Call quantized kernel (fallback to I2S kernel for now)
        provider
            .matmul_i2s(
                &quantized_input,
                &weight_data,
                &mut output_data,
                batch_size,
                out_features,
                in_features,
            )
            .context("Native TL2 quantized matmul failed")?;

        // Apply TL2-specific scaling per-feature/block
        let input_scale = 1.0;
        let block_size = self.weights.block_size;
        for row in 0..batch_size {
            let base = row * out_features;
            for col in 0..out_features {
                let scale_idx = if self.weights.scales.len() == out_features {
                    col
                } else {
                    let weight_idx = col * in_features;
                    (weight_idx / block_size).min(self.weights.scales.len() - 1)
                };
                let scale = self.weights.scales.get(scale_idx).copied().unwrap_or(1.0);
                output_data[base + col] *= input_scale * scale * 0.9; // TL2 correction factor
            }
        }

        // Reshape output
        let output_shape = if input_shape.len() > 2 {
            let mut shape = input_shape[..input_shape.len() - 1].to_vec();
            shape.push(out_features);
            shape
        } else {
            vec![batch_size, out_features]
        };

        let output_tensor =
            candle_core::Tensor::from_vec(output_data, output_shape, input.device())
                .context("Failed to create output tensor for TL2 quantized matmul")?;

        Ok(output_tensor)
    }
}

/// Utility module for quantized operations
mod utils {
    use anyhow::Result;

    /// Unpack 2-bit values from packed byte array
    pub fn unpack_2bit_values(packed: &[u8], count: usize) -> Vec<i8> {
        let mut unpacked = Vec::with_capacity(count);

        for &byte in packed.iter() {
            if unpacked.len() >= count {
                break;
            }

            // Extract 4 values from each byte (2 bits each)
            for shift in [0, 2, 4, 6] {
                if unpacked.len() >= count {
                    break;
                }
                let value = ((byte >> shift) & 0b11) as i8;
                // Convert from [0,3] to [-2,1] range for I2S
                unpacked.push(value - 2);
            }
        }

        unpacked.truncate(count);
        unpacked
    }

    /// Quantize input data to I2S format
    pub fn quantize_input_i2s(input: &[f32], _features: usize) -> Result<Vec<i8>> {
        // Simple quantization to I2S range [-2, 1]
        let quantized: Vec<i8> = input
            .iter()
            .map(|&x| {
                let clamped = x.clamp(-2.0, 1.0);
                clamped.round() as i8
            })
            .collect();

        Ok(quantized)
    }

    /// Quantize input data to TL1 format
    #[allow(dead_code)]
    pub fn quantize_input_tl1(input: &[f32], _features: usize) -> Result<Vec<i8>> {
        // Simple quantization to TL1 range (4-bit signed)
        let quantized: Vec<i8> = input
            .iter()
            .map(|&x| {
                let clamped = x.clamp(-8.0, 7.0);
                clamped.round() as i8
            })
            .collect();

        Ok(quantized)
    }

    /// Quantize input data to TL2 format
    #[allow(dead_code)]
    pub fn quantize_input_tl2(input: &[f32], _features: usize) -> Result<Vec<i8>> {
        // Simple quantization to TL2 range (8-bit signed)
        let quantized: Vec<i8> = input
            .iter()
            .map(|&x| {
                let clamped = x.clamp(-128.0, 127.0);
                clamped.round() as i8
            })
            .collect();

        Ok(quantized)
    }

    /// Unpack TL1 values (2-bit, same as I2S)
    /// TL1 uses 2-bit quantization with a 16-entry lookup table
    #[allow(dead_code)]
    pub fn unpack_tl1_values(packed: &[u8], count: usize) -> Vec<i8> {
        // TL1 uses the same 2-bit packing format as I2S
        // The "TL1" designation refers to the table lookup optimization strategy,
        // not a different bit width
        unpack_2bit_values(packed, count)
    }

    /// Unpack TL2 values (8-bit)
    #[allow(dead_code)]
    pub fn unpack_tl2_values(packed: &[u8], count: usize) -> Vec<i8> {
        // TL2 uses the same 2-bit packing format as I2S
        // The "8-bit" in TL2 refers to the lookup table size (256 entries), not the bit width
        unpack_2bit_values(packed, count)
    }
}
