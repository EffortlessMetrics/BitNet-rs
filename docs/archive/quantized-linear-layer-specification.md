# Quantized Linear Layer Specification

**Component**: Core neural network linear layers with I2S/TL1/TL2 quantization integration
**Location**: `bitnet-inference/src/layers/quantized_linear.rs`
**Dependencies**: bitnet-quantization, bitnet-kernels

## Overview

Quantized linear layers form the backbone of BitNet transformer computation, replacing standard FP32 linear operations with 1-bit and 2-bit quantized matrix multiplications. This specification defines production-ready quantized linear layers that integrate seamlessly with BitNet-rs quantization infrastructure while maintaining >99% accuracy compared to full-precision computation.

## Architecture Design

### Core Components

```rust
// Primary quantized linear layer implementation
pub struct QuantizedLinear {
    // Quantized weight storage
    weights: QuantizedTensor,              // I2S/TL1/TL2 quantized weights
    bias: Option<Tensor>,                  // Optional bias (kept in FP32)

    // Quantization infrastructure
    quantizer: DeviceAwareQuantizer,       // Kernel selection (CPU/GPU)
    qtype: QuantizationType,              // I2S, TL1, or TL2

    // Layer metadata
    in_features: usize,                   // Input dimension
    out_features: usize,                  // Output dimension
    device: Device,                       // CPU or GPU device context

    // Performance optimization
    workspace: Option<Tensor>,            // Pre-allocated workspace for GPU kernels
    scale_cache: Option<Tensor>,          // Cached scales for repeated operations
}

impl QuantizedLinear {
    /// Create quantized linear layer from GGUF tensor
    pub fn from_gguf(
        weights: &GgufTensor,
        bias: Option<&GgufTensor>,
        device: &Device,
        qtype: QuantizationType
    ) -> Result<Self>;

    /// Forward pass with quantized matrix multiplication
    /// Input: [batch_size, seq_len, in_features]
    /// Output: [batch_size, seq_len, out_features]
    pub fn forward(&self, input: &Tensor) -> Result<Tensor>;

    /// Device-aware forward pass with automatic GPU/CPU selection
    pub fn forward_device_aware(&self, input: &Tensor) -> Result<Tensor>;

    /// Get memory usage in bytes (for optimization)
    pub fn memory_usage(&self) -> usize;

    /// Validate quantization accuracy against FP32 reference
    pub fn validate_accuracy(&self, fp32_weights: &Tensor, tolerance: f32) -> Result<f32>;
}
```

## Quantization Integration

### I2S (2-bit Signed) Integration

```rust
impl QuantizedLinear {
    /// I2S-specific forward pass implementation
    fn forward_i2s(&self, input: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, in_features) = input.dims3()?;

        // Reshape input for matrix multiplication: [B*S, I] × [I, O] = [B*S, O]
        let input_2d = input.reshape(&[batch_size * seq_len, in_features])?;

        // Device-aware I2S matrix multiplication
        let output_2d = match self.device {
            Device::Cuda(_) => self.forward_i2s_gpu(&input_2d)?,
            Device::Cpu => self.forward_i2s_cpu(&input_2d)?,
        };

        // Reshape back to 3D: [B*S, O] → [B, S, O]
        let mut output = output_2d.reshape(&[batch_size, seq_len, self.out_features])?;

        // Add bias if present
        if let Some(ref bias) = self.bias {
            output = output.broadcast_add(bias)?;
        }

        Ok(output)
    }

    /// I2S CPU implementation using SIMD kernels
    fn forward_i2s_cpu(&self, input: &Tensor) -> Result<Tensor> {
        let kernel = self.quantizer.select_cpu_kernel()?;

        // Extract quantized weight data (82-byte blocks)
        let weight_data = &self.weights.data;
        let scales = &self.weights.scales;
        let zero_points = self.weights.zero_points.as_ref();

        // Allocate output tensor
        let output_shape = [input.dims()[0], self.out_features];
        let mut output = Tensor::zeros(&output_shape, DType::F32, &self.device)?;

        // Call I2S kernel with proper block handling
        kernel.matmul_i2s_blocked(
            input.as_ptr::<f32>(),
            weight_data.as_ptr(),
            scales.as_ptr(),
            zero_points.map(|zp| zp.as_ptr()).unwrap_or(std::ptr::null()),
            output.as_mut_ptr::<f32>(),
            input.dims()[0],         // M (batch * seq_len)
            self.out_features,       // N
            self.in_features,        // K
            82,                      // I2S block size
        )?;

        Ok(output)
    }

    /// I2S GPU implementation using CUDA kernels
    #[cfg(feature = "gpu")]
    fn forward_i2s_gpu(&self, input: &Tensor) -> Result<Tensor> {
        let cuda_kernel = self.quantizer.select_gpu_kernel()?;

        // Use pre-allocated workspace to avoid dynamic allocation
        let workspace = self.workspace.as_ref()
            .ok_or(InferenceError::DeviceError {
                reason: "GPU workspace not allocated".to_string()
            })?;

        // CUDA I2S kernel with mixed precision (FP16 intermediate)
        let output = cuda_kernel.matmul_i2s_mixed_precision(
            input,
            &self.weights,
            workspace,
            true, // use_fp16_intermediate
        )?;

        Ok(output)
    }
}
```

### TL1 (Table Lookup for ARM NEON)

```rust
impl QuantizedLinear {
    /// TL1 forward pass optimized for ARM NEON
    fn forward_tl1(&self, input: &Tensor) -> Result<Tensor> {
        #[cfg(target_arch = "aarch64")]
        {
            let neon_kernel = self.quantizer.select_neon_kernel()?;

            // TL1 uses lookup tables for 4-bit quantization
            let lookup_tables = self.extract_tl1_lookup_tables()?;
            let indices = &self.weights.data;

            let output = neon_kernel.matmul_tl1_vectorized(
                input,
                indices,
                &lookup_tables,
                self.in_features,
                self.out_features,
            )?;

            self.add_bias_if_present(output)
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            // Fallback to generic implementation
            self.forward_generic(input)
        }
    }

    /// Extract TL1 lookup tables from quantized weights
    fn extract_tl1_lookup_tables(&self) -> Result<Vec<f32>> {
        let scales = &self.weights.scales;
        let block_size = self.weights.block_size;

        // TL1 uses 16-entry lookup tables per block
        let mut lookup_tables = Vec::with_capacity(scales.len() * 16);

        for scale in scales {
            // Generate lookup table: [-8, -7, ..., 7] * scale
            for i in -8i8..=7i8 {
                lookup_tables.push((i as f32) * scale);
            }
        }

        Ok(lookup_tables)
    }
}
```

### TL2 (Table Lookup for x86 AVX2)

```rust
impl QuantizedLinear {
    /// TL2 forward pass optimized for x86 AVX2/AVX-512
    fn forward_tl2(&self, input: &Tensor) -> Result<Tensor> {
        #[cfg(target_arch = "x86_64")]
        {
            // Prefer AVX-512 if available, fallback to AVX2
            let kernel = if is_x86_feature_detected!("avx512f") {
                self.quantizer.select_avx512_kernel()?
            } else if is_x86_feature_detected!("avx2") {
                self.quantizer.select_avx2_kernel()?
            } else {
                return self.forward_generic(input);
            };

            // TL2 uses 8-bit indices with 256-entry lookup tables
            let lookup_tables = self.extract_tl2_lookup_tables()?;
            let indices = &self.weights.data;

            let output = kernel.matmul_tl2_simd(
                input.as_ptr::<f32>(),
                indices.as_ptr(),
                lookup_tables.as_ptr(),
                input.dims()[0],      // M
                self.out_features,    // N
                self.in_features,     // K
                32,                   // TL2 block size
            )?;

            self.add_bias_if_present(output)
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            // Fallback for non-x86 architectures
            self.forward_generic(input)
        }
    }

    /// Extract TL2 lookup tables (256 entries per block)
    fn extract_tl2_lookup_tables(&self) -> Result<Vec<f32>> {
        let scales = &self.weights.scales;
        let zero_points = self.weights.zero_points.as_ref()
            .ok_or(InferenceError::QuantizationError {
                context: "TL2 requires zero points".to_string()
            })?;

        let mut lookup_tables = Vec::with_capacity(scales.len() * 256);

        for (scale, zero_point) in scales.iter().zip(zero_points.iter()) {
            // Generate 256-entry lookup table
            for i in 0u8..=255 {
                let dequantized = (i as f32 - *zero_point as f32) * scale;
                lookup_tables.push(dequantized);
            }
        }

        Ok(lookup_tables)
    }
}
```

## Device-Aware Kernel Selection

### Automatic Device Selection

```rust
pub struct DeviceAwareQuantizer {
    cpu_kernels: Vec<Box<dyn CpuKernel>>,
    gpu_kernels: Vec<Box<dyn GpuKernel>>,
    selected_cpu: OnceLock<usize>,
    selected_gpu: OnceLock<usize>,
}

impl DeviceAwareQuantizer {
    pub fn new() -> Self {
        let mut cpu_kernels: Vec<Box<dyn CpuKernel>> = vec![];
        let mut gpu_kernels: Vec<Box<dyn GpuKernel>> = vec![];

        // CPU kernel priority order (best first)
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        cpu_kernels.push(Box::new(Avx512I2SKernel::new()));

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        cpu_kernels.push(Box::new(Avx2TL2Kernel::new()));

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        cpu_kernels.push(Box::new(NeonTL1Kernel::new()));

        // Always include fallback
        cpu_kernels.push(Box::new(FallbackKernel::new()));

        // GPU kernel detection
        #[cfg(feature = "gpu")]
        {
            if let Ok(cuda_kernel) = CudaI2SKernel::new() {
                gpu_kernels.push(Box::new(cuda_kernel));
            }
        }

        Self {
            cpu_kernels,
            gpu_kernels,
            selected_cpu: OnceLock::new(),
            selected_gpu: OnceLock::new(),
        }
    }

    /// Select best CPU kernel for quantization type
    pub fn select_cpu_kernel(&self, qtype: QuantizationType) -> Result<&dyn CpuKernel> {
        let selected_idx = self.selected_cpu.get_or_init(|| {
            for (i, kernel) in self.cpu_kernels.iter().enumerate() {
                if kernel.supports_quantization(qtype) && kernel.is_available() {
                    log::info!("Selected CPU kernel: {} for {:?}", kernel.name(), qtype);
                    return i;
                }
            }
            self.cpu_kernels.len() - 1 // Fallback
        });

        Ok(self.cpu_kernels[*selected_idx].as_ref())
    }

    /// Select best GPU kernel with graceful fallback
    #[cfg(feature = "gpu")]
    pub fn select_gpu_kernel(&self, qtype: QuantizationType) -> Result<&dyn GpuKernel> {
        if self.gpu_kernels.is_empty() {
            return Err(InferenceError::DeviceError {
                reason: "No GPU kernels available".to_string()
            });
        }

        let selected_idx = self.selected_gpu.get_or_init(|| {
            for (i, kernel) in self.gpu_kernels.iter().enumerate() {
                if kernel.supports_quantization(qtype) && kernel.is_available() {
                    log::info!("Selected GPU kernel: {} for {:?}", kernel.name(), qtype);
                    return i;
                }
            }
            0 // First available GPU kernel
        });

        Ok(self.gpu_kernels[*selected_idx].as_ref())
    }
}
```

## Performance Optimization

### Memory Layout Optimization

```rust
impl QuantizedLinear {
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
        let block_size = 82;
        let num_blocks = (self.in_features * self.out_features + 31) / 32; // 32 elements per block

        // Ensure weight data is properly aligned for SIMD operations
        let mut aligned_data = vec![0u8; num_blocks * block_size];
        let alignment = match self.device {
            Device::Cuda(_) => 128,  // GPU alignment
            Device::Cpu => 32,       // CPU SIMD alignment
        };

        // Copy with proper alignment
        aligned_data.copy_from_slice(&self.weights.data[..num_blocks * block_size]);
        self.weights.data = aligned_data;

        Ok(())
    }

    /// Pre-allocate GPU workspace to avoid runtime allocation
    #[cfg(feature = "gpu")]
    fn allocate_gpu_workspace(&mut self) -> Result<()> {
        let workspace_size = self.calculate_gpu_workspace_size()?;
        self.workspace = Some(Tensor::zeros(
            &[workspace_size],
            DType::F32,
            &self.device
        )?);

        log::debug!("Allocated GPU workspace: {} MB", workspace_size * 4 / 1024 / 1024);
        Ok(())
    }

    fn calculate_gpu_workspace_size(&self) -> Result<usize> {
        // GPU kernels need temporary storage for:
        // - Dequantized weights (if needed): in_features * out_features * sizeof(f16)
        // - Intermediate results: max_batch_size * out_features * sizeof(f32)
        let max_batch_size = 64; // Conservative estimate
        let dequant_size = self.in_features * self.out_features * 2; // FP16
        let intermediate_size = max_batch_size * self.out_features * 4; // FP32

        Ok(dequant_size + intermediate_size)
    }
}
```

### Batch Processing Optimization

```rust
impl QuantizedLinear {
    /// Optimized forward pass for batch processing
    pub fn forward_batch(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        if inputs.is_empty() {
            return Ok(vec![]);
        }

        // Check if batch processing is beneficial
        let total_elements: usize = inputs.iter().map(|t| t.elem_count()).sum();
        let batch_threshold = 1024; // Empirically determined

        if total_elements < batch_threshold {
            // Process individually for small batches
            inputs.iter().map(|input| self.forward(input)).collect()
        } else {
            // Concatenate for efficient batch processing
            self.forward_batched_concat(inputs)
        }
    }

    /// Efficient batched processing by concatenation
    fn forward_batched_concat(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        // Concatenate all inputs along batch dimension
        let concatenated = Tensor::cat(inputs, 0)?;
        let batched_output = self.forward(&concatenated)?;

        // Split output back into individual results
        let mut outputs = Vec::with_capacity(inputs.len());
        let mut start_idx = 0;

        for input in inputs {
            let batch_size = input.dims()[0];
            let output_slice = batched_output.narrow(0, start_idx, batch_size)?;
            outputs.push(output_slice);
            start_idx += batch_size;
        }

        Ok(outputs)
    }
}
```

## Error Handling and Validation

### Comprehensive Error Handling

```rust
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

impl QuantizedLinear {
    /// Validate input tensor before forward pass
    fn validate_input(&self, input: &Tensor) -> Result<(), QuantizedLinearError> {
        // Check device compatibility
        if !self.is_device_compatible(input) {
            return Err(QuantizedLinearError::DeviceMismatch {
                tensor_device: format!("{:?}", input.device()),
                layer_device: format!("{:?}", self.device),
            });
        }

        // Check shape compatibility
        let input_dims = input.dims();
        if input_dims.len() < 2 || input_dims[input_dims.len() - 1] != self.in_features {
            return Err(QuantizedLinearError::ShapeMismatch {
                input: input_dims.to_vec(),
                weight: vec![self.in_features, self.out_features],
            });
        }

        // Check for NaN/Inf values in debug mode
        #[cfg(debug_assertions)]
        {
            if self.contains_invalid_values(input)? {
                return Err(QuantizedLinearError::KernelError {
                    kernel: "input_validation".to_string(),
                    reason: "Input contains NaN or Inf values".to_string(),
                });
            }
        }

        Ok(())
    }

    /// Check for NaN/Inf values (debug builds only)
    #[cfg(debug_assertions)]
    fn contains_invalid_values(&self, tensor: &Tensor) -> Result<bool> {
        let data: Vec<f32> = tensor.flatten_all()?.to_vec1()?;
        Ok(data.iter().any(|&x| !x.is_finite()))
    }
}
```

### Accuracy Validation

```rust
impl QuantizedLinear {
    /// Validate quantization accuracy against FP32 reference
    pub fn validate_accuracy_comprehensive(&self, test_inputs: &[Tensor]) -> Result<AccuracyReport> {
        let mut correlations = Vec::new();
        let mut mse_values = Vec::new();

        // Create FP32 reference layer for comparison
        let fp32_layer = self.create_fp32_reference()?;

        for input in test_inputs {
            // Quantized forward pass
            let quantized_output = self.forward(input)?;

            // FP32 reference forward pass
            let fp32_output = fp32_layer.forward(input)?;

            // Compute metrics
            let correlation = self.compute_correlation(&quantized_output, &fp32_output)?;
            let mse = self.compute_mse(&quantized_output, &fp32_output)?;

            correlations.push(correlation);
            mse_values.push(mse);
        }

        Ok(AccuracyReport {
            mean_correlation: correlations.iter().sum::<f32>() / correlations.len() as f32,
            min_correlation: correlations.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
            mean_mse: mse_values.iter().sum::<f32>() / mse_values.len() as f32,
            max_mse: mse_values.iter().fold(0.0, |a, &b| a.max(b)),
            samples_tested: test_inputs.len(),
        })
    }

    /// Create FP32 reference layer for accuracy validation
    fn create_fp32_reference(&self) -> Result<FP32Linear> {
        // Dequantize weights back to FP32
        let fp32_weights = self.weights.dequantize()?;

        Ok(FP32Linear::new(
            fp32_weights.into_tensor()?,
            self.bias.clone(),
            self.device.clone(),
        ))
    }

    /// Compute Pearson correlation coefficient between tensors
    fn compute_correlation(&self, a: &Tensor, b: &Tensor) -> Result<f32> {
        let a_flat: Vec<f32> = a.flatten_all()?.to_vec1()?;
        let b_flat: Vec<f32> = b.flatten_all()?.to_vec1()?;

        if a_flat.len() != b_flat.len() {
            return Err(QuantizedLinearError::ShapeMismatch {
                input: vec![a_flat.len()],
                weight: vec![b_flat.len()],
            }.into());
        }

        let n = a_flat.len() as f32;
        let mean_a = a_flat.iter().sum::<f32>() / n;
        let mean_b = b_flat.iter().sum::<f32>() / n;

        let mut num = 0.0;
        let mut den_a = 0.0;
        let mut den_b = 0.0;

        for (&a_val, &b_val) in a_flat.iter().zip(b_flat.iter()) {
            let da = a_val - mean_a;
            let db = b_val - mean_b;
            num += da * db;
            den_a += da * da;
            den_b += db * db;
        }

        let correlation = num / (den_a * den_b).sqrt();
        Ok(correlation)
    }
}

#[derive(Debug, Clone)]
pub struct AccuracyReport {
    pub mean_correlation: f32,
    pub min_correlation: f32,
    pub mean_mse: f32,
    pub max_mse: f32,
    pub samples_tested: usize,
}

impl AccuracyReport {
    /// Check if accuracy meets requirements (>99% correlation, <1e-6 MSE)
    pub fn meets_requirements(&self) -> bool {
        self.mean_correlation > 0.99 && self.min_correlation > 0.95 && self.mean_mse < 1e-6
    }
}
```

## Integration with BitNet-rs Ecosystem

### GGUF Loading Integration

```rust
impl QuantizedLinear {
    /// Load quantized linear layer from GGUF tensor
    pub fn from_gguf_tensor(
        gguf_tensor: &GgufTensor,
        layer_name: &str,
        device: &Device
    ) -> Result<Self> {
        // Extract quantization metadata from GGUF
        let qtype = gguf_tensor.quantization_type()
            .ok_or_else(|| QuantizedLinearError::UnsupportedQuantization {
                qtype: QuantizationType::I2S, // Default for error
                device: format!("{:?}", device),
            })?;

        // Load quantized weights with zero-copy when possible
        let weights = if device.is_cpu() && gguf_tensor.is_memory_mapped() {
            // Zero-copy loading for CPU
            QuantizedTensor::from_memory_mapped(gguf_tensor)?
        } else {
            // Copy to device (necessary for GPU)
            gguf_tensor.to_quantized_tensor(device)?
        };

        // Load optional bias
        let bias = gguf_tensor.bias_tensor()
            .map(|bias_gguf| bias_gguf.to_tensor(device))
            .transpose()?;

        let (in_features, out_features) = gguf_tensor.linear_dimensions()?;

        let mut layer = Self {
            weights,
            bias,
            quantizer: DeviceAwareQuantizer::new(),
            qtype,
            in_features,
            out_features,
            device: device.clone(),
            workspace: None,
            scale_cache: None,
        };

        // Optimize memory layout and pre-allocate GPU workspace
        layer.optimize_memory_layout()?;

        log::info!("Loaded quantized linear layer '{}': [{} × {}] {:?} on {:?}",
                   layer_name, in_features, out_features, qtype, device);

        Ok(layer)
    }
}
```

### Cross-Validation Integration

```rust
impl QuantizedLinear {
    /// Cross-validate against C++ reference implementation
    pub fn cross_validate_against_cpp(
        &self,
        cpp_reference: &CppLinearLayer,
        test_cases: &[Tensor]
    ) -> Result<CrossValidationReport> {
        let mut reports = Vec::new();

        for (i, input) in test_cases.iter().enumerate() {
            // Rust quantized forward pass
            let rust_output = self.forward(input)?;

            // C++ reference forward pass
            let cpp_output = cpp_reference.forward(input)?;

            // Compare outputs
            let correlation = self.compute_correlation(&rust_output, &cpp_output)?;
            let mse = self.compute_mse(&rust_output, &cpp_output)?;

            reports.push(LayerValidationResult {
                test_case: i,
                correlation,
                mse,
                passed: correlation > 0.999 && mse < 1e-6,
            });
        }

        let passed_count = reports.iter().filter(|r| r.passed).count();

        Ok(CrossValidationReport {
            layer_name: "quantized_linear".to_string(),
            total_tests: test_cases.len(),
            passed_tests: passed_count,
            overall_correlation: reports.iter().map(|r| r.correlation).sum::<f32>() / reports.len() as f32,
            results: reports,
        })
    }
}

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
```

## Testing Strategy

### Unit Tests with AC Coverage

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantized_linear_i2s_accuracy() {  // AC:2
        let layer = create_test_quantized_linear(QuantizationType::I2S).unwrap();
        let input = create_test_input([2, 10, 512]).unwrap();

        let output = layer.forward(&input).unwrap();

        assert_eq!(output.shape(), [2, 10, 256]);

        // Validate accuracy against FP32 reference
        let accuracy_report = layer.validate_accuracy_comprehensive(&[input]).unwrap();
        assert!(accuracy_report.meets_requirements(),
                "I2S accuracy too low: correlation={:.4}", accuracy_report.mean_correlation);
    }

    #[test]
    fn test_device_aware_kernel_selection() {  // AC:6
        let cpu_layer = create_test_quantized_linear_cpu().unwrap();
        let selected_kernel = cpu_layer.quantizer.select_cpu_kernel(QuantizationType::I2S).unwrap();

        // Should select best available kernel for current architecture
        #[cfg(target_arch = "x86_64")]
        assert!(selected_kernel.name().contains("avx") || selected_kernel.name() == "fallback");

        #[cfg(target_arch = "aarch64")]
        assert!(selected_kernel.name().contains("neon") || selected_kernel.name() == "fallback");
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn test_gpu_acceleration() {  // AC:5
        let gpu_layer = create_test_quantized_linear_gpu().unwrap();
        let input = create_test_input([1, 100, 512]).unwrap();

        let start = std::time::Instant::now();
        let _output = gpu_layer.forward(&input).unwrap();
        let gpu_time = start.elapsed();

        let cpu_layer = gpu_layer.to_cpu().unwrap();
        let start = std::time::Instant::now();
        let _output = cpu_layer.forward(&input.to_device(&Device::Cpu).unwrap()).unwrap();
        let cpu_time = start.elapsed();

        // GPU should be faster for sufficiently large tensors
        if input.elem_count() > 10000 {
            assert!(gpu_time < cpu_time, "GPU not faster: {}ms vs {}ms",
                    gpu_time.as_millis(), cpu_time.as_millis());
        }
    }

    #[test]
    fn test_error_handling_shape_mismatch() {  // AC:10
        let layer = create_test_quantized_linear(QuantizationType::I2S).unwrap();
        let wrong_input = create_test_input([2, 10, 256]).unwrap(); // Wrong last dimension

        let result = layer.forward(&wrong_input);
        assert!(result.is_err());

        match result.unwrap_err().downcast::<QuantizedLinearError>() {
            Ok(QuantizedLinearError::ShapeMismatch { input, weight }) => {
                assert_eq!(input[2], 256);
                assert_eq!(weight[0], 512);
            }
            _ => panic!("Expected ShapeMismatch error"),
        }
    }

    #[test]
    fn test_batch_processing_optimization() {  // AC:5
        let layer = create_test_quantized_linear(QuantizationType::I2S).unwrap();
        let inputs: Vec<Tensor> = (0..10)
            .map(|_| create_test_input([1, 20, 512]).unwrap())
            .collect();

        // Process individually
        let start = std::time::Instant::now();
        let individual_outputs: Result<Vec<_>, _> = inputs.iter()
            .map(|input| layer.forward(input))
            .collect();
        let individual_time = start.elapsed();

        // Process as batch
        let start = std::time::Instant::now();
        let batch_outputs = layer.forward_batch(&inputs).unwrap();
        let batch_time = start.elapsed();

        // Results should be identical
        let individual_outputs = individual_outputs.unwrap();
        for (individual, batch) in individual_outputs.iter().zip(batch_outputs.iter()) {
            let diff = (individual - batch).unwrap().abs().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
            assert!(diff < 1e-6, "Batch processing changed results: diff={}", diff);
        }

        // Batch processing should be faster for multiple inputs
        assert!(batch_time <= individual_time * 2, "Batch processing not efficient");
    }
}
```

This specification provides a comprehensive blueprint for implementing quantized linear layers in BitNet-rs, ensuring seamless integration with the existing quantization infrastructure while delivering production-grade performance and accuracy.
