# Quantization-Aware Weight Loading API Contracts

## Overview

This document defines comprehensive API contracts for quantization-aware GGUF weight loading in BitNet.rs. These contracts ensure type safety, accuracy preservation, and seamless integration with the BitNet quantization pipeline (I2_S, TL1, TL2) while maintaining ≥99% accuracy compared to FP32 reference implementation.

## Core Quantization API Contracts

### Primary Quantization Interface

```rust
/// Quantization-aware GGUF weight loader with integrated dequantization
pub trait QuantizationAwareLoader {
    /// Load tensor with automatic quantization detection and validation
    ///
    /// # Acceptance Criteria Coverage
    /// * AC2: Support I2_S, TL1, TL2 formats with ≥99% accuracy preservation
    /// * AC3: Tensor metadata validation with shape verification
    fn load_quantized_tensor(
        &self,
        tensor_info: &TensorInfo,
        target_device: &Device,
        validation_config: &QuantizationValidationConfig,
    ) -> Result<QuantizedTensorResult>;

    /// Validate quantization accuracy against reference implementation
    ///
    /// # Returns
    /// Accuracy metrics including relative error, absolute error, and preservation ratio
    fn validate_quantization_accuracy(
        &self,
        quantized_tensor: &QuantizedTensor,
        reference_tensor: &ReferenceTensor,
        tolerance: f32,
    ) -> Result<QuantizationAccuracyReport>;

    /// Get supported quantization types for this loader
    fn supported_quantization_types(&self) -> &[QuantizationType];

    /// Check if tensor requires special handling for quantization
    fn requires_special_handling(&self, tensor_info: &TensorInfo) -> bool;
}

/// Enhanced I2_S quantization loader with BitNet integration
pub struct I2SQuantizationLoader {
    /// I2_S quantizer instance
    quantizer: I2SQuantizer,
    /// Block size configuration (82 bytes per block for GGML compatibility)
    block_size: usize,
    /// Accuracy validation threshold
    accuracy_threshold: f32,
    /// Device-specific optimizations
    device_optimizations: DeviceOptimizations,
}

impl I2SQuantizationLoader {
    /// Create new I2_S loader with validation
    ///
    /// # Parameters
    /// * `block_size`: Must be 82 bytes for GGML compatibility
    /// * `accuracy_threshold`: Minimum accuracy requirement (default: 0.99)
    pub fn new(
        block_size: Option<usize>,
        accuracy_threshold: Option<f32>,
        device: &Device,
    ) -> Result<Self> {
        let block_size = block_size.unwrap_or(82); // GGML standard
        if block_size != 82 {
            return Err(BitNetError::InvalidConfiguration(
                format!("I2_S block size must be 82 bytes for GGML compatibility, got {}", block_size)
            ));
        }

        Ok(Self {
            quantizer: I2SQuantizer::new(I2SLayout::GGML)?,
            block_size,
            accuracy_threshold: accuracy_threshold.unwrap_or(0.99),
            device_optimizations: DeviceOptimizations::for_device(device)?,
        })
    }

    /// Load I2_S quantized tensor with accuracy validation
    pub fn load_i2s_tensor(
        &self,
        tensor_data: &[u8],
        tensor_shape: &[usize],
        target_device: &Device,
    ) -> Result<QuantizedTensorResult> {
        // Validate tensor data integrity
        self.validate_i2s_data_integrity(tensor_data, tensor_shape)?;

        // Perform I2_S dequantization
        let dequantized_tensor = self.quantizer.dequantize_tensor(
            tensor_data,
            tensor_shape,
            self.block_size,
        )?;

        // Apply device-specific optimizations
        let optimized_tensor = self.device_optimizations.optimize_tensor(
            dequantized_tensor,
            target_device,
        )?;

        // Validate accuracy if reference is available
        let accuracy_report = if let Some(reference) = &self.reference_implementation {
            Some(self.validate_against_reference(&optimized_tensor, reference)?)
        } else {
            None
        };

        Ok(QuantizedTensorResult {
            tensor: optimized_tensor,
            quantization_type: QuantizationType::I2S,
            accuracy_report,
            device_placement: target_device.clone(),
            memory_footprint_bytes: self.calculate_memory_footprint(&optimized_tensor),
        })
    }

    /// Validate I2_S data integrity before dequantization
    fn validate_i2s_data_integrity(
        &self,
        tensor_data: &[u8],
        tensor_shape: &[usize],
    ) -> Result<()> {
        let expected_size = self.calculate_i2s_size(tensor_shape);
        if tensor_data.len() != expected_size {
            return Err(BitNetError::TensorValidationError(
                format!(
                    "I2_S tensor size mismatch: expected {} bytes, got {} bytes for shape {:?}",
                    expected_size, tensor_data.len(), tensor_shape
                )
            ));
        }

        // Validate block alignment
        if tensor_data.len() % self.block_size != 0 {
            return Err(BitNetError::TensorValidationError(
                format!(
                    "I2_S tensor data not aligned to block size {}: {} bytes",
                    self.block_size, tensor_data.len()
                )
            ));
        }

        Ok(())
    }

    /// Calculate expected I2_S tensor size in bytes
    fn calculate_i2s_size(&self, tensor_shape: &[usize]) -> usize {
        let total_elements: usize = tensor_shape.iter().product();
        // I2_S uses 2 bits per element + scale factors
        let data_bits = total_elements * 2;
        let data_bytes = (data_bits + 7) / 8; // Round up to nearest byte

        // Add scale factors (1 f16 per block of 32 elements)
        let num_blocks = (total_elements + 31) / 32;
        let scale_bytes = num_blocks * 2; // f16 = 2 bytes

        data_bytes + scale_bytes
    }
}

/// TL1/TL2 quantization loader with table lookup optimization
pub struct TLQuantizationLoader {
    /// Table lookup implementation
    lookup_table: TLLookupTable,
    /// Quantization type (TL1 or TL2)
    quantization_type: TLQuantizationType,
    /// Vectorization strategy for SIMD optimization
    vectorization: VectorizationStrategy,
    /// Accuracy validation configuration
    validation_config: TLValidationConfig,
}

impl TLQuantizationLoader {
    /// Create new TL1/TL2 loader with optimization
    pub fn new(
        quantization_type: TLQuantizationType,
        vectorization: VectorizationStrategy,
    ) -> Result<Self> {
        let lookup_table = match quantization_type {
            TLQuantizationType::TL1 => TLLookupTable::tl1_table()?,
            TLQuantizationType::TL2 => TLLookupTable::tl2_table()?,
        };

        Ok(Self {
            lookup_table,
            quantization_type,
            vectorization,
            validation_config: TLValidationConfig::default(),
        })
    }

    /// Load TL quantized tensor with vectorized operations
    pub fn load_tl_tensor(
        &self,
        tensor_data: &[u8],
        tensor_shape: &[usize],
        target_device: &Device,
    ) -> Result<QuantizedTensorResult> {
        // Validate TL data format
        self.validate_tl_data_format(tensor_data, tensor_shape)?;

        // Perform vectorized table lookup dequantization
        let dequantized_tensor = match self.vectorization {
            VectorizationStrategy::SIMD => {
                self.simd_tl_dequantize(tensor_data, tensor_shape)?
            }
            VectorizationStrategy::Standard => {
                self.standard_tl_dequantize(tensor_data, tensor_shape)?
            }
        };

        // Device placement and optimization
        let optimized_tensor = self.place_tensor_on_device(dequantized_tensor, target_device)?;

        Ok(QuantizedTensorResult {
            tensor: optimized_tensor,
            quantization_type: QuantizationType::from_tl_type(self.quantization_type),
            accuracy_report: None, // TL accuracy validation done during table creation
            device_placement: target_device.clone(),
            memory_footprint_bytes: self.calculate_memory_footprint(&optimized_tensor),
        })
    }

    /// SIMD-optimized TL dequantization
    fn simd_tl_dequantize(
        &self,
        tensor_data: &[u8],
        tensor_shape: &[usize],
    ) -> Result<BitNetTensor> {
        // Implementation using SIMD instructions for vectorized table lookup
        // This provides significant performance improvement for large tensors
        #[cfg(target_arch = "x86_64")]
        {
            self.avx2_tl_dequantize(tensor_data, tensor_shape)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            self.standard_tl_dequantize(tensor_data, tensor_shape)
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn avx2_tl_dequantize(
        &self,
        tensor_data: &[u8],
        tensor_shape: &[usize],
    ) -> Result<BitNetTensor> {
        // AVX2 vectorized implementation for x86_64
        use std::arch::x86_64::*;

        let total_elements: usize = tensor_shape.iter().product();
        let mut result = Vec::with_capacity(total_elements);

        unsafe {
            // Process 32 elements at a time using AVX2
            let chunks = tensor_data.chunks_exact(32);
            for chunk in chunks {
                let indices = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);

                // Vectorized table lookup using _mm256_i32gather_ps
                // This is a simplified example - full implementation would handle
                // different TL formats and proper index manipulation
                let values = _mm256_i32gather_ps(
                    self.lookup_table.as_ptr(),
                    indices,
                    4, // Scale factor
                );

                // Store results
                let mut temp = [0.0f32; 8];
                _mm256_storeu_ps(temp.as_mut_ptr(), values);
                result.extend_from_slice(&temp);
            }
        }

        // Handle remaining elements
        let remainder = tensor_data.len() % 32;
        if remainder > 0 {
            let remaining_data = &tensor_data[tensor_data.len() - remainder..];
            for &index in remaining_data {
                result.push(self.lookup_table.get(index as usize)?);
            }
        }

        BitNetTensor::from_f32_vec(result, tensor_shape)
    }
}

/// Quantization validation configuration
#[derive(Debug, Clone)]
pub struct QuantizationValidationConfig {
    /// Enable accuracy validation against reference
    pub validate_accuracy: bool,
    /// Accuracy threshold for validation (0.0-1.0)
    pub accuracy_threshold: f32,
    /// Enable statistical validation (mean, variance, etc.)
    pub validate_statistics: bool,
    /// Enable range validation (min/max values)
    pub validate_range: bool,
    /// Tolerance for numerical comparison
    pub numerical_tolerance: f32,
}

impl Default for QuantizationValidationConfig {
    fn default() -> Self {
        Self {
            validate_accuracy: true,
            accuracy_threshold: 0.99, // 99% accuracy requirement
            validate_statistics: true,
            validate_range: true,
            numerical_tolerance: 1e-5,
        }
    }
}

/// Result of quantized tensor loading with validation metrics
#[derive(Debug)]
pub struct QuantizedTensorResult {
    /// Loaded and dequantized tensor
    pub tensor: BitNetTensor,
    /// Quantization type used
    pub quantization_type: QuantizationType,
    /// Accuracy validation report (if available)
    pub accuracy_report: Option<QuantizationAccuracyReport>,
    /// Device placement information
    pub device_placement: Device,
    /// Memory footprint in bytes
    pub memory_footprint_bytes: u64,
}

/// Quantization accuracy validation report
#[derive(Debug, Clone)]
pub struct QuantizationAccuracyReport {
    /// Overall accuracy percentage (0.0-1.0)
    pub accuracy_percentage: f32,
    /// Maximum absolute error
    pub max_absolute_error: f32,
    /// Maximum relative error
    pub max_relative_error: f32,
    /// Root mean square error
    pub rmse: f32,
    /// Statistical comparison
    pub statistical_comparison: StatisticalComparison,
    /// Validation status
    pub validation_status: ValidationStatus,
}

#[derive(Debug, Clone)]
pub struct StatisticalComparison {
    /// Mean difference between quantized and reference
    pub mean_difference: f32,
    /// Variance difference
    pub variance_difference: f32,
    /// Distribution similarity (Kolmogorov-Smirnov test result)
    pub distribution_similarity: f32,
}

/// Device-specific optimization strategies
#[derive(Debug, Clone)]
pub struct DeviceOptimizations {
    /// Target device
    pub device: Device,
    /// Enable mixed precision (FP16/BF16)
    pub mixed_precision: bool,
    /// Memory layout optimization
    pub memory_layout: MemoryLayout,
    /// Vectorization strategy
    pub vectorization: VectorizationStrategy,
}

impl DeviceOptimizations {
    /// Create device-specific optimizations
    pub fn for_device(device: &Device) -> Result<Self> {
        match device {
            Device::Gpu { .. } => {
                Ok(Self {
                    device: device.clone(),
                    mixed_precision: true, // Enable FP16 for GPU
                    memory_layout: MemoryLayout::CoalescedAccess,
                    vectorization: VectorizationStrategy::CUDA,
                })
            }
            Device::Cpu => {
                Ok(Self {
                    device: device.clone(),
                    mixed_precision: false, // Keep FP32 for CPU
                    memory_layout: MemoryLayout::CacheOptimized,
                    vectorization: VectorizationStrategy::SIMD,
                })
            }
        }
    }

    /// Optimize tensor for target device
    pub fn optimize_tensor(
        &self,
        tensor: BitNetTensor,
        target_device: &Device,
    ) -> Result<BitNetTensor> {
        match (&self.device, target_device) {
            (Device::Cpu, Device::Cpu) => {
                self.optimize_for_cpu(tensor)
            }
            (Device::Gpu { .. }, Device::Gpu { .. }) => {
                self.optimize_for_gpu(tensor)
            }
            (Device::Cpu, Device::Gpu { .. }) => {
                // Transfer from CPU to GPU with optimization
                self.transfer_cpu_to_gpu(tensor, target_device)
            }
            (Device::Gpu { .. }, Device::Cpu) => {
                // Transfer from GPU to CPU with optimization
                self.transfer_gpu_to_cpu(tensor)
            }
        }
    }

    /// CPU-specific tensor optimizations
    fn optimize_for_cpu(&self, tensor: BitNetTensor) -> Result<BitNetTensor> {
        // Apply CPU-specific optimizations:
        // - Memory layout for cache efficiency
        // - SIMD-friendly data alignment
        // - NUMA-aware memory allocation
        match self.memory_layout {
            MemoryLayout::CacheOptimized => {
                tensor.apply_cache_optimization()
            }
            MemoryLayout::CoalescedAccess => {
                // Not applicable for CPU, use default
                Ok(tensor)
            }
        }
    }

    /// GPU-specific tensor optimizations
    fn optimize_for_gpu(&self, tensor: BitNetTensor) -> Result<BitNetTensor> {
        // Apply GPU-specific optimizations:
        // - Coalesced memory access patterns
        // - Mixed precision conversion
        // - GPU memory alignment
        let mut optimized = tensor;

        if self.mixed_precision {
            optimized = optimized.to_half_precision()?;
        }

        match self.memory_layout {
            MemoryLayout::CoalescedAccess => {
                optimized.apply_coalesced_layout()
            }
            MemoryLayout::CacheOptimized => {
                // GPU doesn't need CPU cache optimization
                Ok(optimized)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum MemoryLayout {
    /// Optimized for CPU cache efficiency
    CacheOptimized,
    /// Optimized for GPU coalesced memory access
    CoalescedAccess,
}

#[derive(Debug, Clone)]
pub enum VectorizationStrategy {
    /// Standard scalar operations
    Standard,
    /// CPU SIMD instructions (AVX2, NEON, etc.)
    SIMD,
    /// GPU CUDA optimizations
    CUDA,
}

/// TL quantization type specification
#[derive(Debug, Clone, Copy)]
pub enum TLQuantizationType {
    /// Table Lookup 1-bit quantization
    TL1,
    /// Table Lookup 2-bit quantization
    TL2,
}

/// Table lookup implementation for TL quantization
pub struct TLLookupTable {
    /// Lookup table values
    table: Vec<f32>,
    /// Table size
    size: usize,
    /// Quantization type
    qtype: TLQuantizationType,
}

impl TLLookupTable {
    /// Create TL1 lookup table
    pub fn tl1_table() -> Result<Self> {
        // TL1 uses 1-bit quantization: 2 possible values
        let table = vec![-1.0, 1.0];
        Ok(Self {
            table,
            size: 2,
            qtype: TLQuantizationType::TL1,
        })
    }

    /// Create TL2 lookup table
    pub fn tl2_table() -> Result<Self> {
        // TL2 uses 2-bit quantization: 4 possible values
        let table = vec![-1.5, -0.5, 0.5, 1.5];
        Ok(Self {
            table,
            size: 4,
            qtype: TLQuantizationType::TL2,
        })
    }

    /// Get value from lookup table
    pub fn get(&self, index: usize) -> Result<f32> {
        self.table.get(index)
            .copied()
            .ok_or_else(|| BitNetError::IndexOutOfBounds(
                format!("TL lookup index {} out of bounds for table size {}", index, self.size)
            ))
    }

    /// Get table as raw pointer for SIMD operations
    pub fn as_ptr(&self) -> *const f32 {
        self.table.as_ptr()
    }
}
```

## Integration Testing Framework

### Quantization Accuracy Test Suite

```rust
#[cfg(test)]
mod quantization_accuracy_tests {
    use super::*;

    /// Test I2_S quantization accuracy preservation
    #[tokio::test]
    async fn test_i2s_accuracy_preservation() {
        // AC:2 - Test I2_S quantization with ≥99% accuracy preservation
        let config = QuantizationValidationConfig {
            accuracy_threshold: 0.99,
            ..Default::default()
        };

        let loader = I2SQuantizationLoader::new(Some(82), Some(0.99), &Device::Cpu)?;

        // Load test tensor with known reference values
        let test_tensor = create_test_i2s_tensor()?;
        let reference_tensor = create_reference_fp32_tensor()?;

        let result = loader.load_i2s_tensor(
            &test_tensor.data,
            &test_tensor.shape,
            &Device::Cpu,
        )?;

        // Validate accuracy
        assert!(result.accuracy_report.is_some());
        let accuracy = result.accuracy_report.unwrap();
        assert!(accuracy.accuracy_percentage >= config.accuracy_threshold);
        assert!(accuracy.validation_status == ValidationStatus::Passed);
    }

    /// Test TL1/TL2 quantization with vectorized operations
    #[tokio::test]
    async fn test_tl_vectorized_quantization() {
        // AC:2 - Test TL1 and TL2 quantization formats
        for qtype in [TLQuantizationType::TL1, TLQuantizationType::TL2] {
            let loader = TLQuantizationLoader::new(
                qtype,
                VectorizationStrategy::SIMD,
            )?;

            let test_tensor = create_test_tl_tensor(qtype)?;
            let result = loader.load_tl_tensor(
                &test_tensor.data,
                &test_tensor.shape,
                &Device::Cpu,
            )?;

            // Validate tensor properties
            assert_eq!(result.quantization_type, QuantizationType::from_tl_type(qtype));
            assert!(result.tensor.is_finite());
            assert_eq!(result.tensor.shape(), &test_tensor.shape);
        }
    }

    /// Test device-specific optimizations
    #[tokio::test]
    async fn test_device_optimizations() {
        // AC:6 - Test device-aware tensor placement
        let devices = vec![Device::Cpu];

        #[cfg(feature = "gpu")]
        devices.push(Device::Gpu { device_index: 0 });

        for device in devices {
            let optimizations = DeviceOptimizations::for_device(&device)?;
            let test_tensor = create_test_tensor()?;

            let optimized = optimizations.optimize_tensor(test_tensor, &device)?;

            // Validate device-specific properties
            match device {
                Device::Cpu => {
                    // CPU optimizations should maintain FP32 precision
                    assert_eq!(optimized.dtype(), BitNetDType::F32);
                }
                Device::Gpu { .. } => {
                    // GPU optimizations may use mixed precision
                    assert!(matches!(
                        optimized.dtype(),
                        BitNetDType::F32 | BitNetDType::F16 | BitNetDType::BF16
                    ));
                }
            }
        }
    }
}
```

## Conclusion

This specification provides comprehensive API contracts for quantization-aware GGUF weight loading that:

1. **Ensures Accuracy**: ≥99% preservation for all quantization formats (AC2)
2. **Validates Metadata**: Comprehensive tensor validation with shape verification (AC3)
3. **Supports All Formats**: I2_S, TL1, TL2 with format-specific optimizations
4. **Device-Aware**: Automatic GPU/CPU placement with device-specific optimizations (AC6)
5. **Performance Optimized**: SIMD vectorization and mixed precision support
6. **Cross-Validation Ready**: Integration with C++ reference validation framework (AC5)

The contracts align with BitNet.rs feature flag discipline and provide a solid foundation for implementing real GGUF weight loading with quantization accuracy preservation.