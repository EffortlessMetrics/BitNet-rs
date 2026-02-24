//! Quantization algorithms for BitNet models
//!
//! This crate provides quantization algorithms for BitNet models, including:
//! - I2_S: 2-bit signed quantization with bit-packing
//! - TL1: Table lookup quantization optimized for ARM NEON
//! - TL2: Table lookup quantization optimized for x86 AVX2/AVX-512
//!
//! All quantization methods support round-trip accuracy validation and
//! comprehensive benchmarking against reference implementations.

use bitnet_common::{BitNetTensor, QuantizationType, Result};
// Candle imports removed - not currently used

// Enable accuracy validation tests for production-ready quantization
pub mod accuracy_validation_tests;
pub mod device_aware_quantizer;
// pub mod edge_case_tests; // Temporarily disabled - needs API fixes
// pub mod error_handling_tests; // Temporarily disabled - needs API fixes
pub mod i2s;
pub mod i2s_qk256; // GGML I2_S (QK=256) scalar reference kernels
pub mod i2s_qk256_avx2; // GGML I2_S (QK=256) AVX2 SIMD kernels
pub mod property_based_tests;
#[cfg(feature = "cpu")]
pub mod qk256_dispatch; // Sprint-2 Track A PR1: QK256 SIMD dispatch scaffolding
// pub mod robustness_tests; // Keep disabled until needed
pub mod simd_ops;
pub mod tl1;
pub mod tl2;
pub mod utils;
pub mod validation;

pub use device_aware_quantizer::{
    AccuracyValidator, DeviceAwareQuantizer, QuantizationType as DeviceQuantizationType,
    ToleranceConfig,
};
pub use i2s::{I2SLayout, I2SQuantizer};
pub use tl1::TL1Quantizer;
pub use tl2::TL2Quantizer;

// Compatibility re-export: tests/benches historically used this path
pub use bitnet_common::config::QuantizationConfig;

// AC2: QK256 tolerance constant for tensor size validation (Issue #469)
/// QK256 tolerance percentage for GGUF tensor size validation.
///
/// This constant defines the acceptable deviation between expected and actual
/// tensor sizes for QK256 (GGML I2_S 256-element block) format.
///
/// **Value:** 0.001 (0.1%)
///
/// **Rationale:**
/// - Accounts for GGUF metadata padding and alignment requirements
/// - Rejects tensors with structural issues (wrong block size, corrupted data)
/// - Typical padding: 0-128 bytes for tensors in 128KB-10MB range
///
/// **Usage:**
/// - Permissive mode: Accept tensors within ±0.1% of expected size
/// - Strict mode: Accept tensors with exact size match only (tolerance = 0)
///
/// **See also:**
/// - `qk256_tolerance_bytes()` for per-tensor tolerance calculation
/// - `docs/reference/quantization-support.md` for QK256 format details
pub const QK256_SIZE_TOLERANCE_PERCENT: f64 = 0.001;

/// Calculate tolerance bytes for QK256 tensor size validation.
///
/// Computes 0.1% of expected bytes with ceiling rounding for fractional bytes.
///
/// # Arguments
/// * `expected_bytes` - Expected tensor size in bytes
///
/// # Returns
/// Tolerance in bytes (minimum 8 bytes for alignment padding)
///
/// # Examples
/// ```
/// use bitnet_quantization::qk256_tolerance_bytes;
///
/// assert_eq!(qk256_tolerance_bytes(1_000_000), 1000);  // 1 MB → 1 KB tolerance
/// assert_eq!(qk256_tolerance_bytes(131_072), 132);     // 128 KB → 132 bytes (ceiling)
/// assert_eq!(qk256_tolerance_bytes(100_000), 100);     // 100 KB → 100 bytes
/// assert_eq!(qk256_tolerance_bytes(1_000), 8);         // 1 KB → 8 bytes (minimum)
/// assert_eq!(qk256_tolerance_bytes(20), 8);            // 20 bytes → 8 bytes (minimum)
/// ```
pub fn qk256_tolerance_bytes(expected_bytes: usize) -> usize {
    let tolerance = (expected_bytes as f64) * QK256_SIZE_TOLERANCE_PERCENT;
    // Ceiling rounding ensures fractional bytes round up, minimum 8 bytes for alignment padding
    tolerance.ceil().max(8.0) as usize
}

/// Quantization trait for tensor quantization and dequantization operations
pub trait Quantize {
    /// Quantize a tensor using the specified quantization type
    fn quantize(&self, qtype: QuantizationType) -> Result<QuantizedTensor>;

    /// Dequantize back to a full precision tensor
    fn dequantize(&self) -> Result<BitNetTensor>;
}

/// Quantized tensor representation with compressed data and metadata
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Compressed quantized data
    pub data: Vec<u8>,
    /// Scale factors for dequantization
    pub scales: Vec<f32>,
    /// Zero points for asymmetric quantization (if needed)
    pub zero_points: Option<Vec<i32>>,
    /// Original tensor shape
    pub shape: Vec<usize>,
    /// Quantization type used
    pub qtype: QuantizationType,
    /// Block size for grouped quantization
    pub block_size: usize,
}

impl QuantizedTensor {
    /// Create a new quantized tensor
    pub fn new(
        data: Vec<u8>,
        scales: Vec<f32>,
        shape: Vec<usize>,
        qtype: QuantizationType,
    ) -> Self {
        Self {
            data,
            scales,
            zero_points: None,
            shape,
            qtype,
            block_size: 32, // Default block size
        }
    }

    /// Create a new quantized tensor with all parameters
    pub fn new_with_params(
        data: Vec<u8>,
        scales: Vec<f32>,
        zero_points: Option<Vec<i32>>,
        shape: Vec<usize>,
        qtype: QuantizationType,
        block_size: usize,
    ) -> Self {
        Self { data, scales, zero_points, shape, qtype, block_size }
    }

    /// Get the number of elements in the original tensor
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get the compression ratio compared to FP32
    pub fn compression_ratio(&self) -> f32 {
        let original_bytes = self.numel() * 4; // FP32 = 4 bytes per element
        let compressed_bytes = self.data.len() + self.scales.len() * 4;
        if compressed_bytes == 0 {
            1.0 // Avoid division by zero
        } else {
            (original_bytes as f32 / compressed_bytes as f32).max(1.0)
        }
    }
}

impl Quantize for QuantizedTensor {
    fn quantize(&self, qtype: QuantizationType) -> Result<QuantizedTensor> {
        if self.qtype == qtype {
            return Ok(self.clone());
        }

        // Convert between quantization formats by dequantizing and re-quantizing
        let dequantized = self.dequantize()?;
        dequantized.quantize(qtype)
    }

    fn dequantize(&self) -> Result<BitNetTensor> {
        match self.qtype {
            QuantizationType::I2S => I2SQuantizer::new().dequantize_tensor(self),
            QuantizationType::TL1 => TL1Quantizer::new().dequantize_tensor(self),
            QuantizationType::TL2 => TL2Quantizer::new().dequantize_tensor(self),
        }
    }
}

impl Quantize for BitNetTensor {
    fn quantize(&self, qtype: QuantizationType) -> Result<QuantizedTensor> {
        match qtype {
            QuantizationType::I2S => I2SQuantizer::new().quantize_tensor(self),
            QuantizationType::TL1 => TL1Quantizer::new().quantize_tensor(self),
            QuantizationType::TL2 => TL2Quantizer::new().quantize_tensor(self),
        }
    }

    fn dequantize(&self) -> Result<BitNetTensor> {
        // Already dequantized
        Ok(self.clone())
    }
}

/// Quantizer factory for creating appropriate quantizers
pub struct QuantizerFactory;

impl QuantizerFactory {
    /// Create a quantizer for the specified type
    pub fn create(qtype: QuantizationType) -> Box<dyn QuantizerTrait> {
        match qtype {
            QuantizationType::I2S => Box::new(I2SQuantizer::new()),
            QuantizationType::TL1 => Box::new(TL1Quantizer::new()),
            QuantizationType::TL2 => Box::new(TL2Quantizer::new()),
        }
    }

    /// Get the best quantization type for the current architecture
    pub fn best_for_arch() -> QuantizationType {
        #[cfg(target_arch = "aarch64")]
        {
            QuantizationType::TL1
        }
        #[cfg(target_arch = "x86_64")]
        {
            QuantizationType::TL2
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            QuantizationType::I2S
        }
    }
}

/// Trait for quantizer implementations
pub trait QuantizerTrait: Send + Sync {
    /// Quantize a tensor
    fn quantize_tensor(&self, tensor: &BitNetTensor) -> Result<QuantizedTensor>;

    /// Dequantize a tensor
    fn dequantize_tensor(&self, tensor: &QuantizedTensor) -> Result<BitNetTensor>;

    /// Get the quantization type
    fn quantization_type(&self) -> QuantizationType;

    /// Check if this quantizer is available on the current platform
    fn is_available(&self) -> bool {
        true
    }
}

/// Convert between different quantization formats
pub fn convert_quantization(
    tensor: &QuantizedTensor,
    target_qtype: QuantizationType,
) -> Result<QuantizedTensor> {
    if tensor.qtype == target_qtype {
        return Ok(tensor.clone());
    }

    // Dequantize and re-quantize
    let dequantized = tensor.dequantize()?;
    dequantized.quantize(target_qtype)
}

/// Validate quantization round-trip accuracy
pub fn validate_round_trip(
    original: &BitNetTensor,
    qtype: QuantizationType,
    tolerance: f32,
) -> Result<bool> {
    let quantized = original.quantize(qtype)?;
    let dequantized = quantized.dequantize()?;

    let original_data = original.to_vec()?;
    let deq_data = dequantized.to_vec()?;

    if original_data.len() != deq_data.len() {
        return Ok(false);
    }

    let max_abs_err = original_data
        .iter()
        .zip(deq_data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    Ok(max_abs_err <= tolerance)
}
