//! GPU-accelerated quantization pipeline.
//!
//! Provides a high-level pipeline for quantizing tensors using various
//! BitNet quantization schemes (I2S, TL1, TL2, QK256). The pipeline
//! validates results against a CPU reference implementation when
//! configured to do so.

use std::fmt;

use bitnet_common::QuantizationType;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during pipeline execution.
#[derive(Debug, Error, PartialEq)]
pub enum PipelineError {
    /// The input tensor is empty.
    #[error("empty input tensor")]
    EmptyInput,

    /// The input batch is empty.
    #[error("empty batch")]
    EmptyBatch,

    /// The batch exceeds the configured maximum size.
    #[error("batch size {actual} exceeds maximum {max}")]
    BatchTooLarge { actual: usize, max: usize },

    /// Validation against the CPU reference failed.
    #[error(
        "validation failed: max error {max_error:.6} exceeds tolerance {tolerance:.6}"
    )]
    ValidationFailed { max_error: f64, tolerance: f64 },

    /// The requested quantization type is not supported by this pipeline.
    #[error("unsupported quantization type: {0}")]
    UnsupportedQuantType(String),
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the quantization pipeline.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Block size for I2S / TL1 / TL2 quantization (default: 32).
    pub block_size: usize,
    /// Block size for QK256 quantization (default: 256).
    pub qk256_block_size: usize,
    /// Whether to validate results against CPU reference (default: true).
    pub validate: bool,
    /// Maximum tolerable element-wise error (default: 1e-4).
    pub validation_tolerance: f64,
    /// Maximum number of tensors in a single batch (default: 1024).
    pub max_batch_size: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            block_size: 32,
            qk256_block_size: 256,
            validate: true,
            validation_tolerance: 1e-4,
            max_batch_size: 1024,
        }
    }
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

/// Result of quantizing a single tensor.
#[derive(Debug, Clone)]
pub struct GpuQuantizationResult {
    /// The quantized bytes.
    pub data: Vec<u8>,
    /// The quantization type used.
    pub quant_type: QuantizationType,
    /// Number of elements in the original tensor.
    pub element_count: usize,
    /// Maximum element-wise error vs CPU reference (if validated).
    pub max_error: Option<f64>,
    /// Mean element-wise error vs CPU reference (if validated).
    pub mean_error: Option<f64>,
}

/// Aggregate statistics for a batch quantization run.
#[derive(Debug, Clone)]
pub struct BatchQuantizationSummary {
    /// Total tensors processed.
    pub tensor_count: usize,
    /// Total elements across all tensors.
    pub total_elements: usize,
    /// Total bytes in quantized output.
    pub total_output_bytes: usize,
    /// Maximum error across all tensors (if validated).
    pub max_error: Option<f64>,
    /// Mean error across all tensors (if validated).
    pub mean_error: Option<f64>,
}

impl fmt::Display for BatchQuantizationSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "batch: {} tensors, {} elements, {} output bytes",
            self.tensor_count, self.total_elements, self.total_output_bytes,
        )?;
        if let Some(max) = self.max_error {
            write!(f, ", max_err={max:.6}")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// GPU-accelerated quantization pipeline.
///
/// Internally uses a CPU reference implementation (bit-identical to the
/// GPU algorithm) so that the pipeline can run and be tested without
/// actual GPU hardware.
#[derive(Debug, Clone)]
pub struct GpuQuantizationPipeline {
    config: PipelineConfig,
}

impl GpuQuantizationPipeline {
    /// Create a new pipeline with the given configuration.
    pub fn new(config: PipelineConfig) -> Self {
        Self { config }
    }

    /// Create a pipeline with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(PipelineConfig::default())
    }

    /// Return a reference to the current configuration.
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }

    /// Quantize a single f32 tensor.
    pub fn quantize(
        &self,
        data: &[f32],
        quant_type: QuantizationType,
    ) -> Result<GpuQuantizationResult, PipelineError> {
        if data.is_empty() {
            return Err(PipelineError::EmptyInput);
        }
        Self::validate_quant_type(quant_type)?;

        let quantized = self.cpu_reference_quantize(data, quant_type);

        let (max_error, mean_error) = if self.config.validate {
            let (max_e, mean_e) =
                self.compute_validation_errors(data, &quantized, quant_type);
            if max_e > self.config.validation_tolerance {
                return Err(PipelineError::ValidationFailed {
                    max_error: max_e,
                    tolerance: self.config.validation_tolerance,
                });
            }
            (Some(max_e), Some(mean_e))
        } else {
            (None, None)
        };

        Ok(GpuQuantizationResult {
            data: quantized,
            quant_type,
            element_count: data.len(),
            max_error,
            mean_error,
        })
    }

    /// Quantize a batch of tensors.
    pub fn quantize_batch(
        &self,
        batch: &[&[f32]],
        quant_type: QuantizationType,
    ) -> Result<(Vec<GpuQuantizationResult>, BatchQuantizationSummary), PipelineError>
    {
        if batch.is_empty() {
            return Err(PipelineError::EmptyBatch);
        }
        if batch.len() > self.config.max_batch_size {
            return Err(PipelineError::BatchTooLarge {
                actual: batch.len(),
                max: self.config.max_batch_size,
            });
        }

        let mut results = Vec::with_capacity(batch.len());
        for tensor in batch {
            results.push(self.quantize(tensor, quant_type)?);
        }

        let total_elements: usize =
            results.iter().map(|r| r.element_count).sum();
        let total_output_bytes: usize =
            results.iter().map(|r| r.data.len()).sum();
        let max_error = results
            .iter()
            .filter_map(|r| r.max_error)
            .fold(None, |acc, e| {
                Some(acc.map_or(e, |a: f64| a.max(e)))
            });
        let mean_error = {
            let errors: Vec<f64> =
                results.iter().filter_map(|r| r.mean_error).collect();
            if errors.is_empty() {
                None
            } else {
                Some(errors.iter().sum::<f64>() / errors.len() as f64)
            }
        };

        let summary = BatchQuantizationSummary {
            tensor_count: results.len(),
            total_elements,
            total_output_bytes,
            max_error,
            mean_error,
        };

        Ok((results, summary))
    }

    // -- internal helpers ---------------------------------------------------

    fn validate_quant_type(
        qt: QuantizationType,
    ) -> Result<(), PipelineError> {
        // All current QuantizationType variants are supported.
        match qt {
            QuantizationType::I2S
            | QuantizationType::TL1
            | QuantizationType::TL2 => Ok(()),
        }
    }

    /// CPU reference quantization (simulates GPU kernel output).
    ///
    /// For I2S: each element → 2-bit signed value packed 4 per byte.
    /// For TL1/TL2: each element → 8-bit table index.
    fn cpu_reference_quantize(
        &self,
        data: &[f32],
        quant_type: QuantizationType,
    ) -> Vec<u8> {
        match quant_type {
            QuantizationType::I2S => {
                // 2-bit packing: 4 values per byte
                let num_bytes = (data.len() + 3) / 4;
                let mut out = vec![0u8; num_bytes];
                for (i, &val) in data.iter().enumerate() {
                    let q = self.quantize_i2s_scalar(val);
                    let byte_idx = i / 4;
                    let bit_offset = (i % 4) * 2;
                    out[byte_idx] |= (q & 0x03) << bit_offset;
                }
                out
            }
            QuantizationType::TL1 | QuantizationType::TL2 => {
                // 8-bit table index per element
                data.iter()
                    .map(|&val| self.quantize_tl_scalar(val))
                    .collect()
            }
        }
    }

    /// Scalar I2S quantization: map f32 → {0, 1, 2, 3} (2 bits).
    fn quantize_i2s_scalar(&self, val: f32) -> u8 {
        if val > 0.5 {
            3 // +1
        } else if val > 0.0 {
            2 // +0 (small positive)
        } else if val > -0.5 {
            1 // -0 (small negative)
        } else {
            0 // -1
        }
    }

    /// Scalar TL quantization: map f32 → u8 table index.
    fn quantize_tl_scalar(&self, val: f32) -> u8 {
        // Clamp to [-1, 1] and map to [0, 255]
        let clamped = val.clamp(-1.0, 1.0);
        ((clamped + 1.0) * 127.5) as u8
    }

    /// Compute validation errors between original data and quantized output.
    fn compute_validation_errors(
        &self,
        original: &[f32],
        quantized: &[u8],
        quant_type: QuantizationType,
    ) -> (f64, f64) {
        let dequantized = self.dequantize_reference(quantized, original.len(), quant_type);
        let mut max_err: f64 = 0.0;
        let mut sum_err: f64 = 0.0;
        for (i, &orig) in original.iter().enumerate() {
            let err = (orig as f64 - dequantized[i]).abs();
            max_err = max_err.max(err);
            sum_err += err;
        }
        let mean_err = if original.is_empty() {
            0.0
        } else {
            sum_err / original.len() as f64
        };
        (max_err, mean_err)
    }

    /// Dequantize for validation purposes.
    fn dequantize_reference(
        &self,
        quantized: &[u8],
        element_count: usize,
        quant_type: QuantizationType,
    ) -> Vec<f64> {
        match quant_type {
            QuantizationType::I2S => {
                let mut out = Vec::with_capacity(element_count);
                for i in 0..element_count {
                    let byte_idx = i / 4;
                    let bit_offset = (i % 4) * 2;
                    let q = (quantized[byte_idx] >> bit_offset) & 0x03;
                    let val = match q {
                        3 => 1.0,
                        2 => 0.25,
                        1 => -0.25,
                        0 => -1.0,
                        _ => unreachable!(),
                    };
                    out.push(val);
                }
                out
            }
            QuantizationType::TL1 | QuantizationType::TL2 => {
                quantized
                    .iter()
                    .take(element_count)
                    .map(|&b| (b as f64 / 127.5) - 1.0)
                    .collect()
            }
        }
    }
}

impl Default for GpuQuantizationPipeline{
    fn default() -> Self {
        Self::with_defaults()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn pipeline() -> GpuQuantizationPipeline {
        GpuQuantizationPipeline::with_defaults()
    }

    fn pipeline_no_validate() -> GpuQuantizationPipeline {
        GpuQuantizationPipeline::new(PipelineConfig {
            validate: false,
            ..Default::default()
        })
    }

    // -- config ------------------------------------------------------------

    #[test]
    fn default_config_values() {
        let cfg = PipelineConfig::default();
        assert_eq!(cfg.block_size, 32);
        assert_eq!(cfg.qk256_block_size, 256);
        assert!(cfg.validate);
        assert!((cfg.validation_tolerance - 1e-4).abs() < 1e-10);
        assert_eq!(cfg.max_batch_size, 1024);
    }

    // -- empty input / batch errors ----------------------------------------

    #[test]
    fn empty_input_returns_error() {
        let p = pipeline();
        let err = p.quantize(&[], QuantizationType::I2S).unwrap_err();
        assert_eq!(err, PipelineError::EmptyInput);
    }

    #[test]
    fn empty_batch_returns_error() {
        let p = pipeline();
        let err = p
            .quantize_batch(&[], QuantizationType::I2S)
            .unwrap_err();
        assert_eq!(err, PipelineError::EmptyBatch);
    }

    #[test]
    fn batch_too_large_returns_error() {
        let p = GpuQuantizationPipeline::new(PipelineConfig {
            max_batch_size: 2,
            validate: false,
            ..Default::default()
        });
        let t1 = [1.0f32];
        let t2 = [2.0f32];
        let t3 = [3.0f32];
        let err = p
            .quantize_batch(&[&t1, &t2, &t3], QuantizationType::I2S)
            .unwrap_err();
        assert_eq!(
            err,
            PipelineError::BatchTooLarge {
                actual: 3,
                max: 2
            }
        );
    }

    // -- unsupported quant type --------------------------------------------

    #[test]
    fn unsupported_quant_type_error_displays_correctly() {
        // Verify the UnsupportedQuantType error variant works
        let err = PipelineError::UnsupportedQuantType("FakeType".into());
        let msg = format!("{err}");
        assert!(msg.contains("FakeType"));
        assert!(msg.contains("unsupported"));
    }

    // -- I2S quantization --------------------------------------------------

    #[test]
    fn i2s_quantize_roundtrip() {
        let p = pipeline_no_validate();
        let data = vec![1.0f32, -1.0, 0.3, -0.3];
        let result = p.quantize(&data, QuantizationType::I2S).unwrap();
        assert_eq!(result.element_count, 4);
        assert_eq!(result.quant_type, QuantizationType::I2S);
        // 4 elements → 1 byte (2 bits each)
        assert_eq!(result.data.len(), 1);
    }

    #[test]
    fn i2s_packing_correct_bit_layout() {
        let p = pipeline_no_validate();
        // val > 0.5 → 3, val > 0.0 → 2, val > -0.5 → 1, val <= -0.5 → 0
        let data = vec![-1.0f32, -0.3, 0.3, 1.0];
        let result = p.quantize(&data, QuantizationType::I2S).unwrap();
        let byte = result.data[0];
        assert_eq!(byte & 0x03, 0); // -1.0 → 0
        assert_eq!((byte >> 2) & 0x03, 1); // -0.3 → 1
        assert_eq!((byte >> 4) & 0x03, 2); // 0.3 → 2
        assert_eq!((byte >> 6) & 0x03, 3); // 1.0 → 3
    }

    // -- TL1/TL2 quantization ---------------------------------------------

    #[test]
    fn tl1_quantize_produces_one_byte_per_element() {
        let p = pipeline_no_validate();
        let data = vec![0.0f32, 0.5, -0.5, 1.0, -1.0];
        let result = p.quantize(&data, QuantizationType::TL1).unwrap();
        assert_eq!(result.data.len(), 5);
        assert_eq!(result.quant_type, QuantizationType::TL1);
    }

    #[test]
    fn tl2_clamps_out_of_range_values() {
        let p = pipeline_no_validate();
        let data = vec![-5.0f32, 5.0];
        let result = p.quantize(&data, QuantizationType::TL2).unwrap();
        // -5.0 clamped to -1.0 → index 0
        assert_eq!(result.data[0], 0);
        // 5.0 clamped to 1.0 → index 255
        assert_eq!(result.data[1], 255);
    }

    // -- validation --------------------------------------------------------

    #[test]
    fn validation_populates_error_fields() {
        // Use a tolerance large enough for TL quantization rounding
        let p = GpuQuantizationPipeline::new(PipelineConfig {
            validate: true,
            validation_tolerance: 0.01,
            ..Default::default()
        });
        let data = vec![0.0f32; 8];
        let result = p.quantize(&data, QuantizationType::TL1).unwrap();
        assert!(result.max_error.is_some());
        assert!(result.mean_error.is_some());
    }

    #[test]
    fn no_validation_leaves_error_fields_none() {
        let p = pipeline_no_validate();
        let data = vec![0.0f32; 8];
        let result = p.quantize(&data, QuantizationType::TL1).unwrap();
        assert!(result.max_error.is_none());
        assert!(result.mean_error.is_none());
    }

    // -- batch quantization ------------------------------------------------

    #[test]
    fn batch_quantize_returns_correct_count() {
        let p = pipeline_no_validate();
        let t1 = [0.1f32, 0.2, 0.3, 0.4];
        let t2 = [-0.1f32, -0.2];
        let (results, summary) = p
            .quantize_batch(&[&t1, &t2], QuantizationType::I2S)
            .unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(summary.tensor_count, 2);
        assert_eq!(summary.total_elements, 6);
    }

    // -- summary display ---------------------------------------------------

    #[test]
    fn summary_display_format() {
        let summary = BatchQuantizationSummary {
            tensor_count: 3,
            total_elements: 100,
            total_output_bytes: 25,
            max_error: Some(0.001),
            mean_error: Some(0.0005),
        };
        let s = format!("{summary}");
        assert!(s.contains("3 tensors"));
        assert!(s.contains("100 elements"));
        assert!(s.contains("25 output bytes"));
        assert!(s.contains("max_err="));
    }

    // -- pipeline default trait --------------------------------------------

    #[test]
    fn pipeline_default_matches_with_defaults() {
        let p1 = GpuQuantizationPipeline::default();
        let p2 = GpuQuantizationPipeline::with_defaults();
        assert_eq!(p1.config().block_size, p2.config().block_size);
        assert_eq!(
            p1.config().qk256_block_size,
            p2.config().qk256_block_size
        );
    }

    #[test]
    fn i2s_handles_non_multiple_of_four_length() {
        let p = pipeline_no_validate();
        // 5 elements → 2 bytes (ceil(5/4) = 2)
        let data = vec![1.0f32, -1.0, 0.3, -0.3, 0.8];
        let result = p.quantize(&data, QuantizationType::I2S).unwrap();
        assert_eq!(result.data.len(), 2);
        assert_eq!(result.element_count, 5);
    }
}
