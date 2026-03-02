//! Model format conversion pipeline for transforming models between formats and dtypes.
//!
//! Provides a planning-based approach to model conversion: build a [`ConversionConfig`],
//! generate a [`ConversionPlan`] with [`plan_conversion`], then execute the plan.
//! Supports SafeTensors, GGUF, ONNX, and PyTorch formats with F32/F16/BF16/Int8/Int4 dtypes.
//!
//! # Examples
//!
//! ```
//! use bitnet_models::conversion::{
//!     ConversionConfig, DType, ModelFormat, plan_conversion,
//! };
//!
//! let config = ConversionConfig {
//!     source_format: ModelFormat::SafeTensors,
//!     target_format: ModelFormat::GGUF,
//!     target_dtype: DType::Int8,
//!     quantization_config: None,
//! };
//! let plan = plan_conversion("model.safetensors", &config);
//! assert!(!plan.steps.is_empty());
//! ```

use std::fmt;

// ---------------------------------------------------------------------------
// ModelFormat
// ---------------------------------------------------------------------------

/// Supported model serialization formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelFormat {
    /// HuggingFace SafeTensors (`.safetensors`).
    SafeTensors,
    /// GGML/GGUF binary format (`.gguf`).
    GGUF,
    /// Open Neural Network Exchange (`.onnx`).
    ONNX,
    /// PyTorch checkpoint (`.pt` / `.bin`).
    PyTorch,
}

impl fmt::Display for ModelFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SafeTensors => write!(f, "SafeTensors"),
            Self::GGUF => write!(f, "GGUF"),
            Self::ONNX => write!(f, "ONNX"),
            Self::PyTorch => write!(f, "PyTorch"),
        }
    }
}

impl ModelFormat {
    /// File extension typically associated with this format.
    pub fn extension(&self) -> &str {
        match self {
            Self::SafeTensors => "safetensors",
            Self::GGUF => "gguf",
            Self::ONNX => "onnx",
            Self::PyTorch => "bin",
        }
    }
}

// ---------------------------------------------------------------------------
// DType
// ---------------------------------------------------------------------------

/// Data types for model weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// 32-bit floating point.
    F32,
    /// 16-bit floating point (IEEE 754).
    F16,
    /// 16-bit brain floating point.
    BF16,
    /// 8-bit integer (quantized).
    Int8,
    /// 4-bit integer (quantized).
    Int4,
}

impl DType {
    /// Bytes per element for this dtype.
    ///
    /// For sub-byte types (Int4), returns 1 since elements are packed in pairs.
    pub fn bytes_per_element(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Int8 => 1,
            // 4-bit: two values per byte, but we report the effective byte cost
            // per element for size-estimation purposes (0.5 bytes each).
            Self::Int4 => 1,
        }
    }

    /// Whether this dtype represents a quantized (integer) type.
    pub fn is_quantized(&self) -> bool {
        matches!(self, Self::Int8 | Self::Int4)
    }

    /// Human-readable name for display.
    pub fn display_name(&self) -> &str {
        match self {
            Self::F32 => "float32",
            Self::F16 => "float16",
            Self::BF16 => "bfloat16",
            Self::Int8 => "int8",
            Self::Int4 => "int4",
        }
    }

    /// Compression ratio relative to F32 (e.g. F16 → 0.5, Int4 → 0.125).
    fn compression_ratio(&self) -> f64 {
        match self {
            Self::F32 => 1.0,
            Self::F16 | Self::BF16 => 0.5,
            Self::Int8 => 0.25,
            Self::Int4 => 0.125,
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

// ---------------------------------------------------------------------------
// Quantization
// ---------------------------------------------------------------------------

/// Quantization method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantMethod {
    /// Symmetric quantization (zero-point is always 0).
    Symmetric,
    /// Asymmetric quantization (zero-point can be non-zero).
    Asymmetric,
}

impl fmt::Display for QuantMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Symmetric => write!(f, "symmetric"),
            Self::Asymmetric => write!(f, "asymmetric"),
        }
    }
}

/// Quantization configuration for weight compression.
#[derive(Debug, Clone)]
pub struct QuantizationSpec {
    /// Quantization algorithm to use.
    pub method: QuantMethod,
    /// Optional group size for block-wise quantization.
    pub group_size: Option<usize>,
    /// Whether to quantize per-channel (true) or per-tensor (false).
    pub per_channel: bool,
    /// Number of calibration samples for range estimation.
    pub calibration_samples: usize,
}

impl Default for QuantizationSpec {
    fn default() -> Self {
        Self {
            method: QuantMethod::Symmetric,
            group_size: None,
            per_channel: true,
            calibration_samples: 128,
        }
    }
}

// ---------------------------------------------------------------------------
// ConversionConfig
// ---------------------------------------------------------------------------

/// Configuration for a model format conversion.
#[derive(Debug, Clone)]
pub struct ConversionConfig {
    /// Format of the source model.
    pub source_format: ModelFormat,
    /// Desired output format.
    pub target_format: ModelFormat,
    /// Desired output dtype.
    pub target_dtype: DType,
    /// Optional quantization parameters (required when `target_dtype` is quantized).
    pub quantization_config: Option<QuantizationSpec>,
}

// ---------------------------------------------------------------------------
// ConversionStep
// ---------------------------------------------------------------------------

/// A single step in a conversion pipeline.
#[derive(Debug, Clone, PartialEq)]
pub enum ConversionStep {
    /// Load weights from the source format.
    LoadWeights {
        /// Format to load from.
        source_format: ModelFormat,
    },
    /// Convert between floating-point dtypes.
    ConvertDtype {
        /// Original dtype.
        from: DType,
        /// Target dtype.
        to: DType,
    },
    /// Quantize weights to a lower-precision integer type.
    Quantize {
        /// Quantization algorithm.
        method: QuantMethod,
        /// Target bit width.
        bits: u8,
    },
    /// Pack tensors into the target serialization format.
    PackTensors {
        /// Target format to write.
        target_format: ModelFormat,
    },
    /// Validate the output file (checksum, tensor count, shape checks).
    ValidateOutput,
}

// ---------------------------------------------------------------------------
// ConversionPlan
// ---------------------------------------------------------------------------

/// A planned conversion pipeline with cost estimates.
#[derive(Debug, Clone)]
pub struct ConversionPlan {
    /// Path to the source model.
    pub source_path: String,
    /// Path where the output will be written.
    pub target_path: String,
    /// The conversion configuration.
    pub config: ConversionConfig,
    /// Ordered list of steps to execute.
    pub steps: Vec<ConversionStep>,
    /// Estimated wall-clock time in seconds.
    pub estimated_time_secs: u64,
    /// Estimated output file size in bytes.
    pub estimated_output_size_bytes: u64,
}

// ---------------------------------------------------------------------------
// ConversionResult
// ---------------------------------------------------------------------------

/// Result of executing a conversion plan.
#[derive(Debug, Clone)]
pub struct ConversionResult {
    /// Whether the conversion completed successfully.
    pub success: bool,
    /// Number of steps that completed (including on failure).
    pub steps_completed: usize,
    /// Total number of steps in the plan.
    pub total_steps: usize,
    /// Path to the output file.
    pub output_path: String,
    /// Size of the output file in bytes.
    pub output_size_bytes: u64,
    /// Wall-clock conversion time in milliseconds.
    pub conversion_time_ms: u64,
    /// Non-fatal warnings emitted during conversion.
    pub warnings: Vec<String>,
}

// ---------------------------------------------------------------------------
// Planning helpers
// ---------------------------------------------------------------------------

/// Estimate output file size given a parameter count and target dtype.
///
/// Uses the dtype's compression ratio relative to F32 (4 bytes per param).
pub fn estimate_output_size(num_params: u64, target_dtype: &DType) -> u64 {
    let f32_bytes = num_params * 4;
    (f32_bytes as f64 * target_dtype.compression_ratio()) as u64
}

/// Infer a default source dtype from the model format.
fn default_source_dtype(format: &ModelFormat) -> DType {
    match format {
        // SafeTensors models from HF are typically stored in BF16 or F16;
        // we assume F32 as the safe baseline for conversion planning.
        ModelFormat::SafeTensors | ModelFormat::PyTorch | ModelFormat::ONNX => DType::F32,
        ModelFormat::GGUF => DType::F16,
    }
}

/// Determine the quantization bit-width for a target dtype.
fn quant_bits(dtype: &DType) -> u8 {
    match dtype {
        DType::Int8 => 8,
        DType::Int4 => 4,
        _ => 0,
    }
}

/// Build an output path from the source path and target format.
fn derive_target_path(source_path: &str, target_format: &ModelFormat) -> String {
    let stem = source_path.rsplit_once('.').map_or(source_path, |(s, _)| s);
    format!("{}.{}", stem, target_format.extension())
}

/// Generate a conversion plan for the given source and configuration.
///
/// The plan describes the ordered sequence of steps and provides size/time
/// estimates. It does **not** execute any I/O.
pub fn plan_conversion(source_path: &str, config: &ConversionConfig) -> ConversionPlan {
    let mut steps = Vec::new();

    // Step 1: Load weights from source format.
    steps.push(ConversionStep::LoadWeights { source_format: config.source_format });

    let source_dtype = default_source_dtype(&config.source_format);

    if config.target_dtype.is_quantized() {
        // Quantization path: quantize then pack.
        let method =
            config.quantization_config.as_ref().map_or(QuantMethod::Symmetric, |q| q.method);
        steps.push(ConversionStep::Quantize { method, bits: quant_bits(&config.target_dtype) });
    } else if source_dtype != config.target_dtype {
        // Float-to-float dtype conversion.
        steps.push(ConversionStep::ConvertDtype { from: source_dtype, to: config.target_dtype });
    }

    // If target format differs from source, pack into the new format.
    if config.source_format != config.target_format {
        steps.push(ConversionStep::PackTensors { target_format: config.target_format });
    }

    // Always validate.
    steps.push(ConversionStep::ValidateOutput);

    let target_path = derive_target_path(source_path, &config.target_format);

    // Rough time estimate: 10s base + 5s per step.
    let estimated_time_secs = 10 + (steps.len() as u64) * 5;

    // Assume 7B parameters as a reasonable default when we don't know the model.
    let estimated_params: u64 = 7_000_000_000;
    let estimated_output_size_bytes = estimate_output_size(estimated_params, &config.target_dtype);

    ConversionPlan {
        source_path: source_path.to_string(),
        target_path,
        config: config.clone(),
        steps,
        estimated_time_secs,
        estimated_output_size_bytes,
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- ModelFormat --------------------------------------------------------

    #[test]
    fn test_model_format_display_safetensors() {
        assert_eq!(ModelFormat::SafeTensors.to_string(), "SafeTensors");
    }

    #[test]
    fn test_model_format_display_gguf() {
        assert_eq!(ModelFormat::GGUF.to_string(), "GGUF");
    }

    #[test]
    fn test_model_format_display_onnx() {
        assert_eq!(ModelFormat::ONNX.to_string(), "ONNX");
    }

    #[test]
    fn test_model_format_display_pytorch() {
        assert_eq!(ModelFormat::PyTorch.to_string(), "PyTorch");
    }

    #[test]
    fn test_model_format_extension() {
        assert_eq!(ModelFormat::SafeTensors.extension(), "safetensors");
        assert_eq!(ModelFormat::GGUF.extension(), "gguf");
        assert_eq!(ModelFormat::ONNX.extension(), "onnx");
        assert_eq!(ModelFormat::PyTorch.extension(), "bin");
    }

    // -- DType -------------------------------------------------------------

    #[test]
    fn test_dtype_bytes_per_element_f32() {
        assert_eq!(DType::F32.bytes_per_element(), 4);
    }

    #[test]
    fn test_dtype_bytes_per_element_f16() {
        assert_eq!(DType::F16.bytes_per_element(), 2);
    }

    #[test]
    fn test_dtype_bytes_per_element_bf16() {
        assert_eq!(DType::BF16.bytes_per_element(), 2);
    }

    #[test]
    fn test_dtype_bytes_per_element_int8() {
        assert_eq!(DType::Int8.bytes_per_element(), 1);
    }

    #[test]
    fn test_dtype_bytes_per_element_int4() {
        assert_eq!(DType::Int4.bytes_per_element(), 1);
    }

    #[test]
    fn test_dtype_is_quantized_float_types() {
        assert!(!DType::F32.is_quantized());
        assert!(!DType::F16.is_quantized());
        assert!(!DType::BF16.is_quantized());
    }

    #[test]
    fn test_dtype_is_quantized_int_types() {
        assert!(DType::Int8.is_quantized());
        assert!(DType::Int4.is_quantized());
    }

    #[test]
    fn test_dtype_display_names() {
        assert_eq!(DType::F32.display_name(), "float32");
        assert_eq!(DType::F16.display_name(), "float16");
        assert_eq!(DType::BF16.display_name(), "bfloat16");
        assert_eq!(DType::Int8.display_name(), "int8");
        assert_eq!(DType::Int4.display_name(), "int4");
    }

    #[test]
    fn test_dtype_display_trait() {
        assert_eq!(format!("{}", DType::F32), "float32");
        assert_eq!(format!("{}", DType::Int4), "int4");
    }

    // -- QuantizationSpec --------------------------------------------------

    #[test]
    fn test_quantization_spec_defaults() {
        let spec = QuantizationSpec::default();
        assert_eq!(spec.method, QuantMethod::Symmetric);
        assert!(spec.group_size.is_none());
        assert!(spec.per_channel);
        assert_eq!(spec.calibration_samples, 128);
    }

    #[test]
    fn test_quantization_spec_custom() {
        let spec = QuantizationSpec {
            method: QuantMethod::Asymmetric,
            group_size: Some(128),
            per_channel: false,
            calibration_samples: 256,
        };
        assert_eq!(spec.method, QuantMethod::Asymmetric);
        assert_eq!(spec.group_size, Some(128));
        assert!(!spec.per_channel);
        assert_eq!(spec.calibration_samples, 256);
    }

    // -- ConversionConfig --------------------------------------------------

    #[test]
    fn test_conversion_config_construction() {
        let config = ConversionConfig {
            source_format: ModelFormat::SafeTensors,
            target_format: ModelFormat::GGUF,
            target_dtype: DType::F16,
            quantization_config: None,
        };
        assert_eq!(config.source_format, ModelFormat::SafeTensors);
        assert_eq!(config.target_format, ModelFormat::GGUF);
        assert_eq!(config.target_dtype, DType::F16);
        assert!(config.quantization_config.is_none());
    }

    // -- ConversionStep ordering -------------------------------------------

    #[test]
    fn test_plan_safetensors_f32_to_gguf_f16() {
        let config = ConversionConfig {
            source_format: ModelFormat::SafeTensors,
            target_format: ModelFormat::GGUF,
            target_dtype: DType::F16,
            quantization_config: None,
        };
        let plan = plan_conversion("model.safetensors", &config);
        // LoadWeights → ConvertDtype → PackTensors → Validate
        assert_eq!(plan.steps.len(), 4);
        assert!(matches!(
            plan.steps[0],
            ConversionStep::LoadWeights { source_format: ModelFormat::SafeTensors }
        ));
        assert!(matches!(
            plan.steps[1],
            ConversionStep::ConvertDtype { from: DType::F32, to: DType::F16 }
        ));
        assert!(matches!(
            plan.steps[2],
            ConversionStep::PackTensors { target_format: ModelFormat::GGUF }
        ));
        assert!(matches!(plan.steps[3], ConversionStep::ValidateOutput));
    }

    #[test]
    fn test_plan_safetensors_bf16_to_gguf_int8() {
        let config = ConversionConfig {
            source_format: ModelFormat::SafeTensors,
            target_format: ModelFormat::GGUF,
            target_dtype: DType::Int8,
            quantization_config: Some(QuantizationSpec::default()),
        };
        let plan = plan_conversion("model.safetensors", &config);
        // LoadWeights → Quantize → PackTensors → Validate
        assert_eq!(plan.steps.len(), 4);
        assert!(matches!(plan.steps[0], ConversionStep::LoadWeights { .. }));
        assert!(matches!(
            plan.steps[1],
            ConversionStep::Quantize { method: QuantMethod::Symmetric, bits: 8 }
        ));
        assert!(matches!(
            plan.steps[2],
            ConversionStep::PackTensors { target_format: ModelFormat::GGUF }
        ));
        assert!(matches!(plan.steps[3], ConversionStep::ValidateOutput));
    }

    #[test]
    fn test_plan_safetensors_f16_to_safetensors_int4() {
        let config = ConversionConfig {
            source_format: ModelFormat::SafeTensors,
            target_format: ModelFormat::SafeTensors,
            target_dtype: DType::Int4,
            quantization_config: Some(QuantizationSpec {
                method: QuantMethod::Asymmetric,
                ..QuantizationSpec::default()
            }),
        };
        let plan = plan_conversion("model.safetensors", &config);
        // LoadWeights → Quantize → Validate (same format, no PackTensors)
        assert_eq!(plan.steps.len(), 3);
        assert!(matches!(plan.steps[0], ConversionStep::LoadWeights { .. }));
        assert!(matches!(
            plan.steps[1],
            ConversionStep::Quantize { method: QuantMethod::Asymmetric, bits: 4 }
        ));
        assert!(matches!(plan.steps[2], ConversionStep::ValidateOutput));
    }

    // -- plan_conversion: format combos ------------------------------------

    #[test]
    fn test_plan_gguf_to_safetensors_f32() {
        let config = ConversionConfig {
            source_format: ModelFormat::GGUF,
            target_format: ModelFormat::SafeTensors,
            target_dtype: DType::F32,
            quantization_config: None,
        };
        let plan = plan_conversion("model.gguf", &config);
        // GGUF default dtype is F16 → convert to F32 → pack safetensors → validate
        assert!(plan.steps.len() >= 3);
        assert_eq!(plan.target_path, "model.safetensors");
    }

    #[test]
    fn test_plan_pytorch_to_onnx_f16() {
        let config = ConversionConfig {
            source_format: ModelFormat::PyTorch,
            target_format: ModelFormat::ONNX,
            target_dtype: DType::F16,
            quantization_config: None,
        };
        let plan = plan_conversion("model.bin", &config);
        assert!(plan.steps.len() >= 3);
        assert_eq!(plan.target_path, "model.onnx");
    }

    // -- estimate_output_size: 7B model ------------------------------------

    #[test]
    fn test_estimate_output_size_7b_f32() {
        let size = estimate_output_size(7_000_000_000, &DType::F32);
        assert_eq!(size, 28_000_000_000); // 7B × 4 bytes
    }

    #[test]
    fn test_estimate_output_size_7b_f16() {
        let size = estimate_output_size(7_000_000_000, &DType::F16);
        assert_eq!(size, 14_000_000_000); // 7B × 2 bytes
    }

    #[test]
    fn test_estimate_output_size_7b_int8() {
        let size = estimate_output_size(7_000_000_000, &DType::Int8);
        assert_eq!(size, 7_000_000_000); // 7B × 1 byte
    }

    #[test]
    fn test_estimate_output_size_7b_int4() {
        let size = estimate_output_size(7_000_000_000, &DType::Int4);
        assert_eq!(size, 3_500_000_000); // 7B × 0.5 bytes
    }

    // -- ConversionResult --------------------------------------------------

    #[test]
    fn test_conversion_result_success() {
        let result = ConversionResult {
            success: true,
            steps_completed: 4,
            total_steps: 4,
            output_path: "model.gguf".to_string(),
            output_size_bytes: 14_000_000_000,
            conversion_time_ms: 45_000,
            warnings: vec![],
        };
        assert!(result.success);
        assert_eq!(result.steps_completed, result.total_steps);
        assert!(result.warnings.is_empty());
    }

    #[test]
    fn test_conversion_result_failure() {
        let result = ConversionResult {
            success: false,
            steps_completed: 2,
            total_steps: 4,
            output_path: String::new(),
            output_size_bytes: 0,
            conversion_time_ms: 12_000,
            warnings: vec!["quantization range overflow".to_string()],
        };
        assert!(!result.success);
        assert!(result.steps_completed < result.total_steps);
        assert_eq!(result.warnings.len(), 1);
    }

    // -- Edge cases --------------------------------------------------------

    #[test]
    fn test_same_format_no_dtype_change() {
        let config = ConversionConfig {
            source_format: ModelFormat::SafeTensors,
            target_format: ModelFormat::SafeTensors,
            target_dtype: DType::F32,
            quantization_config: None,
        };
        let plan = plan_conversion("model.safetensors", &config);
        // LoadWeights → Validate only (no convert, no pack)
        assert_eq!(plan.steps.len(), 2);
        assert!(matches!(plan.steps[0], ConversionStep::LoadWeights { .. }));
        assert!(matches!(plan.steps[1], ConversionStep::ValidateOutput));
    }

    #[test]
    fn test_derive_target_path_preserves_stem() {
        let config = ConversionConfig {
            source_format: ModelFormat::SafeTensors,
            target_format: ModelFormat::GGUF,
            target_dtype: DType::F32,
            quantization_config: None,
        };
        let plan = plan_conversion("path/to/my_model.safetensors", &config);
        assert_eq!(plan.target_path, "path/to/my_model.gguf");
    }

    #[test]
    fn test_plan_has_time_estimate() {
        let config = ConversionConfig {
            source_format: ModelFormat::SafeTensors,
            target_format: ModelFormat::GGUF,
            target_dtype: DType::F16,
            quantization_config: None,
        };
        let plan = plan_conversion("model.safetensors", &config);
        assert!(plan.estimated_time_secs > 0);
    }

    #[test]
    fn test_plan_has_size_estimate() {
        let config = ConversionConfig {
            source_format: ModelFormat::SafeTensors,
            target_format: ModelFormat::GGUF,
            target_dtype: DType::Int4,
            quantization_config: None,
        };
        let plan = plan_conversion("model.safetensors", &config);
        assert!(plan.estimated_output_size_bytes > 0);
        // Int4 estimate should be smaller than F32
        let f32_plan = plan_conversion(
            "model.safetensors",
            &ConversionConfig { target_dtype: DType::F32, ..config },
        );
        assert!(plan.estimated_output_size_bytes < f32_plan.estimated_output_size_bytes);
    }

    #[test]
    fn test_quant_method_display() {
        assert_eq!(QuantMethod::Symmetric.to_string(), "symmetric");
        assert_eq!(QuantMethod::Asymmetric.to_string(), "asymmetric");
    }
}
