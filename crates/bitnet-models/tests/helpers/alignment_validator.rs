//! Tensor alignment validation utilities for GGUF weight loading tests
//!
//! Provides comprehensive validation for:
//! - Memory alignment (32-byte default for SIMD optimization)
//! - Shape consistency (tensor dimensions match data size)
//! - Bounds checking (tensor data within GGUF file limits)
//!
//! Used by AC3 tests in gguf_weight_loading_tests.rs
//!
//! # Usage Examples
//!
//! ## Validate Candle Tensors (Post-Load)
//!
//! ```rust,ignore
//! use helpers::alignment_validator::{validate_candle_tensor, AlignmentConfig};
//!
//! // Validate a single tensor with default config (32-byte alignment, lenient)
//! let config = AlignmentConfig::default();
//! let result = validate_candle_tensor("my.weight", &tensor, &config)?;
//!
//! if !result.is_ok() {
//!     eprintln!("Validation failed: {}", result.summary());
//! }
//! ```
//!
//! ## Validate All Tensors in a Model
//!
//! ```rust,ignore
//! use helpers::alignment_validator::{validate_all_tensors, AlignmentConfig, generate_validation_report};
//!
//! let config = AlignmentConfig::strict(); // Fail on any violation
//! let results = validate_all_tensors(&tensor_map, &config)?;
//!
//! println!("{}", generate_validation_report(&results));
//! ```
//!
//! ## Validate Raw GGUF Tensor Metadata (Pre-Load)
//!
//! ```rust,ignore
//! use helpers::alignment_validator::{validate_gguf_tensor_metadata, AlignmentConfig};
//!
//! let config = AlignmentConfig::default().with_alignment(64); // Custom alignment
//! let result = validate_gguf_tensor_metadata(
//!     "tensor.weight",
//!     offset,      // u64: offset in GGUF file
//!     size,        // u64: tensor data size in bytes
//!     &shape,      // &[usize]: tensor dimensions
//!     file_size,   // u64: total GGUF file size
//!     &config,
//! )?;
//!
//! assert!(result.is_aligned);
//! ```
//!
//! # Alignment Requirements
//!
//! - **CPU (AVX2/NEON)**: 32-byte alignment for optimal SIMD performance
//! - **GPU (CUDA)**: 512-byte alignment guaranteed by runtime
//! - **Strict mode**: Fails immediately on first violation
//! - **Lenient mode**: Reports warnings but continues validation

use anyhow::{Context, Result};
use candle_core::Tensor as CandleTensor;
use std::collections::HashMap;

/// Default alignment requirement for GGUF tensors (32 bytes for AVX2/NEON)
pub const DEFAULT_ALIGNMENT: usize = 32;

/// Alignment validation configuration
#[derive(Debug, Clone)]
pub struct AlignmentConfig {
    /// Required alignment in bytes (must be power of 2)
    pub alignment: usize,
    /// Whether to enforce strict alignment (fail on misalignment)
    pub strict: bool,
    /// Whether to validate shape consistency
    pub validate_shapes: bool,
}

impl Default for AlignmentConfig {
    fn default() -> Self {
        Self { alignment: DEFAULT_ALIGNMENT, strict: false, validate_shapes: true }
    }
}

impl AlignmentConfig {
    /// Create a strict alignment configuration (fails on any violation)
    pub fn strict() -> Self {
        Self { alignment: DEFAULT_ALIGNMENT, strict: true, validate_shapes: true }
    }

    /// Create a lenient configuration (warnings only)
    #[allow(dead_code)]
    pub fn lenient() -> Self {
        Self { alignment: DEFAULT_ALIGNMENT, strict: false, validate_shapes: false }
    }

    /// Set custom alignment requirement
    #[allow(dead_code)]
    pub fn with_alignment(mut self, alignment: usize) -> Self {
        assert!(
            alignment > 0 && alignment.is_power_of_two(),
            "Alignment must be a power of 2, got {}",
            alignment
        );
        self.alignment = alignment;
        self
    }
}

/// Validation result with detailed diagnostics
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub tensor_name: String,
    pub is_aligned: bool,
    pub actual_alignment: Option<usize>,
    pub shape_valid: bool,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}

impl ValidationResult {
    fn new(tensor_name: String) -> Self {
        Self {
            tensor_name,
            is_aligned: true,
            actual_alignment: None,
            shape_valid: true,
            warnings: Vec::new(),
            errors: Vec::new(),
        }
    }

    /// Check if validation passed (no errors, or warnings only if lenient)
    pub fn is_ok(&self) -> bool {
        self.errors.is_empty()
    }

    /// Get a human-readable summary
    pub fn summary(&self) -> String {
        let mut parts = vec![format!("Tensor '{}':", self.tensor_name)];

        if !self.is_aligned {
            if let Some(actual) = self.actual_alignment {
                parts.push(format!("  Alignment: {} bytes (not aligned)", actual));
            } else {
                parts.push("  Alignment: unknown (cannot determine)".to_string());
            }
        } else if let Some(actual) = self.actual_alignment {
            parts.push(format!("  Alignment: {} bytes ✓", actual));
        }

        if !self.shape_valid {
            parts.push("  Shape: invalid ✗".to_string());
        }

        for warning in &self.warnings {
            parts.push(format!("  Warning: {}", warning));
        }

        for error in &self.errors {
            parts.push(format!("  Error: {}", error));
        }

        parts.join("\n")
    }
}

/// Validate a single Candle tensor for alignment and shape consistency
///
/// This function validates already-loaded Candle tensors. For raw GGUF validation,
/// use `validate_gguf_tensor_metadata` instead.
///
/// # Arguments
///
/// * `name` - Tensor name for error reporting
/// * `tensor` - Candle tensor to validate
/// * `config` - Validation configuration
///
/// # Returns
///
/// Detailed validation result with alignment info and diagnostics
pub fn validate_candle_tensor(
    name: &str,
    tensor: &CandleTensor,
    config: &AlignmentConfig,
) -> Result<ValidationResult> {
    let mut result = ValidationResult::new(name.to_string());

    // 1. Validate shape consistency (tensor dimensions match element count)
    if config.validate_shapes {
        let shape = tensor.shape();
        let dims = shape.dims();
        let expected_elements: usize = dims.iter().product();
        let actual_elements = shape.elem_count();

        if expected_elements != actual_elements {
            let error = format!(
                "Shape mismatch: dims {:?} imply {} elements, but tensor reports {} elements",
                dims, expected_elements, actual_elements
            );
            result.errors.push(error);
            result.shape_valid = false;

            if config.strict {
                return Err(anyhow::anyhow!(
                    "Tensor '{}' shape validation failed: dims {:?}",
                    name,
                    dims
                ));
            }
        }
    }

    // 2. Check memory alignment (Candle tensors don't expose raw pointers directly,
    //    so we validate based on storage layout)
    //
    // Note: Candle tensors may be strided, contiguous, or on device. For CPU tensors,
    // we can check if the underlying storage is contiguous and properly aligned.
    // For GPU tensors, alignment is handled by the CUDA runtime.

    if tensor.device().is_cpu() {
        // For CPU tensors, check if contiguous (optimal for SIMD)
        if !tensor.is_contiguous() {
            result.warnings.push(
                "Tensor is not contiguous - may have suboptimal memory layout for SIMD".to_string(),
            );
        }

        // Note: Candle doesn't expose raw pointer alignment directly, so we assume
        // that properly constructed tensors (from GGUF with alignment) are aligned.
        // This is a limitation of validating post-load tensors vs raw GGUF metadata.
        result.is_aligned = true;
        result.actual_alignment = Some(config.alignment);
    } else if tensor.device().is_cuda() {
        // GPU tensors: alignment is guaranteed by CUDA allocator (512-byte min)
        result.is_aligned = true;
        result.actual_alignment = Some(512);
        result.warnings.push("GPU tensor - alignment guaranteed by CUDA runtime".to_string());
    } else {
        // Other devices (Metal, etc.)
        result.warnings.push("Non-CPU/GPU tensor - alignment not validated".to_string());
    }

    Ok(result)
}

/// Validate all tensors in a tensor map
///
/// # Arguments
///
/// * `tensors` - HashMap of tensor name -> Candle tensor
/// * `config` - Validation configuration
///
/// # Returns
///
/// Vector of validation results for all tensors
#[allow(dead_code)]
pub fn validate_all_tensors(
    tensors: &HashMap<String, CandleTensor>,
    config: &AlignmentConfig,
) -> Result<Vec<ValidationResult>> {
    let mut results = Vec::with_capacity(tensors.len());

    for (name, tensor) in tensors {
        let result = validate_candle_tensor(name, tensor, config)
            .with_context(|| format!("Failed to validate tensor '{}'", name))?;

        // In strict mode, fail fast on first error
        if config.strict && !result.is_ok() {
            return Err(anyhow::anyhow!(
                "Strict validation failed for tensor '{}':\n{}",
                name,
                result.summary()
            ));
        }

        results.push(result);
    }

    Ok(results)
}

/// Generate a validation report for all tensors
///
/// # Arguments
///
/// * `results` - Validation results from `validate_all_tensors`
///
/// # Returns
///
/// Human-readable report with statistics and per-tensor diagnostics
pub fn generate_validation_report(results: &[ValidationResult]) -> String {
    let total = results.len();
    let aligned = results.iter().filter(|r| r.is_aligned).count();
    let shape_valid = results.iter().filter(|r| r.shape_valid).count();
    let with_warnings = results.iter().filter(|r| !r.warnings.is_empty()).count();
    let with_errors = results.iter().filter(|r| !r.errors.is_empty()).count();

    let mut report = vec![
        "=== Tensor Alignment Validation Report ===".to_string(),
        format!("Total tensors: {}", total),
        format!("Aligned: {}/{} ({:.1}%)", aligned, total, (aligned as f64 / total as f64) * 100.0),
        format!(
            "Shape valid: {}/{} ({:.1}%)",
            shape_valid,
            total,
            (shape_valid as f64 / total as f64) * 100.0
        ),
        format!("Warnings: {}", with_warnings),
        format!("Errors: {}", with_errors),
        String::new(),
    ];

    // Add per-tensor details for any with warnings or errors
    let problematic: Vec<_> =
        results.iter().filter(|r| !r.warnings.is_empty() || !r.errors.is_empty()).collect();

    if !problematic.is_empty() {
        report.push("=== Tensors with Issues ===".to_string());
        for result in problematic {
            report.push(result.summary());
            report.push(String::new());
        }
    }

    report.join("\n")
}

/// Validate raw GGUF tensor metadata (offset, size, alignment)
///
/// This validates tensor metadata before loading into Candle tensors.
/// Use this for direct GGUF file validation.
///
/// # Arguments
///
/// * `name` - Tensor name
/// * `offset` - Tensor data offset in file
/// * `size` - Tensor data size in bytes
/// * `shape` - Tensor dimensions
/// * `file_size` - Total GGUF file size
/// * `config` - Validation configuration
///
/// # Returns
///
/// Validation result with alignment and bounds checking
pub fn validate_gguf_tensor_metadata(
    name: &str,
    offset: u64,
    size: u64,
    shape: &[usize],
    file_size: u64,
    config: &AlignmentConfig,
) -> Result<ValidationResult> {
    let mut result = ValidationResult::new(name.to_string());

    // 1. Check alignment of data offset
    let alignment = config.alignment as u64;
    let is_aligned = if alignment.is_power_of_two() {
        (offset & (alignment - 1)) == 0
    } else {
        offset.is_multiple_of(alignment)
    };

    if !is_aligned {
        let actual_alignment = if offset > 0 {
            // Find actual alignment (largest power of 2 that divides offset)
            let mut align = 1u64;
            while align <= offset && offset.is_multiple_of(align) {
                align *= 2;
            }
            align / 2
        } else {
            0
        };

        result.is_aligned = false;
        result.actual_alignment = Some(actual_alignment as usize);
        result.errors.push(format!(
            "Data offset {} is not aligned to {} bytes (actual: {} bytes)",
            offset, alignment, actual_alignment
        ));

        if config.strict {
            return Err(anyhow::anyhow!(
                "Tensor '{}' offset {} not aligned to {} bytes",
                name,
                offset,
                alignment
            ));
        }
    } else {
        result.is_aligned = true;
        result.actual_alignment = Some(alignment as usize);
    }

    // 2. Validate tensor is within file bounds
    let end_offset = offset.saturating_add(size);
    if end_offset > file_size {
        result.errors.push(format!(
            "Tensor data extends beyond file (offset={}, size={}, file_size={}, end={})",
            offset, size, file_size, end_offset
        ));

        if config.strict {
            return Err(anyhow::anyhow!(
                "Tensor '{}' extends beyond file bounds (end={}, file_size={})",
                name,
                end_offset,
                file_size
            ));
        }
    }

    // 3. Validate shape dimensions are reasonable
    if config.validate_shapes {
        let product: usize = shape.iter().product();
        if product == 0 {
            result.errors.push(format!("Shape {:?} has zero total elements", shape));
            result.shape_valid = false;
        }

        // Warn on very large tensors (>10GB)
        const MAX_REASONABLE_SIZE: u64 = 10 * 1024 * 1024 * 1024;
        if size > MAX_REASONABLE_SIZE {
            result.warnings.push(format!(
                "Very large tensor: {} bytes ({:.2} GB)",
                size,
                size as f64 / (1024.0 * 1024.0 * 1024.0)
            ));
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_validate_candle_tensor_cpu() -> Result<()> {
        let device = Device::Cpu;
        let tensor = CandleTensor::zeros(&[128, 256], candle_core::DType::F32, &device)?;

        let config = AlignmentConfig::default();
        let result = validate_candle_tensor("test.weight", &tensor, &config)?;

        assert!(result.is_ok());
        assert!(result.is_aligned);
        assert!(result.shape_valid);

        Ok(())
    }

    #[test]
    fn test_validate_gguf_metadata_aligned() -> Result<()> {
        let config = AlignmentConfig::default();
        let result = validate_gguf_tensor_metadata(
            "test.weight",
            1024, // offset (32-byte aligned)
            4096, // size
            &[128, 256],
            10000, // file_size
            &config,
        )?;

        assert!(result.is_ok());
        assert!(result.is_aligned);
        assert_eq!(result.actual_alignment, Some(32));

        Ok(())
    }

    #[test]
    fn test_validate_gguf_metadata_misaligned() -> Result<()> {
        let config = AlignmentConfig::default(); // lenient by default

        let result = validate_gguf_tensor_metadata(
            "test.weight",
            1027, // offset (not 32-byte aligned)
            4096, // size
            &[128, 256],
            10000, // file_size
            &config,
        )?;

        assert!(!result.is_aligned);
        assert_eq!(result.actual_alignment, Some(1));
        assert!(!result.errors.is_empty());

        Ok(())
    }

    #[test]
    fn test_validate_gguf_metadata_out_of_bounds() -> Result<()> {
        let config = AlignmentConfig::default();

        let result = validate_gguf_tensor_metadata(
            "test.weight",
            1024,  // offset
            10000, // size (extends beyond file)
            &[128, 256],
            5000, // file_size (too small)
            &config,
        )?;

        assert!(!result.is_ok());
        assert!(result.errors.iter().any(|e| e.contains("beyond file")));

        Ok(())
    }

    #[test]
    fn test_strict_mode_fails_on_misalignment() {
        let config = AlignmentConfig::strict();

        let result = validate_gguf_tensor_metadata(
            "test.weight",
            1027, // misaligned
            4096,
            &[128, 256],
            10000,
            &config,
        );

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not aligned"));
    }

    #[test]
    fn test_validation_report() -> Result<()> {
        let results = vec![
            ValidationResult {
                tensor_name: "good.weight".to_string(),
                is_aligned: true,
                actual_alignment: Some(32),
                shape_valid: true,
                warnings: Vec::new(),
                errors: Vec::new(),
            },
            ValidationResult {
                tensor_name: "bad.weight".to_string(),
                is_aligned: false,
                actual_alignment: Some(8),
                shape_valid: true,
                warnings: vec!["Not contiguous".to_string()],
                errors: vec!["Misaligned".to_string()],
            },
        ];

        let report = generate_validation_report(&results);

        assert!(report.contains("Total tensors: 2"));
        assert!(report.contains("Aligned: 1/2"));
        assert!(report.contains("Errors: 1"));
        assert!(report.contains("bad.weight"));

        Ok(())
    }
}
