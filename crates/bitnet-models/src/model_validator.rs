//! Model validator for checking GGUF headers, tensor shapes, vocab size,
//! hidden dimensions, layer counts, and weight dtypes.

/// GGUF magic number (`GGUF` in little-endian).
pub const GGUF_MAGIC: u32 = 0x46475547;

/// Minimum supported GGUF version.
pub const GGUF_VERSION_MIN: u32 = 2;

/// Maximum supported GGUF version.
pub const GGUF_VERSION_MAX: u32 = 3;

/// Result of a single validation check.
#[derive(Debug, Clone)]
pub struct ValidationCheck {
    pub name: String,
    pub passed: bool,
    pub message: String,
}

/// Aggregated result of running multiple validation checks.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub passed: bool,
    pub checks: Vec<ValidationCheck>,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}

/// Validates model file integrity and configuration.
#[derive(Debug, Clone)]
pub struct ModelValidator;

impl ModelValidator {
    pub fn new() -> Self {
        Self
    }

    /// Check that `magic` equals [`GGUF_MAGIC`] and `version` is within the
    /// supported range.
    pub fn validate_header(&self, magic: u32, version: u32) -> ValidationCheck {
        if magic != GGUF_MAGIC {
            return ValidationCheck {
                name: "header".into(),
                passed: false,
                message: format!("invalid magic: expected {GGUF_MAGIC:#010X}, got {magic:#010X}"),
            };
        }
        if !(GGUF_VERSION_MIN..=GGUF_VERSION_MAX).contains(&version) {
            return ValidationCheck {
                name: "header".into(),
                passed: false,
                message: format!(
                    "unsupported version {version} (expected {GGUF_VERSION_MIN}..={GGUF_VERSION_MAX})"
                ),
            };
        }
        ValidationCheck {
            name: "header".into(),
            passed: true,
            message: format!("GGUF v{version} header OK"),
        }
    }

    /// Check that `count` is within `[min, max]`.
    pub fn validate_tensor_count(&self, count: usize, min: usize, max: usize) -> ValidationCheck {
        let passed = count >= min && count <= max;
        ValidationCheck {
            name: "tensor_count".into(),
            passed,
            message: if passed {
                format!("tensor count {count} within [{min}, {max}]")
            } else {
                format!("tensor count {count} outside [{min}, {max}]")
            },
        }
    }

    /// Check that `shape` has exactly `expected_dims` dimensions.
    pub fn validate_tensor_shape(
        &self,
        name: &str,
        shape: &[usize],
        expected_dims: usize,
    ) -> ValidationCheck {
        let passed = shape.len() == expected_dims;
        ValidationCheck {
            name: format!("tensor_shape:{name}"),
            passed,
            message: if passed {
                format!("{name}: shape has {expected_dims} dims OK")
            } else {
                format!("{name}: expected {expected_dims} dims, got {}", shape.len())
            },
        }
    }

    /// Check that `size` is within `[expected - tolerance, expected + tolerance]`.
    pub fn validate_vocab_size(
        &self,
        size: usize,
        expected: usize,
        tolerance: usize,
    ) -> ValidationCheck {
        let lo = expected.saturating_sub(tolerance);
        let hi = expected.saturating_add(tolerance);
        let passed = size >= lo && size <= hi;
        ValidationCheck {
            name: "vocab_size".into(),
            passed,
            message: if passed {
                format!("vocab size {size} within tolerance of {expected} (±{tolerance})")
            } else {
                format!("vocab size {size} outside tolerance of {expected} (±{tolerance})")
            },
        }
    }

    /// Check that `size` equals `expected`.
    pub fn validate_hidden_size(&self, size: usize, expected: usize) -> ValidationCheck {
        let passed = size == expected;
        ValidationCheck {
            name: "hidden_size".into(),
            passed,
            message: if passed {
                format!("hidden size {size} matches expected")
            } else {
                format!("hidden size {size} != expected {expected}")
            },
        }
    }

    /// Check that `count` equals `expected`.
    pub fn validate_layer_count(&self, count: usize, expected: usize) -> ValidationCheck {
        let passed = count == expected;
        ValidationCheck {
            name: "layer_count".into(),
            passed,
            message: if passed {
                format!("layer count {count} matches expected")
            } else {
                format!("layer count {count} != expected {expected}")
            },
        }
    }

    /// Check that `dtype` is one of `allowed`.
    pub fn validate_weight_dtype(
        &self,
        name: &str,
        dtype: &str,
        allowed: &[&str],
    ) -> ValidationCheck {
        let passed = allowed.contains(&dtype);
        ValidationCheck {
            name: format!("weight_dtype:{name}"),
            passed,
            message: if passed {
                format!("{name}: dtype '{dtype}' is allowed")
            } else {
                format!("{name}: dtype '{dtype}' not in {allowed:?}")
            },
        }
    }

    /// Aggregate a list of checks into a [`ValidationResult`].
    pub fn run_all_checks(&self, checks: Vec<ValidationCheck>) -> ValidationResult {
        let mut warnings = Vec::new();
        let mut errors = Vec::new();

        for check in &checks {
            if !check.passed {
                errors.push(check.message.clone());
            }
        }

        let passed = errors.is_empty();

        // Non-critical checks could be promoted to warnings in the future;
        // for now the warnings vec stays empty unless callers extend it.
        let _ = &mut warnings;

        ValidationResult { passed, checks, warnings, errors }
    }
}

impl Default for ModelValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_gguf_header() {
        let v = ModelValidator::new();
        let c = v.validate_header(GGUF_MAGIC, 3);
        assert!(c.passed);
        assert!(c.message.contains("OK"));
    }

    #[test]
    fn test_invalid_magic_number() {
        let v = ModelValidator::new();
        let c = v.validate_header(0xDEADBEEF, 3);
        assert!(!c.passed);
        assert!(c.message.contains("invalid magic"));
    }

    #[test]
    fn test_invalid_version_too_low() {
        let v = ModelValidator::new();
        let c = v.validate_header(GGUF_MAGIC, 1);
        assert!(!c.passed);
        assert!(c.message.contains("unsupported version"));
    }

    #[test]
    fn test_invalid_version_too_high() {
        let v = ModelValidator::new();
        let c = v.validate_header(GGUF_MAGIC, 4);
        assert!(!c.passed);
        assert!(c.message.contains("unsupported version"));
    }

    #[test]
    fn test_tensor_count_in_range() {
        let v = ModelValidator::new();
        let c = v.validate_tensor_count(100, 10, 500);
        assert!(c.passed);
    }

    #[test]
    fn test_tensor_count_out_of_range() {
        let v = ModelValidator::new();
        let c = v.validate_tensor_count(5, 10, 500);
        assert!(!c.passed);
        assert!(c.message.contains("outside"));
    }

    #[test]
    fn test_shape_correct_dims() {
        let v = ModelValidator::new();
        let c = v.validate_tensor_shape("attn.weight", &[4096, 4096], 2);
        assert!(c.passed);
    }

    #[test]
    fn test_shape_wrong_dims() {
        let v = ModelValidator::new();
        let c = v.validate_tensor_shape("attn.weight", &[4096, 4096], 3);
        assert!(!c.passed);
        assert!(c.message.contains("expected 3 dims"));
    }

    #[test]
    fn test_vocab_size_within_tolerance() {
        let v = ModelValidator::new();
        let c = v.validate_vocab_size(32001, 32000, 10);
        assert!(c.passed);
    }

    #[test]
    fn test_vocab_size_outside_tolerance() {
        let v = ModelValidator::new();
        let c = v.validate_vocab_size(40000, 32000, 10);
        assert!(!c.passed);
    }

    #[test]
    fn test_hidden_size_match() {
        let v = ModelValidator::new();
        let c = v.validate_hidden_size(4096, 4096);
        assert!(c.passed);
    }

    #[test]
    fn test_hidden_size_mismatch() {
        let v = ModelValidator::new();
        let c = v.validate_hidden_size(2048, 4096);
        assert!(!c.passed);
        assert!(c.message.contains("!="));
    }

    #[test]
    fn test_layer_count_match() {
        let v = ModelValidator::new();
        let c = v.validate_layer_count(32, 32);
        assert!(c.passed);
    }

    #[test]
    fn test_layer_count_mismatch() {
        let v = ModelValidator::new();
        let c = v.validate_layer_count(24, 32);
        assert!(!c.passed);
    }

    #[test]
    fn test_weight_dtype_allowed() {
        let v = ModelValidator::new();
        let c = v.validate_weight_dtype("layer.0", "f16", &["f16", "f32", "i2_s"]);
        assert!(c.passed);
    }

    #[test]
    fn test_weight_dtype_not_allowed() {
        let v = ModelValidator::new();
        let c = v.validate_weight_dtype("layer.0", "bf16", &["f16", "f32"]);
        assert!(!c.passed);
        assert!(c.message.contains("not in"));
    }

    #[test]
    fn test_run_all_checks_all_pass() {
        let v = ModelValidator::new();
        let checks = vec![
            v.validate_header(GGUF_MAGIC, 3),
            v.validate_tensor_count(100, 10, 500),
            v.validate_hidden_size(4096, 4096),
        ];
        let result = v.run_all_checks(checks);
        assert!(result.passed);
        assert_eq!(result.checks.len(), 3);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_run_all_checks_some_fail() {
        let v = ModelValidator::new();
        let checks = vec![
            v.validate_header(GGUF_MAGIC, 3),
            v.validate_hidden_size(2048, 4096),
            v.validate_layer_count(24, 32),
        ];
        let result = v.run_all_checks(checks);
        assert!(!result.passed);
        assert_eq!(result.errors.len(), 2);
        assert_eq!(result.checks.len(), 3);
    }
}
