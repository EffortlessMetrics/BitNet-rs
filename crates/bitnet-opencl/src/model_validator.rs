//! Pre-flight model validation before GPU inference.
//!
//! Validates model weights, architecture configuration, and GPU
//! compatibility before committing resources to inference.

use std::fmt;

// ── Severity & report types ──────────────────────────────────────────

/// Severity of a validation finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ValidationSeverity {
    Info,
    Warning,
    Error,
}

impl fmt::Display for ValidationSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARN"),
            Self::Error => write!(f, "ERROR"),
        }
    }
}

/// A single validation finding.
#[derive(Debug, Clone)]
pub struct ValidationFinding {
    pub severity: ValidationSeverity,
    pub message: String,
    pub suggestion: Option<String>,
}

impl fmt::Display for ValidationFinding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.severity, self.message)?;
        if let Some(ref s) = self.suggestion {
            write!(f, " (suggestion: {s})")?;
        }
        Ok(())
    }
}

/// Aggregated validation report.
#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub findings: Vec<ValidationFinding>,
}

impl ValidationReport {
    #[must_use]
    pub fn new() -> Self {
        Self { findings: Vec::new() }
    }

    /// Record a finding.
    pub fn add(
        &mut self,
        severity: ValidationSeverity,
        message: impl Into<String>,
        suggestion: Option<String>,
    ) {
        self.findings.push(ValidationFinding { severity, message: message.into(), suggestion });
    }

    /// `true` when no errors were recorded.
    #[must_use]
    pub fn passed(&self) -> bool {
        !self.findings.iter().any(|f| f.severity == ValidationSeverity::Error)
    }

    #[must_use]
    pub fn errors(&self) -> Vec<&ValidationFinding> {
        self.findings.iter().filter(|f| f.severity == ValidationSeverity::Error).collect()
    }

    #[must_use]
    pub fn warnings(&self) -> Vec<&ValidationFinding> {
        self.findings.iter().filter(|f| f.severity == ValidationSeverity::Warning).collect()
    }

    #[must_use]
    pub fn infos(&self) -> Vec<&ValidationFinding> {
        self.findings.iter().filter(|f| f.severity == ValidationSeverity::Info).collect()
    }

    /// Merge another report into this one.
    pub fn merge(&mut self, other: ValidationReport) {
        self.findings.extend(other.findings);
    }
}

impl Default for ValidationReport {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ValidationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status = if self.passed() { "PASS" } else { "FAIL" };
        writeln!(f, "Validation Report: {status}")?;
        writeln!(
            f,
            "  {} error(s), {} warning(s), {} info(s)",
            self.errors().len(),
            self.warnings().len(),
            self.infos().len(),
        )?;
        for finding in &self.findings {
            writeln!(f, "  {finding}")?;
        }
        Ok(())
    }
}

// ── Domain types (lightweight stand-ins) ─────────────────────────────

/// Simulated model weights for validation.
#[derive(Debug, Clone)]
pub struct ModelWeights {
    /// Per-layer LayerNorm weight vectors.
    pub layer_norm_weights: Vec<Vec<f32>>,
    /// Per-layer projection weight matrices (flattened, with shape).
    pub projection_weights: Vec<ProjectionWeight>,
}

/// A single projection weight matrix.
#[derive(Debug, Clone)]
pub struct ProjectionWeight {
    pub name: String,
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

/// Transformer architecture configuration.
#[derive(Debug, Clone)]
pub struct TransformerConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
    pub intermediate_size: usize,
}

/// Metadata about a loaded model.
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    /// Total model size in bytes.
    pub model_size_bytes: u64,
    /// Whether the model requires FP16 precision.
    pub requires_fp16: bool,
    /// Whether the model requires FP32 precision.
    pub requires_fp32: bool,
}

/// GPU device capabilities.
#[derive(Debug, Clone)]
pub struct GpuDeviceCapabilities {
    /// Total device memory in bytes.
    pub total_memory_bytes: u64,
    /// Available device memory in bytes.
    pub available_memory_bytes: u64,
    /// Whether the device supports FP16.
    pub supports_fp16: bool,
    /// Whether the device supports FP32.
    pub supports_fp32: bool,
    /// Device name (for diagnostics).
    pub device_name: String,
}

// ── ModelValidator ───────────────────────────────────────────────────

/// Pre-flight validator for model weights, architecture, and GPU fit.
pub struct ModelValidator {
    /// Tolerance for LayerNorm weight mean deviation from 1.0.
    pub ln_mean_tolerance: f32,
    /// Minimum acceptable RMS for projection matrices.
    pub proj_rms_min: f32,
    /// Maximum acceptable RMS for projection matrices.
    pub proj_rms_max: f32,
}

impl ModelValidator {
    #[must_use]
    pub fn new() -> Self {
        Self { ln_mean_tolerance: 0.5, proj_rms_min: 0.001, proj_rms_max: 100.0 }
    }

    /// Validate model weights (LayerNorm means, projection norms).
    #[must_use]
    pub fn validate_weights(&self, weights: &ModelWeights) -> ValidationReport {
        let mut report = ValidationReport::new();

        // Check LayerNorm weights
        for (i, ln) in weights.layer_norm_weights.iter().enumerate() {
            if ln.is_empty() {
                report.add(
                    ValidationSeverity::Error,
                    format!("Layer {i}: LayerNorm weights are empty"),
                    Some("Check model loading".into()),
                );
                continue;
            }

            let mean = ln.iter().copied().sum::<f32>() / ln.len() as f32;

            if ln.iter().all(|&v| v == 0.0) {
                report.add(
                    ValidationSeverity::Warning,
                    format!("Layer {i}: LayerNorm weights are all zero"),
                    Some("Model may not have been trained properly".into()),
                );
            } else if (mean - 1.0).abs() > self.ln_mean_tolerance {
                report.add(
                    ValidationSeverity::Warning,
                    format!(
                        "Layer {i}: LayerNorm weight mean = {mean:.4}, \
                         expected ≈ 1.0"
                    ),
                    Some("Unusual LayerNorm initialization".into()),
                );
            }

            // Check for suspiciously uniform weights
            if ln.len() > 1 {
                let variance =
                    ln.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / ln.len() as f32;
                if variance < 1e-12 && mean != 0.0 {
                    report.add(
                        ValidationSeverity::Warning,
                        format!(
                            "Layer {i}: LayerNorm weights are \
                             suspiciously uniform (var={variance:.2e})"
                        ),
                        None,
                    );
                }
            }
        }

        // Check projection matrices
        for proj in &weights.projection_weights {
            if proj.data.is_empty() {
                report.add(
                    ValidationSeverity::Error,
                    format!("Projection '{}': weight data is empty", proj.name),
                    Some("Check model loading".into()),
                );
                continue;
            }

            let rms =
                (proj.data.iter().map(|v| v * v).sum::<f32>() / proj.data.len() as f32).sqrt();

            if rms < self.proj_rms_min {
                report.add(
                    ValidationSeverity::Warning,
                    format!(
                        "Projection '{}': vanishing weights \
                         (RMS={rms:.2e})",
                        proj.name,
                    ),
                    Some("Weights may have collapsed during training".into()),
                );
            } else if rms > self.proj_rms_max {
                report.add(
                    ValidationSeverity::Warning,
                    format!(
                        "Projection '{}': exploding weights \
                         (RMS={rms:.2e})",
                        proj.name,
                    ),
                    Some("Consider weight normalization".into()),
                );
            }
        }

        if report.findings.is_empty() {
            report.add(ValidationSeverity::Info, "All weight checks passed", None);
        }

        report
    }

    /// Validate transformer architecture configuration.
    #[must_use]
    pub fn validate_architecture(&self, config: &TransformerConfig) -> ValidationReport {
        let mut report = ValidationReport::new();

        // Hidden size must be divisible by num_heads
        if config.num_heads == 0 {
            report.add(
                ValidationSeverity::Error,
                "num_heads is zero",
                Some("num_heads must be > 0".into()),
            );
        } else if !config.hidden_size.is_multiple_of(config.num_heads) {
            report.add(
                ValidationSeverity::Error,
                format!(
                    "hidden_size ({}) not divisible by num_heads ({})",
                    config.hidden_size, config.num_heads,
                ),
                Some(format!(
                    "hidden_size must be a multiple of num_heads; \
                     head_dim would be {:.2}",
                    config.hidden_size as f64 / config.num_heads as f64,
                )),
            );
        } else {
            let head_dim = config.hidden_size / config.num_heads;
            report.add(ValidationSeverity::Info, format!("head_dim = {head_dim}"), None);
        }

        // KV heads must divide evenly into attention heads (GQA)
        if config.num_kv_heads == 0 {
            report.add(
                ValidationSeverity::Error,
                "num_kv_heads is zero",
                Some("num_kv_heads must be > 0".into()),
            );
        } else if !config.num_heads.is_multiple_of(config.num_kv_heads) {
            report.add(
                ValidationSeverity::Error,
                format!(
                    "num_heads ({}) not divisible by num_kv_heads \
                     ({}) — GQA requires even division",
                    config.num_heads, config.num_kv_heads,
                ),
                Some("Adjust num_kv_heads to be a divisor of num_heads".into()),
            );
        }

        // Sanity: at least 1 layer
        if config.num_layers == 0 {
            report.add(
                ValidationSeverity::Error,
                "num_layers is zero",
                Some("Model must have at least one layer".into()),
            );
        }

        // Vocab size sanity
        if config.vocab_size == 0 {
            report.add(ValidationSeverity::Error, "vocab_size is zero", None);
        }

        if report.errors().is_empty() && report.warnings().is_empty() {
            report.add(ValidationSeverity::Info, "Architecture validation passed", None);
        }

        report
    }

    /// Validate that a model fits on the target GPU device.
    #[must_use]
    pub fn validate_gpu_compatibility(
        &self,
        model: &ModelMetadata,
        device: &GpuDeviceCapabilities,
    ) -> ValidationReport {
        let mut report = ValidationReport::new();

        // Memory check
        if model.model_size_bytes > device.available_memory_bytes {
            let model_mb = model.model_size_bytes as f64 / (1024.0 * 1024.0);
            let avail_mb = device.available_memory_bytes as f64 / (1024.0 * 1024.0);
            report.add(
                ValidationSeverity::Error,
                format!(
                    "Model ({model_mb:.0} MB) exceeds available GPU \
                     memory ({avail_mb:.0} MB) on '{}'",
                    device.device_name,
                ),
                Some("Use a smaller model or a device with more memory".into()),
            );
        } else {
            let usage_pct = if device.available_memory_bytes > 0 {
                (model.model_size_bytes as f64 / device.available_memory_bytes as f64) * 100.0
            } else {
                100.0
            };
            if usage_pct > 90.0 {
                report.add(
                    ValidationSeverity::Warning,
                    format!(
                        "Model uses {usage_pct:.1}% of available \
                         GPU memory — may cause OOM with KV cache"
                    ),
                    Some("Consider a device with more headroom".into()),
                );
            } else {
                report.add(
                    ValidationSeverity::Info,
                    format!(
                        "Model fits in GPU memory ({usage_pct:.1}% \
                         utilization)"
                    ),
                    None,
                );
            }
        }

        // Precision support
        if model.requires_fp16 && !device.supports_fp16 {
            report.add(
                ValidationSeverity::Error,
                format!("Model requires FP16 but '{}' does not support it", device.device_name,),
                Some(
                    "Use FP32 model variant or a device with FP16 \
                     support"
                        .into(),
                ),
            );
        }

        if model.requires_fp32 && !device.supports_fp32 {
            report.add(
                ValidationSeverity::Error,
                format!("Model requires FP32 but '{}' does not support it", device.device_name,),
                None,
            );
        }

        if report.errors().is_empty() && report.warnings().is_empty() {
            report.add(
                ValidationSeverity::Info,
                format!("Model compatible with '{}'", device.device_name),
                None,
            );
        }

        report
    }
}

impl Default for ModelValidator {
    fn default() -> Self {
        Self::new()
    }
}

// ── QuickValidator ──────────────────────────────────────────────────

/// Fast sanity-check validator (< 1 second for typical models).
///
/// Runs only the cheapest checks: architecture dimensions and basic
/// memory fit.  Skips per-element weight analysis.
pub struct QuickValidator;

impl QuickValidator {
    /// Run fast architecture + memory sanity checks.
    #[must_use]
    pub fn validate(
        config: &TransformerConfig,
        model: &ModelMetadata,
        device: &GpuDeviceCapabilities,
    ) -> ValidationReport {
        let v = ModelValidator::new();
        let mut report = v.validate_architecture(config);
        report.merge(v.validate_gpu_compatibility(model, device));
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_weights() -> ModelWeights {
        ModelWeights {
            layer_norm_weights: vec![vec![1.0, 1.0, 1.0, 1.0], vec![0.99, 1.01, 1.0, 1.0]],
            projection_weights: vec![ProjectionWeight {
                name: "q_proj".into(),
                data: vec![0.01, -0.02, 0.03, -0.01],
                rows: 2,
                cols: 2,
            }],
        }
    }

    fn sample_config() -> TransformerConfig {
        TransformerConfig {
            hidden_size: 2048,
            num_heads: 32,
            num_kv_heads: 8,
            num_layers: 24,
            vocab_size: 32000,
            intermediate_size: 8192,
        }
    }

    fn sample_metadata() -> ModelMetadata {
        ModelMetadata {
            model_size_bytes: 500 * 1024 * 1024, // 500 MB
            requires_fp16: true,
            requires_fp32: false,
        }
    }

    fn sample_device() -> GpuDeviceCapabilities {
        GpuDeviceCapabilities {
            total_memory_bytes: 8 * 1024 * 1024 * 1024, // 8 GB
            available_memory_bytes: 6 * 1024 * 1024 * 1024,
            supports_fp16: true,
            supports_fp32: true,
            device_name: "Test GPU".into(),
        }
    }

    #[test]
    fn valid_weights_pass() {
        let v = ModelValidator::new();
        let report = v.validate_weights(&sample_weights());
        assert!(report.passed());
    }

    #[test]
    fn valid_architecture_passes() {
        let v = ModelValidator::new();
        let report = v.validate_architecture(&sample_config());
        assert!(report.passed());
    }

    #[test]
    fn valid_gpu_compat_passes() {
        let v = ModelValidator::new();
        let report = v.validate_gpu_compatibility(&sample_metadata(), &sample_device());
        assert!(report.passed());
    }

    #[test]
    fn quick_validator_passes_valid_model() {
        let report =
            QuickValidator::validate(&sample_config(), &sample_metadata(), &sample_device());
        assert!(report.passed());
    }
}
