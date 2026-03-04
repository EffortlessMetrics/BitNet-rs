//! Model validation and sanity checking before inference.
//!
//! Provides pre-inference validation for model tensors, architecture conformance,
//! weight health, and device compatibility. All implementations are CPU reference
//! code — no OpenCL runtime required.
//!
//! # Validation levels
//!
//! - [`QuickCheck`] — header-only validation (shape, dtype, tensor count). Fast.
//! - [`FullCheck`] — scans all tensor data for NaN, Inf, zeros, extreme values.
//!
//! # Key types
//!
//! - [`ModelSpec`] — expected architecture description.
//! - [`TensorValidator`] — per-tensor shape/dtype/value checks.
//! - [`ArchitectureValidator`] — tensor names and counts match the spec.
//! - [`WeightHealthChecker`] — detects suspicious weight patterns.
//! - [`CompatibilityChecker`] — device memory and op-support checks.
//! - [`ValidationReport`] — aggregated pass/fail with severity and recommendations.

use std::collections::{HashMap, HashSet};
use std::fmt;

// ---------------------------------------------------------------------------
// ValidationSeverity
// ---------------------------------------------------------------------------

/// Severity level for a single validation finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ValidationSeverity {
    /// Informational note — not a problem.
    Info,
    /// Potential issue that may degrade quality.
    Warning,
    /// Critical problem that will prevent correct inference.
    Error,
}

impl fmt::Display for ValidationSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARNING"),
            Self::Error => write!(f, "ERROR"),
        }
    }
}

// ---------------------------------------------------------------------------
// ValidationFinding
// ---------------------------------------------------------------------------

/// A single validation finding attached to a tensor or the model as a whole.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidationFinding {
    /// Severity of this finding.
    pub severity: ValidationSeverity,
    /// Which tensor (if any) this finding relates to.
    pub tensor_name: Option<String>,
    /// Human-readable description of the issue.
    pub message: String,
    /// Optional recommendation for how to fix the issue.
    pub recommendation: Option<String>,
}

impl fmt::Display for ValidationFinding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let tensor = self.tensor_name.as_deref().unwrap_or("model");
        write!(f, "[{}] {}: {}", self.severity, tensor, self.message)?;
        if let Some(rec) = &self.recommendation {
            write!(f, " (recommendation: {rec})")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ValidationReport
// ---------------------------------------------------------------------------

/// Aggregated results from one or more validation passes.
#[derive(Debug, Clone, Default)]
pub struct ValidationReport {
    /// All findings collected during validation.
    pub findings: Vec<ValidationFinding>,
}

impl ValidationReport {
    /// Create an empty report.
    pub fn new() -> Self {
        Self { findings: Vec::new() }
    }

    /// Add a finding.
    pub fn add(&mut self, finding: ValidationFinding) {
        self.findings.push(finding);
    }

    /// Merge another report into this one.
    pub fn merge(&mut self, other: ValidationReport) {
        self.findings.extend(other.findings);
    }

    /// Returns `true` when there are no Error-level findings.
    pub fn passed(&self) -> bool {
        !self.findings.iter().any(|f| f.severity == ValidationSeverity::Error)
    }

    /// Count of findings at the given severity.
    pub fn count(&self, severity: ValidationSeverity) -> usize {
        self.findings.iter().filter(|f| f.severity == severity).count()
    }

    /// All error-level findings.
    pub fn errors(&self) -> Vec<&ValidationFinding> {
        self.findings.iter().filter(|f| f.severity == ValidationSeverity::Error).collect()
    }

    /// All warning-level findings.
    pub fn warnings(&self) -> Vec<&ValidationFinding> {
        self.findings.iter().filter(|f| f.severity == ValidationSeverity::Warning).collect()
    }
}

impl fmt::Display for ValidationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let errors = self.count(ValidationSeverity::Error);
        let warnings = self.count(ValidationSeverity::Warning);
        let infos = self.count(ValidationSeverity::Info);
        writeln!(
            f,
            "Validation: {} error(s), {} warning(s), {} info(s) — {}",
            errors,
            warnings,
            infos,
            if self.passed() { "PASSED" } else { "FAILED" }
        )?;
        for finding in &self.findings {
            writeln!(f, "  {finding}")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// TensorDtype — lightweight dtype enum (no OpenCL dependency)
// ---------------------------------------------------------------------------

/// Lightweight data-type tag for tensor elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorDtype {
    F32,
    F16,
    BF16,
    I8,
    I2,
    U8,
}

impl TensorDtype {
    /// Size of a single element in bytes.
    pub fn element_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::I8 | Self::U8 => 1,
            // I2 packs 4 values per byte, but we report the per-element
            // logical size here; callers must account for packing externally.
            Self::I2 => 1,
        }
    }
}

impl fmt::Display for TensorDtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F32 => write!(f, "f32"),
            Self::F16 => write!(f, "f16"),
            Self::BF16 => write!(f, "bf16"),
            Self::I8 => write!(f, "i8"),
            Self::I2 => write!(f, "i2"),
            Self::U8 => write!(f, "u8"),
        }
    }
}

// ---------------------------------------------------------------------------
// TensorInfo — metadata for one tensor in the model
// ---------------------------------------------------------------------------

/// Metadata describing a single tensor in the model file.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorInfo {
    /// Tensor name (e.g. `"blk.0.attn_q.weight"`).
    pub name: String,
    /// Shape dimensions (e.g. `[2560, 2560]`).
    pub shape: Vec<usize>,
    /// Element data type.
    pub dtype: TensorDtype,
    /// Optional: raw data for full-scan validation.
    pub data: Option<Vec<f32>>,
}

impl TensorInfo {
    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Estimated bytes (without I2 packing adjustment).
    pub fn size_bytes(&self) -> usize {
        self.numel() * self.dtype.element_bytes()
    }
}

// ---------------------------------------------------------------------------
// ModelSpec — expected architecture description
// ---------------------------------------------------------------------------

/// Describes the expected architecture so validators can check conformance.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelSpec {
    /// Number of transformer layers.
    pub num_layers: u32,
    /// Hidden dimension (embedding width).
    pub hidden_dim: u32,
    /// Vocabulary size.
    pub vocab_size: u32,
    /// Number of attention heads.
    pub num_heads: u32,
    /// Optional: intermediate (FFN) dimension. Defaults to 4 × hidden_dim.
    pub intermediate_dim: Option<u32>,
    /// Expected quantization dtype for weight tensors.
    pub weight_dtype: TensorDtype,
}

impl ModelSpec {
    /// Effective intermediate dimension.
    pub fn ffn_dim(&self) -> u32 {
        self.intermediate_dim.unwrap_or(self.hidden_dim * 4)
    }

    /// Head dimension.
    pub fn head_dim(&self) -> u32 {
        if self.num_heads == 0 {
            return 0;
        }
        self.hidden_dim / self.num_heads
    }

    /// Generate the set of expected tensor names for a standard BitNet
    /// transformer (attention Q/K/V/O, FFN gate/up/down, norms, embeddings).
    pub fn expected_tensor_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        // Token embedding
        names.push("token_embd.weight".into());
        // Per-layer tensors
        for i in 0..self.num_layers {
            for suffix in &[
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_output.weight",
                "attn_norm.weight",
                "ffn_gate.weight",
                "ffn_up.weight",
                "ffn_down.weight",
                "ffn_norm.weight",
            ] {
                names.push(format!("blk.{i}.{suffix}"));
            }
        }
        // Output norm + head
        names.push("output_norm.weight".into());
        names.push("output.weight".into());
        names
    }

    /// Estimate total model memory in bytes (rough).
    pub fn estimated_memory_bytes(&self) -> u64 {
        let h = self.hidden_dim as u64;
        let v = self.vocab_size as u64;
        let l = self.num_layers as u64;
        let ffn = self.ffn_dim() as u64;
        let elem = self.weight_dtype.element_bytes() as u64;

        // Embedding: v * h
        // Per-layer: 4*h*h (attn) + 3*h*ffn (FFN) + 2*h (norms, f32)
        // Output: h*v + h (norm)
        let embedding = v * h * elem;
        let per_layer = (4 * h * h + 3 * h * ffn) * elem + 2 * h * 4;
        let output = h * v * elem + h * 4;
        embedding + l * per_layer + output
    }

    /// Convenience builder for a BitNet-2B-style model.
    pub fn bitnet_2b() -> Self {
        Self {
            num_layers: 24,
            hidden_dim: 2560,
            vocab_size: 32000,
            num_heads: 32,
            intermediate_dim: None,
            weight_dtype: TensorDtype::I2,
        }
    }

    /// Tiny test model for unit tests.
    pub fn tiny_test() -> Self {
        Self {
            num_layers: 2,
            hidden_dim: 64,
            vocab_size: 256,
            num_heads: 4,
            intermediate_dim: None,
            weight_dtype: TensorDtype::I2,
        }
    }

    /// Single-layer model for edge-case tests.
    pub fn single_layer() -> Self {
        Self {
            num_layers: 1,
            hidden_dim: 32,
            vocab_size: 128,
            num_heads: 2,
            intermediate_dim: None,
            weight_dtype: TensorDtype::I2,
        }
    }
}

// ---------------------------------------------------------------------------
// DeviceCapabilities — what the target device supports
// ---------------------------------------------------------------------------

/// Describes the capabilities of the target inference device.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceCapabilities {
    /// Total device memory in bytes (e.g. VRAM for GPU, RAM for CPU).
    pub total_memory_bytes: u64,
    /// Available (free) device memory in bytes.
    pub available_memory_bytes: u64,
    /// Supported element dtypes.
    pub supported_dtypes: HashSet<TensorDtype>,
    /// Maximum single-allocation size in bytes.
    pub max_allocation_bytes: u64,
    /// Device name for diagnostics.
    pub device_name: String,
}

impl DeviceCapabilities {
    /// A770-like defaults (16 GB VRAM).
    pub fn a770_default() -> Self {
        let mut dtypes = HashSet::new();
        dtypes.insert(TensorDtype::F32);
        dtypes.insert(TensorDtype::F16);
        dtypes.insert(TensorDtype::I8);
        dtypes.insert(TensorDtype::I2);
        dtypes.insert(TensorDtype::U8);
        dtypes.insert(TensorDtype::BF16);
        Self {
            total_memory_bytes: 16 * 1024 * 1024 * 1024,
            available_memory_bytes: 14 * 1024 * 1024 * 1024,
            supported_dtypes: dtypes,
            max_allocation_bytes: 4 * 1024 * 1024 * 1024,
            device_name: "Intel Arc A770".into(),
        }
    }

    /// Tiny device for testing OOM paths.
    pub fn tiny_device() -> Self {
        let mut dtypes = HashSet::new();
        dtypes.insert(TensorDtype::F32);
        dtypes.insert(TensorDtype::I2);
        Self {
            total_memory_bytes: 1024 * 1024, // 1 MB
            available_memory_bytes: 512 * 1024,
            supported_dtypes: dtypes,
            max_allocation_bytes: 256 * 1024,
            device_name: "TinyTestDevice".into(),
        }
    }
}

// ---------------------------------------------------------------------------
// TensorValidator
// ---------------------------------------------------------------------------

/// Validates individual tensor shape, dtype, and data health.
pub struct TensorValidator;

impl TensorValidator {
    /// Check tensor shape against an expected shape.
    pub fn check_shape(tensor: &TensorInfo, expected: &[usize], report: &mut ValidationReport) {
        if tensor.shape != expected {
            report.add(ValidationFinding {
                severity: ValidationSeverity::Error,
                tensor_name: Some(tensor.name.clone()),
                message: format!("shape mismatch: expected {:?}, got {:?}", expected, tensor.shape),
                recommendation: Some(
                    "verify model export matched the expected architecture".into(),
                ),
            });
        }
    }

    /// Check tensor dtype against an expected dtype.
    pub fn check_dtype(tensor: &TensorInfo, expected: TensorDtype, report: &mut ValidationReport) {
        if tensor.dtype != expected {
            report.add(ValidationFinding {
                severity: ValidationSeverity::Error,
                tensor_name: Some(tensor.name.clone()),
                message: format!("dtype mismatch: expected {}, got {}", expected, tensor.dtype),
                recommendation: Some("re-quantize the model with the correct dtype".into()),
            });
        }
    }

    /// Scan f32 data for NaN values.
    pub fn check_nan(tensor: &TensorInfo, report: &mut ValidationReport) {
        if let Some(data) = &tensor.data {
            let nan_count = data.iter().filter(|v| v.is_nan()).count();
            if nan_count > 0 {
                report.add(ValidationFinding {
                    severity: ValidationSeverity::Error,
                    tensor_name: Some(tensor.name.clone()),
                    message: format!(
                        "contains {} NaN value(s) out of {} elements",
                        nan_count,
                        data.len()
                    ),
                    recommendation: Some(
                        "re-export or re-train; NaN weights cause garbage output".into(),
                    ),
                });
            }
        }
    }

    /// Scan f32 data for Inf values.
    pub fn check_inf(tensor: &TensorInfo, report: &mut ValidationReport) {
        if let Some(data) = &tensor.data {
            let inf_count = data.iter().filter(|v| v.is_infinite()).count();
            if inf_count > 0 {
                report.add(ValidationFinding {
                    severity: ValidationSeverity::Error,
                    tensor_name: Some(tensor.name.clone()),
                    message: format!(
                        "contains {} Inf value(s) out of {} elements",
                        inf_count,
                        data.len()
                    ),
                    recommendation: Some(
                        "re-export or re-train; Inf weights cause divergence".into(),
                    ),
                });
            }
        }
    }

    /// Check that values fall within `[min_val, max_val]`.
    pub fn check_range(
        tensor: &TensorInfo,
        min_val: f32,
        max_val: f32,
        report: &mut ValidationReport,
    ) {
        if let Some(data) = &tensor.data {
            let oob: usize =
                data.iter().filter(|v| !v.is_nan() && (**v < min_val || **v > max_val)).count();
            if oob > 0 {
                report.add(ValidationFinding {
                    severity: ValidationSeverity::Warning,
                    tensor_name: Some(tensor.name.clone()),
                    message: format!("{} value(s) outside [{}, {}]", oob, min_val, max_val),
                    recommendation: Some("check quantization or training for outliers".into()),
                });
            }
        }
    }

    /// Run all per-tensor data checks (NaN, Inf, range).
    pub fn full_data_check(tensor: &TensorInfo, range: (f32, f32), report: &mut ValidationReport) {
        Self::check_nan(tensor, report);
        Self::check_inf(tensor, report);
        Self::check_range(tensor, range.0, range.1, report);
    }
}

// ---------------------------------------------------------------------------
// ArchitectureValidator
// ---------------------------------------------------------------------------

/// Validates that the tensor set matches the expected architecture.
pub struct ArchitectureValidator;

impl ArchitectureValidator {
    /// Check that every expected tensor name is present.
    pub fn check_tensor_names(
        spec: &ModelSpec,
        tensors: &[TensorInfo],
        report: &mut ValidationReport,
    ) {
        let expected: HashSet<String> = spec.expected_tensor_names().into_iter().collect();
        let actual: HashSet<String> = tensors.iter().map(|t| t.name.clone()).collect();

        for name in &expected {
            if !actual.contains(name) {
                report.add(ValidationFinding {
                    severity: ValidationSeverity::Error,
                    tensor_name: Some(name.clone()),
                    message: "missing expected tensor".into(),
                    recommendation: Some("ensure model file contains all required weights".into()),
                });
            }
        }

        for name in &actual {
            if !expected.contains(name) {
                report.add(ValidationFinding {
                    severity: ValidationSeverity::Info,
                    tensor_name: Some(name.clone()),
                    message: "unexpected extra tensor".into(),
                    recommendation: None,
                });
            }
        }
    }

    /// Check that the total number of tensors matches expectations.
    pub fn check_tensor_count(
        spec: &ModelSpec,
        tensors: &[TensorInfo],
        report: &mut ValidationReport,
    ) {
        let expected = spec.expected_tensor_names().len();
        let actual = tensors.len();
        if actual < expected {
            report.add(ValidationFinding {
                severity: ValidationSeverity::Error,
                tensor_name: None,
                message: format!("tensor count too low: expected {}, got {}", expected, actual),
                recommendation: Some("model file may be truncated".into()),
            });
        } else if actual > expected {
            report.add(ValidationFinding {
                severity: ValidationSeverity::Info,
                tensor_name: None,
                message: format!("extra tensors present: expected {}, got {}", expected, actual),
                recommendation: None,
            });
        }
    }

    /// Validate per-layer tensor shapes against the spec.
    pub fn check_layer_shapes(
        spec: &ModelSpec,
        tensors: &[TensorInfo],
        report: &mut ValidationReport,
    ) {
        let h = spec.hidden_dim as usize;
        let ffn = spec.ffn_dim() as usize;
        let v = spec.vocab_size as usize;

        let by_name: HashMap<&str, &TensorInfo> =
            tensors.iter().map(|t| (t.name.as_str(), t)).collect();

        // Embedding
        if let Some(t) = by_name.get("token_embd.weight") {
            TensorValidator::check_shape(t, &[v, h], report);
        }

        // Output
        if let Some(t) = by_name.get("output.weight") {
            TensorValidator::check_shape(t, &[v, h], report);
        }

        // Per-layer checks
        for layer in 0..spec.num_layers {
            let prefix = format!("blk.{layer}");

            let attn_shapes: &[(&str, &[usize])] = &[
                ("attn_q.weight", &[h, h]),
                ("attn_k.weight", &[h, h]),
                ("attn_v.weight", &[h, h]),
                ("attn_output.weight", &[h, h]),
            ];
            for (suffix, shape) in attn_shapes {
                let key = format!("{prefix}.{suffix}");
                if let Some(t) = by_name.get(key.as_str()) {
                    TensorValidator::check_shape(t, shape, report);
                }
            }

            let ffn_shapes: &[(&str, &[usize])] = &[
                ("ffn_gate.weight", &[ffn, h]),
                ("ffn_up.weight", &[ffn, h]),
                ("ffn_down.weight", &[h, ffn]),
            ];
            for (suffix, shape) in ffn_shapes {
                let key = format!("{prefix}.{suffix}");
                if let Some(t) = by_name.get(key.as_str()) {
                    TensorValidator::check_shape(t, shape, report);
                }
            }

            // Norm weights are 1-D [hidden_dim]
            for suffix in &["attn_norm.weight", "ffn_norm.weight"] {
                let key = format!("{prefix}.{suffix}");
                if let Some(t) = by_name.get(key.as_str()) {
                    TensorValidator::check_shape(t, &[h], report);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// WeightHealthChecker
// ---------------------------------------------------------------------------

/// Detects suspicious weight patterns that indicate a broken model.
pub struct WeightHealthChecker;

impl WeightHealthChecker {
    /// Check for all-zero tensors.
    pub fn check_all_zeros(tensor: &TensorInfo, report: &mut ValidationReport) {
        if let Some(data) = &tensor.data
            && !data.is_empty()
            && data.iter().all(|&v| v == 0.0)
        {
            report.add(ValidationFinding {
                severity: ValidationSeverity::Warning,
                tensor_name: Some(tensor.name.clone()),
                message: "all weights are zero".into(),
                recommendation: Some("tensor may not have been initialized during training".into()),
            });
        }
    }

    /// Check for extreme values (outside ±`threshold`).
    pub fn check_extreme_values(
        tensor: &TensorInfo,
        threshold: f32,
        report: &mut ValidationReport,
    ) {
        if let Some(data) = &tensor.data {
            let extreme_count = data.iter().filter(|v| v.abs() > threshold).count();
            if extreme_count > 0 {
                let pct = (extreme_count as f64 / data.len().max(1) as f64) * 100.0;
                report.add(ValidationFinding {
                    severity: if pct > 1.0 {
                        ValidationSeverity::Warning
                    } else {
                        ValidationSeverity::Info
                    },
                    tensor_name: Some(tensor.name.clone()),
                    message: format!(
                        "{} extreme value(s) (|v| > {}) — {:.2}% of elements",
                        extreme_count, threshold, pct
                    ),
                    recommendation: Some("large weights may indicate training instability".into()),
                });
            }
        }
    }

    /// Check for low variance (all values nearly identical).
    pub fn check_low_variance(
        tensor: &TensorInfo,
        min_variance: f32,
        report: &mut ValidationReport,
    ) {
        if let Some(data) = &tensor.data {
            if data.len() < 2 {
                return;
            }
            let n = data.len() as f64;
            let mean = data.iter().map(|&v| v as f64).sum::<f64>() / n;
            let var = data.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
            if (var as f32) < min_variance {
                report.add(ValidationFinding {
                    severity: ValidationSeverity::Warning,
                    tensor_name: Some(tensor.name.clone()),
                    message: format!("very low variance ({:.2e}), weights may be degenerate", var),
                    recommendation: Some("model may not have converged during training".into()),
                });
            }
        }
    }

    /// Run all health checks on a single tensor.
    pub fn full_health_check(
        tensor: &TensorInfo,
        extreme_threshold: f32,
        min_variance: f32,
        report: &mut ValidationReport,
    ) {
        Self::check_all_zeros(tensor, report);
        Self::check_extreme_values(tensor, extreme_threshold, report);
        Self::check_low_variance(tensor, min_variance, report);
    }
}

// ---------------------------------------------------------------------------
// CompatibilityChecker
// ---------------------------------------------------------------------------

/// Verifies that a model is compatible with the target device.
pub struct CompatibilityChecker;

impl CompatibilityChecker {
    /// Check whether the model fits in device memory.
    pub fn check_memory_fit(
        spec: &ModelSpec,
        device: &DeviceCapabilities,
        report: &mut ValidationReport,
    ) {
        let model_bytes = spec.estimated_memory_bytes();
        if model_bytes > device.available_memory_bytes {
            report.add(ValidationFinding {
                severity: ValidationSeverity::Error,
                tensor_name: None,
                message: format!(
                    "model requires ~{:.1} MB but device '{}' has {:.1} MB available",
                    model_bytes as f64 / (1024.0 * 1024.0),
                    device.device_name,
                    device.available_memory_bytes as f64 / (1024.0 * 1024.0),
                ),
                recommendation: Some("use a smaller model or a device with more memory".into()),
            });
        } else {
            let utilization = model_bytes as f64 / device.available_memory_bytes as f64;
            if utilization > 0.9 {
                report.add(ValidationFinding {
                    severity: ValidationSeverity::Warning,
                    tensor_name: None,
                    message: format!(
                        "model uses {:.0}% of available memory — little headroom for KV cache",
                        utilization * 100.0
                    ),
                    recommendation: Some("consider reducing context length or batch size".into()),
                });
            }
        }
    }

    /// Check that the model's weight dtype is supported by the device.
    pub fn check_dtype_support(
        spec: &ModelSpec,
        device: &DeviceCapabilities,
        report: &mut ValidationReport,
    ) {
        if !device.supported_dtypes.contains(&spec.weight_dtype) {
            report.add(ValidationFinding {
                severity: ValidationSeverity::Error,
                tensor_name: None,
                message: format!(
                    "device '{}' does not support {} weights",
                    device.device_name, spec.weight_dtype,
                ),
                recommendation: Some("use a model quantized to a supported dtype".into()),
            });
        }
    }

    /// Check that no single tensor exceeds the device max allocation.
    pub fn check_allocation_limits(
        tensors: &[TensorInfo],
        device: &DeviceCapabilities,
        report: &mut ValidationReport,
    ) {
        for t in tensors {
            let bytes = t.size_bytes() as u64;
            if bytes > device.max_allocation_bytes {
                report.add(ValidationFinding {
                    severity: ValidationSeverity::Error,
                    tensor_name: Some(t.name.clone()),
                    message: format!(
                        "tensor size ({:.1} MB) exceeds device max allocation ({:.1} MB)",
                        bytes as f64 / (1024.0 * 1024.0),
                        device.max_allocation_bytes as f64 / (1024.0 * 1024.0),
                    ),
                    recommendation: Some("split large tensors or use tensor parallelism".into()),
                });
            }
        }
    }

    /// Run all compatibility checks.
    pub fn full_compatibility_check(
        spec: &ModelSpec,
        tensors: &[TensorInfo],
        device: &DeviceCapabilities,
        report: &mut ValidationReport,
    ) {
        Self::check_memory_fit(spec, device, report);
        Self::check_dtype_support(spec, device, report);
        Self::check_allocation_limits(tensors, device, report);
    }
}

// ---------------------------------------------------------------------------
// QuickCheck — header-only validation
// ---------------------------------------------------------------------------

/// Fast validation that inspects only tensor metadata (shape, dtype, names).
/// Does not read tensor data.
pub struct QuickCheck;

impl QuickCheck {
    /// Run a quick (header-only) validation pass.
    pub fn run(spec: &ModelSpec, tensors: &[TensorInfo]) -> ValidationReport {
        let mut report = ValidationReport::new();

        // Architecture conformance
        ArchitectureValidator::check_tensor_names(spec, tensors, &mut report);
        ArchitectureValidator::check_tensor_count(spec, tensors, &mut report);
        ArchitectureValidator::check_layer_shapes(spec, tensors, &mut report);

        // Dtype checks
        for t in tensors {
            // Norm tensors are expected to be f32
            if t.name.contains("_norm.") {
                TensorValidator::check_dtype(t, TensorDtype::F32, &mut report);
            }
        }

        report
    }
}

// ---------------------------------------------------------------------------
// FullCheck — thorough data-scan validation
// ---------------------------------------------------------------------------

/// Thorough validation that scans all tensor data.
pub struct FullCheck;

impl FullCheck {
    /// Default extreme-value threshold for weight health.
    const DEFAULT_EXTREME_THRESHOLD: f32 = 100.0;
    /// Default minimum variance for weight health.
    const DEFAULT_MIN_VARIANCE: f32 = 1e-10;
    /// Default valid value range for tensor data.
    const DEFAULT_RANGE: (f32, f32) = (-1000.0, 1000.0);

    /// Run a full validation pass (header checks + data scan).
    pub fn run(
        spec: &ModelSpec,
        tensors: &[TensorInfo],
        device: Option<&DeviceCapabilities>,
    ) -> ValidationReport {
        let mut report = QuickCheck::run(spec, tensors);

        // Data-level checks
        for t in tensors {
            TensorValidator::full_data_check(t, Self::DEFAULT_RANGE, &mut report);
            WeightHealthChecker::full_health_check(
                t,
                Self::DEFAULT_EXTREME_THRESHOLD,
                Self::DEFAULT_MIN_VARIANCE,
                &mut report,
            );
        }

        // Device compatibility
        if let Some(dev) = device {
            CompatibilityChecker::full_compatibility_check(spec, tensors, dev, &mut report);
        }

        report
    }
}

// ---------------------------------------------------------------------------
// Helper: build a valid tensor set from a spec (for testing)
// ---------------------------------------------------------------------------

/// Build a minimal valid tensor set matching `spec` with clean data.
pub fn build_valid_tensors(spec: &ModelSpec) -> Vec<TensorInfo> {
    let h = spec.hidden_dim as usize;
    let ffn = spec.ffn_dim() as usize;
    let v = spec.vocab_size as usize;

    let mut tensors = Vec::new();

    // Token embedding
    tensors.push(TensorInfo {
        name: "token_embd.weight".into(),
        shape: vec![v, h],
        dtype: spec.weight_dtype,
        data: Some(vec![0.01; v * h]),
    });

    for i in 0..spec.num_layers {
        // Attention
        for suffix in &["attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight"] {
            tensors.push(TensorInfo {
                name: format!("blk.{i}.{suffix}"),
                shape: vec![h, h],
                dtype: spec.weight_dtype,
                data: Some(vec![0.02; h * h]),
            });
        }
        // Attention norm (f32)
        tensors.push(TensorInfo {
            name: format!("blk.{i}.attn_norm.weight"),
            shape: vec![h],
            dtype: TensorDtype::F32,
            data: Some(vec![1.0; h]),
        });
        // FFN
        for suffix in &["ffn_gate.weight", "ffn_up.weight"] {
            tensors.push(TensorInfo {
                name: format!("blk.{i}.{suffix}"),
                shape: vec![ffn, h],
                dtype: spec.weight_dtype,
                data: Some(vec![0.03; ffn * h]),
            });
        }
        tensors.push(TensorInfo {
            name: format!("blk.{i}.ffn_down.weight"),
            shape: vec![h, ffn],
            dtype: spec.weight_dtype,
            data: Some(vec![0.03; h * ffn]),
        });
        // FFN norm (f32)
        tensors.push(TensorInfo {
            name: format!("blk.{i}.ffn_norm.weight"),
            shape: vec![h],
            dtype: TensorDtype::F32,
            data: Some(vec![1.0; h]),
        });
    }

    // Output norm + head
    tensors.push(TensorInfo {
        name: "output_norm.weight".into(),
        shape: vec![h],
        dtype: TensorDtype::F32,
        data: Some(vec![1.0; h]),
    });
    tensors.push(TensorInfo {
        name: "output.weight".into(),
        shape: vec![v, h],
        dtype: spec.weight_dtype,
        data: Some(vec![0.01; v * h]),
    });

    tensors
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers ----------------------------------------------------------

    fn tiny_spec() -> ModelSpec {
        ModelSpec::tiny_test()
    }

    fn single_layer_spec() -> ModelSpec {
        ModelSpec::single_layer()
    }

    fn valid_tensors(spec: &ModelSpec) -> Vec<TensorInfo> {
        build_valid_tensors(spec)
    }

    fn make_tensor(name: &str, shape: Vec<usize>, data: Vec<f32>) -> TensorInfo {
        TensorInfo { name: name.into(), shape, dtype: TensorDtype::F32, data: Some(data) }
    }

    // =====================================================================
    // ValidationSeverity ordering
    // =====================================================================

    #[test]
    fn severity_ordering() {
        assert!(ValidationSeverity::Info < ValidationSeverity::Warning);
        assert!(ValidationSeverity::Warning < ValidationSeverity::Error);
    }

    #[test]
    fn severity_display() {
        assert_eq!(format!("{}", ValidationSeverity::Info), "INFO");
        assert_eq!(format!("{}", ValidationSeverity::Warning), "WARNING");
        assert_eq!(format!("{}", ValidationSeverity::Error), "ERROR");
    }

    // =====================================================================
    // ValidationReport
    // =====================================================================

    #[test]
    fn empty_report_passes() {
        let report = ValidationReport::new();
        assert!(report.passed());
        assert_eq!(report.count(ValidationSeverity::Error), 0);
    }

    #[test]
    fn report_with_warning_still_passes() {
        let mut report = ValidationReport::new();
        report.add(ValidationFinding {
            severity: ValidationSeverity::Warning,
            tensor_name: None,
            message: "minor issue".into(),
            recommendation: None,
        });
        assert!(report.passed());
        assert_eq!(report.count(ValidationSeverity::Warning), 1);
    }

    #[test]
    fn report_with_error_fails() {
        let mut report = ValidationReport::new();
        report.add(ValidationFinding {
            severity: ValidationSeverity::Error,
            tensor_name: Some("blk.0.attn_q.weight".into()),
            message: "bad tensor".into(),
            recommendation: Some("fix it".into()),
        });
        assert!(!report.passed());
        assert_eq!(report.errors().len(), 1);
    }

    #[test]
    fn report_merge() {
        let mut r1 = ValidationReport::new();
        r1.add(ValidationFinding {
            severity: ValidationSeverity::Warning,
            tensor_name: None,
            message: "w1".into(),
            recommendation: None,
        });
        let mut r2 = ValidationReport::new();
        r2.add(ValidationFinding {
            severity: ValidationSeverity::Error,
            tensor_name: None,
            message: "e1".into(),
            recommendation: None,
        });
        r1.merge(r2);
        assert_eq!(r1.findings.len(), 2);
        assert!(!r1.passed());
    }

    #[test]
    fn report_display_contains_status() {
        let report = ValidationReport::new();
        let s = format!("{report}");
        assert!(s.contains("PASSED"));
    }

    #[test]
    fn report_display_failed() {
        let mut report = ValidationReport::new();
        report.add(ValidationFinding {
            severity: ValidationSeverity::Error,
            tensor_name: None,
            message: "fail".into(),
            recommendation: None,
        });
        let s = format!("{report}");
        assert!(s.contains("FAILED"));
    }

    // =====================================================================
    // TensorDtype
    // =====================================================================

    #[test]
    fn dtype_element_bytes() {
        assert_eq!(TensorDtype::F32.element_bytes(), 4);
        assert_eq!(TensorDtype::F16.element_bytes(), 2);
        assert_eq!(TensorDtype::BF16.element_bytes(), 2);
        assert_eq!(TensorDtype::I8.element_bytes(), 1);
        assert_eq!(TensorDtype::U8.element_bytes(), 1);
        assert_eq!(TensorDtype::I2.element_bytes(), 1);
    }

    #[test]
    fn dtype_display() {
        assert_eq!(format!("{}", TensorDtype::F32), "f32");
        assert_eq!(format!("{}", TensorDtype::I2), "i2");
    }

    // =====================================================================
    // TensorInfo
    // =====================================================================

    #[test]
    fn tensor_info_numel() {
        let t = TensorInfo {
            name: "x".into(),
            shape: vec![3, 4, 5],
            dtype: TensorDtype::F32,
            data: None,
        };
        assert_eq!(t.numel(), 60);
    }

    #[test]
    fn tensor_info_size_bytes() {
        let t = TensorInfo {
            name: "x".into(),
            shape: vec![10, 20],
            dtype: TensorDtype::F32,
            data: None,
        };
        assert_eq!(t.size_bytes(), 800);
    }

    #[test]
    fn tensor_info_empty_shape() {
        let t = TensorInfo {
            name: "scalar".into(),
            shape: vec![],
            dtype: TensorDtype::F32,
            data: None,
        };
        // Product of empty iterator is 1
        assert_eq!(t.numel(), 1);
    }

    // =====================================================================
    // ModelSpec
    // =====================================================================

    #[test]
    fn model_spec_ffn_dim_default() {
        let spec = tiny_spec();
        assert_eq!(spec.ffn_dim(), 64 * 4);
    }

    #[test]
    fn model_spec_ffn_dim_custom() {
        let mut spec = tiny_spec();
        spec.intermediate_dim = Some(128);
        assert_eq!(spec.ffn_dim(), 128);
    }

    #[test]
    fn model_spec_head_dim() {
        let spec = tiny_spec();
        assert_eq!(spec.head_dim(), 64 / 4);
    }

    #[test]
    fn model_spec_head_dim_zero_heads() {
        let mut spec = tiny_spec();
        spec.num_heads = 0;
        assert_eq!(spec.head_dim(), 0);
    }

    #[test]
    fn model_spec_expected_tensor_names_count() {
        let spec = tiny_spec();
        // 1 (embd) + 9*num_layers + 2 (output norm + head)
        let expected = 1 + 9 * spec.num_layers as usize + 2;
        assert_eq!(spec.expected_tensor_names().len(), expected);
    }

    #[test]
    fn model_spec_expected_names_single_layer() {
        let spec = single_layer_spec();
        let names = spec.expected_tensor_names();
        assert!(names.contains(&"token_embd.weight".to_string()));
        assert!(names.contains(&"blk.0.attn_q.weight".to_string()));
        assert!(names.contains(&"output.weight".to_string()));
        assert!(!names.contains(&"blk.1.attn_q.weight".to_string()));
    }

    #[test]
    fn model_spec_estimated_memory_positive() {
        let spec = tiny_spec();
        assert!(spec.estimated_memory_bytes() > 0);
    }

    #[test]
    fn bitnet_2b_spec_layers() {
        let spec = ModelSpec::bitnet_2b();
        assert_eq!(spec.num_layers, 24);
        assert_eq!(spec.hidden_dim, 2560);
    }

    // =====================================================================
    // TensorValidator — shape
    // =====================================================================

    #[test]
    fn shape_match_passes() {
        let t = make_tensor("t", vec![4, 8], vec![0.0; 32]);
        let mut report = ValidationReport::new();
        TensorValidator::check_shape(&t, &[4, 8], &mut report);
        assert!(report.passed());
        assert_eq!(report.findings.len(), 0);
    }

    #[test]
    fn shape_mismatch_detected() {
        let t = make_tensor("t", vec![4, 8], vec![0.0; 32]);
        let mut report = ValidationReport::new();
        TensorValidator::check_shape(&t, &[8, 4], &mut report);
        assert!(!report.passed());
        assert!(report.findings[0].message.contains("shape mismatch"));
    }

    // =====================================================================
    // TensorValidator — dtype
    // =====================================================================

    #[test]
    fn dtype_match_passes() {
        let t = make_tensor("t", vec![4], vec![1.0; 4]);
        let mut report = ValidationReport::new();
        TensorValidator::check_dtype(&t, TensorDtype::F32, &mut report);
        assert!(report.passed());
    }

    #[test]
    fn dtype_mismatch_detected() {
        let t = make_tensor("t", vec![4], vec![1.0; 4]);
        let mut report = ValidationReport::new();
        TensorValidator::check_dtype(&t, TensorDtype::I8, &mut report);
        assert!(!report.passed());
    }

    // =====================================================================
    // TensorValidator — NaN
    // =====================================================================

    #[test]
    fn nan_detected() {
        let t = make_tensor("t", vec![4], vec![1.0, f32::NAN, 2.0, f32::NAN]);
        let mut report = ValidationReport::new();
        TensorValidator::check_nan(&t, &mut report);
        assert!(!report.passed());
        assert!(report.findings[0].message.contains("2 NaN"));
    }

    #[test]
    fn no_nan_passes() {
        let t = make_tensor("t", vec![4], vec![1.0, 2.0, 3.0, 4.0]);
        let mut report = ValidationReport::new();
        TensorValidator::check_nan(&t, &mut report);
        assert!(report.passed());
    }

    #[test]
    fn nan_check_no_data_passes() {
        let t =
            TensorInfo { name: "t".into(), shape: vec![4], dtype: TensorDtype::F32, data: None };
        let mut report = ValidationReport::new();
        TensorValidator::check_nan(&t, &mut report);
        assert!(report.passed());
    }

    // =====================================================================
    // TensorValidator — Inf
    // =====================================================================

    #[test]
    fn inf_detected() {
        let t = make_tensor("t", vec![3], vec![1.0, f32::INFINITY, f32::NEG_INFINITY]);
        let mut report = ValidationReport::new();
        TensorValidator::check_inf(&t, &mut report);
        assert!(!report.passed());
        assert!(report.findings[0].message.contains("2 Inf"));
    }

    #[test]
    fn no_inf_passes() {
        let t = make_tensor("t", vec![3], vec![1.0, 2.0, 3.0]);
        let mut report = ValidationReport::new();
        TensorValidator::check_inf(&t, &mut report);
        assert!(report.passed());
    }

    // =====================================================================
    // TensorValidator — range
    // =====================================================================

    #[test]
    fn range_violation_detected() {
        let t = make_tensor("t", vec![5], vec![0.0, 1.0, 200.0, -200.0, 0.5]);
        let mut report = ValidationReport::new();
        TensorValidator::check_range(&t, -100.0, 100.0, &mut report);
        // Warning, not error
        assert!(report.passed());
        assert_eq!(report.count(ValidationSeverity::Warning), 1);
        assert!(report.findings[0].message.contains("2 value(s)"));
    }

    #[test]
    fn range_all_in_bounds_passes() {
        let t = make_tensor("t", vec![3], vec![0.0, 0.5, -0.5]);
        let mut report = ValidationReport::new();
        TensorValidator::check_range(&t, -1.0, 1.0, &mut report);
        assert_eq!(report.findings.len(), 0);
    }

    #[test]
    fn range_nan_values_not_counted() {
        let t = make_tensor("t", vec![2], vec![f32::NAN, 0.5]);
        let mut report = ValidationReport::new();
        TensorValidator::check_range(&t, -1.0, 1.0, &mut report);
        // NaN is filtered out, only 0.5 in range → no findings
        assert_eq!(report.findings.len(), 0);
    }

    // =====================================================================
    // TensorValidator — full data check
    // =====================================================================

    #[test]
    fn full_data_check_clean() {
        let t = make_tensor("t", vec![4], vec![0.1, 0.2, 0.3, 0.4]);
        let mut report = ValidationReport::new();
        TensorValidator::full_data_check(&t, (-1.0, 1.0), &mut report);
        assert!(report.passed());
        assert_eq!(report.findings.len(), 0);
    }

    #[test]
    fn full_data_check_nan_and_inf() {
        let t = make_tensor("t", vec![4], vec![f32::NAN, f32::INFINITY, 0.1, 0.2]);
        let mut report = ValidationReport::new();
        TensorValidator::full_data_check(&t, (-1.0, 1.0), &mut report);
        assert!(!report.passed());
        assert!(report.count(ValidationSeverity::Error) >= 2);
    }

    // =====================================================================
    // ArchitectureValidator — tensor names
    // =====================================================================

    #[test]
    fn valid_architecture_passes_name_check() {
        let spec = tiny_spec();
        let tensors = valid_tensors(&spec);
        let mut report = ValidationReport::new();
        ArchitectureValidator::check_tensor_names(&spec, &tensors, &mut report);
        assert_eq!(report.count(ValidationSeverity::Error), 0);
    }

    #[test]
    fn missing_tensor_detected() {
        let spec = tiny_spec();
        let mut tensors = valid_tensors(&spec);
        tensors.retain(|t| t.name != "token_embd.weight");
        let mut report = ValidationReport::new();
        ArchitectureValidator::check_tensor_names(&spec, &tensors, &mut report);
        assert!(
            report
                .errors()
                .iter()
                .any(|f| { f.tensor_name.as_deref() == Some("token_embd.weight") })
        );
    }

    #[test]
    fn extra_tensor_reported_as_info() {
        let spec = tiny_spec();
        let mut tensors = valid_tensors(&spec);
        tensors.push(TensorInfo {
            name: "bonus.weight".into(),
            shape: vec![10],
            dtype: TensorDtype::F32,
            data: None,
        });
        let mut report = ValidationReport::new();
        ArchitectureValidator::check_tensor_names(&spec, &tensors, &mut report);
        assert_eq!(report.count(ValidationSeverity::Error), 0);
        assert!(report.count(ValidationSeverity::Info) > 0);
    }

    // =====================================================================
    // ArchitectureValidator — tensor count
    // =====================================================================

    #[test]
    fn tensor_count_exact_match() {
        let spec = tiny_spec();
        let tensors = valid_tensors(&spec);
        let mut report = ValidationReport::new();
        ArchitectureValidator::check_tensor_count(&spec, &tensors, &mut report);
        assert_eq!(report.findings.len(), 0);
    }

    #[test]
    fn tensor_count_too_low() {
        let spec = tiny_spec();
        let mut report = ValidationReport::new();
        ArchitectureValidator::check_tensor_count(&spec, &[], &mut report);
        assert!(!report.passed());
    }

    #[test]
    fn tensor_count_too_high_info() {
        let spec = tiny_spec();
        let mut tensors = valid_tensors(&spec);
        tensors.push(TensorInfo {
            name: "extra".into(),
            shape: vec![1],
            dtype: TensorDtype::F32,
            data: None,
        });
        let mut report = ValidationReport::new();
        ArchitectureValidator::check_tensor_count(&spec, &tensors, &mut report);
        assert!(report.passed());
        assert_eq!(report.count(ValidationSeverity::Info), 1);
    }

    // =====================================================================
    // ArchitectureValidator — layer shapes
    // =====================================================================

    #[test]
    fn correct_shapes_pass() {
        let spec = tiny_spec();
        let tensors = valid_tensors(&spec);
        let mut report = ValidationReport::new();
        ArchitectureValidator::check_layer_shapes(&spec, &tensors, &mut report);
        assert_eq!(report.count(ValidationSeverity::Error), 0);
    }

    #[test]
    fn wrong_attn_shape_detected() {
        let spec = tiny_spec();
        let mut tensors = valid_tensors(&spec);
        // Corrupt blk.0.attn_q.weight shape
        if let Some(t) = tensors.iter_mut().find(|t| t.name == "blk.0.attn_q.weight") {
            t.shape = vec![32, 32]; // wrong
        }
        let mut report = ValidationReport::new();
        ArchitectureValidator::check_layer_shapes(&spec, &tensors, &mut report);
        assert!(report.count(ValidationSeverity::Error) > 0);
    }

    #[test]
    fn wrong_ffn_shape_detected() {
        let spec = tiny_spec();
        let mut tensors = valid_tensors(&spec);
        if let Some(t) = tensors.iter_mut().find(|t| t.name == "blk.0.ffn_gate.weight") {
            t.shape = vec![10, 10]; // wrong
        }
        let mut report = ValidationReport::new();
        ArchitectureValidator::check_layer_shapes(&spec, &tensors, &mut report);
        assert!(report.count(ValidationSeverity::Error) > 0);
    }

    #[test]
    fn wrong_norm_shape_detected() {
        let spec = tiny_spec();
        let mut tensors = valid_tensors(&spec);
        if let Some(t) = tensors.iter_mut().find(|t| t.name == "blk.0.attn_norm.weight") {
            t.shape = vec![10]; // wrong, should be hidden_dim
        }
        let mut report = ValidationReport::new();
        ArchitectureValidator::check_layer_shapes(&spec, &tensors, &mut report);
        assert!(report.count(ValidationSeverity::Error) > 0);
    }

    #[test]
    fn wrong_embedding_shape_detected() {
        let spec = tiny_spec();
        let mut tensors = valid_tensors(&spec);
        if let Some(t) = tensors.iter_mut().find(|t| t.name == "token_embd.weight") {
            t.shape = vec![100, 100]; // wrong
        }
        let mut report = ValidationReport::new();
        ArchitectureValidator::check_layer_shapes(&spec, &tensors, &mut report);
        assert!(report.count(ValidationSeverity::Error) > 0);
    }

    // =====================================================================
    // WeightHealthChecker — all zeros
    // =====================================================================

    #[test]
    fn all_zeros_detected() {
        let t = make_tensor("t", vec![4], vec![0.0, 0.0, 0.0, 0.0]);
        let mut report = ValidationReport::new();
        WeightHealthChecker::check_all_zeros(&t, &mut report);
        assert_eq!(report.count(ValidationSeverity::Warning), 1);
    }

    #[test]
    fn not_all_zeros_passes() {
        let t = make_tensor("t", vec![4], vec![0.0, 0.1, 0.0, 0.0]);
        let mut report = ValidationReport::new();
        WeightHealthChecker::check_all_zeros(&t, &mut report);
        assert_eq!(report.findings.len(), 0);
    }

    #[test]
    fn zeros_empty_data_passes() {
        let t = make_tensor("t", vec![0], vec![]);
        let mut report = ValidationReport::new();
        WeightHealthChecker::check_all_zeros(&t, &mut report);
        assert_eq!(report.findings.len(), 0);
    }

    // =====================================================================
    // WeightHealthChecker — extreme values
    // =====================================================================

    #[test]
    fn extreme_values_detected() {
        let t = make_tensor("t", vec![4], vec![0.1, 0.2, 500.0, -500.0]);
        let mut report = ValidationReport::new();
        WeightHealthChecker::check_extreme_values(&t, 100.0, &mut report);
        assert!(report.findings.len() > 0);
        assert!(report.findings[0].message.contains("extreme"));
    }

    #[test]
    fn no_extreme_values_passes() {
        let t = make_tensor("t", vec![4], vec![0.1, 0.2, -0.3, 0.4]);
        let mut report = ValidationReport::new();
        WeightHealthChecker::check_extreme_values(&t, 100.0, &mut report);
        assert_eq!(report.findings.len(), 0);
    }

    // =====================================================================
    // WeightHealthChecker — low variance
    // =====================================================================

    #[test]
    fn low_variance_detected() {
        let t = make_tensor("t", vec![4], vec![1.0, 1.0, 1.0, 1.0]);
        let mut report = ValidationReport::new();
        WeightHealthChecker::check_low_variance(&t, 1e-8, &mut report);
        assert_eq!(report.count(ValidationSeverity::Warning), 1);
    }

    #[test]
    fn normal_variance_passes() {
        let t = make_tensor("t", vec![4], vec![0.0, 1.0, 2.0, 3.0]);
        let mut report = ValidationReport::new();
        WeightHealthChecker::check_low_variance(&t, 1e-8, &mut report);
        assert_eq!(report.findings.len(), 0);
    }

    #[test]
    fn variance_single_element_skipped() {
        let t = make_tensor("t", vec![1], vec![5.0]);
        let mut report = ValidationReport::new();
        WeightHealthChecker::check_low_variance(&t, 1e-8, &mut report);
        assert_eq!(report.findings.len(), 0);
    }

    // =====================================================================
    // WeightHealthChecker — full check
    // =====================================================================

    #[test]
    fn full_health_check_clean() {
        let t = make_tensor("t", vec![4], vec![0.1, 0.2, -0.3, 0.4]);
        let mut report = ValidationReport::new();
        WeightHealthChecker::full_health_check(&t, 100.0, 1e-10, &mut report);
        assert_eq!(report.findings.len(), 0);
    }

    #[test]
    fn full_health_check_multiple_issues() {
        // All zeros → all-zero warning + low-variance warning
        let t = make_tensor("t", vec![4], vec![0.0, 0.0, 0.0, 0.0]);
        let mut report = ValidationReport::new();
        WeightHealthChecker::full_health_check(&t, 100.0, 1e-10, &mut report);
        assert!(report.count(ValidationSeverity::Warning) >= 2);
    }

    // =====================================================================
    // CompatibilityChecker — memory
    // =====================================================================

    #[test]
    fn memory_fits_passes() {
        let spec = tiny_spec();
        let device = DeviceCapabilities::a770_default();
        let mut report = ValidationReport::new();
        CompatibilityChecker::check_memory_fit(&spec, &device, &mut report);
        assert_eq!(report.count(ValidationSeverity::Error), 0);
    }

    #[test]
    fn memory_oom_detected() {
        let spec = ModelSpec::bitnet_2b();
        let device = DeviceCapabilities::tiny_device();
        let mut report = ValidationReport::new();
        CompatibilityChecker::check_memory_fit(&spec, &device, &mut report);
        assert!(!report.passed());
    }

    #[test]
    fn memory_tight_warns() {
        let spec = tiny_spec();
        let model_bytes = spec.estimated_memory_bytes();
        // Device available = model * 1.05 → > 90% utilization
        let available = (model_bytes as f64 * 1.05) as u64;
        let device = DeviceCapabilities {
            total_memory_bytes: available * 2,
            available_memory_bytes: available,
            supported_dtypes: {
                let mut s = HashSet::new();
                s.insert(TensorDtype::I2);
                s
            },
            max_allocation_bytes: available,
            device_name: "tight".into(),
        };
        let mut report = ValidationReport::new();
        CompatibilityChecker::check_memory_fit(&spec, &device, &mut report);
        assert!(report.count(ValidationSeverity::Warning) > 0);
    }

    // =====================================================================
    // CompatibilityChecker — dtype support
    // =====================================================================

    #[test]
    fn dtype_supported_passes() {
        let spec = tiny_spec();
        let device = DeviceCapabilities::a770_default();
        let mut report = ValidationReport::new();
        CompatibilityChecker::check_dtype_support(&spec, &device, &mut report);
        assert!(report.passed());
    }

    #[test]
    fn dtype_unsupported_detected() {
        let spec = tiny_spec(); // weight_dtype = I2
        let device = DeviceCapabilities {
            total_memory_bytes: u64::MAX,
            available_memory_bytes: u64::MAX,
            supported_dtypes: {
                let mut s = HashSet::new();
                s.insert(TensorDtype::F32); // no I2
                s
            },
            max_allocation_bytes: u64::MAX,
            device_name: "limited".into(),
        };
        let mut report = ValidationReport::new();
        CompatibilityChecker::check_dtype_support(&spec, &device, &mut report);
        assert!(!report.passed());
    }

    // =====================================================================
    // CompatibilityChecker — allocation limits
    // =====================================================================

    #[test]
    fn allocation_within_limit_passes() {
        let spec = tiny_spec();
        let tensors = valid_tensors(&spec);
        let device = DeviceCapabilities::a770_default();
        let mut report = ValidationReport::new();
        CompatibilityChecker::check_allocation_limits(&tensors, &device, &mut report);
        assert!(report.passed());
    }

    #[test]
    fn allocation_exceeds_limit_detected() {
        let tensors = vec![TensorInfo {
            name: "huge".into(),
            shape: vec![100_000, 100_000],
            dtype: TensorDtype::F32,
            data: None,
        }];
        let device = DeviceCapabilities::tiny_device();
        let mut report = ValidationReport::new();
        CompatibilityChecker::check_allocation_limits(&tensors, &device, &mut report);
        assert!(!report.passed());
    }

    // =====================================================================
    // QuickCheck
    // =====================================================================

    #[test]
    fn quick_check_valid_model_passes() {
        let spec = tiny_spec();
        let tensors = valid_tensors(&spec);
        let report = QuickCheck::run(&spec, &tensors);
        assert!(report.passed());
    }

    #[test]
    fn quick_check_missing_tensor_fails() {
        let spec = tiny_spec();
        let report = QuickCheck::run(&spec, &[]);
        assert!(!report.passed());
    }

    #[test]
    fn quick_check_wrong_norm_dtype() {
        let spec = tiny_spec();
        let mut tensors = valid_tensors(&spec);
        // Change norm dtype to I8 (should be f32)
        if let Some(t) = tensors.iter_mut().find(|t| t.name.contains("_norm.")) {
            t.dtype = TensorDtype::I8;
        }
        let report = QuickCheck::run(&spec, &tensors);
        assert!(!report.passed());
    }

    #[test]
    fn quick_check_does_not_scan_data() {
        let spec = tiny_spec();
        let mut tensors = valid_tensors(&spec);
        // Inject NaN data — quick check should NOT catch it
        if let Some(t) = tensors.iter_mut().find(|t| t.name == "blk.0.attn_q.weight") {
            if let Some(d) = &mut t.data {
                d[0] = f32::NAN;
            }
        }
        let report = QuickCheck::run(&spec, &tensors);
        // Quick check only checks metadata, NaN in data should not cause failure
        assert!(report.passed());
    }

    // =====================================================================
    // FullCheck
    // =====================================================================

    #[test]
    fn full_check_valid_model_passes() {
        let spec = tiny_spec();
        let tensors = valid_tensors(&spec);
        let device = DeviceCapabilities::a770_default();
        let report = FullCheck::run(&spec, &tensors, Some(&device));
        assert!(report.passed(), "report: {report}");
    }

    #[test]
    fn full_check_detects_nan() {
        let spec = tiny_spec();
        let mut tensors = valid_tensors(&spec);
        if let Some(t) = tensors.iter_mut().find(|t| t.name == "blk.0.attn_q.weight") {
            if let Some(d) = &mut t.data {
                d[0] = f32::NAN;
            }
        }
        let report = FullCheck::run(&spec, &tensors, None);
        assert!(!report.passed());
    }

    #[test]
    fn full_check_detects_inf() {
        let spec = tiny_spec();
        let mut tensors = valid_tensors(&spec);
        if let Some(t) = tensors.iter_mut().find(|t| t.name == "blk.0.attn_v.weight") {
            if let Some(d) = &mut t.data {
                d[0] = f32::INFINITY;
            }
        }
        let report = FullCheck::run(&spec, &tensors, None);
        assert!(!report.passed());
    }

    #[test]
    fn full_check_no_device_still_works() {
        let spec = tiny_spec();
        let tensors = valid_tensors(&spec);
        let report = FullCheck::run(&spec, &tensors, None);
        assert!(report.passed());
    }

    #[test]
    fn full_check_oom_device() {
        let spec = ModelSpec::bitnet_2b();
        let tensors = valid_tensors(&tiny_spec()); // mismatched but we care about compat
        let device = DeviceCapabilities::tiny_device();
        let report = FullCheck::run(&spec, &tensors, Some(&device));
        assert!(!report.passed());
    }

    // =====================================================================
    // Edge cases: empty model
    // =====================================================================

    #[test]
    fn empty_tensor_set_fails() {
        let spec = tiny_spec();
        let report = QuickCheck::run(&spec, &[]);
        assert!(!report.passed());
    }

    #[test]
    fn empty_tensor_set_full_check_fails() {
        let spec = tiny_spec();
        let report = FullCheck::run(&spec, &[], None);
        assert!(!report.passed());
    }

    // =====================================================================
    // Edge cases: single layer
    // =====================================================================

    #[test]
    fn single_layer_valid() {
        let spec = single_layer_spec();
        let tensors = valid_tensors(&spec);
        let report = QuickCheck::run(&spec, &tensors);
        assert!(report.passed(), "report: {report}");
    }

    #[test]
    fn single_layer_full_check_valid() {
        let spec = single_layer_spec();
        let tensors = valid_tensors(&spec);
        let report = FullCheck::run(&spec, &tensors, None);
        assert!(report.passed(), "report: {report}");
    }

    // =====================================================================
    // Property-like tests: valid input always passes
    // =====================================================================

    #[test]
    fn valid_tiny_always_passes_quick() {
        let spec = tiny_spec();
        let tensors = valid_tensors(&spec);
        for _ in 0..5 {
            assert!(QuickCheck::run(&spec, &tensors).passed());
        }
    }

    #[test]
    fn valid_tiny_always_passes_full() {
        let spec = tiny_spec();
        let tensors = valid_tensors(&spec);
        let dev = DeviceCapabilities::a770_default();
        for _ in 0..5 {
            assert!(FullCheck::run(&spec, &tensors, Some(&dev)).passed());
        }
    }

    #[test]
    fn valid_single_layer_always_passes() {
        let spec = single_layer_spec();
        let tensors = valid_tensors(&spec);
        for _ in 0..5 {
            assert!(QuickCheck::run(&spec, &tensors).passed());
        }
    }

    #[test]
    fn valid_2b_always_passes_quick() {
        let spec = ModelSpec::bitnet_2b();
        // Use header-only tensors (no data) for speed
        let tensors: Vec<TensorInfo> = build_valid_tensors(&spec)
            .into_iter()
            .map(|mut t| {
                t.data = None;
                t
            })
            .collect();
        assert!(QuickCheck::run(&spec, &tensors).passed());
    }

    // =====================================================================
    // DeviceCapabilities
    // =====================================================================

    #[test]
    fn a770_default_has_all_dtypes() {
        let dev = DeviceCapabilities::a770_default();
        assert!(dev.supported_dtypes.contains(&TensorDtype::F32));
        assert!(dev.supported_dtypes.contains(&TensorDtype::F16));
        assert!(dev.supported_dtypes.contains(&TensorDtype::I2));
    }

    #[test]
    fn tiny_device_limited_dtypes() {
        let dev = DeviceCapabilities::tiny_device();
        assert!(!dev.supported_dtypes.contains(&TensorDtype::F16));
    }

    // =====================================================================
    // ValidationFinding display
    // =====================================================================

    #[test]
    fn finding_display_with_tensor() {
        let f = ValidationFinding {
            severity: ValidationSeverity::Error,
            tensor_name: Some("blk.0.attn_q.weight".into()),
            message: "shape mismatch".into(),
            recommendation: Some("fix it".into()),
        };
        let s = format!("{f}");
        assert!(s.contains("ERROR"));
        assert!(s.contains("blk.0.attn_q.weight"));
        assert!(s.contains("shape mismatch"));
        assert!(s.contains("fix it"));
    }

    #[test]
    fn finding_display_without_tensor() {
        let f = ValidationFinding {
            severity: ValidationSeverity::Info,
            tensor_name: None,
            message: "note".into(),
            recommendation: None,
        };
        let s = format!("{f}");
        assert!(s.contains("model"));
        assert!(s.contains("note"));
    }

    // =====================================================================
    // build_valid_tensors helper
    // =====================================================================

    #[test]
    fn build_valid_tensors_matches_spec() {
        let spec = tiny_spec();
        let tensors = build_valid_tensors(&spec);
        let expected = spec.expected_tensor_names();
        assert_eq!(tensors.len(), expected.len());
        for name in &expected {
            assert!(tensors.iter().any(|t| &t.name == name), "missing tensor: {name}");
        }
    }

    #[test]
    fn build_valid_tensors_all_have_data() {
        let spec = tiny_spec();
        let tensors = build_valid_tensors(&spec);
        for t in &tensors {
            assert!(t.data.is_some(), "tensor {} has no data", t.name);
        }
    }

    // =====================================================================
    // Misc edge cases
    // =====================================================================

    #[test]
    fn nan_in_norm_detected_by_full_check() {
        let spec = tiny_spec();
        let mut tensors = valid_tensors(&spec);
        if let Some(t) = tensors.iter_mut().find(|t| t.name == "blk.0.attn_norm.weight") {
            t.data = Some(vec![f32::NAN; spec.hidden_dim as usize]);
        }
        let report = FullCheck::run(&spec, &tensors, None);
        assert!(!report.passed());
    }

    #[test]
    fn extreme_in_output_head_detected() {
        let spec = tiny_spec();
        let mut tensors = valid_tensors(&spec);
        if let Some(t) = tensors.iter_mut().find(|t| t.name == "output.weight") {
            if let Some(d) = &mut t.data {
                d[0] = 999.0;
            }
        }
        let report = FullCheck::run(&spec, &tensors, None);
        // 999.0 is within default range (-1000, 1000) but > extreme threshold (100)
        // Should produce at least an info-level finding
        assert!(report.findings.iter().any(|f| {
            f.tensor_name.as_deref() == Some("output.weight") && f.message.contains("extreme")
        }));
    }

    #[test]
    fn report_warnings_accessor() {
        let mut report = ValidationReport::new();
        report.add(ValidationFinding {
            severity: ValidationSeverity::Warning,
            tensor_name: None,
            message: "w".into(),
            recommendation: None,
        });
        report.add(ValidationFinding {
            severity: ValidationSeverity::Error,
            tensor_name: None,
            message: "e".into(),
            recommendation: None,
        });
        assert_eq!(report.warnings().len(), 1);
        assert_eq!(report.errors().len(), 1);
    }
}
