//! # Inference Receipt Generation (AC4)
//!
//! Generates receipt artifacts documenting real inference execution.
//! Implements schema version 1.0.0 as specified in issue-254-real-inference-spec.md.
//!
//! # Schema Requirements (AC4)
//! - `compute_path`: Must be "real" (not "mock")
//! - `backend`: "cpu" | "cuda" | "metal"
//! - `kernels`: List of executed kernels (e.g., ["i2s_gemv", "rope_apply"])
//! - `deterministic`: Boolean indicating BITNET_DETERMINISTIC=1
//! - `environment`: Environment variables used
//! - `model_info`: Model configuration details
//! - `test_results`: Test execution summary
//! - `performance_baseline`: Performance metrics

use anyhow::{Result, anyhow};
use bitnet_common::CorrectionRecord;
use bitnet_honest_compute::{
    classify_compute_path, validate_compute_path as validate_honest_compute_path,
    validate_kernel_ids as validate_honest_kernel_ids,
};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Schema version for receipt format
pub const RECEIPT_SCHEMA_VERSION: &str = "1.0.0";

/// Alias for schema version (for consistency)
pub const RECEIPT_SCHEMA: &str = RECEIPT_SCHEMA_VERSION;

/// Model information in receipt
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantization_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layers: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hidden_size: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_attention_heads: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_key_value_heads: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vocab_size: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effective_correction_digest: Option<String>,
}

/// Test execution results
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TestResults {
    pub total_tests: usize,
    pub passed: usize,
    pub failed: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skipped: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accuracy_tests: Option<AccuracyTestResults>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub determinism_tests: Option<DeterminismTestResults>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_cache_tests: Option<KVCacheTestResults>,
}

/// Accuracy test results (AC5)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyTestResults {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub i2s_accuracy: Option<AccuracyMetric>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tl1_accuracy: Option<AccuracyMetric>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tl2_accuracy: Option<AccuracyMetric>,
}

/// Individual accuracy metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyMetric {
    pub mse: f64,
    pub tolerance: f64,
    pub passed: bool,
}

/// Determinism test results (AC3, AC6)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeterminismTestResults {
    pub identical_sequences: bool,
    pub runs: usize,
    pub tokens_per_run: usize,
}

/// KV-cache test results (AC7)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KVCacheTestResults {
    pub prefill_decode_parity: bool,
    pub cache_hit_rate: f64,
}

/// Performance baseline metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PerformanceBaseline {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens_generated: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_time_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens_per_second: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_token_latency_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub average_token_latency_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_usage_mb: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_efficiency: Option<CacheEfficiency>,
}

/// Cache efficiency metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEfficiency {
    pub kv_cache_hit_rate: f64,
    pub tensor_cache_hits: usize,
    pub tensor_cache_misses: usize,
}

/// Cross-validation metrics (deprecated - use ParityMetadata instead)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CrossValidation {
    pub cpp_reference_available: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tolerance: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parity_tests_passed: Option<bool>,
}

/// Parity validation metadata (AC4)
///
/// Captures C++ reference comparison metrics for reproducibility and CI validation.
///
/// # Schema Version: 1.0.0
///
/// Status values:
/// - "ok": Rust and C++ outputs match (cosine ≥ 0.99, exact_match_rate = 1.0)
/// - "rust_only": C++ reference not available
/// - "divergence": Outputs differ (cosine < 0.99 or exact_match_rate < 1.0)
/// - "timeout": Parity test exceeded timeout
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParityMetadata {
    /// C++ reference available for comparison
    pub cpp_available: bool,

    /// Cosine similarity between Rust and C++ logits (0.0 to 1.0)
    /// Present only when cpp_available=true
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cosine_similarity: Option<f32>,

    /// Exact match rate for generated tokens (0.0 to 1.0)
    /// Present only when cpp_available=true
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exact_match_rate: Option<f32>,

    /// Parity status: "ok" | "rust_only" | "divergence" | "timeout"
    pub status: String,
}

/// Main inference receipt structure (AC4)
///
/// # Schema Version: 1.0.0
///
/// Provides comprehensive documentation of inference execution including:
/// - Compute path verification (real vs mock)
/// - Backend selection (CPU/GPU)
/// - Kernel execution tracking
/// - Determinism validation
/// - Performance baselines
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceReceipt {
    /// Schema version (always "1.0.0")
    pub schema_version: String,

    /// ISO 8601 timestamp of receipt generation
    pub timestamp: String,

    /// Compute path: "real" (required) or "mock" (fails validation)
    pub compute_path: String,

    /// Backend used: "cpu" | "cuda" | "metal"
    pub backend: String,

    /// Backend selection summary: "requested=X detected=[Y] selected=Z"
    /// Populated from BackendSelectionResult::summary() at receipt generation time.
    #[serde(default)]
    pub backend_summary: String,

    /// Kernels executed during inference
    /// Examples: ["i2s_gemv", "rope_apply", "attention_real"]
    pub kernels: Vec<String>,

    /// Deterministic mode enabled (BITNET_DETERMINISTIC=1)
    pub deterministic: bool,

    /// Environment variables
    pub environment: HashMap<String, String>,

    /// Model configuration
    pub model_info: ModelInfo,

    /// Test execution results
    pub test_results: TestResults,

    /// Performance metrics baseline
    pub performance_baseline: PerformanceBaseline,

    /// Cross-validation results (optional, deprecated - use parity instead)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cross_validation: Option<CrossValidation>,

    /// Parity validation results (AC4)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parity: Option<ParityMetadata>,

    /// Model corrections applied (LayerNorm rescaling, etc.)
    /// Empty if no corrections applied
    pub corrections: Vec<CorrectionRecord>,
}

impl InferenceReceipt {
    /// Generate receipt from inference execution
    ///
    /// # AC4 Contract
    /// - Sets `compute_path="real"` if no mock kernels detected
    /// - Sets `compute_path="mock"` if any mock kernels detected
    /// - Collects environment variables (BITNET_*, RAYON_*)
    /// - Records kernel execution list
    ///
    /// # Example
    /// ```no_run
    /// use bitnet_receipts::InferenceReceipt;
    ///
    /// let receipt = InferenceReceipt::generate(
    ///     "cpu",
    ///     vec!["i2s_gemv".to_string(), "rope_apply".to_string()],
    ///     None,
    /// ).unwrap();
    ///
    /// assert_eq!(receipt.compute_path, "real");
    /// ```
    pub fn generate(
        backend: &str,
        kernels: Vec<String>,
        backend_summary: Option<String>,
    ) -> Result<Self> {
        // AC4: Detect mock kernels (case-insensitive)
        let compute_path = classify_compute_path(kernels.iter().map(String::as_str));

        Ok(Self {
            schema_version: RECEIPT_SCHEMA_VERSION.to_string(),
            timestamp: Utc::now().to_rfc3339(),
            compute_path: compute_path.to_string(),
            backend: backend.to_string(),
            backend_summary: backend_summary.unwrap_or_default(),
            kernels,
            deterministic: std::env::var("BITNET_DETERMINISTIC").is_ok(),
            environment: Self::collect_env_vars(),
            model_info: ModelInfo::default(),
            test_results: TestResults::default(),
            performance_baseline: PerformanceBaseline::default(),
            cross_validation: None,
            parity: None,
            corrections: Vec::new(),
        })
    }

    /// Backward-compatible alias for [`generate`] with no backend summary.
    ///
    /// Equivalent to `generate(backend, kernels, None)`. Prefer `generate()` for new code.
    #[deprecated(since = "0.1.1", note = "use generate(backend, kernels, None) instead")]
    pub fn generate_basic(backend: &str, kernels: Vec<String>) -> Result<Self> {
        Self::generate(backend, kernels, None)
    }

    /// Collect relevant environment variables
    fn collect_env_vars() -> HashMap<String, String> {
        let mut env_vars = HashMap::new();

        // Determinism variables
        if let Ok(val) = std::env::var("BITNET_DETERMINISTIC") {
            env_vars.insert("BITNET_DETERMINISTIC".to_string(), val);
        }
        if let Ok(val) = std::env::var("BITNET_SEED") {
            env_vars.insert("BITNET_SEED".to_string(), val);
        }
        if let Ok(val) = std::env::var("RAYON_NUM_THREADS") {
            env_vars.insert("RAYON_NUM_THREADS".to_string(), val);
        }

        // Model path
        if let Ok(val) = std::env::var("BITNET_GGUF") {
            env_vars.insert("BITNET_GGUF".to_string(), val);
        }

        // System info
        env_vars.insert("RUST_VERSION".to_string(), rustc_version_runtime::version().to_string());
        env_vars.insert("BITNET_VERSION".to_string(), env!("CARGO_PKG_VERSION").to_string());
        env_vars.insert(
            "OS".to_string(),
            format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
        );

        // Add CPU and GPU fingerprints (best-effort)
        env_vars.insert("CPU_BRAND".to_string(), detect_cpu_brand());
        if let Some(gpu_info) = detect_gpu_info() {
            env_vars.insert("GPU_INFO".to_string(), gpu_info);
        }

        env_vars
    }

    /// Load receipt from JSON file
    ///
    /// # Example
    /// ```no_run
    /// use std::path::Path;
    /// use bitnet_receipts::InferenceReceipt;
    ///
    /// let receipt = InferenceReceipt::load(Path::new("ci/inference.json")).unwrap();
    /// assert_eq!(receipt.schema_version, "1.0.0");
    /// ```
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let receipt: InferenceReceipt = serde_json::from_str(&content)?;
        Ok(receipt)
    }

    /// Save receipt to JSON file
    ///
    /// # AC4 Contract
    /// - Serializes to pretty JSON
    /// - Creates parent directory if it doesn't exist
    /// - Writes atomically (temp file + rename)
    /// - Typically saved to `ci/inference.json`
    ///
    /// # Example
    /// ```no_run
    /// use std::path::Path;
    /// use bitnet_receipts::InferenceReceipt;
    ///
    /// let receipt = InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
    /// receipt.save(Path::new("ci/inference.json")).unwrap();
    /// ```
    pub fn save(&self, path: &Path) -> Result<()> {
        // Create parent directory if it doesn't exist
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Serialize to pretty JSON
        let json = serde_json::to_string_pretty(self)?;

        // Atomic write: write to temp file, then rename
        let temp_path = path.with_extension("tmp");
        std::fs::write(&temp_path, json)?;
        std::fs::rename(&temp_path, path)?;

        Ok(())
    }

    /// Validate receipt against AC9 requirements
    ///
    /// # AC9 Contract
    /// - MUST have `compute_path="real"` (fail if "mock")
    /// - MUST NOT have mock kernels (case-insensitive check)
    /// - MUST have zero failed tests
    /// - MUST pass accuracy tests (if present)
    /// - MUST pass determinism tests (if deterministic mode enabled)
    /// - MUST have valid kernel IDs (hygiene checks)
    ///
    /// # Example
    /// ```no_run
    /// use bitnet_receipts::InferenceReceipt;
    ///
    /// let receipt = InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
    /// assert!(receipt.validate().is_ok());
    /// ```
    pub fn validate(&self) -> Result<()> {
        // Validate schema version
        self.validate_schema()?;

        // AC9: Check compute path
        self.validate_compute_path()?;

        // AC9: Check for mock kernels and validate kernel ID hygiene
        self.validate_kernel_ids()?;

        // AC9: Check test results
        if self.test_results.failed > 0 {
            return Err(anyhow!("Failed tests detected: {}", self.test_results.failed));
        }

        // AC9: Validate accuracy tests (if present)
        if let Some(ref accuracy) = self.test_results.accuracy_tests {
            if let Some(ref i2s) = accuracy.i2s_accuracy
                && !i2s.passed
            {
                return Err(anyhow!(
                    "I2S accuracy test failed: MSE {} > tolerance {}",
                    i2s.mse,
                    i2s.tolerance
                ));
            }
            if let Some(ref tl1) = accuracy.tl1_accuracy
                && !tl1.passed
            {
                return Err(anyhow!(
                    "TL1 accuracy test failed: MSE {} > tolerance {}",
                    tl1.mse,
                    tl1.tolerance
                ));
            }
            if let Some(ref tl2) = accuracy.tl2_accuracy
                && !tl2.passed
            {
                return Err(anyhow!(
                    "TL2 accuracy test failed: MSE {} > tolerance {}",
                    tl2.mse,
                    tl2.tolerance
                ));
            }
        }

        // AC9: Validate determinism tests (if deterministic mode)
        if self.deterministic
            && let Some(ref det_tests) = self.test_results.determinism_tests
            && !det_tests.identical_sequences
        {
            return Err(anyhow!("Determinism test failed: sequences not identical"));
        }

        // Soft gate: if backend_summary is non-empty, verify it has the expected format.
        if !self.backend_summary.is_empty() && !self.backend_summary.contains("selected=") {
            return Err(anyhow!(
                "backend_summary format invalid: expected to contain \"selected=\", got: {:?}",
                self.backend_summary
            ));
        }

        Ok(())
    }

    /// Validate schema version
    ///
    /// # Requirements
    /// - Schema version must be "1.0.0"
    ///
    /// # Example
    /// ```no_run
    /// use bitnet_receipts::InferenceReceipt;
    ///
    /// let receipt = InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
    /// assert!(receipt.validate_schema().is_ok());
    /// ```
    pub fn validate_schema(&self) -> Result<()> {
        if self.schema_version != "1.0.0" {
            return Err(anyhow!(
                "Invalid schema version: {} (expected '1.0.0')",
                self.schema_version
            ));
        }
        Ok(())
    }

    /// Validate compute path
    ///
    /// # Requirements
    /// - Compute path must be "real" (not "mock")
    ///
    /// # Example
    /// ```no_run
    /// use bitnet_receipts::InferenceReceipt;
    ///
    /// let receipt = InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
    /// assert!(receipt.validate_compute_path().is_ok());
    /// ```
    pub fn validate_compute_path(&self) -> Result<()> {
        validate_honest_compute_path(&self.compute_path).map_err(Into::into)
    }

    /// Validate kernel IDs
    ///
    /// # Requirements
    /// - Kernel array must be non-empty
    /// - No kernel ID can be empty string
    /// - No kernel ID can be whitespace-only
    /// - Each kernel ID must be ≤ 128 characters
    /// - Total kernel count must be ≤ 10,000
    /// - No kernel ID can contain "mock" (case-insensitive)
    ///
    /// # Example
    /// ```no_run
    /// use bitnet_receipts::InferenceReceipt;
    ///
    /// let receipt = InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
    /// assert!(receipt.validate_kernel_ids().is_ok());
    /// ```
    pub fn validate_kernel_ids(&self) -> Result<()> {
        validate_honest_kernel_ids(self.kernels.iter().map(String::as_str)).map_err(Into::into)
    }

    /// Builder for test results
    pub fn with_test_results(mut self, test_results: TestResults) -> Self {
        self.test_results = test_results;
        self
    }

    /// Builder for model info
    pub fn with_model_info(mut self, model_info: ModelInfo) -> Self {
        self.model_info = model_info;
        self
    }

    /// Builder for performance baseline
    pub fn with_performance_baseline(mut self, performance: PerformanceBaseline) -> Self {
        self.performance_baseline = performance;
        self
    }

    /// Builder for cross-validation
    pub fn with_cross_validation(mut self, cross_val: CrossValidation) -> Self {
        self.cross_validation = Some(cross_val);
        self
    }

    /// Builder for parity metadata (AC4)
    pub fn with_parity(mut self, parity: ParityMetadata) -> Self {
        self.parity = Some(parity);
        self
    }

    /// Builder for corrections
    pub fn with_corrections(mut self, corrections: Vec<CorrectionRecord>) -> Self {
        self.corrections = corrections;
        self
    }

    /// Add a single correction record
    pub fn add_correction(&mut self, correction: CorrectionRecord) {
        self.corrections.push(correction);
    }
}

/// Detect CPU brand string (best-effort).
/// Linux: reads `/proc/cpuinfo` model name; otherwise returns arch.
fn detect_cpu_brand() -> String {
    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = std::fs::read_to_string("/proc/cpuinfo") {
            for line in content.lines() {
                if line.starts_with("model name")
                    && let Some(brand) = line.split(':').nth(1)
                {
                    return brand.trim().to_string();
                }
            }
        }
    }
    std::env::consts::ARCH.to_string()
}

/// Detect GPU information (best-effort)
///
/// Uses bitnet-kernels GPU utilities to detect available GPUs.
/// Returns GPU name and compute capability if available.
fn detect_gpu_info() -> Option<String> {
    #[cfg(any(feature = "gpu", feature = "cuda"))]
    {
        use bitnet_kernels::gpu;
        // Try to get first CUDA device info if available
        if let Ok(devices) = gpu::list_cuda_devices()
            && let Some(device) = devices.first()
        {
            return Some(format!(
                "{} (CC: {}.{})",
                device.name, device.compute_capability.0, device.compute_capability.1
            ));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_receipt_generation_real_path() {
        let receipt = InferenceReceipt::generate(
            "cpu",
            vec!["i2s_gemv".to_string(), "rope_apply".to_string()],
            None,
        )
        .unwrap();

        assert_eq!(receipt.schema_version, "1.0.0");
        assert_eq!(receipt.compute_path, "real");
        assert_eq!(receipt.backend, "cpu");
        assert!(receipt.kernels.contains(&"i2s_gemv".to_string()));
    }

    #[test]
    fn test_receipt_generation_mock_detected() {
        let receipt = InferenceReceipt::generate(
            "cpu",
            vec!["mock_gemv".to_string(), "i2s_gemv".to_string()],
            None,
        )
        .unwrap();

        assert_eq!(receipt.compute_path, "mock");
    }

    #[test]
    fn test_receipt_validation_passes() {
        let receipt =
            InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();

        assert!(receipt.validate().is_ok());
    }

    #[test]
    fn test_receipt_validation_fails_mock_path() {
        let mut receipt =
            InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();

        receipt.compute_path = "mock".to_string();
        assert!(receipt.validate().is_err());
    }

    #[test]
    fn test_receipt_validation_fails_mock_kernels() {
        let receipt =
            InferenceReceipt::generate("cpu", vec!["mock_gemv".to_string()], None).unwrap();

        assert!(receipt.validate().is_err());
    }

    #[test]
    fn test_receipt_validation_fails_failed_tests() {
        let mut receipt =
            InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();

        receipt.test_results.failed = 1;
        assert!(receipt.validate().is_err());
    }

    #[test]
    fn test_receipt_with_corrections() {
        let mut receipt =
            InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();

        // Add a correction record
        let correction = CorrectionRecord {
            layer: "model.layers.0.input_layernorm.weight".to_string(),
            correction_type: "ln_gamma_rescale_rms".to_string(),
            rms_before: Some(0.5),
            rms_after: Some(1.0),
            factor: Some(2.0),
            policy_fingerprint: "BITNET_FIX_LN_SCALE=1".to_string(),
            metadata: None,
        };
        receipt.add_correction(correction.clone());

        // Verify correction is present
        assert_eq!(receipt.corrections.len(), 1);
        assert_eq!(receipt.corrections[0].layer, "model.layers.0.input_layernorm.weight");
        assert_eq!(receipt.corrections[0].correction_type, "ln_gamma_rescale_rms");
        assert_eq!(receipt.corrections[0].rms_before, Some(0.5));
        assert_eq!(receipt.corrections[0].rms_after, Some(1.0));
        assert_eq!(receipt.corrections[0].factor, Some(2.0));
        assert_eq!(receipt.corrections[0].policy_fingerprint, "BITNET_FIX_LN_SCALE=1");
    }

    #[test]
    fn test_receipt_empty_corrections_by_default() {
        let receipt =
            InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
        assert!(receipt.corrections.is_empty(), "Corrections should be empty by default");
    }

    #[test]
    fn test_receipt_serialization_with_corrections() {
        let mut receipt =
            InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();

        let correction = CorrectionRecord {
            layer: "test.layer".to_string(),
            correction_type: "ln_gamma_rescale_rms".to_string(),
            rms_before: Some(0.75),
            rms_after: Some(1.0),
            factor: Some(1.33),
            policy_fingerprint: "BITNET_FIX_LN_SCALE=1".to_string(),
            metadata: None,
        };
        receipt.add_correction(correction);

        // Serialize to JSON
        let json = serde_json::to_string_pretty(&receipt).unwrap();

        // Verify JSON contains corrections
        assert!(json.contains("corrections"));
        assert!(json.contains("test.layer"));
        assert!(json.contains("ln_gamma_rescale_rms"));
        assert!(json.contains("BITNET_FIX_LN_SCALE=1"));

        // Deserialize and verify
        let deserialized: InferenceReceipt = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.corrections.len(), 1);
        assert_eq!(deserialized.corrections[0].layer, "test.layer");
    }

    #[test]
    fn test_receipt_with_model_metadata() {
        let mut receipt =
            InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();

        // Add model SHA256 and correction digest
        receipt.model_info.sha256 = Some("abc123def456".to_string());
        receipt.model_info.effective_correction_digest = Some("digest789".to_string());

        // Serialize and verify
        let json = serde_json::to_string_pretty(&receipt).unwrap();
        assert!(json.contains("sha256"));
        assert!(json.contains("abc123def456"));
        assert!(json.contains("effective_correction_digest"));
        assert!(json.contains("digest789"));
    }

    /// Test validate_schema with invalid version
    #[test]
    fn test_validate_schema_invalid_version() {
        let mut receipt =
            InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
        receipt.schema_version = "2.0.0".to_string();

        let result = receipt.validate_schema();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid schema version"));
    }

    /// Test validate_schema with valid version
    #[test]
    fn test_validate_schema_valid() {
        let receipt =
            InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
        assert!(receipt.validate_schema().is_ok());
    }

    /// Test validate_compute_path with invalid path
    #[test]
    fn test_validate_compute_path_invalid() {
        let mut receipt =
            InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
        receipt.compute_path = "mock".to_string();

        let result = receipt.validate_compute_path();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid compute_path"));
    }

    /// Test validate_compute_path with valid path
    #[test]
    fn test_validate_compute_path_valid() {
        let receipt =
            InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
        assert!(receipt.validate_compute_path().is_ok());
    }

    /// Test validate_kernel_ids with empty array
    #[test]
    fn test_validate_kernel_ids_empty_array() {
        let mut receipt =
            InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
        receipt.kernels = vec![];

        let result = receipt.validate_kernel_ids();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Kernel array is empty"));
    }

    /// Test validate_kernel_ids with empty string
    #[test]
    fn test_validate_kernel_ids_empty_string() {
        let mut receipt =
            InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
        receipt.kernels = vec!["".to_string()];

        let result = receipt.validate_kernel_ids();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Empty kernel ID"));
    }

    /// Test validate_kernel_ids with whitespace-only string
    #[test]
    fn test_validate_kernel_ids_whitespace_only() {
        let mut receipt =
            InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
        receipt.kernels = vec!["   ".to_string()];

        let result = receipt.validate_kernel_ids();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Whitespace-only kernel ID"));
    }

    /// Test validate_kernel_ids with excessive length
    #[test]
    fn test_validate_kernel_ids_excessive_length() {
        let mut receipt =
            InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
        receipt.kernels = vec!["a".repeat(129)];

        let result = receipt.validate_kernel_ids();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("exceeds 128 characters"));
    }

    /// Test validate_kernel_ids at exact 128 character boundary (should pass)
    #[test]
    fn test_validate_kernel_ids_exact_128_chars() {
        let mut receipt =
            InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
        receipt.kernels = vec!["a".repeat(128)];

        assert!(receipt.validate_kernel_ids().is_ok());
    }

    /// Test validate_kernel_ids with excessive count
    #[test]
    fn test_validate_kernel_ids_excessive_count() {
        let mut receipt =
            InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
        receipt.kernels = vec!["kernel".to_string(); 10_001];

        let result = receipt.validate_kernel_ids();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("exceeds 10,000 limit"));
    }

    /// Test validate_kernel_ids at exact 10,000 count boundary (should pass)
    #[test]
    fn test_validate_kernel_ids_exact_10k_count() {
        let mut receipt =
            InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
        receipt.kernels = vec!["kernel".to_string(); 10_000];

        assert!(receipt.validate_kernel_ids().is_ok());
    }

    /// Test validate_kernel_ids with mock kernel (case-insensitive)
    #[test]
    fn test_validate_kernel_ids_mock_kernel() {
        let test_cases = vec!["mock_kernel", "MOCK_kernel", "kernel_mock", "kernel_MOCK_suffix"];

        for kernel_id in test_cases {
            let mut receipt =
                InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
            receipt.kernels = vec![kernel_id.to_string()];

            let result = receipt.validate_kernel_ids();
            assert!(result.is_err(), "Kernel ID '{}' should be rejected as mock", kernel_id);
            assert!(result.unwrap_err().to_string().contains("Mock kernel detected"));
        }
    }

    /// Test validate_kernel_ids with mixed valid and invalid kernels
    #[test]
    fn test_validate_kernel_ids_mixed_kernels() {
        let mut receipt =
            InferenceReceipt::generate("cpu", vec!["i2s_gemv".to_string()], None).unwrap();
        receipt.kernels = vec!["valid_kernel".to_string(), "".to_string()];

        let result = receipt.validate_kernel_ids();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Empty kernel ID at index 1"));
    }

    /// Test validate_kernel_ids with valid realistic CPU kernels
    #[test]
    fn test_validate_kernel_ids_valid_cpu_kernels() {
        let receipt = InferenceReceipt::generate(
            "cpu",
            vec![
                "i2s_cpu_quantized_matmul".to_string(),
                "tl1_lut_dequant_forward".to_string(),
                "tl2_lut_backward".to_string(),
                "cpu_attention_qkvo".to_string(),
            ],
            None,
        )
        .unwrap();

        assert!(receipt.validate_kernel_ids().is_ok());
    }

    /// Test validate_kernel_ids with valid realistic GPU kernels
    #[test]
    fn test_validate_kernel_ids_valid_gpu_kernels() {
        let receipt = InferenceReceipt::generate(
            "cuda",
            vec![
                "gemm_gpu_fp16".to_string(),
                "cuda_i2s_quantize".to_string(),
                "gpu_attention_flash".to_string(),
            ],
            None,
        )
        .unwrap();

        assert!(receipt.validate_kernel_ids().is_ok());
    }

    /// Test that environment variable collection returns non-empty HashMap with valid content
    /// Kills 3 mutation survivors in receipts.rs:221 (empty HashMap, single empty entry, dummy values)
    #[test]
    fn test_receipt_env_vars_content_validation() {
        // Set test environment variables to ensure we have predictable content
        // SAFETY: This is test code running in isolation. We clean up at the end.
        unsafe {
            std::env::set_var("BITNET_DETERMINISTIC", "1");
            std::env::set_var("BITNET_SEED", "42");
        }

        let vars = InferenceReceipt::collect_env_vars();

        // Kill survivor 1: empty HashMap return
        assert!(!vars.is_empty(), "Environment variables should not be empty");

        // Kill survivor 2 & 3: single empty entry or dummy values
        for (key, value) in &vars {
            assert!(!key.is_empty(), "Environment variable key should not be empty");
            assert!(!value.is_empty(), "Environment variable value should not be empty");

            // Validate actual content - keys should be recognizable environment variables
            assert!(
                key.starts_with("BITNET_")
                    || key.starts_with("RAYON_")
                    || key == "RUST_VERSION"
                    || key == "OS"
                    || key == "CPU_BRAND"
                    || key == "GPU_INFO",
                "Key '{}' should be a valid BitNet/Rayon/Rust environment variable",
                key
            );
        }

        // Verify specific expected variables are present with correct values
        assert!(vars.contains_key("BITNET_DETERMINISTIC"), "Should contain BITNET_DETERMINISTIC");
        assert_eq!(
            vars.get("BITNET_DETERMINISTIC"),
            Some(&"1".to_string()),
            "BITNET_DETERMINISTIC should have value '1'"
        );

        assert!(vars.contains_key("BITNET_SEED"), "Should contain BITNET_SEED when set");
        assert_eq!(
            vars.get("BITNET_SEED"),
            Some(&"42".to_string()),
            "BITNET_SEED should have value '42'"
        );

        assert!(vars.contains_key("RUST_VERSION"), "Should always contain RUST_VERSION");
        let rust_version = vars.get("RUST_VERSION").unwrap();
        assert!(
            rust_version.contains('.'),
            "RUST_VERSION should be a valid version string with dots"
        );

        assert!(vars.contains_key("BITNET_VERSION"), "Should always contain BITNET_VERSION");
        assert!(vars.contains_key("OS"), "Should always contain OS");

        // Clean up test environment variables
        // SAFETY: This is test cleanup code running in isolation.
        unsafe {
            std::env::remove_var("BITNET_DETERMINISTIC");
            std::env::remove_var("BITNET_SEED");
        }
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    /// generate_basic always produces a receipt that passes schema validation.
    proptest! {
        #[test]
        fn generate_basic_passes_schema_validation(
            backend in prop_oneof![Just("cpu"), Just("cuda"), Just("gpu")],
            kernel_count in 0usize..=8,
        ) {
            let kernels: Vec<String> = (0..kernel_count)
                .map(|i| format!("kernel_{i}"))
                .collect();
            let receipt = InferenceReceipt::generate_basic(backend, kernels).unwrap();
            prop_assert!(
                receipt.validate_schema().is_ok(),
                "schema validation failed for backend={:?}",
                backend
            );
        }
    }

    /// generate_basic with compute_path "real" always passes validate_compute_path.
    proptest! {
        #[test]
        fn generate_basic_has_real_compute_path(
            backend in "[a-z]{1,8}",
        ) {
            let receipt = InferenceReceipt::generate_basic(&backend, vec!["k".to_string()])
                .unwrap();
            prop_assert_eq!(receipt.compute_path.as_str(), "real");
            prop_assert!(receipt.validate_compute_path().is_ok());
        }
    }

    /// validate_kernel_ids accepts any kernel IDs that are non-empty and ≤128 chars.
    proptest! {
        #[test]
        fn validate_kernel_ids_accepts_valid_ids(
            ids in prop::collection::vec("[a-z_]{1,32}", 1..=16),
        ) {
            let mut receipt =
                InferenceReceipt::generate_basic("cpu", ids.clone()).unwrap();
            receipt.kernels = ids;
            prop_assert!(
                receipt.validate_kernel_ids().is_ok(),
                "expected Ok for valid kernel IDs"
            );
        }
    }

    /// validate_kernel_ids rejects any slice that contains an empty string.
    proptest! {
        #[test]
        fn validate_kernel_ids_rejects_empty_id(
            prefix in prop::collection::vec("[a-z_]{1,16}", 0..=8),
            suffix in prop::collection::vec("[a-z_]{1,16}", 0..=8),
        ) {
            let mut kernels = prefix;
            kernels.push(String::new()); // inject empty id
            kernels.extend(suffix);
            let mut receipt =
                InferenceReceipt::generate_basic("cpu", vec!["ok".to_string()]).unwrap();
            receipt.kernels = kernels;
            prop_assert!(
                receipt.validate_kernel_ids().is_err(),
                "expected Err when empty kernel ID present"
            );
        }
    }
}
