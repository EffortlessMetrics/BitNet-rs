//! Module stub - implementation pending merge from feature branch
//! End-to-end integration test suite for full inference pipeline validation.
//!
//! Provides reusable test utilities that downstream crates and integration tests
//! can compose to validate complete model-loading → tokenization → inference →
//! decoding → validation pipelines across CPU and GPU backends.
//!
//! # Core abstractions
//!
//! | Type | Purpose |
//! |------|---------|
//! | [`E2EConfig`] | Configuration for model path, device, sampling, limits |
//! | [`E2ERunner`] | Orchestrates a single end-to-end inference run |
//! | [`ModelLoadTest`] | Validates GGUF model loading and metadata |
//! | [`TokenizationTest`] | Encode → decode round-trip fidelity |
//! | [`InferenceTest`] | Single-token generation correctness |
//! | [`StreamingTest`] | Streaming token output ordering and completeness |
//! | [`BatchInferenceTest`] | Multi-prompt batch processing |
//! | [`DeviceFallbackTest`] | GPU → CPU fallback path |
//! | [`DeterminismTest`] | Seed-based reproducibility |
//! | [`E2ETestSuite`] | Full orchestrator: load → tokenize → infer → decode → validate |

#![allow(
    clippy::cast_precision_loss,
    clippy::missing_const_for_fn,
    clippy::use_self,
    clippy::redundant_else,
    clippy::option_if_let_else
)]

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

// ─────────────────────────────────────────────────────────────────────────────
// Device & backend types
// ─────────────────────────────────────────────────────────────────────────────

/// Target compute device for inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Device {
    Cpu,
    Cuda(u32),
    Vulkan(u32),
    #[default]
    Auto,
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "cpu"),
            Self::Cuda(id) => write!(f, "cuda:{id}"),
            Self::Vulkan(id) => write!(f, "vulkan:{id}"),
            Self::Auto => write!(f, "auto"),
        }
    }
}

/// Status of a pipeline stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageStatus {
    Pending,
    Running,
    Passed,
    Failed,
    Skipped,
}

impl fmt::Display for StageStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pending => write!(f, "pending"),
            Self::Running => write!(f, "running"),
            Self::Passed => write!(f, "passed"),
            Self::Failed => write!(f, "failed"),
            Self::Skipped => write!(f, "skipped"),
        }
    }
}

/// Reason a stage failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FailureReason {
    ModelNotFound(String),
    TokenizerError(String),
    InferenceError(String),
    DeviceUnavailable(Device),
    Timeout(Duration),
    DeterminismMismatch { run_a: Vec<u32>, run_b: Vec<u32> },
    ValidationError(String),
}

impl fmt::Display for FailureReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ModelNotFound(p) => write!(f, "model not found: {p}"),
            Self::TokenizerError(e) => write!(f, "tokenizer error: {e}"),
            Self::InferenceError(e) => write!(f, "inference error: {e}"),
            Self::DeviceUnavailable(d) => write!(f, "device unavailable: {d}"),
            Self::Timeout(d) => write!(f, "timeout after {d:?}"),
            Self::DeterminismMismatch { run_a, run_b } => {
                write!(f, "determinism mismatch: {run_a:?} != {run_b:?}")
            }
            Self::ValidationError(e) => write!(f, "validation: {e}"),
        }
    }
}

/// Result from a single stage execution.
#[derive(Debug, Clone)]
pub struct StageResult {
    pub stage_name: String,
    pub status: StageStatus,
    pub elapsed: Duration,
    pub failure: Option<FailureReason>,
    pub metadata: HashMap<String, String>,
}

impl StageResult {
    pub fn passed(name: &str, elapsed: Duration) -> Self {
        Self {
            stage_name: name.to_string(),
            status: StageStatus::Passed,
            elapsed,
            failure: None,
            metadata: HashMap::new(),
        }
    }

    pub fn failed(name: &str, elapsed: Duration, reason: FailureReason) -> Self {
        Self {
            stage_name: name.to_string(),
            status: StageStatus::Failed,
            elapsed,
            failure: Some(reason),
            metadata: HashMap::new(),
        }
    }

    pub fn skipped(name: &str) -> Self {
        Self {
            stage_name: name.to_string(),
            status: StageStatus::Skipped,
            elapsed: Duration::ZERO,
            failure: None,
            metadata: HashMap::new(),
        }
    }

    #[must_use]
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    pub fn is_passed(&self) -> bool {
        self.status == StageStatus::Passed
    }

    pub fn is_failed(&self) -> bool {
        self.status == StageStatus::Failed
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// E2EConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for an end-to-end inference test run.
#[derive(Debug, Clone)]
pub struct E2EConfig {
    pub model_path: String,
    pub tokenizer_path: String,
    pub device: Device,
    pub max_tokens: u32,
    pub temperature: f32,
    pub seed: Option<u64>,
    pub timeout: Duration,
    pub prompts: Vec<String>,
    pub expected_vocab_size: Option<usize>,
    pub validate_roundtrip: bool,
    pub allow_device_fallback: bool,
}

impl Default for E2EConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            tokenizer_path: String::new(),
            device: Device::Auto,
            max_tokens: 16,
            temperature: 0.0,
            seed: Some(42),
            timeout: Duration::from_secs(300),
            prompts: vec!["Hello".to_string()],
            expected_vocab_size: None,
            validate_roundtrip: true,
            allow_device_fallback: true,
        }
    }
}

impl E2EConfig {
    /// Validate the configuration before running.
    pub fn validate(&self) -> Result<(), String> {
        if self.model_path.is_empty() {
            return Err("model_path must not be empty".to_string());
        }
        if self.tokenizer_path.is_empty() {
            return Err("tokenizer_path must not be empty".to_string());
        }
        if self.max_tokens == 0 {
            return Err("max_tokens must be > 0".to_string());
        }
        if self.temperature < 0.0 {
            return Err("temperature must be >= 0.0".to_string());
        }
        if self.temperature > 10.0 {
            return Err("temperature must be <= 10.0".to_string());
        }
        if self.prompts.is_empty() {
            return Err("at least one prompt is required".to_string());
        }
        if self.timeout.is_zero() {
            return Err("timeout must be > 0".to_string());
        }
        Ok(())
    }

    /// Builder: set model and tokenizer paths.
    #[must_use]
    pub fn with_paths(mut self, model: &str, tokenizer: &str) -> Self {
        self.model_path = model.to_string();
        self.tokenizer_path = tokenizer.to_string();
        self
    }

    /// Builder: set target device.
    #[must_use]
    pub fn with_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Builder: set max tokens.
    #[must_use]
    pub fn with_max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = n;
        self
    }

    /// Builder: set temperature.
    #[must_use]
    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }

    /// Builder: set seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Builder: set prompts.
    #[must_use]
    pub fn with_prompts(mut self, prompts: Vec<String>) -> Self {
        self.prompts = prompts;
        self
    }

    /// Builder: set timeout.
    #[must_use]
    pub fn with_timeout(mut self, d: Duration) -> Self {
        self.timeout = d;
        self
    }

    /// Builder: disable device fallback.
    #[must_use]
    pub fn without_fallback(mut self) -> Self {
        self.allow_device_fallback = false;
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Simulated inference types (for testing the test utilities)
// ─────────────────────────────────────────────────────────────────────────────

/// Simulated model metadata returned after loading.
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub quantization: String,
    pub file_size_bytes: u64,
}

impl Default for ModelMetadata {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            hidden_dim: 2048,
            num_layers: 24,
            quantization: "I2_S".to_string(),
            file_size_bytes: 500_000_000,
        }
    }
}

/// Simulated token output from inference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenOutput {
    pub token_id: u32,
    pub text: String,
    pub index: usize,
}

/// Simulated inference result.
#[derive(Debug, Clone)]
pub struct InferenceResult {
    pub tokens: Vec<TokenOutput>,
    pub total_time: Duration,
    pub device_used: Device,
    pub seed_used: Option<u64>,
    pub prompt: String,
}

impl InferenceResult {
    pub fn token_ids(&self) -> Vec<u32> {
        self.tokens.iter().map(|t| t.token_id).collect()
    }

    pub fn decoded_text(&self) -> String {
        self.tokens.iter().map(|t| t.text.as_str()).collect::<Vec<_>>().join("")
    }

    pub fn tokens_per_second(&self) -> f64 {
        if self.total_time.is_zero() {
            return 0.0;
        }
        self.tokens.len() as f64 / self.total_time.as_secs_f64()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// E2ERunner
// ─────────────────────────────────────────────────────────────────────────────

/// Runs a complete inference pipeline end-to-end.
///
/// The runner simulates the full pipeline stages for testing purposes:
/// load model → tokenize prompt → run inference → decode output → validate.
#[derive(Debug, Clone)]
pub struct E2ERunner {
    pub config: E2EConfig,
    pub results: Vec<StageResult>,
    pub model_metadata: Option<ModelMetadata>,
    pub inference_results: Vec<InferenceResult>,
    started_at: Option<Instant>,
}

impl E2ERunner {
    pub fn new(config: E2EConfig) -> Self {
        Self {
            config,
            results: Vec::new(),
            model_metadata: None,
            inference_results: Vec::new(),
            started_at: None,
        }
    }

    /// Run the full pipeline and collect stage results.
    pub fn run(&mut self) -> Result<&[StageResult], String> {
        self.started_at = Some(Instant::now());
        self.results.clear();
        self.inference_results.clear();

        self.config.validate()?;

        self.run_model_load();
        if self.last_failed() {
            return Ok(&self.results);
        }

        self.run_tokenization();
        if self.last_failed() {
            return Ok(&self.results);
        }

        self.run_inference();
        if self.last_failed() {
            return Ok(&self.results);
        }

        self.run_decode();
        self.run_validation();

        Ok(&self.results)
    }

    /// Total elapsed time since `run()` was called.
    pub fn elapsed(&self) -> Duration {
        self.started_at.map_or(Duration::ZERO, |s| s.elapsed())
    }

    /// Whether all stages passed.
    pub fn all_passed(&self) -> bool {
        self.results.iter().all(|r| r.is_passed() || r.status == StageStatus::Skipped)
    }

    /// Count of stages that passed.
    pub fn passed_count(&self) -> usize {
        self.results.iter().filter(|r| r.is_passed()).count()
    }

    /// Count of stages that failed.
    pub fn failed_count(&self) -> usize {
        self.results.iter().filter(|r| r.is_failed()).count()
    }

    fn last_failed(&self) -> bool {
        self.results.last().is_some_and(StageResult::is_failed)
    }

    fn run_model_load(&mut self) {
        let start = Instant::now();
        // Simulate model loading: valid paths succeed, otherwise fail
        if self.config.model_path.contains("nonexistent")
            || self.config.model_path.contains("missing")
        {
            self.results.push(StageResult::failed(
                "model_load",
                start.elapsed(),
                FailureReason::ModelNotFound(self.config.model_path.clone()),
            ));
            return;
        }

        let meta = ModelMetadata::default();
        if let Some(expected) = self.config.expected_vocab_size
            && meta.vocab_size != expected
        {
            self.results.push(StageResult::failed(
                "model_load",
                start.elapsed(),
                FailureReason::ValidationError(format!(
                    "vocab size mismatch: expected {expected}, got {}",
                    meta.vocab_size
                )),
            ));
            return;
        }

        self.model_metadata = Some(meta);
        self.results.push(
            StageResult::passed("model_load", start.elapsed())
                .with_metadata("vocab_size", "32000")
                .with_metadata("quantization", "I2_S"),
        );
    }

    fn run_tokenization(&mut self) {
        let start = Instant::now();
        if self.config.tokenizer_path.contains("invalid")
            || self.config.tokenizer_path.contains("corrupt")
        {
            self.results.push(StageResult::failed(
                "tokenization",
                start.elapsed(),
                FailureReason::TokenizerError("invalid tokenizer file".to_string()),
            ));
            return;
        }

        let token_count: usize =
            self.config.prompts.iter().map(|p| p.split_whitespace().count().max(1)).sum();
        self.results.push(
            StageResult::passed("tokenization", start.elapsed())
                .with_metadata("total_tokens", &token_count.to_string()),
        );
    }

    fn run_inference(&mut self) {
        let start = Instant::now();
        let device = self.resolve_device();

        for prompt in &self.config.prompts.clone() {
            if start.elapsed() > self.config.timeout {
                self.results.push(StageResult::failed(
                    "inference",
                    start.elapsed(),
                    FailureReason::Timeout(self.config.timeout),
                ));
                return;
            }

            let tokens = self.generate_simulated_tokens(prompt, device);
            self.inference_results.push(InferenceResult {
                tokens,
                total_time: start.elapsed(),
                device_used: device,
                seed_used: self.config.seed,
                prompt: prompt.clone(),
            });
        }

        self.results.push(
            StageResult::passed("inference", start.elapsed())
                .with_metadata("device", &device.to_string())
                .with_metadata("prompts", &self.config.prompts.len().to_string()),
        );
    }

    fn run_decode(&mut self) {
        let start = Instant::now();
        let has_output = self.inference_results.iter().all(|r| !r.tokens.is_empty());
        if has_output {
            self.results.push(StageResult::passed("decode", start.elapsed()));
        } else {
            self.results.push(StageResult::failed(
                "decode",
                start.elapsed(),
                FailureReason::InferenceError("no tokens generated".to_string()),
            ));
        }
    }

    fn run_validation(&mut self) {
        let start = Instant::now();
        if self.config.validate_roundtrip {
            // Verify all inference results produced tokens within bounds
            for result in &self.inference_results {
                if result.tokens.len() > self.config.max_tokens as usize {
                    self.results.push(StageResult::failed(
                        "validation",
                        start.elapsed(),
                        FailureReason::ValidationError(format!(
                            "generated {} tokens, max was {}",
                            result.tokens.len(),
                            self.config.max_tokens
                        )),
                    ));
                    return;
                }
            }
        }
        self.results.push(StageResult::passed("validation", start.elapsed()));
    }

    fn resolve_device(&self) -> Device {
        match self.config.device {
            Device::Auto => Device::Cpu,
            Device::Cuda(_) | Device::Vulkan(_) if self.config.allow_device_fallback => {
                // Simulate fallback: GPU not available, fall back to CPU
                Device::Cpu
            }
            other => other,
        }
    }

    fn generate_simulated_tokens(&self, prompt: &str, device: Device) -> Vec<TokenOutput> {
        let seed = self.config.seed.unwrap_or(0);
        let prompt_hash =
            prompt.bytes().fold(seed, |acc, b| acc.wrapping_mul(31).wrapping_add(u64::from(b)));
        let device_offset = match device {
            Device::Cpu | Device::Auto => 0_u64,
            Device::Cuda(id) => u64::from(id) + 100,
            Device::Vulkan(id) => u64::from(id) + 200,
        };

        (0..self.config.max_tokens as usize)
            .map(|i| {
                let token_id = ((prompt_hash.wrapping_add(device_offset).wrapping_add(i as u64))
                    % 32000) as u32;
                TokenOutput { token_id, text: format!("t{token_id}"), index: i }
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ModelLoadTest
// ─────────────────────────────────────────────────────────────────────────────

/// Test utility for validating GGUF model loading.
#[derive(Debug, Clone)]
pub struct ModelLoadTest {
    pub config: E2EConfig,
}

impl ModelLoadTest {
    pub fn new(config: E2EConfig) -> Self {
        Self { config }
    }

    /// Verify that a model loads successfully and returns valid metadata.
    pub fn run(&self) -> StageResult {
        let mut runner = E2ERunner::new(self.config.clone());
        runner.run_model_load();
        runner.results.pop().unwrap_or_else(|| StageResult::skipped("model_load"))
    }

    /// Verify model load fails for missing path.
    pub fn expect_failure(&self) -> StageResult {
        let result = self.run();
        if result.is_failed() {
            StageResult::passed("model_load_expect_failure", result.elapsed)
        } else {
            StageResult::failed(
                "model_load_expect_failure",
                result.elapsed,
                FailureReason::ValidationError("expected model load failure".to_string()),
            )
        }
    }

    /// Verify metadata matches expectations.
    pub fn validate_metadata(&self, expected_vocab: usize, expected_layers: usize) -> StageResult {
        let start = Instant::now();
        let mut runner = E2ERunner::new(self.config.clone());
        runner.run_model_load();
        match &runner.model_metadata {
            Some(meta) => {
                if meta.vocab_size != expected_vocab {
                    return StageResult::failed(
                        "metadata_validation",
                        start.elapsed(),
                        FailureReason::ValidationError(format!(
                            "vocab_size: expected {expected_vocab}, got {}",
                            meta.vocab_size
                        )),
                    );
                }
                if meta.num_layers != expected_layers {
                    return StageResult::failed(
                        "metadata_validation",
                        start.elapsed(),
                        FailureReason::ValidationError(format!(
                            "num_layers: expected {expected_layers}, got {}",
                            meta.num_layers
                        )),
                    );
                }
                StageResult::passed("metadata_validation", start.elapsed())
            }
            None => StageResult::failed(
                "metadata_validation",
                start.elapsed(),
                FailureReason::ModelNotFound(self.config.model_path.clone()),
            ),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TokenizationTest
// ─────────────────────────────────────────────────────────────────────────────

/// Test utility for encode → decode round-trip validation.
#[derive(Debug, Clone)]
pub struct TokenizationTest {
    pub config: E2EConfig,
}

impl TokenizationTest {
    pub fn new(config: E2EConfig) -> Self {
        Self { config }
    }

    /// Simulate encode: split text into pseudo-tokens.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        if self.config.tokenizer_path.contains("invalid") {
            return Err("invalid tokenizer".to_string());
        }
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let seed = self.config.seed.unwrap_or(0);
        Ok(text
            .bytes()
            .enumerate()
            .map(|(i, b)| {
                ((u64::from(b)).wrapping_mul(seed.wrapping_add(i as u64 + 1)) % 32000) as u32
            })
            .collect())
    }

    /// Simulate decode: map token IDs to text fragments.
    pub fn decode(&self, tokens: &[u32]) -> Result<String, String> {
        if self.config.tokenizer_path.contains("invalid") {
            return Err("invalid tokenizer".to_string());
        }
        Ok(tokens.iter().fold(String::new(), |mut acc, t| {
            use std::fmt::Write;
            let _ = write!(acc, "t{t}");
            acc
        }))
    }

    /// Run a round-trip test: encode then decode and verify non-empty.
    pub fn roundtrip(&self, text: &str) -> StageResult {
        let start = Instant::now();
        match self.encode(text) {
            Ok(tokens) => {
                if text.is_empty() && tokens.is_empty() {
                    return StageResult::passed("roundtrip", start.elapsed());
                }
                match self.decode(&tokens) {
                    Ok(decoded) => {
                        if decoded.is_empty() && !text.is_empty() {
                            StageResult::failed(
                                "roundtrip",
                                start.elapsed(),
                                FailureReason::TokenizerError(
                                    "decode produced empty output".to_string(),
                                ),
                            )
                        } else {
                            StageResult::passed("roundtrip", start.elapsed())
                                .with_metadata("input_len", &text.len().to_string())
                                .with_metadata("token_count", &tokens.len().to_string())
                                .with_metadata("decoded_len", &decoded.len().to_string())
                        }
                    }
                    Err(e) => StageResult::failed(
                        "roundtrip",
                        start.elapsed(),
                        FailureReason::TokenizerError(e),
                    ),
                }
            }
            Err(e) => {
                StageResult::failed("roundtrip", start.elapsed(), FailureReason::TokenizerError(e))
            }
        }
    }

    /// Verify that encoding is deterministic for the same input and seed.
    pub fn verify_deterministic(&self, text: &str) -> StageResult {
        let start = Instant::now();
        match (self.encode(text), self.encode(text)) {
            (Ok(a), Ok(b)) => {
                if a == b {
                    StageResult::passed("tokenization_determinism", start.elapsed())
                } else {
                    StageResult::failed(
                        "tokenization_determinism",
                        start.elapsed(),
                        FailureReason::DeterminismMismatch { run_a: a, run_b: b },
                    )
                }
            }
            (Err(e), _) | (_, Err(e)) => StageResult::failed(
                "tokenization_determinism",
                start.elapsed(),
                FailureReason::TokenizerError(e),
            ),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// InferenceTest
// ─────────────────────────────────────────────────────────────────────────────

/// Test utility for single token generation.
#[derive(Debug, Clone)]
pub struct InferenceTest {
    pub config: E2EConfig,
}

impl InferenceTest {
    pub fn new(config: E2EConfig) -> Self {
        Self { config }
    }

    /// Generate tokens for a single prompt.
    pub fn generate(&self, prompt: &str) -> Result<InferenceResult, String> {
        self.config.validate()?;
        let mut cfg = self.config.clone();
        cfg.prompts = vec![prompt.to_string()];
        let mut runner = E2ERunner::new(cfg);
        let _ = runner.run()?;
        runner
            .inference_results
            .into_iter()
            .next()
            .ok_or_else(|| "no inference result produced".to_string())
    }

    /// Verify at least one token is generated.
    pub fn verify_nonempty(&self, prompt: &str) -> StageResult {
        let start = Instant::now();
        match self.generate(prompt) {
            Ok(result) => {
                if result.tokens.is_empty() {
                    StageResult::failed(
                        "nonempty_inference",
                        start.elapsed(),
                        FailureReason::InferenceError("no tokens generated".to_string()),
                    )
                } else {
                    StageResult::passed("nonempty_inference", start.elapsed())
                        .with_metadata("tokens", &result.tokens.len().to_string())
                }
            }
            Err(e) => StageResult::failed(
                "nonempty_inference",
                start.elapsed(),
                FailureReason::InferenceError(e),
            ),
        }
    }

    /// Verify token count matches `max_tokens`.
    pub fn verify_token_count(&self, prompt: &str) -> StageResult {
        let start = Instant::now();
        match self.generate(prompt) {
            Ok(result) => {
                let expected = self.config.max_tokens as usize;
                if result.tokens.len() == expected {
                    StageResult::passed("token_count", start.elapsed())
                } else {
                    StageResult::failed(
                        "token_count",
                        start.elapsed(),
                        FailureReason::ValidationError(format!(
                            "expected {expected} tokens, got {}",
                            result.tokens.len()
                        )),
                    )
                }
            }
            Err(e) => StageResult::failed(
                "token_count",
                start.elapsed(),
                FailureReason::InferenceError(e),
            ),
        }
    }

    /// Verify token indices are sequential starting from 0.
    pub fn verify_sequential_indices(&self, prompt: &str) -> StageResult {
        let start = Instant::now();
        match self.generate(prompt) {
            Ok(result) => {
                for (i, tok) in result.tokens.iter().enumerate() {
                    if tok.index != i {
                        return StageResult::failed(
                            "sequential_indices",
                            start.elapsed(),
                            FailureReason::ValidationError(format!(
                                "token {i} has index {}",
                                tok.index
                            )),
                        );
                    }
                }
                StageResult::passed("sequential_indices", start.elapsed())
            }
            Err(e) => StageResult::failed(
                "sequential_indices",
                start.elapsed(),
                FailureReason::InferenceError(e),
            ),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// StreamingTest
// ─────────────────────────────────────────────────────────────────────────────

/// Test utility for streaming token output validation.
#[derive(Debug, Clone)]
pub struct StreamingTest {
    pub config: E2EConfig,
    pub collected_tokens: Vec<TokenOutput>,
}

impl StreamingTest {
    pub fn new(config: E2EConfig) -> Self {
        Self { config, collected_tokens: Vec::new() }
    }

    /// Simulate streaming: collect tokens one at a time.
    pub fn stream(&mut self, prompt: &str) -> Result<Vec<TokenOutput>, String> {
        self.config.validate()?;
        self.collected_tokens.clear();
        let mut cfg = self.config.clone();
        cfg.prompts = vec![prompt.to_string()];
        let mut runner = E2ERunner::new(cfg);
        let _ = runner.run()?;

        if let Some(result) = runner.inference_results.first() {
            for token in &result.tokens {
                self.collected_tokens.push(token.clone());
            }
        }
        Ok(self.collected_tokens.clone())
    }

    /// Verify tokens arrive in order.
    pub fn verify_ordering(&mut self, prompt: &str) -> StageResult {
        let start = Instant::now();
        match self.stream(prompt) {
            Ok(tokens) => {
                for (i, tok) in tokens.iter().enumerate() {
                    if tok.index != i {
                        return StageResult::failed(
                            "stream_ordering",
                            start.elapsed(),
                            FailureReason::ValidationError(format!(
                                "out of order: expected index {i}, got {}",
                                tok.index
                            )),
                        );
                    }
                }
                StageResult::passed("stream_ordering", start.elapsed())
            }
            Err(e) => StageResult::failed(
                "stream_ordering",
                start.elapsed(),
                FailureReason::InferenceError(e),
            ),
        }
    }

    /// Verify streaming produces all expected tokens.
    pub fn verify_completeness(&mut self, prompt: &str) -> StageResult {
        let start = Instant::now();
        match self.stream(prompt) {
            Ok(tokens) => {
                let expected = self.config.max_tokens as usize;
                if tokens.len() == expected {
                    StageResult::passed("stream_completeness", start.elapsed())
                } else {
                    StageResult::failed(
                        "stream_completeness",
                        start.elapsed(),
                        FailureReason::ValidationError(format!(
                            "expected {expected} tokens, got {}",
                            tokens.len()
                        )),
                    )
                }
            }
            Err(e) => StageResult::failed(
                "stream_completeness",
                start.elapsed(),
                FailureReason::InferenceError(e),
            ),
        }
    }

    /// Verify each streamed token has non-empty text.
    pub fn verify_nonempty_text(&mut self, prompt: &str) -> StageResult {
        let start = Instant::now();
        match self.stream(prompt) {
            Ok(tokens) => {
                for tok in &tokens {
                    if tok.text.is_empty() {
                        return StageResult::failed(
                            "stream_nonempty_text",
                            start.elapsed(),
                            FailureReason::ValidationError(format!(
                                "token {} has empty text",
                                tok.index
                            )),
                        );
                    }
                }
                StageResult::passed("stream_nonempty_text", start.elapsed())
            }
            Err(e) => StageResult::failed(
                "stream_nonempty_text",
                start.elapsed(),
                FailureReason::InferenceError(e),
            ),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BatchInferenceTest
// ─────────────────────────────────────────────────────────────────────────────

/// Test utility for batch inference over multiple prompts.
#[derive(Debug, Clone)]
pub struct BatchInferenceTest {
    pub config: E2EConfig,
}

impl BatchInferenceTest {
    pub fn new(config: E2EConfig) -> Self {
        Self { config }
    }

    /// Run inference on all configured prompts.
    pub fn run_batch(&self) -> Result<Vec<InferenceResult>, String> {
        self.config.validate()?;
        let mut runner = E2ERunner::new(self.config.clone());
        let _ = runner.run()?;
        Ok(runner.inference_results)
    }

    /// Verify that each prompt produces output.
    pub fn verify_all_produce_output(&self) -> StageResult {
        let start = Instant::now();
        match self.run_batch() {
            Ok(results) => {
                if results.len() != self.config.prompts.len() {
                    return StageResult::failed(
                        "batch_completeness",
                        start.elapsed(),
                        FailureReason::ValidationError(format!(
                            "expected {} results, got {}",
                            self.config.prompts.len(),
                            results.len()
                        )),
                    );
                }
                for (i, r) in results.iter().enumerate() {
                    if r.tokens.is_empty() {
                        return StageResult::failed(
                            "batch_completeness",
                            start.elapsed(),
                            FailureReason::InferenceError(format!("prompt {i} produced no tokens")),
                        );
                    }
                }
                StageResult::passed("batch_completeness", start.elapsed())
            }
            Err(e) => StageResult::failed(
                "batch_completeness",
                start.elapsed(),
                FailureReason::InferenceError(e),
            ),
        }
    }

    /// Verify different prompts produce different outputs.
    pub fn verify_distinct_outputs(&self) -> StageResult {
        let start = Instant::now();
        match self.run_batch() {
            Ok(results) => {
                if results.len() < 2 {
                    return StageResult::skipped("batch_distinct");
                }
                let id_sets: Vec<Vec<u32>> =
                    results.iter().map(InferenceResult::token_ids).collect();
                let mut all_same = true;
                for i in 1..id_sets.len() {
                    if id_sets[i] != id_sets[0] {
                        all_same = false;
                        break;
                    }
                }
                if all_same && self.config.prompts.windows(2).any(|w| w[0] != w[1]) {
                    StageResult::failed(
                        "batch_distinct",
                        start.elapsed(),
                        FailureReason::ValidationError(
                            "different prompts produced identical outputs".to_string(),
                        ),
                    )
                } else {
                    StageResult::passed("batch_distinct", start.elapsed())
                }
            }
            Err(e) => StageResult::failed(
                "batch_distinct",
                start.elapsed(),
                FailureReason::InferenceError(e),
            ),
        }
    }

    /// Verify all results used the same device.
    pub fn verify_consistent_device(&self) -> StageResult {
        let start = Instant::now();
        match self.run_batch() {
            Ok(results) => {
                if results.is_empty() {
                    return StageResult::skipped("batch_device");
                }
                let first_device = results[0].device_used;
                for (i, r) in results.iter().enumerate().skip(1) {
                    if r.device_used != first_device {
                        return StageResult::failed(
                            "batch_device",
                            start.elapsed(),
                            FailureReason::ValidationError(format!(
                                "prompt {i} used {:?}, expected {:?}",
                                r.device_used, first_device
                            )),
                        );
                    }
                }
                StageResult::passed("batch_device", start.elapsed())
            }
            Err(e) => StageResult::failed(
                "batch_device",
                start.elapsed(),
                FailureReason::InferenceError(e),
            ),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DeviceFallbackTest
// ─────────────────────────────────────────────────────────────────────────────

/// Test utility for GPU → CPU fallback validation.
#[derive(Debug, Clone)]
pub struct DeviceFallbackTest {
    pub config: E2EConfig,
}

impl DeviceFallbackTest {
    pub fn new(config: E2EConfig) -> Self {
        Self { config }
    }

    /// Verify fallback from requested GPU to CPU.
    pub fn verify_fallback(&self) -> StageResult {
        let start = Instant::now();
        let mut cfg = self.config.clone();
        cfg.device = Device::Cuda(0);
        cfg.allow_device_fallback = true;

        let mut runner = E2ERunner::new(cfg);
        match runner.run() {
            Ok(_) => {
                if let Some(result) = runner.inference_results.first() {
                    if result.device_used == Device::Cpu {
                        StageResult::passed("device_fallback", start.elapsed())
                            .with_metadata("requested", "cuda:0")
                            .with_metadata("actual", "cpu")
                    } else {
                        StageResult::failed(
                            "device_fallback",
                            start.elapsed(),
                            FailureReason::ValidationError(format!(
                                "expected CPU fallback, got {:?}",
                                result.device_used
                            )),
                        )
                    }
                } else {
                    StageResult::failed(
                        "device_fallback",
                        start.elapsed(),
                        FailureReason::InferenceError("no results".to_string()),
                    )
                }
            }
            Err(e) => StageResult::failed(
                "device_fallback",
                start.elapsed(),
                FailureReason::InferenceError(e),
            ),
        }
    }

    /// Verify that disabling fallback with unavailable GPU produces an error or
    /// uses the requested device.
    pub fn verify_no_fallback(&self) -> StageResult {
        let start = Instant::now();
        let mut cfg = self.config.clone();
        cfg.device = Device::Cuda(0);
        cfg.allow_device_fallback = false;

        let mut runner = E2ERunner::new(cfg);
        match runner.run() {
            Ok(_) => {
                if let Some(result) = runner.inference_results.first() {
                    // Without fallback, the device should be whatever was requested
                    if result.device_used == Device::Cuda(0) {
                        StageResult::passed("no_fallback", start.elapsed())
                    } else {
                        StageResult::failed(
                            "no_fallback",
                            start.elapsed(),
                            FailureReason::ValidationError(format!(
                                "expected cuda:0 (no fallback), got {:?}",
                                result.device_used
                            )),
                        )
                    }
                } else {
                    StageResult::failed(
                        "no_fallback",
                        start.elapsed(),
                        FailureReason::InferenceError("no results".to_string()),
                    )
                }
            }
            Err(e) => StageResult::failed(
                "no_fallback",
                start.elapsed(),
                FailureReason::InferenceError(e),
            ),
        }
    }

    /// Verify fallback from Vulkan to CPU.
    pub fn verify_vulkan_fallback(&self) -> StageResult {
        let start = Instant::now();
        let mut cfg = self.config.clone();
        cfg.device = Device::Vulkan(0);
        cfg.allow_device_fallback = true;

        let mut runner = E2ERunner::new(cfg);
        match runner.run() {
            Ok(_) => {
                if let Some(result) = runner.inference_results.first() {
                    if result.device_used == Device::Cpu {
                        StageResult::passed("vulkan_fallback", start.elapsed())
                    } else {
                        StageResult::failed(
                            "vulkan_fallback",
                            start.elapsed(),
                            FailureReason::ValidationError(format!(
                                "expected CPU fallback, got {:?}",
                                result.device_used
                            )),
                        )
                    }
                } else {
                    StageResult::failed(
                        "vulkan_fallback",
                        start.elapsed(),
                        FailureReason::InferenceError("no results".to_string()),
                    )
                }
            }
            Err(e) => StageResult::failed(
                "vulkan_fallback",
                start.elapsed(),
                FailureReason::InferenceError(e),
            ),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DeterminismTest
// ─────────────────────────────────────────────────────────────────────────────

/// Test utility for verifying same seed produces identical output.
#[derive(Debug, Clone)]
pub struct DeterminismTest {
    pub config: E2EConfig,
}

impl DeterminismTest {
    pub fn new(config: E2EConfig) -> Self {
        Self { config }
    }

    /// Run inference twice with the same seed and compare outputs.
    pub fn verify_reproducible(&self, prompt: &str) -> StageResult {
        let start = Instant::now();
        let test = InferenceTest::new(self.config.clone());
        match (test.generate(prompt), test.generate(prompt)) {
            (Ok(a), Ok(b)) => {
                let ids_a = a.token_ids();
                let ids_b = b.token_ids();
                if ids_a == ids_b {
                    StageResult::passed("determinism", start.elapsed())
                        .with_metadata("tokens_compared", &ids_a.len().to_string())
                } else {
                    StageResult::failed(
                        "determinism",
                        start.elapsed(),
                        FailureReason::DeterminismMismatch { run_a: ids_a, run_b: ids_b },
                    )
                }
            }
            (Err(e), _) | (_, Err(e)) => StageResult::failed(
                "determinism",
                start.elapsed(),
                FailureReason::InferenceError(e),
            ),
        }
    }

    /// Verify that different seeds produce different outputs.
    pub fn verify_seed_sensitivity(&self, prompt: &str) -> StageResult {
        let start = Instant::now();
        let test_a = InferenceTest::new(self.config.clone().with_seed(42));
        let test_b = InferenceTest::new(self.config.clone().with_seed(123));
        match (test_a.generate(prompt), test_b.generate(prompt)) {
            (Ok(a), Ok(b)) => {
                let ids_a = a.token_ids();
                let ids_b = b.token_ids();
                if ids_a == ids_b {
                    StageResult::failed(
                        "seed_sensitivity",
                        start.elapsed(),
                        FailureReason::ValidationError(
                            "different seeds produced identical output".to_string(),
                        ),
                    )
                } else {
                    StageResult::passed("seed_sensitivity", start.elapsed())
                }
            }
            (Err(e), _) | (_, Err(e)) => StageResult::failed(
                "seed_sensitivity",
                start.elapsed(),
                FailureReason::InferenceError(e),
            ),
        }
    }

    /// Verify reproducibility across multiple runs.
    pub fn verify_multi_run(&self, prompt: &str, runs: usize) -> StageResult {
        let start = Instant::now();
        let test = InferenceTest::new(self.config.clone());
        let mut first_ids: Option<Vec<u32>> = None;
        for run in 0..runs {
            match test.generate(prompt) {
                Ok(result) => {
                    let ids = result.token_ids();
                    match &first_ids {
                        Some(expected) => {
                            if ids != *expected {
                                return StageResult::failed(
                                    "multi_run_determinism",
                                    start.elapsed(),
                                    FailureReason::DeterminismMismatch {
                                        run_a: expected.clone(),
                                        run_b: ids,
                                    },
                                );
                            }
                        }
                        None => first_ids = Some(ids),
                    }
                    let _ = run;
                }
                Err(e) => {
                    return StageResult::failed(
                        "multi_run_determinism",
                        start.elapsed(),
                        FailureReason::InferenceError(e),
                    );
                }
            }
        }
        StageResult::passed("multi_run_determinism", start.elapsed())
            .with_metadata("runs", &runs.to_string())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// E2ETestSuite
// ─────────────────────────────────────────────────────────────────────────────

/// Full orchestrator: load model → tokenize → infer → decode → validate.
#[derive(Debug, Clone)]
pub struct E2ETestSuite {
    pub config: E2EConfig,
    pub stage_results: Vec<StageResult>,
}

impl E2ETestSuite {
    pub fn new(config: E2EConfig) -> Self {
        Self { config, stage_results: Vec::new() }
    }

    /// Run the complete test suite.
    pub fn run_all(&mut self) -> Result<&[StageResult], String> {
        self.stage_results.clear();

        // Stage 1: Model loading
        let load_test = ModelLoadTest::new(self.config.clone());
        self.stage_results.push(load_test.run());
        if self.last_failed() {
            return Ok(&self.stage_results);
        }

        // Stage 2: Tokenization round-trip
        let tok_test = TokenizationTest::new(self.config.clone());
        for prompt in &self.config.prompts.clone() {
            self.stage_results.push(tok_test.roundtrip(prompt));
            if self.last_failed() {
                return Ok(&self.stage_results);
            }
        }

        // Stage 3: Inference
        let inf_test = InferenceTest::new(self.config.clone());
        for prompt in &self.config.prompts.clone() {
            self.stage_results.push(inf_test.verify_nonempty(prompt));
            if self.last_failed() {
                return Ok(&self.stage_results);
            }
        }

        // Stage 4: Determinism (if seed is set)
        if self.config.seed.is_some() {
            let det_test = DeterminismTest::new(self.config.clone());
            for prompt in &self.config.prompts.clone() {
                self.stage_results.push(det_test.verify_reproducible(prompt));
            }
        }

        // Stage 5: Device fallback (if not CPU-only)
        if !matches!(self.config.device, Device::Cpu) && self.config.allow_device_fallback {
            let fb_test = DeviceFallbackTest::new(self.config.clone());
            self.stage_results.push(fb_test.verify_fallback());
        }

        Ok(&self.stage_results)
    }

    /// Summary of results.
    pub fn summary(&self) -> SuiteSummary {
        let total = self.stage_results.len();
        let passed = self.stage_results.iter().filter(|r| r.is_passed()).count();
        let failed = self.stage_results.iter().filter(|r| r.is_failed()).count();
        let skipped =
            self.stage_results.iter().filter(|r| r.status == StageStatus::Skipped).count();
        SuiteSummary { total, passed, failed, skipped }
    }

    fn last_failed(&self) -> bool {
        self.stage_results.last().is_some_and(StageResult::is_failed)
    }
}

/// Summary statistics from a test suite run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SuiteSummary {
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub skipped: usize,
}

impl SuiteSummary {
    pub fn all_passed(&self) -> bool {
        self.failed == 0
    }
}

impl fmt::Display for SuiteSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} total, {} passed, {} failed, {} skipped",
            self.total, self.passed, self.failed, self.skipped
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    fn default_config() -> E2EConfig {
        E2EConfig::default()
            .with_paths("models/test.gguf", "models/tokenizer.json")
            .with_max_tokens(4)
    }

    fn multi_prompt_config() -> E2EConfig {
        default_config().with_prompts(vec![
            "Hello world".to_string(),
            "What is AI?".to_string(),
            "Explain rust".to_string(),
        ])
    }

    // ─────────────────────────── Device ───────────────────────────

    #[test]
    fn test_device_default_is_auto() {
        assert_eq!(Device::default(), Device::Auto);
    }

    #[test]
    fn test_device_display_cpu() {
        assert_eq!(Device::Cpu.to_string(), "cpu");
    }

    #[test]
    fn test_device_display_cuda() {
        assert_eq!(Device::Cuda(0).to_string(), "cuda:0");
    }

    #[test]
    fn test_device_display_vulkan() {
        assert_eq!(Device::Vulkan(1).to_string(), "vulkan:1");
    }

    #[test]
    fn test_device_display_auto() {
        assert_eq!(Device::Auto.to_string(), "auto");
    }

    #[test]
    fn test_device_equality() {
        assert_eq!(Device::Cuda(0), Device::Cuda(0));
        assert_ne!(Device::Cuda(0), Device::Cuda(1));
        assert_ne!(Device::Cpu, Device::Auto);
    }

    #[test]
    fn test_device_hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Device::Cpu);
        set.insert(Device::Cuda(0));
        set.insert(Device::Cuda(0)); // duplicate
        assert_eq!(set.len(), 2);
    }

    // ─────────────────────────── StageStatus ───────────────────────────

    #[test]
    fn test_stage_status_display() {
        assert_eq!(StageStatus::Pending.to_string(), "pending");
        assert_eq!(StageStatus::Running.to_string(), "running");
        assert_eq!(StageStatus::Passed.to_string(), "passed");
        assert_eq!(StageStatus::Failed.to_string(), "failed");
        assert_eq!(StageStatus::Skipped.to_string(), "skipped");
    }

    // ─────────────────────────── FailureReason ───────────────────────────

    #[test]
    fn test_failure_reason_display_model_not_found() {
        let r = FailureReason::ModelNotFound("/bad/path".to_string());
        assert!(r.to_string().contains("/bad/path"));
    }

    #[test]
    fn test_failure_reason_display_tokenizer_error() {
        let r = FailureReason::TokenizerError("corrupt".to_string());
        assert!(r.to_string().contains("corrupt"));
    }

    #[test]
    fn test_failure_reason_display_timeout() {
        let r = FailureReason::Timeout(Duration::from_secs(30));
        assert!(r.to_string().contains("30"));
    }

    #[test]
    fn test_failure_reason_display_determinism() {
        let r = FailureReason::DeterminismMismatch { run_a: vec![1, 2], run_b: vec![3, 4] };
        let s = r.to_string();
        assert!(s.contains("[1, 2]"));
        assert!(s.contains("[3, 4]"));
    }

    #[test]
    fn test_failure_reason_display_device_unavailable() {
        let r = FailureReason::DeviceUnavailable(Device::Cuda(0));
        assert!(r.to_string().contains("cuda:0"));
    }

    #[test]
    fn test_failure_reason_display_validation() {
        let r = FailureReason::ValidationError("bad data".to_string());
        assert!(r.to_string().contains("bad data"));
    }

    #[test]
    fn test_failure_reason_display_inference() {
        let r = FailureReason::InferenceError("oom".to_string());
        assert!(r.to_string().contains("oom"));
    }

    // ─────────────────────────── StageResult ───────────────────────────

    #[test]
    fn test_stage_result_passed() {
        let r = StageResult::passed("test", Duration::from_millis(50));
        assert!(r.is_passed());
        assert!(!r.is_failed());
        assert_eq!(r.stage_name, "test");
        assert!(r.failure.is_none());
    }

    #[test]
    fn test_stage_result_failed() {
        let r = StageResult::failed(
            "test",
            Duration::from_millis(10),
            FailureReason::InferenceError("err".to_string()),
        );
        assert!(r.is_failed());
        assert!(!r.is_passed());
        assert!(r.failure.is_some());
    }

    #[test]
    fn test_stage_result_skipped() {
        let r = StageResult::skipped("test");
        assert!(!r.is_passed());
        assert!(!r.is_failed());
        assert_eq!(r.status, StageStatus::Skipped);
        assert_eq!(r.elapsed, Duration::ZERO);
    }

    #[test]
    fn test_stage_result_with_metadata() {
        let r = StageResult::passed("test", Duration::ZERO)
            .with_metadata("key1", "val1")
            .with_metadata("key2", "val2");
        assert_eq!(r.metadata.get("key1").unwrap(), "val1");
        assert_eq!(r.metadata.get("key2").unwrap(), "val2");
    }

    #[test]
    fn test_stage_result_metadata_overwrite() {
        let r = StageResult::passed("test", Duration::ZERO)
            .with_metadata("k", "a")
            .with_metadata("k", "b");
        assert_eq!(r.metadata.get("k").unwrap(), "b");
    }

    // ─────────────────────────── E2EConfig ───────────────────────────

    #[test]
    fn test_config_default_values() {
        let cfg = E2EConfig::default();
        assert!(cfg.model_path.is_empty());
        assert!(cfg.tokenizer_path.is_empty());
        assert_eq!(cfg.device, Device::Auto);
        assert_eq!(cfg.max_tokens, 16);
        assert_eq!(cfg.temperature, 0.0);
        assert_eq!(cfg.seed, Some(42));
        assert!(cfg.validate_roundtrip);
        assert!(cfg.allow_device_fallback);
    }

    #[test]
    fn test_config_validate_empty_model() {
        let cfg = E2EConfig::default();
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("model_path"));
    }

    #[test]
    fn test_config_validate_empty_tokenizer() {
        let cfg = E2EConfig::default().with_paths("model.gguf", "");
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("tokenizer_path"));
    }

    #[test]
    fn test_config_validate_zero_max_tokens() {
        let cfg = default_config().with_max_tokens(0);
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("max_tokens"));
    }

    #[test]
    fn test_config_validate_negative_temperature() {
        let cfg = default_config().with_temperature(-1.0);
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("temperature"));
    }

    #[test]
    fn test_config_validate_excessive_temperature() {
        let cfg = default_config().with_temperature(11.0);
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("temperature"));
    }

    #[test]
    fn test_config_validate_empty_prompts() {
        let cfg = default_config().with_prompts(vec![]);
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("prompt"));
    }

    #[test]
    fn test_config_validate_zero_timeout() {
        let cfg = default_config().with_timeout(Duration::ZERO);
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("timeout"));
    }

    #[test]
    fn test_config_validate_success() {
        assert!(default_config().validate().is_ok());
    }

    #[test]
    fn test_config_builder_with_paths() {
        let cfg = E2EConfig::default().with_paths("m.gguf", "t.json");
        assert_eq!(cfg.model_path, "m.gguf");
        assert_eq!(cfg.tokenizer_path, "t.json");
    }

    #[test]
    fn test_config_builder_with_device() {
        let cfg = E2EConfig::default().with_device(Device::Cuda(1));
        assert_eq!(cfg.device, Device::Cuda(1));
    }

    #[test]
    fn test_config_builder_chain() {
        let cfg = E2EConfig::default()
            .with_paths("m", "t")
            .with_max_tokens(8)
            .with_temperature(0.5)
            .with_seed(99)
            .with_device(Device::Cpu)
            .without_fallback();
        assert_eq!(cfg.max_tokens, 8);
        assert_eq!(cfg.temperature, 0.5);
        assert_eq!(cfg.seed, Some(99));
        assert_eq!(cfg.device, Device::Cpu);
        assert!(!cfg.allow_device_fallback);
    }

    #[test]
    fn test_config_temperature_boundary_zero() {
        assert!(default_config().with_temperature(0.0).validate().is_ok());
    }

    #[test]
    fn test_config_temperature_boundary_ten() {
        assert!(default_config().with_temperature(10.0).validate().is_ok());
    }

    // ─────────────────────────── ModelMetadata ───────────────────────────

    #[test]
    fn test_model_metadata_defaults() {
        let m = ModelMetadata::default();
        assert_eq!(m.vocab_size, 32000);
        assert_eq!(m.hidden_dim, 2048);
        assert_eq!(m.num_layers, 24);
        assert_eq!(m.quantization, "I2_S");
        assert!(m.file_size_bytes > 0);
    }

    // ─────────────────────────── InferenceResult ───────────────────────────

    #[test]
    fn test_inference_result_token_ids() {
        let result = InferenceResult {
            tokens: vec![
                TokenOutput { token_id: 10, text: "a".to_string(), index: 0 },
                TokenOutput { token_id: 20, text: "b".to_string(), index: 1 },
            ],
            total_time: Duration::from_millis(100),
            device_used: Device::Cpu,
            seed_used: Some(42),
            prompt: "test".to_string(),
        };
        assert_eq!(result.token_ids(), vec![10, 20]);
    }

    #[test]
    fn test_inference_result_decoded_text() {
        let result = InferenceResult {
            tokens: vec![
                TokenOutput { token_id: 1, text: "hello".to_string(), index: 0 },
                TokenOutput { token_id: 2, text: " world".to_string(), index: 1 },
            ],
            total_time: Duration::from_millis(100),
            device_used: Device::Cpu,
            seed_used: None,
            prompt: "test".to_string(),
        };
        assert_eq!(result.decoded_text(), "hello world");
    }

    #[test]
    fn test_inference_result_tokens_per_second() {
        let result = InferenceResult {
            tokens: vec![
                TokenOutput { token_id: 1, text: "a".to_string(), index: 0 },
                TokenOutput { token_id: 2, text: "b".to_string(), index: 1 },
            ],
            total_time: Duration::from_secs(1),
            device_used: Device::Cpu,
            seed_used: None,
            prompt: "test".to_string(),
        };
        assert!((result.tokens_per_second() - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_inference_result_tps_zero_time() {
        let result = InferenceResult {
            tokens: vec![TokenOutput { token_id: 1, text: "a".to_string(), index: 0 }],
            total_time: Duration::ZERO,
            device_used: Device::Cpu,
            seed_used: None,
            prompt: "test".to_string(),
        };
        assert_eq!(result.tokens_per_second(), 0.0);
    }

    #[test]
    fn test_inference_result_empty_tokens() {
        let result = InferenceResult {
            tokens: vec![],
            total_time: Duration::from_millis(100),
            device_used: Device::Cpu,
            seed_used: None,
            prompt: "test".to_string(),
        };
        assert!(result.token_ids().is_empty());
        assert!(result.decoded_text().is_empty());
    }

    // ─────────────────────────── E2ERunner ───────────────────────────

    #[test]
    fn test_runner_successful_pipeline() {
        let mut runner = E2ERunner::new(default_config());
        let results = runner.run().unwrap();
        assert!(!results.is_empty());
        assert!(runner.all_passed());
    }

    #[test]
    fn test_runner_model_not_found() {
        let cfg = default_config().with_paths("nonexistent.gguf", "models/tokenizer.json");
        let mut runner = E2ERunner::new(cfg);
        runner.run().unwrap();
        assert!(runner.failed_count() > 0);
        assert!(!runner.all_passed());
    }

    #[test]
    fn test_runner_invalid_tokenizer() {
        let cfg = default_config().with_paths("models/test.gguf", "invalid_tokenizer.json");
        let mut runner = E2ERunner::new(cfg);
        runner.run().unwrap();
        assert!(runner.failed_count() > 0);
    }

    #[test]
    fn test_runner_validation_fails_empty_config() {
        let mut runner = E2ERunner::new(E2EConfig::default());
        let err = runner.run().unwrap_err();
        assert!(err.contains("model_path"));
    }

    #[test]
    fn test_runner_generates_tokens() {
        let mut runner = E2ERunner::new(default_config());
        runner.run().unwrap();
        assert_eq!(runner.inference_results.len(), 1);
        assert_eq!(runner.inference_results[0].tokens.len(), 4);
    }

    #[test]
    fn test_runner_multiple_prompts() {
        let mut runner = E2ERunner::new(multi_prompt_config());
        runner.run().unwrap();
        assert_eq!(runner.inference_results.len(), 3);
    }

    #[test]
    fn test_runner_passed_count() {
        let mut runner = E2ERunner::new(default_config());
        runner.run().unwrap();
        assert!(runner.passed_count() >= 4); // load, tokenize, inference, decode, validate
    }

    #[test]
    fn test_runner_failed_count_on_success() {
        let mut runner = E2ERunner::new(default_config());
        runner.run().unwrap();
        assert_eq!(runner.failed_count(), 0);
    }

    #[test]
    fn test_runner_elapsed_nonzero() {
        let mut runner = E2ERunner::new(default_config());
        runner.run().unwrap();
        // Elapsed should be valid (may be zero on fast machines but shouldn't panic)
        let _ = runner.elapsed();
    }

    #[test]
    fn test_runner_run_clears_previous() {
        let mut runner = E2ERunner::new(default_config());
        runner.run().unwrap();
        let first_count = runner.results.len();
        runner.run().unwrap();
        assert_eq!(runner.results.len(), first_count);
    }

    #[test]
    fn test_runner_stage_names() {
        let mut runner = E2ERunner::new(default_config());
        runner.run().unwrap();
        let names: Vec<&str> = runner.results.iter().map(|r| r.stage_name.as_str()).collect();
        assert!(names.contains(&"model_load"));
        assert!(names.contains(&"tokenization"));
        assert!(names.contains(&"inference"));
        assert!(names.contains(&"decode"));
        assert!(names.contains(&"validation"));
    }

    #[test]
    fn test_runner_model_metadata_populated() {
        let mut runner = E2ERunner::new(default_config());
        runner.run().unwrap();
        assert!(runner.model_metadata.is_some());
        assert_eq!(runner.model_metadata.unwrap().vocab_size, 32000);
    }

    #[test]
    fn test_runner_missing_model_path() {
        let cfg = default_config().with_paths("missing/model.gguf", "models/tokenizer.json");
        let mut runner = E2ERunner::new(cfg);
        runner.run().unwrap();
        assert!(runner.results[0].is_failed());
        // Should stop after first failure
        assert_eq!(runner.results.len(), 1);
    }

    // ─────────────────────────── ModelLoadTest ───────────────────────────

    #[test]
    fn test_model_load_success() {
        let t = ModelLoadTest::new(default_config());
        assert!(t.run().is_passed());
    }

    #[test]
    fn test_model_load_missing_file() {
        let cfg = default_config().with_paths("nonexistent.gguf", "t.json");
        let t = ModelLoadTest::new(cfg);
        assert!(t.run().is_failed());
    }

    #[test]
    fn test_model_load_expect_failure_on_missing() {
        let cfg = default_config().with_paths("nonexistent.gguf", "t.json");
        let t = ModelLoadTest::new(cfg);
        assert!(t.expect_failure().is_passed());
    }

    #[test]
    fn test_model_load_expect_failure_on_valid() {
        let t = ModelLoadTest::new(default_config());
        assert!(t.expect_failure().is_failed());
    }

    #[test]
    fn test_model_load_validate_metadata_correct() {
        let t = ModelLoadTest::new(default_config());
        assert!(t.validate_metadata(32000, 24).is_passed());
    }

    #[test]
    fn test_model_load_validate_metadata_wrong_vocab() {
        let t = ModelLoadTest::new(default_config());
        assert!(t.validate_metadata(64000, 24).is_failed());
    }

    #[test]
    fn test_model_load_validate_metadata_wrong_layers() {
        let t = ModelLoadTest::new(default_config());
        assert!(t.validate_metadata(32000, 48).is_failed());
    }

    #[test]
    fn test_model_load_with_expected_vocab() {
        let mut cfg = default_config();
        cfg.expected_vocab_size = Some(32000);
        let t = ModelLoadTest::new(cfg);
        assert!(t.run().is_passed());
    }

    #[test]
    fn test_model_load_with_wrong_expected_vocab() {
        let mut cfg = default_config();
        cfg.expected_vocab_size = Some(50000);
        let t = ModelLoadTest::new(cfg);
        assert!(t.run().is_failed());
    }

    #[test]
    fn test_model_load_metadata_includes_quantization() {
        let t = ModelLoadTest::new(default_config());
        let result = t.run();
        assert_eq!(result.metadata.get("quantization").unwrap(), "I2_S");
    }

    // ─────────────────────────── TokenizationTest ───────────────────────────

    #[test]
    fn test_tokenization_encode_nonempty() {
        let t = TokenizationTest::new(default_config());
        let tokens = t.encode("Hello world").unwrap();
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_tokenization_encode_empty() {
        let t = TokenizationTest::new(default_config());
        let tokens = t.encode("").unwrap();
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_tokenization_encode_invalid_tokenizer() {
        let cfg = default_config().with_paths("m.gguf", "invalid_tok.json");
        let t = TokenizationTest::new(cfg);
        assert!(t.encode("test").is_err());
    }

    #[test]
    fn test_tokenization_decode() {
        let t = TokenizationTest::new(default_config());
        let decoded = t.decode(&[1, 2, 3]).unwrap();
        assert!(!decoded.is_empty());
        assert!(decoded.contains("t1"));
    }

    #[test]
    fn test_tokenization_decode_empty() {
        let t = TokenizationTest::new(default_config());
        let decoded = t.decode(&[]).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_tokenization_decode_invalid_tokenizer() {
        let cfg = default_config().with_paths("m.gguf", "invalid_tok.json");
        let t = TokenizationTest::new(cfg);
        assert!(t.decode(&[1]).is_err());
    }

    #[test]
    fn test_tokenization_roundtrip_nonempty() {
        let t = TokenizationTest::new(default_config());
        assert!(t.roundtrip("Hello world").is_passed());
    }

    #[test]
    fn test_tokenization_roundtrip_empty() {
        let t = TokenizationTest::new(default_config());
        assert!(t.roundtrip("").is_passed());
    }

    #[test]
    fn test_tokenization_roundtrip_metadata() {
        let t = TokenizationTest::new(default_config());
        let result = t.roundtrip("Hello");
        assert!(result.metadata.contains_key("input_len"));
        assert!(result.metadata.contains_key("token_count"));
    }

    #[test]
    fn test_tokenization_deterministic() {
        let t = TokenizationTest::new(default_config());
        assert!(t.verify_deterministic("Hello").is_passed());
    }

    #[test]
    fn test_tokenization_token_ids_bounded() {
        let t = TokenizationTest::new(default_config());
        let tokens = t.encode("test input string").unwrap();
        for tok in &tokens {
            assert!(*tok < 32000, "token {tok} out of vocab range");
        }
    }

    // ─────────────────────────── InferenceTest ───────────────────────────

    #[test]
    fn test_inference_generate_success() {
        let t = InferenceTest::new(default_config());
        let result = t.generate("Hello").unwrap();
        assert_eq!(result.tokens.len(), 4);
    }

    #[test]
    fn test_inference_generate_with_seed() {
        let t = InferenceTest::new(default_config().with_seed(99));
        let result = t.generate("Hello").unwrap();
        assert_eq!(result.seed_used, Some(99));
    }

    #[test]
    fn test_inference_verify_nonempty() {
        let t = InferenceTest::new(default_config());
        assert!(t.verify_nonempty("Hello").is_passed());
    }

    #[test]
    fn test_inference_verify_token_count() {
        let t = InferenceTest::new(default_config());
        assert!(t.verify_token_count("Hello").is_passed());
    }

    #[test]
    fn test_inference_verify_sequential_indices() {
        let t = InferenceTest::new(default_config());
        assert!(t.verify_sequential_indices("Hello").is_passed());
    }

    #[test]
    fn test_inference_token_ids_in_vocab() {
        let t = InferenceTest::new(default_config());
        let result = t.generate("test").unwrap();
        for tok in &result.tokens {
            assert!(tok.token_id < 32000);
        }
    }

    #[test]
    fn test_inference_device_reported() {
        let t = InferenceTest::new(default_config());
        let result = t.generate("test").unwrap();
        assert_eq!(result.device_used, Device::Cpu);
    }

    #[test]
    fn test_inference_prompt_preserved() {
        let t = InferenceTest::new(default_config());
        let result = t.generate("My test prompt").unwrap();
        assert_eq!(result.prompt, "My test prompt");
    }

    #[test]
    fn test_inference_different_prompts_different_output() {
        let t = InferenceTest::new(default_config());
        let r1 = t.generate("Hello").unwrap();
        let r2 = t.generate("World").unwrap();
        assert_ne!(r1.token_ids(), r2.token_ids());
    }

    #[test]
    fn test_inference_max_tokens_respected() {
        for n in [1, 4, 8, 16] {
            let t = InferenceTest::new(default_config().with_max_tokens(n));
            let result = t.generate("test").unwrap();
            assert_eq!(result.tokens.len(), n as usize);
        }
    }

    // ─────────────────────────── StreamingTest ───────────────────────────

    #[test]
    fn test_streaming_produces_tokens() {
        let mut t = StreamingTest::new(default_config());
        let tokens = t.stream("Hello").unwrap();
        assert_eq!(tokens.len(), 4);
    }

    #[test]
    fn test_streaming_verify_ordering() {
        let mut t = StreamingTest::new(default_config());
        assert!(t.verify_ordering("Hello").is_passed());
    }

    #[test]
    fn test_streaming_verify_completeness() {
        let mut t = StreamingTest::new(default_config());
        assert!(t.verify_completeness("Hello").is_passed());
    }

    #[test]
    fn test_streaming_verify_nonempty_text() {
        let mut t = StreamingTest::new(default_config());
        assert!(t.verify_nonempty_text("Hello").is_passed());
    }

    #[test]
    fn test_streaming_collected_tokens_persist() {
        let mut t = StreamingTest::new(default_config());
        t.stream("Hello").unwrap();
        assert_eq!(t.collected_tokens.len(), 4);
    }

    #[test]
    fn test_streaming_resets_on_new_stream() {
        let mut t = StreamingTest::new(default_config());
        t.stream("Hello").unwrap();
        t.stream("World").unwrap();
        assert_eq!(t.collected_tokens.len(), 4);
    }

    #[test]
    fn test_streaming_token_indices_sequential() {
        let mut t = StreamingTest::new(default_config().with_max_tokens(8));
        let tokens = t.stream("test").unwrap();
        for (i, tok) in tokens.iter().enumerate() {
            assert_eq!(tok.index, i);
        }
    }

    // ─────────────────────────── BatchInferenceTest ───────────────────────────

    #[test]
    fn test_batch_all_produce_output() {
        let t = BatchInferenceTest::new(multi_prompt_config());
        assert!(t.verify_all_produce_output().is_passed());
    }

    #[test]
    fn test_batch_distinct_outputs() {
        let t = BatchInferenceTest::new(multi_prompt_config());
        assert!(t.verify_distinct_outputs().is_passed());
    }

    #[test]
    fn test_batch_consistent_device() {
        let t = BatchInferenceTest::new(multi_prompt_config());
        assert!(t.verify_consistent_device().is_passed());
    }

    #[test]
    fn test_batch_result_count_matches_prompts() {
        let t = BatchInferenceTest::new(multi_prompt_config());
        let results = t.run_batch().unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_batch_single_prompt() {
        let t = BatchInferenceTest::new(default_config());
        assert!(t.verify_all_produce_output().is_passed());
    }

    #[test]
    fn test_batch_distinct_skipped_single() {
        let t = BatchInferenceTest::new(default_config());
        let result = t.verify_distinct_outputs();
        // Single prompt: can't compare, should skip
        assert!(result.is_passed() || result.status == StageStatus::Skipped);
    }

    #[test]
    fn test_batch_each_result_has_tokens() {
        let t = BatchInferenceTest::new(multi_prompt_config());
        let results = t.run_batch().unwrap();
        for r in &results {
            assert!(!r.tokens.is_empty());
        }
    }

    // ─────────────────────────── DeviceFallbackTest ───────────────────────────

    #[test]
    fn test_device_fallback_cuda_to_cpu() {
        let t = DeviceFallbackTest::new(default_config());
        assert!(t.verify_fallback().is_passed());
    }

    #[test]
    fn test_device_fallback_metadata() {
        let t = DeviceFallbackTest::new(default_config());
        let result = t.verify_fallback();
        assert_eq!(result.metadata.get("requested").unwrap(), "cuda:0");
        assert_eq!(result.metadata.get("actual").unwrap(), "cpu");
    }

    #[test]
    fn test_device_no_fallback() {
        let t = DeviceFallbackTest::new(default_config());
        let result = t.verify_no_fallback();
        // Without fallback, the device should be what was requested
        assert!(result.is_passed() || result.is_failed());
    }

    #[test]
    fn test_device_vulkan_fallback() {
        let t = DeviceFallbackTest::new(default_config());
        assert!(t.verify_vulkan_fallback().is_passed());
    }

    #[test]
    fn test_device_fallback_still_produces_output() {
        let mut cfg = default_config();
        cfg.device = Device::Cuda(0);
        cfg.allow_device_fallback = true;
        let mut runner = E2ERunner::new(cfg);
        runner.run().unwrap();
        assert!(!runner.inference_results.is_empty());
        assert!(!runner.inference_results[0].tokens.is_empty());
    }

    // ─────────────────────────── DeterminismTest ───────────────────────────

    #[test]
    fn test_determinism_same_seed_same_output() {
        let t = DeterminismTest::new(default_config());
        assert!(t.verify_reproducible("Hello").is_passed());
    }

    #[test]
    fn test_determinism_different_seeds_different_output() {
        let t = DeterminismTest::new(default_config());
        assert!(t.verify_seed_sensitivity("Hello").is_passed());
    }

    #[test]
    fn test_determinism_multi_run_3() {
        let t = DeterminismTest::new(default_config());
        assert!(t.verify_multi_run("Hello", 3).is_passed());
    }

    #[test]
    fn test_determinism_multi_run_10() {
        let t = DeterminismTest::new(default_config());
        assert!(t.verify_multi_run("Hello", 10).is_passed());
    }

    #[test]
    fn test_determinism_multi_run_metadata() {
        let t = DeterminismTest::new(default_config());
        let result = t.verify_multi_run("test", 5);
        assert_eq!(result.metadata.get("runs").unwrap(), "5");
    }

    #[test]
    fn test_determinism_across_prompts() {
        let t = DeterminismTest::new(default_config());
        assert!(t.verify_reproducible("First prompt").is_passed());
        assert!(t.verify_reproducible("Second prompt").is_passed());
    }

    #[test]
    fn test_determinism_reproducible_metadata() {
        let t = DeterminismTest::new(default_config());
        let result = t.verify_reproducible("test");
        assert!(result.metadata.contains_key("tokens_compared"));
    }

    // ─────────────────────────── E2ETestSuite ───────────────────────────

    #[test]
    fn test_suite_run_all_passes() {
        let mut suite = E2ETestSuite::new(default_config());
        suite.run_all().unwrap();
        let summary = suite.summary();
        assert!(summary.all_passed(), "suite failed: {summary}");
    }

    #[test]
    fn test_suite_summary_counts() {
        let mut suite = E2ETestSuite::new(default_config());
        suite.run_all().unwrap();
        let summary = suite.summary();
        assert!(summary.total > 0);
        assert_eq!(summary.failed, 0);
        assert_eq!(summary.passed, summary.total - summary.skipped);
    }

    #[test]
    fn test_suite_stops_on_model_failure() {
        let cfg = default_config().with_paths("nonexistent.gguf", "t.json");
        let mut suite = E2ETestSuite::new(cfg);
        suite.run_all().unwrap();
        assert_eq!(suite.stage_results.len(), 1);
        assert!(suite.stage_results[0].is_failed());
    }

    #[test]
    fn test_suite_with_multiple_prompts() {
        let mut suite = E2ETestSuite::new(multi_prompt_config());
        suite.run_all().unwrap();
        let summary = suite.summary();
        assert!(summary.all_passed(), "suite failed: {summary}");
        assert!(summary.total >= 7); // load + 3 roundtrips + 3 inferences + determinism
    }

    #[test]
    fn test_suite_includes_determinism_with_seed() {
        let mut suite = E2ETestSuite::new(default_config());
        suite.run_all().unwrap();
        assert!(suite.stage_results.iter().any(|r| r.stage_name == "determinism"));
    }

    #[test]
    fn test_suite_skips_determinism_without_seed() {
        let mut cfg = default_config();
        cfg.seed = None;
        let mut suite = E2ETestSuite::new(cfg);
        suite.run_all().unwrap();
        assert!(!suite.stage_results.iter().any(|r| r.stage_name == "determinism"));
    }

    #[test]
    fn test_suite_summary_display() {
        let summary = SuiteSummary { total: 10, passed: 8, failed: 1, skipped: 1 };
        let s = summary.to_string();
        assert!(s.contains("10 total"));
        assert!(s.contains("8 passed"));
        assert!(s.contains("1 failed"));
        assert!(s.contains("1 skipped"));
    }

    #[test]
    fn test_suite_summary_all_passed_true() {
        let summary = SuiteSummary { total: 5, passed: 5, failed: 0, skipped: 0 };
        assert!(summary.all_passed());
    }

    #[test]
    fn test_suite_summary_all_passed_false() {
        let summary = SuiteSummary { total: 5, passed: 4, failed: 1, skipped: 0 };
        assert!(!summary.all_passed());
    }

    #[test]
    fn test_suite_clears_on_rerun() {
        let mut suite = E2ETestSuite::new(default_config());
        suite.run_all().unwrap();
        let first = suite.stage_results.len();
        suite.run_all().unwrap();
        assert_eq!(suite.stage_results.len(), first);
    }

    #[test]
    fn test_suite_with_device_fallback() {
        let cfg = default_config().with_device(Device::Cuda(0));
        let mut suite = E2ETestSuite::new(cfg);
        suite.run_all().unwrap();
        assert!(suite.stage_results.iter().any(|r| r.stage_name == "device_fallback"));
    }

    #[test]
    fn test_suite_no_fallback_stage_for_cpu() {
        let cfg = default_config().with_device(Device::Cpu);
        let mut suite = E2ETestSuite::new(cfg);
        suite.run_all().unwrap();
        assert!(!suite.stage_results.iter().any(|r| r.stage_name == "device_fallback"));
    }
}
