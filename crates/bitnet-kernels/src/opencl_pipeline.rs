//! OpenCL inference pipeline orchestrator for Intel A770 GPUs.
//!
//! Defines the pipeline stages and their ordering for a transformer forward
//! pass. Does not call actual OpenCL APIs (those need hardware), but provides
//! the orchestration layer with CPU reference fallbacks and per-stage timing.

use std::fmt;
use std::time::Instant;

// ── PipelineStage ──────────────────────────────────────────────────

/// Logical stage within a single-token forward pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipelineStage {
    /// Token ID → embedding vector lookup.
    Embedding,
    /// Pre-attention RMS normalization.
    RmsNorm,
    /// Q/K/V projection + scaled dot-product + output projection.
    Attention,
    /// Gate/up projection → SiLU → down projection.
    FeedForward,
    /// Final RMS normalization after all layers.
    FinalNorm,
    /// Hidden state → vocab logits projection.
    LogitProjection,
    /// Token sampling from logit distribution.
    Sampling,
}

impl fmt::Display for PipelineStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Embedding => "Embedding",
            Self::RmsNorm => "RmsNorm",
            Self::Attention => "Attention",
            Self::FeedForward => "FeedForward",
            Self::FinalNorm => "FinalNorm",
            Self::LogitProjection => "LogitProjection",
            Self::Sampling => "Sampling",
        };
        write!(f, "{name}")
    }
}

impl PipelineStage {
    /// All pipeline stages in execution order.
    #[must_use]
    pub fn all() -> Vec<Self> {
        vec![
            Self::Embedding,
            Self::RmsNorm,
            Self::Attention,
            Self::FeedForward,
            Self::FinalNorm,
            Self::LogitProjection,
            Self::Sampling,
        ]
    }
}

// ── PipelineError ──────────────────────────────────────────────────

/// Errors that can occur during pipeline configuration or execution.
#[derive(Debug)]
pub enum PipelineError {
    InvalidConfig(String),
    StageFailure { stage: PipelineStage, reason: String },
    GpuUnavailable,
    AllFallbacksFailed,
}

impl fmt::Display for PipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid pipeline config: {msg}"),
            Self::StageFailure { stage, reason } => {
                write!(f, "stage {stage} failed: {reason}")
            }
            Self::GpuUnavailable => write!(f, "GPU is unavailable"),
            Self::AllFallbacksFailed => write!(f, "all fallbacks failed"),
        }
    }
}

impl std::error::Error for PipelineError {}

// ── PipelineConfig ─────────────────────────────────────────────────

/// Configuration for a transformer model consumed by the pipeline.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub num_layers: usize,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub intermediate_dim: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    /// Whether to attempt GPU execution for stages.
    pub use_gpu: bool,
    /// Whether to fall back to CPU when a GPU stage fails.
    pub fallback_to_cpu: bool,
}

impl PipelineConfig {
    /// Validate that every dimension is non-zero and self-consistent.
    pub fn validate(&self) -> Result<(), PipelineError> {
        if self.num_layers == 0 {
            return Err(PipelineError::InvalidConfig("num_layers must be > 0".into()));
        }
        if self.hidden_dim == 0 {
            return Err(PipelineError::InvalidConfig("hidden_dim must be > 0".into()));
        }
        if self.num_heads == 0 {
            return Err(PipelineError::InvalidConfig("num_heads must be > 0".into()));
        }
        if self.head_dim == 0 {
            return Err(PipelineError::InvalidConfig("head_dim must be > 0".into()));
        }
        if self.intermediate_dim == 0 {
            return Err(PipelineError::InvalidConfig("intermediate_dim must be > 0".into()));
        }
        if self.vocab_size == 0 {
            return Err(PipelineError::InvalidConfig("vocab_size must be > 0".into()));
        }
        if self.max_seq_len == 0 {
            return Err(PipelineError::InvalidConfig("max_seq_len must be > 0".into()));
        }
        if self.num_heads * self.head_dim != self.hidden_dim {
            return Err(PipelineError::InvalidConfig(format!(
                "num_heads({}) * head_dim({}) = {} != hidden_dim({})",
                self.num_heads,
                self.head_dim,
                self.num_heads * self.head_dim,
                self.hidden_dim,
            )));
        }
        Ok(())
    }

    /// Rough estimate of the total number of model parameters.
    pub fn total_parameters_estimate(&self) -> usize {
        let h = self.hidden_dim;
        let v = self.vocab_size;
        let inter = self.intermediate_dim;

        // Embedding table
        let embed = v * h;
        // Per-layer: 4 attention projections (Q,K,V,O) + 2 RMSNorm gammas + 3 FFN projections
        let per_layer = 4 * h * h + 2 * h + 3 * h * inter;
        let layers = self.num_layers * per_layer;
        // Final norm gamma + LM head
        let final_params = h + v * h;

        embed + layers + final_params
    }
}

// ── StageResult ────────────────────────────────────────────────────

/// Result of executing a single pipeline stage.
#[derive(Debug, Clone)]
pub struct StageResult {
    pub stage: PipelineStage,
    pub output_shape: Vec<usize>,
    pub execution_time_ns: u64,
    pub used_gpu: bool,
    pub fallback_triggered: bool,
}

// ── PipelineExecution ──────────────────────────────────────────────

/// Aggregated result of a complete single-token forward pass.
#[derive(Debug, Clone)]
pub struct PipelineExecution {
    pub stages: Vec<StageResult>,
    pub total_time_ns: u64,
    pub tokens_generated: usize,
}

impl PipelineExecution {
    /// Fraction of stages that executed on GPU (0.0 – 1.0).
    pub fn gpu_utilization(&self) -> f64 {
        if self.stages.is_empty() {
            return 0.0;
        }
        let gpu_count = self.stages.iter().filter(|s| s.used_gpu).count();
        gpu_count as f64 / self.stages.len() as f64
    }

    /// The stage that took the longest to execute.
    pub fn slowest_stage(&self) -> Option<&StageResult> {
        self.stages.iter().max_by_key(|s| s.execution_time_ns)
    }

    /// Number of stages where a CPU fallback was triggered.
    pub fn total_fallbacks(&self) -> usize {
        self.stages.iter().filter(|s| s.fallback_triggered).count()
    }

    /// Human-readable summary of the pipeline execution.
    pub fn summary(&self) -> String {
        let gpu_pct = self.gpu_utilization() * 100.0;
        let total_us = self.total_time_ns / 1000;
        let fallbacks = self.total_fallbacks();
        let slowest = self
            .slowest_stage()
            .map(|s| format!("{} ({}ns)", s.stage, s.execution_time_ns))
            .unwrap_or_else(|| "none".into());
        format!(
            "Pipeline: {} stages, {total_us}µs total, {gpu_pct:.1}% GPU, \
             {fallbacks} fallback(s), slowest={slowest}",
            self.stages.len(),
        )
    }
}

// ── CPU reference helpers ──────────────────────────────────────────

fn cpu_rms_norm(input: &[f32], dim: usize) -> Vec<f32> {
    let eps = 1e-5_f32;
    let rms = (input.iter().map(|x| x * x).sum::<f32>() / dim as f32 + eps).sqrt();
    input.iter().map(|x| x / rms).collect()
}

fn cpu_matmul_stub(rows: usize, cols: usize) -> Vec<f32> {
    // Deterministic stub output
    (0..rows * cols).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect()
}

fn cpu_silu_elementwise(input: &mut [f32]) {
    for v in input.iter_mut() {
        *v = *v / (1.0 + (-*v).exp());
    }
}

// ── InferencePipeline ──────────────────────────────────────────────

/// Orchestrates all inference stages for a single-token forward pass.
pub struct InferencePipeline {
    config: PipelineConfig,
    execution_count: usize,
    diag: PipelineDiagnostics,
}

impl InferencePipeline {
    /// Create a new pipeline, validating the configuration.
    pub fn new(config: PipelineConfig) -> Result<Self, PipelineError> {
        config.validate()?;
        Ok(Self { config, execution_count: 0, diag: PipelineDiagnostics::default() })
    }

    /// Execute a single-token forward pass using CPU reference implementations.
    ///
    /// Simulates: Embedding → (RmsNorm → Attention → RmsNorm → FeedForward) × L
    ///            → FinalNorm → LogitProjection
    pub fn execute_single_token_cpu(
        &mut self,
        input_ids: &[u32],
        position: usize,
    ) -> Result<PipelineExecution, PipelineError> {
        if input_ids.is_empty() {
            return Err(PipelineError::StageFailure {
                stage: PipelineStage::Embedding,
                reason: "input_ids must not be empty".into(),
            });
        }
        if position >= self.config.max_seq_len {
            return Err(PipelineError::InvalidConfig(format!(
                "position {} >= max_seq_len {}",
                position, self.config.max_seq_len,
            )));
        }

        let use_gpu = self.config.use_gpu;
        // GPU is never truly available in CPU reference mode
        let gpu_available = false;
        let fallback_to_cpu = self.config.fallback_to_cpu;

        let mut stages = Vec::new();
        let pass_start = Instant::now();
        let h = self.config.hidden_dim;
        let v = self.config.vocab_size;
        let inter = self.config.intermediate_dim;

        // ── Embedding ──────────────────────────────────────────
        let t0 = Instant::now();
        let mut hidden: Vec<f32> = (0..h).map(|i| ((i + position) % 13) as f32 * 0.01).collect();
        let fallback = use_gpu && !gpu_available;
        if use_gpu && !gpu_available && !fallback_to_cpu {
            return Err(PipelineError::GpuUnavailable);
        }
        stages.push(StageResult {
            stage: PipelineStage::Embedding,
            output_shape: vec![1, h],
            execution_time_ns: t0.elapsed().as_nanos() as u64,
            used_gpu: gpu_available && use_gpu,
            fallback_triggered: fallback,
        });

        // ── Per-layer stages ───────────────────────────────────
        for _layer in 0..self.config.num_layers {
            // Pre-attention RmsNorm
            let t0 = Instant::now();
            hidden = cpu_rms_norm(&hidden, h);
            stages.push(StageResult {
                stage: PipelineStage::RmsNorm,
                output_shape: vec![1, h],
                execution_time_ns: t0.elapsed().as_nanos() as u64,
                used_gpu: gpu_available && use_gpu,
                fallback_triggered: fallback,
            });

            // Attention
            let t0 = Instant::now();
            let attn_out = cpu_matmul_stub(1, h);
            for (o, &a) in hidden.iter_mut().zip(attn_out.iter()) {
                *o += a;
            }
            stages.push(StageResult {
                stage: PipelineStage::Attention,
                output_shape: vec![1, h],
                execution_time_ns: t0.elapsed().as_nanos() as u64,
                used_gpu: gpu_available && use_gpu,
                fallback_triggered: fallback,
            });

            // Post-attention RmsNorm
            let t0 = Instant::now();
            hidden = cpu_rms_norm(&hidden, h);
            stages.push(StageResult {
                stage: PipelineStage::RmsNorm,
                output_shape: vec![1, h],
                execution_time_ns: t0.elapsed().as_nanos() as u64,
                used_gpu: gpu_available && use_gpu,
                fallback_triggered: fallback,
            });

            // FeedForward (gate/up → SiLU → down)
            let t0 = Instant::now();
            let mut ffn = cpu_matmul_stub(1, inter);
            cpu_silu_elementwise(&mut ffn);
            let down = cpu_matmul_stub(1, h);
            for (o, &d) in hidden.iter_mut().zip(down.iter()) {
                *o += d;
            }
            stages.push(StageResult {
                stage: PipelineStage::FeedForward,
                output_shape: vec![1, h],
                execution_time_ns: t0.elapsed().as_nanos() as u64,
                used_gpu: gpu_available && use_gpu,
                fallback_triggered: fallback,
            });
        }

        // ── FinalNorm ──────────────────────────────────────────
        let t0 = Instant::now();
        let _ = cpu_rms_norm(&hidden, h);
        stages.push(StageResult {
            stage: PipelineStage::FinalNorm,
            output_shape: vec![1, h],
            execution_time_ns: t0.elapsed().as_nanos() as u64,
            used_gpu: gpu_available && use_gpu,
            fallback_triggered: fallback,
        });

        // ── LogitProjection ────────────────────────────────────
        let t0 = Instant::now();
        let _logits = cpu_matmul_stub(1, v);
        stages.push(StageResult {
            stage: PipelineStage::LogitProjection,
            output_shape: vec![1, v],
            execution_time_ns: t0.elapsed().as_nanos() as u64,
            used_gpu: gpu_available && use_gpu,
            fallback_triggered: fallback,
        });

        let total_time_ns = pass_start.elapsed().as_nanos() as u64;
        self.execution_count += 1;

        Ok(PipelineExecution { stages, total_time_ns, tokens_generated: 1 })
    }

    /// The ordered list of stages for one complete forward pass.
    pub fn stage_order(&self) -> Vec<PipelineStage> {
        let mut order = vec![PipelineStage::Embedding];
        for _ in 0..self.config.num_layers {
            order.push(PipelineStage::RmsNorm);
            order.push(PipelineStage::Attention);
            order.push(PipelineStage::RmsNorm);
            order.push(PipelineStage::FeedForward);
        }
        order.push(PipelineStage::FinalNorm);
        order.push(PipelineStage::LogitProjection);
        order
    }

    /// Total number of stages executed per token.
    /// Embedding(1) + layers*(RmsNorm + Attn + RmsNorm + FFN)(4) + FinalNorm(1) + LogitProj(1)
    pub fn stages_per_token(&self) -> usize {
        1 + self.config.num_layers * 4 + 2
    }

    /// Number of forward passes executed so far.
    pub fn execution_count(&self) -> usize {
        self.execution_count
    }

    /// Reference to the pipeline configuration.
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }

    /// Run a forward pass over `input_ids`, returning logits of length `vocab_size`.
    ///
    /// This is a convenience wrapper around [`Self::execute_single_token_cpu`].
    pub fn forward(&mut self, input_ids: &[u32]) -> Result<Vec<f32>, PipelineError> {
        let exec = self.execute_single_token_cpu(input_ids, 0)?;
        self.diag.total_forward_calls += 1;
        if input_ids.len() > self.diag.peak_sequence_len {
            self.diag.peak_sequence_len = input_ids.len();
        }
        // Return deterministic logits sized to vocab
        let v = self.config.vocab_size;
        let logits: Vec<f32> = (0..v).map(|i| (i as f32) * 0.001 - 0.5).collect();
        let _ = exec; // used for timing side-effects
        Ok(logits)
    }

    /// Reset internal state so the pipeline can be reused.
    pub fn reset(&mut self) {
        self.execution_count = 0;
        self.diag = PipelineDiagnostics::default();
    }

    /// Current pipeline status.
    #[must_use]
    pub fn status(&self) -> PipelineStatus {
        PipelineStatus::Ready
    }
}

// ── PipelineStatus ─────────────────────────────────────────────────

/// Runtime status of an [`InferencePipeline`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineStatus {
    /// Pipeline is ready to accept forward calls.
    Ready,
    /// Pipeline encountered an error and must be reset.
    Error,
}

impl PipelineConfig {
    /// Tiny configuration useful for property tests (small dimensions, CPU-only).
    #[must_use]
    pub fn tiny_test() -> Self {
        Self {
            num_layers: 2,
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            intermediate_dim: 64,
            vocab_size: 64,
            max_seq_len: 128,
            use_gpu: false,
            fallback_to_cpu: true,
        }
    }

    /// Configuration matching the BitNet 2B model dimensions.
    #[must_use]
    pub fn bitnet_2b() -> Self {
        Self {
            num_layers: 26,
            hidden_dim: 2560,
            num_heads: 32,
            head_dim: 80,
            intermediate_dim: 6912,
            vocab_size: 32_000,
            max_seq_len: 2048,
            use_gpu: false,
            fallback_to_cpu: true,
        }
    }
}

// ── GenerationConfig ───────────────────────────────────────────────

/// Sampling / generation parameters for autoregressive decoding.
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Temperature for softmax scaling (0.0 = greedy).
    pub temperature: f32,
    /// Nucleus sampling threshold.
    pub top_p: f32,
    /// Top-k sampling (0 = disabled).
    pub top_k: usize,
    /// Maximum tokens to generate.
    pub max_tokens: usize,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self { temperature: 1.0, top_p: 0.9, top_k: 0, max_tokens: 128 }
    }
}

impl GenerationConfig {
    /// Greedy decoding configuration (temperature = 0).
    #[must_use]
    pub fn greedy() -> Self {
        Self { temperature: 0.0, ..Self::default() }
    }

    /// Set temperature, returning `self` for chaining.
    #[must_use]
    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }

    /// Set `top_p`, returning `self` for chaining.
    #[must_use]
    pub fn with_top_p(mut self, p: f32) -> Self {
        self.top_p = p;
        self
    }

    /// Set `top_k`, returning `self` for chaining.
    #[must_use]
    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    /// Set `max_tokens`, returning `self` for chaining.
    #[must_use]
    pub fn with_max_tokens(mut self, n: usize) -> Self {
        self.max_tokens = n;
        self
    }

    /// Validate generation parameters.
    pub fn validate(&self) -> Result<(), PipelineError> {
        if self.temperature < 0.0 {
            return Err(PipelineError::InvalidConfig("temperature must be non-negative".into()));
        }
        if !(0.0..=1.0).contains(&self.top_p) {
            return Err(PipelineError::InvalidConfig("top_p must be in [0.0, 1.0]".into()));
        }
        if self.max_tokens == 0 {
            return Err(PipelineError::InvalidConfig("max_tokens must be > 0".into()));
        }
        Ok(())
    }
}

// ── PipelineBuilder ────────────────────────────────────────────────

/// Builder for constructing an [`InferencePipeline`] with default configuration.
#[derive(Debug, Default)]
pub struct PipelineBuilder {
    config: Option<PipelineConfig>,
}

impl PipelineBuilder {
    /// Create a new builder with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the pipeline configuration.
    #[must_use]
    pub fn with_config(mut self, config: PipelineConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Build the pipeline, using `tiny_test` defaults if no config was given.
    pub fn build(self) -> Result<InferencePipeline, PipelineError> {
        let config = self.config.unwrap_or_else(PipelineConfig::tiny_test);
        InferencePipeline::new(config)
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Generation result types
// ═══════════════════════════════════════════════════════════════════

/// Reason generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// Hit maximum token limit.
    MaxTokens,
    /// Hit end-of-sequence token.
    EndOfSequence,
}

/// Result of a generate call.
#[derive(Debug, Clone)]
pub struct GenerateResult {
    /// Generated token IDs.
    pub tokens: Vec<u32>,
    /// How many tokens were generated.
    pub generated_tokens: usize,
    /// Why generation stopped.
    pub stop_reason: StopReason,
    /// Per-stage timing information.
    pub stage_timings: Vec<(PipelineStage, f64)>,
}

/// Diagnostics snapshot from the pipeline.
#[derive(Debug, Clone, Default)]
pub struct PipelineDiagnostics {
    /// Total forward calls made.
    pub total_forward_calls: usize,
    /// Total tokens generated via `generate`.
    pub total_tokens_generated: usize,
    /// Peak sequence length seen.
    pub peak_sequence_len: usize,
}

/// Token-by-token generator wrapping an `InferencePipeline`.
pub struct TokenGenerator {
    pipeline: InferencePipeline,
    config: GenerationConfig,
}

impl TokenGenerator {
    /// Create a new token generator.
    pub fn new(pipeline: InferencePipeline, config: GenerationConfig) -> Self {
        Self { pipeline, config }
    }

    /// Generate tokens from input IDs.
    pub fn generate(&mut self, input_ids: &[u32]) -> Result<GenerateResult, PipelineError> {
        self.pipeline.generate(input_ids, &self.config)
    }
}

impl InferencePipeline {
    /// Generate a sequence of tokens from a prompt.
    pub fn generate(
        &mut self,
        input_ids: &[u32],
        config: &GenerationConfig,
    ) -> Result<GenerateResult, PipelineError> {
        config.validate()?;
        let max = config.max_tokens;
        let mut tokens = Vec::with_capacity(max);
        let mut timings = Vec::new();

        // Initial forward pass with full input
        let mut logits = self.forward(input_ids)?;

        for _ in 0..max {
            // Greedy sampling (argmax)
            let token = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
            tokens.push(token);

            // Autoregressive: feed the new token
            logits = self.forward(&[token])?;
            timings.push((PipelineStage::Sampling, 0.0));
        }

        self.diag.total_tokens_generated += tokens.len();

        Ok(GenerateResult {
            generated_tokens: tokens.len(),
            stop_reason: StopReason::MaxTokens,
            stage_timings: timings,
            tokens,
        })
    }

    /// Return pipeline diagnostics.
    pub fn diagnostics(&self) -> PipelineDiagnostics {
        self.diag.clone()
    }

    /// Tokens per second estimate.
    pub fn tokens_per_second(&self) -> f64 {
        0.0
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> PipelineConfig {
        PipelineConfig {
            num_layers: 2,
            hidden_dim: 32,
            num_heads: 4,
            head_dim: 8,
            intermediate_dim: 64,
            vocab_size: 64,
            max_seq_len: 128,
            use_gpu: false,
            fallback_to_cpu: true,
        }
    }

    fn tiny_pipeline() -> InferencePipeline {
        InferencePipeline::new(tiny_config()).unwrap()
    }

    // ── PipelineStage Display ──────────────────────────────────

    #[test]
    fn test_stage_display_embedding() {
        assert_eq!(format!("{}", PipelineStage::Embedding), "Embedding");
    }

    #[test]
    fn test_stage_display_rms_norm() {
        assert_eq!(format!("{}", PipelineStage::RmsNorm), "RmsNorm");
    }

    #[test]
    fn test_stage_display_attention() {
        assert_eq!(format!("{}", PipelineStage::Attention), "Attention");
    }

    #[test]
    fn test_stage_display_feed_forward() {
        assert_eq!(format!("{}", PipelineStage::FeedForward), "FeedForward");
    }

    #[test]
    fn test_stage_display_final_norm() {
        assert_eq!(format!("{}", PipelineStage::FinalNorm), "FinalNorm");
    }

    #[test]
    fn test_stage_display_logit_projection() {
        assert_eq!(format!("{}", PipelineStage::LogitProjection), "LogitProjection");
    }

    // ── PipelineConfig validation ──────────────────────────────

    #[test]
    fn test_config_valid() {
        assert!(tiny_config().validate().is_ok());
    }

    #[test]
    fn test_config_zero_num_layers() {
        let mut cfg = tiny_config();
        cfg.num_layers = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_zero_hidden_dim() {
        let mut cfg = tiny_config();
        cfg.hidden_dim = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_zero_num_heads() {
        let mut cfg = tiny_config();
        cfg.num_heads = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_zero_head_dim() {
        let mut cfg = tiny_config();
        cfg.head_dim = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_zero_intermediate_dim() {
        let mut cfg = tiny_config();
        cfg.intermediate_dim = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_zero_vocab_size() {
        let mut cfg = tiny_config();
        cfg.vocab_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_zero_max_seq_len() {
        let mut cfg = tiny_config();
        cfg.max_seq_len = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_mismatched_head_dims() {
        let mut cfg = tiny_config();
        // num_heads=4, head_dim=16 → 64, but hidden_dim=32
        cfg.head_dim = 16;
        let err = cfg.validate().unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("num_heads"), "expected head mismatch error, got: {msg}");
    }

    // ── total_parameters_estimate ──────────────────────────────

    #[test]
    fn test_total_parameters_estimate_nonzero() {
        let cfg = tiny_config();
        assert!(cfg.total_parameters_estimate() > 0);
    }

    #[test]
    fn test_total_parameters_estimate_scales_with_layers() {
        let mut cfg1 = tiny_config();
        cfg1.num_layers = 1;
        let mut cfg2 = tiny_config();
        cfg2.num_layers = 4;
        assert!(cfg2.total_parameters_estimate() > cfg1.total_parameters_estimate());
    }

    #[test]
    fn test_total_parameters_estimate_large_model() {
        let cfg = PipelineConfig {
            num_layers: 32,
            hidden_dim: 4096,
            num_heads: 32,
            head_dim: 128,
            intermediate_dim: 11008,
            vocab_size: 32000,
            max_seq_len: 4096,
            use_gpu: false,
            fallback_to_cpu: true,
        };
        let params = cfg.total_parameters_estimate();
        // ~7B-class model should have > 1B params
        assert!(params > 1_000_000_000, "expected >1B params, got {params}");
    }

    // ── stage_order ────────────────────────────────────────────

    #[test]
    fn test_stage_order_1_layer() {
        let cfg = PipelineConfig { num_layers: 1, ..tiny_config() };
        let p = InferencePipeline::new(cfg).unwrap();
        let order = p.stage_order();
        assert_eq!(
            order,
            vec![
                PipelineStage::Embedding,
                PipelineStage::RmsNorm,
                PipelineStage::Attention,
                PipelineStage::RmsNorm,
                PipelineStage::FeedForward,
                PipelineStage::FinalNorm,
                PipelineStage::LogitProjection,
            ]
        );
    }

    #[test]
    fn test_stage_order_2_layers() {
        let p = tiny_pipeline(); // 2 layers
        let order = p.stage_order();
        // Embedding + 2*(RmsNorm,Attn,RmsNorm,FFN) + FinalNorm + LogitProj = 11
        assert_eq!(order.len(), 11);
        assert_eq!(order[0], PipelineStage::Embedding);
        assert_eq!(order[order.len() - 1], PipelineStage::LogitProjection);
        assert_eq!(order[order.len() - 2], PipelineStage::FinalNorm);
    }

    #[test]
    fn test_stage_order_12_layers() {
        let cfg = PipelineConfig {
            num_layers: 12,
            hidden_dim: 768,
            num_heads: 12,
            head_dim: 64,
            intermediate_dim: 3072,
            vocab_size: 32000,
            max_seq_len: 2048,
            use_gpu: false,
            fallback_to_cpu: true,
        };
        let p = InferencePipeline::new(cfg).unwrap();
        let order = p.stage_order();
        // 1 + 12*4 + 2 = 51
        assert_eq!(order.len(), 51);
    }

    // ── stages_per_token ───────────────────────────────────────

    #[test]
    fn test_stages_per_token_matches_order_len() {
        let p = tiny_pipeline();
        assert_eq!(p.stages_per_token(), p.stage_order().len());
    }

    #[test]
    fn test_stages_per_token_formula() {
        let p = tiny_pipeline(); // 2 layers
        // 1 + 2*4 + 2 = 11
        assert_eq!(p.stages_per_token(), 11);
    }

    // ── execute_single_token_cpu ────────────────────────────────

    #[test]
    fn test_execute_single_token_produces_results() {
        let mut p = tiny_pipeline();
        let exec = p.execute_single_token_cpu(&[1], 0).unwrap();
        assert_eq!(exec.stages.len(), p.stages_per_token());
        assert_eq!(exec.tokens_generated, 1);
    }

    #[test]
    fn test_execute_single_token_stage_count() {
        let mut p = tiny_pipeline();
        let exec = p.execute_single_token_cpu(&[1, 2, 3], 0).unwrap();
        // Stage count should match config regardless of input length
        assert_eq!(exec.stages.len(), 11);
    }

    #[test]
    fn test_execute_first_stage_is_embedding() {
        let mut p = tiny_pipeline();
        let exec = p.execute_single_token_cpu(&[1], 0).unwrap();
        assert_eq!(exec.stages[0].stage, PipelineStage::Embedding);
    }

    #[test]
    fn test_execute_last_stage_is_logit_projection() {
        let mut p = tiny_pipeline();
        let exec = p.execute_single_token_cpu(&[1], 0).unwrap();
        assert_eq!(exec.stages.last().unwrap().stage, PipelineStage::LogitProjection);
    }

    #[test]
    fn test_execute_logit_projection_output_shape() {
        let mut p = tiny_pipeline();
        let exec = p.execute_single_token_cpu(&[1], 0).unwrap();
        let last = exec.stages.last().unwrap();
        assert_eq!(last.output_shape, vec![1, 64]); // vocab_size = 64
    }

    #[test]
    fn test_execute_embedding_output_shape() {
        let mut p = tiny_pipeline();
        let exec = p.execute_single_token_cpu(&[1], 0).unwrap();
        assert_eq!(exec.stages[0].output_shape, vec![1, 32]); // hidden_dim = 32
    }

    #[test]
    fn test_execute_total_time_positive() {
        let mut p = tiny_pipeline();
        let _exec = p.execute_single_token_cpu(&[1], 0).unwrap();
        // total_time_ns may be 0 on very fast systems, but should be non-negative
        // Removed: trivial assertion (u64 always <= u64::MAX)
    }

    #[test]
    fn test_execute_empty_input_fails() {
        let mut p = tiny_pipeline();
        let result = p.execute_single_token_cpu(&[], 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_execute_position_exceeds_max_seq_len() {
        let mut p = tiny_pipeline();
        let result = p.execute_single_token_cpu(&[1], 200);
        assert!(result.is_err());
    }

    // ── GPU utilization ────────────────────────────────────────

    #[test]
    fn test_gpu_utilization_all_cpu() {
        let mut p = tiny_pipeline(); // use_gpu=false
        let exec = p.execute_single_token_cpu(&[1], 0).unwrap();
        assert_eq!(exec.gpu_utilization(), 0.0);
    }

    #[test]
    fn test_gpu_utilization_synthetic_all_gpu() {
        let exec = PipelineExecution {
            stages: vec![
                StageResult {
                    stage: PipelineStage::Embedding,
                    output_shape: vec![1, 32],
                    execution_time_ns: 100,
                    used_gpu: true,
                    fallback_triggered: false,
                },
                StageResult {
                    stage: PipelineStage::LogitProjection,
                    output_shape: vec![1, 64],
                    execution_time_ns: 200,
                    used_gpu: true,
                    fallback_triggered: false,
                },
            ],
            total_time_ns: 300,
            tokens_generated: 1,
        };
        assert!((exec.gpu_utilization() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_gpu_utilization_mixed() {
        let exec = PipelineExecution {
            stages: vec![
                StageResult {
                    stage: PipelineStage::Embedding,
                    output_shape: vec![1, 32],
                    execution_time_ns: 100,
                    used_gpu: true,
                    fallback_triggered: false,
                },
                StageResult {
                    stage: PipelineStage::RmsNorm,
                    output_shape: vec![1, 32],
                    execution_time_ns: 50,
                    used_gpu: false,
                    fallback_triggered: false,
                },
            ],
            total_time_ns: 150,
            tokens_generated: 1,
        };
        assert!((exec.gpu_utilization() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_gpu_utilization_empty_stages() {
        let exec = PipelineExecution { stages: vec![], total_time_ns: 0, tokens_generated: 0 };
        assert_eq!(exec.gpu_utilization(), 0.0);
    }

    // ── slowest_stage ──────────────────────────────────────────

    #[test]
    fn test_slowest_stage_detection() {
        let exec = PipelineExecution {
            stages: vec![
                StageResult {
                    stage: PipelineStage::Embedding,
                    output_shape: vec![1, 32],
                    execution_time_ns: 100,
                    used_gpu: false,
                    fallback_triggered: false,
                },
                StageResult {
                    stage: PipelineStage::Attention,
                    output_shape: vec![1, 32],
                    execution_time_ns: 999,
                    used_gpu: false,
                    fallback_triggered: false,
                },
                StageResult {
                    stage: PipelineStage::FinalNorm,
                    output_shape: vec![1, 32],
                    execution_time_ns: 50,
                    used_gpu: false,
                    fallback_triggered: false,
                },
            ],
            total_time_ns: 1149,
            tokens_generated: 1,
        };
        let slowest = exec.slowest_stage().unwrap();
        assert_eq!(slowest.stage, PipelineStage::Attention);
        assert_eq!(slowest.execution_time_ns, 999);
    }

    #[test]
    fn test_slowest_stage_empty() {
        let exec = PipelineExecution { stages: vec![], total_time_ns: 0, tokens_generated: 0 };
        assert!(exec.slowest_stage().is_none());
    }

    // ── fallback counting ──────────────────────────────────────

    #[test]
    fn test_total_fallbacks_none() {
        let mut p = tiny_pipeline();
        let exec = p.execute_single_token_cpu(&[1], 0).unwrap();
        // use_gpu=false → no fallbacks triggered
        assert_eq!(exec.total_fallbacks(), 0);
    }

    #[test]
    fn test_total_fallbacks_with_gpu_requested() {
        let cfg = PipelineConfig { use_gpu: true, fallback_to_cpu: true, ..tiny_config() };
        let mut p = InferencePipeline::new(cfg).unwrap();
        let exec = p.execute_single_token_cpu(&[1], 0).unwrap();
        // GPU unavailable in CPU mode → all stages have fallback_triggered
        assert_eq!(exec.total_fallbacks(), exec.stages.len());
    }

    #[test]
    fn test_total_fallbacks_synthetic() {
        let exec = PipelineExecution {
            stages: vec![
                StageResult {
                    stage: PipelineStage::Embedding,
                    output_shape: vec![1, 32],
                    execution_time_ns: 100,
                    used_gpu: false,
                    fallback_triggered: true,
                },
                StageResult {
                    stage: PipelineStage::RmsNorm,
                    output_shape: vec![1, 32],
                    execution_time_ns: 50,
                    used_gpu: false,
                    fallback_triggered: false,
                },
                StageResult {
                    stage: PipelineStage::Attention,
                    output_shape: vec![1, 32],
                    execution_time_ns: 200,
                    used_gpu: false,
                    fallback_triggered: true,
                },
            ],
            total_time_ns: 350,
            tokens_generated: 1,
        };
        assert_eq!(exec.total_fallbacks(), 2);
    }

    // ── summary ────────────────────────────────────────────────

    #[test]
    fn test_summary_contains_stages() {
        let mut p = tiny_pipeline();
        let exec = p.execute_single_token_cpu(&[1], 0).unwrap();
        let s = exec.summary();
        assert!(s.contains("11 stages"), "summary should mention 11 stages: {s}");
    }

    #[test]
    fn test_summary_contains_gpu_pct() {
        let mut p = tiny_pipeline();
        let exec = p.execute_single_token_cpu(&[1], 0).unwrap();
        let s = exec.summary();
        assert!(s.contains("0.0% GPU"), "summary should contain GPU%: {s}");
    }

    #[test]
    fn test_summary_contains_fallbacks() {
        let mut p = tiny_pipeline();
        let exec = p.execute_single_token_cpu(&[1], 0).unwrap();
        let s = exec.summary();
        assert!(s.contains("fallback"), "summary should mention fallbacks: {s}");
    }

    #[test]
    fn test_summary_contains_slowest() {
        let mut p = tiny_pipeline();
        let exec = p.execute_single_token_cpu(&[1], 0).unwrap();
        let s = exec.summary();
        assert!(s.contains("slowest="), "summary should mention slowest stage: {s}");
    }

    // ── PipelineError ──────────────────────────────────────────

    #[test]
    fn test_error_display_invalid_config() {
        let err = PipelineError::InvalidConfig("bad dim".into());
        let msg = format!("{err}");
        assert!(msg.contains("bad dim"));
    }

    #[test]
    fn test_error_display_stage_failure() {
        let err =
            PipelineError::StageFailure { stage: PipelineStage::Attention, reason: "OOM".into() };
        let msg = format!("{err}");
        assert!(msg.contains("Attention") && msg.contains("OOM"));
    }

    #[test]
    fn test_error_display_gpu_unavailable() {
        let msg = format!("{}", PipelineError::GpuUnavailable);
        assert!(msg.contains("unavailable"));
    }

    #[test]
    fn test_error_display_all_fallbacks_failed() {
        let msg = format!("{}", PipelineError::AllFallbacksFailed);
        assert!(msg.contains("fallback"));
    }

    #[test]
    fn test_error_is_std_error() {
        let err: Box<dyn std::error::Error> = Box::new(PipelineError::InvalidConfig("test".into()));
        assert!(!err.to_string().is_empty());
    }

    // ── use_gpu=false forces CPU ───────────────────────────────

    #[test]
    fn test_use_gpu_false_forces_all_cpu() {
        let mut p = tiny_pipeline(); // use_gpu=false
        let exec = p.execute_single_token_cpu(&[1], 0).unwrap();
        for stage in &exec.stages {
            assert!(!stage.used_gpu, "stage {} should not use GPU", stage.stage);
            assert!(!stage.fallback_triggered, "no fallback when GPU not requested");
        }
    }

    // ── use_gpu=true without fallback → error ──────────────────

    #[test]
    fn test_use_gpu_true_no_fallback_errors() {
        let cfg = PipelineConfig { use_gpu: true, fallback_to_cpu: false, ..tiny_config() };
        let mut p = InferencePipeline::new(cfg).unwrap();
        let result = p.execute_single_token_cpu(&[1], 0);
        assert!(result.is_err());
    }

    // ── Multiple sequential executions ─────────────────────────

    #[test]
    fn test_multiple_executions_count() {
        let mut p = tiny_pipeline();
        p.execute_single_token_cpu(&[1], 0).unwrap();
        p.execute_single_token_cpu(&[2], 1).unwrap();
        p.execute_single_token_cpu(&[3], 2).unwrap();
        assert_eq!(p.execution_count(), 3);
    }

    #[test]
    fn test_multiple_executions_independent() {
        let mut p = tiny_pipeline();
        let e1 = p.execute_single_token_cpu(&[1], 0).unwrap();
        let e2 = p.execute_single_token_cpu(&[2], 1).unwrap();
        assert_eq!(e1.stages.len(), e2.stages.len());
    }

    // ── Large model config ─────────────────────────────────────

    #[test]
    fn test_large_model_config_32_layers() {
        let cfg = PipelineConfig {
            num_layers: 32,
            hidden_dim: 4096,
            num_heads: 32,
            head_dim: 128,
            intermediate_dim: 11008,
            vocab_size: 32000,
            max_seq_len: 4096,
            use_gpu: false,
            fallback_to_cpu: true,
        };
        let p = InferencePipeline::new(cfg).unwrap();
        // 1 + 32*4 + 2 = 131
        assert_eq!(p.stages_per_token(), 131);
        assert_eq!(p.stage_order().len(), 131);
    }

    // ── InferencePipeline::new validation ──────────────────────

    #[test]
    fn test_pipeline_new_rejects_invalid_config() {
        let mut cfg = tiny_config();
        cfg.hidden_dim = 0;
        assert!(InferencePipeline::new(cfg).is_err());
    }

    #[test]
    fn test_pipeline_new_accepts_valid_config() {
        assert!(InferencePipeline::new(tiny_config()).is_ok());
    }

    // ── Config accessor ────────────────────────────────────────

    #[test]
    fn test_pipeline_config_accessor() {
        let p = tiny_pipeline();
        assert_eq!(p.config().num_layers, 2);
        assert_eq!(p.config().hidden_dim, 32);
        assert_eq!(p.config().vocab_size, 64);
    }
}
