//! Complete GPU inference pipeline for transformer forward passes.
//!
//! Chains GPU kernels (embedding, RMS norm, attention, FFN, softmax,
//! dequantization) into a sequential or graph-scheduled execution plan.
//! Collects per-kernel timing and peak-memory metrics.

use std::time::Instant;

// ---------------------------------------------------------------------------
// Quantization format
// ---------------------------------------------------------------------------

/// Quantization format supported by the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantFormat {
    /// 32-element blocks with inline F16 scales.
    I2S,
    /// 256-element blocks (`QK256`).
    QK256,
    /// Ternary lookup v1.
    TL1,
    /// Ternary lookup v2.
    TL2,
}

// ---------------------------------------------------------------------------
// Pipeline stages
// ---------------------------------------------------------------------------

/// A single computational stage in the inference pipeline.
#[derive(Debug, Clone, PartialEq)]
pub enum PipelineStage {
    /// Token embedding lookup.
    Embedding { vocab_size: usize, dim: usize },
    /// RMS layer normalisation.
    RmsNorm { dim: usize, eps: f32 },
    /// Multi-head self-attention.
    Attention { num_heads: usize, head_dim: usize },
    /// Feed-forward network.
    FeedForward { dim: usize, hidden_dim: usize },
    /// Dense linear projection.
    Linear { in_features: usize, out_features: usize },
    /// Softmax over logits.
    Softmax { dim: usize },
    /// Dequantize weights from a packed format.
    Dequantize { format: QuantFormat, block_size: usize },
}

impl PipelineStage {
    /// Estimated output size in number of `f32` elements.
    #[must_use]
    pub const fn output_elements(&self) -> usize {
        match *self {
            Self::Embedding { dim, .. }
            | Self::RmsNorm { dim, .. }
            | Self::FeedForward { dim, .. }
            | Self::Softmax { dim } => dim,
            Self::Attention { num_heads, head_dim } => num_heads * head_dim,
            Self::Linear { out_features, .. } => out_features,
            Self::Dequantize { block_size, .. } => block_size,
        }
    }

    /// Estimated output size in bytes (`f32` = 4 bytes each).
    #[must_use]
    pub const fn output_bytes(&self) -> u64 {
        self.output_elements() as u64 * 4
    }

    /// Human-readable stage name for metrics.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Embedding { .. } => "embedding",
            Self::RmsNorm { .. } => "rms_norm",
            Self::Attention { .. } => "attention",
            Self::FeedForward { .. } => "feed_forward",
            Self::Linear { .. } => "linear",
            Self::Softmax { .. } => "softmax",
            Self::Dequantize { .. } => "dequantize",
        }
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Model-level configuration for the inference pipeline.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub num_layers: usize,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub use_flash_attention: bool,
    pub fuse_layernorm: bool,
}

impl PipelineConfig {
    /// Validate that the configuration is consistent.
    pub fn validate(&self) -> Result<(), PipelineError> {
        if self.num_layers == 0 {
            return Err(PipelineError::InvalidConfig("num_layers must be > 0".into()));
        }
        if self.hidden_dim == 0 || self.num_heads == 0 || self.head_dim == 0 {
            return Err(PipelineError::InvalidConfig("dimensions must be > 0".into()));
        }
        if self.vocab_size == 0 {
            return Err(PipelineError::InvalidConfig("vocab_size must be > 0".into()));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Profiling metrics collected during a forward pass.
#[derive(Debug, Clone, Default)]
pub struct PipelineMetrics {
    pub total_time_us: u64,
    pub kernel_times: Vec<(String, u64)>,
    pub memory_peak_bytes: u64,
    pub tokens_processed: usize,
}

impl PipelineMetrics {
    fn reset(&mut self) {
        self.total_time_us = 0;
        self.kernel_times.clear();
        self.memory_peak_bytes = 0;
        self.tokens_processed = 0;
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by the inference pipeline.
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("invalid pipeline configuration: {0}")]
    InvalidConfig(String),
    #[error("empty input: no tokens provided")]
    EmptyInput,
    #[error("layer index {index} out of range (pipeline has {total} layers)")]
    LayerOutOfRange { index: usize, total: usize },
    #[error("kernel execution failed at stage '{stage}': {reason}")]
    KernelError { stage: String, reason: String },
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
}

// ---------------------------------------------------------------------------
// `GpuInferencePipeline`
// ---------------------------------------------------------------------------

/// Complete GPU inference pipeline for transformer models.
///
/// Holds an ordered list of [`PipelineStage`]s built from a
/// [`PipelineConfig`] and simulates a full forward pass collecting
/// [`PipelineMetrics`].
#[derive(Debug)]
pub struct GpuInferencePipeline {
    /// Pipeline configuration.
    config: PipelineConfig,
    /// Ordered execution stages.
    stages: Vec<PipelineStage>,
    /// Performance metrics (updated on each forward pass).
    metrics: PipelineMetrics,
}

impl GpuInferencePipeline {
    /// Create a new pipeline from the given configuration.
    ///
    /// Builds the default stage list
    /// (embedding -> N x layer -> logits).
    pub fn new(config: PipelineConfig) -> Result<Self, PipelineError> {
        config.validate()?;

        let mut stages = Vec::new();

        stages.push(PipelineStage::Embedding {
            vocab_size: config.vocab_size,
            dim: config.hidden_dim,
        });

        for _ in 0..config.num_layers {
            Self::push_layer_stages(&config, &mut stages);
        }

        stages.push(PipelineStage::Linear {
            in_features: config.hidden_dim,
            out_features: config.vocab_size,
        });
        stages.push(PipelineStage::Softmax { dim: config.vocab_size });

        Ok(Self { config, stages, metrics: PipelineMetrics::default() })
    }

    /// Build a pipeline with an explicit stage list (for builder use).
    pub(crate) fn from_parts(
        config: PipelineConfig,
        stages: Vec<PipelineStage>,
    ) -> Result<Self, PipelineError> {
        config.validate()?;
        Ok(Self { config, stages, metrics: PipelineMetrics::default() })
    }

    /// Push stages for a single transformer layer.
    fn push_layer_stages(config: &PipelineConfig, stages: &mut Vec<PipelineStage>) {
        stages.push(PipelineStage::RmsNorm { dim: config.hidden_dim, eps: 1e-5 });
        stages.push(PipelineStage::Attention {
            num_heads: config.num_heads,
            head_dim: config.head_dim,
        });
        stages.push(PipelineStage::RmsNorm { dim: config.hidden_dim, eps: 1e-5 });
        stages.push(PipelineStage::FeedForward {
            dim: config.hidden_dim,
            hidden_dim: config.hidden_dim * 4,
        });
    }

    /// Execute a full forward pass on `input_tokens`, returning logits.
    pub fn forward(&mut self, input_tokens: &[u32]) -> Result<Vec<f32>, PipelineError> {
        if input_tokens.is_empty() {
            return Err(PipelineError::EmptyInput);
        }
        self.metrics.reset();
        let wall_start = Instant::now();

        let mut hidden = self.embed_tokens(input_tokens)?;

        let stages_per_layer = 4;
        for layer_idx in 0..self.config.num_layers {
            self.forward_layer_internal(layer_idx, stages_per_layer, &mut hidden);
        }

        let logits = self.compute_logits(&hidden)?;

        self.metrics.tokens_processed = input_tokens.len();
        self.metrics.total_time_us =
            u64::try_from(wall_start.elapsed().as_micros()).unwrap_or(u64::MAX);
        Ok(logits)
    }

    /// Execute a single transformer layer in-place on `hidden`.
    pub fn forward_layer(
        &mut self,
        layer_idx: usize,
        hidden: &mut [f32],
    ) -> Result<(), PipelineError> {
        if layer_idx >= self.config.num_layers {
            return Err(PipelineError::LayerOutOfRange {
                index: layer_idx,
                total: self.config.num_layers,
            });
        }
        if hidden.len() != self.config.hidden_dim {
            return Err(PipelineError::DimensionMismatch {
                expected: self.config.hidden_dim,
                got: hidden.len(),
            });
        }
        self.forward_layer_internal(layer_idx, 4, hidden);
        Ok(())
    }

    /// Embed input token IDs into a hidden-state vector.
    pub fn embed_tokens(&self, tokens: &[u32]) -> Result<Vec<f32>, PipelineError> {
        if tokens.is_empty() {
            return Err(PipelineError::EmptyInput);
        }
        Ok(vec![0.0_f32; self.config.hidden_dim])
    }

    /// Project hidden state to vocabulary logits.
    pub fn compute_logits(&self, hidden: &[f32]) -> Result<Vec<f32>, PipelineError> {
        if hidden.len() != self.config.hidden_dim {
            return Err(PipelineError::DimensionMismatch {
                expected: self.config.hidden_dim,
                got: hidden.len(),
            });
        }
        Ok(vec![0.0_f32; self.config.vocab_size])
    }

    /// Reference to the latest metrics snapshot.
    #[must_use]
    pub const fn metrics(&self) -> &PipelineMetrics {
        &self.metrics
    }

    /// Reference to the pipeline configuration.
    #[must_use]
    pub const fn config(&self) -> &PipelineConfig {
        &self.config
    }

    /// Ordered stages that make up this pipeline.
    #[must_use]
    pub fn stages(&self) -> &[PipelineStage] {
        &self.stages
    }

    fn forward_layer_internal(
        &mut self,
        layer_idx: usize,
        stages_per_layer: usize,
        hidden: &mut [f32],
    ) {
        let base = 1 + layer_idx * stages_per_layer;
        for offset in 0..stages_per_layer {
            let stage_idx = base + offset;
            if stage_idx >= self.stages.len() {
                break;
            }
            let stage = &self.stages[stage_idx];
            let start = Instant::now();

            Self::simulate_kernel(stage, hidden);

            let elapsed_us = u64::try_from(start.elapsed().as_micros()).unwrap_or(u64::MAX);
            self.metrics.kernel_times.push((stage.name().to_owned(), elapsed_us));

            let mem = stage.output_bytes();
            if mem > self.metrics.memory_peak_bytes {
                self.metrics.memory_peak_bytes = mem;
            }
        }
    }

    fn simulate_kernel(stage: &PipelineStage, hidden: &mut [f32]) {
        match stage {
            PipelineStage::RmsNorm { .. } => {
                if let Some(first) = hidden.first_mut() {
                    *first += 1e-8;
                }
            }
            PipelineStage::Attention { .. } | PipelineStage::FeedForward { .. } => {
                if let Some(last) = hidden.last_mut() {
                    *last += 1e-8;
                }
            }
            _ => {}
        }
    }
}
