//! CUDA quantized inference pipeline orchestrator.
//!
//! Coordinates the sequence of CUDA kernels needed for a single transformer
//! layer inference pass. The pipeline planner produces an ordered list of
//! [`PipelineStep`]s from a [`TransformerLayerConfig`], validates dimensional
//! consistency, and estimates device memory requirements.
//!
//! # Usage
//!
//! ```rust,ignore
//! use bitnet_kernels::cuda::pipeline::{InferencePipeline, TransformerLayerConfig};
//!
//! let config = TransformerLayerConfig::new(128, 32, 4096, 11008);
//! let pipeline = InferencePipeline::plan(&config);
//! pipeline.validate().expect("pipeline should be consistent");
//! let mem = pipeline.estimate_memory(1, 512);
//! ```
//!
//! # CPU fallback
//!
//! The pipeline plan itself is device-agnostic — it describes *what* to execute,
//! not *how*.  Actual kernel dispatch is handled by the caller, which may use
//! GPU launches or CPU fallback functions from sibling modules.

use bitnet_common::{KernelError, Result};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for a single transformer layer.
///
/// All dimensions must be positive and `hidden_dim` must be divisible by
/// `num_heads` (so that `head_dim == hidden_dim / num_heads`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransformerLayerConfig {
    /// Dimension of each attention head (e.g. 128).
    pub head_dim: usize,
    /// Number of attention heads (e.g. 32).
    pub num_heads: usize,
    /// Hidden (model) dimension — must equal `head_dim * num_heads`.
    pub hidden_dim: usize,
    /// FFN intermediate dimension (e.g. 11008 for LLaMA-style SwiGLU).
    pub intermediate_dim: usize,
}

impl TransformerLayerConfig {
    /// Create a new layer configuration.
    ///
    /// Returns an error when any dimension is zero or when `hidden_dim` is
    /// not equal to `head_dim * num_heads`.
    pub fn new(
        head_dim: usize,
        num_heads: usize,
        hidden_dim: usize,
        intermediate_dim: usize,
    ) -> Result<Self> {
        if head_dim == 0 || num_heads == 0 || hidden_dim == 0 || intermediate_dim == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "all dimensions must be positive".into(),
            }
            .into());
        }
        if hidden_dim != head_dim * num_heads {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "hidden_dim ({hidden_dim}) must equal head_dim * num_heads ({})",
                    head_dim * num_heads
                ),
            }
            .into());
        }
        Ok(Self { head_dim, num_heads, hidden_dim, intermediate_dim })
    }
}

// ---------------------------------------------------------------------------
// Pipeline steps
// ---------------------------------------------------------------------------

/// A single step in the inference pipeline.
///
/// Steps are executed in order; the orchestrator selects the appropriate CUDA
/// kernel (or CPU fallback) for each step at dispatch time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipelineStep {
    /// Token + position embedding lookup.
    Embedding,
    /// Pre-attention layer normalisation (RMSNorm).
    LayerNorm,
    /// Multi-head self-attention (QKV projection → scaled dot-product → output projection).
    Attention,
    /// Residual (skip) connection addition.
    Residual,
    /// Feed-forward network (gate projection → activation → down projection).
    FFN,
    /// Final logit projection to vocabulary.
    Output,
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// An ordered sequence of [`PipelineStep`]s with the originating config.
///
/// Built via [`InferencePipeline::plan`], which encodes the canonical
/// transformer-layer execution order used by LLaMA / BitNet architectures:
///
/// ```text
/// Embedding → LayerNorm → Attention → Residual → LayerNorm → FFN → Residual → Output
/// ```
#[derive(Debug, Clone)]
pub struct InferencePipeline {
    steps: Vec<PipelineStep>,
    config: TransformerLayerConfig,
}

impl InferencePipeline {
    /// Create the canonical execution plan for a transformer layer.
    ///
    /// The plan follows the pre-norm transformer architecture:
    ///
    /// 1. **Embedding** — token + position lookup
    /// 2. **LayerNorm** — pre-attention RMSNorm
    /// 3. **Attention** — multi-head self-attention
    /// 4. **Residual** — skip connection (input + attention output)
    /// 5. **LayerNorm** — pre-FFN RMSNorm
    /// 6. **FFN** — SwiGLU feed-forward
    /// 7. **Residual** — skip connection (attention output + FFN output)
    /// 8. **Output** — logit projection
    pub fn plan(config: &TransformerLayerConfig) -> Self {
        let steps = vec![
            PipelineStep::Embedding,
            PipelineStep::LayerNorm,
            PipelineStep::Attention,
            PipelineStep::Residual,
            PipelineStep::LayerNorm,
            PipelineStep::FFN,
            PipelineStep::Residual,
            PipelineStep::Output,
        ];
        Self { steps, config: config.clone() }
    }

    /// Validate that the pipeline is internally consistent.
    ///
    /// Checks:
    /// - Pipeline is non-empty.
    /// - First step is [`PipelineStep::Embedding`].
    /// - Last step is [`PipelineStep::Output`].
    /// - Every [`PipelineStep::Attention`] and [`PipelineStep::FFN`] is
    ///   preceded by a [`PipelineStep::LayerNorm`].
    /// - Every [`PipelineStep::Attention`] and [`PipelineStep::FFN`] is
    ///   followed by a [`PipelineStep::Residual`].
    /// - `hidden_dim == head_dim * num_heads`.
    pub fn validate(&self) -> Result<()> {
        if self.steps.is_empty() {
            return Err(
                KernelError::InvalidArguments { reason: "pipeline has no steps".into() }.into()
            );
        }

        // First / last bookends.
        if self.steps[0] != PipelineStep::Embedding {
            return Err(KernelError::InvalidArguments {
                reason: format!("pipeline must start with Embedding, found {:?}", self.steps[0]),
            }
            .into());
        }
        if *self.steps.last().unwrap() != PipelineStep::Output {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "pipeline must end with Output, found {:?}",
                    self.steps.last().unwrap()
                ),
            }
            .into());
        }

        // Structural: Attention/FFN must be preceded by LayerNorm and followed
        // by Residual.
        for (i, step) in self.steps.iter().enumerate() {
            if matches!(step, PipelineStep::Attention | PipelineStep::FFN) {
                if i == 0 || !matches!(self.steps[i - 1], PipelineStep::LayerNorm) {
                    return Err(KernelError::InvalidArguments {
                        reason: format!("{step:?} at index {i} must be preceded by LayerNorm"),
                    }
                    .into());
                }
                if i + 1 >= self.steps.len() || !matches!(self.steps[i + 1], PipelineStep::Residual)
                {
                    return Err(KernelError::InvalidArguments {
                        reason: format!("{step:?} at index {i} must be followed by Residual"),
                    }
                    .into());
                }
            }
        }

        // Dimensional consistency.
        if self.config.hidden_dim != self.config.head_dim * self.config.num_heads {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "hidden_dim ({}) != head_dim * num_heads ({})",
                    self.config.hidden_dim,
                    self.config.head_dim * self.config.num_heads
                ),
            }
            .into());
        }

        Ok(())
    }

    /// Return the ordered steps in this pipeline.
    pub fn steps(&self) -> &[PipelineStep] {
        &self.steps
    }

    /// Return the layer configuration backing this pipeline.
    pub fn config(&self) -> &TransformerLayerConfig {
        &self.config
    }

    /// Estimate peak device memory (bytes) for a given batch and sequence length.
    ///
    /// The estimate accounts for the major activations held simultaneously:
    ///
    /// | Tensor              | Shape                                 | Bytes (f32) |
    /// |---------------------|---------------------------------------|-------------|
    /// | hidden activations  | `batch × seq × hidden`                | 4·B·S·H     |
    /// | QKV projections     | `3 × batch × heads × seq × head_dim` | 12·B·S·H    |
    /// | attention scores    | `batch × heads × seq × seq`           | 4·B·n·S²    |
    /// | FFN intermediate    | `batch × seq × intermediate`          | 4·B·S·I     |
    /// | residual buffer     | `batch × seq × hidden`                | 4·B·S·H     |
    ///
    /// Total ≈ `4 * B * S * (17·H + I + n·S)` bytes (f32).
    ///
    /// Returns `0` when any input is zero.
    pub fn estimate_memory(&self, batch_size: usize, seq_len: usize) -> usize {
        if batch_size == 0 || seq_len == 0 {
            return 0;
        }

        let b = batch_size;
        let s = seq_len;
        let h = self.config.hidden_dim;
        let n = self.config.num_heads;
        let i = self.config.intermediate_dim;

        let bytes_per_elem: usize = 4; // f32

        // hidden activations + residual buffer: 2 × B·S·H
        let hidden = 2 * b * s * h * bytes_per_elem;
        // QKV projections: 3 × B·S·H (packed as B × n × S × head_dim)
        let qkv = 3 * b * s * h * bytes_per_elem;
        // attention score matrix: B × n × S × S
        let attn_scores = b * n * s * s * bytes_per_elem;
        // FFN intermediate: B × S × I
        let ffn = b * s * i * bytes_per_elem;

        hidden + qkv + attn_scores + ffn
    }

    /// Return the total number of steps in the pipeline.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Return `true` if the pipeline has no steps.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────

    /// Standard LLaMA-2 7B–style config.
    fn llama2_7b_config() -> TransformerLayerConfig {
        TransformerLayerConfig::new(128, 32, 4096, 11008).unwrap()
    }

    /// Small config useful for quick assertions.
    fn tiny_config() -> TransformerLayerConfig {
        TransformerLayerConfig::new(4, 2, 8, 16).unwrap()
    }

    // ── TransformerLayerConfig tests ────────────────────────────

    #[test]
    fn config_valid_construction() {
        let cfg = llama2_7b_config();
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.num_heads, 32);
        assert_eq!(cfg.hidden_dim, 4096);
        assert_eq!(cfg.intermediate_dim, 11008);
    }

    #[test]
    fn config_rejects_zero_head_dim() {
        assert!(TransformerLayerConfig::new(0, 32, 4096, 11008).is_err());
    }

    #[test]
    fn config_rejects_zero_num_heads() {
        assert!(TransformerLayerConfig::new(128, 0, 4096, 11008).is_err());
    }

    #[test]
    fn config_rejects_zero_hidden_dim() {
        assert!(TransformerLayerConfig::new(128, 32, 0, 11008).is_err());
    }

    #[test]
    fn config_rejects_zero_intermediate_dim() {
        assert!(TransformerLayerConfig::new(128, 32, 4096, 0).is_err());
    }

    #[test]
    fn config_rejects_mismatched_hidden() {
        // hidden_dim=5000 != 128*32=4096
        assert!(TransformerLayerConfig::new(128, 32, 5000, 11008).is_err());
    }

    #[test]
    fn config_clone_eq() {
        let a = llama2_7b_config();
        let b = a.clone();
        assert_eq!(a, b);
    }

    // ── plan() tests ────────────────────────────────────────────

    #[test]
    fn plan_step_count() {
        let pipe = InferencePipeline::plan(&llama2_7b_config());
        assert_eq!(pipe.len(), 8);
        assert!(!pipe.is_empty());
    }

    #[test]
    fn plan_starts_with_embedding() {
        let pipe = InferencePipeline::plan(&llama2_7b_config());
        assert_eq!(pipe.steps()[0], PipelineStep::Embedding);
    }

    #[test]
    fn plan_ends_with_output() {
        let pipe = InferencePipeline::plan(&llama2_7b_config());
        assert_eq!(*pipe.steps().last().unwrap(), PipelineStep::Output);
    }

    #[test]
    fn plan_canonical_sequence() {
        let pipe = InferencePipeline::plan(&llama2_7b_config());
        let expected = &[
            PipelineStep::Embedding,
            PipelineStep::LayerNorm,
            PipelineStep::Attention,
            PipelineStep::Residual,
            PipelineStep::LayerNorm,
            PipelineStep::FFN,
            PipelineStep::Residual,
            PipelineStep::Output,
        ];
        assert_eq!(pipe.steps(), expected);
    }

    #[test]
    fn plan_preserves_config() {
        let cfg = llama2_7b_config();
        let pipe = InferencePipeline::plan(&cfg);
        assert_eq!(pipe.config(), &cfg);
    }

    // ── validate() tests ────────────────────────────────────────

    #[test]
    fn validate_canonical_ok() {
        let pipe = InferencePipeline::plan(&llama2_7b_config());
        assert!(pipe.validate().is_ok());
    }

    #[test]
    fn validate_tiny_config_ok() {
        let pipe = InferencePipeline::plan(&tiny_config());
        assert!(pipe.validate().is_ok());
    }

    #[test]
    fn validate_empty_pipeline_fails() {
        let pipe = InferencePipeline { steps: vec![], config: llama2_7b_config() };
        assert!(pipe.validate().is_err());
    }

    #[test]
    fn validate_missing_embedding_start_fails() {
        let mut pipe = InferencePipeline::plan(&llama2_7b_config());
        pipe.steps[0] = PipelineStep::LayerNorm;
        assert!(pipe.validate().is_err());
    }

    #[test]
    fn validate_missing_output_end_fails() {
        let mut pipe = InferencePipeline::plan(&llama2_7b_config());
        let last = pipe.steps.len() - 1;
        pipe.steps[last] = PipelineStep::Residual;
        assert!(pipe.validate().is_err());
    }

    #[test]
    fn validate_attention_without_layernorm_fails() {
        let mut pipe = InferencePipeline::plan(&llama2_7b_config());
        // Replace pre-attention LayerNorm with Residual.
        pipe.steps[1] = PipelineStep::Residual;
        assert!(pipe.validate().is_err());
    }

    #[test]
    fn validate_ffn_without_residual_after_fails() {
        let mut pipe = InferencePipeline::plan(&llama2_7b_config());
        // Replace post-FFN Residual with LayerNorm.
        pipe.steps[6] = PipelineStep::LayerNorm;
        assert!(pipe.validate().is_err());
    }

    // ── estimate_memory() tests ─────────────────────────────────

    #[test]
    fn memory_zero_batch() {
        let pipe = InferencePipeline::plan(&llama2_7b_config());
        assert_eq!(pipe.estimate_memory(0, 512), 0);
    }

    #[test]
    fn memory_zero_seq() {
        let pipe = InferencePipeline::plan(&llama2_7b_config());
        assert_eq!(pipe.estimate_memory(4, 0), 0);
    }

    #[test]
    fn memory_positive() {
        let pipe = InferencePipeline::plan(&llama2_7b_config());
        let mem = pipe.estimate_memory(1, 1);
        assert!(mem > 0, "single-token memory must be positive");
    }

    #[test]
    fn memory_scales_with_batch() {
        let pipe = InferencePipeline::plan(&llama2_7b_config());
        let m1 = pipe.estimate_memory(1, 512);
        let m2 = pipe.estimate_memory(2, 512);
        // Attention scores scale as B·n·S², linear terms scale as B, so
        // m2 should be exactly 2× m1.
        assert_eq!(m2, 2 * m1);
    }

    #[test]
    fn memory_scales_superlinearly_with_seq() {
        let pipe = InferencePipeline::plan(&llama2_7b_config());
        let m1 = pipe.estimate_memory(1, 128);
        let m2 = pipe.estimate_memory(1, 256);
        // Contains an S² term, so doubling seq_len more than doubles memory.
        assert!(m2 > 2 * m1, "memory should grow super-linearly with seq_len");
    }

    #[test]
    fn memory_reasonable_order_of_magnitude() {
        // 1 batch, 512 seq, 4096 hidden  →  expect tens–hundreds of MB.
        let pipe = InferencePipeline::plan(&llama2_7b_config());
        let mem = pipe.estimate_memory(1, 512);
        let mb = mem / (1024 * 1024);
        assert!(mb >= 10 && mb <= 2048, "expected 10–2048 MB, got {mb} MB");
    }

    #[test]
    fn memory_tiny_config_small() {
        let pipe = InferencePipeline::plan(&tiny_config());
        let mem = pipe.estimate_memory(1, 4);
        // Tiny: hidden=8, intermediate=16, heads=2, seq=4
        // Should be well under 1 KB.
        assert!(mem <= 1024, "tiny config memory should be <= 1 KB, got {mem}");
    }

    // ── edge-case / large config tests ──────────────────────────

    #[test]
    fn plan_single_head() {
        let cfg = TransformerLayerConfig::new(64, 1, 64, 128).unwrap();
        let pipe = InferencePipeline::plan(&cfg);
        assert!(pipe.validate().is_ok());
        assert_eq!(pipe.config().num_heads, 1);
    }

    #[test]
    fn plan_large_config_validates() {
        // 70B-ish: head_dim=128, 64 heads, hidden=8192, intermediate=28672
        let cfg = TransformerLayerConfig::new(128, 64, 8192, 28672).unwrap();
        let pipe = InferencePipeline::plan(&cfg);
        assert!(pipe.validate().is_ok());
    }

    #[test]
    fn memory_large_batch_seq() {
        let pipe = InferencePipeline::plan(&llama2_7b_config());
        let mem = pipe.estimate_memory(32, 4096);
        // Should be multiple GB — just assert it does not overflow.
        assert!(mem > 0);
    }

    // ── property tests ──────────────────────────────────────────

    fn proptest_config() -> proptest::prelude::ProptestConfig {
        proptest::prelude::ProptestConfig { cases: 64, ..Default::default() }
    }

    proptest::proptest! {
        #![proptest_config(proptest_config())]

        #[test]
        fn prop_plan_always_starts_embedding(
            head_dim in 1_usize..=256,
            num_heads in 1_usize..=128,
            inter_mul in 1_usize..=8,
        ) {
            let hidden = head_dim * num_heads;
            let intermediate = hidden * inter_mul;
            let cfg = TransformerLayerConfig::new(head_dim, num_heads, hidden, intermediate).unwrap();
            let pipe = InferencePipeline::plan(&cfg);
            assert_eq!(pipe.steps()[0], PipelineStep::Embedding);
        }

        #[test]
        fn prop_plan_always_ends_output(
            head_dim in 1_usize..=256,
            num_heads in 1_usize..=128,
            inter_mul in 1_usize..=8,
        ) {
            let hidden = head_dim * num_heads;
            let intermediate = hidden * inter_mul;
            let cfg = TransformerLayerConfig::new(head_dim, num_heads, hidden, intermediate).unwrap();
            let pipe = InferencePipeline::plan(&cfg);
            assert_eq!(*pipe.steps().last().unwrap(), PipelineStep::Output);
        }

        #[test]
        fn prop_plan_always_validates(
            head_dim in 1_usize..=256,
            num_heads in 1_usize..=128,
            inter_mul in 1_usize..=8,
        ) {
            let hidden = head_dim * num_heads;
            let intermediate = hidden * inter_mul;
            let cfg = TransformerLayerConfig::new(head_dim, num_heads, hidden, intermediate).unwrap();
            let pipe = InferencePipeline::plan(&cfg);
            pipe.validate().unwrap();
        }

        #[test]
        fn prop_memory_monotonic_in_batch(
            batch_a in 1_usize..=64,
            batch_b in 1_usize..=64,
        ) {
            let pipe = InferencePipeline::plan(&TransformerLayerConfig::new(64, 8, 512, 1024).unwrap());
            let (lo, hi) = if batch_a <= batch_b { (batch_a, batch_b) } else { (batch_b, batch_a) };
            assert!(pipe.estimate_memory(lo, 32) <= pipe.estimate_memory(hi, 32));
        }

        #[test]
        fn prop_memory_monotonic_in_seq(
            seq_a in 1_usize..=512,
            seq_b in 1_usize..=512,
        ) {
            let pipe = InferencePipeline::plan(&TransformerLayerConfig::new(64, 8, 512, 1024).unwrap());
            let (lo, hi) = if seq_a <= seq_b { (seq_a, seq_b) } else { (seq_b, seq_a) };
            assert!(pipe.estimate_memory(4, lo) <= pipe.estimate_memory(4, hi));
        }
    }
}
