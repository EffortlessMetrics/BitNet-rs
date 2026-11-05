//! OpenCL output projection head for Intel Arc A770 (Xe-HPG).
//!
//! Final linear layer that projects hidden states to vocabulary logits
//! with tied/untied weight support and efficient partial vocabulary decoding.
//!
//! # Components
//!
//! - [`OutputHeadConfig`] — dimension and projection parameters
//! - [`OutputHead`] — hidden state → logits projection
//! - [`TiedWeights`] — shares embedding matrix for output projection (transposed)
//! - [`PartialVocabDecoder`] — top-K candidate selection for efficiency
//! - [`LogitNormalizer`] — subtract-max normalization for numerical stability
//! - [`OutputStats`] — logits range, entropy, and top-K coverage
//! - [`EfficientProjection`] — tiled matmul for large vocabulary projection
//!
//! # CPU reference
//!
//! All projection operations include scalar CPU reference implementations
//! for correctness testing and non-GPU environments.

use bitnet_common::{KernelError, Result};

// ── OpenCL kernel source ─────────────────────────────────────────

/// OpenCL kernel source for output head projection operations.
pub const OUTPUT_HEAD_CL: &str = include_str!("gpu/kernels/output_head.cl");

// ── Configuration ────────────────────────────────────────────────

/// Configuration for the output projection head.
#[derive(Debug, Clone)]
pub struct OutputHeadConfig {
    /// Hidden dimension of the transformer (input size).
    pub hidden_dim: usize,
    /// Vocabulary size (output size).
    pub vocab_size: usize,
    /// Whether to share embedding weights for projection.
    pub tied_weights: bool,
    /// Whether to add a bias term after projection.
    pub use_bias: bool,
}

impl OutputHeadConfig {
    /// Create a new output head configuration.
    pub fn new(hidden_dim: usize, vocab_size: usize) -> Self {
        Self { hidden_dim, vocab_size, tied_weights: false, use_bias: false }
    }

    /// Enable tied weights (shared with embedding).
    #[must_use]
    pub fn with_tied_weights(mut self) -> Self {
        self.tied_weights = true;
        self
    }

    /// Enable bias addition.
    #[must_use]
    pub fn with_bias(mut self) -> Self {
        self.use_bias = true;
        self
    }
}

// ── OutputHead ───────────────────────────────────────────────────

/// Output projection head: projects hidden states to vocabulary logits.
///
/// Computes `logits = hidden @ weight^T [+ bias]` where weight is
/// `[vocab_size, hidden_dim]`.
#[derive(Debug, Clone)]
pub struct OutputHead {
    /// Weight matrix: `[vocab_size, hidden_dim]`.
    weight: Vec<f32>,
    /// Optional bias vector: `[vocab_size]`.
    bias: Option<Vec<f32>>,
    /// Configuration.
    config: OutputHeadConfig,
}

impl OutputHead {
    /// Create a new output head with the given weights and optional bias.
    ///
    /// # Errors
    /// Returns an error if weight dimensions don't match config.
    pub fn new(weight: Vec<f32>, bias: Option<Vec<f32>>, config: OutputHeadConfig) -> Result<Self> {
        let expected = config.vocab_size * config.hidden_dim;
        if weight.len() != expected {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "weight length {} != vocab_size({}) * hidden_dim({})",
                    weight.len(),
                    config.vocab_size,
                    config.hidden_dim,
                ),
            }
            .into());
        }
        if let Some(ref b) = bias
            && b.len() != config.vocab_size
        {
            return Err(KernelError::InvalidArguments {
                reason: format!("bias length {} != vocab_size({})", b.len(), config.vocab_size),
            }
            .into());
        }
        Ok(Self { weight, bias, config })
    }

    /// Project hidden states to logits.
    ///
    /// * `hidden`: `[seq_len, hidden_dim]`
    /// * `output`: `[seq_len, vocab_size]`
    pub fn forward(&self, hidden: &[f32], output: &mut [f32], seq_len: usize) -> Result<()> {
        projection_ref(
            hidden,
            &self.weight,
            self.bias.as_deref(),
            output,
            seq_len,
            self.config.hidden_dim,
            self.config.vocab_size,
        )
    }

    /// Get configuration.
    pub fn config(&self) -> &OutputHeadConfig {
        &self.config
    }

    /// Get a reference to the weight matrix.
    pub fn weight(&self) -> &[f32] {
        &self.weight
    }
}

// ── TiedWeights ──────────────────────────────────────────────────

/// Tied weights: shares the embedding weight matrix for output projection.
///
/// The embedding matrix `[vocab_size, hidden_dim]` is used directly
/// (transposed) for the output projection `logits = hidden @ weight^T`.
#[derive(Debug, Clone)]
pub struct TiedWeights {
    /// Shared weight matrix: `[vocab_size, hidden_dim]`.
    weight: Vec<f32>,
    /// Hidden dimension.
    hidden_dim: usize,
    /// Vocabulary size.
    vocab_size: usize,
}

impl TiedWeights {
    /// Create tied weights from an embedding weight matrix.
    pub fn new(weight: Vec<f32>, vocab_size: usize, hidden_dim: usize) -> Result<Self> {
        let expected = vocab_size * hidden_dim;
        if weight.len() != expected {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "weight length {} != vocab_size({}) * hidden_dim({})",
                    weight.len(),
                    vocab_size,
                    hidden_dim,
                ),
            }
            .into());
        }
        Ok(Self { weight, hidden_dim, vocab_size })
    }

    /// Project hidden states to logits using the shared embedding weights.
    pub fn project(&self, hidden: &[f32], output: &mut [f32], seq_len: usize) -> Result<()> {
        projection_ref(
            hidden,
            &self.weight,
            None,
            output,
            seq_len,
            self.hidden_dim,
            self.vocab_size,
        )
    }

    /// Get a reference to the shared weight.
    pub fn weight(&self) -> &[f32] {
        &self.weight
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Hidden dimension.
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }
}

// ── PartialVocabDecoder ──────────────────────────────────────────

/// Decodes only the top-K candidate tokens from logits for efficiency.
///
/// Instead of computing a full softmax over the vocabulary, this selects
/// the K highest-scoring token indices and their logit values.
#[derive(Debug, Clone)]
pub struct PartialVocabDecoder {
    /// Number of candidates to select.
    top_k: usize,
}

impl PartialVocabDecoder {
    /// Create a new partial vocab decoder selecting `top_k` candidates.
    ///
    /// # Errors
    /// Returns an error if `top_k` is zero.
    pub fn new(top_k: usize) -> Result<Self> {
        if top_k == 0 {
            return Err(
                KernelError::InvalidArguments { reason: "top_k must be >= 1".to_string() }.into()
            );
        }
        Ok(Self { top_k })
    }

    /// Select top-K candidates from logits.
    ///
    /// Returns `(indices, values)` sorted descending by logit value.
    /// If `vocab_size < top_k`, returns all vocab entries.
    pub fn decode(&self, logits: &[f32]) -> (Vec<u32>, Vec<f32>) {
        partial_topk_ref(logits, self.top_k)
    }

    /// The number of candidates this decoder selects.
    pub fn k(&self) -> usize {
        self.top_k
    }
}

// ── LogitNormalizer ──────────────────────────────────────────────

/// Normalizes logits by subtracting the maximum value (log-sum-exp trick).
///
/// This prevents numerical overflow in downstream softmax computation
/// without changing the relative ordering or softmax output.
#[derive(Debug, Clone, Copy)]
pub struct LogitNormalizer;

impl LogitNormalizer {
    /// Normalize logits in-place by subtracting the row-wise maximum.
    ///
    /// * `logits`: `[seq_len, vocab_size]` — modified in-place
    pub fn normalize(logits: &mut [f32], seq_len: usize, vocab_size: usize) -> Result<()> {
        if vocab_size == 0 {
            return Ok(());
        }
        if logits.len() < seq_len * vocab_size {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "logits length {} < seq_len({}) * vocab_size({})",
                    logits.len(),
                    seq_len,
                    vocab_size,
                ),
            }
            .into());
        }
        for s in 0..seq_len {
            let start = s * vocab_size;
            let row = &mut logits[start..start + vocab_size];
            let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            if max_val.is_finite() {
                for v in row.iter_mut() {
                    *v -= max_val;
                }
            }
        }
        Ok(())
    }
}

// ── OutputStats ──────────────────────────────────────────────────

/// Statistics computed over output logits for diagnostics.
#[derive(Debug, Clone)]
pub struct OutputStats {
    /// Range of logit values: `(min, max)`.
    pub logits_range: (f32, f32),
    /// Shannon entropy of the softmax distribution (nats).
    pub entropy: f32,
    /// Cumulative probability mass of the top-K tokens.
    pub top_k_coverage: f32,
}

impl OutputStats {
    /// Compute statistics for a single row of logits.
    ///
    /// * `logits`: raw logit values for one sequence position
    /// * `top_k`: number of top tokens for coverage calculation
    pub fn compute(logits: &[f32], top_k: usize) -> Self {
        if logits.is_empty() {
            return Self { logits_range: (0.0, 0.0), entropy: 0.0, top_k_coverage: 0.0 };
        }

        let min_val = logits.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Stable softmax for entropy and coverage
        let max_for_softmax = max_val;
        let exp_sum: f32 = logits.iter().map(|&v| (v - max_for_softmax).exp()).sum();

        let entropy = if exp_sum > 0.0 {
            let mut h = 0.0f32;
            for &v in logits {
                let p = (v - max_for_softmax).exp() / exp_sum;
                if p > 0.0 {
                    h -= p * p.ln();
                }
            }
            h
        } else {
            0.0
        };

        // Top-K coverage: sum of top-K probabilities
        let mut probs: Vec<f32> =
            logits.iter().map(|&v| (v - max_for_softmax).exp() / exp_sum).collect();
        probs.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let k = top_k.min(probs.len());
        let top_k_coverage: f32 = probs[..k].iter().sum();

        Self { logits_range: (min_val, max_val), entropy, top_k_coverage }
    }
}

// ── EfficientProjection ──────────────────────────────────────────

/// Tiled matrix multiplication for large vocabulary projection.
///
/// Uses a tile size of 16×16 to improve cache locality, matching the
/// OpenCL kernel's `TILE_SIZE`.
#[derive(Debug, Clone)]
pub struct EfficientProjection {
    /// Tile size for blocking.
    tile_size: usize,
}

impl EfficientProjection {
    /// Default tile size matching the OpenCL kernel.
    pub const DEFAULT_TILE_SIZE: usize = 16;

    /// Create a new efficient projection with the given tile size.
    pub fn new(tile_size: usize) -> Self {
        let tile_size = if tile_size == 0 { Self::DEFAULT_TILE_SIZE } else { tile_size };
        Self { tile_size }
    }

    /// Tiled projection: `output = hidden @ weight^T`.
    ///
    /// * `hidden`: `[seq_len, hidden_dim]`
    /// * `weight`: `[vocab_size, hidden_dim]`
    /// * `output`: `[seq_len, vocab_size]`
    pub fn project(
        &self,
        hidden: &[f32],
        weight: &[f32],
        output: &mut [f32],
        seq_len: usize,
        hidden_dim: usize,
        vocab_size: usize,
    ) -> Result<()> {
        tiled_projection_ref(
            hidden,
            weight,
            output,
            seq_len,
            hidden_dim,
            vocab_size,
            self.tile_size,
        )
    }

    /// Get the tile size.
    pub fn tile_size(&self) -> usize {
        self.tile_size
    }
}

impl Default for EfficientProjection {
    fn default() -> Self {
        Self::new(Self::DEFAULT_TILE_SIZE)
    }
}

// ── CPU reference: projection ────────────────────────────────────

/// Output projection: hidden → logits with optional bias (CPU reference).
///
/// Computes `output = hidden @ weight^T [+ bias]` where:
/// * `hidden`: `[seq_len, hidden_dim]`
/// * `weight`: `[vocab_size, hidden_dim]`
/// * `output`: `[seq_len, vocab_size]`
pub fn projection_ref(
    hidden: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    seq_len: usize,
    hidden_dim: usize,
    vocab_size: usize,
) -> Result<()> {
    if hidden.len() < seq_len * hidden_dim {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "hidden length {} < seq_len({}) * hidden_dim({})",
                hidden.len(),
                seq_len,
                hidden_dim,
            ),
        }
        .into());
    }
    if weight.len() < vocab_size * hidden_dim {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "weight length {} < vocab_size({}) * hidden_dim({})",
                weight.len(),
                vocab_size,
                hidden_dim,
            ),
        }
        .into());
    }
    if output.len() < seq_len * vocab_size {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "output length {} < seq_len({}) * vocab_size({})",
                output.len(),
                seq_len,
                vocab_size,
            ),
        }
        .into());
    }
    if let Some(b) = bias
        && b.len() < vocab_size
    {
        return Err(KernelError::InvalidArguments {
            reason: format!("bias length {} < vocab_size({})", b.len(), vocab_size),
        }
        .into());
    }

    for s in 0..seq_len {
        for v in 0..vocab_size {
            let mut acc = 0.0f32;
            let h_off = s * hidden_dim;
            let w_off = v * hidden_dim;
            for k in 0..hidden_dim {
                acc += hidden[h_off + k] * weight[w_off + k];
            }
            if let Some(b) = bias {
                acc += b[v];
            }
            output[s * vocab_size + v] = acc;
        }
    }
    Ok(())
}

/// Tiled output projection (CPU reference).
///
/// Same computation as [`projection_ref`] but uses tile-based blocking
/// for improved cache locality with large vocabularies.
pub fn tiled_projection_ref(
    hidden: &[f32],
    weight: &[f32],
    output: &mut [f32],
    seq_len: usize,
    hidden_dim: usize,
    vocab_size: usize,
    tile_size: usize,
) -> Result<()> {
    if hidden.len() < seq_len * hidden_dim {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "hidden length {} < seq_len({}) * hidden_dim({})",
                hidden.len(),
                seq_len,
                hidden_dim,
            ),
        }
        .into());
    }
    if weight.len() < vocab_size * hidden_dim {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "weight length {} < vocab_size({}) * hidden_dim({})",
                weight.len(),
                vocab_size,
                hidden_dim,
            ),
        }
        .into());
    }
    if output.len() < seq_len * vocab_size {
        return Err(KernelError::InvalidArguments {
            reason: format!(
                "output length {} < seq_len({}) * vocab_size({})",
                output.len(),
                seq_len,
                vocab_size,
            ),
        }
        .into());
    }

    let ts = if tile_size == 0 { 16 } else { tile_size };

    // Zero the output
    output[..seq_len * vocab_size].fill(0.0);

    // Tiled accumulation over hidden_dim
    for s in 0..seq_len {
        for v_block in (0..vocab_size).step_by(ts) {
            let v_end = (v_block + ts).min(vocab_size);
            for k_block in (0..hidden_dim).step_by(ts) {
                let k_end = (k_block + ts).min(hidden_dim);
                for v in v_block..v_end {
                    let h_off = s * hidden_dim;
                    let w_off = v * hidden_dim;
                    let o_idx = s * vocab_size + v;
                    for k in k_block..k_end {
                        output[o_idx] += hidden[h_off + k] * weight[w_off + k];
                    }
                }
            }
        }
    }
    Ok(())
}

/// Select top-K indices and values from a logit vector (CPU reference).
///
/// Returns `(indices, values)` sorted descending by value.
pub fn partial_topk_ref(logits: &[f32], top_k: usize) -> (Vec<u32>, Vec<f32>) {
    let k = top_k.min(logits.len());
    if k == 0 {
        return (vec![], vec![]);
    }

    // Build (index, value) pairs and partial-sort
    let mut indexed: Vec<(u32, f32)> =
        logits.iter().enumerate().map(|(i, &v)| (i as u32, v)).collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);

    let indices = indexed.iter().map(|&(i, _)| i).collect();
    let values = indexed.iter().map(|&(_, v)| v).collect();
    (indices, values)
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    // ── OpenCL kernel source validation ──────────────────────

    #[test]
    fn opencl_source_is_not_empty() {
        assert!(!OUTPUT_HEAD_CL.is_empty());
    }

    #[test]
    fn opencl_source_contains_kernel_keyword() {
        assert!(OUTPUT_HEAD_CL.contains("__kernel"));
    }

    #[test]
    fn opencl_source_has_projection_kernel() {
        assert!(OUTPUT_HEAD_CL.contains("output_head_projection"));
    }

    #[test]
    fn opencl_source_has_tiled_kernel() {
        assert!(OUTPUT_HEAD_CL.contains("output_head_projection_tiled"));
    }

    #[test]
    fn opencl_source_has_normalize_kernel() {
        assert!(OUTPUT_HEAD_CL.contains("logit_normalize"));
    }

    #[test]
    fn opencl_source_has_topk_kernel() {
        assert!(OUTPUT_HEAD_CL.contains("partial_vocab_topk"));
    }

    // ── OutputHeadConfig ─────────────────────────────────────

    #[test]
    fn config_basic() {
        let cfg = OutputHeadConfig::new(2048, 32000);
        assert_eq!(cfg.hidden_dim, 2048);
        assert_eq!(cfg.vocab_size, 32000);
        assert!(!cfg.tied_weights);
        assert!(!cfg.use_bias);
    }

    #[test]
    fn config_with_options() {
        let cfg = OutputHeadConfig::new(512, 1000).with_tied_weights().with_bias();
        assert!(cfg.tied_weights);
        assert!(cfg.use_bias);
    }

    // ── OutputHead ───────────────────────────────────────────

    #[test]
    fn output_head_rejects_wrong_weight_size() {
        let cfg = OutputHeadConfig::new(4, 3);
        assert!(OutputHead::new(vec![0.0; 10], None, cfg).is_err());
    }

    #[test]
    fn output_head_rejects_wrong_bias_size() {
        let cfg = OutputHeadConfig::new(4, 3);
        let weight = vec![0.0; 12];
        assert!(OutputHead::new(weight, Some(vec![0.0; 2]), cfg).is_err());
    }

    #[test]
    fn output_head_basic_projection() {
        // hidden: [1, 3], weight: [2, 3] → output: [1, 2]
        let cfg = OutputHeadConfig::new(3, 2);
        let weight = vec![
            1.0, 0.0, 0.0, // vocab 0: [1,0,0]
            0.0, 1.0, 0.0, // vocab 1: [0,1,0]
        ];
        let head = OutputHead::new(weight, None, cfg).unwrap();
        let hidden = vec![3.0, 5.0, 7.0];
        let mut output = vec![0.0; 2];
        head.forward(&hidden, &mut output, 1).unwrap();
        assert!(approx_eq(output[0], 3.0, EPS)); // dot([3,5,7], [1,0,0])
        assert!(approx_eq(output[1], 5.0, EPS)); // dot([3,5,7], [0,1,0])
    }

    #[test]
    fn output_head_with_bias() {
        let cfg = OutputHeadConfig::new(2, 3).with_bias();
        let weight = vec![
            1.0, 0.0, // vocab 0
            0.0, 1.0, // vocab 1
            1.0, 1.0, // vocab 2
        ];
        let bias = vec![0.5, -0.5, 1.0];
        let head = OutputHead::new(weight, Some(bias), cfg).unwrap();
        let hidden = vec![2.0, 3.0];
        let mut output = vec![0.0; 3];
        head.forward(&hidden, &mut output, 1).unwrap();
        assert!(approx_eq(output[0], 2.5, EPS)); // 2 + 0.5
        assert!(approx_eq(output[1], 2.5, EPS)); // 3 - 0.5
        assert!(approx_eq(output[2], 6.0, EPS)); // 2+3+1
    }

    #[test]
    fn output_head_multi_seq() {
        let cfg = OutputHeadConfig::new(2, 2);
        let weight = vec![1.0, 0.0, 0.0, 1.0];
        let head = OutputHead::new(weight, None, cfg).unwrap();
        let hidden = vec![1.0, 2.0, 3.0, 4.0]; // [2, 2]
        let mut output = vec![0.0; 4]; // [2, 2]
        head.forward(&hidden, &mut output, 2).unwrap();
        assert!(approx_eq(output[0], 1.0, EPS));
        assert!(approx_eq(output[1], 2.0, EPS));
        assert!(approx_eq(output[2], 3.0, EPS));
        assert!(approx_eq(output[3], 4.0, EPS));
    }

    #[test]
    fn output_head_zero_weights() {
        let cfg = OutputHeadConfig::new(3, 2);
        let weight = vec![0.0; 6];
        let head = OutputHead::new(weight, None, cfg).unwrap();
        let hidden = vec![1.0, 2.0, 3.0];
        let mut output = vec![99.0; 2];
        head.forward(&hidden, &mut output, 1).unwrap();
        assert!(approx_eq(output[0], 0.0, EPS));
        assert!(approx_eq(output[1], 0.0, EPS));
    }

    #[test]
    fn output_head_config_accessor() {
        let cfg = OutputHeadConfig::new(64, 100);
        let head = OutputHead::new(vec![0.0; 6400], None, cfg).unwrap();
        assert_eq!(head.config().hidden_dim, 64);
        assert_eq!(head.config().vocab_size, 100);
    }

    // ── TiedWeights ──────────────────────────────────────────

    #[test]
    fn tied_weights_rejects_wrong_size() {
        assert!(TiedWeights::new(vec![0.0; 10], 3, 4).is_err());
    }

    #[test]
    fn tied_weights_basic_projection() {
        let weight = vec![
            1.0, 2.0, // vocab 0
            3.0, 4.0, // vocab 1
            5.0, 6.0, // vocab 2
        ];
        let tw = TiedWeights::new(weight, 3, 2).unwrap();
        let hidden = vec![1.0, 1.0];
        let mut output = vec![0.0; 3];
        tw.project(&hidden, &mut output, 1).unwrap();
        assert!(approx_eq(output[0], 3.0, EPS)); // 1+2
        assert!(approx_eq(output[1], 7.0, EPS)); // 3+4
        assert!(approx_eq(output[2], 11.0, EPS)); // 5+6
    }

    #[test]
    fn tied_vs_untied_equivalence() {
        // Same weights should produce identical logits
        let weight = vec![
            0.5, -0.3, 0.7, // vocab 0
            -0.1, 0.9, 0.2, // vocab 1
        ];
        let cfg = OutputHeadConfig::new(3, 2);
        let head = OutputHead::new(weight.clone(), None, cfg).unwrap();
        let tw = TiedWeights::new(weight, 2, 3).unwrap();

        let hidden = vec![1.0, -0.5, 2.0];
        let mut out_head = vec![0.0; 2];
        let mut out_tied = vec![0.0; 2];
        head.forward(&hidden, &mut out_head, 1).unwrap();
        tw.project(&hidden, &mut out_tied, 1).unwrap();

        for i in 0..2 {
            assert!(approx_eq(out_head[i], out_tied[i], EPS));
        }
    }

    #[test]
    fn tied_weights_accessors() {
        let tw = TiedWeights::new(vec![0.0; 6], 3, 2).unwrap();
        assert_eq!(tw.vocab_size(), 3);
        assert_eq!(tw.hidden_dim(), 2);
        assert_eq!(tw.weight().len(), 6);
    }

    // ── PartialVocabDecoder ──────────────────────────────────

    #[test]
    fn partial_decode_rejects_zero_k() {
        assert!(PartialVocabDecoder::new(0).is_err());
    }

    #[test]
    fn partial_decode_top3() {
        let decoder = PartialVocabDecoder::new(3).unwrap();
        let logits = vec![0.1, 0.9, 0.5, 0.3, 0.7];
        let (indices, values) = decoder.decode(&logits);
        assert_eq!(indices.len(), 3);
        assert_eq!(indices[0], 1); // 0.9
        assert_eq!(indices[1], 4); // 0.7
        assert_eq!(indices[2], 2); // 0.5
        assert!(approx_eq(values[0], 0.9, EPS));
        assert!(approx_eq(values[1], 0.7, EPS));
        assert!(approx_eq(values[2], 0.5, EPS));
    }

    #[test]
    fn partial_decode_k_exceeds_vocab() {
        let decoder = PartialVocabDecoder::new(10).unwrap();
        let logits = vec![1.0, 2.0, 3.0];
        let (indices, values) = decoder.decode(&logits);
        assert_eq!(indices.len(), 3);
        assert_eq!(indices[0], 2); // largest first
        assert!(approx_eq(values[0], 3.0, EPS));
    }

    #[test]
    fn partial_decode_k_equals_one() {
        let decoder = PartialVocabDecoder::new(1).unwrap();
        let logits = vec![0.1, 0.9, 0.5];
        let (indices, values) = decoder.decode(&logits);
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], 1);
        assert!(approx_eq(values[0], 0.9, EPS));
    }

    #[test]
    fn partial_decode_empty_logits() {
        let decoder = PartialVocabDecoder::new(5).unwrap();
        let (indices, values) = decoder.decode(&[]);
        assert!(indices.is_empty());
        assert!(values.is_empty());
    }

    #[test]
    fn partial_decode_accessor() {
        let decoder = PartialVocabDecoder::new(42).unwrap();
        assert_eq!(decoder.k(), 42);
    }

    // ── LogitNormalizer ──────────────────────────────────────

    #[test]
    fn normalize_basic() {
        let mut logits = vec![1.0, 3.0, 2.0];
        LogitNormalizer::normalize(&mut logits, 1, 3).unwrap();
        assert!(approx_eq(logits[0], -2.0, EPS));
        assert!(approx_eq(logits[1], 0.0, EPS));
        assert!(approx_eq(logits[2], -1.0, EPS));
    }

    #[test]
    fn normalize_multi_row() {
        let mut logits = vec![
            1.0, 3.0, // row 0: max=3
            5.0, 2.0, // row 1: max=5
        ];
        LogitNormalizer::normalize(&mut logits, 2, 2).unwrap();
        assert!(approx_eq(logits[0], -2.0, EPS));
        assert!(approx_eq(logits[1], 0.0, EPS));
        assert!(approx_eq(logits[2], 0.0, EPS));
        assert!(approx_eq(logits[3], -3.0, EPS));
    }

    #[test]
    fn normalize_all_same() {
        let mut logits = vec![5.0, 5.0, 5.0];
        LogitNormalizer::normalize(&mut logits, 1, 3).unwrap();
        for &v in &logits {
            assert!(approx_eq(v, 0.0, EPS));
        }
    }

    #[test]
    fn normalize_single_element() {
        let mut logits = vec![42.0];
        LogitNormalizer::normalize(&mut logits, 1, 1).unwrap();
        assert!(approx_eq(logits[0], 0.0, EPS));
    }

    #[test]
    fn normalize_negative_values() {
        let mut logits = vec![-3.0, -1.0, -5.0];
        LogitNormalizer::normalize(&mut logits, 1, 3).unwrap();
        assert!(approx_eq(logits[0], -2.0, EPS));
        assert!(approx_eq(logits[1], 0.0, EPS));
        assert!(approx_eq(logits[2], -4.0, EPS));
    }

    #[test]
    fn normalize_zero_vocab() {
        // Should be a no-op
        let mut logits: Vec<f32> = vec![];
        LogitNormalizer::normalize(&mut logits, 0, 0).unwrap();
    }

    #[test]
    fn normalize_rejects_short_buffer() {
        let mut logits = vec![1.0, 2.0];
        assert!(LogitNormalizer::normalize(&mut logits, 1, 5).is_err());
    }

    // ── OutputStats ──────────────────────────────────────────

    #[test]
    fn stats_basic() {
        let logits = vec![1.0, 2.0, 3.0];
        let stats = OutputStats::compute(&logits, 2);
        assert!(approx_eq(stats.logits_range.0, 1.0, EPS));
        assert!(approx_eq(stats.logits_range.1, 3.0, EPS));
        assert!(stats.entropy > 0.0);
        assert!(stats.top_k_coverage > 0.0);
        assert!(stats.top_k_coverage <= 1.0);
    }

    #[test]
    fn stats_uniform_distribution() {
        // Uniform logits → maximum entropy
        let logits = vec![0.0; 4];
        let stats = OutputStats::compute(&logits, 2);
        // Uniform over 4 → entropy = ln(4) ≈ 1.386
        let expected_entropy = (4.0f32).ln();
        assert!(approx_eq(stats.entropy, expected_entropy, 0.01));
        assert!(approx_eq(stats.top_k_coverage, 0.5, 0.01));
    }

    #[test]
    fn stats_peaked_distribution() {
        // One logit much larger → low entropy, high top-1 coverage
        let logits = vec![0.0, 0.0, 100.0, 0.0];
        let stats = OutputStats::compute(&logits, 1);
        assert!(stats.entropy < 0.1); // very peaked
        assert!(stats.top_k_coverage > 0.99); // top-1 ≈ 1.0
    }

    #[test]
    fn stats_empty_logits() {
        let stats = OutputStats::compute(&[], 5);
        assert!(approx_eq(stats.entropy, 0.0, EPS));
        assert!(approx_eq(stats.top_k_coverage, 0.0, EPS));
    }

    #[test]
    fn stats_single_logit() {
        let stats = OutputStats::compute(&[42.0], 1);
        assert!(approx_eq(stats.logits_range.0, 42.0, EPS));
        assert!(approx_eq(stats.logits_range.1, 42.0, EPS));
        assert!(approx_eq(stats.entropy, 0.0, EPS));
        assert!(approx_eq(stats.top_k_coverage, 1.0, EPS));
    }

    // ── EfficientProjection ──────────────────────────────────

    #[test]
    fn efficient_projection_matches_naive() {
        let hidden = vec![1.0, 2.0, 3.0, 4.0]; // [2, 2]
        let weight = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // [3, 2]
        let mut naive_out = vec![0.0; 6]; // [2, 3]
        let mut tiled_out = vec![0.0; 6]; // [2, 3]

        projection_ref(&hidden, &weight, None, &mut naive_out, 2, 2, 3).unwrap();
        let proj = EfficientProjection::default();
        proj.project(&hidden, &weight, &mut tiled_out, 2, 2, 3).unwrap();

        for i in 0..6 {
            assert!(
                approx_eq(naive_out[i], tiled_out[i], EPS),
                "mismatch at {i}: naive={} tiled={}",
                naive_out[i],
                tiled_out[i]
            );
        }
    }

    #[test]
    fn efficient_projection_larger_matrix() {
        // Test with dimensions that aren't tile-aligned
        let hidden_dim = 17;
        let vocab_size = 13;
        let seq_len = 3;
        let hidden: Vec<f32> = (0..seq_len * hidden_dim).map(|i| (i as f32) * 0.1).collect();
        let weight: Vec<f32> = (0..vocab_size * hidden_dim).map(|i| (i as f32) * 0.01).collect();
        let mut naive_out = vec![0.0; seq_len * vocab_size];
        let mut tiled_out = vec![0.0; seq_len * vocab_size];

        projection_ref(&hidden, &weight, None, &mut naive_out, seq_len, hidden_dim, vocab_size)
            .unwrap();
        let proj = EfficientProjection::new(4);
        proj.project(&hidden, &weight, &mut tiled_out, seq_len, hidden_dim, vocab_size).unwrap();

        for i in 0..naive_out.len() {
            assert!(
                approx_eq(naive_out[i], tiled_out[i], 0.01),
                "mismatch at {i}: naive={} tiled={}",
                naive_out[i],
                tiled_out[i]
            );
        }
    }

    #[test]
    fn efficient_projection_tile_size() {
        let proj = EfficientProjection::default();
        assert_eq!(proj.tile_size(), 16);
        let proj2 = EfficientProjection::new(8);
        assert_eq!(proj2.tile_size(), 8);
    }

    #[test]
    fn efficient_projection_zero_tile_defaults() {
        let proj = EfficientProjection::new(0);
        assert_eq!(proj.tile_size(), 16);
    }

    // ── Edge cases ───────────────────────────────────────────

    #[test]
    fn edge_vocab_size_one() {
        let cfg = OutputHeadConfig::new(3, 1);
        let weight = vec![1.0, 1.0, 1.0];
        let head = OutputHead::new(weight, None, cfg).unwrap();
        let hidden = vec![2.0, 3.0, 4.0];
        let mut output = vec![0.0; 1];
        head.forward(&hidden, &mut output, 1).unwrap();
        assert!(approx_eq(output[0], 9.0, EPS));
    }

    #[test]
    fn edge_hidden_dim_one() {
        let cfg = OutputHeadConfig::new(1, 3);
        let weight = vec![2.0, 3.0, 4.0];
        let head = OutputHead::new(weight, None, cfg).unwrap();
        let hidden = vec![5.0];
        let mut output = vec![0.0; 3];
        head.forward(&hidden, &mut output, 1).unwrap();
        assert!(approx_eq(output[0], 10.0, EPS));
        assert!(approx_eq(output[1], 15.0, EPS));
        assert!(approx_eq(output[2], 20.0, EPS));
    }

    // ── Property tests ───────────────────────────────────────

    #[test]
    fn property_normalize_max_is_zero() {
        // After normalization, the max in each row should be 0.0
        let mut logits = vec![1.5, -0.3, 2.7, 0.0, -1.2, 3.1];
        LogitNormalizer::normalize(&mut logits, 2, 3).unwrap();
        let max_row0 = logits[0..3].iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let max_row1 = logits[3..6].iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(approx_eq(max_row0, 0.0, EPS));
        assert!(approx_eq(max_row1, 0.0, EPS));
    }

    #[test]
    fn property_topk_sorted_descending() {
        let logits = vec![0.3, 0.1, 0.9, 0.7, 0.5];
        let (_, values) = partial_topk_ref(&logits, 4);
        for w in values.windows(2) {
            assert!(w[0] >= w[1], "not sorted descending: {} < {}", w[0], w[1]);
        }
    }

    #[test]
    fn property_projection_linearity() {
        // f(a*x) = a * f(x) for linear projection
        let weight = vec![1.0, 2.0, 3.0, 4.0]; // [2, 2]
        let hidden = vec![1.0, 1.0];
        let scaled_hidden = vec![3.0, 3.0];
        let mut out1 = vec![0.0; 2];
        let mut out2 = vec![0.0; 2];

        projection_ref(&hidden, &weight, None, &mut out1, 1, 2, 2).unwrap();
        projection_ref(&scaled_hidden, &weight, None, &mut out2, 1, 2, 2).unwrap();

        for i in 0..2 {
            assert!(approx_eq(out2[i], 3.0 * out1[i], EPS));
        }
    }

    #[test]
    fn property_tied_weights_same_as_full() {
        // TiedWeights.project should match OutputHead.forward with same weights
        let weight: Vec<f32> = (0..20).map(|i| (i as f32) * 0.1 - 1.0).collect();
        let vocab = 4;
        let hidden_dim = 5;

        let cfg = OutputHeadConfig::new(hidden_dim, vocab);
        let head = OutputHead::new(weight.clone(), None, cfg).unwrap();
        let tied = TiedWeights::new(weight, vocab, hidden_dim).unwrap();

        let hidden: Vec<f32> = (0..hidden_dim).map(|i| (i as f32) * 0.5).collect();
        let mut out_head = vec![0.0; vocab];
        let mut out_tied = vec![0.0; vocab];

        head.forward(&hidden, &mut out_head, 1).unwrap();
        tied.project(&hidden, &mut out_tied, 1).unwrap();

        for i in 0..vocab {
            assert!(
                approx_eq(out_head[i], out_tied[i], EPS),
                "mismatch at vocab {i}: head={} tied={}",
                out_head[i],
                out_tied[i]
            );
        }
    }

    #[test]
    fn property_stats_coverage_monotonic() {
        // top_k_coverage should increase or stay same as K increases
        let logits = vec![0.1, 0.5, 0.3, 0.8, 0.2];
        let mut prev_coverage = 0.0;
        for k in 1..=5 {
            let stats = OutputStats::compute(&logits, k);
            assert!(
                stats.top_k_coverage >= prev_coverage - EPS,
                "coverage decreased: k={k}, prev={prev_coverage}, cur={}",
                stats.top_k_coverage
            );
            prev_coverage = stats.top_k_coverage;
        }
    }

    #[test]
    fn property_full_coverage_is_one() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let stats = OutputStats::compute(&logits, 4);
        assert!(approx_eq(stats.top_k_coverage, 1.0, 0.001));
    }

    #[test]
    fn property_normalize_preserves_ordering() {
        let logits_orig = vec![1.0, 5.0, 3.0, 2.0];
        let mut logits = logits_orig.clone();
        LogitNormalizer::normalize(&mut logits, 1, 4).unwrap();
        // Relative ordering preserved
        assert!(logits[1] > logits[2]); // 5 > 3
        assert!(logits[2] > logits[3]); // 3 > 2
        assert!(logits[3] > logits[0]); // 2 > 1
    }
}
