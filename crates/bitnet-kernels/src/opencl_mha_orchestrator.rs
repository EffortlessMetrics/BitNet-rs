//! Multi-head attention orchestrator for OpenCL GPU inference.
//!
//! # Overview
//!
//! Composes the full MHA pipeline: QKV projection → head split → RoPE →
//! scaled dot-product attention → head merge → output projection. Supports
//! standard MHA, grouped-query attention (GQA), and multi-query attention
//! (MQA) through configurable head counts.
//!
//! # CPU reference
//!
//! All components have pure-CPU scalar implementations for correctness
//! testing and non-GPU environments. No OpenCL runtime is required.
//!
//! # Components
//!
//! - [`MhaConfig`] — per-layer attention geometry and RoPE parameters
//! - [`QkvProjection`] — linear projections from hidden → Q, K, V
//! - [`HeadSplitter`] — reshape `[batch, seq, hidden]` → `[batch, heads, seq, head_dim]`
//! - [`RoPEApplier`] — rotary position embeddings on Q and K
//! - [`AttentionScorer`] — scaled dot-product `QK^T / sqrt(d_k)` with causal mask
//! - [`AttentionApplier`] — `softmax(scores) @ V`
//! - [`OutputProjection`] — concat heads and project back to hidden_dim
//! - [`MhaOrchestrator`] — end-to-end pipeline orchestrating all stages
//! - [`MhaStats`] — per-stage timing and throughput metrics

use bitnet_common::{KernelError, Result};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Multi-head attention configuration.
#[derive(Debug, Clone)]
pub struct MhaConfig {
    /// Number of query heads.
    pub num_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Number of key/value heads (< num_heads for GQA, 1 for MQA).
    pub num_kv_heads: usize,
    /// Base frequency for rotary position embeddings.
    pub rope_theta: f32,
    /// Maximum supported sequence length.
    pub max_seq_len: usize,
}

impl MhaConfig {
    /// Create a standard MHA config (num_kv_heads == num_heads).
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if any dimension is zero.
    pub fn new(num_heads: usize, head_dim: usize, max_seq_len: usize) -> Result<Self> {
        Self::new_gqa(num_heads, num_heads, head_dim, 10000.0, max_seq_len)
    }

    /// Create a GQA config with explicit KV head count and RoPE theta.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError::InvalidArguments`] if dimensions are zero or
    /// `num_heads` is not divisible by `num_kv_heads`.
    pub fn new_gqa(
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_theta: f32,
        max_seq_len: usize,
    ) -> Result<Self> {
        if num_heads == 0 || num_kv_heads == 0 || head_dim == 0 || max_seq_len == 0 {
            return Err(KernelError::InvalidArguments {
                reason: "all MHA dimensions must be > 0".into(),
            }
            .into());
        }
        if !num_heads.is_multiple_of(num_kv_heads) {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "num_heads ({num_heads}) must be divisible by \
                     num_kv_heads ({num_kv_heads})"
                ),
            }
            .into());
        }
        if rope_theta <= 0.0 {
            return Err(KernelError::InvalidArguments {
                reason: "rope_theta must be positive".into(),
            }
            .into());
        }
        Ok(Self { num_heads, head_dim, num_kv_heads, rope_theta, max_seq_len })
    }

    /// Hidden dimension: `num_heads * head_dim`.
    #[inline]
    pub fn hidden_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }

    /// KV projection size: `num_kv_heads * head_dim`.
    #[inline]
    pub fn kv_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    /// Number of query heads sharing each KV head.
    #[inline]
    pub fn heads_per_kv_group(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }

    /// True when this is a GQA or MQA configuration.
    #[inline]
    pub fn is_gqa(&self) -> bool {
        self.num_kv_heads != self.num_heads
    }

    /// True when this is multi-query attention (single KV head).
    #[inline]
    pub fn is_mqa(&self) -> bool {
        self.num_kv_heads == 1 && self.num_heads > 1
    }

    /// Scaling factor for dot-product attention: `1 / sqrt(head_dim)`.
    #[inline]
    pub fn scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }
}

// ---------------------------------------------------------------------------
// QKV Projection
// ---------------------------------------------------------------------------

/// Projects hidden states into query, key, and value tensors.
///
/// Weight layout:
/// - `wq`: `[hidden_dim, hidden_dim]` (Q projection)
/// - `wk`: `[hidden_dim, kv_dim]` (K projection)
/// - `wv`: `[hidden_dim, kv_dim]` (V projection)
#[derive(Debug, Clone)]
pub struct QkvProjection {
    /// Q projection weight `[hidden_dim, hidden_dim]`.
    pub wq: Vec<f32>,
    /// K projection weight `[hidden_dim, kv_dim]`.
    pub wk: Vec<f32>,
    /// V projection weight `[hidden_dim, kv_dim]`.
    pub wv: Vec<f32>,
    /// Optional Q bias `[hidden_dim]`.
    pub bq: Option<Vec<f32>>,
    /// Optional K bias `[kv_dim]`.
    pub bk: Option<Vec<f32>>,
    /// Optional V bias `[kv_dim]`.
    pub bv: Option<Vec<f32>>,
}

impl QkvProjection {
    /// Create a new QKV projection from weight matrices (no bias).
    pub fn new(wq: Vec<f32>, wk: Vec<f32>, wv: Vec<f32>) -> Self {
        Self { wq, wk, wv, bq: None, bk: None, bv: None }
    }

    /// Create with bias terms.
    pub fn with_bias(
        wq: Vec<f32>,
        wk: Vec<f32>,
        wv: Vec<f32>,
        bq: Vec<f32>,
        bk: Vec<f32>,
        bv: Vec<f32>,
    ) -> Self {
        Self { wq, wk, wv, bq: Some(bq), bk: Some(bk), bv: Some(bv) }
    }

    /// Create zero-initialized projection weights for the given config.
    pub fn zeros(config: &MhaConfig) -> Self {
        let h = config.hidden_dim();
        let kv = config.kv_dim();
        Self {
            wq: vec![0.0; h * h],
            wk: vec![0.0; h * kv],
            wv: vec![0.0; h * kv],
            bq: None,
            bk: None,
            bv: None,
        }
    }

    /// Create identity-like projection weights (useful for testing).
    ///
    /// Sets diagonal elements to 1.0 so that projection is a pass-through
    /// for appropriately sized inputs.
    pub fn identity(config: &MhaConfig) -> Self {
        let h = config.hidden_dim();
        let kv = config.kv_dim();
        let mut wq = vec![0.0; h * h];
        for i in 0..h {
            wq[i * h + i] = 1.0;
        }
        let mut wk = vec![0.0; h * kv];
        for i in 0..kv {
            wk[i * kv + i] = 1.0;
        }
        let mut wv = vec![0.0; h * kv];
        for i in 0..kv {
            wv[i * kv + i] = 1.0;
        }
        Self { wq, wk, wv, bq: None, bk: None, bv: None }
    }

    /// Validate weight dimensions against the config.
    pub fn validate(&self, config: &MhaConfig) -> Result<()> {
        let h = config.hidden_dim();
        let kv = config.kv_dim();
        if self.wq.len() != h * h {
            return Err(KernelError::InvalidArguments {
                reason: format!("wq: expected {}, got {}", h * h, self.wq.len()),
            }
            .into());
        }
        if self.wk.len() != h * kv {
            return Err(KernelError::InvalidArguments {
                reason: format!("wk: expected {}, got {}", h * kv, self.wk.len()),
            }
            .into());
        }
        if self.wv.len() != h * kv {
            return Err(KernelError::InvalidArguments {
                reason: format!("wv: expected {}, got {}", h * kv, self.wv.len()),
            }
            .into());
        }
        if let Some(ref bq) = self.bq
            && bq.len() != h
        {
            return Err(KernelError::InvalidArguments {
                reason: format!("bq: expected {h}, got {}", bq.len()),
            }
            .into());
        }
        if let Some(ref bk) = self.bk
            && bk.len() != kv
        {
            return Err(KernelError::InvalidArguments {
                reason: format!("bk: expected {kv}, got {}", bk.len()),
            }
            .into());
        }
        if let Some(ref bv) = self.bv
            && bv.len() != kv
        {
            return Err(KernelError::InvalidArguments {
                reason: format!("bv: expected {kv}, got {}", bv.len()),
            }
            .into());
        }
        Ok(())
    }

    /// Project input to Q, K, V tensors (CPU reference).
    ///
    /// - `input`: `[seq_len, hidden_dim]`
    /// - returns `(q, k, v)` where:
    ///   - `q`: `[seq_len, hidden_dim]`
    ///   - `k`: `[seq_len, kv_dim]`
    ///   - `v`: `[seq_len, kv_dim]`
    pub fn forward(
        &self,
        input: &[f32],
        seq_len: usize,
        config: &MhaConfig,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        self.validate(config)?;
        let h = config.hidden_dim();
        let kv = config.kv_dim();
        if input.len() != seq_len * h {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "input size mismatch: expected {}, got {}",
                    seq_len * h,
                    input.len()
                ),
            }
            .into());
        }

        let mut q = vec![0.0f32; seq_len * h];
        let mut k = vec![0.0f32; seq_len * kv];
        let mut v = vec![0.0f32; seq_len * kv];

        matmul_ref(input, &self.wq, &mut q, seq_len, h, h);
        matmul_ref(input, &self.wk, &mut k, seq_len, h, kv);
        matmul_ref(input, &self.wv, &mut v, seq_len, h, kv);

        if let Some(ref bq) = self.bq {
            add_bias(&mut q, bq, seq_len, h);
        }
        if let Some(ref bk) = self.bk {
            add_bias(&mut k, bk, seq_len, kv);
        }
        if let Some(ref bv) = self.bv {
            add_bias(&mut v, bv, seq_len, kv);
        }

        Ok((q, k, v))
    }
}

// ---------------------------------------------------------------------------
// Head splitter
// ---------------------------------------------------------------------------

/// Reshapes `[seq_len, num_heads * head_dim]` → per-head slices.
///
/// Provides utilities to extract and scatter per-head data for attention
/// computation without physical transposition.
pub struct HeadSplitter;

impl HeadSplitter {
    /// Extract a single head's data from interleaved layout.
    ///
    /// Input: `[seq_len, num_heads * head_dim]` row-major
    /// Output: `[seq_len, head_dim]` contiguous slice for head `h`.
    pub fn extract_head(
        data: &[f32],
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        head_idx: usize,
    ) -> Vec<f32> {
        let mut out = Vec::with_capacity(seq_len * head_dim);
        let stride = num_heads * head_dim;
        for t in 0..seq_len {
            let start = t * stride + head_idx * head_dim;
            out.extend_from_slice(&data[start..start + head_dim]);
        }
        out
    }

    /// Scatter a single head's output back into interleaved layout.
    ///
    /// Writes `[seq_len, head_dim]` data from `head_data` into the
    /// appropriate positions in `output` for head `head_idx`.
    pub fn scatter_head(
        output: &mut [f32],
        head_data: &[f32],
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        head_idx: usize,
    ) {
        let stride = num_heads * head_dim;
        for t in 0..seq_len {
            let dst = t * stride + head_idx * head_dim;
            let src = t * head_dim;
            output[dst..dst + head_dim].copy_from_slice(&head_data[src..src + head_dim]);
        }
    }

    /// Split interleaved data into all heads.
    ///
    /// Returns `Vec<Vec<f32>>` of length `num_heads`, each `[seq_len, head_dim]`.
    pub fn split_all(
        data: &[f32],
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Vec<Vec<f32>> {
        (0..num_heads).map(|h| Self::extract_head(data, seq_len, num_heads, head_dim, h)).collect()
    }

    /// Merge per-head outputs into interleaved layout.
    ///
    /// Takes `heads`: `Vec<Vec<f32>>` each `[seq_len, head_dim]`
    /// Returns `[seq_len, num_heads * head_dim]`.
    pub fn merge_all(heads: &[Vec<f32>], seq_len: usize, head_dim: usize) -> Vec<f32> {
        let num_heads = heads.len();
        let mut out = vec![0.0f32; seq_len * num_heads * head_dim];
        for (h, head_data) in heads.iter().enumerate() {
            Self::scatter_head(&mut out, head_data, seq_len, num_heads, head_dim, h);
        }
        out
    }
}

// ---------------------------------------------------------------------------
// RoPE applier
// ---------------------------------------------------------------------------

/// Applies Rotary Position Embeddings (RoPE) to Q and K tensors.
///
/// RoPE encodes position information by rotating pairs of dimensions
/// using sinusoidal functions with geometrically spaced frequencies.
pub struct RoPEApplier {
    /// Precomputed cosine table `[max_seq_len, head_dim / 2]`.
    cos_table: Vec<f32>,
    /// Precomputed sine table `[max_seq_len, head_dim / 2]`.
    sin_table: Vec<f32>,
    head_dim: usize,
    max_seq_len: usize,
}

impl RoPEApplier {
    /// Build RoPE tables for the given configuration.
    pub fn new(config: &MhaConfig) -> Self {
        let half_dim = config.head_dim / 2;
        let mut cos_table = vec![0.0f32; config.max_seq_len * half_dim];
        let mut sin_table = vec![0.0f32; config.max_seq_len * half_dim];

        for pos in 0..config.max_seq_len {
            for i in 0..half_dim {
                let freq =
                    1.0 / (config.rope_theta as f64).powf(2.0 * i as f64 / config.head_dim as f64);
                let angle = pos as f64 * freq;
                cos_table[pos * half_dim + i] = angle.cos() as f32;
                sin_table[pos * half_dim + i] = angle.sin() as f32;
            }
        }

        Self { cos_table, sin_table, head_dim: config.head_dim, max_seq_len: config.max_seq_len }
    }

    /// Apply RoPE to a single head's Q or K tensor in-place.
    ///
    /// `data`: `[seq_len, head_dim]`, modified in-place.
    /// `start_pos`: offset for autoregressive decoding.
    pub fn apply(&self, data: &mut [f32], seq_len: usize, start_pos: usize) -> Result<()> {
        if start_pos + seq_len > self.max_seq_len {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "RoPE position overflow: {} + {} > {}",
                    start_pos, seq_len, self.max_seq_len
                ),
            }
            .into());
        }
        if data.len() != seq_len * self.head_dim {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "RoPE data size mismatch: expected {}, got {}",
                    seq_len * self.head_dim,
                    data.len()
                ),
            }
            .into());
        }

        let half_dim = self.head_dim / 2;
        for t in 0..seq_len {
            let pos = start_pos + t;
            for i in 0..half_dim {
                let cos_val = self.cos_table[pos * half_dim + i];
                let sin_val = self.sin_table[pos * half_dim + i];
                let x0 = data[t * self.head_dim + i];
                let x1 = data[t * self.head_dim + half_dim + i];
                data[t * self.head_dim + i] = x0 * cos_val - x1 * sin_val;
                data[t * self.head_dim + half_dim + i] = x0 * sin_val + x1 * cos_val;
            }
        }
        Ok(())
    }

    /// Apply RoPE to all heads in an interleaved Q or K tensor.
    ///
    /// `data`: `[seq_len, num_heads * head_dim]`, modified in-place.
    pub fn apply_all_heads(
        &self,
        data: &mut [f32],
        seq_len: usize,
        num_heads: usize,
        start_pos: usize,
    ) -> Result<()> {
        if start_pos + seq_len > self.max_seq_len {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "RoPE position overflow: {} + {} > {}",
                    start_pos, seq_len, self.max_seq_len
                ),
            }
            .into());
        }

        let half_dim = self.head_dim / 2;
        let stride = num_heads * self.head_dim;
        for t in 0..seq_len {
            let pos = start_pos + t;
            for h in 0..num_heads {
                let base = t * stride + h * self.head_dim;
                for i in 0..half_dim {
                    let cos_val = self.cos_table[pos * half_dim + i];
                    let sin_val = self.sin_table[pos * half_dim + i];
                    let x0 = data[base + i];
                    let x1 = data[base + half_dim + i];
                    data[base + i] = x0 * cos_val - x1 * sin_val;
                    data[base + half_dim + i] = x0 * sin_val + x1 * cos_val;
                }
            }
        }
        Ok(())
    }

    /// Get the cosine value for a specific position and dimension pair.
    #[inline]
    pub fn cos_at(&self, pos: usize, dim_pair: usize) -> f32 {
        self.cos_table[pos * (self.head_dim / 2) + dim_pair]
    }

    /// Get the sine value for a specific position and dimension pair.
    #[inline]
    pub fn sin_at(&self, pos: usize, dim_pair: usize) -> f32 {
        self.sin_table[pos * (self.head_dim / 2) + dim_pair]
    }
}

// ---------------------------------------------------------------------------
// Attention scorer
// ---------------------------------------------------------------------------

/// Computes scaled dot-product attention scores `QK^T / sqrt(d_k)` with
/// optional causal masking.
pub struct AttentionScorer;

impl AttentionScorer {
    /// Compute raw attention scores for a single head.
    ///
    /// - `q`: `[seq_len, head_dim]`
    /// - `k`: `[kv_len, head_dim]`
    /// - Returns `[seq_len, kv_len]` score matrix.
    pub fn score(
        q: &[f32],
        k: &[f32],
        seq_len: usize,
        kv_len: usize,
        head_dim: usize,
        scale: f32,
    ) -> Vec<f32> {
        let mut scores = vec![0.0f32; seq_len * kv_len];
        for i in 0..seq_len {
            for j in 0..kv_len {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[i * head_dim + d] * k[j * head_dim + d];
                }
                scores[i * kv_len + j] = dot * scale;
            }
        }
        scores
    }

    /// Apply causal mask: positions where `j > i + offset` become `-inf`.
    pub fn apply_causal_mask(scores: &mut [f32], seq_len: usize, kv_len: usize, offset: usize) {
        for i in 0..seq_len {
            let query_pos = offset + i;
            for j in 0..kv_len {
                if j > query_pos {
                    scores[i * kv_len + j] = f32::NEG_INFINITY;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Attention applier (softmax + V weighting)
// ---------------------------------------------------------------------------

/// Applies softmax to attention scores and computes weighted sum of values.
pub struct AttentionApplier;

impl AttentionApplier {
    /// Apply softmax row-wise in-place.
    pub fn softmax_inplace(scores: &mut [f32], seq_len: usize, kv_len: usize) {
        for i in 0..seq_len {
            let row = &mut scores[i * kv_len..(i + 1) * kv_len];
            softmax_row(row);
        }
    }

    /// Compute weighted sum: `weights @ V`.
    ///
    /// - `weights`: `[seq_len, kv_len]` (post-softmax)
    /// - `v`: `[kv_len, head_dim]`
    /// - `output`: `[seq_len, head_dim]`
    pub fn apply(
        weights: &[f32],
        v: &[f32],
        output: &mut [f32],
        seq_len: usize,
        kv_len: usize,
        head_dim: usize,
    ) {
        for i in 0..seq_len {
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for j in 0..kv_len {
                    acc += weights[i * kv_len + j] * v[j * head_dim + d];
                }
                output[i * head_dim + d] = acc;
            }
        }
    }

    /// Full single-head attention: score → mask → softmax → weight V.
    #[allow(clippy::too_many_arguments)]
    pub fn full_attention(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        output: &mut [f32],
        seq_len: usize,
        kv_len: usize,
        head_dim: usize,
        scale: f32,
        causal: bool,
        offset: usize,
    ) {
        let mut scores = AttentionScorer::score(q, k, seq_len, kv_len, head_dim, scale);
        if causal {
            AttentionScorer::apply_causal_mask(&mut scores, seq_len, kv_len, offset);
        }
        AttentionApplier::softmax_inplace(&mut scores, seq_len, kv_len);
        AttentionApplier::apply(&scores, v, output, seq_len, kv_len, head_dim);
    }
}

// ---------------------------------------------------------------------------
// Output projection
// ---------------------------------------------------------------------------

/// Projects concatenated head outputs back to hidden dimension.
///
/// Weight layout: `wo` is `[hidden_dim, hidden_dim]` row-major.
#[derive(Debug, Clone)]
pub struct OutputProjection {
    /// Output projection weight `[hidden_dim, hidden_dim]`.
    pub wo: Vec<f32>,
    /// Optional output bias `[hidden_dim]`.
    pub bo: Option<Vec<f32>>,
}

impl OutputProjection {
    /// Create a new output projection (no bias).
    pub fn new(wo: Vec<f32>) -> Self {
        Self { wo, bo: None }
    }

    /// Create with bias.
    pub fn with_bias(wo: Vec<f32>, bo: Vec<f32>) -> Self {
        Self { wo, bo: Some(bo) }
    }

    /// Create zero-initialized output projection.
    pub fn zeros(config: &MhaConfig) -> Self {
        let h = config.hidden_dim();
        Self { wo: vec![0.0; h * h], bo: None }
    }

    /// Create an identity output projection (pass-through).
    pub fn identity(config: &MhaConfig) -> Self {
        let h = config.hidden_dim();
        let mut wo = vec![0.0f32; h * h];
        for i in 0..h {
            wo[i * h + i] = 1.0;
        }
        Self { wo, bo: None }
    }

    /// Validate weight dimensions.
    pub fn validate(&self, config: &MhaConfig) -> Result<()> {
        let h = config.hidden_dim();
        if self.wo.len() != h * h {
            return Err(KernelError::InvalidArguments {
                reason: format!("wo: expected {}, got {}", h * h, self.wo.len()),
            }
            .into());
        }
        if let Some(ref bo) = self.bo
            && bo.len() != h
        {
            return Err(KernelError::InvalidArguments {
                reason: format!("bo: expected {h}, got {}", bo.len()),
            }
            .into());
        }
        Ok(())
    }

    /// Project concatenated head output to hidden dim (CPU reference).
    ///
    /// - `input`: `[seq_len, hidden_dim]`
    /// - `output`: `[seq_len, hidden_dim]`
    pub fn forward(
        &self,
        input: &[f32],
        output: &mut [f32],
        seq_len: usize,
        config: &MhaConfig,
    ) -> Result<()> {
        self.validate(config)?;
        let h = config.hidden_dim();
        if input.len() != seq_len * h {
            return Err(KernelError::InvalidArguments {
                reason: format!(
                    "output projection input: expected {}, got {}",
                    seq_len * h,
                    input.len()
                ),
            }
            .into());
        }
        matmul_ref(input, &self.wo, output, seq_len, h, h);
        if let Some(ref bo) = self.bo {
            add_bias(output, bo, seq_len, h);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// MHA statistics
// ---------------------------------------------------------------------------

/// Per-stage timing and throughput metrics for an MHA forward pass.
#[derive(Debug, Clone, Default)]
pub struct MhaStats {
    /// Time spent in QKV projection (microseconds).
    pub qkv_proj_us: u64,
    /// Time spent in head splitting (microseconds).
    pub head_split_us: u64,
    /// Time spent applying RoPE (microseconds).
    pub rope_us: u64,
    /// Time spent in attention scoring + softmax + V weighting (microseconds).
    pub attention_us: u64,
    /// Time spent merging heads (microseconds).
    pub head_merge_us: u64,
    /// Time spent in output projection (microseconds).
    pub output_proj_us: u64,
    /// Total wall-clock time (microseconds).
    pub total_us: u64,
    /// Batch size processed.
    pub batch_size: usize,
    /// Sequence length processed.
    pub seq_len: usize,
}

impl MhaStats {
    /// Estimated total FLOPs for the MHA forward pass.
    pub fn estimated_flops(&self, config: &MhaConfig) -> u64 {
        let h = config.hidden_dim() as u64;
        let kv = config.kv_dim() as u64;
        let s = self.seq_len as u64;
        let b = self.batch_size.max(1) as u64;

        // QKV projection: 2*s*(h*h + 2*h*kv)
        let qkv_flops = 2 * s * (h * h + 2 * h * kv);
        // Attention: 2*s*s*h (QK^T) + 2*s*s*h (weights@V)
        let attn_flops = 4 * s * s * h;
        // Output projection: 2*s*h*h
        let out_flops = 2 * s * h * h;

        b * (qkv_flops + attn_flops + out_flops)
    }

    /// Estimated GFLOP/s throughput.
    pub fn gflops(&self, config: &MhaConfig) -> f64 {
        if self.total_us == 0 {
            return 0.0;
        }
        let flops = self.estimated_flops(config) as f64;
        flops / (self.total_us as f64) * 1e6 / 1e9
    }

    /// Estimated memory bandwidth utilization (GB/s).
    ///
    /// Counts bytes read for weights + activations.
    pub fn bandwidth_gbs(&self, config: &MhaConfig) -> f64 {
        if self.total_us == 0 {
            return 0.0;
        }
        let h = config.hidden_dim() as u64;
        let kv = config.kv_dim() as u64;
        let s = self.seq_len as u64;
        let b = self.batch_size.max(1) as u64;

        // Weight bytes: (h*h + 2*h*kv + h*h) * 4 bytes
        let weight_bytes = (2 * h * h + 2 * h * kv) * 4;
        // Activation bytes (approximate): input + QKV + output
        let act_bytes = b * s * (h + h + 2 * kv + h) * 4;

        let total_bytes = (weight_bytes + act_bytes) as f64;
        total_bytes / (self.total_us as f64) * 1e6 / 1e9
    }
}

// ---------------------------------------------------------------------------
// MHA Orchestrator
// ---------------------------------------------------------------------------

/// End-to-end multi-head attention orchestrator.
///
/// Orchestrates: QKV projection → head split → RoPE → attention → merge →
/// output projection. Provides CPU reference implementation.
pub struct MhaOrchestrator {
    config: MhaConfig,
    qkv_proj: QkvProjection,
    rope: RoPEApplier,
    output_proj: OutputProjection,
}

impl MhaOrchestrator {
    /// Create a new orchestrator with the given weights.
    pub fn new(
        config: MhaConfig,
        qkv_proj: QkvProjection,
        output_proj: OutputProjection,
    ) -> Result<Self> {
        qkv_proj.validate(&config)?;
        output_proj.validate(&config)?;
        let rope = RoPEApplier::new(&config);
        Ok(Self { config, qkv_proj, rope, output_proj })
    }

    /// Get the configuration.
    pub fn config(&self) -> &MhaConfig {
        &self.config
    }

    /// Run the full MHA forward pass (CPU reference).
    ///
    /// - `input`: `[batch_size * seq_len, hidden_dim]`
    /// - `output`: `[batch_size * seq_len, hidden_dim]`
    /// - `batch_size`: number of sequences in the batch
    /// - `seq_len`: sequence length
    /// - `start_pos`: position offset for autoregressive decoding
    /// - `causal`: whether to apply causal masking
    ///
    /// Returns [`MhaStats`] with per-stage timing.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        input: &[f32],
        output: &mut [f32],
        batch_size: usize,
        seq_len: usize,
        start_pos: usize,
        causal: bool,
    ) -> Result<MhaStats> {
        let h = self.config.hidden_dim();
        let total_tokens = batch_size * seq_len;

        if input.len() != total_tokens * h {
            return Err(KernelError::InvalidArguments {
                reason: format!("input size: expected {}, got {}", total_tokens * h, input.len()),
            }
            .into());
        }
        if output.len() != total_tokens * h {
            return Err(KernelError::InvalidArguments {
                reason: format!("output size: expected {}, got {}", total_tokens * h, output.len()),
            }
            .into());
        }

        let mut stats = MhaStats { batch_size, seq_len, ..Default::default() };
        let overall_start = Instant::now();

        // Process each batch element independently.
        for b in 0..batch_size {
            let in_start = b * seq_len * h;
            let in_end = in_start + seq_len * h;
            let batch_input = &input[in_start..in_end];

            // 1. QKV projection
            let t0 = Instant::now();
            let (mut q, mut k, v) = self.qkv_proj.forward(batch_input, seq_len, &self.config)?;
            stats.qkv_proj_us += t0.elapsed().as_micros() as u64;

            // 2. Apply RoPE to Q and K (all heads)
            let t0 = Instant::now();
            self.rope.apply_all_heads(&mut q, seq_len, self.config.num_heads, start_pos)?;
            self.rope.apply_all_heads(&mut k, seq_len, self.config.num_kv_heads, start_pos)?;
            stats.rope_us += t0.elapsed().as_micros() as u64;

            // 3. Split heads, compute attention, merge
            let t0 = Instant::now();
            let q_heads =
                HeadSplitter::split_all(&q, seq_len, self.config.num_heads, self.config.head_dim);
            let k_heads = HeadSplitter::split_all(
                &k,
                seq_len,
                self.config.num_kv_heads,
                self.config.head_dim,
            );
            let v_heads = HeadSplitter::split_all(
                &v,
                seq_len,
                self.config.num_kv_heads,
                self.config.head_dim,
            );
            stats.head_split_us += t0.elapsed().as_micros() as u64;

            let t0 = Instant::now();
            let heads_per_group = self.config.heads_per_kv_group();
            let mut attn_heads = Vec::with_capacity(self.config.num_heads);

            for (qh, q_head) in q_heads.iter().enumerate() {
                let kv_head = qh / heads_per_group;
                let mut head_out = vec![0.0f32; seq_len * self.config.head_dim];
                AttentionApplier::full_attention(
                    q_head,
                    &k_heads[kv_head],
                    &v_heads[kv_head],
                    &mut head_out,
                    seq_len,
                    seq_len,
                    self.config.head_dim,
                    self.config.scale(),
                    causal,
                    start_pos,
                );
                attn_heads.push(head_out);
            }
            stats.attention_us += t0.elapsed().as_micros() as u64;

            // 4. Merge heads
            let t0 = Instant::now();
            let merged = HeadSplitter::merge_all(&attn_heads, seq_len, self.config.head_dim);
            stats.head_merge_us += t0.elapsed().as_micros() as u64;

            // 5. Output projection
            let t0 = Instant::now();
            let out_start = b * seq_len * h;
            let out_end = out_start + seq_len * h;
            self.output_proj.forward(
                &merged,
                &mut output[out_start..out_end],
                seq_len,
                &self.config,
            )?;
            stats.output_proj_us += t0.elapsed().as_micros() as u64;
        }

        stats.total_us = overall_start.elapsed().as_micros() as u64;
        Ok(stats)
    }

    /// Run a simplified forward pass without RoPE (for testing projections).
    pub fn forward_no_rope(
        &self,
        input: &[f32],
        output: &mut [f32],
        seq_len: usize,
        causal: bool,
    ) -> Result<()> {
        let (q, k, v) = self.qkv_proj.forward(input, seq_len, &self.config)?;

        let q_heads =
            HeadSplitter::split_all(&q, seq_len, self.config.num_heads, self.config.head_dim);
        let k_heads =
            HeadSplitter::split_all(&k, seq_len, self.config.num_kv_heads, self.config.head_dim);
        let v_heads =
            HeadSplitter::split_all(&v, seq_len, self.config.num_kv_heads, self.config.head_dim);

        let heads_per_group = self.config.heads_per_kv_group();
        let mut attn_heads = Vec::with_capacity(self.config.num_heads);
        for (qh, q_head) in q_heads.iter().enumerate() {
            let kv_head = qh / heads_per_group;
            let mut head_out = vec![0.0f32; seq_len * self.config.head_dim];
            AttentionApplier::full_attention(
                q_head,
                &k_heads[kv_head],
                &v_heads[kv_head],
                &mut head_out,
                seq_len,
                seq_len,
                self.config.head_dim,
                self.config.scale(),
                causal,
                0,
            );
            attn_heads.push(head_out);
        }

        let merged = HeadSplitter::merge_all(&attn_heads, seq_len, self.config.head_dim);
        self.output_proj.forward(&merged, output, seq_len, &self.config)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// OpenCL kernel source
// ---------------------------------------------------------------------------

/// OpenCL C source for the fused MHA pipeline.
///
/// Contains kernels for QKV projection, RoPE application, scaled dot-product
/// attention, and output projection. Ready for GPU dispatch on OpenCL devices.
pub const MHA_ORCHESTRATOR_CL: &str = r#"
// --- QKV Projection ---
// Work-item: (col, row) where row=seq_idx, col=output_dim
__kernel void qkv_project(
    __global const float* input,
    __global const float* wq,
    __global const float* wk,
    __global const float* wv,
    __global float* q_out,
    __global float* k_out,
    __global float* v_out,
    const int seq_len,
    const int hidden_dim,
    const int kv_dim)
{
    int row = get_global_id(1);
    if (row >= seq_len) return;

    int col = get_global_id(0);

    // Q projection
    if (col < hidden_dim) {
        float sum = 0.0f;
        for (int k = 0; k < hidden_dim; k++) {
            sum += input[row * hidden_dim + k] * wq[k * hidden_dim + col];
        }
        q_out[row * hidden_dim + col] = sum;
    }

    // K projection
    if (col < kv_dim) {
        float sum = 0.0f;
        for (int k = 0; k < hidden_dim; k++) {
            sum += input[row * hidden_dim + k] * wk[k * kv_dim + col];
        }
        k_out[row * kv_dim + col] = sum;
    }

    // V projection
    if (col < kv_dim) {
        float sum = 0.0f;
        for (int k = 0; k < hidden_dim; k++) {
            sum += input[row * hidden_dim + k] * wv[k * kv_dim + col];
        }
        v_out[row * kv_dim + col] = sum;
    }
}

// --- RoPE Application ---
__kernel void apply_rope(
    __global float* data,
    const int seq_len,
    const int num_heads,
    const int head_dim,
    const int start_pos,
    const float rope_theta)
{
    int tid = get_global_id(0);
    int half_dim = head_dim / 2;
    int total_pairs = seq_len * num_heads * half_dim;
    if (tid >= total_pairs) return;

    int i = tid % half_dim;
    int remaining = tid / half_dim;
    int h = remaining % num_heads;
    int t = remaining / num_heads;

    int pos = start_pos + t;
    float freq = 1.0f / pow(rope_theta, 2.0f * (float)i / (float)head_dim);
    float angle = (float)pos * freq;
    float cos_val = cos(angle);
    float sin_val = sin(angle);

    int stride = num_heads * head_dim;
    int base = t * stride + h * head_dim;
    float x0 = data[base + i];
    float x1 = data[base + half_dim + i];
    data[base + i] = x0 * cos_val - x1 * sin_val;
    data[base + half_dim + i] = x0 * sin_val + x1 * cos_val;
}

// --- Scaled Dot-Product Attention (single head) ---
__kernel void sdpa_single_head(
    __global const float* Q,
    __global const float* K,
    __global const float* V,
    __global float* output,
    const int seq_len,
    const int kv_len,
    const int head_dim,
    const float scale,
    const int causal,
    const int offset)
{
    int seq_idx = get_global_id(0);
    if (seq_idx >= seq_len) return;

    float scores[4096];
    if (kv_len > 4096) return;

    for (int j = 0; j < kv_len; j++) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += Q[seq_idx * head_dim + d] * K[j * head_dim + d];
        }
        scores[j] = dot * scale;
    }

    if (causal) {
        int query_pos = offset + seq_idx;
        for (int j = query_pos + 1; j < kv_len; j++) {
            scores[j] = -1e30f;
        }
    }

    float max_score = scores[0];
    for (int j = 1; j < kv_len; j++) {
        if (scores[j] > max_score) max_score = scores[j];
    }
    float sum = 0.0f;
    for (int j = 0; j < kv_len; j++) {
        scores[j] = exp(scores[j] - max_score);
        sum += scores[j];
    }
    if (sum > 0.0f) {
        for (int j = 0; j < kv_len; j++) {
            scores[j] /= sum;
        }
    }

    for (int d = 0; d < head_dim; d++) {
        float acc = 0.0f;
        for (int j = 0; j < kv_len; j++) {
            acc += scores[j] * V[j * head_dim + d];
        }
        output[seq_idx * head_dim + d] = acc;
    }
}
"#;

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Row-major matrix multiply: C = A @ B.
///
/// A: `[m, k]`, B: `[k, n]`, C: `[m, n]`.
fn matmul_ref(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// Add bias to each row: `data[t, :] += bias[:]`.
fn add_bias(data: &mut [f32], bias: &[f32], rows: usize, cols: usize) {
    for t in 0..rows {
        for j in 0..cols {
            data[t * cols + j] += bias[j];
        }
    }
}

/// Numerically stable softmax over a single row (in-place).
fn softmax_row(row: &mut [f32]) {
    if row.is_empty() {
        return;
    }
    let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max_val == f32::NEG_INFINITY {
        row.iter_mut().for_each(|v| *v = 0.0);
        return;
    }
    let mut sum = 0.0f32;
    for v in row.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in row.iter_mut() {
            *v /= sum;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const ATOL: f32 = 1e-5;

    fn assert_close(a: f32, b: f32, tol: f32) {
        assert!((a - b).abs() <= tol, "values differ: {a} vs {b} (diff={})", (a - b).abs());
    }

    fn assert_slices_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() <= tol, "index {i}: {x} vs {y} (diff={})", (x - y).abs());
        }
    }

    // ===================================================================
    // MhaConfig tests
    // ===================================================================

    #[test]
    fn test_config_standard_mha() {
        let cfg = MhaConfig::new(8, 64, 1024).unwrap();
        assert_eq!(cfg.num_heads, 8);
        assert_eq!(cfg.num_kv_heads, 8);
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(cfg.hidden_dim(), 512);
        assert_eq!(cfg.kv_dim(), 512);
        assert!(!cfg.is_gqa());
        assert!(!cfg.is_mqa());
    }

    #[test]
    fn test_config_gqa() {
        let cfg = MhaConfig::new_gqa(8, 2, 64, 10000.0, 1024).unwrap();
        assert_eq!(cfg.heads_per_kv_group(), 4);
        assert!(cfg.is_gqa());
        assert!(!cfg.is_mqa());
        assert_eq!(cfg.kv_dim(), 128);
    }

    #[test]
    fn test_config_mqa() {
        let cfg = MhaConfig::new_gqa(8, 1, 64, 10000.0, 1024).unwrap();
        assert!(cfg.is_mqa());
        assert!(cfg.is_gqa());
        assert_eq!(cfg.heads_per_kv_group(), 8);
    }

    #[test]
    fn test_config_zero_heads_rejected() {
        assert!(MhaConfig::new(0, 64, 1024).is_err());
    }

    #[test]
    fn test_config_zero_head_dim_rejected() {
        assert!(MhaConfig::new(8, 0, 1024).is_err());
    }

    #[test]
    fn test_config_zero_max_seq_len_rejected() {
        assert!(MhaConfig::new(8, 64, 0).is_err());
    }

    #[test]
    fn test_config_zero_kv_heads_rejected() {
        assert!(MhaConfig::new_gqa(8, 0, 64, 10000.0, 1024).is_err());
    }

    #[test]
    fn test_config_indivisible_heads_rejected() {
        assert!(MhaConfig::new_gqa(8, 3, 64, 10000.0, 1024).is_err());
    }

    #[test]
    fn test_config_negative_rope_theta_rejected() {
        assert!(MhaConfig::new_gqa(8, 8, 64, -1.0, 1024).is_err());
    }

    #[test]
    fn test_config_zero_rope_theta_rejected() {
        assert!(MhaConfig::new_gqa(8, 8, 64, 0.0, 1024).is_err());
    }

    #[test]
    fn test_config_scale() {
        let cfg = MhaConfig::new(4, 64, 128).unwrap();
        assert_close(cfg.scale(), 1.0 / 8.0, ATOL);
    }

    #[test]
    fn test_config_single_head() {
        let cfg = MhaConfig::new(1, 32, 512).unwrap();
        assert_eq!(cfg.hidden_dim(), 32);
        assert!(!cfg.is_gqa());
        assert!(!cfg.is_mqa());
    }

    // ===================================================================
    // QKV Projection tests
    // ===================================================================

    #[test]
    fn test_qkv_proj_identity() {
        let cfg = MhaConfig::new(2, 2, 64).unwrap();
        let proj = QkvProjection::identity(&cfg);
        let input = vec![1.0, 2.0, 3.0, 4.0]; // [1, 4]
        let (q, k, v) = proj.forward(&input, 1, &cfg).unwrap();
        assert_eq!(q.len(), 4);
        assert_eq!(k.len(), 4);
        assert_eq!(v.len(), 4);
        assert_slices_close(&q, &input, ATOL);
    }

    #[test]
    fn test_qkv_proj_zeros() {
        let cfg = MhaConfig::new(2, 2, 64).unwrap();
        let proj = QkvProjection::zeros(&cfg);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let (q, k, v) = proj.forward(&input, 1, &cfg).unwrap();
        for val in q.iter().chain(k.iter()).chain(v.iter()) {
            assert_close(*val, 0.0, ATOL);
        }
    }

    #[test]
    fn test_qkv_proj_with_bias() {
        let cfg = MhaConfig::new(1, 2, 64).unwrap();
        let wq = vec![1.0, 0.0, 0.0, 1.0]; // identity
        let wk = vec![1.0, 0.0, 0.0, 1.0];
        let wv = vec![1.0, 0.0, 0.0, 1.0];
        let bq = vec![0.5, 0.5];
        let bk = vec![1.0, 1.0];
        let bv = vec![-1.0, -1.0];
        let proj = QkvProjection::with_bias(wq, wk, wv, bq, bk, bv);
        let input = vec![1.0, 2.0];
        let (q, k, v) = proj.forward(&input, 1, &cfg).unwrap();
        assert_slices_close(&q, &[1.5, 2.5], ATOL);
        assert_slices_close(&k, &[2.0, 3.0], ATOL);
        assert_slices_close(&v, &[0.0, 1.0], ATOL);
    }

    #[test]
    fn test_qkv_proj_multi_seq() {
        let cfg = MhaConfig::new(1, 2, 64).unwrap();
        let wq = vec![1.0, 0.0, 0.0, 1.0]; // identity
        let wk = vec![1.0, 0.0, 0.0, 1.0];
        let wv = vec![1.0, 0.0, 0.0, 1.0];
        let proj = QkvProjection::new(wq, wk, wv);
        let input = vec![1.0, 2.0, 3.0, 4.0]; // [2, 2]
        let (q, k, v) = proj.forward(&input, 2, &cfg).unwrap();
        assert_slices_close(&q, &input, ATOL);
        assert_slices_close(&k, &input, ATOL);
        assert_slices_close(&v, &input, ATOL);
    }

    #[test]
    fn test_qkv_proj_wrong_input_size() {
        let cfg = MhaConfig::new(2, 2, 64).unwrap();
        let proj = QkvProjection::zeros(&cfg);
        let input = vec![1.0, 2.0]; // too small
        assert!(proj.forward(&input, 1, &cfg).is_err());
    }

    #[test]
    fn test_qkv_proj_validation_wrong_wq_size() {
        let cfg = MhaConfig::new(2, 2, 64).unwrap();
        let proj = QkvProjection::new(vec![0.0; 10], vec![0.0; 16], vec![0.0; 16]);
        assert!(proj.validate(&cfg).is_err());
    }

    #[test]
    fn test_qkv_proj_gqa_sizes() {
        let cfg = MhaConfig::new_gqa(4, 2, 2, 10000.0, 64).unwrap();
        let proj = QkvProjection::zeros(&cfg);
        let input = vec![0.0; 8]; // [1, 8]
        let (q, k, v) = proj.forward(&input, 1, &cfg).unwrap();
        assert_eq!(q.len(), 8); // num_heads * head_dim = 8
        assert_eq!(k.len(), 4); // num_kv_heads * head_dim = 4
        assert_eq!(v.len(), 4);
    }

    #[test]
    fn test_qkv_proj_bias_validation_wrong_bq() {
        let cfg = MhaConfig::new(1, 2, 64).unwrap();
        let proj = QkvProjection::with_bias(
            vec![0.0; 4],
            vec![0.0; 4],
            vec![0.0; 4],
            vec![0.0; 3], // wrong size
            vec![0.0; 2],
            vec![0.0; 2],
        );
        assert!(proj.validate(&cfg).is_err());
    }

    #[test]
    fn test_qkv_proj_bias_validation_wrong_bk() {
        let cfg = MhaConfig::new(1, 2, 64).unwrap();
        let proj = QkvProjection::with_bias(
            vec![0.0; 4],
            vec![0.0; 4],
            vec![0.0; 4],
            vec![0.0; 2],
            vec![0.0; 3], // wrong size
            vec![0.0; 2],
        );
        assert!(proj.validate(&cfg).is_err());
    }

    #[test]
    fn test_qkv_proj_bias_validation_wrong_bv() {
        let cfg = MhaConfig::new(1, 2, 64).unwrap();
        let proj = QkvProjection::with_bias(
            vec![0.0; 4],
            vec![0.0; 4],
            vec![0.0; 4],
            vec![0.0; 2],
            vec![0.0; 2],
            vec![0.0; 3], // wrong size
        );
        assert!(proj.validate(&cfg).is_err());
    }

    // ===================================================================
    // HeadSplitter tests
    // ===================================================================

    #[test]
    fn test_head_splitter_extract_single() {
        // 2 heads, head_dim=2, seq_len=1 => [1, 4] interleaved
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let h0 = HeadSplitter::extract_head(&data, 1, 2, 2, 0);
        let h1 = HeadSplitter::extract_head(&data, 1, 2, 2, 1);
        assert_slices_close(&h0, &[1.0, 2.0], ATOL);
        assert_slices_close(&h1, &[3.0, 4.0], ATOL);
    }

    #[test]
    fn test_head_splitter_extract_multi_seq() {
        // 2 heads, head_dim=2, seq_len=2 => [2, 4]
        let data = vec![
            1.0, 2.0, 3.0, 4.0, // t=0
            5.0, 6.0, 7.0, 8.0, // t=1
        ];
        let h0 = HeadSplitter::extract_head(&data, 2, 2, 2, 0);
        assert_slices_close(&h0, &[1.0, 2.0, 5.0, 6.0], ATOL);
    }

    #[test]
    fn test_head_splitter_roundtrip() {
        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // t=0, 3 heads, dim=2
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // t=1
        ];
        let heads = HeadSplitter::split_all(&data, 2, 3, 2);
        let merged = HeadSplitter::merge_all(&heads, 2, 2);
        assert_slices_close(&merged, &data, ATOL);
    }

    #[test]
    fn test_head_splitter_single_head() {
        let data = vec![1.0, 2.0, 3.0, 4.0]; // [2, 2] single head
        let heads = HeadSplitter::split_all(&data, 2, 1, 2);
        assert_eq!(heads.len(), 1);
        assert_slices_close(&heads[0], &data, ATOL);
        let merged = HeadSplitter::merge_all(&heads, 2, 2);
        assert_slices_close(&merged, &data, ATOL);
    }

    #[test]
    fn test_head_splitter_scatter() {
        let mut output = vec![0.0f32; 8]; // [2, 4] for 2 heads, dim=2
        let h0_data = vec![1.0, 2.0, 5.0, 6.0]; // [2, 2]
        let h1_data = vec![3.0, 4.0, 7.0, 8.0];
        HeadSplitter::scatter_head(&mut output, &h0_data, 2, 2, 2, 0);
        HeadSplitter::scatter_head(&mut output, &h1_data, 2, 2, 2, 1);
        let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert_slices_close(&output, &expected, ATOL);
    }

    // ===================================================================
    // RoPE tests
    // ===================================================================

    #[test]
    fn test_rope_position_zero_is_identity() {
        let cfg = MhaConfig::new(1, 4, 128).unwrap();
        let rope = RoPEApplier::new(&cfg);
        // At position 0, cos=1.0 and sin=0.0, so RoPE is identity
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let original = data.clone();
        rope.apply(&mut data, 1, 0).unwrap();
        assert_slices_close(&data, &original, ATOL);
    }

    #[test]
    fn test_rope_changes_with_position() {
        let cfg = MhaConfig::new(1, 4, 128).unwrap();
        let rope = RoPEApplier::new(&cfg);
        let mut data0 = vec![1.0, 2.0, 3.0, 4.0];
        let mut data1 = vec![1.0, 2.0, 3.0, 4.0];
        rope.apply(&mut data0, 1, 0).unwrap();
        rope.apply(&mut data1, 1, 1).unwrap();
        // Position 1 should differ from position 0
        let differs = data0.iter().zip(data1.iter()).any(|(a, b)| (a - b).abs() > ATOL);
        assert!(differs, "RoPE should produce different outputs for different positions");
    }

    #[test]
    fn test_rope_preserves_norm() {
        let cfg = MhaConfig::new(1, 4, 128).unwrap();
        let rope = RoPEApplier::new(&cfg);
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let norm_before: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        rope.apply(&mut data, 1, 5).unwrap();
        let norm_after: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert_close(norm_before, norm_after, 1e-4);
    }

    #[test]
    fn test_rope_overflow_rejected() {
        let cfg = MhaConfig::new(1, 4, 8).unwrap();
        let rope = RoPEApplier::new(&cfg);
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        assert!(rope.apply(&mut data, 1, 8).is_err());
    }

    #[test]
    fn test_rope_size_mismatch_rejected() {
        let cfg = MhaConfig::new(1, 4, 128).unwrap();
        let rope = RoPEApplier::new(&cfg);
        let mut data = vec![1.0, 2.0]; // too small
        assert!(rope.apply(&mut data, 1, 0).is_err());
    }

    #[test]
    fn test_rope_all_heads() {
        let cfg = MhaConfig::new(2, 4, 128).unwrap();
        let rope = RoPEApplier::new(&cfg);
        // [1, 8] = 2 heads of dim 4
        let mut data = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let original = data.clone();
        rope.apply_all_heads(&mut data, 1, 2, 0).unwrap();
        // Position 0 → identity
        assert_slices_close(&data, &original, ATOL);
    }

    #[test]
    fn test_rope_all_heads_position_changes() {
        let cfg = MhaConfig::new(2, 4, 128).unwrap();
        let rope = RoPEApplier::new(&cfg);
        let mut data0 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut data1 = data0.clone();
        rope.apply_all_heads(&mut data0, 1, 2, 0).unwrap();
        rope.apply_all_heads(&mut data1, 1, 2, 3).unwrap();
        let differs = data0.iter().zip(data1.iter()).any(|(a, b)| (a - b).abs() > ATOL);
        assert!(differs, "Different positions should yield different RoPE outputs");
    }

    #[test]
    fn test_rope_multi_seq() {
        let cfg = MhaConfig::new(1, 2, 128).unwrap();
        let rope = RoPEApplier::new(&cfg);
        // [2, 2] = 2 positions, 1 head, dim=2
        let mut data = vec![1.0, 0.0, 1.0, 0.0];
        rope.apply(&mut data, 2, 0).unwrap();
        // pos 0 → identity; pos 1 should rotate
        assert_close(data[0], 1.0, ATOL);
        assert_close(data[1], 0.0, ATOL);
    }

    #[test]
    fn test_rope_cos_sin_accessors() {
        let cfg = MhaConfig::new(1, 4, 128).unwrap();
        let rope = RoPEApplier::new(&cfg);
        // At position 0: cos=1.0, sin=0.0
        assert_close(rope.cos_at(0, 0), 1.0, ATOL);
        assert_close(rope.sin_at(0, 0), 0.0, ATOL);
    }

    #[test]
    fn test_rope_all_heads_overflow_rejected() {
        let cfg = MhaConfig::new(2, 4, 8).unwrap();
        let rope = RoPEApplier::new(&cfg);
        let mut data = vec![0.0; 16]; // [2, 8]
        assert!(rope.apply_all_heads(&mut data, 2, 2, 7).is_err());
    }

    // ===================================================================
    // AttentionScorer tests
    // ===================================================================

    #[test]
    fn test_scorer_identity_keys() {
        let q = vec![1.0, 0.0];
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let scores = AttentionScorer::score(&q, &k, 1, 2, 2, 1.0);
        assert_close(scores[0], 1.0, ATOL); // dot(q, k0) = 1
        assert_close(scores[1], 0.0, ATOL); // dot(q, k1) = 0
    }

    #[test]
    fn test_scorer_scaling() {
        let q = vec![2.0, 0.0];
        let k = vec![3.0, 0.0];
        let scores = AttentionScorer::score(&q, &k, 1, 1, 2, 0.5);
        assert_close(scores[0], 3.0, ATOL); // 2*3 * 0.5 = 3
    }

    #[test]
    fn test_causal_mask_applied() {
        let mut scores = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
        AttentionScorer::apply_causal_mask(&mut scores, 2, 3, 0);
        // Row 0: pos 0 can attend to j=0 only
        assert_eq!(scores[0], 1.0);
        assert_eq!(scores[1], f32::NEG_INFINITY);
        assert_eq!(scores[2], f32::NEG_INFINITY);
        // Row 1: pos 1 can attend to j=0,1
        assert_eq!(scores[3], 4.0);
        assert_eq!(scores[4], 5.0);
        assert_eq!(scores[5], f32::NEG_INFINITY);
    }

    #[test]
    fn test_causal_mask_with_offset() {
        let mut scores = vec![1.0, 2.0, 3.0]; // [1, 3]
        AttentionScorer::apply_causal_mask(&mut scores, 1, 3, 2);
        // pos 2: can attend to j=0,1,2
        assert_eq!(scores[0], 1.0);
        assert_eq!(scores[1], 2.0);
        assert_eq!(scores[2], 3.0);
    }

    #[test]
    fn test_causal_mask_seq1() {
        let mut scores = vec![5.0]; // [1, 1]
        AttentionScorer::apply_causal_mask(&mut scores, 1, 1, 0);
        assert_eq!(scores[0], 5.0); // pos 0 can attend to j=0
    }

    // ===================================================================
    // AttentionApplier tests
    // ===================================================================

    #[test]
    fn test_softmax_uniform() {
        let mut scores = vec![0.0, 0.0, 0.0]; // [1, 3]
        AttentionApplier::softmax_inplace(&mut scores, 1, 3);
        for s in &scores {
            assert_close(*s, 1.0 / 3.0, ATOL);
        }
    }

    #[test]
    fn test_softmax_dominated() {
        let mut scores = vec![100.0, 0.0]; // [1, 2]
        AttentionApplier::softmax_inplace(&mut scores, 1, 2);
        assert!(scores[0] > 0.99, "dominant score should be ~1.0");
        assert!(scores[1] < 0.01, "non-dominant should be ~0.0");
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let mut scores = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
        AttentionApplier::softmax_inplace(&mut scores, 2, 3);
        let sum0: f32 = scores[0..3].iter().sum();
        let sum1: f32 = scores[3..6].iter().sum();
        assert_close(sum0, 1.0, ATOL);
        assert_close(sum1, 1.0, ATOL);
    }

    #[test]
    fn test_apply_single_kv() {
        // Single KV position → output = V regardless of weights (weight=1.0)
        let weights = vec![1.0]; // [1, 1]
        let v = vec![3.0, 7.0]; // [1, 2]
        let mut out = vec![0.0; 2];
        AttentionApplier::apply(&weights, &v, &mut out, 1, 1, 2);
        assert_slices_close(&out, &[3.0, 7.0], ATOL);
    }

    #[test]
    fn test_apply_weighted_average() {
        let weights = vec![0.5, 0.5]; // [1, 2] uniform
        let v = vec![2.0, 4.0, 6.0, 8.0]; // [2, 2]
        let mut out = vec![0.0; 2];
        AttentionApplier::apply(&weights, &v, &mut out, 1, 2, 2);
        assert_slices_close(&out, &[4.0, 6.0], ATOL);
    }

    #[test]
    fn test_full_attention_non_causal() {
        let q = vec![1.0, 0.0];
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let v = vec![10.0, 20.0, 30.0, 40.0];
        let mut out = vec![0.0; 2];
        AttentionApplier::full_attention(&q, &k, &v, &mut out, 1, 2, 2, 1.0, false, 0);
        // Softmax of [1.0, 0.0] → [e^1/(e^1+e^0), e^0/(e^1+e^0)]
        let e1 = 1.0f32.exp();
        let w0 = e1 / (e1 + 1.0);
        let w1 = 1.0 / (e1 + 1.0);
        let expected = [w0 * 10.0 + w1 * 30.0, w0 * 20.0 + w1 * 40.0];
        assert_slices_close(&out, &expected, 1e-4);
    }

    #[test]
    fn test_full_attention_causal_seq1() {
        let q = vec![1.0, 2.0];
        let k = vec![3.0, 4.0];
        let v = vec![5.0, 6.0];
        let mut out_c = vec![0.0; 2];
        let mut out_nc = vec![0.0; 2];
        AttentionApplier::full_attention(&q, &k, &v, &mut out_c, 1, 1, 2, 1.0, true, 0);
        AttentionApplier::full_attention(&q, &k, &v, &mut out_nc, 1, 1, 2, 1.0, false, 0);
        // With seq_len=kv_len=1, causal doesn't change anything
        assert_slices_close(&out_c, &out_nc, ATOL);
    }

    #[test]
    fn test_full_attention_causal_blocks_future() {
        // seq_len=2, kv_len=2
        let q = vec![1.0, 0.0, 0.0, 1.0]; // [2, 2]
        let k = vec![1.0, 0.0, 0.0, 1.0]; // [2, 2]
        let v = vec![10.0, 20.0, 30.0, 40.0]; // [2, 2]
        let mut out = vec![0.0; 4];
        AttentionApplier::full_attention(&q, &k, &v, &mut out, 2, 2, 2, 1.0, true, 0);
        // First query (pos 0) can only attend to pos 0 → output = V[0]
        assert_slices_close(&out[0..2], &[10.0, 20.0], ATOL);
    }

    // ===================================================================
    // OutputProjection tests
    // ===================================================================

    #[test]
    fn test_output_proj_identity() {
        let cfg = MhaConfig::new(2, 2, 64).unwrap();
        let proj = OutputProjection::identity(&cfg);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = vec![0.0; 4];
        proj.forward(&input, &mut out, 1, &cfg).unwrap();
        assert_slices_close(&out, &input, ATOL);
    }

    #[test]
    fn test_output_proj_zeros() {
        let cfg = MhaConfig::new(2, 2, 64).unwrap();
        let proj = OutputProjection::zeros(&cfg);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = vec![0.0; 4];
        proj.forward(&input, &mut out, 1, &cfg).unwrap();
        for val in &out {
            assert_close(*val, 0.0, ATOL);
        }
    }

    #[test]
    fn test_output_proj_with_bias() {
        let cfg = MhaConfig::new(1, 2, 64).unwrap();
        let wo = vec![1.0, 0.0, 0.0, 1.0]; // identity
        let bo = vec![10.0, 20.0];
        let proj = OutputProjection::with_bias(wo, bo);
        let input = vec![1.0, 2.0];
        let mut out = vec![0.0; 2];
        proj.forward(&input, &mut out, 1, &cfg).unwrap();
        assert_slices_close(&out, &[11.0, 22.0], ATOL);
    }

    #[test]
    fn test_output_proj_wrong_weight_size() {
        let cfg = MhaConfig::new(2, 2, 64).unwrap();
        let proj = OutputProjection::new(vec![0.0; 10]);
        assert!(proj.validate(&cfg).is_err());
    }

    #[test]
    fn test_output_proj_wrong_bias_size() {
        let cfg = MhaConfig::new(1, 2, 64).unwrap();
        let proj = OutputProjection::with_bias(vec![0.0; 4], vec![0.0; 3]);
        assert!(proj.validate(&cfg).is_err());
    }

    // ===================================================================
    // MhaOrchestrator end-to-end tests
    // ===================================================================

    fn make_small_orchestrator(
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> MhaOrchestrator {
        let cfg = MhaConfig::new_gqa(num_heads, num_kv_heads, head_dim, 10000.0, 128).unwrap();
        let qkv = QkvProjection::identity(&cfg);
        let out_proj = OutputProjection::identity(&cfg);
        MhaOrchestrator::new(cfg, qkv, out_proj).unwrap()
    }

    #[test]
    fn test_orchestrator_forward_basic() {
        let orch = make_small_orchestrator(2, 2, 2);
        let input = vec![1.0, 0.0, 0.0, 1.0]; // [1, 4]
        let mut output = vec![0.0; 4];
        let stats = orch.forward(&input, &mut output, 1, 1, 0, false).unwrap();
        assert_eq!(stats.batch_size, 1);
        assert_eq!(stats.seq_len, 1);
        // With identity projections, single token, non-causal:
        // Q=K=V=input, attention on self → output = input (via identity out proj)
        for val in &output {
            assert!(!val.is_nan(), "output should not be NaN");
        }
    }

    #[test]
    fn test_orchestrator_forward_multi_seq() {
        let orch = make_small_orchestrator(2, 2, 2);
        let input = vec![
            1.0, 0.0, 0.0, 1.0, // t=0
            0.0, 1.0, 1.0, 0.0, // t=1
        ];
        let mut output = vec![0.0; 8];
        let stats = orch.forward(&input, &mut output, 1, 2, 0, false).unwrap();
        assert_eq!(stats.seq_len, 2);
        for val in &output {
            assert!(!val.is_nan());
        }
    }

    #[test]
    fn test_orchestrator_forward_causal() {
        let orch = make_small_orchestrator(1, 1, 2);
        let input = vec![1.0, 0.0, 0.0, 1.0]; // [2, 2]
        let mut output = vec![0.0; 4];
        orch.forward(&input, &mut output, 1, 2, 0, true).unwrap();
        for val in &output {
            assert!(!val.is_nan());
        }
    }

    #[test]
    fn test_orchestrator_gqa_mode() {
        let orch = make_small_orchestrator(4, 2, 2);
        let input = vec![0.1; 8]; // [1, 8] = 4 heads * 2 dim
        let mut output = vec![0.0; 8];
        orch.forward(&input, &mut output, 1, 1, 0, false).unwrap();
        for val in &output {
            assert!(!val.is_nan());
        }
    }

    #[test]
    fn test_orchestrator_mqa_mode() {
        let orch = make_small_orchestrator(4, 1, 2);
        let input = vec![0.1; 8];
        let mut output = vec![0.0; 8];
        orch.forward(&input, &mut output, 1, 1, 0, false).unwrap();
        for val in &output {
            assert!(!val.is_nan());
        }
    }

    #[test]
    fn test_orchestrator_multi_batch() {
        let orch = make_small_orchestrator(2, 2, 2);
        let batch_size = 3;
        let seq_len = 2;
        let h = 4;
        let input = vec![0.5; batch_size * seq_len * h];
        let mut output = vec![0.0; batch_size * seq_len * h];
        let stats = orch.forward(&input, &mut output, batch_size, seq_len, 0, false).unwrap();
        assert_eq!(stats.batch_size, 3);
        for val in &output {
            assert!(!val.is_nan());
        }
    }

    #[test]
    fn test_orchestrator_batch_independence() {
        // Each batch element should produce the same output for same input
        let orch = make_small_orchestrator(2, 2, 2);
        let h = 4;
        let input = vec![
            1.0, 2.0, 3.0, 4.0, // batch 0
            1.0, 2.0, 3.0, 4.0, // batch 1
        ];
        let mut output = vec![0.0; 8];
        orch.forward(&input, &mut output, 2, 1, 0, false).unwrap();
        assert_slices_close(&output[0..h], &output[h..2 * h], 1e-4);
    }

    #[test]
    fn test_orchestrator_wrong_input_size() {
        let orch = make_small_orchestrator(2, 2, 2);
        let input = vec![1.0; 3]; // wrong size
        let mut output = vec![0.0; 4];
        assert!(orch.forward(&input, &mut output, 1, 1, 0, false).is_err());
    }

    #[test]
    fn test_orchestrator_wrong_output_size() {
        let orch = make_small_orchestrator(2, 2, 2);
        let input = vec![1.0; 4];
        let mut output = vec![0.0; 3]; // wrong size
        assert!(orch.forward(&input, &mut output, 1, 1, 0, false).is_err());
    }

    #[test]
    fn test_orchestrator_forward_no_rope() {
        let orch = make_small_orchestrator(2, 2, 2);
        let input = vec![1.0, 0.0, 0.0, 1.0];
        let mut output = vec![0.0; 4];
        orch.forward_no_rope(&input, &mut output, 1, false).unwrap();
        for val in &output {
            assert!(!val.is_nan());
        }
    }

    #[test]
    fn test_orchestrator_seq_len_1() {
        let orch = make_small_orchestrator(2, 2, 2);
        let input = vec![0.5, 0.5, 0.5, 0.5];
        let mut output = vec![0.0; 4];
        orch.forward(&input, &mut output, 1, 1, 0, true).unwrap();
        for val in &output {
            assert!(!val.is_nan());
        }
    }

    #[test]
    fn test_orchestrator_head_dim_1() {
        let cfg = MhaConfig::new_gqa(2, 2, 1, 10000.0, 128).unwrap();
        // head_dim=1 means hidden_dim=2
        // For head_dim=1, RoPE half_dim=0 so RoPE is effectively a no-op
        let h = cfg.hidden_dim();
        let kv = cfg.kv_dim();
        let mut wq = vec![0.0f32; h * h];
        for i in 0..h {
            wq[i * h + i] = 1.0;
        }
        let mut wk = vec![0.0f32; h * kv];
        for i in 0..kv {
            wk[i * kv + i] = 1.0;
        }
        let mut wv = vec![0.0f32; h * kv];
        for i in 0..kv {
            wv[i * kv + i] = 1.0;
        }
        let qkv = QkvProjection::new(wq, wk, wv);
        let mut wo = vec![0.0f32; h * h];
        for i in 0..h {
            wo[i * h + i] = 1.0;
        }
        let out_proj = OutputProjection::new(wo);
        let orch = MhaOrchestrator::new(cfg, qkv, out_proj).unwrap();
        let input = vec![1.0, 2.0];
        let mut output = vec![0.0; 2];
        orch.forward(&input, &mut output, 1, 1, 0, false).unwrap();
        for val in &output {
            assert!(!val.is_nan());
        }
    }

    // ===================================================================
    // MhaStats tests
    // ===================================================================

    #[test]
    fn test_stats_default() {
        let stats = MhaStats::default();
        assert_eq!(stats.total_us, 0);
        assert_eq!(stats.batch_size, 0);
    }

    #[test]
    fn test_stats_flops_nonzero() {
        let cfg = MhaConfig::new(8, 64, 1024).unwrap();
        let stats = MhaStats { seq_len: 128, batch_size: 1, ..Default::default() };
        assert!(stats.estimated_flops(&cfg) > 0);
    }

    #[test]
    fn test_stats_gflops_zero_time() {
        let cfg = MhaConfig::new(8, 64, 1024).unwrap();
        let stats = MhaStats { seq_len: 128, batch_size: 1, total_us: 0, ..Default::default() };
        assert_eq!(stats.gflops(&cfg), 0.0);
    }

    #[test]
    fn test_stats_gflops_with_time() {
        let cfg = MhaConfig::new(8, 64, 1024).unwrap();
        let stats =
            MhaStats { seq_len: 128, batch_size: 1, total_us: 1_000_000, ..Default::default() };
        assert!(stats.gflops(&cfg) > 0.0);
    }

    #[test]
    fn test_stats_bandwidth() {
        let cfg = MhaConfig::new(8, 64, 1024).unwrap();
        let stats =
            MhaStats { seq_len: 128, batch_size: 1, total_us: 1_000_000, ..Default::default() };
        assert!(stats.bandwidth_gbs(&cfg) > 0.0);
    }

    #[test]
    fn test_stats_bandwidth_zero_time() {
        let cfg = MhaConfig::new(8, 64, 1024).unwrap();
        let stats = MhaStats { seq_len: 128, batch_size: 1, total_us: 0, ..Default::default() };
        assert_eq!(stats.bandwidth_gbs(&cfg), 0.0);
    }

    #[test]
    fn test_stats_from_orchestrator() {
        let orch = make_small_orchestrator(2, 2, 2);
        let input = vec![1.0; 4];
        let mut output = vec![0.0; 4];
        let stats = orch.forward(&input, &mut output, 1, 1, 0, false).unwrap();
        assert_eq!(stats.batch_size, 1);
        assert_eq!(stats.seq_len, 1);
        // At least some stage should have nonzero time (or zero for very fast)
        assert!(stats.total_us < 10_000_000, "should complete in < 10s");
    }

    // ===================================================================
    // Property-like tests: attention weights sum to 1
    // ===================================================================

    #[test]
    fn test_attention_weights_sum_to_one_seq2() {
        let q = vec![1.0, 2.0, 3.0, 4.0]; // [2, 2]
        let k = vec![0.5, 0.5, 0.5, 0.5]; // [2, 2]
        let mut scores = AttentionScorer::score(&q, &k, 2, 2, 2, 1.0);
        AttentionApplier::softmax_inplace(&mut scores, 2, 2);
        let sum0: f32 = scores[0..2].iter().sum();
        let sum1: f32 = scores[2..4].iter().sum();
        assert_close(sum0, 1.0, ATOL);
        assert_close(sum1, 1.0, ATOL);
    }

    #[test]
    fn test_attention_weights_sum_to_one_seq4() {
        let q: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect(); // [4, 4]
        let k: Vec<f32> = (0..16).map(|i| (15 - i) as f32 * 0.1).collect();
        let mut scores = AttentionScorer::score(&q, &k, 4, 4, 4, 0.5);
        AttentionApplier::softmax_inplace(&mut scores, 4, 4);
        for i in 0..4 {
            let sum: f32 = scores[i * 4..(i + 1) * 4].iter().sum();
            assert_close(sum, 1.0, ATOL);
        }
    }

    #[test]
    fn test_attention_weights_sum_to_one_with_causal_mask() {
        let q: Vec<f32> = (0..12).map(|i| i as f32 * 0.1).collect(); // [3, 4]
        let k: Vec<f32> = (0..12).map(|i| (11 - i) as f32 * 0.1).collect();
        let mut scores = AttentionScorer::score(&q, &k, 3, 3, 4, 0.5);
        AttentionScorer::apply_causal_mask(&mut scores, 3, 3, 0);
        AttentionApplier::softmax_inplace(&mut scores, 3, 3);
        for i in 0..3 {
            let sum: f32 = scores[i * 3..(i + 1) * 3].iter().sum();
            assert_close(sum, 1.0, ATOL);
        }
    }

    #[test]
    fn test_attention_weights_nonnegative() {
        let q: Vec<f32> = vec![-1.0, 2.0, 0.5, -3.0]; // [2, 2]
        let k: Vec<f32> = vec![1.0, -1.0, 2.0, 0.0]; // [2, 2]
        let mut scores = AttentionScorer::score(&q, &k, 2, 2, 2, 1.0);
        AttentionApplier::softmax_inplace(&mut scores, 2, 2);
        for s in &scores {
            assert!(*s >= 0.0, "attention weight should be non-negative, got {s}");
        }
    }

    #[test]
    fn test_attention_weights_sum_to_one_gqa_scenario() {
        // 4 query heads, 2 KV heads, head_dim=2
        let cfg = MhaConfig::new_gqa(4, 2, 2, 10000.0, 128).unwrap();
        let seq_len = 3;
        let scale = cfg.scale();
        // Generate some Q, K per head
        for qh in 0..cfg.num_heads {
            let kv_head = qh / cfg.heads_per_kv_group();
            let q: Vec<f32> =
                (0..seq_len * cfg.head_dim).map(|i| ((i + qh * 7) as f32) * 0.1).collect();
            let k: Vec<f32> =
                (0..seq_len * cfg.head_dim).map(|i| ((i + kv_head * 3) as f32) * 0.1).collect();
            let mut scores = AttentionScorer::score(&q, &k, seq_len, seq_len, cfg.head_dim, scale);
            AttentionApplier::softmax_inplace(&mut scores, seq_len, seq_len);
            for row in 0..seq_len {
                let sum: f32 = scores[row * seq_len..(row + 1) * seq_len].iter().sum();
                assert_close(sum, 1.0, ATOL);
            }
        }
    }

    // ===================================================================
    // Determinism tests
    // ===================================================================

    #[test]
    fn test_orchestrator_deterministic() {
        let orch = make_small_orchestrator(2, 2, 2);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // [2, 4]
        let mut out1 = vec![0.0; 8];
        let mut out2 = vec![0.0; 8];
        orch.forward(&input, &mut out1, 1, 2, 0, true).unwrap();
        orch.forward(&input, &mut out2, 1, 2, 0, true).unwrap();
        assert_slices_close(&out1, &out2, ATOL);
    }

    // ===================================================================
    // Output shape tests
    // ===================================================================

    #[test]
    fn test_output_shape_standard_mha() {
        let orch = make_small_orchestrator(4, 4, 4);
        let h = 16;
        let seq = 3;
        let input = vec![0.1; seq * h];
        let mut output = vec![0.0; seq * h];
        orch.forward(&input, &mut output, 1, seq, 0, false).unwrap();
        assert_eq!(output.len(), seq * h);
    }

    #[test]
    fn test_output_shape_gqa() {
        let orch = make_small_orchestrator(4, 2, 4);
        let h = 16;
        let seq = 3;
        let input = vec![0.1; seq * h];
        let mut output = vec![0.0; seq * h];
        orch.forward(&input, &mut output, 1, seq, 0, false).unwrap();
        assert_eq!(output.len(), seq * h);
    }

    #[test]
    fn test_output_shape_mqa() {
        let orch = make_small_orchestrator(4, 1, 4);
        let h = 16;
        let seq = 3;
        let input = vec![0.1; seq * h];
        let mut output = vec![0.0; seq * h];
        orch.forward(&input, &mut output, 1, seq, 0, false).unwrap();
        assert_eq!(output.len(), seq * h);
    }

    #[test]
    fn test_output_shape_multi_batch() {
        let orch = make_small_orchestrator(2, 2, 2);
        let h = 4;
        let seq = 2;
        let batch = 4;
        let input = vec![0.1; batch * seq * h];
        let mut output = vec![0.0; batch * seq * h];
        orch.forward(&input, &mut output, batch, seq, 0, false).unwrap();
        assert_eq!(output.len(), batch * seq * h);
    }

    // ===================================================================
    // OpenCL kernel source validity
    // ===================================================================

    #[test]
    fn test_opencl_source_not_empty() {
        assert!(!MHA_ORCHESTRATOR_CL.is_empty());
    }

    #[test]
    fn test_opencl_source_contains_qkv_project() {
        assert!(MHA_ORCHESTRATOR_CL.contains("qkv_project"));
    }

    #[test]
    fn test_opencl_source_contains_apply_rope() {
        assert!(MHA_ORCHESTRATOR_CL.contains("apply_rope"));
    }

    #[test]
    fn test_opencl_source_contains_sdpa() {
        assert!(MHA_ORCHESTRATOR_CL.contains("sdpa_single_head"));
    }

    // ===================================================================
    // Helper function tests
    // ===================================================================

    #[test]
    fn test_matmul_ref_identity() {
        let a = vec![1.0, 2.0, 3.0, 4.0]; // [2, 2]
        let b = vec![1.0, 0.0, 0.0, 1.0]; // identity
        let mut c = vec![0.0; 4];
        matmul_ref(&a, &b, &mut c, 2, 2, 2);
        assert_slices_close(&c, &a, ATOL);
    }

    #[test]
    fn test_matmul_ref_rectangular() {
        let a = vec![1.0, 2.0, 3.0]; // [1, 3]
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [3, 2]
        let mut c = vec![0.0; 2]; // [1, 2]
        matmul_ref(&a, &b, &mut c, 1, 3, 2);
        // 1*1+2*3+3*5 = 22, 1*2+2*4+3*6 = 28
        assert_slices_close(&c, &[22.0, 28.0], ATOL);
    }

    #[test]
    fn test_add_bias_single_row() {
        let mut data = vec![1.0, 2.0, 3.0];
        add_bias(&mut data, &[10.0, 20.0, 30.0], 1, 3);
        assert_slices_close(&data, &[11.0, 22.0, 33.0], ATOL);
    }

    #[test]
    fn test_add_bias_multi_row() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0]; // [2, 2]
        add_bias(&mut data, &[0.5, 0.5], 2, 2);
        assert_slices_close(&data, &[1.5, 2.5, 3.5, 4.5], ATOL);
    }

    #[test]
    fn test_softmax_row_empty() {
        let mut row: Vec<f32> = vec![];
        softmax_row(&mut row);
        assert!(row.is_empty());
    }

    #[test]
    fn test_softmax_row_single() {
        let mut row = vec![5.0];
        softmax_row(&mut row);
        assert_close(row[0], 1.0, ATOL);
    }

    #[test]
    fn test_softmax_row_all_neg_inf() {
        let mut row = vec![f32::NEG_INFINITY, f32::NEG_INFINITY];
        softmax_row(&mut row);
        // All -inf → output zeros
        assert_close(row[0], 0.0, ATOL);
        assert_close(row[1], 0.0, ATOL);
    }
}
