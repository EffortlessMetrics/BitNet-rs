//! Advanced attention mechanism variants for GPU-accelerated inference.
//!
//! Provides configurable attention types (MHA, MQA, GQA, Flash, Sliding
//! Window, Linear, Sparse) with RoPE, ALiBi, masking, and KV-head sharing.

use std::fmt;

use crate::HalError;

// ── Attention type enum ──────────────────────────────────────────────────

/// Supported attention mechanism variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttentionType {
    /// Standard multi-head attention (Vaswani et al., 2017).
    MultiHead,
    /// Multi-query attention: all heads share one KV pair (Shazeer, 2019).
    MultiQuery,
    /// Grouped-query attention: KV heads shared across groups (Ainslie+23).
    GroupedQuery,
    /// Memory-efficient tiled attention (Dao et al., 2022).
    FlashAttention,
    /// Local sliding-window attention with optional global tokens.
    SlidingWindow,
    /// Linear-complexity attention via kernel feature maps.
    Linear,
    /// Sparse attention with fixed or learned sparsity patterns.
    Sparse,
}

impl fmt::Display for AttentionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MultiHead => write!(f, "MultiHead"),
            Self::MultiQuery => write!(f, "MultiQuery"),
            Self::GroupedQuery => write!(f, "GroupedQuery"),
            Self::FlashAttention => write!(f, "FlashAttention"),
            Self::SlidingWindow => write!(f, "SlidingWindow"),
            Self::Linear => write!(f, "Linear"),
            Self::Sparse => write!(f, "Sparse"),
        }
    }
}

// ── AttentionEngine ──────────────────────────────────────────────────────

/// Configurable attention engine supporting all seven variants.
#[derive(Debug, Clone)]
pub struct AttentionEngine {
    pub attention_type: AttentionType,
    pub num_heads: usize,
    pub head_dim: usize,
    pub num_kv_heads: usize,
    pub max_seq_len: usize,
}

impl AttentionEngine {
    /// Create a new attention engine, validating configuration.
    pub fn new(
        attention_type: AttentionType,
        num_heads: usize,
        head_dim: usize,
        num_kv_heads: usize,
        max_seq_len: usize,
    ) -> Result<Self, HalError> {
        if num_heads == 0 || head_dim == 0 || num_kv_heads == 0 {
            return Err(HalError::ShapeMismatch { expected: 1, actual: 0 });
        }
        if num_heads % num_kv_heads != 0 {
            return Err(HalError::ShapeMismatch {
                expected: num_kv_heads,
                actual: num_heads % num_kv_heads,
            });
        }
        Ok(Self { attention_type, num_heads, head_dim, num_kv_heads, max_seq_len })
    }

    /// Model dimension (`num_heads * head_dim`).
    pub const fn model_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }

    /// Number of query heads sharing each KV head.
    pub const fn heads_per_kv_group(&self) -> usize {
        if self.num_kv_heads == 0 { 0 } else { self.num_heads / self.num_kv_heads }
    }

    /// Compute memory bytes for KV cache at a given sequence length.
    pub const fn kv_cache_bytes(&self, seq_len: usize) -> usize {
        // 2 (K+V) × kv_heads × seq_len × head_dim × 4 bytes (f32)
        2 * self.num_kv_heads * seq_len * self.head_dim * 4
    }

    /// Estimated FLOPs for a single forward pass at the given seq length.
    pub const fn estimated_flops(&self, seq_len: usize) -> u64 {
        // QK^T: 2 * num_heads * seq_len * seq_len * head_dim
        // AV:   2 * num_heads * seq_len * seq_len * head_dim
        (4 * self.num_heads * seq_len * seq_len * self.head_dim) as u64
    }
}

// ── Multi-head attention ─────────────────────────────────────────────────

/// Standard multi-head attention configuration.
#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    pub num_heads: usize,
    pub head_dim: usize,
    pub dropout: f32,
}

impl MultiHeadAttention {
    pub fn new(num_heads: usize, head_dim: usize) -> Result<Self, HalError> {
        if num_heads == 0 || head_dim == 0 {
            return Err(HalError::ShapeMismatch { expected: 1, actual: 0 });
        }
        Ok(Self { num_heads, head_dim, dropout: 0.0 })
    }

    /// Total model dimension.
    pub const fn model_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }

    /// Compute scaled dot-product attention for one head.
    ///
    /// `q`, `k`, `v` each have length `seq_len * head_dim`.
    /// Returns output of length `seq_len * head_dim`.
    pub fn forward(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        mask: Option<&[f32]>,
    ) -> Result<Vec<f32>, HalError> {
        let expected = seq_len * self.head_dim;
        if q.len() != expected || k.len() != expected || v.len() != expected {
            return Err(HalError::ShapeMismatch { expected, actual: q.len() });
        }
        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0 / (self.head_dim as f32).sqrt();

        let mut output = vec![0.0_f32; expected];
        for i in 0..seq_len {
            let qi = &q[i * self.head_dim..(i + 1) * self.head_dim];
            // compute attention scores for row i
            let mut scores = vec![0.0_f32; seq_len];
            for j in 0..seq_len {
                let kj = &k[j * self.head_dim..(j + 1) * self.head_dim];
                let dot: f32 = qi.iter().zip(kj).map(|(a, b)| a * b).sum();
                scores[j] = dot * scale;
            }
            // apply mask
            if let Some(m) = mask {
                for j in 0..seq_len {
                    scores[j] += m[i * seq_len + j];
                }
            }
            // softmax
            let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0_f32;
            for s in &mut scores {
                *s = (*s - max_s).exp();
                sum += *s;
            }
            if sum > 0.0 {
                for s in &mut scores {
                    *s /= sum;
                }
            }
            // weighted sum of values
            let out_row = &mut output[i * self.head_dim..(i + 1) * self.head_dim];
            for j in 0..seq_len {
                let vj = &v[j * self.head_dim..(j + 1) * self.head_dim];
                for d in 0..self.head_dim {
                    out_row[d] += scores[j] * vj[d];
                }
            }
        }
        Ok(output)
    }
}

// ── Grouped-query attention ──────────────────────────────────────────────

/// GQA configuration with fewer KV heads than query heads.
#[derive(Debug, Clone)]
pub struct GroupedQueryAttention {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl GroupedQueryAttention {
    pub fn new(num_heads: usize, num_kv_heads: usize, head_dim: usize) -> Result<Self, HalError> {
        if num_heads == 0 || num_kv_heads == 0 || head_dim == 0 {
            return Err(HalError::ShapeMismatch { expected: 1, actual: 0 });
        }
        if num_heads % num_kv_heads != 0 {
            return Err(HalError::ShapeMismatch {
                expected: num_kv_heads,
                actual: num_heads % num_kv_heads,
            });
        }
        Ok(Self { num_heads, num_kv_heads, head_dim })
    }

    /// Number of query heads per KV group.
    pub const fn group_size(&self) -> usize {
        if self.num_kv_heads == 0 { 0 } else { self.num_heads / self.num_kv_heads }
    }

    /// Is this effectively multi-query attention (1 KV head)?
    pub const fn is_multi_query(&self) -> bool {
        self.num_kv_heads == 1
    }

    /// Is this effectively standard MHA (kv_heads == heads)?
    pub const fn is_standard_mha(&self) -> bool {
        self.num_kv_heads == self.num_heads
    }

    /// Map a query head index to its KV head index.
    pub const fn kv_head_for_query(&self, query_head: usize) -> usize {
        if self.num_kv_heads == 0 { 0 } else { query_head / (self.num_heads / self.num_kv_heads) }
    }

    /// Memory savings ratio compared to standard MHA.
    pub fn memory_ratio(&self) -> f32 {
        if self.num_heads == 0 {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        {
            self.num_kv_heads as f32 / self.num_heads as f32
        }
    }
}

// ── Flash attention config ───────────────────────────────────────────────

/// Tiling parameters for memory-efficient flash attention.
#[derive(Debug, Clone)]
pub struct FlashAttentionConfig {
    pub block_size_q: usize,
    pub block_size_kv: usize,
    pub num_warps: usize,
    pub use_causal_mask: bool,
    pub softmax_scale: Option<f32>,
}

impl FlashAttentionConfig {
    /// Create with sensible defaults for the given head dimension.
    pub fn for_head_dim(head_dim: usize) -> Self {
        let block = if head_dim <= 64 { 128 } else { 64 };
        Self {
            block_size_q: block,
            block_size_kv: block,
            num_warps: 4,
            use_causal_mask: true,
            softmax_scale: None,
        }
    }

    /// Effective softmax scaling factor.
    pub fn scale(&self, head_dim: usize) -> f32 {
        self.softmax_scale.unwrap_or_else(|| {
            #[allow(clippy::cast_precision_loss)]
            {
                1.0 / (head_dim as f32).sqrt()
            }
        })
    }

    /// Number of Q tiles for a given sequence length.
    pub const fn num_q_tiles(&self, seq_len: usize) -> usize {
        seq_len.div_ceil(self.block_size_q)
    }

    /// Number of KV tiles for a given sequence length.
    pub const fn num_kv_tiles(&self, seq_len: usize) -> usize {
        seq_len.div_ceil(self.block_size_kv)
    }

    /// SRAM bytes required per block tile.
    pub const fn sram_per_tile(&self, head_dim: usize) -> usize {
        // Q tile + K tile + V tile + output accumulator, each f32
        (self.block_size_q * head_dim
            + self.block_size_kv * head_dim
            + self.block_size_kv * head_dim
            + self.block_size_q * head_dim)
            * 4
    }
}

// ── Sliding window config ────────────────────────────────────────────────

/// Sliding-window attention with optional global attention positions.
#[derive(Debug, Clone)]
pub struct SlidingWindowConfig {
    pub window_size: usize,
    pub global_positions: Vec<usize>,
    pub symmetric: bool,
}

impl SlidingWindowConfig {
    pub fn new(window_size: usize) -> Self {
        Self { window_size, global_positions: Vec::new(), symmetric: false }
    }

    /// Add positions that attend to/from every other position.
    pub fn with_global_positions(mut self, positions: Vec<usize>) -> Self {
        self.global_positions = positions;
        self
    }

    /// Whether position `i` can attend to position `j`.
    pub fn can_attend(&self, i: usize, j: usize) -> bool {
        // Global positions attend everywhere and are attended by everyone.
        if self.global_positions.contains(&i) || self.global_positions.contains(&j) {
            return true;
        }
        if self.symmetric {
            let dist = if i >= j { i - j } else { j - i };
            dist <= self.window_size
        } else {
            // Causal: j <= i and within window
            j <= i && (i - j) <= self.window_size
        }
    }

    /// Build a mask for the given sequence length.
    ///
    /// Attended positions are `0.0`, blocked positions `NEG_INFINITY`.
    pub fn build_mask(&self, seq_len: usize) -> Vec<f32> {
        let mut mask = vec![f32::NEG_INFINITY; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                if self.can_attend(i, j) {
                    mask[i * seq_len + j] = 0.0;
                }
            }
        }
        mask
    }

    /// Fraction of positions attended (sparsity metric).
    pub fn density(&self, seq_len: usize) -> f32 {
        if seq_len == 0 {
            return 0.0;
        }
        let attended: usize = (0..seq_len)
            .flat_map(|i| (0..seq_len).map(move |j| (i, j)))
            .filter(|&(i, j)| self.can_attend(i, j))
            .count();
        #[allow(clippy::cast_precision_loss)]
        {
            attended as f32 / (seq_len * seq_len) as f32
        }
    }
}

// ── Attention mask ───────────────────────────────────────────────────────

/// Pre-built attention mask patterns.
#[derive(Debug, Clone, PartialEq)]
pub enum AttentionMask {
    /// Lower-triangular causal mask.
    Causal,
    /// No masking — all positions attend to all others.
    Bidirectional,
    /// User-supplied mask of length `seq_len * seq_len`.
    Custom(Vec<f32>),
}

impl AttentionMask {
    /// Materialise the mask for the given sequence length.
    pub fn build(&self, seq_len: usize) -> Vec<f32> {
        match self {
            Self::Causal => {
                let mut m = vec![0.0_f32; seq_len * seq_len];
                for i in 0..seq_len {
                    for j in (i + 1)..seq_len {
                        m[i * seq_len + j] = f32::NEG_INFINITY;
                    }
                }
                m
            }
            Self::Bidirectional => vec![0.0_f32; seq_len * seq_len],
            Self::Custom(v) => v.clone(),
        }
    }

    /// Whether this is a causal (autoregressive) mask.
    pub const fn is_causal(&self) -> bool {
        matches!(self, Self::Causal)
    }
}

// ── Rotary position embedding ────────────────────────────────────────────

/// Configurable RoPE with base frequency and optional interpolation.
#[derive(Debug, Clone)]
pub struct RotaryPositionEmbedding {
    pub dim: usize,
    pub base: f32,
    pub max_seq_len: usize,
    /// Scale factor for NTK-aware interpolation (1.0 = no interpolation).
    pub scaling_factor: f32,
}

impl RotaryPositionEmbedding {
    pub fn new(dim: usize, max_seq_len: usize) -> Result<Self, HalError> {
        if dim == 0 || dim % 2 != 0 {
            return Err(HalError::ShapeMismatch { expected: 2, actual: dim % 2 });
        }
        Ok(Self { dim, base: 10_000.0, max_seq_len, scaling_factor: 1.0 })
    }

    /// Use a custom base frequency.
    pub const fn with_base(mut self, base: f32) -> Self {
        self.base = base;
        self
    }

    /// Enable NTK-aware interpolation with the given factor.
    pub const fn with_scaling(mut self, factor: f32) -> Self {
        self.scaling_factor = factor;
        self
    }

    /// Compute the inverse-frequency vector of length `dim / 2`.
    pub fn inv_freq(&self) -> Vec<f32> {
        let half = self.dim / 2;
        let effective_base = if self.scaling_factor != 1.0 {
            #[allow(clippy::cast_precision_loss)]
            {
                self.base * self.scaling_factor.powf(self.dim as f32 / (self.dim as f32 - 2.0))
            }
        } else {
            self.base
        };
        (0..half)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                {
                    1.0 / effective_base.powf(2.0 * i as f32 / self.dim as f32)
                }
            })
            .collect()
    }

    /// Build cos/sin tables for positions `0..seq_len`.
    pub fn build_tables(&self, seq_len: usize) -> (Vec<f32>, Vec<f32>) {
        let inv = self.inv_freq();
        let half = self.dim / 2;
        let mut cos_t = Vec::with_capacity(seq_len * half);
        let mut sin_t = Vec::with_capacity(seq_len * half);
        for pos in 0..seq_len {
            for f in &inv {
                #[allow(clippy::cast_precision_loss)]
                let angle = pos as f32 * f;
                cos_t.push(angle.cos());
                sin_t.push(angle.sin());
            }
        }
        (cos_t, sin_t)
    }

    /// Apply RoPE to a vector of length `dim`.
    pub fn apply(&self, x: &[f32], pos: usize) -> Result<Vec<f32>, HalError> {
        if x.len() != self.dim {
            return Err(HalError::ShapeMismatch { expected: self.dim, actual: x.len() });
        }
        let inv = self.inv_freq();
        let half = self.dim / 2;
        let mut out = vec![0.0_f32; self.dim];
        for i in 0..half {
            #[allow(clippy::cast_precision_loss)]
            let angle = pos as f32 * inv[i];
            let (sin_a, cos_a) = angle.sin_cos();
            out[2 * i] = x[2 * i].mul_add(cos_a, -(x[2 * i + 1] * sin_a));
            out[2 * i + 1] = x[2 * i].mul_add(sin_a, x[2 * i + 1] * cos_a);
        }
        Ok(out)
    }
}

// ── ALiBi position bias ──────────────────────────────────────────────────

/// Attention with Linear Biases (Press et al., 2022).
#[derive(Debug, Clone)]
pub struct AlibiPositionBias {
    pub num_heads: usize,
}

impl AlibiPositionBias {
    pub fn new(num_heads: usize) -> Result<Self, HalError> {
        if num_heads == 0 {
            return Err(HalError::ShapeMismatch { expected: 1, actual: 0 });
        }
        Ok(Self { num_heads })
    }

    /// Compute the per-head slope.
    ///
    /// Slopes are powers of `2^{-8/num_heads}` for each head.
    pub fn slopes(&self) -> Vec<f32> {
        let ratio = 2.0_f32.powf(-8.0 / self.num_heads as f32);
        (0..self.num_heads).map(|h| ratio.powi((h + 1) as i32)).collect()
    }

    /// Build bias matrix for one head at a given slope and seq length.
    ///
    /// `bias[i][j] = -slope * |i - j|` (causal: only j <= i).
    pub fn build_bias(&self, head_idx: usize, seq_len: usize) -> Vec<f32> {
        let slopes = self.slopes();
        let slope = slopes[head_idx.min(self.num_heads - 1)];
        let mut bias = vec![f32::NEG_INFINITY; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..=i {
                #[allow(clippy::cast_precision_loss)]
                {
                    bias[i * seq_len + j] = -slope * (i - j) as f32;
                }
            }
        }
        bias
    }

    /// Build bias matrices for all heads.
    pub fn build_all_biases(&self, seq_len: usize) -> Vec<Vec<f32>> {
        (0..self.num_heads).map(|h| self.build_bias(h, seq_len)).collect()
    }
}

// ── KV head sharing ──────────────────────────────────────────────────────

/// KV head sharing patterns for efficient inference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvHeadSharing {
    /// No sharing — each query head has its own KV head (standard MHA).
    None,
    /// All query heads share a single KV head (MQA).
    Full,
    /// Query heads are grouped; each group shares one KV head (GQA).
    Grouped { num_groups: usize },
}

impl KvHeadSharing {
    /// Number of distinct KV heads needed.
    pub const fn num_kv_heads(&self, num_query_heads: usize) -> usize {
        match self {
            Self::None => num_query_heads,
            Self::Full => 1,
            Self::Grouped { num_groups } => *num_groups,
        }
    }

    /// Map a query head to its KV head index.
    pub const fn kv_index(&self, query_head: usize, num_query_heads: usize) -> usize {
        match self {
            Self::None => query_head,
            Self::Full => 0,
            Self::Grouped { num_groups } => {
                if *num_groups == 0 {
                    0
                } else {
                    query_head / (num_query_heads / *num_groups)
                }
            }
        }
    }

    /// Memory saving factor relative to full MHA (0.0–1.0).
    pub fn savings(&self, num_query_heads: usize) -> f32 {
        if num_query_heads == 0 {
            return 0.0;
        }
        let kv = self.num_kv_heads(num_query_heads);
        #[allow(clippy::cast_precision_loss)]
        {
            1.0 - (kv as f32 / num_query_heads as f32)
        }
    }
}

// ── Attention metrics ────────────────────────────────────────────────────

/// Runtime metrics for attention computation.
#[derive(Debug, Clone, Default)]
pub struct AttentionMetrics {
    /// Bytes consumed by the KV cache.
    pub kv_cache_bytes: usize,
    /// Estimated floating-point operations.
    pub flops: u64,
    /// Fraction of attention weights that are effectively zero.
    pub sparsity: f32,
    /// Peak memory usage in bytes.
    pub peak_memory_bytes: usize,
    /// Number of attention heads computed.
    pub heads_computed: usize,
}

impl AttentionMetrics {
    /// Create metrics for a given engine configuration and seq length.
    pub fn for_engine(engine: &AttentionEngine, seq_len: usize) -> Self {
        Self {
            kv_cache_bytes: engine.kv_cache_bytes(seq_len),
            flops: engine.estimated_flops(seq_len),
            sparsity: 0.0,
            peak_memory_bytes: engine.kv_cache_bytes(seq_len)
                + engine.num_heads * seq_len * engine.head_dim * 4,
            heads_computed: engine.num_heads,
        }
    }

    /// Effective compute utilization (0.0–1.0).
    pub fn utilization(&self) -> f32 {
        if self.flops == 0 {
            return 0.0;
        }
        1.0 - self.sparsity
    }
}

impl fmt::Display for AttentionMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "kv_cache={}B flops={} sparsity={:.2}% heads={}",
            self.kv_cache_bytes,
            self.flops,
            self.sparsity * 100.0,
            self.heads_computed,
        )
    }
}

// ══════════════════════════════════════════════════════════════════════════
//  Tests
// ══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── AttentionType tests ──────────────────────────────────────────

    #[test]
    fn attention_type_display() {
        assert_eq!(AttentionType::MultiHead.to_string(), "MultiHead");
        assert_eq!(AttentionType::MultiQuery.to_string(), "MultiQuery");
        assert_eq!(AttentionType::GroupedQuery.to_string(), "GroupedQuery");
        assert_eq!(AttentionType::FlashAttention.to_string(), "FlashAttention");
        assert_eq!(AttentionType::SlidingWindow.to_string(), "SlidingWindow");
        assert_eq!(AttentionType::Linear.to_string(), "Linear");
        assert_eq!(AttentionType::Sparse.to_string(), "Sparse");
    }

    #[test]
    fn attention_type_equality() {
        assert_eq!(AttentionType::MultiHead, AttentionType::MultiHead);
        assert_ne!(AttentionType::MultiHead, AttentionType::MultiQuery);
    }

    #[test]
    fn attention_type_clone() {
        let t = AttentionType::FlashAttention;
        let t2 = t;
        assert_eq!(t, t2);
    }

    // ── AttentionEngine tests ────────────────────────────────────────

    #[test]
    fn engine_new_valid() {
        let e = AttentionEngine::new(AttentionType::MultiHead, 32, 128, 32, 2048).unwrap();
        assert_eq!(e.model_dim(), 32 * 128);
        assert_eq!(e.heads_per_kv_group(), 1);
    }

    #[test]
    fn engine_gqa_config() {
        let e = AttentionEngine::new(AttentionType::GroupedQuery, 32, 128, 8, 2048).unwrap();
        assert_eq!(e.heads_per_kv_group(), 4);
    }

    #[test]
    fn engine_mqa_config() {
        let e = AttentionEngine::new(AttentionType::MultiQuery, 32, 128, 1, 2048).unwrap();
        assert_eq!(e.heads_per_kv_group(), 32);
    }

    #[test]
    fn engine_rejects_zero_heads() {
        assert!(AttentionEngine::new(AttentionType::MultiHead, 0, 128, 1, 1024,).is_err());
    }

    #[test]
    fn engine_rejects_misaligned_kv_heads() {
        // 32 heads but 5 kv_heads: 32 % 5 != 0
        assert!(AttentionEngine::new(AttentionType::GroupedQuery, 32, 128, 5, 1024,).is_err());
    }

    #[test]
    fn engine_kv_cache_bytes() {
        let e = AttentionEngine::new(AttentionType::MultiHead, 8, 64, 8, 512).unwrap();
        // 2 * 8 * 100 * 64 * 4 = 409600
        assert_eq!(e.kv_cache_bytes(100), 409_600);
    }

    #[test]
    fn engine_estimated_flops() {
        let e = AttentionEngine::new(AttentionType::MultiHead, 8, 64, 8, 512).unwrap();
        let flops = e.estimated_flops(10);
        // 4 * 8 * 10 * 10 * 64 = 204800
        assert_eq!(flops, 204_800);
    }

    #[test]
    fn engine_zero_seq_len_flops() {
        let e = AttentionEngine::new(AttentionType::MultiHead, 8, 64, 8, 512).unwrap();
        assert_eq!(e.estimated_flops(0), 0);
    }

    // ── MultiHeadAttention tests ─────────────────────────────────────

    #[test]
    fn mha_new_valid() {
        let mha = MultiHeadAttention::new(8, 64).unwrap();
        assert_eq!(mha.model_dim(), 512);
        assert_eq!(mha.dropout, 0.0);
    }

    #[test]
    fn mha_rejects_zero_heads() {
        assert!(MultiHeadAttention::new(0, 64).is_err());
    }

    #[test]
    fn mha_rejects_zero_dim() {
        assert!(MultiHeadAttention::new(8, 0).is_err());
    }

    #[test]
    fn mha_forward_identity() {
        let mha = MultiHeadAttention::new(1, 4).unwrap();
        let seq_len = 2;
        let q = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let k = q.clone();
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let out = mha.forward(&q, &k, &v, seq_len, None).unwrap();
        assert_eq!(out.len(), seq_len * 4);
    }

    #[test]
    fn mha_forward_with_causal_mask() {
        let mha = MultiHeadAttention::new(1, 2).unwrap();
        let seq_len = 3;
        let n = seq_len * 2;
        let q = vec![1.0; n];
        let k = vec![1.0; n];
        let v = vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5];
        let mask = AttentionMask::Causal.build(seq_len);
        let out = mha.forward(&q, &k, &v, seq_len, Some(&mask)).unwrap();
        // First position can only see itself
        assert!((out[0] - 1.0).abs() < 1e-5);
        assert!((out[1] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn mha_forward_shape_mismatch() {
        let mha = MultiHeadAttention::new(1, 4).unwrap();
        let q = vec![1.0; 8];
        let k = vec![1.0; 4]; // wrong size
        let v = vec![1.0; 8];
        assert!(mha.forward(&q, &k, &v, 2, None).is_err());
    }

    #[test]
    fn mha_forward_single_token() {
        let mha = MultiHeadAttention::new(1, 4).unwrap();
        let q = vec![1.0, 0.0, 0.0, 0.0];
        let k = q.clone();
        let v = vec![0.5, 0.5, 0.5, 0.5];
        let out = mha.forward(&q, &k, &v, 1, None).unwrap();
        // Single token: attention is trivially [1.0], output = v
        for (a, b) in out.iter().zip(v.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    // ── GroupedQueryAttention tests ──────────────────────────────────

    #[test]
    fn gqa_new_valid() {
        let gqa = GroupedQueryAttention::new(32, 8, 128).unwrap();
        assert_eq!(gqa.group_size(), 4);
    }

    #[test]
    fn gqa_rejects_zero() {
        assert!(GroupedQueryAttention::new(0, 8, 128).is_err());
    }

    #[test]
    fn gqa_rejects_misaligned() {
        assert!(GroupedQueryAttention::new(32, 7, 128).is_err());
    }

    #[test]
    fn gqa_is_multi_query() {
        let gqa = GroupedQueryAttention::new(32, 1, 128).unwrap();
        assert!(gqa.is_multi_query());
        assert!(!gqa.is_standard_mha());
    }

    #[test]
    fn gqa_is_standard_mha() {
        let gqa = GroupedQueryAttention::new(32, 32, 128).unwrap();
        assert!(gqa.is_standard_mha());
        assert!(!gqa.is_multi_query());
    }

    #[test]
    fn gqa_kv_head_mapping() {
        let gqa = GroupedQueryAttention::new(8, 2, 64).unwrap();
        // group_size = 4, heads 0..3 → kv0, heads 4..7 → kv1
        assert_eq!(gqa.kv_head_for_query(0), 0);
        assert_eq!(gqa.kv_head_for_query(3), 0);
        assert_eq!(gqa.kv_head_for_query(4), 1);
        assert_eq!(gqa.kv_head_for_query(7), 1);
    }

    #[test]
    fn gqa_memory_ratio() {
        let gqa = GroupedQueryAttention::new(32, 8, 128).unwrap();
        assert!((gqa.memory_ratio() - 0.25).abs() < 1e-6);
    }

    #[test]
    fn gqa_memory_ratio_mqa() {
        let gqa = GroupedQueryAttention::new(32, 1, 128).unwrap();
        assert!((gqa.memory_ratio() - 1.0 / 32.0).abs() < 1e-6);
    }

    // ── FlashAttentionConfig tests ───────────────────────────────────

    #[test]
    fn flash_config_default_small_head() {
        let cfg = FlashAttentionConfig::for_head_dim(64);
        assert_eq!(cfg.block_size_q, 128);
        assert_eq!(cfg.block_size_kv, 128);
        assert!(cfg.use_causal_mask);
    }

    #[test]
    fn flash_config_large_head() {
        let cfg = FlashAttentionConfig::for_head_dim(128);
        assert_eq!(cfg.block_size_q, 64);
    }

    #[test]
    fn flash_config_scale_default() {
        let cfg = FlashAttentionConfig::for_head_dim(64);
        let s = cfg.scale(64);
        assert!((s - 1.0 / 8.0).abs() < 1e-6);
    }

    #[test]
    fn flash_config_custom_scale() {
        let mut cfg = FlashAttentionConfig::for_head_dim(64);
        cfg.softmax_scale = Some(0.5);
        assert!((cfg.scale(64) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn flash_tile_counts() {
        let cfg = FlashAttentionConfig::for_head_dim(64);
        assert_eq!(cfg.num_q_tiles(256), 2);
        assert_eq!(cfg.num_kv_tiles(300), 3); // ceil(300/128)
    }

    #[test]
    fn flash_sram_per_tile() {
        let cfg = FlashAttentionConfig::for_head_dim(64);
        // (128*64 + 128*64 + 128*64 + 128*64) * 4
        let expected = 4 * 128 * 64 * 4;
        assert_eq!(cfg.sram_per_tile(64), expected);
    }

    // ── SlidingWindowConfig tests ────────────────────────────────────

    #[test]
    fn sliding_window_basic() {
        let sw = SlidingWindowConfig::new(2);
        // causal: j <= i and i-j <= 2
        assert!(sw.can_attend(3, 3));
        assert!(sw.can_attend(3, 2));
        assert!(sw.can_attend(3, 1));
        assert!(!sw.can_attend(3, 0)); // distance 3 > window 2
        assert!(!sw.can_attend(1, 3)); // causal: j > i
    }

    #[test]
    fn sliding_window_global_positions() {
        let sw = SlidingWindowConfig::new(1).with_global_positions(vec![0]);
        // Position 0 is global: attends everywhere, attended by everyone
        assert!(sw.can_attend(0, 5));
        assert!(sw.can_attend(5, 0));
        // Non-global still follows window
        assert!(!sw.can_attend(5, 3)); // distance 2 > window 1
    }

    #[test]
    fn sliding_window_symmetric() {
        let mut sw = SlidingWindowConfig::new(2);
        sw.symmetric = true;
        assert!(sw.can_attend(1, 3)); // distance 2 <= window 2
        assert!(!sw.can_attend(0, 3)); // distance 3 > window 2
    }

    #[test]
    fn sliding_window_mask_shape() {
        let sw = SlidingWindowConfig::new(2);
        let mask = sw.build_mask(4);
        assert_eq!(mask.len(), 16);
    }

    #[test]
    fn sliding_window_mask_values() {
        let sw = SlidingWindowConfig::new(1);
        let mask = sw.build_mask(3);
        // Row 0: [0, -inf, -inf]
        assert_eq!(mask[0], 0.0);
        assert_eq!(mask[1], f32::NEG_INFINITY);
        // Row 1: [0, 0, -inf]
        assert_eq!(mask[3], 0.0);
        assert_eq!(mask[4], 0.0);
        assert_eq!(mask[5], f32::NEG_INFINITY);
        // Row 2: [-inf, 0, 0]
        assert_eq!(mask[6], f32::NEG_INFINITY);
        assert_eq!(mask[7], 0.0);
        assert_eq!(mask[8], 0.0);
    }

    #[test]
    fn sliding_window_density() {
        let sw = SlidingWindowConfig::new(100);
        // Large window with small seq: should be fairly dense
        let d = sw.density(4);
        assert!(d > 0.5);
    }

    #[test]
    fn sliding_window_density_empty() {
        let sw = SlidingWindowConfig::new(1);
        assert_eq!(sw.density(0), 0.0);
    }

    // ── AttentionMask tests ──────────────────────────────────────────

    #[test]
    fn mask_causal_build() {
        let mask = AttentionMask::Causal.build(3);
        assert_eq!(mask.len(), 9);
        // Diagonal and below should be 0.0
        assert_eq!(mask[0], 0.0); // (0,0)
        assert_eq!(mask[3], 0.0); // (1,0)
        assert_eq!(mask[4], 0.0); // (1,1)
        // Above diagonal should be -inf
        assert_eq!(mask[1], f32::NEG_INFINITY); // (0,1)
        assert_eq!(mask[2], f32::NEG_INFINITY); // (0,2)
    }

    #[test]
    fn mask_bidirectional_build() {
        let mask = AttentionMask::Bidirectional.build(4);
        assert!(mask.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn mask_custom() {
        let custom = vec![0.0; 4];
        let mask = AttentionMask::Custom(custom.clone());
        assert_eq!(mask.build(2), custom);
    }

    #[test]
    fn mask_is_causal() {
        assert!(AttentionMask::Causal.is_causal());
        assert!(!AttentionMask::Bidirectional.is_causal());
    }

    // ── RotaryPositionEmbedding tests ────────────────────────────────

    #[test]
    fn rope_new_valid() {
        let rope = RotaryPositionEmbedding::new(64, 2048).unwrap();
        assert_eq!(rope.dim, 64);
        assert_eq!(rope.base, 10_000.0);
        assert_eq!(rope.scaling_factor, 1.0);
    }

    #[test]
    fn rope_rejects_odd_dim() {
        assert!(RotaryPositionEmbedding::new(63, 1024).is_err());
    }

    #[test]
    fn rope_rejects_zero_dim() {
        assert!(RotaryPositionEmbedding::new(0, 1024).is_err());
    }

    #[test]
    fn rope_custom_base() {
        let rope = RotaryPositionEmbedding::new(64, 1024).unwrap().with_base(500_000.0);
        assert_eq!(rope.base, 500_000.0);
    }

    #[test]
    fn rope_inv_freq_length() {
        let rope = RotaryPositionEmbedding::new(64, 1024).unwrap();
        assert_eq!(rope.inv_freq().len(), 32);
    }

    #[test]
    fn rope_inv_freq_decreasing() {
        let rope = RotaryPositionEmbedding::new(16, 512).unwrap();
        let inv = rope.inv_freq();
        for w in inv.windows(2) {
            assert!(w[0] > w[1], "inv_freq should be decreasing");
        }
    }

    #[test]
    fn rope_build_tables_shape() {
        let rope = RotaryPositionEmbedding::new(8, 512).unwrap();
        let (cos_t, sin_t) = rope.build_tables(16);
        assert_eq!(cos_t.len(), 16 * 4);
        assert_eq!(sin_t.len(), 16 * 4);
    }

    #[test]
    fn rope_apply_preserves_norm() {
        let rope = RotaryPositionEmbedding::new(4, 512).unwrap();
        let x = vec![1.0, 0.0, 0.0, 1.0];
        let rotated = rope.apply(&x, 0).unwrap();
        let norm_orig: f32 = x.iter().map(|v| v * v).sum();
        let norm_rot: f32 = rotated.iter().map(|v| v * v).sum();
        assert!((norm_orig - norm_rot).abs() < 1e-5, "RoPE should preserve vector norm");
    }

    #[test]
    fn rope_position_zero_is_identity() {
        let rope = RotaryPositionEmbedding::new(4, 512).unwrap();
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let out = rope.apply(&x, 0).unwrap();
        // At position 0, angle = 0 for all freqs → cos=1, sin=0
        for (a, b) in out.iter().zip(x.iter()) {
            assert!((a - b).abs() < 1e-5, "Position 0 should be identity");
        }
    }

    #[test]
    fn rope_apply_wrong_dim() {
        let rope = RotaryPositionEmbedding::new(4, 512).unwrap();
        assert!(rope.apply(&[1.0, 2.0], 0).is_err());
    }

    #[test]
    fn rope_with_scaling() {
        let rope = RotaryPositionEmbedding::new(64, 8192).unwrap().with_scaling(4.0);
        assert_eq!(rope.scaling_factor, 4.0);
        // Scaled inv_freq should differ from unscaled
        let unscaled = RotaryPositionEmbedding::new(64, 8192).unwrap().inv_freq();
        let scaled = rope.inv_freq();
        // Index 0 is base^0 = 1.0 for both; compare a later index.
        assert!((unscaled[1] - scaled[1]).abs() > 1e-8, "Scaling should change frequencies");
    }

    // ── AlibiPositionBias tests ──────────────────────────────────────

    #[test]
    fn alibi_new_valid() {
        let alibi = AlibiPositionBias::new(8).unwrap();
        assert_eq!(alibi.num_heads, 8);
    }

    #[test]
    fn alibi_rejects_zero_heads() {
        assert!(AlibiPositionBias::new(0).is_err());
    }

    #[test]
    fn alibi_slopes_length() {
        let alibi = AlibiPositionBias::new(8).unwrap();
        assert_eq!(alibi.slopes().len(), 8);
    }

    #[test]
    fn alibi_slopes_positive_decreasing() {
        let alibi = AlibiPositionBias::new(8).unwrap();
        let slopes = alibi.slopes();
        for s in &slopes {
            assert!(*s > 0.0);
        }
        for w in slopes.windows(2) {
            assert!(w[0] > w[1], "slopes should decrease");
        }
    }

    #[test]
    fn alibi_bias_diagonal_zero() {
        let alibi = AlibiPositionBias::new(4).unwrap();
        let bias = alibi.build_bias(0, 4);
        // Diagonal entries (i==j): distance 0 → bias = 0
        for i in 0..4 {
            assert!((bias[i * 4 + i] - 0.0).abs() < 1e-6, "Diagonal should be zero");
        }
    }

    #[test]
    fn alibi_bias_negative_off_diagonal() {
        let alibi = AlibiPositionBias::new(4).unwrap();
        let bias = alibi.build_bias(0, 4);
        // Below diagonal: should be negative (distance > 0)
        for i in 1..4 {
            for j in 0..i {
                assert!(bias[i * 4 + j] < 0.0, "Off-diagonal should be negative");
            }
        }
    }

    #[test]
    fn alibi_bias_upper_triangle_neg_inf() {
        let alibi = AlibiPositionBias::new(4).unwrap();
        let bias = alibi.build_bias(0, 4);
        for i in 0..4 {
            for j in (i + 1)..4 {
                assert_eq!(
                    bias[i * 4 + j],
                    f32::NEG_INFINITY,
                    "Upper triangle should be -inf (causal)"
                );
            }
        }
    }

    #[test]
    fn alibi_build_all_biases() {
        let alibi = AlibiPositionBias::new(4).unwrap();
        let all = alibi.build_all_biases(3);
        assert_eq!(all.len(), 4);
        for bias in &all {
            assert_eq!(bias.len(), 9);
        }
    }

    // ── KvHeadSharing tests ──────────────────────────────────────────

    #[test]
    fn kv_sharing_none() {
        let s = KvHeadSharing::None;
        assert_eq!(s.num_kv_heads(32), 32);
        assert_eq!(s.kv_index(5, 32), 5);
    }

    #[test]
    fn kv_sharing_full() {
        let s = KvHeadSharing::Full;
        assert_eq!(s.num_kv_heads(32), 1);
        assert_eq!(s.kv_index(15, 32), 0);
    }

    #[test]
    fn kv_sharing_grouped() {
        let s = KvHeadSharing::Grouped { num_groups: 4 };
        assert_eq!(s.num_kv_heads(32), 4);
        // 32 / 4 = 8 heads per group
        assert_eq!(s.kv_index(0, 32), 0);
        assert_eq!(s.kv_index(7, 32), 0);
        assert_eq!(s.kv_index(8, 32), 1);
        assert_eq!(s.kv_index(31, 32), 3);
    }

    #[test]
    fn kv_sharing_savings_none() {
        let s = KvHeadSharing::None;
        assert!((s.savings(32) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn kv_sharing_savings_full() {
        let s = KvHeadSharing::Full;
        let savings = s.savings(32);
        assert!((savings - (1.0 - 1.0 / 32.0)).abs() < 1e-6);
    }

    #[test]
    fn kv_sharing_savings_grouped() {
        let s = KvHeadSharing::Grouped { num_groups: 8 };
        assert!((s.savings(32) - 0.75).abs() < 1e-6);
    }

    #[test]
    fn kv_sharing_equality() {
        assert_eq!(KvHeadSharing::None, KvHeadSharing::None);
        assert_eq!(KvHeadSharing::Full, KvHeadSharing::Full);
        assert_ne!(KvHeadSharing::None, KvHeadSharing::Full);
    }

    // ── AttentionMetrics tests ───────────────────────────────────────

    #[test]
    fn metrics_for_engine() {
        let e = AttentionEngine::new(AttentionType::MultiHead, 8, 64, 8, 512).unwrap();
        let m = AttentionMetrics::for_engine(&e, 32);
        assert!(m.kv_cache_bytes > 0);
        assert!(m.flops > 0);
        assert_eq!(m.heads_computed, 8);
    }

    #[test]
    fn metrics_utilization_full() {
        let m = AttentionMetrics { flops: 1000, sparsity: 0.0, ..Default::default() };
        assert!((m.utilization() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn metrics_utilization_sparse() {
        let m = AttentionMetrics { flops: 1000, sparsity: 0.3, ..Default::default() };
        assert!((m.utilization() - 0.7).abs() < 1e-6);
    }

    #[test]
    fn metrics_utilization_zero_flops() {
        let m = AttentionMetrics::default();
        assert_eq!(m.utilization(), 0.0);
    }

    #[test]
    fn metrics_display() {
        let m = AttentionMetrics {
            kv_cache_bytes: 1024,
            flops: 999,
            sparsity: 0.5,
            peak_memory_bytes: 2048,
            heads_computed: 4,
        };
        let s = m.to_string();
        assert!(s.contains("kv_cache=1024B"));
        assert!(s.contains("flops=999"));
        assert!(s.contains("50.00%"));
    }

    #[test]
    fn metrics_default() {
        let m = AttentionMetrics::default();
        assert_eq!(m.kv_cache_bytes, 0);
        assert_eq!(m.flops, 0);
        assert_eq!(m.sparsity, 0.0);
    }

    // ── Cross-type integration tests ─────────────────────────────────

    #[test]
    fn engine_type_round_trips() {
        for ty in [
            AttentionType::MultiHead,
            AttentionType::MultiQuery,
            AttentionType::GroupedQuery,
            AttentionType::FlashAttention,
            AttentionType::SlidingWindow,
            AttentionType::Linear,
            AttentionType::Sparse,
        ] {
            let kv = if ty == AttentionType::MultiQuery { 1 } else { 8 };
            let e = AttentionEngine::new(ty, 8, 64, kv, 512).unwrap();
            assert_eq!(e.attention_type, ty);
        }
    }

    #[test]
    fn gqa_matches_kv_sharing_grouped() {
        let gqa = GroupedQueryAttention::new(32, 8, 128).unwrap();
        let sharing = KvHeadSharing::Grouped { num_groups: 8 };
        assert_eq!(gqa.group_size(), 4);
        assert_eq!(sharing.num_kv_heads(32), gqa.num_kv_heads as usize);
    }

    #[test]
    fn mqa_matches_kv_sharing_full() {
        let gqa = GroupedQueryAttention::new(32, 1, 128).unwrap();
        let sharing = KvHeadSharing::Full;
        assert!(gqa.is_multi_query());
        assert_eq!(sharing.num_kv_heads(32), 1);
    }

    #[test]
    fn flash_config_with_engine() {
        let e = AttentionEngine::new(AttentionType::FlashAttention, 8, 64, 8, 2048).unwrap();
        let cfg = FlashAttentionConfig::for_head_dim(e.head_dim);
        assert_eq!(cfg.num_q_tiles(2048), 16);
    }

    #[test]
    fn sliding_window_with_mask() {
        let sw = SlidingWindowConfig::new(2);
        let mask = sw.build_mask(4);
        let attn_mask = AttentionMask::Custom(mask.clone());
        let built = attn_mask.build(4);
        assert_eq!(built, mask);
    }

    #[test]
    fn rope_and_alibi_exclusive() {
        // Both can be constructed for the same model, though only one
        // should be used at a time; this tests they coexist.
        let rope = RotaryPositionEmbedding::new(64, 2048).unwrap();
        let alibi = AlibiPositionBias::new(8).unwrap();
        let (cos_t, _) = rope.build_tables(4);
        let biases = alibi.build_all_biases(4);
        assert_eq!(cos_t.len(), 4 * 32);
        assert_eq!(biases.len(), 8);
    }

    #[test]
    fn metrics_consistency_with_engine() {
        let e = AttentionEngine::new(AttentionType::GroupedQuery, 32, 128, 8, 4096).unwrap();
        let m = AttentionMetrics::for_engine(&e, 256);
        assert_eq!(m.kv_cache_bytes, e.kv_cache_bytes(256));
        assert_eq!(m.flops, e.estimated_flops(256));
    }
}
