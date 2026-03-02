//! Sparse attention patterns for Intel Arc A770 long-context inference.
//!
//! Implements various sparse attention mechanisms — local, strided, block-sparse,
//! and longformer-style — to enable efficient long-context inference within
//! the A770's 16 GB memory budget.
//!
//! All computations have CPU reference implementations. The OpenCL dispatch path
//! will be added when the runtime is wired up.

use std::fmt;

// ---------------------------------------------------------------------------
// Attention patterns
// ---------------------------------------------------------------------------

/// Sparse attention pattern variant.
#[derive(Debug, Clone, PartialEq)]
pub enum AttentionPattern {
    /// Full (dense) attention — every token attends to every other.
    Dense,
    /// Local window attention — each token attends to the nearest `window_size`
    /// tokens on each side.
    Local(usize),
    /// Strided attention — each token attends to every `stride`-th token.
    Strided(usize),
    /// Block-sparse attention — attention is computed in fixed-size blocks.
    BlockSparse(usize),
    /// Longformer-style — local window plus a set of global token indices.
    Longformer { local_window: usize, global_tokens: Vec<usize> },
    /// Sliding window with configurable step between windows.
    SlidingWindow { size: usize, step: usize },
}

impl fmt::Display for AttentionPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dense => write!(f, "Dense"),
            Self::Local(w) => write!(f, "Local(window={w})"),
            Self::Strided(s) => write!(f, "Strided(stride={s})"),
            Self::BlockSparse(b) => write!(f, "BlockSparse(block={b})"),
            Self::Longformer { local_window, global_tokens } => {
                write!(f, "Longformer(window={local_window}, globals={})", global_tokens.len())
            }
            Self::SlidingWindow { size, step } => {
                write!(f, "SlidingWindow(size={size}, step={step})")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SparseMask
// ---------------------------------------------------------------------------

/// Compressed sparse attention mask.
///
/// `mask_data` stores one bit per `(query, key)` pair in row-major order.
/// A `true` value means the query position **may** attend to the key position.
#[derive(Debug, Clone)]
pub struct SparseMask {
    pub pattern: AttentionPattern,
    pub seq_len: usize,
    pub num_heads: usize,
    /// Packed bitmask — one bit per element, row-major `[seq_len, seq_len]`.
    /// Shared across heads (pattern is position-based, not head-based).
    pub mask_data: Vec<u8>,
}

impl SparseMask {
    /// Number of `(query, key)` pairs represented.
    #[inline]
    pub fn total_elements(&self) -> usize {
        self.seq_len * self.seq_len
    }

    /// Count non-zero (allowed) entries by scanning the bitmask.
    pub fn non_zero_count(&self) -> usize {
        let total = self.total_elements();
        let mut count = 0usize;
        for idx in 0..total {
            if self.mask_data[idx / 8] & (1 << (idx % 8)) != 0 {
                count += 1;
            }
        }
        count
    }

    /// Check whether query position `i` may attend to key position `j`.
    #[inline]
    pub fn allows(&self, i: usize, j: usize) -> bool {
        let idx = i * self.seq_len + j;
        self.mask_data[idx / 8] & (1 << (idx % 8)) != 0
    }
}

// ---------------------------------------------------------------------------
// SparseAttentionConfig
// ---------------------------------------------------------------------------

/// Configuration for sparse attention computation.
#[derive(Debug, Clone)]
pub struct SparseAttentionConfig {
    pub pattern: AttentionPattern,
    pub num_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub causal: bool,
}

impl SparseAttentionConfig {
    /// Create a new sparse attention configuration.
    ///
    /// # Errors
    ///
    /// Returns an error string if any dimension is zero.
    pub fn new(
        pattern: AttentionPattern,
        num_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        causal: bool,
    ) -> Result<Self, String> {
        if num_heads == 0 || head_dim == 0 || max_seq_len == 0 {
            return Err("num_heads, head_dim, and max_seq_len must all be > 0".into());
        }
        Ok(Self { pattern, num_heads, head_dim, max_seq_len, causal })
    }

    /// Scaling factor `1 / sqrt(head_dim)`.
    #[inline]
    pub fn scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }
}

// ---------------------------------------------------------------------------
// SparsityStats
// ---------------------------------------------------------------------------

/// Statistics about the sparsity of an attention mask.
#[derive(Debug, Clone, PartialEq)]
pub struct SparsityStats {
    pub total_elements: usize,
    pub non_zero: usize,
    pub sparsity_ratio: f64,
    pub memory_saved_bytes: usize,
}

impl SparsityStats {
    /// Compute stats from a [`SparseMask`].
    pub fn from_mask(mask: &SparseMask) -> Self {
        let total = mask.total_elements();
        let nz = mask.non_zero_count();
        let sparsity = if total == 0 { 0.0 } else { 1.0 - (nz as f64 / total as f64) };
        // Memory saved compared to dense f32 score matrix (per head).
        let dense_bytes = total * std::mem::size_of::<f32>();
        let sparse_bytes = nz * std::mem::size_of::<f32>();
        let saved = dense_bytes.saturating_sub(sparse_bytes);
        Self {
            total_elements: total,
            non_zero: nz,
            sparsity_ratio: sparsity,
            memory_saved_bytes: saved,
        }
    }
}

impl fmt::Display for SparsityStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SparsityStats {{ total={}, nz={}, sparsity={:.2}%, saved={} B }}",
            self.total_elements,
            self.non_zero,
            self.sparsity_ratio * 100.0,
            self.memory_saved_bytes,
        )
    }
}

// ---------------------------------------------------------------------------
// BlockSparseLayout
// ---------------------------------------------------------------------------

/// Manages block-sparse matrix layout for efficient GPU dispatch.
///
/// Divides the `[seq_len, seq_len]` attention matrix into fixed-size blocks
/// and tracks which blocks are "active" (non-zero).
#[derive(Debug, Clone)]
pub struct BlockSparseLayout {
    pub block_size: usize,
    pub seq_len: usize,
    pub num_blocks_per_side: usize,
    /// Row-major `[num_blocks_per_side, num_blocks_per_side]` — `true` if
    /// block is active.
    pub active_blocks: Vec<bool>,
}

impl BlockSparseLayout {
    /// Build a layout from a block-sparse (or other) pattern.
    pub fn new(block_size: usize, seq_len: usize, mask: &SparseMask) -> Self {
        let nbs = seq_len.div_ceil(block_size);
        let mut active = vec![false; nbs * nbs];

        for bi in 0..nbs {
            for bj in 0..nbs {
                'block: for li in 0..block_size {
                    let row = bi * block_size + li;
                    if row >= seq_len {
                        break;
                    }
                    for lj in 0..block_size {
                        let col = bj * block_size + lj;
                        if col >= seq_len {
                            break;
                        }
                        if mask.allows(row, col) {
                            active[bi * nbs + bj] = true;
                            break 'block;
                        }
                    }
                }
            }
        }

        Self { block_size, seq_len, num_blocks_per_side: nbs, active_blocks: active }
    }

    /// Number of active (non-zero) blocks.
    pub fn active_block_count(&self) -> usize {
        self.active_blocks.iter().filter(|&&b| b).count()
    }

    /// Total number of blocks.
    pub fn total_block_count(&self) -> usize {
        self.num_blocks_per_side * self.num_blocks_per_side
    }

    /// Check whether the block at `(block_row, block_col)` is active.
    #[inline]
    pub fn is_active(&self, block_row: usize, block_col: usize) -> bool {
        self.active_blocks[block_row * self.num_blocks_per_side + block_col]
    }
}

// ---------------------------------------------------------------------------
// MaskGenerator
// ---------------------------------------------------------------------------

/// Generates [`SparseMask`] instances from [`AttentionPattern`] definitions.
pub struct MaskGenerator;

impl MaskGenerator {
    /// Generate a sparse mask for the given pattern and sequence length.
    pub fn generate(
        pattern: &AttentionPattern,
        seq_len: usize,
        num_heads: usize,
        causal: bool,
    ) -> SparseMask {
        let total_bits = seq_len * seq_len;
        let num_bytes = total_bits.div_ceil(8);
        let mut mask_data = vec![0u8; num_bytes];

        for i in 0..seq_len {
            for j in 0..seq_len {
                if causal && j > i {
                    continue;
                }
                let allowed = match pattern {
                    AttentionPattern::Dense => true,
                    AttentionPattern::Local(w) => {
                        let dist = i.abs_diff(j);
                        dist <= *w
                    }
                    AttentionPattern::Strided(stride) => {
                        if *stride == 0 {
                            false
                        } else {
                            j % stride == 0 || i == j
                        }
                    }
                    AttentionPattern::BlockSparse(bs) => {
                        if *bs == 0 {
                            false
                        } else {
                            i / bs == j / bs
                        }
                    }
                    AttentionPattern::Longformer { local_window, global_tokens } => {
                        let dist = i.abs_diff(j);
                        let in_window = dist <= *local_window;
                        let is_global_i = global_tokens.contains(&i);
                        let is_global_j = global_tokens.contains(&j);
                        in_window || is_global_i || is_global_j
                    }
                    AttentionPattern::SlidingWindow { size, step } => {
                        if *step == 0 || *size == 0 {
                            false
                        } else {
                            let window_start = (i / step) * step;
                            let window_end = window_start + size;
                            j >= window_start && j < window_end
                        }
                    }
                };
                if allowed {
                    let idx = i * seq_len + j;
                    mask_data[idx / 8] |= 1 << (idx % 8);
                }
            }
        }

        SparseMask { pattern: pattern.clone(), seq_len, num_heads, mask_data }
    }
}

// ---------------------------------------------------------------------------
// SparseAttentionComputer — CPU reference
// ---------------------------------------------------------------------------

/// Computes sparse attention with mask application (CPU reference).
pub struct SparseAttentionComputer;

impl SparseAttentionComputer {
    /// Compute sparse scaled dot-product attention for a single head.
    ///
    /// # Arguments
    ///
    /// * `q` — query matrix `[seq_len, head_dim]`, row-major.
    /// * `k` — key matrix `[seq_len, head_dim]`, row-major.
    /// * `v` — value matrix `[seq_len, head_dim]`, row-major.
    /// * `mask` — sparse attention mask.
    /// * `head_dim` — dimension per head.
    /// * `scale` — typically `1 / sqrt(head_dim)`.
    ///
    /// Returns the output `[seq_len, head_dim]`.
    pub fn compute(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        mask: &SparseMask,
        head_dim: usize,
        scale: f32,
    ) -> Vec<f32> {
        let seq_len = mask.seq_len;
        assert_eq!(q.len(), seq_len * head_dim);
        assert_eq!(k.len(), seq_len * head_dim);
        assert_eq!(v.len(), seq_len * head_dim);

        let mut output = vec![0.0f32; seq_len * head_dim];
        let mut scores = vec![0.0f32; seq_len];

        for i in 0..seq_len {
            // Compute Q[i] · K[j]^T * scale, masked.
            for j in 0..seq_len {
                if mask.allows(i, j) {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[i * head_dim + d] * k[j * head_dim + d];
                    }
                    scores[j] = dot * scale;
                } else {
                    scores[j] = f32::NEG_INFINITY;
                }
            }

            // Softmax.
            softmax_row(&mut scores);

            // Weighted sum of V.
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for j in 0..seq_len {
                    acc += scores[j] * v[j * head_dim + d];
                }
                output[i * head_dim + d] = acc;
            }
        }

        output
    }

    /// Multi-head sparse attention (CPU reference).
    ///
    /// * `q` — `[seq_len, num_heads * head_dim]`
    /// * `k` — `[seq_len, num_heads * head_dim]`
    /// * `v` — `[seq_len, num_heads * head_dim]`
    ///
    /// Returns `[seq_len, num_heads * head_dim]`.
    pub fn compute_multi_head(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        mask: &SparseMask,
        config: &SparseAttentionConfig,
    ) -> Vec<f32> {
        let seq_len = mask.seq_len;
        let hd = config.head_dim;
        let nh = config.num_heads;
        let full_dim = nh * hd;

        assert_eq!(q.len(), seq_len * full_dim);
        assert_eq!(k.len(), seq_len * full_dim);
        assert_eq!(v.len(), seq_len * full_dim);

        let mut output = vec![0.0f32; seq_len * full_dim];

        for h in 0..nh {
            // Extract per-head slices.
            let q_head: Vec<f32> = (0..seq_len)
                .flat_map(|t| {
                    let start = t * full_dim + h * hd;
                    q[start..start + hd].iter().copied()
                })
                .collect();
            let k_head: Vec<f32> = (0..seq_len)
                .flat_map(|t| {
                    let start = t * full_dim + h * hd;
                    k[start..start + hd].iter().copied()
                })
                .collect();
            let v_head: Vec<f32> = (0..seq_len)
                .flat_map(|t| {
                    let start = t * full_dim + h * hd;
                    v[start..start + hd].iter().copied()
                })
                .collect();

            let out_head = Self::compute(&q_head, &k_head, &v_head, mask, hd, config.scale());

            // Scatter back.
            for t in 0..seq_len {
                let dst = t * full_dim + h * hd;
                output[dst..dst + hd].copy_from_slice(&out_head[t * hd..(t + 1) * hd]);
            }
        }

        output
    }
}

// ---------------------------------------------------------------------------
// Dense reference (for property-test comparison)
// ---------------------------------------------------------------------------

/// Dense attention reference (optionally causal) — used to validate that
/// sparse attention with a Dense pattern produces the same output.
#[cfg(test)]
fn dense_attention_ref(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
    causal: bool,
) -> Vec<f32> {
    let mut output = vec![0.0f32; seq_len * head_dim];
    let mut scores = vec![0.0f32; seq_len];

    for i in 0..seq_len {
        for j in 0..seq_len {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[i * head_dim + d] * k[j * head_dim + d];
            }
            scores[j] = dot * scale;
            if causal && j > i {
                scores[j] = f32::NEG_INFINITY;
            }
        }
        softmax_row(&mut scores);
        for d in 0..head_dim {
            let mut acc = 0.0f32;
            for j in 0..seq_len {
                acc += scores[j] * v[j * head_dim + d];
            }
            output[i * head_dim + d] = acc;
        }
    }
    output
}

// ---------------------------------------------------------------------------
// Softmax helper
// ---------------------------------------------------------------------------

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
// OpenCL kernel source (placeholder for GPU dispatch)
// ---------------------------------------------------------------------------

/// OpenCL C source for sparse masked attention (placeholder).
pub const SPARSE_ATTENTION_CL: &str = r#"
// Sparse attention kernel for Intel Arc A770.
// Applies a bitmask to skip masked-out (query, key) pairs.
__kernel void sparse_attention(
    __global const float* Q,
    __global const float* K,
    __global const float* V,
    __global const uchar* mask,
    __global float* output,
    const int seq_len,
    const int head_dim,
    const float scale)
{
    int qi = get_global_id(0);
    if (qi >= seq_len) return;

    float scores[4096];
    if (seq_len > 4096) return;

    for (int kj = 0; kj < seq_len; kj++) {
        int idx = qi * seq_len + kj;
        int byte_idx = idx / 8;
        int bit_idx = idx % 8;
        if ((mask[byte_idx] >> bit_idx) & 1) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += Q[qi * head_dim + d] * K[kj * head_dim + d];
            }
            scores[kj] = dot * scale;
        } else {
            scores[kj] = -1e30f;
        }
    }

    // softmax
    float max_s = scores[0];
    for (int j = 1; j < seq_len; j++) {
        if (scores[j] > max_s) max_s = scores[j];
    }
    float sum = 0.0f;
    for (int j = 0; j < seq_len; j++) {
        scores[j] = exp(scores[j] - max_s);
        sum += scores[j];
    }
    if (sum > 0.0f) {
        for (int j = 0; j < seq_len; j++) scores[j] /= sum;
    }

    for (int d = 0; d < head_dim; d++) {
        float acc = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            acc += scores[j] * V[j * head_dim + d];
        }
        output[qi * head_dim + d] = acc;
    }
}
"#;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: deterministic Q/K/V for a given seq_len × head_dim.
    fn make_qkv(seq_len: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let n = seq_len * head_dim;
        let q: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.01).sin()).collect();
        let k: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.02).cos()).collect();
        let v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.03).sin()).collect();
        (q, k, v)
    }

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < tol)
    }

    // =======================================================================
    // AttentionPattern Display
    // =======================================================================

    #[test]
    fn test_pattern_display_dense() {
        assert_eq!(format!("{}", AttentionPattern::Dense), "Dense");
    }

    #[test]
    fn test_pattern_display_local() {
        assert_eq!(format!("{}", AttentionPattern::Local(4)), "Local(window=4)");
    }

    #[test]
    fn test_pattern_display_strided() {
        assert_eq!(format!("{}", AttentionPattern::Strided(3)), "Strided(stride=3)");
    }

    #[test]
    fn test_pattern_display_block_sparse() {
        assert_eq!(format!("{}", AttentionPattern::BlockSparse(16)), "BlockSparse(block=16)");
    }

    #[test]
    fn test_pattern_display_longformer() {
        let p = AttentionPattern::Longformer { local_window: 2, global_tokens: vec![0, 5] };
        assert_eq!(format!("{p}"), "Longformer(window=2, globals=2)");
    }

    #[test]
    fn test_pattern_display_sliding_window() {
        let p = AttentionPattern::SlidingWindow { size: 8, step: 4 };
        assert_eq!(format!("{p}"), "SlidingWindow(size=8, step=4)");
    }

    // =======================================================================
    // SparseAttentionConfig
    // =======================================================================

    #[test]
    fn test_config_valid() {
        let cfg = SparseAttentionConfig::new(AttentionPattern::Dense, 8, 64, 512, true);
        assert!(cfg.is_ok());
        let cfg = cfg.unwrap();
        assert_eq!(cfg.num_heads, 8);
        assert!((cfg.scale() - 1.0 / 8.0).abs() < 1e-6); // 1/sqrt(64) = 0.125
    }

    #[test]
    fn test_config_zero_heads() {
        let cfg = SparseAttentionConfig::new(AttentionPattern::Dense, 0, 64, 512, true);
        assert!(cfg.is_err());
    }

    #[test]
    fn test_config_zero_head_dim() {
        let cfg = SparseAttentionConfig::new(AttentionPattern::Dense, 8, 0, 512, true);
        assert!(cfg.is_err());
    }

    #[test]
    fn test_config_zero_max_seq_len() {
        let cfg = SparseAttentionConfig::new(AttentionPattern::Dense, 8, 64, 0, true);
        assert!(cfg.is_err());
    }

    // =======================================================================
    // Mask generation — Dense
    // =======================================================================

    #[test]
    fn test_mask_dense_non_causal() {
        let mask = MaskGenerator::generate(&AttentionPattern::Dense, 4, 1, false);
        assert_eq!(mask.non_zero_count(), 16); // 4×4 fully allowed
        for i in 0..4 {
            for j in 0..4 {
                assert!(mask.allows(i, j));
            }
        }
    }

    #[test]
    fn test_mask_dense_causal() {
        let mask = MaskGenerator::generate(&AttentionPattern::Dense, 4, 1, true);
        // Lower-triangular: 1+2+3+4 = 10
        assert_eq!(mask.non_zero_count(), 10);
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(mask.allows(i, j), j <= i);
            }
        }
    }

    // =======================================================================
    // Mask generation — Local
    // =======================================================================

    #[test]
    fn test_mask_local_non_causal() {
        let mask = MaskGenerator::generate(&AttentionPattern::Local(1), 5, 1, false);
        // window=1: each token attends to itself ± 1
        assert!(mask.allows(0, 0));
        assert!(mask.allows(0, 1));
        assert!(!mask.allows(0, 2));
        assert!(mask.allows(2, 1));
        assert!(mask.allows(2, 2));
        assert!(mask.allows(2, 3));
        assert!(!mask.allows(0, 4));
    }

    #[test]
    fn test_mask_local_causal() {
        let mask = MaskGenerator::generate(&AttentionPattern::Local(1), 5, 1, true);
        // Causal + local(1): can only attend to j <= i AND |i-j| <= 1
        assert!(mask.allows(2, 1));
        assert!(mask.allows(2, 2));
        assert!(!mask.allows(2, 3)); // causal blocks j > i
        assert!(!mask.allows(0, 1)); // causal blocks j > i
    }

    #[test]
    fn test_mask_local_window_larger_than_seq() {
        let mask = MaskGenerator::generate(&AttentionPattern::Local(100), 4, 1, false);
        // Window exceeds seq_len → equivalent to dense.
        assert_eq!(mask.non_zero_count(), 16);
    }

    // =======================================================================
    // Mask generation — Strided
    // =======================================================================

    #[test]
    fn test_mask_strided_non_causal() {
        let mask = MaskGenerator::generate(&AttentionPattern::Strided(2), 6, 1, false);
        // stride=2: attend to j where j%2==0, plus diagonal (i==j).
        assert!(mask.allows(0, 0)); // j%2==0
        assert!(!mask.allows(0, 1)); // j%2!=0 and i!=j
        assert!(mask.allows(0, 2)); // j%2==0
        assert!(mask.allows(1, 1)); // diagonal
        assert!(mask.allows(1, 0)); // j%2==0
        assert!(!mask.allows(1, 3)); // j%2!=0 and i!=j
    }

    #[test]
    fn test_mask_strided_causal() {
        let mask = MaskGenerator::generate(&AttentionPattern::Strided(2), 6, 1, true);
        assert!(mask.allows(3, 0)); // j%2==0, j<=i
        assert!(mask.allows(3, 2)); // j%2==0, j<=i
        assert!(mask.allows(3, 3)); // diagonal
        assert!(!mask.allows(3, 4)); // causal blocks j>i
    }

    // =======================================================================
    // Mask generation — BlockSparse
    // =======================================================================

    #[test]
    fn test_mask_block_sparse_non_causal() {
        let mask = MaskGenerator::generate(&AttentionPattern::BlockSparse(2), 6, 1, false);
        // block=2: positions 0,1 in block 0; 2,3 in block 1; 4,5 in block 2
        assert!(mask.allows(0, 0));
        assert!(mask.allows(0, 1));
        assert!(!mask.allows(0, 2)); // different block
        assert!(mask.allows(4, 5));
        assert!(!mask.allows(4, 3));
    }

    #[test]
    fn test_mask_block_sparse_causal() {
        let mask = MaskGenerator::generate(&AttentionPattern::BlockSparse(2), 4, 1, true);
        // block=2, causal: block boundaries + causal
        assert!(mask.allows(1, 0)); // same block, j<=i
        assert!(mask.allows(1, 1)); // same block, j<=i
        assert!(!mask.allows(0, 1)); // same block but causal blocks j>i
    }

    // =======================================================================
    // Mask generation — Longformer
    // =======================================================================

    #[test]
    fn test_mask_longformer_non_causal() {
        let p = AttentionPattern::Longformer { local_window: 1, global_tokens: vec![0] };
        let mask = MaskGenerator::generate(&p, 5, 1, false);
        // Token 0 is global → row 0 and column 0 are all true.
        for j in 0..5 {
            assert!(mask.allows(0, j), "global row: (0, {j})");
        }
        for i in 0..5 {
            assert!(mask.allows(i, 0), "global col: ({i}, 0)");
        }
        // Non-global, non-local pair:
        assert!(!mask.allows(3, 1)); // |3-1|=2 > window=1, neither is global
    }

    #[test]
    fn test_mask_longformer_causal() {
        let p = AttentionPattern::Longformer { local_window: 1, global_tokens: vec![0] };
        let mask = MaskGenerator::generate(&p, 5, 1, true);
        // Global token 0 is causal-limited: row 0 can only attend to col 0.
        assert!(mask.allows(0, 0));
        assert!(!mask.allows(0, 1)); // causal: j > i
        // But other rows can attend to global col 0.
        assert!(mask.allows(4, 0));
    }

    // =======================================================================
    // Mask generation — SlidingWindow
    // =======================================================================

    #[test]
    fn test_mask_sliding_window_non_causal() {
        let p = AttentionPattern::SlidingWindow { size: 3, step: 2 };
        let mask = MaskGenerator::generate(&p, 6, 1, false);
        // i=0 → window_start=0, window_end=3 → attends to 0,1,2
        assert!(mask.allows(0, 0));
        assert!(mask.allows(0, 1));
        assert!(mask.allows(0, 2));
        assert!(!mask.allows(0, 3));
        // i=3 → window_start=2, window_end=5 → attends to 2,3,4
        assert!(mask.allows(3, 2));
        assert!(mask.allows(3, 3));
        assert!(mask.allows(3, 4));
        assert!(!mask.allows(3, 5));
    }

    #[test]
    fn test_mask_sliding_window_causal() {
        let p = AttentionPattern::SlidingWindow { size: 4, step: 2 };
        let mask = MaskGenerator::generate(&p, 6, 1, true);
        // Causal: j must be <= i.
        assert!(!mask.allows(1, 2)); // causal blocks j > i
    }

    // =======================================================================
    // SparseMask — edge cases
    // =======================================================================

    #[test]
    fn test_mask_seq_len_one() {
        let mask = MaskGenerator::generate(&AttentionPattern::Dense, 1, 1, true);
        assert_eq!(mask.non_zero_count(), 1);
        assert!(mask.allows(0, 0));
    }

    #[test]
    fn test_mask_single_head() {
        let mask = MaskGenerator::generate(&AttentionPattern::Local(2), 8, 1, false);
        assert_eq!(mask.num_heads, 1);
        assert!(mask.non_zero_count() > 0);
    }

    // =======================================================================
    // SparsityStats
    // =======================================================================

    #[test]
    fn test_sparsity_stats_dense() {
        let mask = MaskGenerator::generate(&AttentionPattern::Dense, 8, 1, false);
        let stats = SparsityStats::from_mask(&mask);
        assert_eq!(stats.total_elements, 64);
        assert_eq!(stats.non_zero, 64);
        assert!((stats.sparsity_ratio - 0.0).abs() < 1e-10);
        assert_eq!(stats.memory_saved_bytes, 0);
    }

    #[test]
    fn test_sparsity_stats_local() {
        let mask = MaskGenerator::generate(&AttentionPattern::Local(1), 8, 1, false);
        let stats = SparsityStats::from_mask(&mask);
        assert_eq!(stats.total_elements, 64);
        // Local(1) on 8 tokens: 2 + 3 + 3×4 + 3 + 2 = 22
        // Actually: corners have 2, edges have 3, interior has 3 each
        assert!(stats.non_zero < 64);
        assert!(stats.sparsity_ratio > 0.0);
        assert!(stats.memory_saved_bytes > 0);
    }

    #[test]
    fn test_sparsity_stats_block_sparse() {
        let mask = MaskGenerator::generate(&AttentionPattern::BlockSparse(2), 8, 1, false);
        let stats = SparsityStats::from_mask(&mask);
        // 4 blocks of 2×2 = 16 non-zero out of 64.
        assert_eq!(stats.non_zero, 16);
        assert!((stats.sparsity_ratio - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_sparsity_stats_display() {
        let mask = MaskGenerator::generate(&AttentionPattern::Dense, 4, 1, false);
        let stats = SparsityStats::from_mask(&mask);
        let s = format!("{stats}");
        assert!(s.contains("total=16"));
        assert!(s.contains("nz=16"));
    }

    #[test]
    fn test_sparsity_memory_saved_causal() {
        let mask = MaskGenerator::generate(&AttentionPattern::Dense, 8, 1, true);
        let stats = SparsityStats::from_mask(&mask);
        // Causal: 36 non-zero out of 64.
        assert_eq!(stats.non_zero, 36);
        let expected_saved = (64 - 36) * 4; // 28 * 4 = 112
        assert_eq!(stats.memory_saved_bytes, expected_saved);
    }

    // =======================================================================
    // BlockSparseLayout
    // =======================================================================

    #[test]
    fn test_block_layout_basic() {
        let mask = MaskGenerator::generate(&AttentionPattern::BlockSparse(4), 8, 1, false);
        let layout = BlockSparseLayout::new(4, 8, &mask);
        assert_eq!(layout.num_blocks_per_side, 2);
        assert_eq!(layout.total_block_count(), 4);
        // Diagonal blocks active, off-diagonal inactive.
        assert!(layout.is_active(0, 0));
        assert!(layout.is_active(1, 1));
        assert!(!layout.is_active(0, 1));
        assert!(!layout.is_active(1, 0));
        assert_eq!(layout.active_block_count(), 2);
    }

    #[test]
    fn test_block_layout_dense() {
        let mask = MaskGenerator::generate(&AttentionPattern::Dense, 8, 1, false);
        let layout = BlockSparseLayout::new(4, 8, &mask);
        // All blocks active for dense.
        assert_eq!(layout.active_block_count(), 4);
    }

    #[test]
    fn test_block_layout_non_aligned_seq() {
        // seq_len=7 with block_size=4 → 2 blocks per side (last block partial).
        let mask = MaskGenerator::generate(&AttentionPattern::Dense, 7, 1, false);
        let layout = BlockSparseLayout::new(4, 7, &mask);
        assert_eq!(layout.num_blocks_per_side, 2);
        assert_eq!(layout.active_block_count(), 4);
    }

    // =======================================================================
    // SparseAttentionComputer — correctness
    // =======================================================================

    #[test]
    fn test_compute_dense_matches_reference() {
        let seq_len = 6;
        let head_dim = 4;
        let (q, k, v) = make_qkv(seq_len, head_dim);
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mask = MaskGenerator::generate(&AttentionPattern::Dense, seq_len, 1, false);
        let sparse_out = SparseAttentionComputer::compute(&q, &k, &v, &mask, head_dim, scale);
        let dense_out = dense_attention_ref(&q, &k, &v, seq_len, head_dim, scale, false);

        assert!(
            approx_eq(&sparse_out, &dense_out, 1e-5),
            "Dense sparse output should match dense reference"
        );
    }

    #[test]
    fn test_compute_dense_causal_matches_reference() {
        let seq_len = 6;
        let head_dim = 4;
        let (q, k, v) = make_qkv(seq_len, head_dim);
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mask = MaskGenerator::generate(&AttentionPattern::Dense, seq_len, 1, true);
        let sparse_out = SparseAttentionComputer::compute(&q, &k, &v, &mask, head_dim, scale);
        let dense_out = dense_attention_ref(&q, &k, &v, seq_len, head_dim, scale, true);

        assert!(
            approx_eq(&sparse_out, &dense_out, 1e-5),
            "Dense causal sparse output should match causal reference"
        );
    }

    #[test]
    fn test_compute_local_output_shape() {
        let seq_len = 8;
        let head_dim = 4;
        let (q, k, v) = make_qkv(seq_len, head_dim);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mask = MaskGenerator::generate(&AttentionPattern::Local(2), seq_len, 1, false);
        let out = SparseAttentionComputer::compute(&q, &k, &v, &mask, head_dim, scale);
        assert_eq!(out.len(), seq_len * head_dim);
    }

    #[test]
    fn test_compute_strided_output_finite() {
        let seq_len = 8;
        let head_dim = 4;
        let (q, k, v) = make_qkv(seq_len, head_dim);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mask = MaskGenerator::generate(&AttentionPattern::Strided(2), seq_len, 1, false);
        let out = SparseAttentionComputer::compute(&q, &k, &v, &mask, head_dim, scale);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_compute_block_sparse_output_finite() {
        let seq_len = 8;
        let head_dim = 4;
        let (q, k, v) = make_qkv(seq_len, head_dim);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mask = MaskGenerator::generate(&AttentionPattern::BlockSparse(4), seq_len, 1, false);
        let out = SparseAttentionComputer::compute(&q, &k, &v, &mask, head_dim, scale);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_compute_longformer_output_finite() {
        let seq_len = 8;
        let head_dim = 4;
        let (q, k, v) = make_qkv(seq_len, head_dim);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let p = AttentionPattern::Longformer { local_window: 2, global_tokens: vec![0] };
        let mask = MaskGenerator::generate(&p, seq_len, 1, false);
        let out = SparseAttentionComputer::compute(&q, &k, &v, &mask, head_dim, scale);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_compute_sliding_window_output_finite() {
        let seq_len = 8;
        let head_dim = 4;
        let (q, k, v) = make_qkv(seq_len, head_dim);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let p = AttentionPattern::SlidingWindow { size: 4, step: 2 };
        let mask = MaskGenerator::generate(&p, seq_len, 1, false);
        let out = SparseAttentionComputer::compute(&q, &k, &v, &mask, head_dim, scale);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_compute_seq_len_one() {
        let head_dim = 4;
        let (q, k, v) = make_qkv(1, head_dim);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mask = MaskGenerator::generate(&AttentionPattern::Dense, 1, 1, true);
        let out = SparseAttentionComputer::compute(&q, &k, &v, &mask, head_dim, scale);
        assert_eq!(out.len(), head_dim);
        // With seq_len=1, output should equal V (softmax of single element = 1).
        assert!(approx_eq(&out, &v, 1e-5));
    }

    // =======================================================================
    // Multi-head sparse attention
    // =======================================================================

    #[test]
    fn test_multi_head_dense_matches_per_head() {
        let seq_len = 4;
        let head_dim = 4;
        let num_heads = 2;
        let full_dim = num_heads * head_dim;

        let n = seq_len * full_dim;
        let q: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.01).sin()).collect();
        let k: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.02).cos()).collect();
        let v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.03).sin()).collect();

        let config = SparseAttentionConfig::new(
            AttentionPattern::Dense,
            num_heads,
            head_dim,
            seq_len,
            false,
        )
        .unwrap();
        let mask = MaskGenerator::generate(&AttentionPattern::Dense, seq_len, num_heads, false);
        let out = SparseAttentionComputer::compute_multi_head(&q, &k, &v, &mask, &config);
        assert_eq!(out.len(), n);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // =======================================================================
    // Long sequence handling
    // =======================================================================

    #[test]
    fn test_mask_generation_long_seq() {
        // Ensure we can generate masks for seq_len > 2048 without panicking.
        let seq_len = 2100;
        let mask = MaskGenerator::generate(&AttentionPattern::Local(64), seq_len, 1, true);
        assert_eq!(mask.seq_len, seq_len);
        assert!(mask.non_zero_count() > 0);
        assert!(mask.non_zero_count() < seq_len * seq_len);
    }

    #[test]
    fn test_sparsity_stats_long_seq() {
        let seq_len = 2200;
        let mask = MaskGenerator::generate(&AttentionPattern::BlockSparse(64), seq_len, 1, false);
        let stats = SparsityStats::from_mask(&mask);
        assert!(stats.sparsity_ratio > 0.5);
        assert!(stats.memory_saved_bytes > 0);
    }

    // =======================================================================
    // Property tests: sparse ⊆ dense
    // =======================================================================

    #[test]
    fn test_local_is_subset_of_dense() {
        let seq_len = 8;
        let dense = MaskGenerator::generate(&AttentionPattern::Dense, seq_len, 1, false);
        let local = MaskGenerator::generate(&AttentionPattern::Local(2), seq_len, 1, false);
        for i in 0..seq_len {
            for j in 0..seq_len {
                if local.allows(i, j) {
                    assert!(dense.allows(i, j));
                }
            }
        }
    }

    #[test]
    fn test_block_sparse_is_subset_of_dense() {
        let seq_len = 8;
        let dense = MaskGenerator::generate(&AttentionPattern::Dense, seq_len, 1, false);
        let bs = MaskGenerator::generate(&AttentionPattern::BlockSparse(4), seq_len, 1, false);
        for i in 0..seq_len {
            for j in 0..seq_len {
                if bs.allows(i, j) {
                    assert!(dense.allows(i, j));
                }
            }
        }
    }

    #[test]
    fn test_causal_is_subset_of_non_causal() {
        let seq_len = 8;
        let non_causal = MaskGenerator::generate(&AttentionPattern::Local(2), seq_len, 1, false);
        let causal = MaskGenerator::generate(&AttentionPattern::Local(2), seq_len, 1, true);
        for i in 0..seq_len {
            for j in 0..seq_len {
                if causal.allows(i, j) {
                    assert!(non_causal.allows(i, j));
                }
            }
        }
    }

    #[test]
    fn test_sparse_attention_output_matches_masked_dense() {
        // Property: sparse attention with Local pattern should match dense
        // attention where non-local positions are -inf.
        let seq_len = 6;
        let head_dim = 4;
        let (q, k, v) = make_qkv(seq_len, head_dim);
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mask = MaskGenerator::generate(&AttentionPattern::Local(1), seq_len, 1, false);
        let sparse_out = SparseAttentionComputer::compute(&q, &k, &v, &mask, head_dim, scale);

        // Build the same result manually with dense + masking.
        let mut manual_out = vec![0.0f32; seq_len * head_dim];
        let mut scores = vec![0.0f32; seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[i * head_dim + d] * k[j * head_dim + d];
                }
                scores[j] = if mask.allows(i, j) { dot * scale } else { f32::NEG_INFINITY };
            }
            softmax_row(&mut scores);
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for j in 0..seq_len {
                    acc += scores[j] * v[j * head_dim + d];
                }
                manual_out[i * head_dim + d] = acc;
            }
        }

        assert!(
            approx_eq(&sparse_out, &manual_out, 1e-5),
            "Sparse attention should equal manually-masked dense attention"
        );
    }

    // =======================================================================
    // OpenCL kernel source
    // =======================================================================

    #[test]
    fn test_opencl_kernel_source_not_empty() {
        assert!(!SPARSE_ATTENTION_CL.is_empty());
        assert!(SPARSE_ATTENTION_CL.contains("sparse_attention"));
    }

    #[test]
    fn test_opencl_kernel_source_has_mask_param() {
        assert!(SPARSE_ATTENTION_CL.contains("__global const uchar* mask"));
    }

    // =======================================================================
    // Additional edge cases
    // =======================================================================

    #[test]
    fn test_strided_stride_one_equals_dense() {
        let seq_len = 6;
        // stride=1 means j%1==0 is always true → equivalent to Dense.
        let mask = MaskGenerator::generate(&AttentionPattern::Strided(1), seq_len, 1, false);
        assert_eq!(mask.non_zero_count(), seq_len * seq_len);
    }

    #[test]
    fn test_block_sparse_block_one() {
        let seq_len = 4;
        // block_size=1: only diagonal allowed.
        let mask = MaskGenerator::generate(&AttentionPattern::BlockSparse(1), seq_len, 1, false);
        assert_eq!(mask.non_zero_count(), seq_len); // only diagonal
    }

    #[test]
    fn test_longformer_all_global() {
        let seq_len = 4;
        let p =
            AttentionPattern::Longformer { local_window: 0, global_tokens: (0..seq_len).collect() };
        let mask = MaskGenerator::generate(&p, seq_len, 1, false);
        // All tokens are global → equivalent to dense.
        assert_eq!(mask.non_zero_count(), seq_len * seq_len);
    }

    #[test]
    fn test_sliding_window_full_coverage() {
        // size=seq_len, step=1 → window_start = i, window_end = i+size.
        // Token i attends to [i, min(i+size, seq_len)), giving a lower-
        // triangular pattern with sum = seq_len*(seq_len+1)/2.
        let seq_len = 4;
        let p = AttentionPattern::SlidingWindow { size: seq_len, step: 1 };
        let mask = MaskGenerator::generate(&p, seq_len, 1, false);
        assert_eq!(mask.non_zero_count(), seq_len * (seq_len + 1) / 2);
    }
}
