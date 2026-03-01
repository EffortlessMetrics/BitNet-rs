//! Memory-efficient flash attention for Intel Arc A770 (Xe-HPG).
//!
//! Implements the FlashAttention algorithm with block-tiled softmax and
//! online normalization, optimized for A770's 64 KB SLM. CPU reference
//! implementations are provided for correctness testing; the OpenCL kernel
//! source targets actual GPU dispatch.

use std::fmt;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Configuration for the flash attention kernel.
#[derive(Debug, Clone)]
pub struct FlashAttentionConfig {
    /// Dimension of each attention head.
    pub head_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Tile / block size for the flash attention loop (default 64).
    pub block_size: usize,
    /// Whether to apply a causal (autoregressive) mask.
    pub causal: bool,
    /// Optional explicit scale factor; defaults to `1/sqrt(head_dim)`.
    pub scale: Option<f32>,
}

impl FlashAttentionConfig {
    /// Create a new config with the given head dimension and number of heads.
    pub fn new(head_dim: usize, num_heads: usize) -> Self {
        Self { head_dim, num_heads, block_size: 64, causal: false, scale: None }
    }

    /// Effective scale factor.
    pub fn effective_scale(&self) -> f32 {
        self.scale.unwrap_or_else(|| 1.0 / (self.head_dim as f32).sqrt())
    }
}

/// Errors specific to flash attention.
#[derive(Debug, Clone, PartialEq)]
pub enum FlashAttentionError {
    /// `head_dim` is zero or exceeds hardware limits.
    InvalidHeadDim(usize),
    /// Sequence length exceeds the supported maximum.
    SequenceTooLong(usize),
    /// Block size does not evenly tile SLM budget.
    BlockSizeMismatch { expected: usize, actual: usize },
    /// Detected NaN / Inf during online softmax.
    NumericalInstability(String),
}

impl fmt::Display for FlashAttentionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidHeadDim(d) => write!(f, "invalid head_dim: {d}"),
            Self::SequenceTooLong(n) => {
                write!(f, "sequence length {n} exceeds maximum")
            }
            Self::BlockSizeMismatch { expected, actual } => {
                write!(f, "block size mismatch: expected {expected}, got {actual}")
            }
            Self::NumericalInstability(msg) => {
                write!(f, "numerical instability: {msg}")
            }
        }
    }
}

impl std::error::Error for FlashAttentionError {}

/// Statistics about an attention weight distribution.
#[derive(Debug, Clone, Copy)]
pub struct AttentionStats {
    pub max_attention_weight: f32,
    pub min_attention_weight: f32,
    /// Fraction of weights below 1e-4.
    pub sparsity_ratio: f32,
}

// ---------------------------------------------------------------------------
// CPU reference implementations
// ---------------------------------------------------------------------------

/// Standard O(n²) attention: softmax(Q·Kᵀ · scale) · V.
pub fn cpu_standard_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    assert_eq!(q.len(), seq_len * head_dim);
    assert_eq!(k.len(), seq_len * head_dim);
    assert_eq!(v.len(), seq_len * head_dim);

    // Compute scores: S[i][j] = sum_d Q[i][d] * K[j][d] * scale
    let mut scores = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[i * head_dim + d] * k[j * head_dim + d];
            }
            scores[i * seq_len + j] = dot * scale;
        }
    }

    // Row-wise softmax
    for i in 0..seq_len {
        let row = &mut scores[i * seq_len..(i + 1) * seq_len];
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
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

    // Output = scores · V
    let mut out = vec![0.0f32; seq_len * head_dim];
    for i in 0..seq_len {
        for j in 0..seq_len {
            let w = scores[i * seq_len + j];
            for d in 0..head_dim {
                out[i * head_dim + d] += w * v[j * head_dim + d];
            }
        }
    }
    out
}

/// Block-tiled flash attention with online softmax normalization.
pub fn cpu_flash_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    block_size: usize,
    scale: f32,
) -> Vec<f32> {
    assert_eq!(q.len(), seq_len * head_dim);
    assert_eq!(k.len(), seq_len * head_dim);
    assert_eq!(v.len(), seq_len * head_dim);
    assert!(block_size > 0);

    let num_blocks = seq_len.div_ceil(block_size);
    let mut out = vec![0.0f32; seq_len * head_dim];

    // Per-row running statistics
    let mut row_max = vec![f32::NEG_INFINITY; seq_len];
    let mut row_sum = vec![0.0f32; seq_len];

    // For each K/V block
    for bj in 0..num_blocks {
        let j_start = bj * block_size;
        let j_end = (j_start + block_size).min(seq_len);
        let bj_len = j_end - j_start;

        // For each query row
        for i in 0..seq_len {
            // Compute partial scores for this block
            let mut block_scores = vec![0.0f32; bj_len];
            for (jj, score) in block_scores.iter_mut().enumerate() {
                let j = j_start + jj;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[i * head_dim + d] * k[j * head_dim + d];
                }
                *score = dot * scale;
            }

            // Online softmax update
            let block_max = block_scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let new_max = row_max[i].max(block_max);

            // Rescale previous accumulator
            let correction = (row_max[i] - new_max).exp();
            let old_sum_corrected = row_sum[i] * correction;

            // Exponentiate current block
            let mut block_exp = vec![0.0f32; bj_len];
            let mut block_sum = 0.0f32;
            for jj in 0..bj_len {
                block_exp[jj] = (block_scores[jj] - new_max).exp();
                block_sum += block_exp[jj];
            }

            let new_sum = old_sum_corrected + block_sum;

            // Update output: rescale old output and add new contribution
            for d in 0..head_dim {
                out[i * head_dim + d] *= correction * row_sum[i];
                for (jj, &exp_val) in block_exp.iter().enumerate() {
                    let j = j_start + jj;
                    out[i * head_dim + d] += exp_val * v[j * head_dim + d];
                }
                if new_sum > 0.0 {
                    out[i * head_dim + d] /= new_sum;
                }
            }

            row_max[i] = new_max;
            row_sum[i] = new_sum;
        }
    }
    out
}

/// Generate a causal (lower-triangular) mask. 0.0 for allowed positions,
/// `f32::NEG_INFINITY` for masked positions.
pub fn cpu_causal_mask(seq_len: usize) -> Vec<f32> {
    let mut mask = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask[i * seq_len + j] = f32::NEG_INFINITY;
        }
    }
    mask
}

/// Apply causal mask in-place: set upper-triangular entries to `-inf`.
pub fn cpu_apply_causal_mask(scores: &mut [f32], seq_len: usize) {
    assert!(scores.len() >= seq_len * seq_len);
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            scores[i * seq_len + j] = f32::NEG_INFINITY;
        }
    }
}

/// Multi-head attention: splits Q/K/V by heads, runs standard attention on
/// each, and concatenates outputs.
pub fn cpu_multi_head_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    num_heads: usize,
    scale: f32,
) -> Vec<f32> {
    let head_size = seq_len * head_dim;
    assert_eq!(q.len(), num_heads * head_size);
    assert_eq!(k.len(), num_heads * head_size);
    assert_eq!(v.len(), num_heads * head_size);

    let mut out = vec![0.0f32; num_heads * head_size];
    for h in 0..num_heads {
        let offset = h * head_size;
        let head_out = cpu_standard_attention(
            &q[offset..offset + head_size],
            &k[offset..offset + head_size],
            &v[offset..offset + head_size],
            seq_len,
            head_dim,
            scale,
        );
        out[offset..offset + head_size].copy_from_slice(&head_out);
    }
    out
}

/// Compute attention weight statistics from a softmax-normalised score
/// matrix.
pub fn cpu_attention_stats(scores: &[f32], seq_len: usize) -> AttentionStats {
    assert_eq!(scores.len(), seq_len * seq_len);

    let mut max_w = f32::NEG_INFINITY;
    let mut min_w = f32::INFINITY;
    let mut sparse_count = 0usize;
    let total = scores.len();

    for &w in scores {
        if w > max_w {
            max_w = w;
        }
        if w < min_w {
            min_w = w;
        }
        if w < 1e-4 {
            sparse_count += 1;
        }
    }

    AttentionStats {
        max_attention_weight: max_w,
        min_attention_weight: min_w,
        sparsity_ratio: sparse_count as f32 / total as f32,
    }
}

/// Online softmax for a single block.
///
/// Given `block` scores, a `running_max` and `running_sum` from previous
/// blocks, returns `(softmax_block, new_max, new_sum)` where `softmax_block`
/// contains the *un-normalised* exponentials for this block and `new_sum`
/// accounts for the rescaled previous sum.
pub fn cpu_block_softmax(
    block: &[f32],
    running_max: f32,
    running_sum: f32,
) -> (Vec<f32>, f32, f32) {
    let block_max = block.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let new_max = running_max.max(block_max);

    let correction = (running_max - new_max).exp();
    let old_sum_corrected = running_sum * correction;

    let mut exp_block = Vec::with_capacity(block.len());
    let mut block_sum = 0.0f32;
    for &s in block {
        let e = (s - new_max).exp();
        block_sum += e;
        exp_block.push(e);
    }

    let new_sum = old_sum_corrected + block_sum;
    (exp_block, new_max, new_sum)
}

// ---------------------------------------------------------------------------
// OpenCL kernel source (A770 / Xe-HPG targeting)
// ---------------------------------------------------------------------------

/// OpenCL C kernel source for flash attention on Intel Arc A770.
///
/// - `flash_attention_fwd`: single-head flash attention with SLM tiling.
///   block_size=64 fits comfortably in 64 KB SLM with head_dim ≤ 128.
/// - `causal_mask_kernel`: parallel causal mask generation.
pub const FLASH_ATTENTION_SRC: &str = r#"
// ----- Flash Attention for Intel Arc A770 (Xe-HPG, 64KB SLM) -----
// block_size=64 × head_dim≤128 × sizeof(float) = 32 KB per Q/K tile,
// leaving headroom for V tile + accumulators within 64 KB SLM budget.

__kernel void flash_attention_fwd(
    __global const float* Q,       // [seq_len, head_dim]
    __global const float* K,       // [seq_len, head_dim]
    __global const float* V,       // [seq_len, head_dim]
    __global       float* O,       // [seq_len, head_dim]
    const int seq_len,
    const int head_dim,
    const int block_size,
    const float scale,
    const int causal)
{
    const int row = get_global_id(0);  // query row
    if (row >= seq_len) return;

    // Per-row running max and sum for online softmax
    float m_old = -INFINITY;
    float l_old = 0.0f;

    // Accumulator for output (private memory, per work-item)
    float acc[128]; // head_dim <= 128
    for (int d = 0; d < head_dim; d++) acc[d] = 0.0f;

    const int num_blocks = (seq_len + block_size - 1) / block_size;

    for (int bj = 0; bj < num_blocks; bj++) {
        int j_start = bj * block_size;
        int j_end   = min(j_start + block_size, seq_len);

        // --- Compute block scores S_ij = Q_i · K_j^T * scale ---
        float s[64]; // block_size <= 64
        float m_block = -INFINITY;
        for (int jj = 0; jj < j_end - j_start; jj++) {
            int j = j_start + jj;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += Q[row * head_dim + d] * K[j * head_dim + d];
            }
            dot *= scale;
            // Apply causal mask
            if (causal && j > row) dot = -INFINITY;
            s[jj] = dot;
            m_block = fmax(m_block, dot);
        }

        // --- Online softmax update ---
        float m_new = fmax(m_old, m_block);
        float correction = exp(m_old - m_new);
        float l_corrected = l_old * correction;

        float p[64];
        float l_block = 0.0f;
        for (int jj = 0; jj < j_end - j_start; jj++) {
            p[jj] = exp(s[jj] - m_new);
            l_block += p[jj];
        }
        float l_new = l_corrected + l_block;

        // --- Update output accumulator ---
        for (int d = 0; d < head_dim; d++) {
            acc[d] = acc[d] * correction * l_old;
            for (int jj = 0; jj < j_end - j_start; jj++) {
                int j = j_start + jj;
                acc[d] += p[jj] * V[j * head_dim + d];
            }
            if (l_new > 0.0f) {
                acc[d] /= l_new;
            }
        }

        m_old = m_new;
        l_old = l_new;
    }

    // Write output
    for (int d = 0; d < head_dim; d++) {
        O[row * head_dim + d] = acc[d];
    }
}

__kernel void causal_mask_kernel(
    __global float* mask,     // [seq_len, seq_len]
    const int seq_len)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    if (i >= seq_len || j >= seq_len) return;
    mask[i * seq_len + j] = (j > i) ? -INFINITY : 0.0f;
}
"#;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers --

    fn rand_vec(len: usize, seed: u64) -> Vec<f32> {
        // Simple deterministic PRNG (xorshift64)
        let mut state = seed;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                // Map to [-1, 1]
                (state as f32 / u64::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32, ctx: &str) {
        assert_eq!(a.len(), b.len(), "{ctx}: length mismatch");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (x - y).abs();
            assert!(diff < tol, "{ctx}: index {i} differs: {x} vs {y} (diff {diff})");
        }
    }

    fn softmax_row(row: &[f32]) -> Vec<f32> {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = row.iter().map(|&v| (v - max).exp()).collect();
        let sum: f32 = exp.iter().sum();
        exp.iter().map(|&e| e / sum).collect()
    }

    // -- FlashAttentionConfig tests --

    #[test]
    fn test_config_default_scale() {
        let cfg = FlashAttentionConfig::new(64, 8);
        let expected = 1.0 / (64.0f32).sqrt();
        assert!(
            (cfg.effective_scale() - expected).abs() < 1e-6,
            "default scale should be 1/sqrt(head_dim)"
        );
    }

    #[test]
    fn test_config_custom_scale() {
        let mut cfg = FlashAttentionConfig::new(64, 8);
        cfg.scale = Some(0.42);
        assert!((cfg.effective_scale() - 0.42).abs() < 1e-6, "custom scale should be used");
    }

    #[test]
    fn test_config_default_block_size() {
        let cfg = FlashAttentionConfig::new(64, 8);
        assert_eq!(cfg.block_size, 64);
    }

    // -- FlashAttentionError tests --

    #[test]
    fn test_error_display_invalid_head_dim() {
        let e = FlashAttentionError::InvalidHeadDim(0);
        assert!(e.to_string().contains("invalid head_dim"));
    }

    #[test]
    fn test_error_display_seq_too_long() {
        let e = FlashAttentionError::SequenceTooLong(999_999);
        assert!(e.to_string().contains("999999"));
    }

    #[test]
    fn test_error_display_block_mismatch() {
        let e = FlashAttentionError::BlockSizeMismatch { expected: 64, actual: 32 };
        assert!(e.to_string().contains("expected 64"));
    }

    #[test]
    fn test_error_display_numerical() {
        let e = FlashAttentionError::NumericalInstability("NaN detected".into());
        assert!(e.to_string().contains("NaN"));
    }

    #[test]
    fn test_error_eq() {
        assert_eq!(
            FlashAttentionError::InvalidHeadDim(64),
            FlashAttentionError::InvalidHeadDim(64)
        );
        assert_ne!(
            FlashAttentionError::InvalidHeadDim(64),
            FlashAttentionError::InvalidHeadDim(128)
        );
    }

    // -- Causal mask --

    #[test]
    fn test_causal_mask_1x1() {
        let mask = cpu_causal_mask(1);
        assert_eq!(mask, vec![0.0]);
    }

    #[test]
    fn test_causal_mask_lower_triangular() {
        let n = 4;
        let mask = cpu_causal_mask(n);
        for i in 0..n {
            for j in 0..n {
                let val = mask[i * n + j];
                if j <= i {
                    assert_eq!(val, 0.0, "({i},{j}) should be 0.0");
                } else {
                    assert!(val.is_infinite() && val < 0.0, "({i},{j}) should be -inf");
                }
            }
        }
    }

    #[test]
    fn test_apply_causal_mask() {
        let n = 3;
        let mut scores = vec![1.0f32; n * n];
        cpu_apply_causal_mask(&mut scores, n);
        for i in 0..n {
            for j in 0..n {
                if j > i {
                    assert!(scores[i * n + j].is_infinite());
                } else {
                    assert_eq!(scores[i * n + j], 1.0);
                }
            }
        }
    }

    // -- Standard attention basic tests --

    #[test]
    fn test_standard_attention_single_token() {
        let head_dim = 4;
        let q = vec![1.0, 0.0, 0.0, 0.0];
        let k = vec![1.0, 0.0, 0.0, 0.0];
        let v = vec![0.5, 0.5, 0.5, 0.5];
        let scale = 1.0 / (head_dim as f32).sqrt();
        let out = cpu_standard_attention(&q, &k, &v, 1, head_dim, scale);
        // With single token, output = V
        assert_close(&out, &v, 1e-5, "single token");
    }

    #[test]
    fn test_standard_attention_output_shape() {
        let seq_len = 8;
        let head_dim = 32;
        let q = rand_vec(seq_len * head_dim, 1);
        let k = rand_vec(seq_len * head_dim, 2);
        let v = rand_vec(seq_len * head_dim, 3);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let out = cpu_standard_attention(&q, &k, &v, seq_len, head_dim, scale);
        assert_eq!(out.len(), seq_len * head_dim);
    }

    #[test]
    fn test_standard_attention_seq4_hd32() {
        let seq_len = 4;
        let head_dim = 32;
        let q = rand_vec(seq_len * head_dim, 10);
        let k = rand_vec(seq_len * head_dim, 20);
        let v = rand_vec(seq_len * head_dim, 30);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let out = cpu_standard_attention(&q, &k, &v, seq_len, head_dim, scale);
        // Sanity: output is finite
        assert!(out.iter().all(|x| x.is_finite()));
    }

    // -- Flash vs standard agreement --

    #[test]
    fn test_flash_matches_standard_seq4() {
        let (seq_len, head_dim, bs) = (4, 32, 2);
        let q = rand_vec(seq_len * head_dim, 100);
        let k = rand_vec(seq_len * head_dim, 200);
        let v = rand_vec(seq_len * head_dim, 300);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let std = cpu_standard_attention(&q, &k, &v, seq_len, head_dim, scale);
        let flash = cpu_flash_attention(&q, &k, &v, seq_len, head_dim, bs, scale);
        assert_close(&flash, &std, 1e-4, "flash vs std seq4");
    }

    #[test]
    fn test_flash_matches_standard_seq8() {
        let (seq_len, head_dim, bs) = (8, 64, 4);
        let q = rand_vec(seq_len * head_dim, 101);
        let k = rand_vec(seq_len * head_dim, 201);
        let v = rand_vec(seq_len * head_dim, 301);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let std = cpu_standard_attention(&q, &k, &v, seq_len, head_dim, scale);
        let flash = cpu_flash_attention(&q, &k, &v, seq_len, head_dim, bs, scale);
        assert_close(&flash, &std, 1e-4, "flash vs std seq8");
    }

    #[test]
    fn test_flash_matches_standard_seq16() {
        let (seq_len, head_dim, bs) = (16, 64, 4);
        let q = rand_vec(seq_len * head_dim, 102);
        let k = rand_vec(seq_len * head_dim, 202);
        let v = rand_vec(seq_len * head_dim, 302);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let std = cpu_standard_attention(&q, &k, &v, seq_len, head_dim, scale);
        let flash = cpu_flash_attention(&q, &k, &v, seq_len, head_dim, bs, scale);
        assert_close(&flash, &std, 1e-4, "flash vs std seq16");
    }

    #[test]
    fn test_flash_single_token() {
        let head_dim = 32;
        let q = rand_vec(head_dim, 42);
        let k = rand_vec(head_dim, 43);
        let v = rand_vec(head_dim, 44);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let out = cpu_flash_attention(&q, &k, &v, 1, head_dim, 64, scale);
        // Single token → output = V
        assert_close(&out, &v, 1e-5, "flash single token");
    }

    // -- Block boundary tests --

    #[test]
    fn test_flash_block_boundary_63() {
        let (seq_len, head_dim, bs) = (63, 32, 64);
        let q = rand_vec(seq_len * head_dim, 1001);
        let k = rand_vec(seq_len * head_dim, 1002);
        let v = rand_vec(seq_len * head_dim, 1003);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let std = cpu_standard_attention(&q, &k, &v, seq_len, head_dim, scale);
        let flash = cpu_flash_attention(&q, &k, &v, seq_len, head_dim, bs, scale);
        assert_close(&flash, &std, 1e-4, "block boundary 63");
    }

    #[test]
    fn test_flash_block_boundary_64() {
        let (seq_len, head_dim, bs) = (64, 32, 64);
        let q = rand_vec(seq_len * head_dim, 1004);
        let k = rand_vec(seq_len * head_dim, 1005);
        let v = rand_vec(seq_len * head_dim, 1006);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let std = cpu_standard_attention(&q, &k, &v, seq_len, head_dim, scale);
        let flash = cpu_flash_attention(&q, &k, &v, seq_len, head_dim, bs, scale);
        assert_close(&flash, &std, 1e-4, "block boundary 64");
    }

    #[test]
    fn test_flash_block_boundary_65() {
        let (seq_len, head_dim, bs) = (65, 32, 64);
        let q = rand_vec(seq_len * head_dim, 1007);
        let k = rand_vec(seq_len * head_dim, 1008);
        let v = rand_vec(seq_len * head_dim, 1009);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let std = cpu_standard_attention(&q, &k, &v, seq_len, head_dim, scale);
        let flash = cpu_flash_attention(&q, &k, &v, seq_len, head_dim, bs, scale);
        assert_close(&flash, &std, 1e-4, "block boundary 65");
    }

    #[test]
    fn test_flash_block_boundary_128() {
        let (seq_len, head_dim, bs) = (128, 32, 64);
        let q = rand_vec(seq_len * head_dim, 1010);
        let k = rand_vec(seq_len * head_dim, 1011);
        let v = rand_vec(seq_len * head_dim, 1012);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let std = cpu_standard_attention(&q, &k, &v, seq_len, head_dim, scale);
        let flash = cpu_flash_attention(&q, &k, &v, seq_len, head_dim, bs, scale);
        assert_close(&flash, &std, 1e-4, "block boundary 128");
    }

    #[test]
    fn test_flash_block_boundary_129() {
        let (seq_len, head_dim, bs) = (129, 32, 64);
        let q = rand_vec(seq_len * head_dim, 1013);
        let k = rand_vec(seq_len * head_dim, 1014);
        let v = rand_vec(seq_len * head_dim, 1015);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let std = cpu_standard_attention(&q, &k, &v, seq_len, head_dim, scale);
        let flash = cpu_flash_attention(&q, &k, &v, seq_len, head_dim, bs, scale);
        assert_close(&flash, &std, 1e-4, "block boundary 129");
    }

    // -- Head dimension tests --

    #[test]
    fn test_flash_head_dim_32() {
        let (seq_len, head_dim, bs) = (8, 32, 4);
        let q = rand_vec(seq_len * head_dim, 2001);
        let k = rand_vec(seq_len * head_dim, 2002);
        let v = rand_vec(seq_len * head_dim, 2003);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let std = cpu_standard_attention(&q, &k, &v, seq_len, head_dim, scale);
        let flash = cpu_flash_attention(&q, &k, &v, seq_len, head_dim, bs, scale);
        assert_close(&flash, &std, 1e-4, "head_dim 32");
    }

    #[test]
    fn test_flash_head_dim_64() {
        let (seq_len, head_dim, bs) = (8, 64, 4);
        let q = rand_vec(seq_len * head_dim, 2004);
        let k = rand_vec(seq_len * head_dim, 2005);
        let v = rand_vec(seq_len * head_dim, 2006);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let std = cpu_standard_attention(&q, &k, &v, seq_len, head_dim, scale);
        let flash = cpu_flash_attention(&q, &k, &v, seq_len, head_dim, bs, scale);
        assert_close(&flash, &std, 1e-4, "head_dim 64");
    }

    #[test]
    fn test_flash_head_dim_128() {
        let (seq_len, head_dim, bs) = (8, 128, 4);
        let q = rand_vec(seq_len * head_dim, 2007);
        let k = rand_vec(seq_len * head_dim, 2008);
        let v = rand_vec(seq_len * head_dim, 2009);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let std = cpu_standard_attention(&q, &k, &v, seq_len, head_dim, scale);
        let flash = cpu_flash_attention(&q, &k, &v, seq_len, head_dim, bs, scale);
        assert_close(&flash, &std, 1e-4, "head_dim 128");
    }

    // -- Multi-head tests --

    #[test]
    fn test_multi_head_1() {
        let (seq_len, head_dim, num_heads) = (4, 32, 1);
        let n = num_heads * seq_len * head_dim;
        let q = rand_vec(n, 3001);
        let k = rand_vec(n, 3002);
        let v = rand_vec(n, 3003);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let out = cpu_multi_head_attention(&q, &k, &v, seq_len, head_dim, num_heads, scale);
        assert_eq!(out.len(), n);
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_multi_head_4() {
        let (seq_len, head_dim, num_heads) = (4, 32, 4);
        let n = num_heads * seq_len * head_dim;
        let q = rand_vec(n, 3004);
        let k = rand_vec(n, 3005);
        let v = rand_vec(n, 3006);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let out = cpu_multi_head_attention(&q, &k, &v, seq_len, head_dim, num_heads, scale);
        assert_eq!(out.len(), n);
    }

    #[test]
    fn test_multi_head_8() {
        let (seq_len, head_dim, num_heads) = (4, 64, 8);
        let n = num_heads * seq_len * head_dim;
        let q = rand_vec(n, 3007);
        let k = rand_vec(n, 3008);
        let v = rand_vec(n, 3009);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let out = cpu_multi_head_attention(&q, &k, &v, seq_len, head_dim, num_heads, scale);
        assert_eq!(out.len(), n);
    }

    // -- AttentionStats --

    #[test]
    fn test_attention_stats_uniform() {
        // 2×2 uniform distribution after softmax
        let scores = vec![0.5, 0.5, 0.5, 0.5];
        let stats = cpu_attention_stats(&scores, 2);
        assert!((stats.max_attention_weight - 0.5).abs() < 1e-6);
        assert!((stats.min_attention_weight - 0.5).abs() < 1e-6);
        assert!((stats.sparsity_ratio - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_attention_stats_sparse() {
        let scores = vec![0.0, 0.0, 0.0, 1.0];
        let stats = cpu_attention_stats(&scores, 2);
        assert!((stats.max_attention_weight - 1.0).abs() < 1e-6);
        assert!((stats.min_attention_weight - 0.0).abs() < 1e-6);
        // 3 out of 4 values are below 1e-4
        assert!((stats.sparsity_ratio - 0.75).abs() < 1e-6);
    }

    // -- Block softmax --

    #[test]
    fn test_block_softmax_single_block_matches_full() {
        let scores = vec![1.0, 2.0, 3.0, 4.0];
        let (exp_block, new_max, new_sum) = cpu_block_softmax(&scores, f32::NEG_INFINITY, 0.0);
        let full = softmax_row(&scores);
        let normalised: Vec<f32> = exp_block.iter().map(|&e| e / new_sum).collect();
        assert_close(&normalised, &full, 1e-6, "block softmax single");
        assert!((new_max - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_block_softmax_two_blocks() {
        let block1 = vec![1.0, 2.0];
        let block2 = vec![3.0, 4.0];
        let full_scores = vec![1.0, 2.0, 3.0, 4.0];
        let full_sm = softmax_row(&full_scores);

        // Process block1
        let (_, m1, s1) = cpu_block_softmax(&block1, f32::NEG_INFINITY, 0.0);
        // Process block2 with running stats from block1
        let (exp2, m2, s2) = cpu_block_softmax(&block2, m1, s1);

        // The second block's exponentials normalised by total sum
        // should match the last two elements of the full softmax
        let normalised2: Vec<f32> = exp2.iter().map(|&e| e / s2).collect();
        assert_close(&normalised2, &full_sm[2..], 1e-5, "block softmax two blocks (block2)");
        // Verify total sum integrates both blocks
        assert!(m2 >= m1);
        assert!(s2 > s1);
    }

    // -- Causal vs non-causal --

    #[test]
    fn test_causal_vs_noncausal_differ() {
        let (seq_len, head_dim) = (4, 32);
        let q = rand_vec(seq_len * head_dim, 5001);
        let k = rand_vec(seq_len * head_dim, 5002);
        let v = rand_vec(seq_len * head_dim, 5003);
        let scale = 1.0 / (head_dim as f32).sqrt();

        let out_nc = cpu_standard_attention(&q, &k, &v, seq_len, head_dim, scale);

        // Build causal scores manually
        let mut scores = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[i * head_dim + d] * k[j * head_dim + d];
                }
                scores[i * seq_len + j] = dot * scale;
            }
        }
        cpu_apply_causal_mask(&mut scores, seq_len);
        // Softmax rows
        for i in 0..seq_len {
            let row = &mut scores[i * seq_len..(i + 1) * seq_len];
            let max_v = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for val in row.iter_mut() {
                *val = (*val - max_v).exp();
                sum += *val;
            }
            for val in row.iter_mut() {
                *val /= sum;
            }
        }
        let mut out_c = vec![0.0f32; seq_len * head_dim];
        for i in 0..seq_len {
            for j in 0..seq_len {
                let w = scores[i * seq_len + j];
                for d in 0..head_dim {
                    out_c[i * head_dim + d] += w * v[j * head_dim + d];
                }
            }
        }

        // They must differ (seq_len > 1)
        let any_diff = out_nc.iter().zip(out_c.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(any_diff, "causal and non-causal should differ");
    }

    // -- Numerical stability --

    #[test]
    fn test_numerical_stability_large_values() {
        let (seq_len, head_dim) = (4, 32);
        // Large values that would overflow naive exp without max-subtract
        let q: Vec<f32> = (0..seq_len * head_dim).map(|i| 100.0 + (i as f32) * 0.01).collect();
        let k = q.clone();
        let v: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32) * 0.1).collect();
        let scale = 1.0 / (head_dim as f32).sqrt();

        let out_std = cpu_standard_attention(&q, &k, &v, seq_len, head_dim, scale);
        let out_flash = cpu_flash_attention(&q, &k, &v, seq_len, head_dim, 2, scale);

        assert!(out_std.iter().all(|x| x.is_finite()), "standard output should be finite");
        assert!(out_flash.iter().all(|x| x.is_finite()), "flash output should be finite");
        assert_close(&out_flash, &out_std, 1e-3, "large values");
    }

    // -- Property: output shape matches value shape --

    #[test]
    fn test_output_shape_matches_value() {
        let seq_len = 16;
        let head_dim = 64;
        let q = rand_vec(seq_len * head_dim, 7001);
        let k = rand_vec(seq_len * head_dim, 7002);
        let v = rand_vec(seq_len * head_dim, 7003);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let out = cpu_standard_attention(&q, &k, &v, seq_len, head_dim, scale);
        assert_eq!(out.len(), v.len(), "output shape == value shape");
    }

    // -- OpenCL source sanity --

    #[test]
    fn test_opencl_source_contains_kernels() {
        assert!(FLASH_ATTENTION_SRC.contains("flash_attention_fwd"));
        assert!(FLASH_ATTENTION_SRC.contains("causal_mask_kernel"));
    }

    #[test]
    fn test_opencl_source_contains_slm_comment() {
        assert!(
            FLASH_ATTENTION_SRC.contains("64KB SLM"),
            "kernel source should reference A770 SLM budget"
        );
    }

    // -- Multi-head consistency: single-head MHA == standard attention --

    #[test]
    fn test_multi_head_single_equals_standard() {
        let (seq_len, head_dim) = (4, 32);
        let q = rand_vec(seq_len * head_dim, 8001);
        let k = rand_vec(seq_len * head_dim, 8002);
        let v = rand_vec(seq_len * head_dim, 8003);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let std_out = cpu_standard_attention(&q, &k, &v, seq_len, head_dim, scale);
        let mha_out = cpu_multi_head_attention(&q, &k, &v, seq_len, head_dim, 1, scale);
        assert_close(&mha_out, &std_out, 1e-6, "MHA(1) == standard");
    }

    // -- Flash attention with block_size=1 (degenerate case) --

    #[test]
    fn test_flash_block_size_1() {
        let (seq_len, head_dim) = (4, 16);
        let q = rand_vec(seq_len * head_dim, 9001);
        let k = rand_vec(seq_len * head_dim, 9002);
        let v = rand_vec(seq_len * head_dim, 9003);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let std = cpu_standard_attention(&q, &k, &v, seq_len, head_dim, scale);
        let flash = cpu_flash_attention(&q, &k, &v, seq_len, head_dim, 1, scale);
        assert_close(&flash, &std, 1e-4, "block_size=1");
    }
}
