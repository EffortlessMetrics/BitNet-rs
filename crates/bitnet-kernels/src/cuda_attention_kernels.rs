//! CUDA-style attention computation kernels with CPU reference implementations.
//!
//! Provides the core transformer attention primitives—QKV projection,
//! scaled dot-product attention, causal masking, ALiBi positional bias,
//! rotary position embeddings (RoPE), and grouped-query attention—as
//! CPU reference implementations suitable for correctness testing and
//! non-GPU environments. The API mirrors typical CUDA attention kernel
//! interfaces so that a future GPU backend can be a drop-in replacement.

use std::fmt;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors specific to attention kernel operations.
#[derive(Debug, Clone, PartialEq)]
pub enum AttentionKernelError {
    /// A tensor dimension does not match the expected shape.
    DimensionMismatch { expected: usize, actual: usize, context: String },
    /// The number of KV heads does not evenly divide query heads.
    InvalidHeadConfig { num_heads: usize, num_kv_heads: usize },
    /// A parameter value is out of range.
    InvalidParameter(String),
    /// Head dimension must be even for rotary embeddings.
    OddHeadDim(usize),
}

impl fmt::Display for AttentionKernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, actual, context } => {
                write!(f, "dimension mismatch in {context}: expected {expected}, got {actual}")
            }
            Self::InvalidHeadConfig { num_heads, num_kv_heads } => {
                write!(
                    f,
                    "num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
                )
            }
            Self::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
            Self::OddHeadDim(d) => write!(f, "head_dim must be even for RoPE, got {d}"),
        }
    }
}

impl std::error::Error for AttentionKernelError {}

/// Convenience result alias.
pub type Result<T> = std::result::Result<T, AttentionKernelError>;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Attention algorithm variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttentionStyle {
    /// Standard scaled dot-product attention (MHA).
    Standard,
    /// FlashAttention-v2 tiled algorithm.
    FlashV2,
    /// Grouped-query attention with a specified number of KV heads.
    GroupedQuery { num_kv_heads: usize },
    /// Multi-query attention (single KV head shared by all query heads).
    MultiQuery,
}

impl fmt::Display for AttentionStyle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Standard => write!(f, "Standard"),
            Self::FlashV2 => write!(f, "FlashV2"),
            Self::GroupedQuery { num_kv_heads } => write!(f, "GQA(kv={num_kv_heads})"),
            Self::MultiQuery => write!(f, "MQA"),
        }
    }
}

/// Parameters for an attention computation.
#[derive(Debug, Clone)]
pub struct AttentionParams {
    pub num_heads: usize,
    pub head_dim: usize,
    pub seq_len: usize,
    pub kv_seq_len: usize,
    pub scale: f32,
    pub causal: bool,
    pub dropout_rate: f32,
}

impl AttentionParams {
    /// Create params with sensible defaults (no dropout, not causal).
    pub fn new(num_heads: usize, head_dim: usize, seq_len: usize) -> Self {
        Self {
            num_heads,
            head_dim,
            seq_len,
            kv_seq_len: seq_len,
            scale: 1.0 / (head_dim as f32).sqrt(),
            causal: false,
            dropout_rate: 0.0,
        }
    }

    /// Builder-style setter for causal masking.
    pub fn with_causal(mut self, causal: bool) -> Self {
        self.causal = causal;
        self
    }

    /// Builder-style setter for KV sequence length (cross-attention).
    pub fn with_kv_seq_len(mut self, kv_seq_len: usize) -> Self {
        self.kv_seq_len = kv_seq_len;
        self
    }
}

/// Output of an attention computation.
#[derive(Debug, Clone)]
pub struct AttentionOutput {
    /// Attention result tensor, shape `[num_heads * seq_len * head_dim]`.
    pub output: Vec<f32>,
    /// Optional attention weight matrix (when requested for diagnostics).
    pub attention_weights: Option<Vec<f32>>,
}

/// Summary statistics for an attention weight matrix.
#[derive(Debug, Clone)]
pub struct AttentionStats {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub entropy: f32,
}

// ---------------------------------------------------------------------------
// QKV projection
// ---------------------------------------------------------------------------

/// Project `input` through weight matrices to produce Q, K, V tensors.
///
/// * `input` — `[seq_len * model_dim]`
/// * `wq`, `wk`, `wv` — `[model_dim * (num_heads * head_dim)]`
///
/// Returns `(Q, K, V)` each of shape `[seq_len * num_heads * head_dim]`.
pub fn compute_qkv_projection(
    input: &[f32],
    wq: &[f32],
    wk: &[f32],
    wv: &[f32],
    params: &AttentionParams,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let model_dim = params.num_heads * params.head_dim;
    let expected_input = params.seq_len * model_dim;
    if input.len() != expected_input {
        return Err(AttentionKernelError::DimensionMismatch {
            expected: expected_input,
            actual: input.len(),
            context: "input".into(),
        });
    }
    let expected_w = model_dim * model_dim;
    for (name, w) in [("wq", wq), ("wk", wk), ("wv", wv)] {
        if w.len() != expected_w {
            return Err(AttentionKernelError::DimensionMismatch {
                expected: expected_w,
                actual: w.len(),
                context: name.into(),
            });
        }
    }

    let project = |w: &[f32]| -> Vec<f32> {
        let mut out = vec![0.0f32; params.seq_len * model_dim];
        for s in 0..params.seq_len {
            let in_row = &input[s * model_dim..(s + 1) * model_dim];
            let out_row = &mut out[s * model_dim..(s + 1) * model_dim];
            for j in 0..model_dim {
                let mut acc = 0.0f32;
                for i in 0..model_dim {
                    acc += in_row[i] * w[i * model_dim + j];
                }
                out_row[j] = acc;
            }
        }
        out
    };

    Ok((project(wq), project(wk), project(wv)))
}

// ---------------------------------------------------------------------------
// Scaled dot-product attention
// ---------------------------------------------------------------------------

/// Compute `softmax(Q @ K^T * scale) @ V` per head.
///
/// * `q` — `[num_heads * seq_len * head_dim]`
/// * `k` — `[num_heads * kv_seq_len * head_dim]`
/// * `v` — `[num_heads * kv_seq_len * head_dim]`
pub fn scaled_dot_product_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    params: &AttentionParams,
) -> Result<AttentionOutput> {
    let q_expected = params.num_heads * params.seq_len * params.head_dim;
    if q.len() != q_expected {
        return Err(AttentionKernelError::DimensionMismatch {
            expected: q_expected,
            actual: q.len(),
            context: "Q".into(),
        });
    }
    let kv_expected = params.num_heads * params.kv_seq_len * params.head_dim;
    if k.len() != kv_expected {
        return Err(AttentionKernelError::DimensionMismatch {
            expected: kv_expected,
            actual: k.len(),
            context: "K".into(),
        });
    }
    if v.len() != kv_expected {
        return Err(AttentionKernelError::DimensionMismatch {
            expected: kv_expected,
            actual: v.len(),
            context: "V".into(),
        });
    }

    let h = params.num_heads;
    let s = params.seq_len;
    let kv_s = params.kv_seq_len;
    let d = params.head_dim;

    let mut output = vec![0.0f32; h * s * d];
    let mut all_weights = Vec::with_capacity(h * s * kv_s);

    for head in 0..h {
        let q_off = head * s * d;
        let k_off = head * kv_s * d;

        // scores = Q @ K^T  → [s, kv_s]
        let mut scores = vec![0.0f32; s * kv_s];
        for i in 0..s {
            for j in 0..kv_s {
                let mut dot = 0.0f32;
                for dd in 0..d {
                    dot += q[q_off + i * d + dd] * k[k_off + j * d + dd];
                }
                scores[i * kv_s + j] = dot * params.scale;
            }
        }

        // Causal mask
        if params.causal {
            scores = apply_causal_mask(&scores, s);
        }

        // Row-wise softmax
        for i in 0..s {
            let row = &mut scores[i * kv_s..(i + 1) * kv_s];
            softmax_inplace(row);
        }

        all_weights.extend_from_slice(&scores);

        // output = weights @ V
        let v_off = head * kv_s * d;
        for i in 0..s {
            for dd in 0..d {
                let mut acc = 0.0f32;
                for j in 0..kv_s {
                    acc += scores[i * kv_s + j] * v[v_off + j * d + dd];
                }
                output[head * s * d + i * d + dd] = acc;
            }
        }
    }

    Ok(AttentionOutput { output, attention_weights: Some(all_weights) })
}

// ---------------------------------------------------------------------------
// Causal mask
// ---------------------------------------------------------------------------

/// Apply a lower-triangular causal mask, setting future positions to `-inf`.
///
/// `scores` is row-major `[seq_len, seq_len]` (self-attention only).
pub fn apply_causal_mask(scores: &[f32], seq_len: usize) -> Vec<f32> {
    let mut masked = scores.to_vec();
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            masked[i * seq_len + j] = f32::NEG_INFINITY;
        }
    }
    masked
}

// ---------------------------------------------------------------------------
// ALiBi positional bias
// ---------------------------------------------------------------------------

/// Apply Attention with Linear Biases (ALiBi) to pre-softmax scores.
///
/// `scores` is `[num_heads * seq_len * seq_len]`.
pub fn apply_alibi_bias(scores: &mut [f32], num_heads: usize, seq_len: usize) {
    for h in 0..num_heads {
        let slope = alibi_slope(h, num_heads);
        let off = h * seq_len * seq_len;
        for i in 0..seq_len {
            for j in 0..seq_len {
                let distance = (j as f32) - (i as f32);
                scores[off + i * seq_len + j] += slope * distance;
            }
        }
    }
}

/// Compute the ALiBi slope for head index `h` among `num_heads` heads.
fn alibi_slope(h: usize, num_heads: usize) -> f32 {
    let ratio = 2.0f64.powf(-(8.0 / num_heads as f64));
    ratio.powi(h as i32 + 1) as f32
}

// ---------------------------------------------------------------------------
// Rotary position embedding (RoPE)
// ---------------------------------------------------------------------------

/// Apply rotary position embeddings in-place to Q and K.
///
/// * `q`, `k` — `[seq_len * head_dim]` for a single head.
/// * `positions` — position index for each token in the sequence.
/// * `head_dim` — must be even.
pub fn rotary_position_embedding(
    q: &mut [f32],
    k: &mut [f32],
    positions: &[usize],
    head_dim: usize,
) -> Result<()> {
    if !head_dim.is_multiple_of(2) {
        return Err(AttentionKernelError::OddHeadDim(head_dim));
    }
    let half = head_dim / 2;
    for (idx, &pos) in positions.iter().enumerate() {
        let base = idx * head_dim;
        for i in 0..half {
            let theta = (pos as f32) / 10000.0f32.powf(2.0 * i as f32 / head_dim as f32);
            let cos_t = theta.cos();
            let sin_t = theta.sin();

            // Q
            let q0 = q[base + i];
            let q1 = q[base + half + i];
            q[base + i] = q0 * cos_t - q1 * sin_t;
            q[base + half + i] = q0 * sin_t + q1 * cos_t;

            // K
            let k0 = k[base + i];
            let k1 = k[base + half + i];
            k[base + i] = k0 * cos_t - k1 * sin_t;
            k[base + half + i] = k0 * sin_t + k1 * cos_t;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Grouped-query attention
// ---------------------------------------------------------------------------

/// Grouped-query (or multi-query) attention.
///
/// * `q` — `[num_heads * seq_len * head_dim]`
/// * `k` — `[num_kv_heads * kv_seq_len * head_dim]`
/// * `v` — `[num_kv_heads * kv_seq_len * head_dim]`
/// * `params.num_heads` — number of query heads.
/// * `style` — must be `GroupedQuery { num_kv_heads }` or `MultiQuery`.
pub fn grouped_query_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    params: &AttentionParams,
    style: AttentionStyle,
) -> Result<AttentionOutput> {
    let num_kv_heads = match style {
        AttentionStyle::GroupedQuery { num_kv_heads } => num_kv_heads,
        AttentionStyle::MultiQuery => 1,
        _ => {
            return Err(AttentionKernelError::InvalidParameter(
                "grouped_query_attention requires GroupedQuery or MultiQuery style".into(),
            ));
        }
    };

    if !params.num_heads.is_multiple_of(num_kv_heads) {
        return Err(AttentionKernelError::InvalidHeadConfig {
            num_heads: params.num_heads,
            num_kv_heads,
        });
    }

    let group_size = params.num_heads / num_kv_heads;
    let s = params.seq_len;
    let kv_s = params.kv_seq_len;
    let d = params.head_dim;

    let q_expected = params.num_heads * s * d;
    if q.len() != q_expected {
        return Err(AttentionKernelError::DimensionMismatch {
            expected: q_expected,
            actual: q.len(),
            context: "Q (GQA)".into(),
        });
    }
    let kv_expected = num_kv_heads * kv_s * d;
    if k.len() != kv_expected {
        return Err(AttentionKernelError::DimensionMismatch {
            expected: kv_expected,
            actual: k.len(),
            context: "K (GQA)".into(),
        });
    }
    if v.len() != kv_expected {
        return Err(AttentionKernelError::DimensionMismatch {
            expected: kv_expected,
            actual: v.len(),
            context: "V (GQA)".into(),
        });
    }

    let mut output = vec![0.0f32; params.num_heads * s * d];
    let mut all_weights = Vec::with_capacity(params.num_heads * s * kv_s);

    for head in 0..params.num_heads {
        let kv_head = head / group_size;
        let q_off = head * s * d;
        let k_off = kv_head * kv_s * d;

        let mut scores = vec![0.0f32; s * kv_s];
        for i in 0..s {
            for j in 0..kv_s {
                let mut dot = 0.0f32;
                for dd in 0..d {
                    dot += q[q_off + i * d + dd] * k[k_off + j * d + dd];
                }
                scores[i * kv_s + j] = dot * params.scale;
            }
        }

        if params.causal {
            for i in 0..s {
                for j in (i + 1)..kv_s {
                    scores[i * kv_s + j] = f32::NEG_INFINITY;
                }
            }
        }

        for i in 0..s {
            softmax_inplace(&mut scores[i * kv_s..(i + 1) * kv_s]);
        }

        all_weights.extend_from_slice(&scores);

        let v_off = kv_head * kv_s * d;
        for i in 0..s {
            for dd in 0..d {
                let mut acc = 0.0f32;
                for j in 0..kv_s {
                    acc += scores[i * kv_s + j] * v[v_off + j * d + dd];
                }
                output[head * s * d + i * d + dd] = acc;
            }
        }
    }

    Ok(AttentionOutput { output, attention_weights: Some(all_weights) })
}

// ---------------------------------------------------------------------------
// Head splitting / merging utilities
// ---------------------------------------------------------------------------

/// Reshape `[seq_len * (num_heads * head_dim)]` → `[num_heads * seq_len * head_dim]`.
pub fn split_heads(input: &[f32], num_heads: usize, head_dim: usize) -> Result<Vec<f32>> {
    let model_dim = num_heads * head_dim;
    if !input.len().is_multiple_of(model_dim) {
        return Err(AttentionKernelError::DimensionMismatch {
            expected: model_dim,
            actual: input.len() % model_dim,
            context: "split_heads: input not divisible by model_dim".into(),
        });
    }
    let seq_len = input.len() / model_dim;
    let mut out = vec![0.0f32; input.len()];
    for s in 0..seq_len {
        for h in 0..num_heads {
            for d in 0..head_dim {
                out[h * seq_len * head_dim + s * head_dim + d] =
                    input[s * model_dim + h * head_dim + d];
            }
        }
    }
    Ok(out)
}

/// Reshape `[num_heads * seq_len * head_dim]` → `[seq_len * (num_heads * head_dim)]`.
pub fn merge_heads(input: &[f32], num_heads: usize, head_dim: usize) -> Result<Vec<f32>> {
    let model_dim = num_heads * head_dim;
    if !input.len().is_multiple_of(model_dim) {
        return Err(AttentionKernelError::DimensionMismatch {
            expected: model_dim,
            actual: input.len() % model_dim,
            context: "merge_heads: input not divisible by model_dim".into(),
        });
    }
    let seq_len = input.len() / model_dim;
    let mut out = vec![0.0f32; input.len()];
    for s in 0..seq_len {
        for h in 0..num_heads {
            for d in 0..head_dim {
                out[s * model_dim + h * head_dim + d] =
                    input[h * seq_len * head_dim + s * head_dim + d];
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Attention statistics
// ---------------------------------------------------------------------------

/// Compute summary statistics for an attention weight matrix.
pub fn compute_attention_stats(weights: &[f32]) -> AttentionStats {
    if weights.is_empty() {
        return AttentionStats { min: 0.0, max: 0.0, mean: 0.0, entropy: 0.0 };
    }
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut entropy = 0.0f64;

    for &w in weights {
        if w < min {
            min = w;
        }
        if w > max {
            max = w;
        }
        sum += w as f64;
        if w > 0.0 {
            entropy -= (w as f64) * (w as f64).ln();
        }
    }

    AttentionStats { min, max, mean: (sum / weights.len() as f64) as f32, entropy: entropy as f32 }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Numerically stable in-place softmax over a slice.
fn softmax_inplace(row: &mut [f32]) {
    if row.is_empty() {
        return;
    }
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

// ===========================================================================
// Tests
// ===========================================================================
#[cfg(test)]
mod tests {
    use super::*;

    // Helper: identity-like weight matrix (eye)
    fn eye(dim: usize) -> Vec<f32> {
        let mut m = vec![0.0f32; dim * dim];
        for i in 0..dim {
            m[i * dim + i] = 1.0;
        }
        m
    }

    // Helper: constant vector
    fn constant(len: usize, val: f32) -> Vec<f32> {
        vec![val; len]
    }

    // ------------------------------------------------------------------
    // AttentionStyle Display
    // ------------------------------------------------------------------
    #[test]
    fn test_attention_style_display() {
        assert_eq!(AttentionStyle::Standard.to_string(), "Standard");
        assert_eq!(AttentionStyle::FlashV2.to_string(), "FlashV2");
        assert_eq!(AttentionStyle::GroupedQuery { num_kv_heads: 4 }.to_string(), "GQA(kv=4)");
        assert_eq!(AttentionStyle::MultiQuery.to_string(), "MQA");
    }

    // ------------------------------------------------------------------
    // AttentionParams builder
    // ------------------------------------------------------------------
    #[test]
    fn test_params_defaults() {
        let p = AttentionParams::new(8, 64, 10);
        assert_eq!(p.num_heads, 8);
        assert_eq!(p.head_dim, 64);
        assert_eq!(p.seq_len, 10);
        assert_eq!(p.kv_seq_len, 10);
        assert!(!p.causal);
        assert!((p.scale - 1.0 / 8.0).abs() < 1e-5); // 1/sqrt(64)
    }

    #[test]
    fn test_params_builder_causal() {
        let p = AttentionParams::new(4, 16, 5).with_causal(true);
        assert!(p.causal);
    }

    #[test]
    fn test_params_builder_kv_seq_len() {
        let p = AttentionParams::new(4, 16, 5).with_kv_seq_len(20);
        assert_eq!(p.kv_seq_len, 20);
    }

    // ------------------------------------------------------------------
    // QKV projection
    // ------------------------------------------------------------------
    #[test]
    fn test_qkv_identity_projection() {
        let params = AttentionParams::new(1, 4, 2);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2×4
        let w = eye(4);
        let (q, k, v) = compute_qkv_projection(&input, &w, &w, &w, &params).unwrap();
        assert_eq!(q, input);
        assert_eq!(k, input);
        assert_eq!(v, input);
    }

    #[test]
    fn test_qkv_projection_dim_mismatch_input() {
        let params = AttentionParams::new(1, 4, 2);
        let input = vec![1.0; 5]; // wrong size
        let w = eye(4);
        let err = compute_qkv_projection(&input, &w, &w, &w, &params).unwrap_err();
        assert!(matches!(err, AttentionKernelError::DimensionMismatch { .. }));
    }

    #[test]
    fn test_qkv_projection_dim_mismatch_weight() {
        let params = AttentionParams::new(1, 4, 2);
        let input = vec![1.0; 8];
        let w_ok = eye(4);
        let w_bad = vec![1.0; 10];
        let err = compute_qkv_projection(&input, &w_bad, &w_ok, &w_ok, &params).unwrap_err();
        assert!(matches!(err, AttentionKernelError::DimensionMismatch { .. }));
    }

    // ------------------------------------------------------------------
    // Scaled dot-product attention
    // ------------------------------------------------------------------
    #[test]
    fn test_sdpa_single_head_single_token() {
        let params = AttentionParams::new(1, 4, 1);
        let q = vec![1.0, 0.0, 0.0, 0.0];
        let k = vec![1.0, 0.0, 0.0, 0.0];
        let v = vec![0.0, 1.0, 2.0, 3.0];
        let out = scaled_dot_product_attention(&q, &k, &v, &params).unwrap();
        // Single token: softmax of single score == 1.0, so output == v
        for (a, b) in out.output.iter().zip(v.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn test_sdpa_uniform_attention() {
        // 1 head, 2 tokens, 2-d head, uniform Q/K → equal weights
        let mut params = AttentionParams::new(1, 2, 2);
        params.scale = 1.0;
        let q = vec![0.0, 0.0, 0.0, 0.0]; // zero → uniform attention
        let k = vec![0.0, 0.0, 0.0, 0.0];
        let v = vec![1.0, 0.0, 0.0, 1.0];
        let out = scaled_dot_product_attention(&q, &k, &v, &params).unwrap();
        // Each row attends equally → output = mean of v rows = [0.5, 0.5]
        for i in 0..2 {
            assert!((out.output[i * 2] - 0.5).abs() < 1e-5);
            assert!((out.output[i * 2 + 1] - 0.5).abs() < 1e-5);
        }
    }

    #[test]
    fn test_sdpa_multi_head() {
        let params = AttentionParams::new(2, 2, 1);
        let q = vec![1.0, 0.0, 0.0, 1.0]; // 2 heads × 1 token × 2
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let v = vec![10.0, 20.0, 30.0, 40.0];
        let out = scaled_dot_product_attention(&q, &k, &v, &params).unwrap();
        assert_eq!(out.output.len(), 4);
        // Single KV token per head → output == v
        for (a, b) in out.output.iter().zip(v.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn test_sdpa_dim_mismatch_q() {
        let params = AttentionParams::new(1, 4, 1);
        let q = vec![1.0; 3]; // wrong
        let k = vec![1.0; 4];
        let v = vec![1.0; 4];
        assert!(scaled_dot_product_attention(&q, &k, &v, &params).is_err());
    }

    #[test]
    fn test_sdpa_dim_mismatch_k() {
        let params = AttentionParams::new(1, 4, 1);
        let q = vec![1.0; 4];
        let k = vec![1.0; 3]; // wrong
        let v = vec![1.0; 4];
        assert!(scaled_dot_product_attention(&q, &k, &v, &params).is_err());
    }

    #[test]
    fn test_sdpa_dim_mismatch_v() {
        let params = AttentionParams::new(1, 4, 1);
        let q = vec![1.0; 4];
        let k = vec![1.0; 4];
        let v = vec![1.0; 3]; // wrong
        assert!(scaled_dot_product_attention(&q, &k, &v, &params).is_err());
    }

    #[test]
    fn test_sdpa_with_causal() {
        let params = AttentionParams::new(1, 2, 3).with_causal(true);
        // Q=K=constant → without causal, uniform attention.
        // With causal: first token only sees itself.
        let q = constant(6, 0.0);
        let k = constant(6, 0.0);
        let v = vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let out = scaled_dot_product_attention(&q, &k, &v, &params).unwrap();
        // First row: attends only to token 0 → [1.0, 0.0]
        assert!((out.output[0] - 1.0).abs() < 1e-5);
        assert!((out.output[1] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_sdpa_weights_returned() {
        let params = AttentionParams::new(1, 2, 2);
        let q = constant(4, 0.0);
        let k = constant(4, 0.0);
        let v = constant(4, 1.0);
        let out = scaled_dot_product_attention(&q, &k, &v, &params).unwrap();
        let w = out.attention_weights.unwrap();
        assert_eq!(w.len(), 4); // 1 head × 2 × 2
        // Uniform → each weight ≈ 0.5
        for &wt in &w {
            assert!((wt - 0.5).abs() < 1e-5);
        }
    }

    #[test]
    fn test_sdpa_cross_attention() {
        let params = AttentionParams::new(1, 2, 1).with_kv_seq_len(3);
        let q = vec![1.0, 0.0]; // 1 token query
        let k = vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // 3 KV tokens
        let v = vec![10.0, 0.0, 0.0, 10.0, 0.0, 0.0];
        let out = scaled_dot_product_attention(&q, &k, &v, &params).unwrap();
        assert_eq!(out.output.len(), 2);
    }

    // ------------------------------------------------------------------
    // Causal mask
    // ------------------------------------------------------------------
    #[test]
    fn test_causal_mask_identity_for_seq1() {
        let scores = vec![5.0];
        let masked = apply_causal_mask(&scores, 1);
        assert_eq!(masked, vec![5.0]);
    }

    #[test]
    fn test_causal_mask_upper_triangle_neg_inf() {
        let scores = vec![1.0, 2.0, 3.0, 4.0]; // 2×2
        let masked = apply_causal_mask(&scores, 2);
        assert_eq!(masked[0], 1.0); // (0,0) kept
        assert!(masked[1].is_infinite() && masked[1] < 0.0); // (0,1) masked
        assert_eq!(masked[2], 3.0); // (1,0) kept
        assert_eq!(masked[3], 4.0); // (1,1) kept
    }

    #[test]
    fn test_causal_mask_3x3() {
        let scores = vec![1.0; 9];
        let masked = apply_causal_mask(&scores, 3);
        // Lower triangular + diagonal should be 1.0
        assert_eq!(masked[0], 1.0);
        assert!(masked[1] == f32::NEG_INFINITY);
        assert!(masked[2] == f32::NEG_INFINITY);
        assert_eq!(masked[3], 1.0);
        assert_eq!(masked[4], 1.0);
        assert!(masked[5] == f32::NEG_INFINITY);
        assert_eq!(masked[6], 1.0);
        assert_eq!(masked[7], 1.0);
        assert_eq!(masked[8], 1.0);
    }

    // ------------------------------------------------------------------
    // ALiBi
    // ------------------------------------------------------------------
    #[test]
    fn test_alibi_slope_monotonic() {
        // Slopes should decrease as head index increases.
        let nh = 8;
        let slopes: Vec<f32> = (0..nh).map(|h| alibi_slope(h, nh)).collect();
        for i in 1..slopes.len() {
            assert!(slopes[i] < slopes[i - 1], "slopes must decrease");
        }
    }

    #[test]
    fn test_alibi_bias_modifies_scores() {
        let nh = 2;
        let sl = 3;
        let mut scores = vec![0.0f32; nh * sl * sl];
        apply_alibi_bias(&mut scores, nh, sl);
        // Position (0,0) distance=0 → bias=0, but others != 0
        assert!((scores[0]).abs() < 1e-7);
        // Position (0,1) distance=1 → positive bias (for head 0)
        assert!(scores[1] > 0.0);
        // Position (1,0) distance=-1 → negative bias (for head 0)
        assert!(scores[1 * sl + 0] < 0.0);
    }

    #[test]
    fn test_alibi_bias_head0_vs_head1() {
        let nh = 2;
        let sl = 4;
        let mut scores = vec![0.0f32; nh * sl * sl];
        apply_alibi_bias(&mut scores, nh, sl);
        // Head 0 should have larger slope magnitude than head 1
        let head0_01 = scores[0 * sl * sl + 0 * sl + 1]; // (0,1) in head 0
        let head1_01 = scores[1 * sl * sl + 0 * sl + 1]; // (0,1) in head 1
        assert!(head0_01.abs() > head1_01.abs());
    }

    // ------------------------------------------------------------------
    // RoPE
    // ------------------------------------------------------------------
    #[test]
    fn test_rope_preserves_norm() {
        let hd = 4;
        let positions = vec![0, 1, 2];
        let mut q = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let mut k = q.clone();
        let norm_before: f32 = q.iter().map(|x| x * x).sum();
        rotary_position_embedding(&mut q, &mut k, &positions, hd).unwrap();
        let norm_after: f32 = q.iter().map(|x| x * x).sum();
        assert!((norm_before - norm_after).abs() < 1e-4, "RoPE should preserve vector norm");
    }

    #[test]
    fn test_rope_position_zero_is_identity() {
        let hd = 4;
        let positions = vec![0];
        let mut q = vec![1.0, 2.0, 3.0, 4.0];
        let mut k = vec![5.0, 6.0, 7.0, 8.0];
        let q_orig = q.clone();
        let k_orig = k.clone();
        rotary_position_embedding(&mut q, &mut k, &positions, hd).unwrap();
        // At position 0, theta=0 → cos=1, sin=0 → identity
        for i in 0..hd {
            assert!((q[i] - q_orig[i]).abs() < 1e-5);
            assert!((k[i] - k_orig[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_rope_odd_head_dim_error() {
        let mut q = vec![1.0; 3];
        let mut k = vec![1.0; 3];
        let err = rotary_position_embedding(&mut q, &mut k, &[0], 3).unwrap_err();
        assert!(matches!(err, AttentionKernelError::OddHeadDim(3)));
    }

    #[test]
    fn test_rope_different_positions_differ() {
        let hd = 4;
        let mut q1 = vec![1.0, 0.0, 0.0, 1.0];
        let mut k1 = q1.clone();
        let mut q2 = q1.clone();
        let mut k2 = q1.clone();
        rotary_position_embedding(&mut q1, &mut k1, &[1], hd).unwrap();
        rotary_position_embedding(&mut q2, &mut k2, &[5], hd).unwrap();
        assert!(q1 != q2, "different positions must produce different embeddings");
    }

    // ------------------------------------------------------------------
    // Grouped-query attention
    // ------------------------------------------------------------------
    #[test]
    fn test_gqa_matches_standard_when_groups_equal_heads() {
        let params = AttentionParams::new(2, 4, 3);
        let q = constant(2 * 3 * 4, 0.5);
        let k = constant(2 * 3 * 4, 0.5);
        let v: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let std_out = scaled_dot_product_attention(&q, &k, &v, &params).unwrap();
        let gqa_out = grouped_query_attention(
            &q,
            &k,
            &v,
            &params,
            AttentionStyle::GroupedQuery { num_kv_heads: 2 },
        )
        .unwrap();
        for (a, b) in std_out.output.iter().zip(gqa_out.output.iter()) {
            assert!((a - b).abs() < 1e-4);
        }
    }

    #[test]
    fn test_gqa_with_fewer_kv_heads() {
        let params = AttentionParams::new(4, 2, 2);
        let q = constant(4 * 2 * 2, 1.0);
        let k = constant(2 * 2 * 2, 1.0); // 2 KV heads
        let v = constant(2 * 2 * 2, 1.0);
        let out = grouped_query_attention(
            &q,
            &k,
            &v,
            &params,
            AttentionStyle::GroupedQuery { num_kv_heads: 2 },
        )
        .unwrap();
        assert_eq!(out.output.len(), 4 * 2 * 2);
    }

    #[test]
    fn test_mqa_single_kv_head() {
        let params = AttentionParams::new(4, 2, 2);
        let q = constant(4 * 2 * 2, 1.0);
        let k = constant(1 * 2 * 2, 1.0); // 1 KV head
        let v = constant(1 * 2 * 2, 1.0);
        let out = grouped_query_attention(&q, &k, &v, &params, AttentionStyle::MultiQuery).unwrap();
        assert_eq!(out.output.len(), 4 * 2 * 2);
    }

    #[test]
    fn test_gqa_invalid_head_config() {
        let params = AttentionParams::new(5, 2, 2);
        let q = constant(5 * 2 * 2, 1.0);
        let k = constant(3 * 2 * 2, 1.0);
        let v = constant(3 * 2 * 2, 1.0);
        let err = grouped_query_attention(
            &q,
            &k,
            &v,
            &params,
            AttentionStyle::GroupedQuery { num_kv_heads: 3 },
        )
        .unwrap_err();
        assert!(matches!(err, AttentionKernelError::InvalidHeadConfig { .. }));
    }

    #[test]
    fn test_gqa_wrong_style() {
        let params = AttentionParams::new(2, 2, 1);
        let q = constant(4, 1.0);
        let k = constant(4, 1.0);
        let v = constant(4, 1.0);
        let err =
            grouped_query_attention(&q, &k, &v, &params, AttentionStyle::Standard).unwrap_err();
        assert!(matches!(err, AttentionKernelError::InvalidParameter(_)));
    }

    #[test]
    fn test_gqa_causal() {
        let params = AttentionParams::new(2, 2, 3).with_causal(true);
        let q = constant(2 * 3 * 2, 0.0);
        let k = constant(1 * 3 * 2, 0.0);
        let v = vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // 1 kv-head, 3 tokens, 2-d
        let out = grouped_query_attention(&q, &k, &v, &params, AttentionStyle::MultiQuery).unwrap();
        // First token for both heads attends only to itself → [1.0, 0.0]
        assert!((out.output[0] - 1.0).abs() < 1e-5);
        assert!((out.output[1] - 0.0).abs() < 1e-5);
    }

    // ------------------------------------------------------------------
    // split_heads / merge_heads
    // ------------------------------------------------------------------
    #[test]
    fn test_split_merge_roundtrip() {
        let nh = 2;
        let hd = 3;
        let sl = 4;
        let input: Vec<f32> = (0..sl * nh * hd).map(|i| i as f32).collect();
        let split = split_heads(&input, nh, hd).unwrap();
        let merged = merge_heads(&split, nh, hd).unwrap();
        assert_eq!(input, merged);
    }

    #[test]
    fn test_split_heads_shape() {
        let nh = 4;
        let hd = 8;
        let sl = 3;
        let input = vec![1.0f32; sl * nh * hd];
        let split = split_heads(&input, nh, hd).unwrap();
        assert_eq!(split.len(), nh * sl * hd);
    }

    #[test]
    fn test_split_heads_bad_size() {
        let err = split_heads(&[1.0; 7], 2, 3).unwrap_err();
        assert!(matches!(err, AttentionKernelError::DimensionMismatch { .. }));
    }

    #[test]
    fn test_merge_heads_bad_size() {
        let err = merge_heads(&[1.0; 7], 2, 3).unwrap_err();
        assert!(matches!(err, AttentionKernelError::DimensionMismatch { .. }));
    }

    // ------------------------------------------------------------------
    // Attention stats
    // ------------------------------------------------------------------
    #[test]
    fn test_stats_uniform() {
        let w = vec![0.25f32; 4]; // uniform over 4 elements
        let stats = compute_attention_stats(&w);
        assert!((stats.min - 0.25).abs() < 1e-6);
        assert!((stats.max - 0.25).abs() < 1e-6);
        assert!((stats.mean - 0.25).abs() < 1e-6);
        assert!(stats.entropy > 0.0);
    }

    #[test]
    fn test_stats_empty() {
        let stats = compute_attention_stats(&[]);
        assert_eq!(stats.min, 0.0);
        assert_eq!(stats.max, 0.0);
        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.entropy, 0.0);
    }

    #[test]
    fn test_stats_peaked() {
        let w = vec![1.0, 0.0, 0.0, 0.0];
        let stats = compute_attention_stats(&w);
        assert_eq!(stats.max, 1.0);
        assert_eq!(stats.min, 0.0);
        assert!((stats.mean - 0.25).abs() < 1e-6);
        assert_eq!(stats.entropy, 0.0); // -1*ln(1) + 0 = 0
    }

    // ------------------------------------------------------------------
    // Error display
    // ------------------------------------------------------------------
    #[test]
    fn test_error_display() {
        let e = AttentionKernelError::DimensionMismatch {
            expected: 10,
            actual: 5,
            context: "test".into(),
        };
        assert!(e.to_string().contains("test"));
        assert!(e.to_string().contains("10"));
    }

    #[test]
    fn test_error_invalid_head_config_display() {
        let e = AttentionKernelError::InvalidHeadConfig { num_heads: 7, num_kv_heads: 3 };
        assert!(e.to_string().contains("7"));
        assert!(e.to_string().contains("3"));
    }

    #[test]
    fn test_error_odd_head_dim_display() {
        let e = AttentionKernelError::OddHeadDim(5);
        assert!(e.to_string().contains("5"));
    }

    // ------------------------------------------------------------------
    // Softmax sanity
    // ------------------------------------------------------------------
    #[test]
    fn test_softmax_sums_to_one() {
        let mut row = vec![1.0, 2.0, 3.0, 4.0];
        softmax_inplace(&mut row);
        let sum: f32 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_with_neg_inf() {
        let mut row = vec![1.0, f32::NEG_INFINITY, 2.0];
        softmax_inplace(&mut row);
        assert!((row[1]).abs() < 1e-7); // masked position → 0
        let sum: f32 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_empty() {
        let mut row: Vec<f32> = vec![];
        softmax_inplace(&mut row); // should not panic
    }

    // ------------------------------------------------------------------
    // Proptest properties
    // ------------------------------------------------------------------
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        // P1: softmax output always sums to 1 and is non-negative
        proptest! {
            #[test]
            fn softmax_is_valid_distribution(
                vals in proptest::collection::vec(-100.0f32..100.0f32, 1..64)
            ) {
                let mut row = vals;
                softmax_inplace(&mut row);
                let sum: f32 = row.iter().sum();
                prop_assert!((sum - 1.0).abs() < 1e-4,
                    "softmax sum = {sum}, expected 1.0");
                for &v in &row {
                    prop_assert!(v >= 0.0, "softmax output must be non-negative, got {v}");
                }
            }
        }

        // P2: split_heads then merge_heads is identity
        proptest! {
            #[test]
            fn split_merge_roundtrip(
                nh in 1usize..=4,
                hd in 1usize..=8,
                sl in 1usize..=6,
            ) {
                let len = sl * nh * hd;
                let input: Vec<f32> = (0..len).map(|i| i as f32).collect();
                let split = split_heads(&input, nh, hd).unwrap();
                let merged = merge_heads(&split, nh, hd).unwrap();
                prop_assert_eq!(&input, &merged);
            }
        }

        // P3: causal mask preserves diagonal and lower triangle
        proptest! {
            #[test]
            fn causal_mask_preserves_lower_triangle(
                sl in 1usize..=16,
            ) {
                let scores: Vec<f32> = (0..(sl * sl)).map(|i| i as f32).collect();
                let masked = apply_causal_mask(&scores, sl);
                for i in 0..sl {
                    for j in 0..=i {
                        prop_assert_eq!(
                            masked[i * sl + j],
                            scores[i * sl + j],
                            "lower-triangle value altered at ({},{})", i, j
                        );
                    }
                    for j in (i + 1)..sl {
                        prop_assert!(
                            masked[i * sl + j] == f32::NEG_INFINITY,
                            "upper-triangle not masked at ({},{})", i, j
                        );
                    }
                }
            }
        }

        // P4: RoPE preserves L2 norm of each token
        proptest! {
            #[test]
            fn rope_preserves_per_token_norm(
                hd_half in 1usize..=8,
                n_tokens in 1usize..=4,
            ) {
                let hd = hd_half * 2;
                let len = n_tokens * hd;
                let mut q: Vec<f32> = (0..len).map(|i| (i as f32) * 0.1).collect();
                let mut k = q.clone();
                let positions: Vec<usize> = (0..n_tokens).collect();

                let norms_before: Vec<f64> = (0..n_tokens)
                    .map(|t| {
                        q[t * hd..(t + 1) * hd]
                            .iter()
                            .map(|x| (*x as f64) * (*x as f64))
                            .sum::<f64>()
                            .sqrt()
                    })
                    .collect();

                rotary_position_embedding(&mut q, &mut k, &positions, hd).unwrap();

                for t in 0..n_tokens {
                    let norm_after: f64 = q[t * hd..(t + 1) * hd]
                        .iter()
                        .map(|x| (*x as f64) * (*x as f64))
                        .sum::<f64>()
                        .sqrt();
                    prop_assert!(
                        (norms_before[t] - norm_after).abs() < 1e-3,
                        "norm changed at token {t}: {:.6} → {:.6}",
                        norms_before[t], norm_after,
                    );
                }
            }
        }

        // P5: SDPA output length equals num_heads * seq_len * head_dim
        proptest! {
            #[test]
            fn sdpa_output_has_correct_shape(
                nh in 1usize..=4,
                hd in 1usize..=8,
                sl in 1usize..=4,
            ) {
                let params = AttentionParams::new(nh, hd, sl);
                let total = nh * sl * hd;
                let q = vec![0.0f32; total];
                let k = vec![0.0f32; total];
                let v = vec![1.0f32; total];
                let out = scaled_dot_product_attention(&q, &k, &v, &params).unwrap();
                prop_assert_eq!(out.output.len(), total);
            }
        }
    }
}
