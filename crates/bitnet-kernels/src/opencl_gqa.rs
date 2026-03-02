//! OpenCL Grouped-Query Attention (GQA) and Multi-Query Attention (MQA).
//!
//! Efficient KV cache usage on Intel Arc A770 (Xe-HPG) by sharing key/value
//! heads across groups of query heads. Three attention modes are supported:
//!
//! - **MHA** — standard multi-head attention (`num_kv_heads == num_q_heads`).
//! - **GQA** — grouped-query attention (`num_kv_heads < num_q_heads`,
//!   `num_q_heads % num_kv_heads == 0`).
//! - **MQA** — multi-query attention (`num_kv_heads == 1`).
//!
//! CPU reference implementations are provided for correctness testing. An
//! embedded OpenCL C kernel source targets GPU dispatch.

use std::fmt;

// ---------------------------------------------------------------------------
// Attention type
// ---------------------------------------------------------------------------

/// Attention variant inferred from the head configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttentionType {
    /// Standard multi-head attention: each query head has its own KV head.
    Mha,
    /// Grouped-query attention: multiple query heads share a KV head.
    Gqa,
    /// Multi-query attention: all query heads share a single KV head.
    Mqa,
}

impl AttentionType {
    /// Infer the attention type from head counts.
    pub fn infer(num_q_heads: usize, num_kv_heads: usize) -> Self {
        if num_kv_heads == num_q_heads {
            Self::Mha
        } else if num_kv_heads == 1 {
            Self::Mqa
        } else {
            Self::Gqa
        }
    }
}

impl fmt::Display for AttentionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mha => write!(f, "MHA"),
            Self::Gqa => write!(f, "GQA"),
            Self::Mqa => write!(f, "MQA"),
        }
    }
}

// ---------------------------------------------------------------------------
// GQA configuration
// ---------------------------------------------------------------------------

/// Configuration for grouped-query attention.
#[derive(Debug, Clone)]
pub struct GqaConfig {
    /// Number of query heads.
    pub num_q_heads: usize,
    /// Number of key/value heads (≤ `num_q_heads`).
    pub num_kv_heads: usize,
    /// Dimensionality of each head.
    pub head_dim: usize,
    /// Maximum sequence length (KV cache pre-allocation bound).
    pub max_seq_len: usize,
    /// Inferred attention type.
    pub attention_type: AttentionType,
}

/// Errors specific to GQA configuration or execution.
#[derive(Debug, Clone, PartialEq)]
pub enum GqaError {
    /// A dimension parameter is zero.
    ZeroDimension(String),
    /// `num_q_heads` is not divisible by `num_kv_heads`.
    UnevenGrouping { num_q_heads: usize, num_kv_heads: usize },
    /// `num_kv_heads` exceeds `num_q_heads`.
    TooManyKvHeads { num_q_heads: usize, num_kv_heads: usize },
    /// Buffer length does not match the expected size.
    BufferMismatch { expected: usize, actual: usize },
    /// Sequence length exceeds the configured maximum.
    SequenceTooLong { seq_len: usize, max_seq_len: usize },
}

impl fmt::Display for GqaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension(name) => write!(f, "{name} must be > 0"),
            Self::UnevenGrouping { num_q_heads, num_kv_heads } => {
                write!(
                    f,
                    "num_q_heads ({num_q_heads}) must be divisible by \
                     num_kv_heads ({num_kv_heads})"
                )
            }
            Self::TooManyKvHeads { num_q_heads, num_kv_heads } => {
                write!(
                    f,
                    "num_kv_heads ({num_kv_heads}) must be ≤ \
                     num_q_heads ({num_q_heads})"
                )
            }
            Self::BufferMismatch { expected, actual } => {
                write!(f, "buffer length mismatch: expected {expected}, got {actual}")
            }
            Self::SequenceTooLong { seq_len, max_seq_len } => {
                write!(f, "seq_len {seq_len} exceeds max_seq_len {max_seq_len}")
            }
        }
    }
}

impl std::error::Error for GqaError {}

impl GqaConfig {
    /// Create a new GQA configuration.
    ///
    /// # Errors
    ///
    /// Returns [`GqaError`] if any dimension is zero, `num_kv_heads` exceeds
    /// `num_q_heads`, or `num_q_heads` is not evenly divisible by
    /// `num_kv_heads`.
    pub fn new(
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Result<Self, GqaError> {
        if num_q_heads == 0 {
            return Err(GqaError::ZeroDimension("num_q_heads".into()));
        }
        if num_kv_heads == 0 {
            return Err(GqaError::ZeroDimension("num_kv_heads".into()));
        }
        if head_dim == 0 {
            return Err(GqaError::ZeroDimension("head_dim".into()));
        }
        if max_seq_len == 0 {
            return Err(GqaError::ZeroDimension("max_seq_len".into()));
        }
        if num_kv_heads > num_q_heads {
            return Err(GqaError::TooManyKvHeads { num_q_heads, num_kv_heads });
        }
        if !num_q_heads.is_multiple_of(num_kv_heads) {
            return Err(GqaError::UnevenGrouping { num_q_heads, num_kv_heads });
        }
        let attention_type = AttentionType::infer(num_q_heads, num_kv_heads);
        Ok(Self { num_q_heads, num_kv_heads, head_dim, max_seq_len, attention_type })
    }

    /// Number of query heads per KV group.
    pub fn group_size(&self) -> usize {
        self.num_q_heads / self.num_kv_heads
    }

    /// Scaling factor `1 / sqrt(head_dim)`.
    pub fn scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }
}

// ---------------------------------------------------------------------------
// Head grouping
// ---------------------------------------------------------------------------

/// Maps each query head index to its corresponding KV head index.
#[derive(Debug, Clone)]
pub struct HeadGrouping {
    /// `mapping[q_head] = kv_head` for each query head.
    mapping: Vec<usize>,
    /// Number of query heads per KV group.
    group_size: usize,
}

impl HeadGrouping {
    /// Build a head-grouping map from the given config.
    pub fn from_config(config: &GqaConfig) -> Self {
        let group_size = config.group_size();
        let mapping: Vec<usize> = (0..config.num_q_heads).map(|q| q / group_size).collect();
        Self { mapping, group_size }
    }

    /// Return the KV head index for a given query head.
    pub fn kv_head_for(&self, q_head: usize) -> usize {
        self.mapping[q_head]
    }

    /// Number of query heads that share each KV head.
    pub fn group_size(&self) -> usize {
        self.group_size
    }

    /// Return the full mapping slice.
    pub fn mapping(&self) -> &[usize] {
        &self.mapping
    }

    /// Return query head indices belonging to the given KV group.
    pub fn q_heads_in_group(&self, kv_head: usize) -> Vec<usize> {
        self.mapping.iter().enumerate().filter(|&(_, &kv)| kv == kv_head).map(|(q, _)| q).collect()
    }
}

// ---------------------------------------------------------------------------
// KV expander
// ---------------------------------------------------------------------------

/// Repeats KV head data so that the resulting buffer has one copy per query
/// head. This is the non-optimized fallback path; the optimized path uses
/// group-aware indexing to avoid the copy.
pub struct KvExpander;

impl KvExpander {
    /// Expand KV tensor of shape `[num_kv_heads, seq_len, head_dim]` to
    /// `[num_q_heads, seq_len, head_dim]` by repeating each KV head
    /// `group_size` times.
    pub fn expand(kv: &[f32], config: &GqaConfig, seq_len: usize) -> Result<Vec<f32>, GqaError> {
        let expected = config.num_kv_heads * seq_len * config.head_dim;
        if kv.len() != expected {
            return Err(GqaError::BufferMismatch { expected, actual: kv.len() });
        }
        let group_size = config.group_size();
        let head_stride = seq_len * config.head_dim;
        let mut out = Vec::with_capacity(config.num_q_heads * head_stride);
        for kv_h in 0..config.num_kv_heads {
            let src = &kv[kv_h * head_stride..(kv_h + 1) * head_stride];
            for _ in 0..group_size {
                out.extend_from_slice(src);
            }
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Causal grouped mask
// ---------------------------------------------------------------------------

/// Causal mask generator compatible with GQA head groups.
///
/// All query heads in the same group share the same causal mask (since they
/// attend to the same KV sequence). The mask value is `0.0` for allowed
/// positions and `f32::NEG_INFINITY` for masked positions.
pub struct CausalGroupedMask;

impl CausalGroupedMask {
    /// Generate a causal mask of shape `[seq_len, seq_len]`.
    ///
    /// `mask[i][j] = 0.0` if `j <= i`, else `NEG_INFINITY`.
    pub fn generate(seq_len: usize) -> Vec<f32> {
        let mut mask = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                mask[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
        mask
    }

    /// Apply causal mask in-place to a score matrix `[seq_len, seq_len]`.
    pub fn apply_in_place(scores: &mut [f32], seq_len: usize) {
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                scores[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
    }

    /// Check whether a position pair is causally valid.
    pub fn is_valid(query_pos: usize, key_pos: usize) -> bool {
        key_pos <= query_pos
    }
}

// ---------------------------------------------------------------------------
// Grouped scorer
// ---------------------------------------------------------------------------

/// Computes Q×Kᵀ dot-product scores with group-aware KV indexing.
///
/// Instead of expanding KV heads, the scorer indexes into the correct KV
/// head for each query head, saving memory bandwidth.
pub struct GroupedScorer;

impl GroupedScorer {
    /// Compute attention scores for all query heads.
    ///
    /// - `q`: `[num_q_heads, seq_len, head_dim]`
    /// - `k`: `[num_kv_heads, seq_len, head_dim]`
    ///
    /// Returns scores `[num_q_heads, seq_len, seq_len]` scaled by
    /// `1/sqrt(head_dim)`.
    pub fn compute(
        q: &[f32],
        k: &[f32],
        config: &GqaConfig,
        seq_len: usize,
    ) -> Result<Vec<f32>, GqaError> {
        let q_expected = config.num_q_heads * seq_len * config.head_dim;
        if q.len() != q_expected {
            return Err(GqaError::BufferMismatch { expected: q_expected, actual: q.len() });
        }
        let k_expected = config.num_kv_heads * seq_len * config.head_dim;
        if k.len() != k_expected {
            return Err(GqaError::BufferMismatch { expected: k_expected, actual: k.len() });
        }

        let grouping = HeadGrouping::from_config(config);
        let scale = config.scale();
        let head_stride = seq_len * config.head_dim;
        let mut scores = vec![0.0f32; config.num_q_heads * seq_len * seq_len];

        for qh in 0..config.num_q_heads {
            let kv_h = grouping.kv_head_for(qh);
            let q_base = qh * head_stride;
            let k_base = kv_h * head_stride;
            let s_base = qh * seq_len * seq_len;

            for i in 0..seq_len {
                for j in 0..seq_len {
                    let mut dot = 0.0f32;
                    for d in 0..config.head_dim {
                        dot += q[q_base + i * config.head_dim + d]
                            * k[k_base + j * config.head_dim + d];
                    }
                    scores[s_base + i * seq_len + j] = dot * scale;
                }
            }
        }
        Ok(scores)
    }
}

// ---------------------------------------------------------------------------
// GQA statistics
// ---------------------------------------------------------------------------

/// Statistics about a GQA configuration.
#[derive(Debug, Clone, Copy)]
pub struct GqaStats {
    /// KV memory savings compared to standard MHA (0.0 – 1.0).
    /// E.g., GQA with 32 Q / 8 KV heads saves 75 % → `0.75`.
    pub kv_memory_savings: f32,
    /// Effective memory bandwidth ratio (KV reads / Q reads).
    pub effective_bandwidth_ratio: f32,
    /// Attention FLOPs for a given sequence length.
    pub attention_flops: u64,
}

impl GqaStats {
    /// Compute statistics for the given configuration and sequence length.
    pub fn compute(config: &GqaConfig, seq_len: usize) -> Self {
        let savings = 1.0 - (config.num_kv_heads as f32 / config.num_q_heads as f32);
        let bandwidth_ratio = config.num_kv_heads as f32 / config.num_q_heads as f32;
        // FLOPs: for each Q head, 2 * seq_len * seq_len * head_dim (QK^T)
        //        + 2 * seq_len * seq_len * head_dim (scores @ V)
        let flops_per_head = 2 * (seq_len as u64) * (seq_len as u64) * (config.head_dim as u64);
        let total_flops = (config.num_q_heads as u64) * 2 * flops_per_head;
        Self {
            kv_memory_savings: savings,
            effective_bandwidth_ratio: bandwidth_ratio,
            attention_flops: total_flops,
        }
    }
}

// ---------------------------------------------------------------------------
// GQA attention (CPU reference)
// ---------------------------------------------------------------------------

/// Grouped-query attention: `softmax(Q·Kᵀ / √d) · V` with KV head sharing.
pub struct GqaAttention;

impl GqaAttention {
    /// Compute GQA output.
    ///
    /// - `q`: `[num_q_heads, seq_len, head_dim]`
    /// - `k`: `[num_kv_heads, seq_len, head_dim]`
    /// - `v`: `[num_kv_heads, seq_len, head_dim]`
    ///
    /// Returns output `[num_q_heads, seq_len, head_dim]`.
    pub fn compute(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        config: &GqaConfig,
        seq_len: usize,
        causal: bool,
    ) -> Result<Vec<f32>, GqaError> {
        let q_expected = config.num_q_heads * seq_len * config.head_dim;
        if q.len() != q_expected {
            return Err(GqaError::BufferMismatch { expected: q_expected, actual: q.len() });
        }
        let kv_expected = config.num_kv_heads * seq_len * config.head_dim;
        if k.len() != kv_expected {
            return Err(GqaError::BufferMismatch { expected: kv_expected, actual: k.len() });
        }
        if v.len() != kv_expected {
            return Err(GqaError::BufferMismatch { expected: kv_expected, actual: v.len() });
        }

        let grouping = HeadGrouping::from_config(config);
        let scale = config.scale();
        let head_stride = seq_len * config.head_dim;
        let mut output = vec![0.0f32; config.num_q_heads * head_stride];

        for qh in 0..config.num_q_heads {
            let kv_h = grouping.kv_head_for(qh);
            let q_base = qh * head_stride;
            let k_base = kv_h * head_stride;
            let v_base = kv_h * head_stride;
            let o_base = qh * head_stride;

            // Compute scores: Q·Kᵀ * scale
            let mut scores = vec![0.0f32; seq_len * seq_len];
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let mut dot = 0.0f32;
                    for d in 0..config.head_dim {
                        dot += q[q_base + i * config.head_dim + d]
                            * k[k_base + j * config.head_dim + d];
                    }
                    scores[i * seq_len + j] = dot * scale;
                }
            }

            // Causal mask
            if causal {
                CausalGroupedMask::apply_in_place(&mut scores, seq_len);
            }

            // Row-wise softmax
            for i in 0..seq_len {
                let row = &mut scores[i * seq_len..(i + 1) * seq_len];
                let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in row.iter_mut() {
                    *s = (*s - max_val).exp();
                    sum += *s;
                }
                if sum > 0.0 {
                    for s in row.iter_mut() {
                        *s /= sum;
                    }
                }
            }

            // Output = scores · V
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let w = scores[i * seq_len + j];
                    for d in 0..config.head_dim {
                        output[o_base + i * config.head_dim + d] +=
                            w * v[v_base + j * config.head_dim + d];
                    }
                }
            }
        }
        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// CPU reference: standalone single-head attention
// ---------------------------------------------------------------------------

/// Single-head scaled dot-product attention (CPU reference).
///
/// `q`, `k`, `v` each have shape `[seq_len, head_dim]`.
pub fn cpu_single_head_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    causal: bool,
) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();
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
    if causal {
        CausalGroupedMask::apply_in_place(&mut scores, seq_len);
    }
    softmax_rows(&mut scores, seq_len);
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

/// In-place row-wise softmax over `[rows, cols]` where `cols == row_len`.
fn softmax_rows(data: &mut [f32], row_len: usize) {
    let num_rows = data.len() / row_len;
    for i in 0..num_rows {
        let row = &mut data[i * row_len..(i + 1) * row_len];
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
}

// ---------------------------------------------------------------------------
// OpenCL kernel source
// ---------------------------------------------------------------------------

/// OpenCL C kernel source for group-aware scaled dot-product attention.
///
/// Work-group layout: one work-item per (query_head, query_row) pair.
/// The kernel reads from the correct KV head using `q_head / group_size`.
pub const GQA_ATTENTION_CL: &str = r#"
__kernel void gqa_attention(
    __global const float* Q,       // [num_q_heads, seq_len, head_dim]
    __global const float* K,       // [num_kv_heads, seq_len, head_dim]
    __global const float* V,       // [num_kv_heads, seq_len, head_dim]
    __global float* output,        // [num_q_heads, seq_len, head_dim]
    const int seq_len,
    const int head_dim,
    const int num_q_heads,
    const int num_kv_heads,
    const int group_size,
    const float scale,
    const int causal
) {
    int qh = get_global_id(0);  // query head index
    int qi = get_global_id(1);  // query row index

    if (qh >= num_q_heads || qi >= seq_len) return;

    int kv_h = qh / group_size;
    int head_stride = seq_len * head_dim;
    int q_base = qh * head_stride + qi * head_dim;
    int k_base = kv_h * head_stride;
    int v_base = kv_h * head_stride;
    int o_base = qh * head_stride + qi * head_dim;

    // --- Compute scores and find max ---
    float max_score = -INFINITY;
    for (int j = 0; j < seq_len; j++) {
        if (causal && j > qi) break;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += Q[q_base + d] * K[k_base + j * head_dim + d];
        }
        dot *= scale;
        if (dot > max_score) max_score = dot;
    }

    // --- Softmax numerator + denominator ---
    float sum_exp = 0.0f;
    for (int j = 0; j < seq_len; j++) {
        if (causal && j > qi) break;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += Q[q_base + d] * K[k_base + j * head_dim + d];
        }
        sum_exp += exp(dot * scale - max_score);
    }

    // --- Weighted sum over V ---
    for (int d = 0; d < head_dim; d++) {
        float acc = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            if (causal && j > qi) break;
            float dot = 0.0f;
            for (int dd = 0; dd < head_dim; dd++) {
                dot += Q[q_base + dd] * K[k_base + j * head_dim + dd];
            }
            float w = exp(dot * scale - max_score) / sum_exp;
            acc += w * V[v_base + j * head_dim + d];
        }
        output[o_base + d] = acc;
    }
}
"#;

/// Return the OpenCL kernel source string.
pub fn kernel_source() -> &'static str {
    GQA_ATTENTION_CL
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Deterministic pseudo-random f32 in `[-1, 1]` seeded by index.
    fn pseudo_rand(idx: usize) -> f32 {
        let x = ((idx as u64)
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407)) as f32;
        (x / f32::MAX).sin()
    }

    fn make_tensor(len: usize, seed: usize) -> Vec<f32> {
        (0..len).map(|i| pseudo_rand(i + seed)).collect()
    }

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    fn assert_tensors_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(approx_eq(*x, *y, tol), "mismatch at index {i}: {x} vs {y} (tol={tol})");
        }
    }

    // -----------------------------------------------------------------------
    // AttentionType inference
    // -----------------------------------------------------------------------

    #[test]
    fn test_attention_type_mha() {
        assert_eq!(AttentionType::infer(8, 8), AttentionType::Mha);
    }

    #[test]
    fn test_attention_type_gqa() {
        assert_eq!(AttentionType::infer(32, 8), AttentionType::Gqa);
    }

    #[test]
    fn test_attention_type_mqa() {
        assert_eq!(AttentionType::infer(32, 1), AttentionType::Mqa);
    }

    #[test]
    fn test_attention_type_display() {
        assert_eq!(format!("{}", AttentionType::Mha), "MHA");
        assert_eq!(format!("{}", AttentionType::Gqa), "GQA");
        assert_eq!(format!("{}", AttentionType::Mqa), "MQA");
    }

    #[test]
    fn test_attention_type_single_head_mha() {
        assert_eq!(AttentionType::infer(1, 1), AttentionType::Mha);
    }

    // -----------------------------------------------------------------------
    // GqaConfig
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_mha() {
        let cfg = GqaConfig::new(8, 8, 64, 512).unwrap();
        assert_eq!(cfg.attention_type, AttentionType::Mha);
        assert_eq!(cfg.group_size(), 1);
    }

    #[test]
    fn test_config_gqa() {
        let cfg = GqaConfig::new(32, 8, 64, 512).unwrap();
        assert_eq!(cfg.attention_type, AttentionType::Gqa);
        assert_eq!(cfg.group_size(), 4);
    }

    #[test]
    fn test_config_mqa() {
        let cfg = GqaConfig::new(32, 1, 64, 512).unwrap();
        assert_eq!(cfg.attention_type, AttentionType::Mqa);
        assert_eq!(cfg.group_size(), 32);
    }

    #[test]
    fn test_config_zero_q_heads() {
        let err = GqaConfig::new(0, 1, 64, 512).unwrap_err();
        assert_eq!(err, GqaError::ZeroDimension("num_q_heads".into()));
    }

    #[test]
    fn test_config_zero_kv_heads() {
        let err = GqaConfig::new(8, 0, 64, 512).unwrap_err();
        assert_eq!(err, GqaError::ZeroDimension("num_kv_heads".into()));
    }

    #[test]
    fn test_config_zero_head_dim() {
        let err = GqaConfig::new(8, 8, 0, 512).unwrap_err();
        assert_eq!(err, GqaError::ZeroDimension("head_dim".into()));
    }

    #[test]
    fn test_config_zero_max_seq_len() {
        let err = GqaConfig::new(8, 8, 64, 0).unwrap_err();
        assert_eq!(err, GqaError::ZeroDimension("max_seq_len".into()));
    }

    #[test]
    fn test_config_uneven_grouping() {
        let err = GqaConfig::new(7, 3, 64, 512).unwrap_err();
        assert_eq!(err, GqaError::UnevenGrouping { num_q_heads: 7, num_kv_heads: 3 });
    }

    #[test]
    fn test_config_too_many_kv_heads() {
        let err = GqaConfig::new(4, 8, 64, 512).unwrap_err();
        assert_eq!(err, GqaError::TooManyKvHeads { num_q_heads: 4, num_kv_heads: 8 });
    }

    #[test]
    fn test_config_scale() {
        let cfg = GqaConfig::new(8, 8, 64, 512).unwrap();
        assert!(approx_eq(cfg.scale(), 1.0 / 8.0, 1e-6));
    }

    #[test]
    fn test_config_scale_head_dim_128() {
        let cfg = GqaConfig::new(8, 8, 128, 512).unwrap();
        let expected = 1.0 / (128.0f32).sqrt();
        assert!(approx_eq(cfg.scale(), expected, 1e-6));
    }

    #[test]
    fn test_error_display() {
        let e = GqaError::ZeroDimension("head_dim".into());
        assert_eq!(format!("{e}"), "head_dim must be > 0");
    }

    // -----------------------------------------------------------------------
    // HeadGrouping
    // -----------------------------------------------------------------------

    #[test]
    fn test_grouping_mha() {
        let cfg = GqaConfig::new(4, 4, 64, 256).unwrap();
        let g = HeadGrouping::from_config(&cfg);
        assert_eq!(g.mapping(), &[0, 1, 2, 3]);
    }

    #[test]
    fn test_grouping_gqa_32_8() {
        let cfg = GqaConfig::new(32, 8, 64, 256).unwrap();
        let g = HeadGrouping::from_config(&cfg);
        assert_eq!(g.group_size(), 4);
        // First 4 Q heads → KV head 0
        for i in 0..4 {
            assert_eq!(g.kv_head_for(i), 0);
        }
        // Next 4 → KV head 1
        for i in 4..8 {
            assert_eq!(g.kv_head_for(i), 1);
        }
        // Last group → KV head 7
        for i in 28..32 {
            assert_eq!(g.kv_head_for(i), 7);
        }
    }

    #[test]
    fn test_grouping_mqa() {
        let cfg = GqaConfig::new(8, 1, 64, 256).unwrap();
        let g = HeadGrouping::from_config(&cfg);
        for i in 0..8 {
            assert_eq!(g.kv_head_for(i), 0);
        }
    }

    #[test]
    fn test_grouping_q_heads_in_group() {
        let cfg = GqaConfig::new(8, 2, 64, 256).unwrap();
        let g = HeadGrouping::from_config(&cfg);
        assert_eq!(g.q_heads_in_group(0), vec![0, 1, 2, 3]);
        assert_eq!(g.q_heads_in_group(1), vec![4, 5, 6, 7]);
    }

    #[test]
    fn test_grouping_single_head() {
        let cfg = GqaConfig::new(1, 1, 64, 256).unwrap();
        let g = HeadGrouping::from_config(&cfg);
        assert_eq!(g.mapping(), &[0]);
        assert_eq!(g.group_size(), 1);
    }

    // -----------------------------------------------------------------------
    // KvExpander
    // -----------------------------------------------------------------------

    #[test]
    fn test_kv_expand_mha_identity() {
        let cfg = GqaConfig::new(4, 4, 2, 256).unwrap();
        let kv = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 4 heads, seq=1, dim=2
        let expanded = KvExpander::expand(&kv, &cfg, 1).unwrap();
        assert_eq!(expanded, kv);
    }

    #[test]
    fn test_kv_expand_gqa() {
        let cfg = GqaConfig::new(4, 2, 2, 256).unwrap();
        // 2 KV heads × seq_len=1 × head_dim=2
        let kv = vec![1.0, 2.0, 3.0, 4.0];
        let expanded = KvExpander::expand(&kv, &cfg, 1).unwrap();
        // Each KV head repeated 2×: [1,2,1,2, 3,4,3,4]
        assert_eq!(expanded, vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
    }

    #[test]
    fn test_kv_expand_mqa() {
        let cfg = GqaConfig::new(3, 1, 2, 256).unwrap();
        let kv = vec![1.0, 2.0]; // 1 KV head × seq=1 × dim=2
        let expanded = KvExpander::expand(&kv, &cfg, 1).unwrap();
        assert_eq!(expanded, vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn test_kv_expand_buffer_mismatch() {
        let cfg = GqaConfig::new(4, 2, 2, 256).unwrap();
        let kv = vec![1.0, 2.0]; // too short
        let err = KvExpander::expand(&kv, &cfg, 1).unwrap_err();
        assert_eq!(err, GqaError::BufferMismatch { expected: 4, actual: 2 });
    }

    #[test]
    fn test_kv_expand_longer_seq() {
        let cfg = GqaConfig::new(4, 2, 2, 256).unwrap();
        // 2 KV heads × seq_len=2 × head_dim=2 = 8 elements
        let kv = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let expanded = KvExpander::expand(&kv, &cfg, 2).unwrap();
        // Head 0: [1,2,3,4] repeated 2× then Head 1: [5,6,7,8] repeated 2×
        assert_eq!(expanded.len(), 16);
        assert_eq!(&expanded[0..4], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&expanded[4..8], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&expanded[8..12], &[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(&expanded[12..16], &[5.0, 6.0, 7.0, 8.0]);
    }

    // -----------------------------------------------------------------------
    // CausalGroupedMask
    // -----------------------------------------------------------------------

    #[test]
    fn test_causal_mask_1x1() {
        let mask = CausalGroupedMask::generate(1);
        assert_eq!(mask, vec![0.0]);
    }

    #[test]
    fn test_causal_mask_3x3() {
        let mask = CausalGroupedMask::generate(3);
        // Row 0: [0, -inf, -inf]
        // Row 1: [0, 0, -inf]
        // Row 2: [0, 0, 0]
        assert_eq!(mask[0], 0.0);
        assert!(mask[1].is_infinite() && mask[1] < 0.0);
        assert!(mask[2].is_infinite() && mask[2] < 0.0);
        assert_eq!(mask[3], 0.0);
        assert_eq!(mask[4], 0.0);
        assert!(mask[5].is_infinite() && mask[5] < 0.0);
        assert_eq!(mask[6], 0.0);
        assert_eq!(mask[7], 0.0);
        assert_eq!(mask[8], 0.0);
    }

    #[test]
    fn test_causal_mask_is_valid() {
        assert!(CausalGroupedMask::is_valid(5, 5));
        assert!(CausalGroupedMask::is_valid(5, 3));
        assert!(!CausalGroupedMask::is_valid(3, 5));
        assert!(CausalGroupedMask::is_valid(0, 0));
    }

    #[test]
    fn test_causal_mask_apply_in_place() {
        let mut scores = vec![1.0; 4]; // 2×2
        CausalGroupedMask::apply_in_place(&mut scores, 2);
        assert_eq!(scores[0], 1.0);
        assert!(scores[1].is_infinite());
        assert_eq!(scores[2], 1.0);
        assert_eq!(scores[3], 1.0);
    }

    // -----------------------------------------------------------------------
    // GroupedScorer
    // -----------------------------------------------------------------------

    #[test]
    fn test_scorer_mha_identity() {
        let cfg = GqaConfig::new(2, 2, 2, 256).unwrap();
        let q = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let k = q.clone();
        let scores = GroupedScorer::compute(&q, &k, &cfg, 2).unwrap();
        // 2 heads × 2×2 score matrices = 8 values
        assert_eq!(scores.len(), 8);
    }

    #[test]
    fn test_scorer_gqa_shared_kv() {
        let cfg = GqaConfig::new(4, 2, 2, 256).unwrap();
        // 4 Q heads × seq=1 × dim=2
        let q = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5];
        // 2 KV heads × seq=1 × dim=2
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let scores = GroupedScorer::compute(&q, &k, &cfg, 1).unwrap();
        // 4 heads × 1×1 = 4 scores
        assert_eq!(scores.len(), 4);
        // Q heads 0,1 share KV head 0; Q heads 2,3 share KV head 1
        let scale = cfg.scale();
        assert!(approx_eq(scores[0], 1.0 * scale, 1e-5)); // q[0]·k[0]
        assert!(approx_eq(scores[1], 0.0 * scale, 1e-5)); // q[1]·k[0]
        assert!(approx_eq(scores[2], 1.0 * scale, 1e-5)); // q[2]·k[1]
        assert!(approx_eq(scores[3], 0.5 * scale, 1e-5)); // q[3]·k[1]
    }

    #[test]
    fn test_scorer_buffer_mismatch_q() {
        let cfg = GqaConfig::new(2, 2, 2, 256).unwrap();
        let q = vec![1.0]; // wrong size
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let err = GroupedScorer::compute(&q, &k, &cfg, 1).unwrap_err();
        matches!(err, GqaError::BufferMismatch { .. });
    }

    #[test]
    fn test_scorer_buffer_mismatch_k() {
        let cfg = GqaConfig::new(2, 2, 2, 256).unwrap();
        let q = vec![1.0, 0.0, 0.0, 1.0];
        let k = vec![1.0]; // wrong size
        let err = GroupedScorer::compute(&q, &k, &cfg, 1).unwrap_err();
        matches!(err, GqaError::BufferMismatch { .. });
    }

    // -----------------------------------------------------------------------
    // GqaStats
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_mha_zero_savings() {
        let cfg = GqaConfig::new(8, 8, 64, 512).unwrap();
        let stats = GqaStats::compute(&cfg, 128);
        assert!(approx_eq(stats.kv_memory_savings, 0.0, 1e-6));
        assert!(approx_eq(stats.effective_bandwidth_ratio, 1.0, 1e-6));
    }

    #[test]
    fn test_stats_gqa_75_savings() {
        let cfg = GqaConfig::new(32, 8, 64, 512).unwrap();
        let stats = GqaStats::compute(&cfg, 128);
        assert!(approx_eq(stats.kv_memory_savings, 0.75, 1e-6));
        assert!(approx_eq(stats.effective_bandwidth_ratio, 0.25, 1e-6));
    }

    #[test]
    fn test_stats_mqa_max_savings() {
        let cfg = GqaConfig::new(32, 1, 64, 512).unwrap();
        let stats = GqaStats::compute(&cfg, 128);
        let expected_savings = 1.0 - 1.0 / 32.0;
        assert!(approx_eq(stats.kv_memory_savings, expected_savings, 1e-6));
    }

    #[test]
    fn test_stats_flops_nonzero() {
        let cfg = GqaConfig::new(8, 8, 64, 512).unwrap();
        let stats = GqaStats::compute(&cfg, 128);
        assert!(stats.attention_flops > 0);
    }

    #[test]
    fn test_stats_flops_scale_with_heads() {
        let cfg4 = GqaConfig::new(4, 4, 64, 512).unwrap();
        let cfg8 = GqaConfig::new(8, 8, 64, 512).unwrap();
        let s4 = GqaStats::compute(&cfg4, 128);
        let s8 = GqaStats::compute(&cfg8, 128);
        assert_eq!(s8.attention_flops, s4.attention_flops * 2);
    }

    // -----------------------------------------------------------------------
    // GqaAttention — MHA mode
    // -----------------------------------------------------------------------

    #[test]
    fn test_gqa_mha_single_head_seq1() {
        let cfg = GqaConfig::new(1, 1, 4, 256).unwrap();
        let q = vec![1.0, 0.0, 0.0, 0.0];
        let k = vec![1.0, 0.0, 0.0, 0.0];
        let v = vec![0.0, 1.0, 0.0, 0.0];
        let out = GqaAttention::compute(&q, &k, &v, &cfg, 1, false).unwrap();
        assert_tensors_close(&out, &v, 1e-5);
    }

    #[test]
    fn test_gqa_mha_matches_reference() {
        let cfg = GqaConfig::new(2, 2, 4, 256).unwrap();
        let seq_len = 3;
        let q = make_tensor(2 * seq_len * 4, 0);
        let k = make_tensor(2 * seq_len * 4, 100);
        let v = make_tensor(2 * seq_len * 4, 200);
        let out = GqaAttention::compute(&q, &k, &v, &cfg, seq_len, false).unwrap();
        // Compare head-by-head against single-head reference
        for h in 0..2 {
            let off = h * seq_len * 4;
            let q_h = &q[off..off + seq_len * 4];
            let k_h = &k[off..off + seq_len * 4];
            let v_h = &v[off..off + seq_len * 4];
            let ref_out = cpu_single_head_attention(q_h, k_h, v_h, seq_len, 4, false);
            assert_tensors_close(&out[off..off + seq_len * 4], &ref_out, 1e-5);
        }
    }

    #[test]
    fn test_gqa_mha_causal() {
        let cfg = GqaConfig::new(1, 1, 2, 256).unwrap();
        let seq_len = 3;
        let q = make_tensor(seq_len * 2, 10);
        let k = make_tensor(seq_len * 2, 20);
        let v = make_tensor(seq_len * 2, 30);
        let out = GqaAttention::compute(&q, &k, &v, &cfg, seq_len, true).unwrap();
        let ref_out = cpu_single_head_attention(&q, &k, &v, seq_len, 2, true);
        assert_tensors_close(&out, &ref_out, 1e-5);
    }

    // -----------------------------------------------------------------------
    // GqaAttention — GQA mode
    // -----------------------------------------------------------------------

    #[test]
    fn test_gqa_32q_8kv() {
        let cfg = GqaConfig::new(32, 8, 4, 256).unwrap();
        let seq_len = 2;
        let q = make_tensor(32 * seq_len * 4, 0);
        let k = make_tensor(8 * seq_len * 4, 50);
        let v = make_tensor(8 * seq_len * 4, 100);
        let out = GqaAttention::compute(&q, &k, &v, &cfg, seq_len, false).unwrap();
        assert_eq!(out.len(), 32 * seq_len * 4);
    }

    #[test]
    fn test_gqa_shared_kv_heads_identical_output() {
        // Q heads in the same group with identical Q should produce identical output
        let cfg = GqaConfig::new(4, 2, 2, 256).unwrap();
        let seq_len = 2;
        // Make Q heads 0 and 1 identical (both map to KV head 0)
        let mut q = vec![0.0f32; 4 * seq_len * 2];
        let pattern = [1.0, 0.5, 0.3, 0.8]; // seq=2, dim=2
        q[0..4].copy_from_slice(&pattern); // head 0
        q[4..8].copy_from_slice(&pattern); // head 1 = same
        // heads 2,3 can differ
        q[8..12].copy_from_slice(&[0.1, 0.2, 0.3, 0.4]);
        q[12..16].copy_from_slice(&[0.5, 0.6, 0.7, 0.8]);
        let k = make_tensor(2 * seq_len * 2, 10);
        let v = make_tensor(2 * seq_len * 2, 20);
        let out = GqaAttention::compute(&q, &k, &v, &cfg, seq_len, false).unwrap();
        // Head 0 and head 1 output should be identical
        let h0 = &out[0..4];
        let h1 = &out[4..8];
        assert_tensors_close(h0, h1, 1e-6);
    }

    #[test]
    fn test_gqa_equivalence_with_expanded_kv() {
        // GQA with group-aware indexing should match MHA with expanded KV
        let cfg_gqa = GqaConfig::new(4, 2, 2, 256).unwrap();
        let cfg_mha = GqaConfig::new(4, 4, 2, 256).unwrap();
        let seq_len = 3;
        let q = make_tensor(4 * seq_len * 2, 0);
        let k_small = make_tensor(2 * seq_len * 2, 50);
        let v_small = make_tensor(2 * seq_len * 2, 100);
        let k_expanded = KvExpander::expand(&k_small, &cfg_gqa, seq_len).unwrap();
        let v_expanded = KvExpander::expand(&v_small, &cfg_gqa, seq_len).unwrap();

        let out_gqa =
            GqaAttention::compute(&q, &k_small, &v_small, &cfg_gqa, seq_len, false).unwrap();
        let out_mha =
            GqaAttention::compute(&q, &k_expanded, &v_expanded, &cfg_mha, seq_len, false).unwrap();
        assert_tensors_close(&out_gqa, &out_mha, 1e-5);
    }

    #[test]
    fn test_gqa_causal_32q_8kv() {
        let cfg = GqaConfig::new(32, 8, 4, 256).unwrap();
        let seq_len = 4;
        let q = make_tensor(32 * seq_len * 4, 0);
        let k = make_tensor(8 * seq_len * 4, 50);
        let v = make_tensor(8 * seq_len * 4, 100);
        let out = GqaAttention::compute(&q, &k, &v, &cfg, seq_len, true).unwrap();
        assert_eq!(out.len(), 32 * seq_len * 4);
        // Verify all values are finite
        assert!(out.iter().all(|x| x.is_finite()));
    }

    // -----------------------------------------------------------------------
    // GqaAttention — MQA mode
    // -----------------------------------------------------------------------

    #[test]
    fn test_mqa_single_kv_head() {
        let cfg = GqaConfig::new(8, 1, 4, 256).unwrap();
        let seq_len = 2;
        let q = make_tensor(8 * seq_len * 4, 0);
        let k = make_tensor(1 * seq_len * 4, 50);
        let v = make_tensor(1 * seq_len * 4, 100);
        let out = GqaAttention::compute(&q, &k, &v, &cfg, seq_len, false).unwrap();
        assert_eq!(out.len(), 8 * seq_len * 4);
    }

    #[test]
    fn test_mqa_all_heads_same_kv() {
        // With identical Q, all heads should produce identical output
        let cfg = GqaConfig::new(4, 1, 2, 256).unwrap();
        let seq_len = 2;
        let pattern: Vec<f32> = vec![1.0, 0.5, 0.3, 0.8];
        let q: Vec<f32> = pattern.iter().copied().cycle().take(4 * seq_len * 2).collect();
        let k = make_tensor(1 * seq_len * 2, 10);
        let v = make_tensor(1 * seq_len * 2, 20);
        let out = GqaAttention::compute(&q, &k, &v, &cfg, seq_len, false).unwrap();
        let stride = seq_len * 2;
        let h0 = &out[0..stride];
        for h in 1..4 {
            assert_tensors_close(&out[h * stride..(h + 1) * stride], h0, 1e-6);
        }
    }

    #[test]
    fn test_mqa_causal() {
        let cfg = GqaConfig::new(4, 1, 2, 256).unwrap();
        let seq_len = 4;
        let q = make_tensor(4 * seq_len * 2, 0);
        let k = make_tensor(1 * seq_len * 2, 50);
        let v = make_tensor(1 * seq_len * 2, 100);
        let out = GqaAttention::compute(&q, &k, &v, &cfg, seq_len, true).unwrap();
        assert!(out.iter().all(|x| x.is_finite()));
    }

    // -----------------------------------------------------------------------
    // Various head dimensions
    // -----------------------------------------------------------------------

    #[test]
    fn test_head_dim_32() {
        let cfg = GqaConfig::new(4, 2, 32, 256).unwrap();
        let seq_len = 3;
        let q = make_tensor(4 * seq_len * 32, 0);
        let k = make_tensor(2 * seq_len * 32, 50);
        let v = make_tensor(2 * seq_len * 32, 100);
        let out = GqaAttention::compute(&q, &k, &v, &cfg, seq_len, false).unwrap();
        assert_eq!(out.len(), 4 * seq_len * 32);
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_head_dim_64() {
        let cfg = GqaConfig::new(8, 4, 64, 256).unwrap();
        let seq_len = 2;
        let q = make_tensor(8 * seq_len * 64, 0);
        let k = make_tensor(4 * seq_len * 64, 50);
        let v = make_tensor(4 * seq_len * 64, 100);
        let out = GqaAttention::compute(&q, &k, &v, &cfg, seq_len, false).unwrap();
        assert_eq!(out.len(), 8 * seq_len * 64);
    }

    #[test]
    fn test_head_dim_128() {
        let cfg = GqaConfig::new(8, 2, 128, 256).unwrap();
        let seq_len = 2;
        let q = make_tensor(8 * seq_len * 128, 0);
        let k = make_tensor(2 * seq_len * 128, 50);
        let v = make_tensor(2 * seq_len * 128, 100);
        let out = GqaAttention::compute(&q, &k, &v, &cfg, seq_len, false).unwrap();
        assert_eq!(out.len(), 8 * seq_len * 128);
        assert!(out.iter().all(|x| x.is_finite()));
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_single_head_single_token() {
        let cfg = GqaConfig::new(1, 1, 2, 256).unwrap();
        let q = vec![1.0, 0.0];
        let k = vec![1.0, 0.0];
        let v = vec![0.5, 0.5];
        let out = GqaAttention::compute(&q, &k, &v, &cfg, 1, false).unwrap();
        // softmax of single element = 1.0, output = v
        assert_tensors_close(&out, &v, 1e-5);
    }

    #[test]
    fn test_seq_len_1_gqa() {
        let cfg = GqaConfig::new(8, 2, 4, 256).unwrap();
        let q = make_tensor(8 * 1 * 4, 0);
        let k = make_tensor(2 * 1 * 4, 50);
        let v = make_tensor(2 * 1 * 4, 100);
        let out = GqaAttention::compute(&q, &k, &v, &cfg, 1, true).unwrap();
        assert_eq!(out.len(), 8 * 1 * 4);
    }

    #[test]
    fn test_buffer_mismatch_q() {
        let cfg = GqaConfig::new(2, 2, 2, 256).unwrap();
        let q = vec![1.0]; // wrong
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let v = vec![1.0, 0.0, 0.0, 1.0];
        let err = GqaAttention::compute(&q, &k, &v, &cfg, 1, false).unwrap_err();
        matches!(err, GqaError::BufferMismatch { .. });
    }

    #[test]
    fn test_buffer_mismatch_k() {
        let cfg = GqaConfig::new(2, 2, 2, 256).unwrap();
        let q = vec![1.0, 0.0, 0.0, 1.0];
        let k = vec![1.0]; // wrong
        let v = vec![1.0, 0.0, 0.0, 1.0];
        let err = GqaAttention::compute(&q, &k, &v, &cfg, 1, false).unwrap_err();
        matches!(err, GqaError::BufferMismatch { .. });
    }

    #[test]
    fn test_buffer_mismatch_v() {
        let cfg = GqaConfig::new(2, 2, 2, 256).unwrap();
        let q = vec![1.0, 0.0, 0.0, 1.0];
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let v = vec![1.0]; // wrong
        let err = GqaAttention::compute(&q, &k, &v, &cfg, 1, false).unwrap_err();
        matches!(err, GqaError::BufferMismatch { .. });
    }

    // -----------------------------------------------------------------------
    // Property: attention outputs in valid range
    // -----------------------------------------------------------------------

    #[test]
    fn test_output_finite_mha_various_seeds() {
        for seed in 0..5 {
            let cfg = GqaConfig::new(4, 4, 8, 256).unwrap();
            let sl = 4;
            let q = make_tensor(4 * sl * 8, seed * 1000);
            let k = make_tensor(4 * sl * 8, seed * 1000 + 100);
            let v = make_tensor(4 * sl * 8, seed * 1000 + 200);
            let out = GqaAttention::compute(&q, &k, &v, &cfg, sl, false).unwrap();
            assert!(out.iter().all(|x| x.is_finite()), "seed={seed}");
        }
    }

    #[test]
    fn test_output_finite_gqa_various_seeds() {
        for seed in 0..5 {
            let cfg = GqaConfig::new(8, 2, 8, 256).unwrap();
            let sl = 4;
            let q = make_tensor(8 * sl * 8, seed * 1000);
            let k = make_tensor(2 * sl * 8, seed * 1000 + 100);
            let v = make_tensor(2 * sl * 8, seed * 1000 + 200);
            let out = GqaAttention::compute(&q, &k, &v, &cfg, sl, true).unwrap();
            assert!(out.iter().all(|x| x.is_finite()), "seed={seed}");
        }
    }

    #[test]
    fn test_output_finite_mqa_various_seeds() {
        for seed in 0..5 {
            let cfg = GqaConfig::new(16, 1, 8, 256).unwrap();
            let sl = 4;
            let q = make_tensor(16 * sl * 8, seed * 1000);
            let k = make_tensor(1 * sl * 8, seed * 1000 + 100);
            let v = make_tensor(1 * sl * 8, seed * 1000 + 200);
            let out = GqaAttention::compute(&q, &k, &v, &cfg, sl, false).unwrap();
            assert!(out.iter().all(|x| x.is_finite()), "seed={seed}");
        }
    }

    // -----------------------------------------------------------------------
    // Property: softmax weights sum to 1
    // -----------------------------------------------------------------------

    #[test]
    fn test_softmax_rows_sum_to_one() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2×3
        softmax_rows(&mut data, 3);
        let sum0: f32 = data[0..3].iter().sum();
        let sum1: f32 = data[3..6].iter().sum();
        assert!(approx_eq(sum0, 1.0, 1e-5));
        assert!(approx_eq(sum1, 1.0, 1e-5));
    }

    #[test]
    fn test_softmax_all_equal() {
        let mut data = vec![0.0; 4]; // 1×4
        softmax_rows(&mut data, 4);
        for &v in &data {
            assert!(approx_eq(v, 0.25, 1e-5));
        }
    }

    // -----------------------------------------------------------------------
    // CPU single-head reference
    // -----------------------------------------------------------------------

    #[test]
    fn test_cpu_single_head_seq1() {
        let q = vec![1.0, 0.0];
        let k = vec![1.0, 0.0];
        let v = vec![0.5, 0.3];
        let out = cpu_single_head_attention(&q, &k, &v, 1, 2, false);
        assert_tensors_close(&out, &v, 1e-5);
    }

    #[test]
    fn test_cpu_single_head_causal_last_row() {
        // With causal mask, last row sees all tokens → non-causal = causal for last row
        let seq_len = 4;
        let dim = 4;
        let q = make_tensor(seq_len * dim, 0);
        let k = make_tensor(seq_len * dim, 50);
        let v = make_tensor(seq_len * dim, 100);
        let out_c = cpu_single_head_attention(&q, &k, &v, seq_len, dim, true);
        let out_nc = cpu_single_head_attention(&q, &k, &v, seq_len, dim, false);
        // Last row should be identical
        let last = (seq_len - 1) * dim;
        assert_tensors_close(&out_c[last..last + dim], &out_nc[last..last + dim], 1e-5);
    }

    // -----------------------------------------------------------------------
    // OpenCL kernel source
    // -----------------------------------------------------------------------

    #[test]
    fn test_kernel_source_nonempty() {
        let src = kernel_source();
        assert!(!src.is_empty());
    }

    #[test]
    fn test_kernel_source_contains_entry() {
        let src = kernel_source();
        assert!(src.contains("gqa_attention"));
    }

    #[test]
    fn test_kernel_source_contains_group_size() {
        let src = kernel_source();
        assert!(src.contains("group_size"));
    }

    #[test]
    fn test_kernel_source_contains_causal() {
        let src = kernel_source();
        assert!(src.contains("causal"));
    }

    // -----------------------------------------------------------------------
    // GqaError display coverage
    // -----------------------------------------------------------------------

    #[test]
    fn test_error_display_uneven() {
        let e = GqaError::UnevenGrouping { num_q_heads: 7, num_kv_heads: 3 };
        let s = format!("{e}");
        assert!(s.contains("7"));
        assert!(s.contains("3"));
    }

    #[test]
    fn test_error_display_too_many_kv() {
        let e = GqaError::TooManyKvHeads { num_q_heads: 4, num_kv_heads: 8 };
        let s = format!("{e}");
        assert!(s.contains("4"));
        assert!(s.contains("8"));
    }

    #[test]
    fn test_error_display_buffer_mismatch() {
        let e = GqaError::BufferMismatch { expected: 10, actual: 5 };
        let s = format!("{e}");
        assert!(s.contains("10"));
        assert!(s.contains("5"));
    }

    #[test]
    fn test_error_display_seq_too_long() {
        let e = GqaError::SequenceTooLong { seq_len: 1024, max_seq_len: 512 };
        let s = format!("{e}");
        assert!(s.contains("1024"));
        assert!(s.contains("512"));
    }

    #[test]
    fn test_error_is_std_error() {
        let e = GqaError::ZeroDimension("test".into());
        let _: &dyn std::error::Error = &e;
    }
}
