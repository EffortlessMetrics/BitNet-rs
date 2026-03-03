//! Multi-Query / Grouped-Query Attention support for the GPU HAL.
//!
//! Implements MHA (multi-head), MQA (multi-query), and GQA (grouped-query)
//! attention variants with projections, KV head expansion, attention
//! computation, cache size estimation, and type conversion utilities.

use std::fmt;

// ── AttentionType ─────────────────────────────────────────────────────────

/// Describes the attention variant: MHA, MQA, or GQA.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionType {
    /// Multi-Head Attention — K/V have the same number of heads as Q.
    MHA { num_heads: usize },
    /// Multi-Query Attention — K/V share a single head.
    MQA { num_heads: usize },
    /// Grouped-Query Attention — K/V have `num_kv_heads` groups.
    GQA { num_heads: usize, num_kv_heads: usize },
}

impl AttentionType {
    /// Number of query heads.
    pub fn num_heads(&self) -> usize {
        match self {
            Self::MHA { num_heads } | Self::MQA { num_heads } | Self::GQA { num_heads, .. } => {
                *num_heads
            }
        }
    }

    /// Number of key/value heads.
    pub fn num_kv_heads(&self) -> usize {
        match self {
            Self::MHA { num_heads } => *num_heads,
            Self::MQA { .. } => 1,
            Self::GQA { num_kv_heads, .. } => *num_kv_heads,
        }
    }

    /// Detect the canonical type from head counts.
    ///
    /// * `num_kv_heads == num_heads` → MHA
    /// * `num_kv_heads == 1` → MQA
    /// * otherwise → GQA
    pub fn from_head_counts(num_heads: usize, num_kv_heads: usize) -> Self {
        assert!(num_heads > 0, "num_heads must be > 0");
        assert!(num_kv_heads > 0, "num_kv_heads must be > 0");
        assert!(
            num_heads % num_kv_heads == 0,
            "num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        );
        if num_kv_heads == num_heads {
            Self::MHA { num_heads }
        } else if num_kv_heads == 1 {
            Self::MQA { num_heads }
        } else {
            Self::GQA { num_heads, num_kv_heads }
        }
    }

    /// The expansion factor: how many Q heads share each KV head.
    pub fn head_expansion_factor(&self) -> usize {
        self.num_heads() / self.num_kv_heads()
    }
}

impl fmt::Display for AttentionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MHA { num_heads } => write!(f, "MHA(heads={num_heads})"),
            Self::MQA { num_heads } => {
                write!(f, "MQA(q_heads={num_heads}, kv_heads=1)")
            }
            Self::GQA { num_heads, num_kv_heads } => {
                write!(f, "GQA(q_heads={num_heads}, kv_heads={num_kv_heads})")
            }
        }
    }
}

// ── AttentionConfig ───────────────────────────────────────────────────────

/// Full configuration for an attention layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttentionConfig {
    pub head_dim: usize,
    pub attention_type: AttentionType,
    pub dropout: f32,
    pub use_alibi: bool,
    pub use_rope: bool,
    pub max_seq_len: usize,
}

impl AttentionConfig {
    pub fn new(head_dim: usize, attention_type: AttentionType) -> Self {
        Self {
            head_dim,
            attention_type,
            dropout: 0.0,
            use_alibi: false,
            use_rope: false,
            max_seq_len: 2048,
        }
    }

    /// Total model dimension (hidden size) for queries.
    pub fn q_hidden_dim(&self) -> usize {
        self.attention_type.num_heads() * self.head_dim
    }

    /// Total model dimension for keys/values.
    pub fn kv_hidden_dim(&self) -> usize {
        self.attention_type.num_kv_heads() * self.head_dim
    }
}

// ── KVHeadExpander ────────────────────────────────────────────────────────

/// Expands KV heads to match Q head count via repetition.
///
/// For MQA/GQA the K and V tensors have fewer heads than Q. Before
/// computing dot-product attention the KV heads must be repeated so
/// their count matches `num_heads`.
pub struct KVHeadExpander;

impl KVHeadExpander {
    /// Expand `kv` of shape `[kv_heads, seq_len, head_dim]` to
    /// `[num_heads, seq_len, head_dim]` by repeating each KV head.
    ///
    /// `kv.len()` must equal `kv_heads * seq_len * head_dim`.
    pub fn expand(
        kv: &[f32],
        kv_heads: usize,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        assert!(num_heads % kv_heads == 0, "num_heads must be divisible by kv_heads");
        let repeats = num_heads / kv_heads;
        let expected_len = kv_heads * seq_len * head_dim;
        assert_eq!(
            kv.len(),
            expected_len,
            "kv length mismatch: expected {expected_len}, got {}",
            kv.len()
        );

        if repeats == 1 {
            return kv.to_vec();
        }

        let head_size = seq_len * head_dim;
        let mut out = Vec::with_capacity(num_heads * head_size);
        for h in 0..kv_heads {
            let src = &kv[h * head_size..(h + 1) * head_size];
            for _ in 0..repeats {
                out.extend_from_slice(src);
            }
        }
        out
    }
}

// ── Projection helpers ────────────────────────────────────────────────────

/// Result of a Q/K/V projection.
#[derive(Debug, Clone)]
pub struct ProjectedQKV {
    /// `[num_heads, seq_len, head_dim]`
    pub q: Vec<f32>,
    /// `[kv_heads, seq_len, head_dim]`
    pub k: Vec<f32>,
    /// `[kv_heads, seq_len, head_dim]`
    pub v: Vec<f32>,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub seq_len: usize,
    pub head_dim: usize,
}

/// Multi-Head Attention projection (Q, K, V each have `num_heads`).
pub struct MHAProjection {
    pub num_heads: usize,
    pub head_dim: usize,
}

impl MHAProjection {
    pub fn new(num_heads: usize, head_dim: usize) -> Self {
        Self { num_heads, head_dim }
    }

    /// Project `hidden` of shape `[seq_len, 3 * num_heads * head_dim]`.
    pub fn project(&self, hidden: &[f32], seq_len: usize) -> ProjectedQKV {
        let h = self.num_heads * self.head_dim;
        assert_eq!(hidden.len(), seq_len * 3 * h, "MHA hidden length mismatch");
        let mut q = Vec::with_capacity(seq_len * h);
        let mut k = Vec::with_capacity(seq_len * h);
        let mut v = Vec::with_capacity(seq_len * h);
        for t in 0..seq_len {
            let base = t * 3 * h;
            q.extend_from_slice(&hidden[base..base + h]);
            k.extend_from_slice(&hidden[base + h..base + 2 * h]);
            v.extend_from_slice(&hidden[base + 2 * h..base + 3 * h]);
        }
        ProjectedQKV {
            q,
            k,
            v,
            num_heads: self.num_heads,
            num_kv_heads: self.num_heads,
            seq_len,
            head_dim: self.head_dim,
        }
    }
}

/// Multi-Query Attention projection (K/V have 1 head).
pub struct MQAProjection {
    pub num_heads: usize,
    pub head_dim: usize,
}

impl MQAProjection {
    pub fn new(num_heads: usize, head_dim: usize) -> Self {
        Self { num_heads, head_dim }
    }

    /// Project `hidden` of shape `[seq_len, (num_heads + 2) * head_dim]`.
    pub fn project(&self, hidden: &[f32], seq_len: usize) -> ProjectedQKV {
        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.head_dim; // 1 head
        let row = q_dim + 2 * kv_dim;
        assert_eq!(hidden.len(), seq_len * row, "MQA hidden length mismatch");
        let mut q = Vec::with_capacity(seq_len * q_dim);
        let mut k = Vec::with_capacity(seq_len * kv_dim);
        let mut v = Vec::with_capacity(seq_len * kv_dim);
        for t in 0..seq_len {
            let base = t * row;
            q.extend_from_slice(&hidden[base..base + q_dim]);
            k.extend_from_slice(&hidden[base + q_dim..base + q_dim + kv_dim]);
            v.extend_from_slice(&hidden[base + q_dim + kv_dim..base + row]);
        }
        ProjectedQKV {
            q,
            k,
            v,
            num_heads: self.num_heads,
            num_kv_heads: 1,
            seq_len,
            head_dim: self.head_dim,
        }
    }
}

/// Grouped-Query Attention projection (K/V have `num_kv_heads`).
pub struct GQAProjection {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl GQAProjection {
    pub fn new(num_heads: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        assert!(num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads");
        Self { num_heads, num_kv_heads, head_dim }
    }

    /// Project `hidden` of shape
    /// `[seq_len, (num_heads + 2 * num_kv_heads) * head_dim]`.
    pub fn project(&self, hidden: &[f32], seq_len: usize) -> ProjectedQKV {
        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;
        let row = q_dim + 2 * kv_dim;
        assert_eq!(hidden.len(), seq_len * row, "GQA hidden length mismatch");
        let mut q = Vec::with_capacity(seq_len * q_dim);
        let mut k = Vec::with_capacity(seq_len * kv_dim);
        let mut v = Vec::with_capacity(seq_len * kv_dim);
        for t in 0..seq_len {
            let base = t * row;
            q.extend_from_slice(&hidden[base..base + q_dim]);
            k.extend_from_slice(&hidden[base + q_dim..base + q_dim + kv_dim]);
            v.extend_from_slice(&hidden[base + q_dim + kv_dim..base + row]);
        }
        ProjectedQKV {
            q,
            k,
            v,
            num_heads: self.num_heads,
            num_kv_heads: self.num_kv_heads,
            seq_len,
            head_dim: self.head_dim,
        }
    }
}

// ── AttentionComputer ─────────────────────────────────────────────────────

/// Computes scaled dot-product attention for any attention type.
///
/// `score = softmax(Q · K^T / sqrt(head_dim) + bias) · V`
pub struct AttentionComputer {
    pub head_dim: usize,
}

impl AttentionComputer {
    pub fn new(head_dim: usize) -> Self {
        Self { head_dim }
    }

    /// Compute attention output for a single head.
    ///
    /// * `q` — `[seq_q, head_dim]`
    /// * `k` — `[seq_k, head_dim]`
    /// * `v` — `[seq_k, head_dim]`
    /// * `bias` — optional `[seq_q, seq_k]` additive bias (e.g. causal mask)
    ///
    /// Returns `[seq_q, head_dim]`.
    pub fn compute_single_head(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        bias: Option<&[f32]>,
    ) -> Vec<f32> {
        let d = self.head_dim;
        let seq_q = q.len() / d;
        let seq_k = k.len() / d;
        assert_eq!(q.len(), seq_q * d);
        assert_eq!(k.len(), seq_k * d);
        assert_eq!(v.len(), seq_k * d);

        let scale = 1.0 / (d as f32).sqrt();

        // scores = Q · K^T  → [seq_q, seq_k]
        let mut scores = vec![0.0f32; seq_q * seq_k];
        for i in 0..seq_q {
            for j in 0..seq_k {
                let mut dot = 0.0f32;
                for dd in 0..d {
                    dot += q[i * d + dd] * k[j * d + dd];
                }
                scores[i * seq_k + j] = dot * scale;
            }
        }

        // add bias
        if let Some(b) = bias {
            assert_eq!(b.len(), seq_q * seq_k);
            for (s, &bi) in scores.iter_mut().zip(b.iter()) {
                *s += bi;
            }
        }

        // softmax per row
        for i in 0..seq_q {
            let row = &mut scores[i * seq_k..(i + 1) * seq_k];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for v in row.iter_mut() {
                *v = (*v - max).exp();
                sum += *v;
            }
            if sum > 0.0 {
                for v in row.iter_mut() {
                    *v /= sum;
                }
            }
        }

        // output = scores · V  → [seq_q, d]
        let mut out = vec![0.0f32; seq_q * d];
        for i in 0..seq_q {
            for j in 0..seq_k {
                let w = scores[i * seq_k + j];
                for dd in 0..d {
                    out[i * d + dd] += w * v[j * d + dd];
                }
            }
        }
        out
    }

    /// Compute multi-head attention from a `ProjectedQKV`.
    ///
    /// K/V heads are expanded if needed, then per-head attention is
    /// computed and the results concatenated.
    ///
    /// Returns `[seq_len, num_heads * head_dim]`.
    pub fn compute_multihead(&self, projected: &ProjectedQKV, bias: Option<&[f32]>) -> Vec<f32> {
        let d = self.head_dim;
        let seq = projected.seq_len;
        let num_heads = projected.num_heads;
        let head_size = seq * d;

        // Expand K/V if needed
        let k_expanded = if projected.num_kv_heads < num_heads {
            KVHeadExpander::expand(&projected.k, projected.num_kv_heads, num_heads, seq, d)
        } else {
            projected.k.clone()
        };
        let v_expanded = if projected.num_kv_heads < num_heads {
            KVHeadExpander::expand(&projected.v, projected.num_kv_heads, num_heads, seq, d)
        } else {
            projected.v.clone()
        };

        let mut output = Vec::with_capacity(seq * num_heads * d);
        // Compute attention per-head, then interleave into
        // [seq, num_heads * d] layout.
        let mut per_head_outputs: Vec<Vec<f32>> = Vec::with_capacity(num_heads);
        for h in 0..num_heads {
            let q_h = &projected.q[h * head_size..(h + 1) * head_size];
            let k_h = &k_expanded[h * head_size..(h + 1) * head_size];
            let v_h = &v_expanded[h * head_size..(h + 1) * head_size];
            per_head_outputs.push(self.compute_single_head(q_h, k_h, v_h, bias));
        }

        for t in 0..seq {
            for h in 0..num_heads {
                output.extend_from_slice(&per_head_outputs[h][t * d..(t + 1) * d]);
            }
        }
        output
    }
}

// ── KVCacheSizeEstimator ──────────────────────────────────────────────────

/// Estimates KV cache memory in bytes for a given attention configuration.
pub struct KVCacheSizeEstimator;

impl KVCacheSizeEstimator {
    /// Estimate KV cache size in bytes for `f32` storage.
    ///
    /// Formula: `2 * num_layers * num_kv_heads * max_seq_len * head_dim * 4`
    pub fn estimate_bytes(
        attention_type: &AttentionType,
        num_layers: usize,
        max_seq_len: usize,
        head_dim: usize,
    ) -> usize {
        let kv_heads = attention_type.num_kv_heads();
        2 * num_layers * kv_heads * max_seq_len * head_dim * size_of::<f32>()
    }

    /// Estimate KV cache size in bytes for `f16` storage.
    pub fn estimate_bytes_f16(
        attention_type: &AttentionType,
        num_layers: usize,
        max_seq_len: usize,
        head_dim: usize,
    ) -> usize {
        let kv_heads = attention_type.num_kv_heads();
        2 * num_layers * kv_heads * max_seq_len * head_dim * 2
    }
}

// ── AttentionConverter ────────────────────────────────────────────────────

/// Converts between attention types by merging or splitting heads.
pub struct AttentionConverter;

impl AttentionConverter {
    /// Convert MHA → GQA by averaging groups of consecutive KV heads.
    ///
    /// `kv` has shape `[num_heads, seq_len, head_dim]` and the output
    /// has shape `[num_kv_groups, seq_len, head_dim]`.
    pub fn mha_to_gqa(
        kv: &[f32],
        num_heads: usize,
        num_kv_groups: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        assert!(num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups");
        let group_size = num_heads / num_kv_groups;
        let head_elems = seq_len * head_dim;
        assert_eq!(kv.len(), num_heads * head_elems);

        let mut out = vec![0.0f32; num_kv_groups * head_elems];
        for g in 0..num_kv_groups {
            for h in 0..group_size {
                let src_head = g * group_size + h;
                let src = &kv[src_head * head_elems..(src_head + 1) * head_elems];
                let dst = &mut out[g * head_elems..(g + 1) * head_elems];
                for (d, s) in dst.iter_mut().zip(src.iter()) {
                    *d += s / group_size as f32;
                }
            }
        }
        out
    }

    /// Convert GQA → MQA by averaging all KV groups into a single head.
    pub fn gqa_to_mqa(
        kv: &[f32],
        num_kv_groups: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        Self::mha_to_gqa(kv, num_kv_groups, 1, seq_len, head_dim)
    }

    /// Convert MHA → MQA by averaging all heads into one.
    pub fn mha_to_mqa(kv: &[f32], num_heads: usize, seq_len: usize, head_dim: usize) -> Vec<f32> {
        Self::mha_to_gqa(kv, num_heads, 1, seq_len, head_dim)
    }
}

// ── MQAMetrics ────────────────────────────────────────────────────────────

/// Metrics for an attention configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct MQAMetrics {
    /// Percentage of KV memory saved relative to MHA.
    pub kv_memory_saved_pct: f64,
    /// Effective number of KV heads.
    pub effective_num_kv_heads: usize,
    /// How many Q heads share each KV head.
    pub head_expansion_factor: usize,
}

impl MQAMetrics {
    /// Compute metrics for the given attention type.
    pub fn compute(attention_type: &AttentionType) -> Self {
        let num_heads = attention_type.num_heads();
        let kv_heads = attention_type.num_kv_heads();
        let expansion = num_heads / kv_heads;
        let saved = 1.0 - (kv_heads as f64 / num_heads as f64);
        Self {
            kv_memory_saved_pct: saved * 100.0,
            effective_num_kv_heads: kv_heads,
            head_expansion_factor: expansion,
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── AttentionType tests ───────────────────────────────────────────

    #[test]
    fn attention_type_mha_heads() {
        let t = AttentionType::MHA { num_heads: 32 };
        assert_eq!(t.num_heads(), 32);
        assert_eq!(t.num_kv_heads(), 32);
        assert_eq!(t.head_expansion_factor(), 1);
    }

    #[test]
    fn attention_type_mqa_heads() {
        let t = AttentionType::MQA { num_heads: 32 };
        assert_eq!(t.num_heads(), 32);
        assert_eq!(t.num_kv_heads(), 1);
        assert_eq!(t.head_expansion_factor(), 32);
    }

    #[test]
    fn attention_type_gqa_heads() {
        let t = AttentionType::GQA { num_heads: 32, num_kv_heads: 8 };
        assert_eq!(t.num_heads(), 32);
        assert_eq!(t.num_kv_heads(), 8);
        assert_eq!(t.head_expansion_factor(), 4);
    }

    #[test]
    fn from_head_counts_detects_mha() {
        let t = AttentionType::from_head_counts(16, 16);
        assert_eq!(t, AttentionType::MHA { num_heads: 16 });
    }

    #[test]
    fn from_head_counts_detects_mqa() {
        let t = AttentionType::from_head_counts(16, 1);
        assert_eq!(t, AttentionType::MQA { num_heads: 16 });
    }

    #[test]
    fn from_head_counts_detects_gqa() {
        let t = AttentionType::from_head_counts(32, 8);
        assert_eq!(t, AttentionType::GQA { num_heads: 32, num_kv_heads: 8 });
    }

    #[test]
    #[should_panic(expected = "must be divisible by num_kv_heads")]
    fn from_head_counts_rejects_indivisible() {
        AttentionType::from_head_counts(32, 5);
    }

    #[test]
    #[should_panic(expected = "num_heads must be > 0")]
    fn from_head_counts_rejects_zero_heads() {
        AttentionType::from_head_counts(0, 0);
    }

    #[test]
    fn attention_type_display_mha() {
        let t = AttentionType::MHA { num_heads: 32 };
        assert_eq!(t.to_string(), "MHA(heads=32)");
    }

    #[test]
    fn attention_type_display_mqa() {
        let t = AttentionType::MQA { num_heads: 16 };
        assert_eq!(t.to_string(), "MQA(q_heads=16, kv_heads=1)");
    }

    #[test]
    fn attention_type_display_gqa() {
        let t = AttentionType::GQA { num_heads: 32, num_kv_heads: 4 };
        assert_eq!(t.to_string(), "GQA(q_heads=32, kv_heads=4)");
    }

    // ── Edge case: GQA where kv_heads == num_heads is MHA ─────────────

    #[test]
    fn gqa_with_equal_heads_is_mha() {
        let t = AttentionType::from_head_counts(8, 8);
        assert!(matches!(t, AttentionType::MHA { num_heads: 8 }));
    }

    #[test]
    fn gqa_with_one_kv_head_is_mqa() {
        let t = AttentionType::from_head_counts(8, 1);
        assert!(matches!(t, AttentionType::MQA { num_heads: 8 }));
    }

    // ── AttentionConfig tests ─────────────────────────────────────────

    #[test]
    fn config_q_hidden_dim_mha() {
        let cfg = AttentionConfig::new(64, AttentionType::MHA { num_heads: 32 });
        assert_eq!(cfg.q_hidden_dim(), 2048);
        assert_eq!(cfg.kv_hidden_dim(), 2048);
    }

    #[test]
    fn config_kv_hidden_dim_mqa() {
        let cfg = AttentionConfig::new(64, AttentionType::MQA { num_heads: 32 });
        assert_eq!(cfg.q_hidden_dim(), 2048);
        assert_eq!(cfg.kv_hidden_dim(), 64);
    }

    #[test]
    fn config_kv_hidden_dim_gqa() {
        let cfg = AttentionConfig::new(128, AttentionType::GQA { num_heads: 32, num_kv_heads: 8 });
        assert_eq!(cfg.q_hidden_dim(), 4096);
        assert_eq!(cfg.kv_hidden_dim(), 1024);
    }

    #[test]
    fn config_defaults() {
        let cfg = AttentionConfig::new(64, AttentionType::MHA { num_heads: 8 });
        assert_eq!(cfg.dropout, 0.0);
        assert!(!cfg.use_alibi);
        assert!(!cfg.use_rope);
        assert_eq!(cfg.max_seq_len, 2048);
    }

    // ── KVHeadExpander tests ──────────────────────────────────────────

    #[test]
    fn expand_identity_when_equal_heads() {
        let kv = vec![1.0, 2.0, 3.0, 4.0]; // 1 head, seq=2, dim=2
        let out = KVHeadExpander::expand(&kv, 1, 1, 2, 2);
        assert_eq!(out, kv);
    }

    #[test]
    fn expand_mqa_single_to_four_heads() {
        // 1 kv head, seq=1, dim=2 → expand to 4 heads
        let kv = vec![1.0, 2.0];
        let out = KVHeadExpander::expand(&kv, 1, 4, 1, 2);
        assert_eq!(out, vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn expand_gqa_two_to_eight_heads() {
        // 2 kv heads, seq=1, dim=2 → expand to 8 heads (4× each)
        let kv = vec![1.0, 2.0, 3.0, 4.0];
        let out = KVHeadExpander::expand(&kv, 2, 8, 1, 2);
        assert_eq!(out.len(), 16);
        // head 0 repeated 4×, then head 1 repeated 4×
        assert_eq!(&out[0..2], &[1.0, 2.0]);
        assert_eq!(&out[2..4], &[1.0, 2.0]);
        assert_eq!(&out[4..6], &[1.0, 2.0]);
        assert_eq!(&out[6..8], &[1.0, 2.0]);
        assert_eq!(&out[8..10], &[3.0, 4.0]);
        assert_eq!(&out[10..12], &[3.0, 4.0]);
        assert_eq!(&out[12..14], &[3.0, 4.0]);
        assert_eq!(&out[14..16], &[3.0, 4.0]);
    }

    #[test]
    fn expand_preserves_seq_dim() {
        // 1 kv head, seq=3, dim=2 → 2 heads
        let kv = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = KVHeadExpander::expand(&kv, 1, 2, 3, 2);
        assert_eq!(out.len(), 12);
        // Both heads identical
        assert_eq!(&out[0..6], &kv[..]);
        assert_eq!(&out[6..12], &kv[..]);
    }

    #[test]
    #[should_panic(expected = "num_heads must be divisible by kv_heads")]
    fn expand_rejects_indivisible() {
        KVHeadExpander::expand(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 5, 1, 2);
    }

    #[test]
    #[should_panic(expected = "kv length mismatch")]
    fn expand_rejects_wrong_length() {
        KVHeadExpander::expand(&[1.0], 1, 2, 1, 2);
    }

    // ── MHAProjection tests ───────────────────────────────────────────

    #[test]
    fn mha_projection_shapes() {
        let proj = MHAProjection::new(4, 8);
        // seq=2, 3 * 4 * 8 = 96 per step → 192 total
        let hidden = vec![0.5; 2 * 3 * 4 * 8];
        let qkv = proj.project(&hidden, 2);
        assert_eq!(qkv.q.len(), 2 * 4 * 8);
        assert_eq!(qkv.k.len(), 2 * 4 * 8);
        assert_eq!(qkv.v.len(), 2 * 4 * 8);
        assert_eq!(qkv.num_heads, 4);
        assert_eq!(qkv.num_kv_heads, 4);
    }

    #[test]
    fn mha_projection_values() {
        let proj = MHAProjection::new(2, 2);
        // seq=1, row = 3*2*2 = 12
        let hidden: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let qkv = proj.project(&hidden, 1);
        assert_eq!(qkv.q, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(qkv.k, &[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(qkv.v, &[9.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    #[should_panic(expected = "MHA hidden length mismatch")]
    fn mha_projection_rejects_bad_length() {
        let proj = MHAProjection::new(4, 8);
        proj.project(&[1.0; 10], 1);
    }

    // ── MQAProjection tests ───────────────────────────────────────────

    #[test]
    fn mqa_projection_shapes() {
        let proj = MQAProjection::new(8, 64);
        // row = (8 + 2) * 64 = 640
        let hidden = vec![0.5; 2 * 640];
        let qkv = proj.project(&hidden, 2);
        assert_eq!(qkv.q.len(), 2 * 8 * 64);
        assert_eq!(qkv.k.len(), 2 * 64); // 1 head
        assert_eq!(qkv.v.len(), 2 * 64); // 1 head
        assert_eq!(qkv.num_heads, 8);
        assert_eq!(qkv.num_kv_heads, 1);
    }

    #[test]
    fn mqa_projection_kv_single_head() {
        let proj = MQAProjection::new(4, 2);
        // row = (4 + 2) * 2 = 12
        let hidden: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let qkv = proj.project(&hidden, 1);
        assert_eq!(qkv.q.len(), 8); // 4 heads * 2 dim
        assert_eq!(qkv.k.len(), 2); // 1 head * 2 dim
        assert_eq!(qkv.v.len(), 2); // 1 head * 2 dim
        assert_eq!(qkv.k, &[9.0, 10.0]);
        assert_eq!(qkv.v, &[11.0, 12.0]);
    }

    #[test]
    #[should_panic(expected = "MQA hidden length mismatch")]
    fn mqa_projection_rejects_bad_length() {
        let proj = MQAProjection::new(8, 64);
        proj.project(&[1.0; 10], 1);
    }

    // ── GQAProjection tests ───────────────────────────────────────────

    #[test]
    fn gqa_projection_shapes() {
        let proj = GQAProjection::new(32, 8, 64);
        // row = (32 + 2*8) * 64 = 3072
        let hidden = vec![0.5; 1 * 3072];
        let qkv = proj.project(&hidden, 1);
        assert_eq!(qkv.q.len(), 32 * 64);
        assert_eq!(qkv.k.len(), 8 * 64);
        assert_eq!(qkv.v.len(), 8 * 64);
        assert_eq!(qkv.num_heads, 32);
        assert_eq!(qkv.num_kv_heads, 8);
    }

    #[test]
    fn gqa_projection_values() {
        let proj = GQAProjection::new(4, 2, 2);
        // row = (4 + 2*2) * 2 = 16
        let hidden: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let qkv = proj.project(&hidden, 1);
        assert_eq!(qkv.q, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(qkv.k, &[9.0, 10.0, 11.0, 12.0]);
        assert_eq!(qkv.v, &[13.0, 14.0, 15.0, 16.0]);
    }

    #[test]
    fn gqa_projection_seq_gt_one() {
        let proj = GQAProjection::new(4, 2, 2);
        // row = 16, seq = 2 → 32 total
        let hidden: Vec<f32> = (1..=32).map(|x| x as f32).collect();
        let qkv = proj.project(&hidden, 2);
        assert_eq!(qkv.q.len(), 2 * 4 * 2);
        assert_eq!(qkv.k.len(), 2 * 2 * 2);
        assert_eq!(qkv.v.len(), 2 * 2 * 2);
        assert_eq!(qkv.seq_len, 2);
    }

    #[test]
    #[should_panic(expected = "num_heads must be divisible by num_kv_heads")]
    fn gqa_projection_rejects_indivisible() {
        GQAProjection::new(7, 3, 64);
    }

    #[test]
    #[should_panic(expected = "GQA hidden length mismatch")]
    fn gqa_projection_rejects_bad_length() {
        let proj = GQAProjection::new(8, 4, 64);
        proj.project(&[1.0; 10], 1);
    }

    // ── AttentionComputer: single head ────────────────────────────────

    #[test]
    fn single_head_attention_output_shape() {
        let comp = AttentionComputer::new(4);
        let q = vec![1.0; 2 * 4]; // seq_q=2
        let k = vec![1.0; 3 * 4]; // seq_k=3
        let v = vec![1.0; 3 * 4];
        let out = comp.compute_single_head(&q, &k, &v, None);
        assert_eq!(out.len(), 2 * 4);
    }

    #[test]
    fn single_head_identity_attention() {
        // seq_q=1, seq_k=1, dim=2 → output should equal V
        let comp = AttentionComputer::new(2);
        let q = vec![1.0, 0.0];
        let k = vec![1.0, 0.0];
        let v = vec![3.0, 7.0];
        let out = comp.compute_single_head(&q, &k, &v, None);
        assert!((out[0] - 3.0).abs() < 1e-5);
        assert!((out[1] - 7.0).abs() < 1e-5);
    }

    #[test]
    fn single_head_uniform_attention() {
        // seq_q=1, seq_k=2, q/k equal → softmax gives ~0.5 each
        let comp = AttentionComputer::new(2);
        let q = vec![1.0, 1.0];
        let k = vec![1.0, 1.0, 1.0, 1.0]; // 2 keys, identical
        let v = vec![2.0, 0.0, 0.0, 4.0]; // 2 values
        let out = comp.compute_single_head(&q, &k, &v, None);
        // output ≈ 0.5 * [2,0] + 0.5 * [0,4] = [1, 2]
        assert!((out[0] - 1.0).abs() < 1e-5);
        assert!((out[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn single_head_with_causal_bias() {
        // seq_q=2, seq_k=2, causal mask: position 0 sees only key 0
        let comp = AttentionComputer::new(1);
        let q = vec![1.0, 1.0];
        let k = vec![1.0, 1.0];
        let v = vec![10.0, 20.0];
        // Causal: [0, -inf; 0, 0]
        let bias = vec![0.0, f32::NEG_INFINITY, 0.0, 0.0];
        let out = comp.compute_single_head(&q, &k, &v, Some(&bias));
        // Row 0: only key 0 visible → output = 10
        assert!((out[0] - 10.0).abs() < 1e-4);
        // Row 1: both visible, equal scores → mean(10, 20) = 15
        assert!((out[1] - 15.0).abs() < 1e-4);
    }

    // ── AttentionComputer: multi-head ─────────────────────────────────

    #[test]
    fn multihead_mha_output_shape() {
        let comp = AttentionComputer::new(4);
        let proj = MHAProjection::new(2, 4);
        let hidden = vec![0.1; 1 * 3 * 2 * 4]; // seq=1
        let qkv = proj.project(&hidden, 1);
        let out = comp.compute_multihead(&qkv, None);
        assert_eq!(out.len(), 1 * 2 * 4); // seq * num_heads * head_dim
    }

    #[test]
    fn multihead_mqa_output_shape() {
        let comp = AttentionComputer::new(4);
        let proj = MQAProjection::new(8, 4);
        // row = (8+2)*4 = 40, seq=2
        let hidden = vec![0.1; 2 * 40];
        let qkv = proj.project(&hidden, 2);
        let out = comp.compute_multihead(&qkv, None);
        assert_eq!(out.len(), 2 * 8 * 4);
    }

    #[test]
    fn multihead_gqa_output_shape() {
        let comp = AttentionComputer::new(4);
        let proj = GQAProjection::new(8, 2, 4);
        // row = (8 + 2*2)*4 = 48, seq=1
        let hidden = vec![0.1; 1 * 48];
        let qkv = proj.project(&hidden, 1);
        let out = comp.compute_multihead(&qkv, None);
        assert_eq!(out.len(), 1 * 8 * 4);
    }

    #[test]
    fn multihead_all_types_produce_same_shape() {
        let head_dim = 4;
        let num_heads = 8;
        let seq = 2;

        let comp = AttentionComputer::new(head_dim);

        // MHA
        let mha = MHAProjection::new(num_heads, head_dim);
        let h_mha = vec![0.1; seq * 3 * num_heads * head_dim];
        let q_mha = mha.project(&h_mha, seq);
        let o_mha = comp.compute_multihead(&q_mha, None);

        // MQA
        let mqa = MQAProjection::new(num_heads, head_dim);
        let h_mqa = vec![0.1; seq * (num_heads + 2) * head_dim];
        let q_mqa = mqa.project(&h_mqa, seq);
        let o_mqa = comp.compute_multihead(&q_mqa, None);

        // GQA
        let gqa = GQAProjection::new(num_heads, 4, head_dim);
        let h_gqa = vec![0.1; seq * (num_heads + 2 * 4) * head_dim];
        let q_gqa = gqa.project(&h_gqa, seq);
        let o_gqa = comp.compute_multihead(&q_gqa, None);

        let expected = seq * num_heads * head_dim;
        assert_eq!(o_mha.len(), expected);
        assert_eq!(o_mqa.len(), expected);
        assert_eq!(o_gqa.len(), expected);
    }

    // ── KVCacheSizeEstimator tests ────────────────────────────────────

    #[test]
    fn kv_cache_mha() {
        let t = AttentionType::MHA { num_heads: 32 };
        let bytes = KVCacheSizeEstimator::estimate_bytes(&t, 24, 2048, 128);
        // 2 * 24 * 32 * 2048 * 128 * 4
        assert_eq!(bytes, 2 * 24 * 32 * 2048 * 128 * 4);
    }

    #[test]
    fn kv_cache_mqa() {
        let t = AttentionType::MQA { num_heads: 32 };
        let bytes = KVCacheSizeEstimator::estimate_bytes(&t, 24, 2048, 128);
        // 2 * 24 * 1 * 2048 * 128 * 4
        assert_eq!(bytes, 2 * 24 * 1 * 2048 * 128 * 4);
    }

    #[test]
    fn kv_cache_gqa() {
        let t = AttentionType::GQA { num_heads: 32, num_kv_heads: 8 };
        let bytes = KVCacheSizeEstimator::estimate_bytes(&t, 24, 2048, 128);
        assert_eq!(bytes, 2 * 24 * 8 * 2048 * 128 * 4);
    }

    #[test]
    fn kv_cache_mqa_is_smallest() {
        let layers = 24;
        let seq = 2048;
        let dim = 128;
        let mha = KVCacheSizeEstimator::estimate_bytes(
            &AttentionType::MHA { num_heads: 32 },
            layers,
            seq,
            dim,
        );
        let gqa = KVCacheSizeEstimator::estimate_bytes(
            &AttentionType::GQA { num_heads: 32, num_kv_heads: 8 },
            layers,
            seq,
            dim,
        );
        let mqa = KVCacheSizeEstimator::estimate_bytes(
            &AttentionType::MQA { num_heads: 32 },
            layers,
            seq,
            dim,
        );
        assert!(mqa < gqa);
        assert!(gqa < mha);
    }

    #[test]
    fn kv_cache_f16_half_of_f32() {
        let t = AttentionType::MHA { num_heads: 8 };
        let f32_bytes = KVCacheSizeEstimator::estimate_bytes(&t, 12, 1024, 64);
        let f16_bytes = KVCacheSizeEstimator::estimate_bytes_f16(&t, 12, 1024, 64);
        assert_eq!(f16_bytes * 2, f32_bytes);
    }

    #[test]
    fn kv_cache_ratio_mha_vs_mqa() {
        let layers = 32;
        let seq = 4096;
        let dim = 128;
        let num_heads = 32;
        let mha = KVCacheSizeEstimator::estimate_bytes(
            &AttentionType::MHA { num_heads },
            layers,
            seq,
            dim,
        );
        let mqa = KVCacheSizeEstimator::estimate_bytes(
            &AttentionType::MQA { num_heads },
            layers,
            seq,
            dim,
        );
        assert_eq!(mha / mqa, num_heads);
    }

    // ── AttentionConverter tests ──────────────────────────────────────

    #[test]
    fn mha_to_gqa_averages_groups() {
        // 4 heads → 2 groups: avg heads (0,1) and (2,3)
        // seq=1, dim=2
        let kv = vec![
            2.0, 4.0, // head 0
            6.0, 8.0, // head 1
            10.0, 12.0, // head 2
            14.0, 16.0, // head 3
        ];
        let out = AttentionConverter::mha_to_gqa(&kv, 4, 2, 1, 2);
        assert_eq!(out.len(), 4);
        // group 0 = avg(head0, head1) = (4, 6)
        assert!((out[0] - 4.0).abs() < 1e-5);
        assert!((out[1] - 6.0).abs() < 1e-5);
        // group 1 = avg(head2, head3) = (12, 14)
        assert!((out[2] - 12.0).abs() < 1e-5);
        assert!((out[3] - 14.0).abs() < 1e-5);
    }

    #[test]
    fn gqa_to_mqa_averages_all_groups() {
        // 4 kv groups → 1 head
        let kv = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let out = AttentionConverter::gqa_to_mqa(&kv, 4, 1, 2);
        assert_eq!(out.len(), 2);
        // avg of 4 heads: (1+3+5+7)/4=4, (2+4+6+8)/4=5
        assert!((out[0] - 4.0).abs() < 1e-5);
        assert!((out[1] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn mha_to_mqa_single_head() {
        let kv = vec![2.0, 4.0, 6.0, 8.0]; // 2 heads, seq=1, dim=2
        let out = AttentionConverter::mha_to_mqa(&kv, 2, 1, 2);
        assert_eq!(out.len(), 2);
        assert!((out[0] - 4.0).abs() < 1e-5);
        assert!((out[1] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn mha_to_gqa_identity_when_equal() {
        // 4 heads → 4 groups (identity)
        let kv = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let out = AttentionConverter::mha_to_gqa(&kv, 4, 4, 1, 2);
        for (a, b) in out.iter().zip(kv.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    #[should_panic(expected = "num_heads must be divisible by num_kv_groups")]
    fn converter_rejects_indivisible() {
        AttentionConverter::mha_to_gqa(&[1.0; 6], 3, 2, 1, 2);
    }

    #[test]
    fn converter_preserves_seq_dim() {
        // 2 heads, seq=2, dim=1 → 1 group
        let kv = vec![10.0, 20.0, 30.0, 40.0];
        let out = AttentionConverter::mha_to_mqa(&kv, 2, 2, 1);
        assert_eq!(out.len(), 2);
        // seq0: avg(10,30)=20, seq1: avg(20,40)=30
        assert!((out[0] - 20.0).abs() < 1e-5);
        assert!((out[1] - 30.0).abs() < 1e-5);
    }

    // ── MQAMetrics tests ──────────────────────────────────────────────

    #[test]
    fn metrics_mha_zero_savings() {
        let m = MQAMetrics::compute(&AttentionType::MHA { num_heads: 32 });
        assert!((m.kv_memory_saved_pct - 0.0).abs() < 1e-5);
        assert_eq!(m.effective_num_kv_heads, 32);
        assert_eq!(m.head_expansion_factor, 1);
    }

    #[test]
    fn metrics_mqa_max_savings() {
        let m = MQAMetrics::compute(&AttentionType::MQA { num_heads: 32 });
        let expected_pct = (1.0 - 1.0 / 32.0) * 100.0;
        assert!((m.kv_memory_saved_pct - expected_pct).abs() < 1e-5);
        assert_eq!(m.effective_num_kv_heads, 1);
        assert_eq!(m.head_expansion_factor, 32);
    }

    #[test]
    fn metrics_gqa_intermediate_savings() {
        let m = MQAMetrics::compute(&AttentionType::GQA { num_heads: 32, num_kv_heads: 8 });
        let expected_pct = (1.0 - 8.0 / 32.0) * 100.0;
        assert!((m.kv_memory_saved_pct - expected_pct).abs() < 1e-5);
        assert_eq!(m.effective_num_kv_heads, 8);
        assert_eq!(m.head_expansion_factor, 4);
    }

    #[test]
    fn metrics_gqa_savings_between_mha_and_mqa() {
        let mha = MQAMetrics::compute(&AttentionType::MHA { num_heads: 32 });
        let gqa = MQAMetrics::compute(&AttentionType::GQA { num_heads: 32, num_kv_heads: 8 });
        let mqa = MQAMetrics::compute(&AttentionType::MQA { num_heads: 32 });
        assert!(gqa.kv_memory_saved_pct > mha.kv_memory_saved_pct);
        assert!(gqa.kv_memory_saved_pct < mqa.kv_memory_saved_pct);
    }

    #[test]
    fn metrics_expansion_factor_various() {
        let cases: Vec<(AttentionType, usize)> = vec![
            (AttentionType::MHA { num_heads: 16 }, 1),
            (AttentionType::MQA { num_heads: 16 }, 16),
            (AttentionType::GQA { num_heads: 16, num_kv_heads: 4 }, 4),
            (AttentionType::GQA { num_heads: 16, num_kv_heads: 2 }, 8),
        ];
        for (at, expected) in &cases {
            let m = MQAMetrics::compute(at);
            assert_eq!(m.head_expansion_factor, *expected, "failed for {at}");
        }
    }

    // ── Integration-style tests ───────────────────────────────────────

    #[test]
    fn full_pipeline_mha() {
        let heads = 4;
        let dim = 2;
        let seq = 1;
        let proj = MHAProjection::new(heads, dim);
        let hidden = vec![1.0; seq * 3 * heads * dim];
        let qkv = proj.project(&hidden, seq);
        let comp = AttentionComputer::new(dim);
        let out = comp.compute_multihead(&qkv, None);
        assert_eq!(out.len(), seq * heads * dim);
        // uniform inputs → output should be all 1.0
        for v in &out {
            assert!((*v - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn full_pipeline_mqa() {
        let heads = 4;
        let dim = 2;
        let seq = 1;
        let proj = MQAProjection::new(heads, dim);
        let hidden = vec![1.0; seq * (heads + 2) * dim];
        let qkv = proj.project(&hidden, seq);
        assert_eq!(qkv.num_kv_heads, 1);
        let comp = AttentionComputer::new(dim);
        let out = comp.compute_multihead(&qkv, None);
        assert_eq!(out.len(), seq * heads * dim);
    }

    #[test]
    fn full_pipeline_gqa() {
        let heads = 8;
        let kv_heads = 2;
        let dim = 4;
        let seq = 2;
        let proj = GQAProjection::new(heads, kv_heads, dim);
        let hidden = vec![0.5; seq * (heads + 2 * kv_heads) * dim];
        let qkv = proj.project(&hidden, seq);
        assert_eq!(qkv.num_kv_heads, 2);
        let comp = AttentionComputer::new(dim);
        let out = comp.compute_multihead(&qkv, None);
        assert_eq!(out.len(), seq * heads * dim);
    }

    #[test]
    fn full_pipeline_with_cache_estimation() {
        let t = AttentionType::GQA { num_heads: 32, num_kv_heads: 8 };
        let cfg = AttentionConfig::new(128, t);
        let cache = KVCacheSizeEstimator::estimate_bytes(&t, 32, cfg.max_seq_len, cfg.head_dim);
        let metrics = MQAMetrics::compute(&t);
        // GQA with 8 kv heads saves 75%
        assert!((metrics.kv_memory_saved_pct - 75.0).abs() < 1e-5);
        // Cache should be non-zero
        assert!(cache > 0);
    }

    #[test]
    fn round_trip_mha_to_gqa_expand() {
        // MHA → GQA conversion then expansion should give equal-length output
        let heads = 4;
        let kv_groups = 2;
        let seq = 1;
        let dim = 2;
        let kv: Vec<f32> = (0..heads * seq * dim).map(|i| i as f32).collect();
        let gqa_kv = AttentionConverter::mha_to_gqa(&kv, heads, kv_groups, seq, dim);
        assert_eq!(gqa_kv.len(), kv_groups * seq * dim);
        let expanded = KVHeadExpander::expand(&gqa_kv, kv_groups, heads, seq, dim);
        assert_eq!(expanded.len(), kv.len());
    }

    #[test]
    fn from_head_counts_gqa_two_groups() {
        let t = AttentionType::from_head_counts(64, 2);
        assert_eq!(t, AttentionType::GQA { num_heads: 64, num_kv_heads: 2 });
        assert_eq!(t.head_expansion_factor(), 32);
    }

    #[test]
    fn config_with_rope_and_alibi() {
        let mut cfg = AttentionConfig::new(64, AttentionType::MQA { num_heads: 16 });
        cfg.use_rope = true;
        cfg.use_alibi = true;
        cfg.max_seq_len = 4096;
        assert!(cfg.use_rope);
        assert!(cfg.use_alibi);
        assert_eq!(cfg.max_seq_len, 4096);
    }

    #[test]
    fn config_with_dropout() {
        let mut cfg =
            AttentionConfig::new(128, AttentionType::GQA { num_heads: 32, num_kv_heads: 4 });
        cfg.dropout = 0.1;
        assert!((cfg.dropout - 0.1).abs() < 1e-7);
    }

    #[test]
    fn expand_large_head_count() {
        // Stress: 1 kv head → 64 heads
        let kv = vec![42.0; 4]; // seq=2, dim=2
        let out = KVHeadExpander::expand(&kv, 1, 64, 2, 2);
        assert_eq!(out.len(), 64 * 4);
        for chunk in out.chunks(4) {
            assert_eq!(chunk, &[42.0, 42.0, 42.0, 42.0]);
        }
    }

    #[test]
    fn kv_cache_single_layer() {
        let t = AttentionType::MQA { num_heads: 16 };
        let bytes = KVCacheSizeEstimator::estimate_bytes(&t, 1, 512, 64);
        assert_eq!(bytes, 2 * 1 * 1 * 512 * 64 * 4);
    }

    #[test]
    fn attention_type_clone_eq() {
        let a = AttentionType::GQA { num_heads: 32, num_kv_heads: 8 };
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn projected_qkv_clone() {
        let proj = MHAProjection::new(2, 2);
        let hidden = vec![1.0; 12];
        let qkv = proj.project(&hidden, 1);
        let qkv2 = qkv.clone();
        assert_eq!(qkv.q, qkv2.q);
        assert_eq!(qkv.k, qkv2.k);
        assert_eq!(qkv.v, qkv2.v);
    }

    #[test]
    fn metrics_debug_display() {
        let m = MQAMetrics::compute(&AttentionType::MQA { num_heads: 8 });
        let dbg = format!("{m:?}");
        assert!(dbg.contains("kv_memory_saved_pct"));
    }
}
