//! Complete CPU multi-head attention (MHA) and grouped-query attention (GQA).
//!
//! Combines existing primitives (matmul, softmax, causal mask) into the
//! full MHA / GQA operation.  Splits Q, K, V into per-head slices,
//! computes scaled dot-product attention for each head, then concatenates
//! the results.
//!
//! Delegates the per-head dot-product to [`AttentionKernel::scaled_dot_product`]
//! which performs runtime AVX2 dispatch on x86_64.

use bitnet_common::{BitNetError, KernelError, Result};

use super::attention::{AttentionKernel, causal_mask};

// ── Configuration ──────────────────────────────────────────────────

/// Configuration for multi-head attention.
#[derive(Debug, Clone)]
pub struct MhaConfig {
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimensionality of each head.
    pub head_dim: usize,
    /// Whether to apply a causal (upper-triangular) mask so that each
    /// position can only attend to itself and earlier positions.
    pub use_causal_mask: bool,
    /// Dropout rate (reserved for future use; currently unused).
    /// Must be in `[0.0, 1.0)`.
    pub dropout_rate: f32,
}

impl MhaConfig {
    /// Total model dimension (`num_heads * head_dim`).
    #[inline]
    pub fn model_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }

    fn validate(&self) -> Result<()> {
        if self.num_heads == 0 {
            return Err(invalid_arg("num_heads must be > 0"));
        }
        if self.head_dim == 0 {
            return Err(invalid_arg("head_dim must be > 0"));
        }
        if !(0.0..1.0).contains(&self.dropout_rate) {
            return Err(invalid_arg("dropout_rate must be in [0.0, 1.0)"));
        }
        Ok(())
    }
}

/// Configuration for grouped-query attention where key/value heads
/// are shared among query heads.
#[derive(Debug, Clone)]
pub struct GqaConfig {
    /// Number of query heads.
    pub num_heads: usize,
    /// Number of key/value heads (must evenly divide `num_heads`).
    pub num_kv_heads: usize,
    /// Dimensionality of each head.
    pub head_dim: usize,
    /// Whether to apply a causal mask.
    pub use_causal_mask: bool,
    /// Dropout rate (reserved for future use; currently unused).
    pub dropout_rate: f32,
}

impl GqaConfig {
    fn validate(&self) -> Result<()> {
        if self.num_heads == 0 {
            return Err(invalid_arg("num_heads must be > 0"));
        }
        if self.num_kv_heads == 0 {
            return Err(invalid_arg("num_kv_heads must be > 0"));
        }
        if self.head_dim == 0 {
            return Err(invalid_arg("head_dim must be > 0"));
        }
        if !self.num_heads.is_multiple_of(self.num_kv_heads) {
            return Err(invalid_arg("num_heads must be a multiple of num_kv_heads"));
        }
        if !(0.0..1.0).contains(&self.dropout_rate) {
            return Err(invalid_arg("dropout_rate must be in [0.0, 1.0)"));
        }
        Ok(())
    }
}

fn invalid_arg(reason: &str) -> BitNetError {
    BitNetError::Kernel(KernelError::InvalidArguments { reason: reason.to_string() })
}

// ── Head extraction / scatter ──────────────────────────────────────

/// Extract head `h` from `[seq_len, num_heads * head_dim]` into
/// a contiguous `[seq_len, head_dim]` buffer.
fn extract_head(
    data: &[f32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    h: usize,
) -> Vec<f32> {
    let stride = num_heads * head_dim;
    let mut head = Vec::with_capacity(seq_len * head_dim);
    for t in 0..seq_len {
        let start = t * stride + h * head_dim;
        head.extend_from_slice(&data[start..start + head_dim]);
    }
    head
}

/// Scatter a `[seq_len, head_dim]` result back into the interleaved
/// output at head position `h`.
fn scatter_head(
    output: &mut [f32],
    head_out: &[f32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    h: usize,
) {
    let stride = num_heads * head_dim;
    for t in 0..seq_len {
        let dst = t * stride + h * head_dim;
        let src = t * head_dim;
        output[dst..dst + head_dim].copy_from_slice(&head_out[src..src + head_dim]);
    }
}

// ── Public API ─────────────────────────────────────────────────────

/// Full multi-head attention.
///
/// * `q` — queries,  shape `[seq_len, num_heads * head_dim]`
/// * `k` — keys,     shape `[seq_len, num_heads * head_dim]`
/// * `v` — values,   shape `[seq_len, num_heads * head_dim]`
///
/// Returns output of shape `[seq_len, num_heads * head_dim]`.
///
/// Internally splits Q/K/V into per-head slices, computes
/// `softmax(Q·K^T / √d_k) · V` per head (with optional causal mask),
/// then concatenates the results.
pub fn multi_head_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    config: &MhaConfig,
    seq_len: usize,
) -> Result<Vec<f32>> {
    config.validate()?;
    if seq_len == 0 {
        return Err(invalid_arg("seq_len must be > 0"));
    }

    let MhaConfig { num_heads, head_dim, use_causal_mask, .. } = *config;
    let model_dim = num_heads * head_dim;
    let expected = seq_len * model_dim;

    if q.len() != expected {
        return Err(invalid_arg("q length does not match seq_len * num_heads * head_dim"));
    }
    if k.len() != expected {
        return Err(invalid_arg("k length does not match seq_len * num_heads * head_dim"));
    }
    if v.len() != expected {
        return Err(invalid_arg("v length does not match seq_len * num_heads * head_dim"));
    }

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mask_vec = if use_causal_mask { Some(causal_mask(seq_len)) } else { None };
    let mask_ref = mask_vec.as_deref();

    let mut output = vec![0.0_f32; expected];

    for h in 0..num_heads {
        let q_head = extract_head(q, seq_len, num_heads, head_dim, h);
        let k_head = extract_head(k, seq_len, num_heads, head_dim, h);
        let v_head = extract_head(v, seq_len, num_heads, head_dim, h);

        let head_out = AttentionKernel::scaled_dot_product(
            &q_head, &k_head, &v_head, mask_ref, scale, seq_len, seq_len, head_dim,
        )?;

        scatter_head(&mut output, &head_out, seq_len, num_heads, head_dim, h);
    }

    Ok(output)
}

/// Grouped-query attention (GQA).
///
/// Like MHA, but key/value tensors have fewer heads than queries.
/// Each KV head is shared among `num_heads / num_kv_heads` query heads.
///
/// * `q` — shape `[seq_len, num_heads * head_dim]`
/// * `k` — shape `[seq_len, num_kv_heads * head_dim]`
/// * `v` — shape `[seq_len, num_kv_heads * head_dim]`
///
/// Returns shape `[seq_len, num_heads * head_dim]`.
pub fn grouped_query_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    config: &GqaConfig,
    seq_len: usize,
) -> Result<Vec<f32>> {
    config.validate()?;
    if seq_len == 0 {
        return Err(invalid_arg("seq_len must be > 0"));
    }

    let GqaConfig { num_heads, num_kv_heads, head_dim, use_causal_mask, .. } = *config;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    if q.len() != seq_len * q_dim {
        return Err(invalid_arg("q length mismatch for GQA"));
    }
    if k.len() != seq_len * kv_dim {
        return Err(invalid_arg("k length mismatch for GQA"));
    }
    if v.len() != seq_len * kv_dim {
        return Err(invalid_arg("v length mismatch for GQA"));
    }

    let group_size = num_heads / num_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mask_vec = if use_causal_mask { Some(causal_mask(seq_len)) } else { None };
    let mask_ref = mask_vec.as_deref();

    let mut output = vec![0.0_f32; seq_len * q_dim];

    for kv_h in 0..num_kv_heads {
        let k_head = extract_head(k, seq_len, num_kv_heads, head_dim, kv_h);
        let v_head = extract_head(v, seq_len, num_kv_heads, head_dim, kv_h);

        for g in 0..group_size {
            let q_idx = kv_h * group_size + g;
            let q_head = extract_head(q, seq_len, num_heads, head_dim, q_idx);

            let head_out = AttentionKernel::scaled_dot_product(
                &q_head, &k_head, &v_head, mask_ref, scale, seq_len, seq_len, head_dim,
            )?;

            scatter_head(&mut output, &head_out, seq_len, num_heads, head_dim, q_idx);
        }
    }

    Ok(output)
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-4;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    /// Row-wise softmax (numerically stable) for reference calculations.
    fn ref_softmax(row: &[f32]) -> Vec<f32> {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|&v| (v - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        if sum == 0.0 { vec![0.0; row.len()] } else { exps.iter().map(|&e| e / sum).collect() }
    }

    /// Reference single-head SDPA for verification.
    fn ref_sdpa(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        head_dim: usize,
        causal: bool,
    ) -> Vec<f32> {
        let scale = 1.0 / (head_dim as f32).sqrt();
        // Q·K^T
        let mut scores = vec![0.0_f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                let mut dot = 0.0_f32;
                for d in 0..head_dim {
                    dot += q[i * head_dim + d] * k[j * head_dim + d];
                }
                scores[i * seq_len + j] = dot * scale;
            }
        }
        // causal mask
        if causal {
            for i in 0..seq_len {
                for j in (i + 1)..seq_len {
                    scores[i * seq_len + j] = f32::NEG_INFINITY;
                }
            }
        }
        // softmax per row
        for i in 0..seq_len {
            let row = &scores[i * seq_len..(i + 1) * seq_len];
            let sm = ref_softmax(row);
            scores[i * seq_len..(i + 1) * seq_len].copy_from_slice(&sm);
        }
        // scores · V
        let mut out = vec![0.0_f32; seq_len * head_dim];
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

    fn default_config(num_heads: usize, head_dim: usize, causal: bool) -> MhaConfig {
        MhaConfig { num_heads, head_dim, use_causal_mask: causal, dropout_rate: 0.0 }
    }

    // ── 1. Single head, known values ───────────────────────────────

    #[test]
    fn single_head_known_values() {
        // seq_len=2, head_dim=2, no causal mask
        let q = vec![1.0, 0.0, 0.0, 1.0];
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let v = vec![1.0, 2.0, 3.0, 4.0];

        let cfg = default_config(1, 2, false);
        let out = multi_head_attention(&q, &k, &v, &cfg, 2).unwrap();
        let expected = ref_sdpa(&q, &k, &v, 2, 2, false);

        assert_eq!(out.len(), 4);
        for (i, (&a, &b)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(approx_eq(a, b), "pos {i}: got {a}, want {b}",);
        }
    }

    #[test]
    fn single_head_identity_keys() {
        // When Q == K (identity-like), attention weights should
        // be uniform for equal queries.
        let q = vec![1.0, 0.0, 1.0, 0.0]; // 2 identical queries
        let k = vec![1.0, 0.0, 1.0, 0.0];
        let v = vec![10.0, 20.0, 30.0, 40.0];

        let cfg = default_config(1, 2, false);
        let out = multi_head_attention(&q, &k, &v, &cfg, 2).unwrap();
        // Both rows should produce the same output (average of v rows).
        assert!(approx_eq(out[0], out[2]));
        assert!(approx_eq(out[1], out[3]));
    }

    // ── 2. Multi-head: output shape ────────────────────────────────

    #[test]
    fn multi_head_output_shape_matches_input() {
        let num_heads = 4;
        let head_dim = 8;
        let seq_len = 6;
        let n = seq_len * num_heads * head_dim;
        let q: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let k = q.clone();
        let v = q.clone();

        let cfg = default_config(num_heads, head_dim, false);
        let out = multi_head_attention(&q, &k, &v, &cfg, seq_len).unwrap();
        assert_eq!(out.len(), n);
    }

    #[test]
    fn multi_head_output_shape_with_causal() {
        let num_heads = 2;
        let head_dim = 4;
        let seq_len = 5;
        let n = seq_len * num_heads * head_dim;
        let data = vec![0.1_f32; n];

        let cfg = default_config(num_heads, head_dim, true);
        let out = multi_head_attention(&data, &data, &data, &cfg, seq_len).unwrap();
        assert_eq!(out.len(), n);
    }

    #[test]
    fn multi_head_8_heads() {
        let num_heads = 8;
        let head_dim = 16;
        let seq_len = 4;
        let n = seq_len * num_heads * head_dim;
        let data = vec![0.05_f32; n];

        let cfg = default_config(num_heads, head_dim, false);
        let out = multi_head_attention(&data, &data, &data, &cfg, seq_len).unwrap();
        assert_eq!(out.len(), n);
        // Uniform input → output should also be uniform.
        for &v in &out {
            assert!(approx_eq(v, 0.05));
        }
    }

    // ── 3. Causal mask: future positions don't contribute ──────────

    #[test]
    fn causal_first_token_sees_only_itself() {
        let head_dim = 4;
        let seq_len = 4;
        let n = seq_len * head_dim;
        // V rows are distinct so we can tell what was attended to.
        let q = vec![0.5_f32; n];
        let k = vec![0.5_f32; n];
        let mut v = vec![0.0_f32; n];
        for t in 0..seq_len {
            for d in 0..head_dim {
                v[t * head_dim + d] = (t + 1) as f32;
            }
        }

        let cfg = default_config(1, head_dim, true);
        let out = multi_head_attention(&q, &k, &v, &cfg, seq_len).unwrap();
        // First token can only attend to position 0 → output == V[0]
        for d in 0..head_dim {
            assert!(approx_eq(out[d], 1.0), "first token dim {d}: got {}, want 1.0", out[d],);
        }
    }

    #[test]
    fn causal_last_token_attends_to_all() {
        // With uniform Q, K the last token gets a uniform average of
        // all V rows.
        let head_dim = 2;
        let seq_len = 3;
        let n = seq_len * head_dim;
        let q = vec![1.0_f32; n];
        let k = vec![1.0_f32; n];
        let v = vec![
            1.0, 1.0, // t=0
            2.0, 2.0, // t=1
            3.0, 3.0, // t=2
        ];

        let cfg = default_config(1, head_dim, true);
        let out = multi_head_attention(&q, &k, &v, &cfg, seq_len).unwrap();
        let expected = ref_sdpa(&q, &k, &v, seq_len, head_dim, true);
        for (i, (&a, &b)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(approx_eq(a, b), "pos {i}: got {a}, want {b}");
        }
    }

    #[test]
    fn causal_vs_noncausal_first_row_differs() {
        // With seq_len > 1, row 0 of causal output should differ from
        // non-causal because the causal mask blocks future tokens.
        let head_dim = 4;
        let seq_len = 3;
        let n = seq_len * head_dim;
        let q: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
        let k = q.clone();
        let v: Vec<f32> = (0..n).map(|i| ((n - i) as f32) * 0.1).collect();

        let causal_cfg = default_config(1, head_dim, true);
        let nocausal_cfg = default_config(1, head_dim, false);
        let c = multi_head_attention(&q, &k, &v, &causal_cfg, seq_len).unwrap();
        let nc = multi_head_attention(&q, &k, &v, &nocausal_cfg, seq_len).unwrap();

        // At least one element in the first row should differ.
        let differs = (0..head_dim).any(|d| !approx_eq(c[d], nc[d]));
        assert!(differs, "causal and non-causal row 0 should differ");
    }

    // ── 4. GQA: correct head sharing ──────────────────────────────

    #[test]
    fn gqa_equal_heads_matches_mha() {
        // When num_kv_heads == num_heads, GQA == MHA.
        let num_heads = 4;
        let head_dim = 4;
        let seq_len = 3;
        let n = seq_len * num_heads * head_dim;
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();

        let mha_cfg = default_config(num_heads, head_dim, false);
        let gqa_cfg = GqaConfig {
            num_heads,
            num_kv_heads: num_heads,
            head_dim,
            use_causal_mask: false,
            dropout_rate: 0.0,
        };

        let mha = multi_head_attention(&data, &data, &data, &mha_cfg, seq_len).unwrap();
        let gqa = grouped_query_attention(&data, &data, &data, &gqa_cfg, seq_len).unwrap();

        assert_eq!(mha.len(), gqa.len());
        for (i, (&a, &b)) in mha.iter().zip(gqa.iter()).enumerate() {
            assert!(approx_eq(a, b), "pos {i}: mha={a} gqa={b}",);
        }
    }

    #[test]
    fn gqa_kv_sharing_output_shape() {
        let num_heads = 8;
        let num_kv_heads = 2;
        let head_dim = 4;
        let seq_len = 3;
        let q = vec![0.1_f32; seq_len * num_heads * head_dim];
        let k = vec![0.1_f32; seq_len * num_kv_heads * head_dim];
        let v = vec![0.1_f32; seq_len * num_kv_heads * head_dim];

        let cfg = GqaConfig {
            num_heads,
            num_kv_heads,
            head_dim,
            use_causal_mask: false,
            dropout_rate: 0.0,
        };
        let out = grouped_query_attention(&q, &k, &v, &cfg, seq_len).unwrap();
        assert_eq!(out.len(), seq_len * num_heads * head_dim);
    }

    #[test]
    fn gqa_shared_heads_produce_same_output() {
        // With uniform Q, shared KV heads should make grouped queries
        // produce identical output within each group.
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 4;
        let seq_len = 3;
        let q = vec![0.5_f32; seq_len * num_heads * head_dim];
        let k = vec![0.5_f32; seq_len * num_kv_heads * head_dim];
        let v: Vec<f32> =
            (0..seq_len * num_kv_heads * head_dim).map(|i| (i as f32) * 0.1).collect();

        let cfg = GqaConfig {
            num_heads,
            num_kv_heads,
            head_dim,
            use_causal_mask: false,
            dropout_rate: 0.0,
        };
        let out = grouped_query_attention(&q, &k, &v, &cfg, seq_len).unwrap();

        // Heads 0,1 share kv_head 0 → with uniform Q they are identical.
        // Heads 2,3 share kv_head 1.
        let group_size = num_heads / num_kv_heads;
        for kv_h in 0..num_kv_heads {
            for t in 0..seq_len {
                let base_idx = kv_h * group_size;
                let base_start = t * num_heads * head_dim + base_idx * head_dim;
                for g in 1..group_size {
                    let cmp_start = t * num_heads * head_dim + (base_idx + g) * head_dim;
                    for d in 0..head_dim {
                        assert!(
                            approx_eq(out[base_start + d], out[cmp_start + d]),
                            "kv_h={kv_h} t={t} g={g} d={d}: {} != {}",
                            out[base_start + d],
                            out[cmp_start + d],
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn gqa_with_causal_mask() {
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 4;
        let seq_len = 4;
        let q = vec![0.2_f32; seq_len * num_heads * head_dim];
        let k = vec![0.2_f32; seq_len * num_kv_heads * head_dim];
        let v: Vec<f32> =
            (0..seq_len * num_kv_heads * head_dim).map(|i| (i as f32) * 0.01).collect();

        let cfg = GqaConfig {
            num_heads,
            num_kv_heads,
            head_dim,
            use_causal_mask: true,
            dropout_rate: 0.0,
        };
        let out = grouped_query_attention(&q, &k, &v, &cfg, seq_len).unwrap();
        assert_eq!(out.len(), seq_len * num_heads * head_dim);
    }

    // ── 5. Property tests ──────────────────────────────────────────

    #[test]
    fn property_output_shape_invariant() {
        // For various configs, output length == input length.
        for &nh in &[1, 2, 4, 8] {
            for &hd in &[2, 4, 16] {
                for &sl in &[1, 3, 8] {
                    let n = sl * nh * hd;
                    let data = vec![0.1_f32; n];
                    let cfg = default_config(nh, hd, false);
                    let out = multi_head_attention(&data, &data, &data, &cfg, sl).unwrap();
                    assert_eq!(out.len(), n, "nh={nh} hd={hd} sl={sl}",);
                }
            }
        }
    }

    #[test]
    fn property_attention_weights_sum_to_one() {
        // Verify via reference: each softmax row sums to ~1.
        let head_dim = 4;
        let seq_len = 5;
        let n = seq_len * head_dim;
        let q: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
        let k = q.clone();
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Compute scores manually.
        let mut scores = vec![0.0_f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                let mut dot = 0.0_f32;
                for d in 0..head_dim {
                    dot += q[i * head_dim + d] * k[j * head_dim + d];
                }
                scores[i * seq_len + j] = dot * scale;
            }
        }
        for i in 0..seq_len {
            let row = &scores[i * seq_len..(i + 1) * seq_len];
            let sm = ref_softmax(row);
            let sum: f32 = sm.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "row {i}: sum={sum}",);
        }
    }

    #[test]
    fn property_output_finite() {
        let num_heads = 2;
        let head_dim = 8;
        let seq_len = 4;
        let n = seq_len * num_heads * head_dim;
        let data = vec![0.5_f32; n];
        let cfg = default_config(num_heads, head_dim, true);
        let out = multi_head_attention(&data, &data, &data, &cfg, seq_len).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn property_gqa_output_shape_invariant() {
        for &nh in &[2, 4, 8] {
            for &kvh in &[1, 2] {
                if nh % kvh != 0 {
                    continue;
                }
                for &hd in &[4, 8] {
                    for &sl in &[1, 3] {
                        let q = vec![0.1; sl * nh * hd];
                        let k = vec![0.1; sl * kvh * hd];
                        let v = vec![0.1; sl * kvh * hd];
                        let cfg = GqaConfig {
                            num_heads: nh,
                            num_kv_heads: kvh,
                            head_dim: hd,
                            use_causal_mask: false,
                            dropout_rate: 0.0,
                        };
                        let out = grouped_query_attention(&q, &k, &v, &cfg, sl).unwrap();
                        assert_eq!(out.len(), sl * nh * hd, "nh={nh} kvh={kvh} hd={hd} sl={sl}",);
                    }
                }
            }
        }
    }

    #[test]
    fn property_causal_monotonic_context() {
        // With causal mask and uniform Q/K, later tokens should have
        // access to more context, so their output should reflect the
        // average of more V rows.
        let head_dim = 2;
        let seq_len = 4;
        let n = seq_len * head_dim;
        let q = vec![1.0_f32; n];
        let k = vec![1.0_f32; n];
        // V rows: [1,1], [2,2], [3,3], [4,4]
        let v: Vec<f32> =
            (0..seq_len).flat_map(|t| std::iter::repeat_n((t + 1) as f32, head_dim)).collect();

        let cfg = default_config(1, head_dim, true);
        let out = multi_head_attention(&q, &k, &v, &cfg, seq_len).unwrap();
        // Row averages should be increasing (1.0, 1.5, 2.0, 2.5).
        let mut prev = f32::NEG_INFINITY;
        for t in 0..seq_len {
            let avg: f32 =
                out[t * head_dim..(t + 1) * head_dim].iter().sum::<f32>() / head_dim as f32;
            assert!(avg >= prev - EPS, "t={t}: avg {avg} < prev {prev}",);
            prev = avg;
        }
    }

    // ── 6. Edge cases ──────────────────────────────────────────────

    #[test]
    fn edge_seq_len_1() {
        let q = vec![1.0, 2.0];
        let k = vec![3.0, 4.0];
        let v = vec![5.0, 6.0];
        let cfg = default_config(1, 2, true);
        let out = multi_head_attention(&q, &k, &v, &cfg, 1).unwrap();
        // With seq_len=1, softmax of a single element is 1.0,
        // so output == V.
        assert_eq!(out.len(), 2);
        assert!(approx_eq(out[0], 5.0));
        assert!(approx_eq(out[1], 6.0));
    }

    #[test]
    fn edge_num_heads_1() {
        let head_dim = 8;
        let seq_len = 3;
        let n = seq_len * head_dim;
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
        let cfg = default_config(1, head_dim, false);
        let out = multi_head_attention(&data, &data, &data, &cfg, seq_len).unwrap();
        assert_eq!(out.len(), n);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn edge_seq_len_1_gqa() {
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 4;
        let q = vec![0.5_f32; num_heads * head_dim];
        let k = vec![0.5_f32; num_kv_heads * head_dim];
        let v: Vec<f32> = (0..num_kv_heads * head_dim).map(|i| (i as f32) * 0.1).collect();

        let cfg = GqaConfig {
            num_heads,
            num_kv_heads,
            head_dim,
            use_causal_mask: false,
            dropout_rate: 0.0,
        };
        let out = grouped_query_attention(&q, &k, &v, &cfg, 1).unwrap();
        assert_eq!(out.len(), num_heads * head_dim);
    }

    #[test]
    fn edge_large_head_dim() {
        let head_dim = 128;
        let seq_len = 2;
        let n = seq_len * head_dim;
        let data = vec![0.01_f32; n];
        let cfg = default_config(1, head_dim, false);
        let out = multi_head_attention(&data, &data, &data, &cfg, seq_len).unwrap();
        assert_eq!(out.len(), n);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── 7. Validation / error paths ────────────────────────────────

    #[test]
    fn error_zero_heads() {
        let cfg = default_config(0, 4, false);
        let r = multi_head_attention(&[0.0; 4], &[0.0; 4], &[0.0; 4], &cfg, 1);
        assert!(r.is_err());
    }

    #[test]
    fn error_zero_head_dim() {
        let cfg = default_config(2, 0, false);
        let r = multi_head_attention(&[], &[], &[], &cfg, 1);
        assert!(r.is_err());
    }

    #[test]
    fn error_zero_seq_len() {
        let cfg = default_config(1, 4, false);
        let r = multi_head_attention(&[], &[], &[], &cfg, 0);
        assert!(r.is_err());
    }

    #[test]
    fn error_q_length_mismatch() {
        let cfg = default_config(1, 4, false);
        // seq_len=2 → expect 8 elements, provide 4
        let r = multi_head_attention(&[0.0; 4], &[0.0; 8], &[0.0; 8], &cfg, 2);
        assert!(r.is_err());
    }

    #[test]
    fn error_gqa_heads_not_divisible() {
        let cfg = GqaConfig {
            num_heads: 5,
            num_kv_heads: 3,
            head_dim: 4,
            use_causal_mask: false,
            dropout_rate: 0.0,
        };
        let r = grouped_query_attention(&[0.0; 20], &[0.0; 12], &[0.0; 12], &cfg, 1);
        assert!(r.is_err());
    }

    #[test]
    fn error_invalid_dropout_rate() {
        let cfg =
            MhaConfig { num_heads: 1, head_dim: 4, use_causal_mask: false, dropout_rate: 1.0 };
        let r = multi_head_attention(&[0.0; 4], &[0.0; 4], &[0.0; 4], &cfg, 1);
        assert!(r.is_err());
    }

    #[test]
    fn error_negative_dropout_rate() {
        let cfg =
            MhaConfig { num_heads: 1, head_dim: 4, use_causal_mask: false, dropout_rate: -0.1 };
        let r = multi_head_attention(&[0.0; 4], &[0.0; 4], &[0.0; 4], &cfg, 1);
        assert!(r.is_err());
    }

    // ── 8. Consistency ─────────────────────────────────────────────

    #[test]
    fn mha_matches_manual_per_head_sdpa() {
        // 2 heads, head_dim=2, seq_len=2
        let num_heads = 2;
        let head_dim = 2;
        let seq_len = 2;
        // Interleaved: [t0_h0_d0, t0_h0_d1, t0_h1_d0, t0_h1_d1, ...]
        let q = vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5];
        let k = vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5];
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let cfg = default_config(num_heads, head_dim, false);
        let out = multi_head_attention(&q, &k, &v, &cfg, seq_len).unwrap();

        // Compute per-head manually.
        let q0 = extract_head(&q, seq_len, num_heads, head_dim, 0);
        let k0 = extract_head(&k, seq_len, num_heads, head_dim, 0);
        let v0 = extract_head(&v, seq_len, num_heads, head_dim, 0);
        let exp0 = ref_sdpa(&q0, &k0, &v0, seq_len, head_dim, false);

        let q1 = extract_head(&q, seq_len, num_heads, head_dim, 1);
        let k1 = extract_head(&k, seq_len, num_heads, head_dim, 1);
        let v1 = extract_head(&v, seq_len, num_heads, head_dim, 1);
        let exp1 = ref_sdpa(&q1, &k1, &v1, seq_len, head_dim, false);

        // Scatter expected back into interleaved format.
        let mut expected = vec![0.0_f32; seq_len * num_heads * head_dim];
        scatter_head(&mut expected, &exp0, seq_len, num_heads, head_dim, 0);
        scatter_head(&mut expected, &exp1, seq_len, num_heads, head_dim, 1);

        for (i, (&a, &b)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(approx_eq(a, b), "pos {i}: got {a}, want {b}",);
        }
    }
}
