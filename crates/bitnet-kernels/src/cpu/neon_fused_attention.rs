//! ARM NEON fused attention kernels for Apple Silicon.
//!
//! Provides SIMD-accelerated fused QKV projection, scaled dot-product
//! attention, and causal-masked attention using `float32x4` NEON intrinsics.
//! Each function processes 4 elements at a time with scalar fallback for
//! remainders.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Number of f32 lanes in a NEON `float32x4_t` register.
const LANES: usize = 4;

// ── Helpers ─────────────────────────────────────────────────────────

/// Scalar dot product used as reference and for tail elements.
#[cfg(test)]
#[inline]
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Scalar softmax in-place over `row[..len]`.
fn softmax_inplace(row: &mut [f32]) {
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

// ── NEON dot product ────────────────────────────────────────────────

/// Compute dot product of two equal-length slices using NEON FMA.
///
/// # Safety
/// Requires `aarch64` target with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    let chunks = len / LANES;
    let remainder = len % LANES;

    let mut acc = vdupq_n_f32(0.0);
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * LANES;
        let va = unsafe { vld1q_f32(a_ptr.add(offset)) };
        let vb = unsafe { vld1q_f32(b_ptr.add(offset)) };
        acc = vfmaq_f32(acc, va, vb);
    }

    // Horizontal sum of accumulator.
    let mut result = vaddvq_f32(acc);

    // Scalar tail.
    let tail_start = chunks * LANES;
    for i in 0..remainder {
        result += a[tail_start + i] * b[tail_start + i];
    }
    result
}

// ── Fused QKV projection ────────────────────────────────────────────

/// Projects `input` to Q, K, V simultaneously using NEON FMA.
///
/// For each position `p` in `[0, seq_len)` and each head dimension `h` in
/// `[0, d_head)`, computes:
///   q[p*d_head + h] = dot(input[p*d_model ..][..d_model], wq[h*d_model ..][..d_model])
///   k[p*d_head + h] = dot(input[p*d_model ..][..d_model], wk[h*d_model ..][..d_model])
///   v[p*d_head + h] = dot(input[p*d_model ..][..d_model], wv[h*d_model ..][..d_model])
///
/// # Panics
/// Panics if slice lengths are inconsistent with the given dimensions.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
pub fn neon_fused_qkv_projection(
    input: &[f32],
    wq: &[f32],
    wk: &[f32],
    wv: &[f32],
    q: &mut [f32],
    k: &mut [f32],
    v: &mut [f32],
    seq_len: usize,
    d_model: usize,
    d_head: usize,
) {
    assert_eq!(input.len(), seq_len * d_model, "input size mismatch");
    assert_eq!(wq.len(), d_head * d_model, "wq size mismatch");
    assert_eq!(wk.len(), d_head * d_model, "wk size mismatch");
    assert_eq!(wv.len(), d_head * d_model, "wv size mismatch");
    assert_eq!(q.len(), seq_len * d_head, "q size mismatch");
    assert_eq!(k.len(), seq_len * d_head, "k size mismatch");
    assert_eq!(v.len(), seq_len * d_head, "v size mismatch");

    for p in 0..seq_len {
        let inp = &input[p * d_model..(p + 1) * d_model];
        for h in 0..d_head {
            let w_row_q = &wq[h * d_model..(h + 1) * d_model];
            let w_row_k = &wk[h * d_model..(h + 1) * d_model];
            let w_row_v = &wv[h * d_model..(h + 1) * d_model];

            // SAFETY: we are on aarch64 (guarded by cfg) with NEON available.
            unsafe {
                q[p * d_head + h] = dot_neon(inp, w_row_q);
                k[p * d_head + h] = dot_neon(inp, w_row_k);
                v[p * d_head + h] = dot_neon(inp, w_row_v);
            }
        }
    }
}

// ── Scaled dot-product attention ────────────────────────────────────

/// Computes `output = softmax(Q·Kᵀ / √d_head) · V`.
///
/// Layout: `q`, `k`, `v`, `output` are flat `[seq_len × d_head]` matrices
/// stored row-major.
///
/// # Panics
/// Panics on dimension mismatch or zero `d_head`.
#[cfg(target_arch = "aarch64")]
pub fn neon_scaled_dot_product_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    d_head: usize,
) {
    assert!(d_head > 0, "d_head must be > 0");
    let total = seq_len * d_head;
    assert_eq!(q.len(), total, "q size mismatch");
    assert_eq!(k.len(), total, "k size mismatch");
    assert_eq!(v.len(), total, "v size mismatch");
    assert_eq!(output.len(), total, "output size mismatch");

    let scale = 1.0 / (d_head as f32).sqrt();

    // scores[i][j] = dot(q_i, k_j) * scale
    let mut scores = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        let qi = &q[i * d_head..(i + 1) * d_head];
        for j in 0..seq_len {
            let kj = &k[j * d_head..(j + 1) * d_head];
            let d = unsafe { dot_neon(qi, kj) };
            scores[i * seq_len + j] = d * scale;
        }
    }

    // Row-wise softmax.
    for i in 0..seq_len {
        softmax_inplace(&mut scores[i * seq_len..(i + 1) * seq_len]);
    }

    // output[i] = sum_j scores[i][j] * v[j]
    neon_weighted_sum(v, &scores, output, seq_len, d_head);
}

// ── Causal mask attention ───────────────────────────────────────────

/// Like [`neon_scaled_dot_product_attention`] but applies a causal
/// (upper-triangular) mask: positions `j > i` are set to `−∞` before
/// softmax so that each token can only attend to earlier tokens and itself.
#[cfg(target_arch = "aarch64")]
pub fn neon_causal_mask_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    d_head: usize,
) {
    assert!(d_head > 0, "d_head must be > 0");
    let total = seq_len * d_head;
    assert_eq!(q.len(), total, "q size mismatch");
    assert_eq!(k.len(), total, "k size mismatch");
    assert_eq!(v.len(), total, "v size mismatch");
    assert_eq!(output.len(), total, "output size mismatch");

    let scale = 1.0 / (d_head as f32).sqrt();

    let mut scores = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        let qi = &q[i * d_head..(i + 1) * d_head];
        for j in 0..seq_len {
            if j > i {
                scores[i * seq_len + j] = f32::NEG_INFINITY;
            } else {
                let kj = &k[j * d_head..(j + 1) * d_head];
                let d = unsafe { dot_neon(qi, kj) };
                scores[i * seq_len + j] = d * scale;
            }
        }
    }

    for i in 0..seq_len {
        softmax_inplace(&mut scores[i * seq_len..(i + 1) * seq_len]);
    }

    neon_weighted_sum(v, &scores, output, seq_len, d_head);
}

// ── Weighted sum (shared by both attention variants) ────────────────

/// Compute `output[i] = Σ_j weights[i,j] · V[j]` using NEON.
#[cfg(target_arch = "aarch64")]
fn neon_weighted_sum(
    v: &[f32],
    weights: &[f32],
    output: &mut [f32],
    seq_len: usize,
    d_head: usize,
) {
    for i in 0..seq_len {
        let out_row = &mut output[i * d_head..(i + 1) * d_head];
        out_row.fill(0.0);

        for j in 0..seq_len {
            let w = weights[i * seq_len + j];
            let vj = &v[j * d_head..(j + 1) * d_head];
            let chunks = d_head / LANES;
            let remainder = d_head % LANES;

            // SAFETY: aarch64 NEON loads/stores within slice bounds.
            unsafe {
                let w_vec = vdupq_n_f32(w);
                for c in 0..chunks {
                    let base = c * LANES;
                    let cur = vld1q_f32(out_row.as_ptr().add(base));
                    let val = vld1q_f32(vj.as_ptr().add(base));
                    let res = vfmaq_f32(cur, val, w_vec);
                    vst1q_f32(out_row.as_mut_ptr().add(base), res);
                }
            }

            let tail_start = chunks * LANES;
            for t in 0..remainder {
                out_row[tail_start + t] += w * vj[tail_start + t];
            }
        }
    }
}

// ── Scalar reference implementations (for testing) ──────────────────

/// Scalar reference implementation of fused QKV projection.
#[cfg(test)]
fn scalar_fused_qkv_projection(
    input: &[f32],
    wq: &[f32],
    wk: &[f32],
    wv: &[f32],
    q: &mut [f32],
    k: &mut [f32],
    v: &mut [f32],
    seq_len: usize,
    d_model: usize,
    d_head: usize,
) {
    for p in 0..seq_len {
        let inp = &input[p * d_model..(p + 1) * d_model];
        for h in 0..d_head {
            q[p * d_head + h] = dot_scalar(inp, &wq[h * d_model..(h + 1) * d_model]);
            k[p * d_head + h] = dot_scalar(inp, &wk[h * d_model..(h + 1) * d_model]);
            v[p * d_head + h] = dot_scalar(inp, &wv[h * d_model..(h + 1) * d_model]);
        }
    }
}

/// Scalar reference for scaled dot-product attention.
#[cfg(test)]
fn scalar_scaled_dot_product_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    d_head: usize,
) {
    let scale = 1.0 / (d_head as f32).sqrt();
    let mut scores = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            scores[i * seq_len + j] =
                dot_scalar(&q[i * d_head..(i + 1) * d_head], &k[j * d_head..(j + 1) * d_head])
                    * scale;
        }
    }
    for i in 0..seq_len {
        softmax_inplace(&mut scores[i * seq_len..(i + 1) * seq_len]);
    }
    for i in 0..seq_len {
        let out = &mut output[i * d_head..(i + 1) * d_head];
        out.fill(0.0);
        for j in 0..seq_len {
            let w = scores[i * seq_len + j];
            for h in 0..d_head {
                out[h] += w * v[j * d_head + h];
            }
        }
    }
}

/// Scalar reference for causal mask attention.
#[cfg(test)]
fn scalar_causal_mask_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    d_head: usize,
) {
    let scale = 1.0 / (d_head as f32).sqrt();
    let mut scores = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            if j > i {
                scores[i * seq_len + j] = f32::NEG_INFINITY;
            } else {
                scores[i * seq_len + j] =
                    dot_scalar(&q[i * d_head..(i + 1) * d_head], &k[j * d_head..(j + 1) * d_head])
                        * scale;
            }
        }
    }
    for i in 0..seq_len {
        softmax_inplace(&mut scores[i * seq_len..(i + 1) * seq_len]);
    }
    for i in 0..seq_len {
        let out = &mut output[i * d_head..(i + 1) * d_head];
        out.fill(0.0);
        for j in 0..seq_len {
            let w = scores[i * seq_len + j];
            for h in 0..d_head {
                out[h] += w * v[j * d_head + h];
            }
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Identity-weight QKV projection: output == input when W = I.
    #[test]
    fn test_qkv_projection_identity() {
        let seq_len = 2;
        let d_model = 4;
        let d_head = 4;
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        // Identity matrix (4×4 stored row-major as d_head * d_model)
        let eye =
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let mut q = vec![0.0; seq_len * d_head];
        let mut k = vec![0.0; seq_len * d_head];
        let mut v = vec![0.0; seq_len * d_head];
        neon_fused_qkv_projection(
            &input, &eye, &eye, &eye, &mut q, &mut k, &mut v, seq_len, d_model, d_head,
        );
        assert_eq!(q, input);
        assert_eq!(k, input);
        assert_eq!(v, input);
    }

    /// QKV projection matches scalar reference on non-trivial weights.
    #[test]
    fn test_qkv_projection_vs_scalar() {
        let seq_len = 3;
        let d_model = 8;
        let d_head = 4;
        let input: Vec<f32> = (0..seq_len * d_model).map(|i| (i as f32) * 0.1).collect();
        let wq: Vec<f32> = (0..d_head * d_model).map(|i| ((i % 7) as f32) * 0.05).collect();
        let wk: Vec<f32> = (0..d_head * d_model).map(|i| ((i % 5) as f32) * 0.03).collect();
        let wv: Vec<f32> = (0..d_head * d_model).map(|i| ((i % 3) as f32) * 0.07).collect();

        let mut q_neon = vec![0.0; seq_len * d_head];
        let mut k_neon = vec![0.0; seq_len * d_head];
        let mut v_neon = vec![0.0; seq_len * d_head];
        neon_fused_qkv_projection(
            &input,
            &wq,
            &wk,
            &wv,
            &mut q_neon,
            &mut k_neon,
            &mut v_neon,
            seq_len,
            d_model,
            d_head,
        );

        let mut q_ref = vec![0.0; seq_len * d_head];
        let mut k_ref = vec![0.0; seq_len * d_head];
        let mut v_ref = vec![0.0; seq_len * d_head];
        scalar_fused_qkv_projection(
            &input, &wq, &wk, &wv, &mut q_ref, &mut k_ref, &mut v_ref, seq_len, d_model, d_head,
        );

        for i in 0..q_neon.len() {
            assert!((q_neon[i] - q_ref[i]).abs() < 1e-4, "q mismatch at {i}");
            assert!((k_neon[i] - k_ref[i]).abs() < 1e-4, "k mismatch at {i}");
            assert!((v_neon[i] - v_ref[i]).abs() < 1e-4, "v mismatch at {i}");
        }
    }

    /// Scaled dot-product attention on a single token is just softmax(0)*V = V.
    #[test]
    fn test_sdpa_single_token() {
        let seq_len = 1;
        let d_head = 4;
        let q = vec![1.0, 0.0, 0.0, 0.0];
        let k = vec![1.0, 0.0, 0.0, 0.0];
        let v = vec![0.5, 0.6, 0.7, 0.8];
        let mut output = vec![0.0; d_head];
        neon_scaled_dot_product_attention(&q, &k, &v, &mut output, seq_len, d_head);
        for i in 0..d_head {
            assert!((output[i] - v[i]).abs() < 1e-5, "mismatch at {i}");
        }
    }

    /// Scaled dot-product attention matches scalar reference.
    #[test]
    fn test_sdpa_vs_scalar() {
        let seq_len = 4;
        let d_head = 8;
        let n = seq_len * d_head;
        let q: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 - 1.0).collect();
        let k: Vec<f32> = (0..n).map(|i| ((i * 3 + 1) as f32) * 0.05 - 0.5).collect();
        let v: Vec<f32> = (0..n).map(|i| ((i * 7 + 2) as f32) * 0.02).collect();

        let mut out_neon = vec![0.0; n];
        let mut out_ref = vec![0.0; n];
        neon_scaled_dot_product_attention(&q, &k, &v, &mut out_neon, seq_len, d_head);
        scalar_scaled_dot_product_attention(&q, &k, &v, &mut out_ref, seq_len, d_head);

        for i in 0..n {
            assert!(
                (out_neon[i] - out_ref[i]).abs() < 1e-4,
                "sdpa mismatch at {i}: neon={} ref={}",
                out_neon[i],
                out_ref[i]
            );
        }
    }

    /// Causal masking: first row sees only itself.
    #[test]
    fn test_causal_first_row_identity() {
        let seq_len = 3;
        let d_head = 4;
        let n = seq_len * d_head;
        let q = vec![1.0; n];
        let k = vec![1.0; n];
        let v: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut output = vec![0.0; n];
        neon_causal_mask_attention(&q, &k, &v, &mut output, seq_len, d_head);

        // First row can only attend to position 0, so output[0..d_head] == v[0..d_head].
        for h in 0..d_head {
            assert!((output[h] - v[h]).abs() < 1e-5, "first row mismatch at {h}");
        }
    }

    /// Causal attention matches scalar reference on larger input.
    #[test]
    fn test_causal_vs_scalar() {
        let seq_len = 4;
        let d_head = 8;
        let n = seq_len * d_head;
        let q: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 - 1.0).collect();
        let k: Vec<f32> = (0..n).map(|i| ((i * 3 + 1) as f32) * 0.05 - 0.5).collect();
        let v: Vec<f32> = (0..n).map(|i| ((i * 7 + 2) as f32) * 0.02).collect();

        let mut out_neon = vec![0.0; n];
        let mut out_ref = vec![0.0; n];
        neon_causal_mask_attention(&q, &k, &v, &mut out_neon, seq_len, d_head);
        scalar_causal_mask_attention(&q, &k, &v, &mut out_ref, seq_len, d_head);

        for i in 0..n {
            assert!(
                (out_neon[i] - out_ref[i]).abs() < 1e-4,
                "causal mismatch at {i}: neon={} ref={}",
                out_neon[i],
                out_ref[i]
            );
        }
    }

    /// Zero input produces zero output for QKV projection.
    #[test]
    fn test_qkv_zero_input() {
        let seq_len = 2;
        let d_model = 4;
        let d_head = 4;
        let input = vec![0.0; seq_len * d_model];
        let wq: Vec<f32> = (0..d_head * d_model).map(|i| i as f32).collect();
        let wk = wq.clone();
        let wv = wq.clone();
        let mut q = vec![999.0; seq_len * d_head];
        let mut k = vec![999.0; seq_len * d_head];
        let mut v = vec![999.0; seq_len * d_head];
        neon_fused_qkv_projection(
            &input, &wq, &wk, &wv, &mut q, &mut k, &mut v, seq_len, d_model, d_head,
        );
        assert!(q.iter().all(|&x| x.abs() < 1e-6), "q should be zero");
        assert!(k.iter().all(|&x| x.abs() < 1e-6), "k should be zero");
        assert!(v.iter().all(|&x| x.abs() < 1e-6), "v should be zero");
    }

    /// Softmax rows sum to 1.0 in SDPA output (stochastic matrix property).
    #[test]
    fn test_sdpa_softmax_normalisation() {
        let seq_len = 3;
        let d_head = 4;
        let n = seq_len * d_head;
        let q: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let k: Vec<f32> = (0..n).map(|i| i as f32 * 0.05).collect();

        // Use uniform V so output equals softmax-weighted sum of identical rows ⇒
        // each output row should equal V[0..d_head].
        let v: Vec<f32> = (0..d_head).map(|i| (i + 1) as f32).cycle().take(n).collect();
        let mut output = vec![0.0; n];
        neon_scaled_dot_product_attention(&q, &k, &v, &mut output, seq_len, d_head);

        // Since all V rows are identical, output rows should match V row.
        for i in 0..seq_len {
            for h in 0..d_head {
                let expected = v[h]; // all V rows identical
                assert!(
                    (output[i * d_head + h] - expected).abs() < 1e-4,
                    "row {i} dim {h}: got {} expected {}",
                    output[i * d_head + h],
                    expected
                );
            }
        }
    }
}

// ── Property tests ──────────────────────────────────────────────────

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    /// Generate a Vec<f32> of the given length with values in [-1, 1].
    fn vec_f32(len: usize) -> impl Strategy<Value = Vec<f32>> {
        proptest::collection::vec(-1.0f32..1.0f32, len)
    }

    proptest! {
        /// NEON QKV projection matches scalar reference for arbitrary inputs.
        #[test]
        fn prop_qkv_matches_scalar(
            input in vec_f32(3 * 8),   // seq_len=3, d_model=8
            wq in vec_f32(4 * 8),      // d_head=4
            wk in vec_f32(4 * 8),
            wv in vec_f32(4 * 8),
        ) {
            let (seq_len, d_model, d_head) = (3, 8, 4);
            let mut q_n = vec![0.0; seq_len * d_head];
            let mut k_n = vec![0.0; seq_len * d_head];
            let mut v_n = vec![0.0; seq_len * d_head];
            neon_fused_qkv_projection(&input, &wq, &wk, &wv, &mut q_n, &mut k_n, &mut v_n, seq_len, d_model, d_head);

            let mut q_s = vec![0.0; seq_len * d_head];
            let mut k_s = vec![0.0; seq_len * d_head];
            let mut v_s = vec![0.0; seq_len * d_head];
            scalar_fused_qkv_projection(&input, &wq, &wk, &wv, &mut q_s, &mut k_s, &mut v_s, seq_len, d_model, d_head);

            for i in 0..q_n.len() {
                prop_assert!((q_n[i] - q_s[i]).abs() < 1e-3, "q diff at {}: {} vs {}", i, q_n[i], q_s[i]);
                prop_assert!((k_n[i] - k_s[i]).abs() < 1e-3, "k diff at {}: {} vs {}", i, k_n[i], k_s[i]);
                prop_assert!((v_n[i] - v_s[i]).abs() < 1e-3, "v diff at {}: {} vs {}", i, v_n[i], v_s[i]);
            }
        }

        /// NEON SDPA matches scalar reference for arbitrary Q/K/V.
        #[test]
        fn prop_sdpa_matches_scalar(
            q in vec_f32(3 * 4),       // seq_len=3, d_head=4
            k in vec_f32(3 * 4),
            v in vec_f32(3 * 4),
        ) {
            let (seq_len, d_head) = (3, 4);
            let n = seq_len * d_head;
            let mut out_n = vec![0.0; n];
            let mut out_s = vec![0.0; n];
            neon_scaled_dot_product_attention(&q, &k, &v, &mut out_n, seq_len, d_head);
            scalar_scaled_dot_product_attention(&q, &k, &v, &mut out_s, seq_len, d_head);

            for i in 0..n {
                prop_assert!((out_n[i] - out_s[i]).abs() < 1e-3, "sdpa diff at {}: {} vs {}", i, out_n[i], out_s[i]);
            }
        }

        /// NEON causal attention matches scalar reference.
        #[test]
        fn prop_causal_matches_scalar(
            q in vec_f32(3 * 4),
            k in vec_f32(3 * 4),
            v in vec_f32(3 * 4),
        ) {
            let (seq_len, d_head) = (3, 4);
            let n = seq_len * d_head;
            let mut out_n = vec![0.0; n];
            let mut out_s = vec![0.0; n];
            neon_causal_mask_attention(&q, &k, &v, &mut out_n, seq_len, d_head);
            scalar_causal_mask_attention(&q, &k, &v, &mut out_s, seq_len, d_head);

            for i in 0..n {
                prop_assert!((out_n[i] - out_s[i]).abs() < 1e-3, "causal diff at {}: {} vs {}", i, out_n[i], out_s[i]);
            }
        }
    }
}
