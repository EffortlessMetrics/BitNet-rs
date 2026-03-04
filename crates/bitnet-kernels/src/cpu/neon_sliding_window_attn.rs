//! NEON sliding window attention kernel for Apple Silicon.
//!
//! Provides efficient local attention with a sliding window mask, restricting
//! each query position to attend only to the last `window_size` key positions
//! (causal). Includes NEON‑accelerated inner loops on `aarch64` and scalar
//! fallbacks everywhere else.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// NEON lane count for `float32x4_t`.
#[cfg(target_arch = "aarch64")]
const LANES: usize = 4;

// ── Helpers ────────────────────────────────────────────────────────────

/// Scalar dot product.
#[inline]
#[allow(dead_code)]
fn scalar_dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// NEON dot product.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_dot(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let chunks = len / LANES;
    let mut vacc = vdupq_n_f32(0.0);
    for i in 0..chunks {
        let va = unsafe { vld1q_f32(a.as_ptr().add(i * LANES)) };
        let vb = unsafe { vld1q_f32(b.as_ptr().add(i * LANES)) };
        vacc = vfmaq_f32(vacc, va, vb);
    }
    let mut acc = vaddvq_f32(vacc);
    for i in (chunks * LANES)..len {
        acc += unsafe { *a.as_ptr().add(i) * *b.as_ptr().add(i) };
    }
    acc
}

/// Scalar softmax in-place (max‑subtract‑exp‑normalise).
#[allow(dead_code)]
fn scalar_softmax_inplace(data: &mut [f32]) {
    if data.is_empty() {
        return;
    }
    let max_val = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max_val == f32::NEG_INFINITY {
        // All masked – leave as‑is (zeros after exp would be 0/0).
        return;
    }
    let mut sum = 0.0f32;
    for v in data.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for v in data.iter_mut() {
            *v *= inv;
        }
    }
}

/// NEON‑accelerated softmax in-place.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_softmax_inplace(data: &mut [f32]) {
    let len = data.len();
    if len == 0 {
        return;
    }

    // Phase 1 – max
    let ptr = data.as_ptr();
    let chunks = len / LANES;
    let mut vmax = vdupq_n_f32(f32::NEG_INFINITY);
    for i in 0..chunks {
        let v = unsafe { vld1q_f32(ptr.add(i * LANES)) };
        vmax = vmaxq_f32(vmax, v);
    }
    let mut max_val = vmaxvq_f32(vmax);
    for val in &data[chunks * LANES..len] {
        max_val = max_val.max(*val);
    }
    if max_val == f32::NEG_INFINITY {
        return;
    }

    // Phase 2 – exp(x − max)
    let vmax_s = vdupq_n_f32(max_val);
    let mut vsum = vdupq_n_f32(0.0);
    let out_ptr = data.as_mut_ptr();
    for i in 0..chunks {
        let v = unsafe { vld1q_f32(ptr.add(i * LANES)) };
        let shifted = vsubq_f32(v, vmax_s);
        let mut arr = [0.0f32; LANES];
        unsafe { vst1q_f32(arr.as_mut_ptr(), shifted) };
        for a in &mut arr {
            *a = a.exp();
        }
        let exp_v = unsafe { vld1q_f32(arr.as_ptr()) };
        vsum = vaddq_f32(vsum, exp_v);
        unsafe { vst1q_f32(out_ptr.add(i * LANES), exp_v) };
    }
    let mut sum = vaddvq_f32(vsum);
    for val in &mut data[chunks * LANES..len] {
        let e = (*val - max_val).exp();
        *val = e;
        sum += e;
    }

    // Phase 3 – normalise
    if sum > 0.0 {
        let inv = 1.0 / sum;
        let vinv = vdupq_n_f32(inv);
        for i in 0..chunks {
            let v = unsafe { vld1q_f32(out_ptr.add(i * LANES) as *const f32) };
            let normed = vmulq_f32(v, vinv);
            unsafe { vst1q_f32(out_ptr.add(i * LANES), normed) };
        }
        for val in &mut data[chunks * LANES..len] {
            *val *= inv;
        }
    }
}

/// Scalar weighted accumulate: `out[i] += src[i] * w`.
#[inline]
#[allow(dead_code)]
fn scalar_weighted_acc(out: &mut [f32], src: &[f32], w: f32) {
    for (o, s) in out.iter_mut().zip(src.iter()) {
        *o += s * w;
    }
}

/// NEON weighted accumulate.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_weighted_acc(out: &mut [f32], src: &[f32], w: f32) {
    let len = out.len().min(src.len());
    let chunks = len / LANES;
    let vw = vdupq_n_f32(w);
    let op = out.as_mut_ptr();
    for i in 0..chunks {
        let vo = unsafe { vld1q_f32(op.add(i * LANES) as *const f32) };
        let vs = unsafe { vld1q_f32(src.as_ptr().add(i * LANES)) };
        let r = vfmaq_f32(vo, vs, vw);
        unsafe { vst1q_f32(op.add(i * LANES), r) };
    }
    for i in (chunks * LANES)..len {
        out[i] += src[i] * w;
    }
}

// ── 1. Build sliding window mask ───────────────────────────────────────

/// Generate a causal sliding window mask.
///
/// Returns a `seq_len × seq_len` boolean vector in row‑major order.
/// `mask[i * seq_len + j]` is `true` when query position `i` may attend
/// to key position `j` (i.e. `j <= i` and `i − j < window_size`).
pub fn build_sliding_window_mask(seq_len: usize, window_size: usize) -> Vec<bool> {
    let ws = window_size.max(1);
    let mut mask = vec![false; seq_len * seq_len];
    for i in 0..seq_len {
        let start = if i >= ws { i - ws + 1 } else { 0 };
        for j in start..=i {
            mask[i * seq_len + j] = true;
        }
    }
    mask
}

// ── 2. Sliding window Q*K^T ────────────────────────────────────────────

/// Compute scaled Q·Kᵀ scores with a causal sliding window mask.
///
/// Returns `seq_len × seq_len` scores. Out-of-window positions are set to
/// `f32::NEG_INFINITY`.
///
/// Layout: Q and K are `[seq_len, head_dim]` in row‑major order.
pub fn sliding_window_qk_neon(
    q: &[f32],
    k: &[f32],
    seq_len: usize,
    head_dim: usize,
    window_size: usize,
    scale: f32,
) -> Vec<f32> {
    assert!(q.len() >= seq_len * head_dim, "q too short: {} < {}", q.len(), seq_len * head_dim);
    assert!(k.len() >= seq_len * head_dim, "k too short: {} < {}", k.len(), seq_len * head_dim);
    if seq_len == 0 || head_dim == 0 {
        return vec![];
    }
    let ws = window_size.max(1);
    let mut scores = vec![f32::NEG_INFINITY; seq_len * seq_len];

    #[cfg(target_arch = "aarch64")]
    {
        for i in 0..seq_len {
            let q_row = &q[i * head_dim..(i + 1) * head_dim];
            let start = if i >= ws { i - ws + 1 } else { 0 };
            for j in start..=i {
                let k_row = &k[j * head_dim..(j + 1) * head_dim];
                scores[i * seq_len + j] = unsafe { neon_dot(q_row, k_row) } * scale;
            }
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..seq_len {
            let q_row = &q[i * head_dim..(i + 1) * head_dim];
            let start = if i >= ws { i - ws + 1 } else { 0 };
            for j in start..=i {
                let k_row = &k[j * head_dim..(j + 1) * head_dim];
                scores[i * seq_len + j] = scalar_dot(q_row, k_row) * scale;
            }
        }
    }

    scores
}

// ── 3. Sliding window softmax ──────────────────────────────────────────

/// Apply softmax over only the valid (non‑masked) window positions for
/// each query row.
///
/// `scores` is `[seq_len, seq_len]` in row‑major order. Out‑of‑window
/// entries must be `f32::NEG_INFINITY`.
///
/// Returns a new `[seq_len, seq_len]` tensor of attention weights.
pub fn sliding_window_softmax_neon(scores: &[f32], window_size: usize, seq_len: usize) -> Vec<f32> {
    if seq_len == 0 {
        return vec![];
    }
    assert!(
        scores.len() >= seq_len * seq_len,
        "scores too short: {} < {}",
        scores.len(),
        seq_len * seq_len
    );

    let mut out = scores[..seq_len * seq_len].to_vec();

    for i in 0..seq_len {
        let row = &mut out[i * seq_len..(i + 1) * seq_len];

        #[cfg(target_arch = "aarch64")]
        unsafe {
            neon_softmax_inplace(row);
        }

        #[cfg(not(target_arch = "aarch64"))]
        scalar_softmax_inplace(row);
    }

    // Zero out masked positions so they contribute nothing to V accumulation.
    let ws = window_size.max(1);
    for i in 0..seq_len {
        let start = if i >= ws { i - ws + 1 } else { 0 };
        // Positions before the window start.
        for j in 0..start {
            out[i * seq_len + j] = 0.0;
        }
        // Positions after the causal boundary.
        for j in (i + 1)..seq_len {
            out[i * seq_len + j] = 0.0;
        }
    }

    out
}

// ── 4. Sliding window attention (single‑head) ─────────────────────────

/// Full single‑head sliding window attention.
///
/// Q, K, V are `[seq_len, head_dim]` in row‑major order.
/// Returns `[seq_len, head_dim]` output.
pub fn sliding_window_attention_neon(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    window_size: usize,
) -> Vec<f32> {
    if seq_len == 0 || head_dim == 0 {
        return vec![];
    }
    let expected = seq_len * head_dim;
    assert!(q.len() >= expected, "q too short");
    assert!(k.len() >= expected, "k too short");
    assert!(v.len() >= expected, "v too short");

    let scale = 1.0 / (head_dim as f32).sqrt();
    let scores = sliding_window_qk_neon(q, k, seq_len, head_dim, window_size, scale);
    let weights = sliding_window_softmax_neon(&scores, window_size, seq_len);

    let mut output = vec![0.0f32; expected];

    #[cfg(target_arch = "aarch64")]
    {
        for i in 0..seq_len {
            let out_row = &mut output[i * head_dim..(i + 1) * head_dim];
            let ws = window_size.max(1);
            let start = if i >= ws { i - ws + 1 } else { 0 };
            for j in start..=i {
                let w = weights[i * seq_len + j];
                if w == 0.0 {
                    continue;
                }
                let v_row = &v[j * head_dim..(j + 1) * head_dim];
                unsafe {
                    neon_weighted_acc(out_row, v_row, w);
                }
            }
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..seq_len {
            let out_row = &mut output[i * head_dim..(i + 1) * head_dim];
            let ws = window_size.max(1);
            let start = if i >= ws { i - ws + 1 } else { 0 };
            for j in start..=i {
                let w = weights[i * seq_len + j];
                if w == 0.0 {
                    continue;
                }
                let v_row = &v[j * head_dim..(j + 1) * head_dim];
                scalar_weighted_acc(out_row, v_row, w);
            }
        }
    }

    output
}

// ── 5. Multi‑head sliding window attention ─────────────────────────────

/// Multi‑head sliding window attention.
///
/// Q, K, V are `[num_heads, seq_len, head_dim]` in row‑major order.
/// Returns `[num_heads, seq_len, head_dim]`.
pub fn multi_head_sliding_window_neon(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    window_size: usize,
) -> Vec<f32> {
    if seq_len == 0 || head_dim == 0 || num_heads == 0 {
        return vec![];
    }
    let head_elems = seq_len * head_dim;
    let total = num_heads * head_elems;
    assert!(q.len() >= total, "q too short for multi‑head");
    assert!(k.len() >= total, "k too short for multi‑head");
    assert!(v.len() >= total, "v too short for multi‑head");

    let mut output = vec![0.0f32; total];

    for h in 0..num_heads {
        let offset = h * head_elems;
        let q_head = &q[offset..offset + head_elems];
        let k_head = &k[offset..offset + head_elems];
        let v_head = &v[offset..offset + head_elems];
        let head_out =
            sliding_window_attention_neon(q_head, k_head, v_head, seq_len, head_dim, window_size);
        output[offset..offset + head_elems].copy_from_slice(&head_out);
    }

    output
}

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;

    // ── Test utilities ─────────────────────────────────────────────────

    /// Reference single‑head attention (no window – full causal).
    fn reference_causal_attention(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut scores = vec![f32::NEG_INFINITY; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..=i {
                let q_row = &q[i * head_dim..(i + 1) * head_dim];
                let k_row = &k[j * head_dim..(j + 1) * head_dim];
                scores[i * seq_len + j] = scalar_dot(q_row, k_row) * scale;
            }
        }
        // softmax per row
        for i in 0..seq_len {
            let row = &mut scores[i * seq_len..(i + 1) * seq_len];
            scalar_softmax_inplace(row);
            // zero masked
            for j in (i + 1)..seq_len {
                row[j] = 0.0;
            }
        }
        // weighted sum
        let mut out = vec![0.0f32; seq_len * head_dim];
        for i in 0..seq_len {
            for j in 0..=i {
                let w = scores[i * seq_len + j];
                for d in 0..head_dim {
                    out[i * head_dim + d] += w * v[j * head_dim + d];
                }
            }
        }
        out
    }

    /// Approximate float comparison.
    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        if a == b {
            return true; // handles ±inf equality
        }
        (a - b).abs() <= tol
    }

    fn assert_slices_approx(a: &[f32], b: &[f32], tol: f32, ctx: &str) {
        assert_eq!(a.len(), b.len(), "{ctx}: length mismatch");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(approx_eq(x, y, tol), "{ctx}[{i}]: {x} vs {y} (tol={tol})");
        }
    }

    /// Simple identity‑like Q/K/V for easy verification.
    fn make_identity_qkv(seq_len: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let n = seq_len * head_dim;
        let q: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let k = q.clone();
        let v: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.001).collect();
        (q, k, v)
    }

    /// Ones Q/K, ones V for trivial verification.
    fn make_ones_qkv(seq_len: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let n = seq_len * head_dim;
        (vec![1.0; n], vec![1.0; n], vec![1.0; n])
    }

    // ── build_sliding_window_mask ──────────────────────────────────────

    #[test]
    fn test_mask_1x1() {
        let mask = build_sliding_window_mask(1, 1);
        assert_eq!(mask, vec![true]);
    }

    #[test]
    fn test_mask_2x2_window1() {
        let mask = build_sliding_window_mask(2, 1);
        // row 0: [T, F]  row 1: [F, T]
        assert_eq!(mask, vec![true, false, false, true]);
    }

    #[test]
    fn test_mask_3x3_window2() {
        let mask = build_sliding_window_mask(3, 2);
        // row 0: [T, F, F]
        // row 1: [T, T, F]
        // row 2: [F, T, T]
        assert_eq!(mask, vec![true, false, false, true, true, false, false, true, true]);
    }

    #[test]
    fn test_mask_4x4_window2() {
        let mask = build_sliding_window_mask(4, 2);
        #[rustfmt::skip]
        let expected = vec![
            true,  false, false, false,
            true,  true,  false, false,
            false, true,  true,  false,
            false, false, true,  true,
        ];
        assert_eq!(mask, expected);
    }

    #[test]
    fn test_mask_window_equals_seq_len() {
        // Window covers entire sequence → full causal mask.
        let mask = build_sliding_window_mask(4, 4);
        #[rustfmt::skip]
        let expected = vec![
            true,  false, false, false,
            true,  true,  false, false,
            true,  true,  true,  false,
            true,  true,  true,  true,
        ];
        assert_eq!(mask, expected);
    }

    #[test]
    fn test_mask_window_larger_than_seq_len() {
        // Window larger than seq_len → same as full causal.
        let mask = build_sliding_window_mask(3, 100);
        #[rustfmt::skip]
        let expected = vec![
            true,  false, false,
            true,  true,  false,
            true,  true,  true,
        ];
        assert_eq!(mask, expected);
    }

    #[test]
    fn test_mask_empty() {
        assert!(build_sliding_window_mask(0, 5).is_empty());
    }

    #[test]
    fn test_mask_window_zero_clamped_to_one() {
        // window_size=0 is clamped to 1 → diagonal only.
        let mask = build_sliding_window_mask(3, 0);
        #[rustfmt::skip]
        let expected = vec![
            true,  false, false,
            false, true,  false,
            false, false, true,
        ];
        assert_eq!(mask, expected);
    }

    #[test]
    fn test_mask_diagonal_count() {
        // Each row should have exactly min(window_size, i+1) true entries.
        for seq in [1, 2, 4, 8, 16] {
            for ws in [1, 2, 3, 4, 8, 32] {
                let mask = build_sliding_window_mask(seq, ws);
                for i in 0..seq {
                    let count = mask[i * seq..(i + 1) * seq].iter().filter(|&&b| b).count();
                    let expected = (i + 1).min(ws);
                    assert_eq!(count, expected, "seq={seq} ws={ws} row={i}");
                }
            }
        }
    }

    #[test]
    fn test_mask_causal_upper_triangle_false() {
        for seq in [1, 3, 5, 8] {
            for ws in [1, 2, 4, 100] {
                let mask = build_sliding_window_mask(seq, ws);
                for i in 0..seq {
                    for j in (i + 1)..seq {
                        assert!(!mask[i * seq + j], "seq={seq} ws={ws} ({i},{j})");
                    }
                }
            }
        }
    }

    // ── sliding_window_qk_neon ─────────────────────────────────────────

    #[test]
    fn test_qk_empty() {
        let out = sliding_window_qk_neon(&[], &[], 0, 0, 2, 1.0);
        assert!(out.is_empty());
    }

    #[test]
    fn test_qk_single_token() {
        let q = vec![1.0, 0.0, 0.0, 0.0];
        let k = vec![1.0, 0.0, 0.0, 0.0];
        let scores = sliding_window_qk_neon(&q, &k, 1, 4, 1, 1.0);
        assert_eq!(scores.len(), 1);
        assert!(approx_eq(scores[0], 1.0, 1e-5));
    }

    #[test]
    fn test_qk_scaling() {
        let q = vec![2.0; 4];
        let k = vec![2.0; 4];
        let scores = sliding_window_qk_neon(&q, &k, 1, 4, 1, 0.5);
        // dot = 2*2*4 = 16, scaled by 0.5 → 8
        assert!(approx_eq(scores[0], 8.0, 1e-5));
    }

    #[test]
    fn test_qk_out_of_window_is_neg_inf() {
        // 3 tokens, window=1 → only diagonal valid
        let (q, k, _) = make_identity_qkv(3, 4);
        let scores = sliding_window_qk_neon(&q, &k, 3, 4, 1, 1.0);
        // (0,1) should be -inf
        assert_eq!(scores[0 * 3 + 1], f32::NEG_INFINITY);
        // (0,2) should be -inf
        assert_eq!(scores[0 * 3 + 2], f32::NEG_INFINITY);
        // (1,0) should be -inf (window=1 means only j==i)
        assert_eq!(scores[1 * 3 + 0], f32::NEG_INFINITY);
        // (2,0) should be -inf
        assert_eq!(scores[2 * 3 + 0], f32::NEG_INFINITY);
        // (2,1) should be -inf
        assert_eq!(scores[2 * 3 + 1], f32::NEG_INFINITY);
        // diag should be finite
        for i in 0..3 {
            assert!(scores[i * 3 + i].is_finite(), "diag[{i}]");
        }
    }

    #[test]
    fn test_qk_window2() {
        // 4 tokens, window=2 → each row has at most 2 valid entries
        let q = vec![1.0; 4 * 2]; // head_dim=2
        let k = vec![1.0; 4 * 2];
        let scores = sliding_window_qk_neon(&q, &k, 4, 2, 2, 1.0);
        // row 0: only (0,0) valid
        assert!(scores[0].is_finite());
        assert_eq!(scores[1], f32::NEG_INFINITY);
        // row 1: (1,0) and (1,1) valid
        assert!(scores[1 * 4 + 0].is_finite());
        assert!(scores[1 * 4 + 1].is_finite());
        // row 2: (2,1) and (2,2) valid
        assert_eq!(scores[2 * 4 + 0], f32::NEG_INFINITY);
        assert!(scores[2 * 4 + 1].is_finite());
        assert!(scores[2 * 4 + 2].is_finite());
        // row 3: (3,2) and (3,3) valid
        assert_eq!(scores[3 * 4 + 0], f32::NEG_INFINITY);
        assert_eq!(scores[3 * 4 + 1], f32::NEG_INFINITY);
        assert!(scores[3 * 4 + 2].is_finite());
        assert!(scores[3 * 4 + 3].is_finite());
    }

    #[test]
    fn test_qk_full_window_equals_causal() {
        let (q, k, _) = make_identity_qkv(4, 4);
        let full = sliding_window_qk_neon(&q, &k, 4, 4, 100, 0.5);
        let causal = sliding_window_qk_neon(&q, &k, 4, 4, 4, 0.5);
        assert_slices_approx(&full, &causal, 1e-5, "full_vs_causal");
    }

    #[test]
    fn test_qk_orthogonal_vectors() {
        // q=[1,0,0,0], k=[0,1,0,0] → dot = 0
        let q = vec![1.0, 0.0, 0.0, 0.0];
        let k = vec![0.0, 1.0, 0.0, 0.0];
        let scores = sliding_window_qk_neon(&q, &k, 1, 4, 1, 1.0);
        assert!(approx_eq(scores[0], 0.0, 1e-6));
    }

    #[test]
    fn test_qk_negative_values() {
        let q = vec![-1.0, -2.0];
        let k = vec![3.0, 4.0];
        let scores = sliding_window_qk_neon(&q, &k, 1, 2, 1, 1.0);
        // dot = -3 + -8 = -11
        assert!(approx_eq(scores[0], -11.0, 1e-5));
    }

    #[test]
    fn test_qk_head_dim_not_multiple_of_4() {
        // head_dim = 3 (not multiple of LANES)
        let q = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 tokens
        let k = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let scores = sliding_window_qk_neon(&q, &k, 2, 3, 10, 1.0);
        // (0,0): dot([1,2,3],[1,1,1]) = 6
        assert!(approx_eq(scores[0], 6.0, 1e-5));
        // (1,1): dot([4,5,6],[1,1,1]) = 15
        assert!(approx_eq(scores[1 * 2 + 1], 15.0, 1e-5));
    }

    // ── sliding_window_softmax_neon ────────────────────────────────────

    #[test]
    fn test_softmax_empty() {
        let out = sliding_window_softmax_neon(&[], 2, 0);
        assert!(out.is_empty());
    }

    #[test]
    fn test_softmax_single_valid() {
        // Only one valid entry → weight = 1.0
        let scores = vec![5.0];
        let out = sliding_window_softmax_neon(&scores, 1, 1);
        assert!(approx_eq(out[0], 1.0, 1e-5));
    }

    #[test]
    fn test_softmax_row_sums_to_one() {
        let (q, k, _) = make_identity_qkv(4, 4);
        let scale = 1.0 / (4.0f32).sqrt();
        let scores = sliding_window_qk_neon(&q, &k, 4, 4, 3, scale);
        let weights = sliding_window_softmax_neon(&scores, 3, 4);
        for i in 0..4 {
            let row_sum: f32 = weights[i * 4..(i + 1) * 4].iter().sum();
            assert!(approx_eq(row_sum, 1.0, 1e-4), "row {i} sum = {row_sum}");
        }
    }

    #[test]
    fn test_softmax_masked_positions_zero() {
        let scores = vec![
            1.0,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            1.0,
            1.0,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            1.0,
            1.0,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            1.0,
            1.0,
        ];
        let out = sliding_window_softmax_neon(&scores, 2, 4);
        // Row 0: only (0,0) valid → 1.0
        assert!(approx_eq(out[0], 1.0, 1e-5));
        for j in 1..4 {
            assert!(approx_eq(out[j], 0.0, 1e-6));
        }
        // Row 2: (2,1) and (2,2) valid, (2,0) and (2,3) = 0
        assert!(approx_eq(out[2 * 4 + 0], 0.0, 1e-6));
        assert!(approx_eq(out[2 * 4 + 3], 0.0, 1e-6));
        assert!(out[2 * 4 + 1] > 0.0);
        assert!(out[2 * 4 + 2] > 0.0);
    }

    #[test]
    fn test_softmax_equal_scores_uniform() {
        // 4 tokens, window=4 (full causal), equal scores → row i has uniform 1/(i+1)
        let mut scores = vec![f32::NEG_INFINITY; 16];
        for i in 0..4 {
            for j in 0..=i {
                scores[i * 4 + j] = 0.0;
            }
        }
        let weights = sliding_window_softmax_neon(&scores, 4, 4);
        // row 0: 1 entry → [1, 0, 0, 0]
        assert!(approx_eq(weights[0], 1.0, 1e-5));
        // row 1: 2 entries → [0.5, 0.5, 0, 0]
        assert!(approx_eq(weights[1 * 4 + 0], 0.5, 1e-5));
        assert!(approx_eq(weights[1 * 4 + 1], 0.5, 1e-5));
        // row 3: 4 entries → [0.25, 0.25, 0.25, 0.25]
        for j in 0..4 {
            assert!(approx_eq(weights[3 * 4 + j], 0.25, 1e-5));
        }
    }

    #[test]
    fn test_softmax_numerical_stability_large_values() {
        // Large scores should not overflow.
        let scores = vec![
            1000.0,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            999.0,
            1000.0,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            999.0,
            1000.0,
        ];
        let out = sliding_window_softmax_neon(&scores, 2, 3);
        for i in 0..3 {
            let row_sum: f32 = out[i * 3..(i + 1) * 3].iter().sum();
            assert!(approx_eq(row_sum, 1.0, 1e-4), "row {i} sum = {row_sum}");
            for j in 0..3 {
                let val = out[i * 3 + j];
                assert!(!val.is_nan(), "NaN at ({i},{j})");
                assert!(!val.is_infinite(), "Inf at ({i},{j})");
            }
        }
    }

    #[test]
    fn test_softmax_numerical_stability_negative_values() {
        let scores = vec![-1000.0, f32::NEG_INFINITY, -999.0, -1000.0];
        let out = sliding_window_softmax_neon(&scores, 2, 2);
        for i in 0..2 {
            let row_sum: f32 = out[i * 2..(i + 1) * 2].iter().sum();
            assert!(approx_eq(row_sum, 1.0, 1e-4), "row {i} sum={row_sum}");
        }
    }

    #[test]
    fn test_softmax_weights_nonnegative() {
        let (q, k, _) = make_identity_qkv(8, 4);
        let scores = sliding_window_qk_neon(&q, &k, 8, 4, 3, 0.5);
        let weights = sliding_window_softmax_neon(&scores, 3, 8);
        for (i, &w) in weights.iter().enumerate() {
            assert!(w >= 0.0, "negative weight at {i}: {w}");
        }
    }

    // ── sliding_window_attention_neon ──────────────────────────────────

    #[test]
    fn test_attn_empty() {
        let out = sliding_window_attention_neon(&[], &[], &[], 0, 0, 2);
        assert!(out.is_empty());
    }

    #[test]
    fn test_attn_single_token() {
        let q = vec![1.0, 0.0, 0.0, 0.0];
        let k = vec![1.0, 0.0, 0.0, 0.0];
        let v = vec![0.5, 0.5, 0.5, 0.5];
        let out = sliding_window_attention_neon(&q, &k, &v, 1, 4, 1);
        // Single token → attention weight = 1.0 → output = v
        assert_slices_approx(&out, &v, 1e-5, "single_token");
    }

    #[test]
    fn test_attn_window1_diagonal() {
        // window=1 → each query only attends to itself → output[i] = v[i]
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 4 tokens, hd=2
        let q = vec![1.0; 8];
        let k = vec![1.0; 8];
        let out = sliding_window_attention_neon(&q, &k, &v, 4, 2, 1);
        assert_slices_approx(&out, &v, 1e-4, "window1_diag");
    }

    #[test]
    fn test_attn_full_window_matches_causal() {
        let (q, k, v) = make_identity_qkv(4, 8);
        let windowed = sliding_window_attention_neon(&q, &k, &v, 4, 8, 100);
        let causal = reference_causal_attention(&q, &k, &v, 4, 8);
        assert_slices_approx(&windowed, &causal, 1e-4, "full_window_vs_causal");
    }

    #[test]
    fn test_attn_ones_uniform() {
        // All-ones Q/K/V → all scores equal within window.
        let (q, k, v) = make_ones_qkv(3, 4);
        let out = sliding_window_attention_neon(&q, &k, &v, 3, 4, 10);
        // Output should be all 1s (weighted average of all-ones V).
        for &x in &out {
            assert!(approx_eq(x, 1.0, 1e-4), "expected 1.0 got {x}");
        }
    }

    #[test]
    fn test_attn_output_length() {
        let (q, k, v) = make_identity_qkv(5, 8);
        let out = sliding_window_attention_neon(&q, &k, &v, 5, 8, 3);
        assert_eq!(out.len(), 5 * 8);
    }

    #[test]
    fn test_attn_no_nan_no_inf() {
        let (q, k, v) = make_identity_qkv(6, 4);
        let out = sliding_window_attention_neon(&q, &k, &v, 6, 4, 3);
        for (i, &x) in out.iter().enumerate() {
            assert!(!x.is_nan(), "NaN at {i}");
            assert!(!x.is_infinite(), "Inf at {i}");
        }
    }

    #[test]
    fn test_attn_window_size_1_isolation() {
        // With window=1, position i only sees itself.
        // Make V rows distinct; output[i] must equal V[i].
        let seq_len = 5;
        let hd = 4;
        let q = vec![1.0; seq_len * hd];
        let k = vec![1.0; seq_len * hd];
        let v: Vec<f32> = (0..seq_len * hd).map(|i| (i as f32) * 0.1 + 1.0).collect();
        let out = sliding_window_attention_neon(&q, &k, &v, seq_len, hd, 1);
        for i in 0..seq_len {
            let expected_row = &v[i * hd..(i + 1) * hd];
            let actual_row = &out[i * hd..(i + 1) * hd];
            assert_slices_approx(actual_row, expected_row, 1e-4, &format!("row {i}"));
        }
    }

    #[test]
    fn test_attn_head_dim_1() {
        let q = vec![1.0, 2.0, 3.0];
        let k = vec![1.0, 1.0, 1.0];
        let v = vec![10.0, 20.0, 30.0];
        let out = sliding_window_attention_neon(&q, &k, &v, 3, 1, 2);
        assert_eq!(out.len(), 3);
        // Position 0: only sees itself → output = v[0] = 10
        assert!(approx_eq(out[0], 10.0, 1e-3));
    }

    #[test]
    fn test_attn_head_dim_not_multiple_of_lanes() {
        // head_dim = 3 (not multiple of 4)
        let q = vec![1.0; 6]; // 2 tokens × hd=3
        let k = vec![1.0; 6];
        let v = vec![2.0; 6];
        let out = sliding_window_attention_neon(&q, &k, &v, 2, 3, 10);
        for &x in &out {
            assert!(approx_eq(x, 2.0, 1e-4));
        }
    }

    #[test]
    fn test_attn_large_head_dim() {
        let hd = 64;
        let seq = 4;
        let (q, k, v) = make_identity_qkv(seq, hd);
        let out = sliding_window_attention_neon(&q, &k, &v, seq, hd, 2);
        assert_eq!(out.len(), seq * hd);
        for &x in &out {
            assert!(!x.is_nan());
            assert!(!x.is_infinite());
        }
    }

    // ── multi_head_sliding_window_neon ─────────────────────────────────

    #[test]
    fn test_mh_empty() {
        let out = multi_head_sliding_window_neon(&[], &[], &[], 0, 0, 0, 2);
        assert!(out.is_empty());
    }

    #[test]
    fn test_mh_single_head_matches_single() {
        let (q, k, v) = make_identity_qkv(4, 8);
        let single = sliding_window_attention_neon(&q, &k, &v, 4, 8, 3);
        let multi = multi_head_sliding_window_neon(&q, &k, &v, 4, 1, 8, 3);
        assert_slices_approx(&multi, &single, 1e-5, "single_head");
    }

    #[test]
    fn test_mh_output_length() {
        let nh = 4;
        let seq = 3;
        let hd = 8;
        let total = nh * seq * hd;
        let q = vec![1.0; total];
        let k = vec![1.0; total];
        let v = vec![1.0; total];
        let out = multi_head_sliding_window_neon(&q, &k, &v, seq, nh, hd, 2);
        assert_eq!(out.len(), total);
    }

    #[test]
    fn test_mh_heads_independent() {
        // Two heads with different V → outputs differ per head.
        let seq = 3;
        let hd = 4;
        let head_elems = seq * hd;
        let q = vec![1.0; 2 * head_elems];
        let k = vec![1.0; 2 * head_elems];
        let mut v = vec![0.0; 2 * head_elems];
        // Head 0: v = 1.0
        for i in 0..head_elems {
            v[i] = 1.0;
        }
        // Head 1: v = 2.0
        for i in head_elems..2 * head_elems {
            v[i] = 2.0;
        }
        let out = multi_head_sliding_window_neon(&q, &k, &v, seq, 2, hd, 10);
        // Head 0 output ≈ 1.0, Head 1 output ≈ 2.0
        for i in 0..head_elems {
            assert!(approx_eq(out[i], 1.0, 1e-4), "head0[{i}]");
        }
        for i in head_elems..2 * head_elems {
            assert!(approx_eq(out[i], 2.0, 1e-4), "head1[{i}]");
        }
    }

    #[test]
    fn test_mh_no_nan() {
        let nh = 2;
        let seq = 5;
        let hd = 4;
        let (q, k, v) = make_identity_qkv(nh * seq, hd);
        // Reinterpret as num_heads=2 (total = 2*5*4 = 40, same as 10*4)
        let out = multi_head_sliding_window_neon(&q, &k, &v, seq, nh, hd, 3);
        for (i, &x) in out.iter().enumerate() {
            assert!(!x.is_nan(), "NaN at {i}");
            assert!(!x.is_infinite(), "Inf at {i}");
        }
    }

    #[test]
    fn test_mh_four_heads() {
        let nh = 4;
        let seq = 4;
        let hd = 8;
        let total = nh * seq * hd;
        let q: Vec<f32> = (0..total).map(|i| ((i % 7) as f32) * 0.1).collect();
        let k: Vec<f32> = (0..total).map(|i| ((i % 5) as f32) * 0.1).collect();
        let v: Vec<f32> = (0..total).map(|i| ((i % 3) as f32) * 0.1 + 0.5).collect();
        let out = multi_head_sliding_window_neon(&q, &k, &v, seq, nh, hd, 2);
        assert_eq!(out.len(), total);
        for &x in &out {
            assert!(!x.is_nan());
        }
    }

    #[test]
    fn test_mh_window1_each_head() {
        // window=1 → output[h][i] = v[h][i]
        let nh = 3;
        let seq = 4;
        let hd = 2;
        let total = nh * seq * hd;
        let q = vec![1.0; total];
        let k = vec![1.0; total];
        let v: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01 + 1.0).collect();
        let out = multi_head_sliding_window_neon(&q, &k, &v, seq, nh, hd, 1);
        assert_slices_approx(&out, &v, 1e-4, "mh_window1");
    }

    // ── Determinism ────────────────────────────────────────────────────

    #[test]
    fn test_determinism_qk() {
        let (q, k, _) = make_identity_qkv(8, 16);
        let a = sliding_window_qk_neon(&q, &k, 8, 16, 4, 0.5);
        let b = sliding_window_qk_neon(&q, &k, 8, 16, 4, 0.5);
        assert_eq!(a, b);
    }

    #[test]
    fn test_determinism_attention() {
        let (q, k, v) = make_identity_qkv(6, 8);
        let a = sliding_window_attention_neon(&q, &k, &v, 6, 8, 3);
        let b = sliding_window_attention_neon(&q, &k, &v, 6, 8, 3);
        assert_eq!(a, b);
    }

    #[test]
    fn test_determinism_multi_head() {
        let (q, k, v) = make_identity_qkv(4 * 2, 8); // 2 heads × 4 tokens
        let a = multi_head_sliding_window_neon(&q, &k, &v, 4, 2, 8, 3);
        let b = multi_head_sliding_window_neon(&q, &k, &v, 4, 2, 8, 3);
        assert_eq!(a, b);
    }

    #[test]
    fn test_determinism_softmax() {
        let scores = vec![
            1.0,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            0.5,
            1.5,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            0.3,
            0.7,
        ];
        let a = sliding_window_softmax_neon(&scores, 2, 3);
        let b = sliding_window_softmax_neon(&scores, 2, 3);
        assert_eq!(a, b);
    }

    // ── Edge cases ─────────────────────────────────────────────────────

    #[test]
    fn test_window_size_equals_seq_len() {
        let seq = 5;
        let hd = 4;
        let (q, k, v) = make_identity_qkv(seq, hd);
        let windowed = sliding_window_attention_neon(&q, &k, &v, seq, hd, seq);
        let causal = reference_causal_attention(&q, &k, &v, seq, hd);
        assert_slices_approx(&windowed, &causal, 1e-4, "window_eq_seq");
    }

    #[test]
    fn test_window_larger_than_seq() {
        let seq = 3;
        let hd = 4;
        let (q, k, v) = make_identity_qkv(seq, hd);
        let windowed = sliding_window_attention_neon(&q, &k, &v, seq, hd, 999);
        let causal = reference_causal_attention(&q, &k, &v, seq, hd);
        assert_slices_approx(&windowed, &causal, 1e-4, "window_gt_seq");
    }

    #[test]
    fn test_seq_len_1_any_window() {
        for ws in [1, 2, 5, 100] {
            let q = vec![1.0; 4];
            let k = vec![1.0; 4];
            let v = vec![42.0; 4];
            let out = sliding_window_attention_neon(&q, &k, &v, 1, 4, ws);
            assert_slices_approx(&out, &v, 1e-5, &format!("seq1_ws{ws}"));
        }
    }

    #[test]
    fn test_head_dim_equals_1() {
        let q = vec![2.0, 3.0];
        let k = vec![2.0, 3.0];
        let v = vec![10.0, 20.0];
        let out = sliding_window_attention_neon(&q, &k, &v, 2, 1, 10);
        assert_eq!(out.len(), 2);
        assert!(!out[0].is_nan());
        assert!(!out[1].is_nan());
    }

    #[test]
    fn test_zero_q_produces_uniform_weights() {
        // Q = 0 → all valid scores = 0 → uniform softmax
        let q = vec![0.0; 12]; // 3 tokens × hd=4
        let k = vec![1.0; 12];
        let v: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let out = sliding_window_attention_neon(&q, &k, &v, 3, 4, 10);
        // Row 0: only position 0 → v[0..4] = [0,1,2,3]
        assert_slices_approx(&out[0..4], &[0.0, 1.0, 2.0, 3.0], 1e-4, "row0");
    }

    // ── Large sequence ─────────────────────────────────────────────────

    #[test]
    fn test_large_seq_no_panic() {
        let seq = 64;
        let hd = 16;
        let n = seq * hd;
        let q: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001).sin()).collect();
        let k: Vec<f32> = (0..n).map(|i| (i as f32 * 0.002).cos()).collect();
        let v: Vec<f32> = (0..n).map(|i| (i as f32 * 0.003).sin()).collect();
        let out = sliding_window_attention_neon(&q, &k, &v, seq, hd, 8);
        assert_eq!(out.len(), n);
        for &x in &out {
            assert!(!x.is_nan());
            assert!(!x.is_infinite());
        }
    }

    #[test]
    fn test_large_seq_window_small() {
        let seq = 128;
        let hd = 8;
        let n = seq * hd;
        let q = vec![0.1; n];
        let k = vec![0.1; n];
        let v = vec![1.0; n];
        let out = sliding_window_attention_neon(&q, &k, &v, seq, hd, 4);
        // All-ones V → output should be ≈ 1.0 everywhere
        for (i, &x) in out.iter().enumerate() {
            assert!(approx_eq(x, 1.0, 0.1), "pos {i}: expected ~1.0 got {x}");
        }
    }

    #[test]
    fn test_large_multi_head() {
        let nh = 8;
        let seq = 16;
        let hd = 16;
        let total = nh * seq * hd;
        let q: Vec<f32> = (0..total).map(|i| (i as f32 * 0.0001).sin()).collect();
        let k: Vec<f32> = (0..total).map(|i| (i as f32 * 0.0002).cos()).collect();
        let v: Vec<f32> = (0..total).map(|i| (i as f32 * 0.0003).sin() + 0.5).collect();
        let out = multi_head_sliding_window_neon(&q, &k, &v, seq, nh, hd, 4);
        assert_eq!(out.len(), total);
        for &x in &out {
            assert!(!x.is_nan());
        }
    }

    // ── Softmax extra tests ────────────────────────────────────────────

    #[test]
    fn test_softmax_all_neg_inf_row() {
        // An entire row of -inf (shouldn't happen in practice, but must not NaN).
        let scores = vec![0.0, f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY];
        let out = sliding_window_softmax_neon(&scores, 1, 2);
        // Row 0: (0,0)=0.0 valid → weight 1.0
        assert!(approx_eq(out[0], 1.0, 1e-5));
        // Row 1: all masked in softmax, but window=1 should zero-out col 0
        // Post-softmax zero-out: col 0 is before window start (start=1 for i=1, ws=1)
        // col 1 is at causal boundary (j=1=i=1, valid)
        // Actually: the scores have both (1,0)=-inf, (1,1)=-inf.
        // Softmax of all -inf → 0. After zero-out, all 0.
        for &x in &out[2..4] {
            assert!(!x.is_nan(), "got NaN");
        }
    }

    #[test]
    fn test_softmax_very_different_scales() {
        let scores = vec![
            100.0,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            -100.0,
            100.0,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            -100.0,
            100.0,
        ];
        let out = sliding_window_softmax_neon(&scores, 2, 3);
        // Row 0: only (0,0) valid → 1.0
        assert!(approx_eq(out[0], 1.0, 1e-5));
        // Row 1: scores -100 and 100 → second dominates
        assert!(out[1 * 3 + 1] > 0.99);
        // Row 2: scores -100 and 100 → second dominates
        assert!(out[2 * 3 + 2] > 0.99);
    }

    // ── Q*K^T additional tests ─────────────────────────────────────────

    #[test]
    fn test_qk_scale_zero() {
        let q = vec![1.0; 4];
        let k = vec![1.0; 4];
        let scores = sliding_window_qk_neon(&q, &k, 1, 4, 1, 0.0);
        assert!(approx_eq(scores[0], 0.0, 1e-6));
    }

    #[test]
    fn test_qk_scale_negative() {
        let q = vec![1.0; 4];
        let k = vec![1.0; 4];
        let scores = sliding_window_qk_neon(&q, &k, 1, 4, 1, -1.0);
        // dot=4, scale=-1 → -4
        assert!(approx_eq(scores[0], -4.0, 1e-5));
    }

    #[test]
    fn test_qk_large_head_dim_32() {
        let hd = 32;
        let q: Vec<f32> = (0..hd).map(|i| (i as f32) * 0.1).collect();
        let k: Vec<f32> = (0..hd).map(|i| 1.0 - (i as f32) * 0.01).collect();
        let scores = sliding_window_qk_neon(&q, &k, 1, hd, 1, 1.0);
        let expected = scalar_dot(&q, &k);
        assert!(approx_eq(scores[0], expected, 1e-3));
    }

    #[test]
    fn test_qk_symmetry_on_diagonal() {
        // When Q == K, diagonal scores should be identical if Q rows are equal.
        let q = vec![1.0; 3 * 4]; // 3 tokens, hd=4, all ones
        let k = q.clone();
        let scores = sliding_window_qk_neon(&q, &k, 3, 4, 10, 1.0);
        // All diagonal entries should equal dot([1,1,1,1],[1,1,1,1]) = 4
        for i in 0..3 {
            assert!(approx_eq(scores[i * 3 + i], 4.0, 1e-5));
        }
    }

    // ── Attention correctness deep-dives ───────────────────────────────

    #[test]
    fn test_attn_window2_blending() {
        // 3 tokens, window=2, hd=1 for easy manual computation.
        // Q=[1,1,1], K=[1,1,1] → all valid scores = 1.0
        // scale = 1/sqrt(1) = 1
        // Scores (causal, window=2):
        //   row 0: [1, -inf, -inf]
        //   row 1: [1, 1, -inf]
        //   row 2: [-inf, 1, 1]
        // Softmax row 0: [1, 0, 0]
        // Softmax row 1: [0.5, 0.5, 0]
        // Softmax row 2: [0, 0.5, 0.5]
        let q = vec![1.0, 1.0, 1.0];
        let k = vec![1.0, 1.0, 1.0];
        let v = vec![10.0, 20.0, 30.0];
        let out = sliding_window_attention_neon(&q, &k, &v, 3, 1, 2);
        assert!(approx_eq(out[0], 10.0, 1e-3), "row0");
        assert!(approx_eq(out[1], 15.0, 1e-3), "row1");
        assert!(approx_eq(out[2], 25.0, 1e-3), "row2");
    }

    #[test]
    fn test_attn_window3_3tokens() {
        // window >= seq_len → full causal
        let q = vec![1.0, 1.0, 1.0];
        let k = vec![1.0, 1.0, 1.0];
        let v = vec![10.0, 20.0, 30.0];
        let out = sliding_window_attention_neon(&q, &k, &v, 3, 1, 3);
        // row 0: [1,0,0] → 10
        // row 1: [0.5,0.5,0] → 15
        // row 2: [0.333,0.333,0.333] → 20
        assert!(approx_eq(out[0], 10.0, 1e-3));
        assert!(approx_eq(out[1], 15.0, 1e-3));
        assert!(approx_eq(out[2], 20.0, 1e-3));
    }

    #[test]
    fn test_attn_preserves_v_magnitude() {
        // Attention weights sum to 1 → output magnitude bounded by V magnitude.
        let (q, k, _) = make_identity_qkv(8, 4);
        let v: Vec<f32> = (0..32).map(|_| 5.0).collect();
        let out = sliding_window_attention_neon(&q, &k, &v, 8, 4, 4);
        for &x in &out {
            assert!(x.abs() <= 5.0 + 1e-3, "output exceeds V magnitude: {x}");
        }
    }

    // ── Mask consistency with attention ─────────────────────────────────

    #[test]
    fn test_mask_agrees_with_qk_scores() {
        let seq = 5;
        let hd = 4;
        let ws = 3;
        let mask = build_sliding_window_mask(seq, ws);
        let (q, k, _) = make_identity_qkv(seq, hd);
        let scores = sliding_window_qk_neon(&q, &k, seq, hd, ws, 1.0);
        for i in 0..seq {
            for j in 0..seq {
                let idx = i * seq + j;
                if mask[idx] {
                    assert!(scores[idx].is_finite(), "mask true but score -inf at ({i},{j})");
                } else {
                    assert_eq!(
                        scores[idx],
                        f32::NEG_INFINITY,
                        "mask false but score finite at ({i},{j})"
                    );
                }
            }
        }
    }

    #[test]
    fn test_mask_agrees_with_softmax_zeros() {
        let seq = 4;
        let ws = 2;
        let mask = build_sliding_window_mask(seq, ws);
        let (q, k, _) = make_identity_qkv(seq, 4);
        let scores = sliding_window_qk_neon(&q, &k, seq, 4, ws, 0.5);
        let weights = sliding_window_softmax_neon(&scores, ws, seq);
        for i in 0..seq {
            for j in 0..seq {
                let idx = i * seq + j;
                if !mask[idx] {
                    assert!(
                        approx_eq(weights[idx], 0.0, 1e-6),
                        "mask false but weight non-zero at ({i},{j}): {}",
                        weights[idx]
                    );
                }
            }
        }
    }

    // ── Scalar fallback correctness ────────────────────────────────────

    #[test]
    fn test_scalar_softmax_basic() {
        let mut data = vec![1.0, 2.0, 3.0];
        scalar_softmax_inplace(&mut data);
        let sum: f32 = data.iter().sum();
        assert!(approx_eq(sum, 1.0, 1e-5));
        // Largest input → largest probability
        assert!(data[2] > data[1]);
        assert!(data[1] > data[0]);
    }

    #[test]
    fn test_scalar_softmax_single() {
        let mut data = vec![42.0];
        scalar_softmax_inplace(&mut data);
        assert!(approx_eq(data[0], 1.0, 1e-6));
    }

    #[test]
    fn test_scalar_softmax_all_neg_inf() {
        let mut data = vec![f32::NEG_INFINITY; 4];
        scalar_softmax_inplace(&mut data);
        // Should not NaN – early exit
        for &x in &data {
            assert!(!x.is_nan());
        }
    }

    #[test]
    fn test_scalar_softmax_large_values() {
        let mut data = vec![1000.0, 1001.0, 999.0];
        scalar_softmax_inplace(&mut data);
        let sum: f32 = data.iter().sum();
        assert!(approx_eq(sum, 1.0, 1e-5));
        assert!(data[1] > data[0]); // 1001 > 1000
        assert!(data[0] > data[2]); // 1000 > 999
    }

    #[test]
    fn test_scalar_dot_basic() {
        assert!(approx_eq(scalar_dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0, 1e-5));
    }

    #[test]
    fn test_scalar_dot_zero() {
        assert!(approx_eq(scalar_dot(&[0.0; 4], &[1.0; 4]), 0.0, 1e-6));
    }

    #[test]
    fn test_scalar_dot_negative() {
        assert!(approx_eq(scalar_dot(&[-1.0, 2.0], &[3.0, -4.0]), -11.0, 1e-5));
    }

    // ── Softmax window boundary ────────────────────────────────────────

    #[test]
    fn test_softmax_window_boundary_exact() {
        // 4 tokens, window=2: row 2 should only have weights at (2,1) and (2,2).
        let mut scores = vec![f32::NEG_INFINITY; 16];
        for i in 0..4 {
            let start = if i >= 2 { i - 1 } else { 0 };
            for j in start..=i {
                scores[i * 4 + j] = 1.0;
            }
        }
        let weights = sliding_window_softmax_neon(&scores, 2, 4);
        // Row 2: positions 0 and 3 must be zero.
        assert!(approx_eq(weights[2 * 4 + 0], 0.0, 1e-6));
        assert!(approx_eq(weights[2 * 4 + 3], 0.0, 1e-6));
        // Positions 1 and 2 should be 0.5 each.
        assert!(approx_eq(weights[2 * 4 + 1], 0.5, 1e-4));
        assert!(approx_eq(weights[2 * 4 + 2], 0.5, 1e-4));
    }

    // ── Various head_dim sizes ─────────────────────────────────────────

    #[test]
    fn test_attn_head_dim_5() {
        let hd = 5;
        let seq = 3;
        let (q, k, v) = make_identity_qkv(seq, hd);
        let out = sliding_window_attention_neon(&q, &k, &v, seq, hd, 2);
        assert_eq!(out.len(), seq * hd);
        for &x in &out {
            assert!(!x.is_nan());
        }
    }

    #[test]
    fn test_attn_head_dim_7() {
        let hd = 7;
        let seq = 4;
        let (q, k, v) = make_identity_qkv(seq, hd);
        let out = sliding_window_attention_neon(&q, &k, &v, seq, hd, 3);
        assert_eq!(out.len(), seq * hd);
    }

    #[test]
    fn test_attn_head_dim_16() {
        let hd = 16;
        let seq = 6;
        let (q, k, v) = make_identity_qkv(seq, hd);
        let windowed = sliding_window_attention_neon(&q, &k, &v, seq, hd, seq);
        let causal = reference_causal_attention(&q, &k, &v, seq, hd);
        assert_slices_approx(&windowed, &causal, 1e-3, "hd16");
    }

    // ── Multi-head with various window sizes ───────────────────────────

    #[test]
    fn test_mh_various_windows() {
        let nh = 2;
        let seq = 4;
        let hd = 4;
        let total = nh * seq * hd;
        let q = vec![1.0; total];
        let k = vec![1.0; total];
        let v = vec![1.0; total];
        for ws in [1, 2, 3, 4, 10] {
            let out = multi_head_sliding_window_neon(&q, &k, &v, seq, nh, hd, ws);
            assert_eq!(out.len(), total, "ws={ws}");
            for &x in &out {
                assert!(approx_eq(x, 1.0, 1e-3), "ws={ws} got {x}");
            }
        }
    }

    // ── Additional Q*K^T edge cases ────────────────────────────────────

    #[test]
    fn test_qk_two_tokens_window1() {
        let q = vec![1.0, 0.0, 0.0, 1.0]; // 2 tokens, hd=2
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let scores = sliding_window_qk_neon(&q, &k, 2, 2, 1, 1.0);
        // window=1: only diagonal valid
        assert!(scores[0 * 2 + 0].is_finite()); // (0,0)
        assert_eq!(scores[0 * 2 + 1], f32::NEG_INFINITY); // (0,1)
        assert_eq!(scores[1 * 2 + 0], f32::NEG_INFINITY); // (1,0)
        assert!(scores[1 * 2 + 1].is_finite()); // (1,1)
    }

    #[test]
    fn test_qk_score_values_correct() {
        // hd=2: q0=[1,2], q1=[3,4], k0=[5,6], k1=[7,8]
        let q = vec![1.0, 2.0, 3.0, 4.0];
        let k = vec![5.0, 6.0, 7.0, 8.0];
        let scores = sliding_window_qk_neon(&q, &k, 2, 2, 10, 1.0);
        // (0,0): dot([1,2],[5,6]) = 17
        assert!(approx_eq(scores[0], 17.0, 1e-4));
        // (1,0): dot([3,4],[5,6]) = 39
        assert!(approx_eq(scores[1 * 2 + 0], 39.0, 1e-4));
        // (1,1): dot([3,4],[7,8]) = 53
        assert!(approx_eq(scores[1 * 2 + 1], 53.0, 1e-4));
    }

    // ── Mask additional tests ──────────────────────────────────────────

    #[test]
    fn test_mask_5x5_window3() {
        let mask = build_sliding_window_mask(5, 3);
        // Row 4: window [2,3,4] → j ∈ {2,3,4}
        assert!(!mask[4 * 5 + 0]);
        assert!(!mask[4 * 5 + 1]);
        assert!(mask[4 * 5 + 2]);
        assert!(mask[4 * 5 + 3]);
        assert!(mask[4 * 5 + 4]);
    }

    #[test]
    fn test_mask_large_seq() {
        let seq = 64;
        let ws = 8;
        let mask = build_sliding_window_mask(seq, ws);
        assert_eq!(mask.len(), seq * seq);
        // Spot-check last row
        let last = seq - 1;
        let start = last - ws + 1;
        for j in 0..start {
            assert!(!mask[last * seq + j]);
        }
        for j in start..=last {
            assert!(mask[last * seq + j]);
        }
    }

    // ── Additional attention edge cases ────────────────────────────────

    #[test]
    fn test_attn_different_q_k_values() {
        // Q and K have different values; verify no crash and finite output.
        let q = vec![0.5, -0.3, 0.8, 0.1, -0.2, 0.6, 0.9, -0.4]; // 2×4
        let k = vec![0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4];
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let out = sliding_window_attention_neon(&q, &k, &v, 2, 4, 2);
        assert_eq!(out.len(), 8);
        for &x in &out {
            assert!(!x.is_nan());
            assert!(!x.is_infinite());
        }
    }

    #[test]
    fn test_attn_sequential_v_accumulation() {
        // 4 tokens, hd=1, window=4 (full causal), Q=K=ones
        // Row i: uniform weights 1/(i+1) over v[0..=i]
        let q = vec![1.0; 4];
        let k = vec![1.0; 4];
        let v = vec![0.0, 4.0, 8.0, 12.0];
        let out = sliding_window_attention_neon(&q, &k, &v, 4, 1, 4);
        // row 0: avg(0) = 0
        assert!(approx_eq(out[0], 0.0, 1e-3));
        // row 1: avg(0,4) = 2
        assert!(approx_eq(out[1], 2.0, 1e-3));
        // row 2: avg(0,4,8) = 4
        assert!(approx_eq(out[2], 4.0, 1e-3));
        // row 3: avg(0,4,8,12) = 6
        assert!(approx_eq(out[3], 6.0, 1e-3));
    }

    // ── Scalar weighted acc test ───────────────────────────────────────

    #[test]
    fn test_scalar_weighted_acc() {
        let mut out = vec![1.0, 2.0, 3.0];
        scalar_weighted_acc(&mut out, &[10.0, 20.0, 30.0], 0.5);
        assert!(approx_eq(out[0], 6.0, 1e-6));
        assert!(approx_eq(out[1], 12.0, 1e-6));
        assert!(approx_eq(out[2], 18.0, 1e-6));
    }

    // ── Extra coverage ─────────────────────────────────────────────────

    #[test]
    fn test_mask_single_element() {
        let mask = build_sliding_window_mask(1, 100);
        assert_eq!(mask, vec![true]);
    }

    #[test]
    fn test_attn_two_tokens_full_window_manual() {
        // 2 tokens, hd=1, window=2 (full causal)
        // Q=[1,1], K=[1,1], V=[0,10]
        // row 0: softmax([1]) → [1] → out=0
        // row 1: scores=[1,1] → softmax → [0.5,0.5] → out=5
        let out = sliding_window_attention_neon(&[1.0, 1.0], &[1.0, 1.0], &[0.0, 10.0], 2, 1, 2);
        assert!(approx_eq(out[0], 0.0, 1e-3));
        assert!(approx_eq(out[1], 5.0, 1e-3));
    }

    #[test]
    fn test_mh_determinism_repeat() {
        let nh = 3;
        let seq = 5;
        let hd = 4;
        let total = nh * seq * hd;
        let q: Vec<f32> = (0..total).map(|i| (i as f32 * 0.01).sin()).collect();
        let k: Vec<f32> = (0..total).map(|i| (i as f32 * 0.02).cos()).collect();
        let v: Vec<f32> = (0..total).map(|i| (i as f32 * 0.03).sin()).collect();
        let results: Vec<_> =
            (0..3).map(|_| multi_head_sliding_window_neon(&q, &k, &v, seq, nh, hd, 3)).collect();
        assert_eq!(results[0], results[1]);
        assert_eq!(results[1], results[2]);
    }

    #[test]
    fn test_qk_head_dim_large_non_aligned() {
        // head_dim = 17 (not a multiple of 4)
        let hd = 17;
        let seq = 3;
        let q: Vec<f32> = (0..seq * hd).map(|i| (i as f32) * 0.1).collect();
        let k: Vec<f32> = (0..seq * hd).map(|i| 1.0 - (i as f32) * 0.05).collect();
        let scores = sliding_window_qk_neon(&q, &k, seq, hd, 2, 0.5);
        // Verify diagonal entries match scalar dot product
        for i in 0..seq {
            let q_row = &q[i * hd..(i + 1) * hd];
            let k_row = &k[i * hd..(i + 1) * hd];
            let expected = scalar_dot(q_row, k_row) * 0.5;
            assert!(
                approx_eq(scores[i * seq + i], expected, 1e-3),
                "diag[{i}]: {} vs {expected}",
                scores[i * seq + i]
            );
        }
    }
}
