//! NEON-optimized pooling v2 operations for Apple Silicon (aarch64).
//!
//! Provides six pooling operations: mean pool, 1D max pool, 1D average pool,
//! weighted mean pool, last-token pool, and CLS-token pool.
//! Each has a NEON fast path and a scalar fallback, with a public dispatcher
//! that selects at runtime via `is_aarch64_feature_detected!`.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ---------------------------------------------------------------------------
// 1. Mean pooling across the sequence dimension
// ---------------------------------------------------------------------------

/// NEON-accelerated mean pooling: averages `seq_len` rows of width `dim`.
///
/// # Safety
///
/// Caller must ensure the target supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_mean_pool_f32(input: &[f32], seq_len: usize, dim: usize, output: &mut [f32]) {
    if seq_len == 0 {
        output[..dim].fill(0.0);
        return;
    }
    let inv_seq = 1.0 / seq_len as f32;
    let inv_v = unsafe { vdupq_n_f32(inv_seq) };
    let chunks = dim / 4;
    let remainder = dim % 4;

    // Zero accumulator
    for c in 0..chunks {
        let mut acc = unsafe { vdupq_n_f32(0.0) };
        for s in 0..seq_len {
            let v = unsafe { vld1q_f32(input.as_ptr().add(s * dim + c * 4)) };
            acc = vaddq_f32(acc, v);
        }
        let result = vmulq_f32(acc, inv_v);
        unsafe { vst1q_f32(output.as_mut_ptr().add(c * 4), result) };
    }

    for r in 0..remainder {
        let idx = chunks * 4 + r;
        let mut sum = 0.0_f32;
        for s in 0..seq_len {
            sum += input[s * dim + idx];
        }
        output[idx] = sum * inv_seq;
    }
}

fn scalar_mean_pool_f32(input: &[f32], seq_len: usize, dim: usize, output: &mut [f32]) {
    if seq_len == 0 {
        output[..dim].fill(0.0);
        return;
    }
    let inv_seq = 1.0 / seq_len as f32;
    for d in 0..dim {
        let mut sum = 0.0_f32;
        for s in 0..seq_len {
            sum += input[s * dim + d];
        }
        output[d] = sum * inv_seq;
    }
}

/// Mean pooling across the sequence dimension.
///
/// `input` is `[seq_len, dim]` row-major, `output` must have length ≥ `dim`.
///
/// # Panics
///
/// Panics if `input.len() < seq_len * dim` or `output.len() < dim`.
pub fn mean_pool_f32(input: &[f32], seq_len: usize, dim: usize, output: &mut [f32]) {
    assert!(
        input.len() >= seq_len * dim,
        "input too small: need {}, got {}",
        seq_len * dim,
        input.len()
    );
    assert!(output.len() >= dim, "output too small: need {dim}, got {}", output.len());

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_mean_pool_f32(input, seq_len, dim, output);
            }
            return;
        }
    }
    scalar_mean_pool_f32(input, seq_len, dim, output);
}

// ---------------------------------------------------------------------------
// 2. 1D max pooling
// ---------------------------------------------------------------------------

/// NEON-accelerated 1D max pooling over `[seq_len, dim]` along the seq axis.
///
/// # Safety
///
/// Caller must ensure the target supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_max_pool_1d_f32(
    input: &[f32],
    seq_len: usize,
    dim: usize,
    kernel_size: usize,
    stride: usize,
    output: &mut [f32],
) {
    if seq_len == 0 || kernel_size > seq_len {
        return;
    }
    let out_len = (seq_len - kernel_size) / stride + 1;
    let chunks = dim / 4;
    let remainder = dim % 4;

    for o in 0..out_len {
        let base = o * stride;
        for c in 0..chunks {
            let col = c * 4;
            let mut acc = unsafe { vdupq_n_f32(f32::NEG_INFINITY) };
            for k in 0..kernel_size {
                let v = unsafe { vld1q_f32(input.as_ptr().add((base + k) * dim + col)) };
                acc = vmaxq_f32(acc, v);
            }
            unsafe { vst1q_f32(output.as_mut_ptr().add(o * dim + col), acc) };
        }
        for r in 0..remainder {
            let col = chunks * 4 + r;
            let mut mx = f32::NEG_INFINITY;
            for k in 0..kernel_size {
                let val = input[(base + k) * dim + col];
                if val > mx {
                    mx = val;
                }
            }
            output[o * dim + col] = mx;
        }
    }
}

fn scalar_max_pool_1d_f32(
    input: &[f32],
    seq_len: usize,
    dim: usize,
    kernel_size: usize,
    stride: usize,
    output: &mut [f32],
) {
    if seq_len == 0 || kernel_size > seq_len {
        return;
    }
    let out_len = (seq_len - kernel_size) / stride + 1;
    for o in 0..out_len {
        let base = o * stride;
        for d in 0..dim {
            let mut mx = f32::NEG_INFINITY;
            for k in 0..kernel_size {
                let val = input[(base + k) * dim + d];
                if val > mx {
                    mx = val;
                }
            }
            output[o * dim + d] = mx;
        }
    }
}

/// 1D max pooling along the sequence dimension with `vmaxq_f32`.
///
/// `input` is `[seq_len, dim]`, `output` must have length ≥ `out_len * dim`
/// where `out_len = (seq_len - kernel_size) / stride + 1`.
///
/// # Panics
///
/// Panics if `stride` is zero, or slices are too small.
pub fn max_pool_1d_f32(
    input: &[f32],
    seq_len: usize,
    dim: usize,
    kernel_size: usize,
    stride: usize,
    output: &mut [f32],
) {
    assert!(stride > 0, "stride must be > 0");
    assert!(
        input.len() >= seq_len * dim,
        "input too small: need {}, got {}",
        seq_len * dim,
        input.len()
    );
    if seq_len == 0 || kernel_size > seq_len || kernel_size == 0 {
        return;
    }
    let out_len = (seq_len - kernel_size) / stride + 1;
    assert!(
        output.len() >= out_len * dim,
        "output too small: need {}, got {}",
        out_len * dim,
        output.len()
    );

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_max_pool_1d_f32(input, seq_len, dim, kernel_size, stride, output);
            }
            return;
        }
    }
    scalar_max_pool_1d_f32(input, seq_len, dim, kernel_size, stride, output);
}

// ---------------------------------------------------------------------------
// 3. 1D average pooling
// ---------------------------------------------------------------------------

/// NEON-accelerated 1D average pooling over `[seq_len, dim]`.
///
/// # Safety
///
/// Caller must ensure the target supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_avg_pool_1d_f32(
    input: &[f32],
    seq_len: usize,
    dim: usize,
    kernel_size: usize,
    stride: usize,
    output: &mut [f32],
) {
    if seq_len == 0 || kernel_size > seq_len {
        return;
    }
    let out_len = (seq_len - kernel_size) / stride + 1;
    let inv_k = 1.0 / kernel_size as f32;
    let inv_v = unsafe { vdupq_n_f32(inv_k) };
    let chunks = dim / 4;
    let remainder = dim % 4;

    for o in 0..out_len {
        let base = o * stride;
        for c in 0..chunks {
            let col = c * 4;
            let mut acc = unsafe { vdupq_n_f32(0.0) };
            for k in 0..kernel_size {
                let v = unsafe { vld1q_f32(input.as_ptr().add((base + k) * dim + col)) };
                acc = vaddq_f32(acc, v);
            }
            let result = vmulq_f32(acc, inv_v);
            unsafe { vst1q_f32(output.as_mut_ptr().add(o * dim + col), result) };
        }
        for r in 0..remainder {
            let col = chunks * 4 + r;
            let mut sum = 0.0_f32;
            for k in 0..kernel_size {
                sum += input[(base + k) * dim + col];
            }
            output[o * dim + col] = sum * inv_k;
        }
    }
}

fn scalar_avg_pool_1d_f32(
    input: &[f32],
    seq_len: usize,
    dim: usize,
    kernel_size: usize,
    stride: usize,
    output: &mut [f32],
) {
    if seq_len == 0 || kernel_size > seq_len {
        return;
    }
    let out_len = (seq_len - kernel_size) / stride + 1;
    let inv_k = 1.0 / kernel_size as f32;
    for o in 0..out_len {
        let base = o * stride;
        for d in 0..dim {
            let mut sum = 0.0_f32;
            for k in 0..kernel_size {
                sum += input[(base + k) * dim + d];
            }
            output[o * dim + d] = sum * inv_k;
        }
    }
}

/// 1D average pooling along the sequence dimension with `vaddq_f32`.
///
/// `input` is `[seq_len, dim]`, `output` must have length ≥ `out_len * dim`.
///
/// # Panics
///
/// Panics if `stride` is zero, or slices are too small.
pub fn avg_pool_1d_f32(
    input: &[f32],
    seq_len: usize,
    dim: usize,
    kernel_size: usize,
    stride: usize,
    output: &mut [f32],
) {
    assert!(stride > 0, "stride must be > 0");
    assert!(
        input.len() >= seq_len * dim,
        "input too small: need {}, got {}",
        seq_len * dim,
        input.len()
    );
    if seq_len == 0 || kernel_size > seq_len || kernel_size == 0 {
        return;
    }
    let out_len = (seq_len - kernel_size) / stride + 1;
    assert!(
        output.len() >= out_len * dim,
        "output too small: need {}, got {}",
        out_len * dim,
        output.len()
    );

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_avg_pool_1d_f32(input, seq_len, dim, kernel_size, stride, output);
            }
            return;
        }
    }
    scalar_avg_pool_1d_f32(input, seq_len, dim, kernel_size, stride, output);
}

// ---------------------------------------------------------------------------
// 4. Weighted mean pooling (attention-weighted)
// ---------------------------------------------------------------------------

/// NEON-accelerated weighted mean pooling using `vfmaq_f32`.
///
/// # Safety
///
/// Caller must ensure the target supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_weighted_mean_pool_f32(
    input: &[f32],
    weights: &[f32],
    seq_len: usize,
    dim: usize,
    output: &mut [f32],
) {
    if seq_len == 0 {
        output[..dim].fill(0.0);
        return;
    }

    // Compute weight sum for normalization
    let mut weight_sum = 0.0_f32;
    for s in 0..seq_len {
        weight_sum += weights[s];
    }
    if weight_sum == 0.0 {
        output[..dim].fill(0.0);
        return;
    }
    let inv_wsum = 1.0 / weight_sum;
    let inv_v = unsafe { vdupq_n_f32(inv_wsum) };
    let chunks = dim / 4;
    let remainder = dim % 4;

    for c in 0..chunks {
        let col = c * 4;
        let mut acc = unsafe { vdupq_n_f32(0.0) };
        for s in 0..seq_len {
            let w = unsafe { vdupq_n_f32(weights[s]) };
            let v = unsafe { vld1q_f32(input.as_ptr().add(s * dim + col)) };
            acc = vfmaq_f32(acc, v, w);
        }
        let result = vmulq_f32(acc, inv_v);
        unsafe { vst1q_f32(output.as_mut_ptr().add(col), result) };
    }

    for r in 0..remainder {
        let col = chunks * 4 + r;
        let mut sum = 0.0_f32;
        for s in 0..seq_len {
            sum += input[s * dim + col] * weights[s];
        }
        output[col] = sum * inv_wsum;
    }
}

fn scalar_weighted_mean_pool_f32(
    input: &[f32],
    weights: &[f32],
    seq_len: usize,
    dim: usize,
    output: &mut [f32],
) {
    if seq_len == 0 {
        output[..dim].fill(0.0);
        return;
    }
    let weight_sum: f32 = weights[..seq_len].iter().sum();
    if weight_sum == 0.0 {
        output[..dim].fill(0.0);
        return;
    }
    let inv_wsum = 1.0 / weight_sum;
    for d in 0..dim {
        let mut sum = 0.0_f32;
        for s in 0..seq_len {
            sum += input[s * dim + d] * weights[s];
        }
        output[d] = sum * inv_wsum;
    }
}

/// Attention-weighted mean pooling using `vfmaq_f32`.
///
/// `input` is `[seq_len, dim]`, `weights` has length ≥ `seq_len`,
/// `output` has length ≥ `dim`.
///
/// # Panics
///
/// Panics if slices are too small.
pub fn weighted_mean_pool_f32(
    input: &[f32],
    weights: &[f32],
    seq_len: usize,
    dim: usize,
    output: &mut [f32],
) {
    assert!(
        input.len() >= seq_len * dim,
        "input too small: need {}, got {}",
        seq_len * dim,
        input.len()
    );
    assert!(
        weights.len() >= seq_len,
        "weights too small: need {seq_len}, got {}",
        weights.len()
    );
    assert!(output.len() >= dim, "output too small: need {dim}, got {}", output.len());

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_weighted_mean_pool_f32(input, weights, seq_len, dim, output);
            }
            return;
        }
    }
    scalar_weighted_mean_pool_f32(input, weights, seq_len, dim, output);
}

// ---------------------------------------------------------------------------
// 5. Last-token pooling
// ---------------------------------------------------------------------------

/// NEON-accelerated last-token extraction (fast memcpy with NEON loads/stores).
///
/// # Safety
///
/// Caller must ensure the target supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_last_token_pool_f32(
    input: &[f32],
    seq_len: usize,
    dim: usize,
    output: &mut [f32],
) {
    if seq_len == 0 {
        output[..dim].fill(0.0);
        return;
    }
    let last_row = (seq_len - 1) * dim;
    let chunks = dim / 4;
    let remainder = dim % 4;

    for c in 0..chunks {
        let v = unsafe { vld1q_f32(input.as_ptr().add(last_row + c * 4)) };
        unsafe { vst1q_f32(output.as_mut_ptr().add(c * 4), v) };
    }
    for r in 0..remainder {
        output[chunks * 4 + r] = input[last_row + chunks * 4 + r];
    }
}

fn scalar_last_token_pool_f32(
    input: &[f32],
    seq_len: usize,
    dim: usize,
    output: &mut [f32],
) {
    if seq_len == 0 {
        output[..dim].fill(0.0);
        return;
    }
    let last_row = (seq_len - 1) * dim;
    output[..dim].copy_from_slice(&input[last_row..last_row + dim]);
}

/// Extract the last token from `[seq_len, dim]` input.
///
/// If `seq_len == 0`, output is zeroed.
///
/// # Panics
///
/// Panics if slices are too small.
pub fn last_token_pool_f32(input: &[f32], seq_len: usize, dim: usize, output: &mut [f32]) {
    assert!(
        input.len() >= seq_len * dim,
        "input too small: need {}, got {}",
        seq_len * dim,
        input.len()
    );
    assert!(output.len() >= dim, "output too small: need {dim}, got {}", output.len());

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_last_token_pool_f32(input, seq_len, dim, output);
            }
            return;
        }
    }
    scalar_last_token_pool_f32(input, seq_len, dim, output);
}

// ---------------------------------------------------------------------------
// 6. CLS-token pooling (first token)
// ---------------------------------------------------------------------------

/// NEON-accelerated CLS-token extraction (fast copy of first row).
///
/// # Safety
///
/// Caller must ensure the target supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_cls_token_pool_f32(input: &[f32], dim: usize, output: &mut [f32]) {
    let chunks = dim / 4;
    let remainder = dim % 4;

    for c in 0..chunks {
        let v = unsafe { vld1q_f32(input.as_ptr().add(c * 4)) };
        unsafe { vst1q_f32(output.as_mut_ptr().add(c * 4), v) };
    }
    for r in 0..remainder {
        output[chunks * 4 + r] = input[chunks * 4 + r];
    }
}

fn scalar_cls_token_pool_f32(input: &[f32], dim: usize, output: &mut [f32]) {
    output[..dim].copy_from_slice(&input[..dim]);
}

/// Extract the first [CLS] token from input of width `dim`.
///
/// # Panics
///
/// Panics if `input.len() < dim` or `output.len() < dim`.
pub fn cls_token_pool_f32(input: &[f32], dim: usize, output: &mut [f32]) {
    assert!(input.len() >= dim, "input too small: need {dim}, got {}", input.len());
    assert!(output.len() >= dim, "output too small: need {dim}, got {}", output.len());

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                neon_cls_token_pool_f32(input, dim, output);
            }
            return;
        }
    }
    scalar_cls_token_pool_f32(input, dim, output);
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers -------------------------------------------------------------

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() <= tol,
                "mismatch at index {i}: {x} vs {y} (tol={tol})"
            );
        }
    }

    /// Simple reference mean pool for verification.
    fn ref_mean_pool(input: &[f32], seq_len: usize, dim: usize) -> Vec<f32> {
        let mut out = vec![0.0_f32; dim];
        if seq_len == 0 {
            return out;
        }
        for d in 0..dim {
            let mut s = 0.0_f32;
            for t in 0..seq_len {
                s += input[t * dim + d];
            }
            out[d] = s / seq_len as f32;
        }
        out
    }

    // ========= mean_pool_f32 tests =========

    #[test]
    fn test_mean_pool_basic() {
        // 2 rows, dim=4: [[1,2,3,4],[5,6,7,8]] → mean = [3,4,5,6]
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut output = vec![0.0; 4];
        mean_pool_f32(&input, 2, 4, &mut output);
        approx_eq(&output, &[3.0, 4.0, 5.0, 6.0], 1e-6);
    }

    #[test]
    fn test_mean_pool_single_row() {
        let input = vec![10.0, 20.0, 30.0];
        let mut output = vec![0.0; 3];
        mean_pool_f32(&input, 1, 3, &mut output);
        approx_eq(&output, &[10.0, 20.0, 30.0], 1e-6);
    }

    #[test]
    fn test_mean_pool_seq_len_zero() {
        let input: Vec<f32> = vec![];
        let mut output = vec![999.0; 4];
        mean_pool_f32(&input, 0, 4, &mut output);
        approx_eq(&output[..4], &[0.0, 0.0, 0.0, 0.0], 0.0);
    }

    #[test]
    fn test_mean_pool_large_dim() {
        let dim = 33; // non-multiple of 4
        let seq_len = 3;
        let input: Vec<f32> = (0..seq_len * dim).map(|i| i as f32).collect();
        let mut output = vec![0.0; dim];
        mean_pool_f32(&input, seq_len, dim, &mut output);
        let expected = ref_mean_pool(&input, seq_len, dim);
        approx_eq(&output, &expected, 1e-5);
    }

    #[test]
    fn test_mean_pool_dim_1() {
        let input = vec![2.0, 4.0, 6.0];
        let mut output = vec![0.0; 1];
        mean_pool_f32(&input, 3, 1, &mut output);
        approx_eq(&output, &[4.0], 1e-6);
    }

    #[test]
    fn test_mean_pool_dim_8() {
        let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut output = vec![0.0; 8];
        mean_pool_f32(&input, 2, 8, &mut output);
        let expected = ref_mean_pool(&input, 2, 8);
        approx_eq(&output, &expected, 1e-6);
    }

    #[test]
    fn test_mean_pool_all_same() {
        let input = vec![5.0; 20];
        let mut output = vec![0.0; 4];
        mean_pool_f32(&input, 5, 4, &mut output);
        approx_eq(&output, &[5.0, 5.0, 5.0, 5.0], 1e-6);
    }

    #[test]
    fn test_mean_pool_negative_values() {
        let input = vec![-1.0, -2.0, -3.0, -4.0, 1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        mean_pool_f32(&input, 2, 4, &mut output);
        approx_eq(&output, &[0.0, 0.0, 0.0, 0.0], 1e-6);
    }

    #[test]
    fn test_mean_pool_many_rows() {
        let seq_len = 100;
        let dim = 16;
        let input: Vec<f32> = (0..seq_len * dim).map(|i| (i % 7) as f32).collect();
        let mut output = vec![0.0; dim];
        mean_pool_f32(&input, seq_len, dim, &mut output);
        let expected = ref_mean_pool(&input, seq_len, dim);
        approx_eq(&output, &expected, 1e-4);
    }

    // ========= max_pool_1d_f32 tests =========

    #[test]
    fn test_max_pool_basic() {
        // 4 rows, dim=2, kernel=2, stride=1 → 3 output rows
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut output = vec![0.0; 6];
        max_pool_1d_f32(&input, 4, 2, 2, 1, &mut output);
        // row 0-1 max: [3,4], row 1-2 max: [5,6], row 2-3 max: [7,8]
        approx_eq(&output, &[3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 1e-6);
    }

    #[test]
    fn test_max_pool_stride_2() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut output = vec![0.0; 4];
        max_pool_1d_f32(&input, 4, 2, 2, 2, &mut output);
        // row 0-1: [3,4], row 2-3: [7,8]
        approx_eq(&output, &[3.0, 4.0, 7.0, 8.0], 1e-6);
    }

    #[test]
    fn test_max_pool_kernel_eq_seq() {
        // 3 rows, dim=2: [[1,5],[3,2],[4,6]] → max per-col = [4,6]
        let input = vec![1.0, 5.0, 3.0, 2.0, 4.0, 6.0];
        let mut output = vec![0.0; 2];
        max_pool_1d_f32(&input, 3, 2, 3, 1, &mut output);
        approx_eq(&output, &[4.0, 6.0], 1e-6);
    }

    #[test]
    fn test_max_pool_kernel_gt_seq() {
        let input = vec![1.0, 2.0];
        let mut output = vec![0.0; 2];
        max_pool_1d_f32(&input, 1, 2, 5, 1, &mut output);
        // kernel_size > seq_len → no output
        approx_eq(&output, &[0.0, 0.0], 0.0);
    }

    #[test]
    fn test_max_pool_seq_zero() {
        let input: Vec<f32> = vec![];
        let mut output = vec![0.0; 4];
        max_pool_1d_f32(&input, 0, 2, 1, 1, &mut output);
        // seq_len=0 → no output written
    }

    #[test]
    fn test_max_pool_negative() {
        // 3 rows, dim=2, kernel=2, stride=1 → 2 output rows
        let input = vec![-5.0, -3.0, -1.0, -4.0, -2.0, -6.0];
        let mut output = vec![0.0; 4];
        max_pool_1d_f32(&input, 3, 2, 2, 1, &mut output);
        // row 0-1: max([-5,-3],[-1,-4])=[-1,-3], row 1-2: max([-1,-4],[-2,-6])=[-1,-4]
        approx_eq(&output, &[-1.0, -3.0, -1.0, -4.0], 1e-6);
    }

    #[test]
    fn test_max_pool_dim_5() {
        // non-aligned dim
        let dim = 5;
        let seq_len = 3;
        let input: Vec<f32> = (0..seq_len * dim).map(|i| i as f32).collect();
        let mut output = vec![0.0; 2 * dim]; // kernel=2, stride=1 → 2 outputs
        max_pool_1d_f32(&input, seq_len, dim, 2, 1, &mut output);
        // row 0-1: max(0..5, 5..10) = [5,6,7,8,9]
        // row 1-2: max(5..10, 10..15) = [10,11,12,13,14]
        approx_eq(
            &output,
            &[5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
            1e-6,
        );
    }

    #[test]
    fn test_max_pool_kernel_1() {
        // kernel=1 is identity
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        max_pool_1d_f32(&input, 2, 2, 1, 1, &mut output);
        approx_eq(&output, &input, 1e-6);
    }

    // ========= avg_pool_1d_f32 tests =========

    #[test]
    fn test_avg_pool_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut output = vec![0.0; 6];
        avg_pool_1d_f32(&input, 4, 2, 2, 1, &mut output);
        // row 0-1 avg: [2,3], row 1-2 avg: [4,5], row 2-3 avg: [6,7]
        approx_eq(&output, &[2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 1e-6);
    }

    #[test]
    fn test_avg_pool_stride_2() {
        let input = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
        let mut output = vec![0.0; 4];
        avg_pool_1d_f32(&input, 4, 2, 2, 2, &mut output);
        // row 0-1: [4,6], row 2-3: [12,14]
        approx_eq(&output, &[4.0, 6.0, 12.0, 14.0], 1e-6);
    }

    #[test]
    fn test_avg_pool_kernel_eq_seq() {
        let input = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0];
        let mut output = vec![0.0; 2];
        avg_pool_1d_f32(&input, 3, 2, 3, 1, &mut output);
        approx_eq(&output, &[6.0, 8.0], 1e-6);
    }

    #[test]
    fn test_avg_pool_kernel_gt_seq() {
        let input = vec![1.0, 2.0];
        let mut output = vec![0.0; 2];
        avg_pool_1d_f32(&input, 1, 2, 5, 1, &mut output);
        approx_eq(&output, &[0.0, 0.0], 0.0);
    }

    #[test]
    fn test_avg_pool_seq_zero() {
        let input: Vec<f32> = vec![];
        let mut output = vec![0.0; 4];
        avg_pool_1d_f32(&input, 0, 2, 1, 1, &mut output);
    }

    #[test]
    fn test_avg_pool_dim_7() {
        let dim = 7;
        let seq_len = 4;
        let input: Vec<f32> = (0..seq_len * dim).map(|i| i as f32).collect();
        let mut output = vec![0.0; 3 * dim]; // kernel=2, stride=1 → 3 outputs
        avg_pool_1d_f32(&input, seq_len, dim, 2, 1, &mut output);
        // Verify first output row: avg of row 0 and row 1
        for d in 0..dim {
            let expected = (input[d] + input[dim + d]) / 2.0;
            assert!((output[d] - expected).abs() < 1e-5, "mismatch at d={d}");
        }
    }

    #[test]
    fn test_avg_pool_kernel_1() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        avg_pool_1d_f32(&input, 2, 2, 1, 1, &mut output);
        approx_eq(&output, &input, 1e-6);
    }

    #[test]
    fn test_avg_pool_large() {
        let dim = 16;
        let seq_len = 10;
        let kernel_size = 3;
        let stride = 2;
        let out_len = (seq_len - kernel_size) / stride + 1;
        let input: Vec<f32> = (0..seq_len * dim).map(|i| (i as f32) * 0.1).collect();
        let mut output = vec![0.0; out_len * dim];
        avg_pool_1d_f32(&input, seq_len, dim, kernel_size, stride, &mut output);
        // Verify first element
        let expected = (input[0] + input[dim] + input[2 * dim]) / 3.0;
        assert!((output[0] - expected).abs() < 1e-5);
    }

    // ========= weighted_mean_pool_f32 tests =========

    #[test]
    fn test_weighted_pool_basic() {
        // 2 rows dim=4, weights=[1,1] → same as mean pool
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let weights = vec![1.0, 1.0];
        let mut output = vec![0.0; 4];
        weighted_mean_pool_f32(&input, &weights, 2, 4, &mut output);
        approx_eq(&output, &[3.0, 4.0, 5.0, 6.0], 1e-6);
    }

    #[test]
    fn test_weighted_pool_unequal_weights() {
        // weights=[1,3] → output = (1*row0 + 3*row1) / 4
        let input = vec![0.0, 0.0, 4.0, 4.0];
        let weights = vec![1.0, 3.0];
        let mut output = vec![0.0; 2];
        weighted_mean_pool_f32(&input, &weights, 2, 2, &mut output);
        // (1*0 + 3*4)/4 = 3.0
        approx_eq(&output, &[3.0, 3.0], 1e-6);
    }

    #[test]
    fn test_weighted_pool_zero_weights() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weights = vec![0.0, 0.0];
        let mut output = vec![999.0; 2];
        weighted_mean_pool_f32(&input, &weights, 2, 2, &mut output);
        approx_eq(&output, &[0.0, 0.0], 0.0);
    }

    #[test]
    fn test_weighted_pool_seq_zero() {
        let input: Vec<f32> = vec![];
        let weights: Vec<f32> = vec![];
        let mut output = vec![999.0; 3];
        weighted_mean_pool_f32(&input, &weights, 0, 3, &mut output);
        approx_eq(&output[..3], &[0.0, 0.0, 0.0], 0.0);
    }

    #[test]
    fn test_weighted_pool_single_row() {
        let input = vec![10.0, 20.0, 30.0, 40.0];
        let weights = vec![5.0];
        let mut output = vec![0.0; 4];
        weighted_mean_pool_f32(&input, &weights, 1, 4, &mut output);
        approx_eq(&output, &[10.0, 20.0, 30.0, 40.0], 1e-6);
    }

    #[test]
    fn test_weighted_pool_dim_5() {
        let dim = 5;
        let input: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let weights = vec![0.5, 1.5];
        let mut output = vec![0.0; dim];
        weighted_mean_pool_f32(&input, &weights, 2, dim, &mut output);
        // expected: (0.5*row0 + 1.5*row1) / 2.0
        let expected: Vec<f32> = (0..dim)
            .map(|d| (0.5 * input[d] + 1.5 * input[dim + d]) / 2.0)
            .collect();
        approx_eq(&output, &expected, 1e-5);
    }

    #[test]
    fn test_weighted_pool_large() {
        let dim = 17;
        let seq_len = 8;
        let input: Vec<f32> = (0..seq_len * dim).map(|i| (i as f32) * 0.1).collect();
        let weights: Vec<f32> = (0..seq_len).map(|i| (i + 1) as f32).collect();
        let mut output = vec![0.0; dim];
        weighted_mean_pool_f32(&input, &weights, seq_len, dim, &mut output);
        let weight_sum: f32 = weights.iter().sum();
        for d in 0..dim {
            let mut expected = 0.0_f32;
            for s in 0..seq_len {
                expected += input[s * dim + d] * weights[s];
            }
            expected /= weight_sum;
            assert!(
                (output[d] - expected).abs() < 1e-4,
                "mismatch at d={d}: {} vs {}",
                output[d],
                expected,
            );
        }
    }

    #[test]
    fn test_weighted_pool_negative_weights() {
        // Negative weights are valid — just weighted sum / total
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weights = vec![-1.0, 3.0]; // sum=2
        let mut output = vec![0.0; 2];
        weighted_mean_pool_f32(&input, &weights, 2, 2, &mut output);
        // (-1*1 + 3*3)/2 = 4, (-1*2 + 3*4)/2 = 5
        approx_eq(&output, &[4.0, 5.0], 1e-6);
    }

    // ========= last_token_pool_f32 tests =========

    #[test]
    fn test_last_token_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut output = vec![0.0; 4];
        last_token_pool_f32(&input, 2, 4, &mut output);
        approx_eq(&output, &[5.0, 6.0, 7.0, 8.0], 0.0);
    }

    #[test]
    fn test_last_token_single_row() {
        let input = vec![42.0, 43.0];
        let mut output = vec![0.0; 2];
        last_token_pool_f32(&input, 1, 2, &mut output);
        approx_eq(&output, &[42.0, 43.0], 0.0);
    }

    #[test]
    fn test_last_token_seq_zero() {
        let input: Vec<f32> = vec![];
        let mut output = vec![999.0; 3];
        last_token_pool_f32(&input, 0, 3, &mut output);
        approx_eq(&output[..3], &[0.0, 0.0, 0.0], 0.0);
    }

    #[test]
    fn test_last_token_dim_5() {
        let input: Vec<f32> = (0..15).map(|i| i as f32).collect();
        let mut output = vec![0.0; 5];
        last_token_pool_f32(&input, 3, 5, &mut output);
        approx_eq(&output, &[10.0, 11.0, 12.0, 13.0, 14.0], 0.0);
    }

    #[test]
    fn test_last_token_dim_16() {
        let dim = 16;
        let seq_len = 4;
        let input: Vec<f32> = (0..seq_len * dim).map(|i| i as f32).collect();
        let mut output = vec![0.0; dim];
        last_token_pool_f32(&input, seq_len, dim, &mut output);
        let expected: Vec<f32> = ((seq_len - 1) * dim..seq_len * dim)
            .map(|i| i as f32)
            .collect();
        approx_eq(&output, &expected, 0.0);
    }

    #[test]
    fn test_last_token_many_rows() {
        let dim = 8;
        let seq_len = 50;
        let input: Vec<f32> = (0..seq_len * dim).map(|i| i as f32).collect();
        let mut output = vec![0.0; dim];
        last_token_pool_f32(&input, seq_len, dim, &mut output);
        for d in 0..dim {
            assert_eq!(output[d], ((seq_len - 1) * dim + d) as f32);
        }
    }

    // ========= cls_token_pool_f32 tests =========

    #[test]
    fn test_cls_token_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut output = vec![0.0; 4];
        cls_token_pool_f32(&input, 4, &mut output);
        approx_eq(&output, &[1.0, 2.0, 3.0, 4.0], 0.0);
    }

    #[test]
    fn test_cls_token_dim_1() {
        let input = vec![42.0, 99.0];
        let mut output = vec![0.0; 1];
        cls_token_pool_f32(&input, 1, &mut output);
        approx_eq(&output, &[42.0], 0.0);
    }

    #[test]
    fn test_cls_token_dim_5() {
        let input: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let mut output = vec![0.0; 5];
        cls_token_pool_f32(&input, 5, &mut output);
        approx_eq(&output, &[0.0, 1.0, 2.0, 3.0, 4.0], 0.0);
    }

    #[test]
    fn test_cls_token_dim_16() {
        let dim = 16;
        let input: Vec<f32> = (0..dim * 3).map(|i| i as f32).collect();
        let mut output = vec![0.0; dim];
        cls_token_pool_f32(&input, dim, &mut output);
        let expected: Vec<f32> = (0..dim).map(|i| i as f32).collect();
        approx_eq(&output, &expected, 0.0);
    }

    #[test]
    fn test_cls_token_negative() {
        let input = vec![-1.0, -2.0, -3.0, -4.0, 5.0, 6.0, 7.0, 8.0];
        let mut output = vec![0.0; 4];
        cls_token_pool_f32(&input, 4, &mut output);
        approx_eq(&output, &[-1.0, -2.0, -3.0, -4.0], 0.0);
    }

    // ========= Cross-check: mean_pool vs avg_pool with kernel=seq_len =========

    #[test]
    fn test_mean_pool_vs_avg_pool_full_kernel() {
        let seq_len = 5;
        let dim = 8;
        let input: Vec<f32> = (0..seq_len * dim).map(|i| i as f32).collect();
        let mut mean_out = vec![0.0; dim];
        let mut avg_out = vec![0.0; dim];
        mean_pool_f32(&input, seq_len, dim, &mut mean_out);
        avg_pool_1d_f32(&input, seq_len, dim, seq_len, 1, &mut avg_out);
        approx_eq(&mean_out, &avg_out, 1e-5);
    }

    // ========= Cross-check: weighted pool with uniform weights == mean pool =========

    #[test]
    fn test_weighted_uniform_eq_mean() {
        let seq_len = 4;
        let dim = 6;
        let input: Vec<f32> = (0..seq_len * dim).map(|i| i as f32).collect();
        let weights = vec![1.0; seq_len];
        let mut w_out = vec![0.0; dim];
        let mut m_out = vec![0.0; dim];
        weighted_mean_pool_f32(&input, &weights, seq_len, dim, &mut w_out);
        mean_pool_f32(&input, seq_len, dim, &mut m_out);
        approx_eq(&w_out, &m_out, 1e-5);
    }

    // ========= Scalar fallback verification =========

    #[test]
    fn test_scalar_mean_pool() {
        // 2 rows, dim=3: [[1,2,3],[4,5,6]] → mean=[2.5,3.5,4.5]
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut output = vec![0.0; 3];
        scalar_mean_pool_f32(&input, 2, 3, &mut output);
        approx_eq(&output, &[2.5, 3.5, 4.5], 1e-6);
    }

    #[test]
    fn test_scalar_max_pool() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut output = vec![0.0; 4];
        scalar_max_pool_1d_f32(&input, 3, 2, 2, 1, &mut output);
        approx_eq(&output, &[3.0, 4.0, 5.0, 6.0], 1e-6);
    }

    #[test]
    fn test_scalar_avg_pool() {
        let input = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0];
        let mut output = vec![0.0; 2];
        scalar_avg_pool_1d_f32(&input, 3, 2, 3, 1, &mut output);
        approx_eq(&output, &[6.0, 8.0], 1e-6);
    }

    #[test]
    fn test_scalar_weighted_pool() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weights = vec![2.0, 1.0];
        let mut output = vec![0.0; 2];
        scalar_weighted_mean_pool_f32(&input, &weights, 2, 2, &mut output);
        // (2*1+1*3)/3=5/3, (2*2+1*4)/3=8/3
        approx_eq(&output, &[5.0 / 3.0, 8.0 / 3.0], 1e-6);
    }

    #[test]
    fn test_scalar_last_token() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 2];
        scalar_last_token_pool_f32(&input, 2, 2, &mut output);
        approx_eq(&output, &[3.0, 4.0], 0.0);
    }

    #[test]
    fn test_scalar_cls_token() {
        let input = vec![10.0, 20.0, 30.0, 40.0];
        let mut output = vec![0.0; 2];
        scalar_cls_token_pool_f32(&input, 2, &mut output);
        approx_eq(&output, &[10.0, 20.0], 0.0);
    }

    // ========= Edge cases =========

    #[test]
    #[should_panic(expected = "stride must be > 0")]
    fn test_max_pool_stride_zero_panics() {
        let input = vec![1.0; 4];
        let mut output = vec![0.0; 4];
        max_pool_1d_f32(&input, 2, 2, 1, 0, &mut output);
    }

    #[test]
    #[should_panic(expected = "stride must be > 0")]
    fn test_avg_pool_stride_zero_panics() {
        let input = vec![1.0; 4];
        let mut output = vec![0.0; 4];
        avg_pool_1d_f32(&input, 2, 2, 1, 0, &mut output);
    }

    #[test]
    fn test_max_pool_kernel_zero() {
        // kernel_size=0 → early return, no output written
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        max_pool_1d_f32(&input, 2, 2, 0, 1, &mut output);
        approx_eq(&output, &[0.0, 0.0, 0.0, 0.0], 0.0);
    }

    #[test]
    fn test_avg_pool_kernel_zero() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];
        avg_pool_1d_f32(&input, 2, 2, 0, 1, &mut output);
        approx_eq(&output, &[0.0, 0.0, 0.0, 0.0], 0.0);
    }

    #[test]
    #[should_panic(expected = "input too small")]
    fn test_mean_pool_input_too_small() {
        let input = vec![1.0, 2.0];
        let mut output = vec![0.0; 4];
        mean_pool_f32(&input, 2, 4, &mut output);
    }

    #[test]
    #[should_panic(expected = "output too small")]
    fn test_mean_pool_output_too_small() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 1];
        mean_pool_f32(&input, 2, 2, &mut output);
    }

    #[test]
    #[should_panic(expected = "input too small")]
    fn test_cls_token_input_too_small() {
        let input = vec![1.0];
        let mut output = vec![0.0; 4];
        cls_token_pool_f32(&input, 4, &mut output);
    }

    #[test]
    fn test_last_token_vs_cls_single_row() {
        // With seq_len=1, last_token and cls_token should agree
        let input = vec![7.0, 8.0, 9.0, 10.0];
        let mut last_out = vec![0.0; 4];
        let mut cls_out = vec![0.0; 4];
        last_token_pool_f32(&input, 1, 4, &mut last_out);
        cls_token_pool_f32(&input, 4, &mut cls_out);
        approx_eq(&last_out, &cls_out, 0.0);
    }

    // ========= Additional coverage for dim=4 boundary =========

    #[test]
    fn test_mean_pool_dim_exactly_4() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut output = vec![0.0; 4];
        mean_pool_f32(&input, 3, 4, &mut output);
        approx_eq(&output, &[5.0, 6.0, 7.0, 8.0], 1e-6);
    }

    #[test]
    fn test_weighted_pool_dim_exactly_4() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let weights = vec![1.0, 1.0];
        let mut output = vec![0.0; 4];
        weighted_mean_pool_f32(&input, &weights, 2, 4, &mut output);
        approx_eq(&output, &[3.0, 4.0, 5.0, 6.0], 1e-6);
    }

    #[test]
    fn test_mean_pool_dim_3() {
        // All remainder, no NEON chunks
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut output = vec![0.0; 3];
        mean_pool_f32(&input, 2, 3, &mut output);
        approx_eq(&output, &[2.5, 3.5, 4.5], 1e-6);
    }
}
