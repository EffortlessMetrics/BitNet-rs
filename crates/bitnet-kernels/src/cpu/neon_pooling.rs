//! ARM NEON pooling operations for Apple Silicon.
//!
//! Provides NEON-optimized 1-D max, average, global average, global max,
//! and adaptive average pooling for `f32` slices.  Each public function
//! dispatches to NEON intrinsics on `aarch64` and falls back to a scalar
//! implementation on other architectures.

// ── NEON intrinsics import ─────────────────────────────────────────
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ===================================================================
// Helper: compute the number of output windows for a 1-D pooling op.
// ===================================================================

#[inline]
fn pool1d_output_len(input_len: usize, pool_size: usize, stride: usize) -> usize {
    if input_len < pool_size {
        return 0;
    }
    (input_len - pool_size) / stride + 1
}

// ===================================================================
// 1. max_pool1d_neon — 1-D max pooling
// ===================================================================

/// 1-D max pooling.
///
/// Slides a window of `pool_size` over `input[..input_len]` with the
/// given `stride`, writing per-window maxima into `output`.
///
/// On `aarch64` targets the hot loop uses NEON `vmaxq_f32`; on other
/// targets a scalar fallback is used.
///
/// # Panics
///
/// Panics if `pool_size` or `stride` is zero, or if `output` is too
/// small for the computed number of windows.
pub fn max_pool1d_neon(
    input: &[f32],
    input_len: usize,
    pool_size: usize,
    stride: usize,
    output: &mut [f32],
) {
    assert!(pool_size > 0, "pool_size must be > 0");
    assert!(stride > 0, "stride must be > 0");

    let len = input_len.min(input.len());
    let out_len = pool1d_output_len(len, pool_size, stride);
    if out_len == 0 {
        return;
    }
    assert!(output.len() >= out_len, "output too small: need {out_len}, got {}", output.len());

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is always available on aarch64.
        unsafe {
            max_pool1d_neon_inner(input, len, pool_size, stride, output);
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        max_pool1d_scalar(input, len, pool_size, stride, output);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn max_pool1d_neon_inner(
    input: &[f32],
    input_len: usize,
    pool_size: usize,
    stride: usize,
    output: &mut [f32],
) {
    let out_len = pool1d_output_len(input_len, pool_size, stride);
    let ptr = input.as_ptr();

    for idx in 0..out_len {
        let base = idx * stride;
        let chunks = pool_size / 4;
        let remainder = pool_size % 4;

        let mut acc = vdupq_n_f32(f32::NEG_INFINITY);
        for c in 0..chunks {
            let v = vld1q_f32(ptr.add(base + c * 4));
            acc = vmaxq_f32(acc, v);
        }

        let mut max_val = vmaxvq_f32(acc);
        for r in 0..remainder {
            let val = *ptr.add(base + chunks * 4 + r);
            if val > max_val {
                max_val = val;
            }
        }
        output[idx] = max_val;
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn max_pool1d_scalar(
    input: &[f32],
    input_len: usize,
    pool_size: usize,
    stride: usize,
    output: &mut [f32],
) {
    let out_len = pool1d_output_len(input_len, pool_size, stride);
    for idx in 0..out_len {
        let base = idx * stride;
        let mut max_val = f32::NEG_INFINITY;
        for k in 0..pool_size {
            let val = input[base + k];
            if val > max_val {
                max_val = val;
            }
        }
        output[idx] = max_val;
    }
}

// ===================================================================
// 2. avg_pool1d_neon — 1-D average pooling
// ===================================================================

/// 1-D average pooling.
///
/// Slides a window of `pool_size` over `input[..input_len]` with the
/// given `stride`, writing per-window means into `output`.
///
/// # Panics
///
/// Panics if `pool_size` or `stride` is zero, or if `output` is too
/// small.
pub fn avg_pool1d_neon(
    input: &[f32],
    input_len: usize,
    pool_size: usize,
    stride: usize,
    output: &mut [f32],
) {
    assert!(pool_size > 0, "pool_size must be > 0");
    assert!(stride > 0, "stride must be > 0");

    let len = input_len.min(input.len());
    let out_len = pool1d_output_len(len, pool_size, stride);
    if out_len == 0 {
        return;
    }
    assert!(output.len() >= out_len, "output too small: need {out_len}, got {}", output.len());

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            avg_pool1d_neon_inner(input, len, pool_size, stride, output);
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        avg_pool1d_scalar(input, len, pool_size, stride, output);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn avg_pool1d_neon_inner(
    input: &[f32],
    input_len: usize,
    pool_size: usize,
    stride: usize,
    output: &mut [f32],
) {
    let out_len = pool1d_output_len(input_len, pool_size, stride);
    let ptr = input.as_ptr();
    let inv_k = 1.0 / pool_size as f32;

    for idx in 0..out_len {
        let base = idx * stride;
        let chunks = pool_size / 4;
        let remainder = pool_size % 4;

        let mut acc = vdupq_n_f32(0.0);
        for c in 0..chunks {
            let v = vld1q_f32(ptr.add(base + c * 4));
            acc = vaddq_f32(acc, v);
        }

        let mut sum = vaddvq_f32(acc);
        for r in 0..remainder {
            sum += *ptr.add(base + chunks * 4 + r);
        }
        output[idx] = sum * inv_k;
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn avg_pool1d_scalar(
    input: &[f32],
    input_len: usize,
    pool_size: usize,
    stride: usize,
    output: &mut [f32],
) {
    let out_len = pool1d_output_len(input_len, pool_size, stride);
    let inv_k = 1.0 / pool_size as f32;
    for idx in 0..out_len {
        let base = idx * stride;
        let mut sum = 0.0f32;
        for k in 0..pool_size {
            sum += input[base + k];
        }
        output[idx] = sum * inv_k;
    }
}

// ===================================================================
// 3. global_avg_pool_neon — global average pooling per channel
// ===================================================================

/// Global average pooling per channel.
///
/// Given a flat buffer laid out as `[channels][spatial_size]`, computes
/// the mean of each spatial slice and writes one value per channel
/// into `output`.
///
/// # Panics
///
/// Panics if `channels` or `spatial_size` is zero, if `input.len()` is
/// less than `channels * spatial_size`, or if `output` is too small.
pub fn global_avg_pool_neon(
    input: &[f32],
    channels: usize,
    spatial_size: usize,
    output: &mut [f32],
) {
    assert!(channels > 0, "channels must be > 0");
    assert!(spatial_size > 0, "spatial_size must be > 0");
    assert!(
        input.len() >= channels * spatial_size,
        "input too short: need {}, got {}",
        channels * spatial_size,
        input.len()
    );
    assert!(output.len() >= channels, "output too small: need {channels}, got {}", output.len());

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            global_avg_pool_neon_inner(input, channels, spatial_size, output);
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        global_avg_pool_scalar(input, channels, spatial_size, output);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn global_avg_pool_neon_inner(
    input: &[f32],
    channels: usize,
    spatial_size: usize,
    output: &mut [f32],
) {
    let ptr = input.as_ptr();
    let inv_s = 1.0 / spatial_size as f32;

    for ch in 0..channels {
        let base = ch * spatial_size;
        let chunks = spatial_size / 4;
        let remainder = spatial_size % 4;

        let mut acc = vdupq_n_f32(0.0);
        for c in 0..chunks {
            let v = vld1q_f32(ptr.add(base + c * 4));
            acc = vaddq_f32(acc, v);
        }

        let mut sum = vaddvq_f32(acc);
        for r in 0..remainder {
            sum += *ptr.add(base + chunks * 4 + r);
        }
        output[ch] = sum * inv_s;
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn global_avg_pool_scalar(input: &[f32], channels: usize, spatial_size: usize, output: &mut [f32]) {
    let inv_s = 1.0 / spatial_size as f32;
    for ch in 0..channels {
        let base = ch * spatial_size;
        let mut sum = 0.0f32;
        for k in 0..spatial_size {
            sum += input[base + k];
        }
        output[ch] = sum * inv_s;
    }
}

// ===================================================================
// 4. global_max_pool_neon — global max pooling per channel
// ===================================================================

/// Global max pooling per channel.
///
/// Given a flat buffer laid out as `[channels][spatial_size]`, computes
/// the maximum of each spatial slice and writes one value per channel
/// into `output`.
///
/// # Panics
///
/// Panics if `channels` or `spatial_size` is zero, if `input.len()` is
/// less than `channels * spatial_size`, or if `output` is too small.
pub fn global_max_pool_neon(
    input: &[f32],
    channels: usize,
    spatial_size: usize,
    output: &mut [f32],
) {
    assert!(channels > 0, "channels must be > 0");
    assert!(spatial_size > 0, "spatial_size must be > 0");
    assert!(
        input.len() >= channels * spatial_size,
        "input too short: need {}, got {}",
        channels * spatial_size,
        input.len()
    );
    assert!(output.len() >= channels, "output too small: need {channels}, got {}", output.len());

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            global_max_pool_neon_inner(input, channels, spatial_size, output);
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        global_max_pool_scalar(input, channels, spatial_size, output);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn global_max_pool_neon_inner(
    input: &[f32],
    channels: usize,
    spatial_size: usize,
    output: &mut [f32],
) {
    let ptr = input.as_ptr();

    for ch in 0..channels {
        let base = ch * spatial_size;
        let chunks = spatial_size / 4;
        let remainder = spatial_size % 4;

        let mut acc = vdupq_n_f32(f32::NEG_INFINITY);
        for c in 0..chunks {
            let v = vld1q_f32(ptr.add(base + c * 4));
            acc = vmaxq_f32(acc, v);
        }

        let mut max_val = vmaxvq_f32(acc);
        for r in 0..remainder {
            let val = *ptr.add(base + chunks * 4 + r);
            if val > max_val {
                max_val = val;
            }
        }
        output[ch] = max_val;
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn global_max_pool_scalar(input: &[f32], channels: usize, spatial_size: usize, output: &mut [f32]) {
    for ch in 0..channels {
        let base = ch * spatial_size;
        let mut max_val = f32::NEG_INFINITY;
        for k in 0..spatial_size {
            let val = input[base + k];
            if val > max_val {
                max_val = val;
            }
        }
        output[ch] = max_val;
    }
}

// ===================================================================
// 5. adaptive_avg_pool1d_neon — adaptive average pooling
// ===================================================================

/// Adaptive 1-D average pooling.
///
/// Divides `input[..input_len]` into `output_len` bins (matching
/// PyTorch `AdaptiveAvgPool1d` bin boundaries) and writes the mean of
/// each bin into `output`.
///
/// # Panics
///
/// Panics if `output_len` is zero, if `output_len > input_len`, or if
/// `output` is too small.
pub fn adaptive_avg_pool1d_neon(
    input: &[f32],
    input_len: usize,
    output_len: usize,
    output: &mut [f32],
) {
    let len = input_len.min(input.len());
    assert!(output_len > 0, "output_len must be > 0");
    assert!(output_len <= len, "output_len ({output_len}) must be <= input length ({len})");
    assert!(
        output.len() >= output_len,
        "output too small: need {output_len}, got {}",
        output.len()
    );

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            adaptive_avg_pool1d_neon_inner(input, len, output_len, output);
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        adaptive_avg_pool1d_scalar(input, len, output_len, output);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn adaptive_avg_pool1d_neon_inner(
    input: &[f32],
    input_len: usize,
    output_len: usize,
    output: &mut [f32],
) {
    let ptr = input.as_ptr();

    for i in 0..output_len {
        let start = (i * input_len) / output_len;
        let end = ((i + 1) * input_len) / output_len;
        let bin_len = end - start;

        let chunks = bin_len / 4;
        let remainder = bin_len % 4;

        let mut acc = vdupq_n_f32(0.0);
        for c in 0..chunks {
            let v = vld1q_f32(ptr.add(start + c * 4));
            acc = vaddq_f32(acc, v);
        }

        let mut sum = vaddvq_f32(acc);
        for r in 0..remainder {
            sum += *ptr.add(start + chunks * 4 + r);
        }
        output[i] = sum / bin_len as f32;
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn adaptive_avg_pool1d_scalar(
    input: &[f32],
    input_len: usize,
    output_len: usize,
    output: &mut [f32],
) {
    for i in 0..output_len {
        let start = (i * input_len) / output_len;
        let end = ((i + 1) * input_len) / output_len;
        let bin_len = end - start;
        let mut sum = 0.0f32;
        for k in start..end {
            sum += input[k];
        }
        output[i] = sum / bin_len as f32;
    }
}

// ===================================================================
// Tests
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-6;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    fn assert_approx_slice(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!(approx_eq(*a, *e), "index {i}: {a} != {e} (diff={})", (a - e).abs());
        }
    }

    // ── max_pool1d_neon ────────────────────────────────────────────

    #[test]
    fn max_pool1d_basic() {
        let input: Vec<f32> = (0..8).map(|x| x as f32).collect();
        let mut out = vec![0.0f32; 3];
        max_pool1d_neon(&input, 8, 3, 2, &mut out);
        assert_eq!(out, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn max_pool1d_stride_1() {
        let input = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let mut out = vec![0.0f32; 6];
        max_pool1d_neon(&input, 8, 3, 1, &mut out);
        assert_eq!(out, vec![4.0, 4.0, 5.0, 9.0, 9.0, 9.0]);
    }

    #[test]
    fn max_pool1d_large_kernel() {
        let input = vec![1.0, 8.0, 3.0, 7.0, 2.0, 9.0, 0.0, 4.0, 6.0, 5.0];
        let mut out = vec![0.0f32; 2];
        max_pool1d_neon(&input, 10, 5, 3, &mut out);
        assert_eq!(out, vec![8.0, 9.0]);
    }

    #[test]
    fn max_pool1d_pool_equals_input() {
        let input = vec![5.0, 1.0, 3.0, 9.0];
        let mut out = vec![0.0f32; 1];
        max_pool1d_neon(&input, 4, 4, 1, &mut out);
        assert_eq!(out[0], 9.0);
    }

    #[test]
    fn max_pool1d_pool_larger_than_input() {
        let input = vec![1.0, 2.0];
        let mut out = vec![0.0f32; 1];
        max_pool1d_neon(&input, 2, 5, 1, &mut out);
        // out_len == 0 → no output written
        assert_eq!(out[0], 0.0);
    }

    #[test]
    fn max_pool1d_single_element() {
        let input = vec![42.0];
        let mut out = vec![0.0f32; 1];
        max_pool1d_neon(&input, 1, 1, 1, &mut out);
        assert_eq!(out[0], 42.0);
    }

    #[test]
    fn max_pool1d_stride_larger_than_pool() {
        let input: Vec<f32> = (0..10).map(|x| x as f32).collect();
        let mut out = vec![0.0f32; 3];
        max_pool1d_neon(&input, 10, 2, 3, &mut out);
        // windows: [0,1]=1, [3,4]=4, [6,7]=7
        assert_eq!(out, vec![1.0, 4.0, 7.0]);
    }

    #[test]
    fn max_pool1d_negative_values() {
        let input = vec![-5.0, -3.0, -8.0, -1.0, -4.0];
        let mut out = vec![0.0f32; 4]; // (5-2)/1+1=4
        max_pool1d_neon(&input, 5, 2, 1, &mut out);
        assert_eq!(out, vec![-3.0, -3.0, -1.0, -1.0]);
    }

    #[test]
    fn max_pool1d_all_same() {
        let input = vec![7.0; 8];
        let mut out = vec![0.0f32; 3]; // (8-4)/2+1=3
        max_pool1d_neon(&input, 8, 4, 2, &mut out);
        assert_eq!(out, vec![7.0, 7.0, 7.0]);
    }

    #[test]
    fn max_pool1d_kernel_4_exact_neon_chunk() {
        let input = vec![1.0, 4.0, 2.0, 3.0, 5.0, 0.0, 8.0, 6.0];
        let mut out = vec![0.0f32; 2];
        max_pool1d_neon(&input, 8, 4, 4, &mut out);
        assert_eq!(out, vec![4.0, 8.0]);
    }

    #[test]
    fn max_pool1d_kernel_5_neon_plus_remainder() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut out = vec![0.0f32; 1];
        max_pool1d_neon(&input, 5, 5, 1, &mut out);
        assert_eq!(out[0], 5.0);
    }

    #[test]
    fn max_pool1d_input_len_less_than_slice() {
        let input = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let mut out = vec![0.0f32; 2]; // (3-2)/1+1=2
        // input_len = 3 → only consider first 3 elements
        max_pool1d_neon(&input, 3, 2, 1, &mut out);
        assert_eq!(out, vec![20.0, 30.0]);
    }

    #[test]
    fn max_pool1d_large_input() {
        let input: Vec<f32> = (0..64).map(|x| x as f32).collect();
        let mut out = vec![0.0f32; 16];
        max_pool1d_neon(&input, 64, 4, 4, &mut out);
        for (i, v) in out.iter().enumerate() {
            assert_eq!(*v, (i * 4 + 3) as f32);
        }
    }

    #[test]
    fn max_pool1d_inf_values() {
        let input = vec![f32::NEG_INFINITY, 0.0, f32::INFINITY, 1.0];
        let mut out = vec![0.0f32; 1];
        max_pool1d_neon(&input, 4, 4, 1, &mut out);
        assert_eq!(out[0], f32::INFINITY);
    }

    #[test]
    #[should_panic(expected = "pool_size must be > 0")]
    fn max_pool1d_zero_pool_panics() {
        let input = vec![1.0];
        let mut out = vec![0.0f32; 1];
        max_pool1d_neon(&input, 1, 0, 1, &mut out);
    }

    #[test]
    #[should_panic(expected = "stride must be > 0")]
    fn max_pool1d_zero_stride_panics() {
        let input = vec![1.0];
        let mut out = vec![0.0f32; 1];
        max_pool1d_neon(&input, 1, 1, 0, &mut out);
    }

    #[test]
    #[should_panic(expected = "output too small")]
    fn max_pool1d_output_too_small_panics() {
        let input = vec![1.0; 8];
        let mut out = vec![0.0f32; 1]; // need 3
        max_pool1d_neon(&input, 8, 3, 2, &mut out);
    }

    // ── avg_pool1d_neon ────────────────────────────────────────────

    #[test]
    fn avg_pool1d_basic() {
        let input: Vec<f32> = (0..8).map(|x| x as f32).collect();
        let mut out = vec![0.0f32; 3];
        avg_pool1d_neon(&input, 8, 3, 2, &mut out);
        assert_approx_slice(&out, &[1.0, 3.0, 5.0]);
    }

    #[test]
    fn avg_pool1d_stride_1() {
        let input = vec![2.0, 4.0, 6.0, 8.0];
        let mut out = vec![0.0f32; 2];
        avg_pool1d_neon(&input, 4, 3, 1, &mut out);
        assert_approx_slice(&out, &[4.0, 6.0]);
    }

    #[test]
    fn avg_pool1d_pool_equals_input() {
        let input = vec![2.0, 4.0, 6.0];
        let mut out = vec![0.0f32; 1];
        avg_pool1d_neon(&input, 3, 3, 1, &mut out);
        assert!(approx_eq(out[0], 4.0));
    }

    #[test]
    fn avg_pool1d_pool_larger_than_input() {
        let input = vec![1.0, 2.0];
        let mut out = vec![0.0f32; 1];
        avg_pool1d_neon(&input, 2, 5, 1, &mut out);
        assert_eq!(out[0], 0.0); // no windows
    }

    #[test]
    fn avg_pool1d_single_element() {
        let input = vec![42.0];
        let mut out = vec![0.0f32; 1];
        avg_pool1d_neon(&input, 1, 1, 1, &mut out);
        assert!(approx_eq(out[0], 42.0));
    }

    #[test]
    fn avg_pool1d_stride_larger_than_pool() {
        let input: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let mut out = vec![0.0f32; 3];
        avg_pool1d_neon(&input, 12, 2, 4, &mut out);
        // windows: (0+1)/2=0.5, (4+5)/2=4.5, (8+9)/2=8.5
        assert_approx_slice(&out, &[0.5, 4.5, 8.5]);
    }

    #[test]
    fn avg_pool1d_negative_values() {
        let input = vec![-4.0, -2.0, -6.0, -8.0];
        let mut out = vec![0.0f32; 3]; // (4-2)/1+1=3
        avg_pool1d_neon(&input, 4, 2, 1, &mut out);
        assert_approx_slice(&out, &[-3.0, -4.0, -7.0]);
    }

    #[test]
    fn avg_pool1d_all_same() {
        let input = vec![5.0; 8];
        let mut out = vec![0.0f32; 2];
        avg_pool1d_neon(&input, 8, 4, 3, &mut out);
        assert_approx_slice(&out, &[5.0, 5.0]);
    }

    #[test]
    fn avg_pool1d_kernel_4_exact_chunk() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut out = vec![0.0f32; 2];
        avg_pool1d_neon(&input, 8, 4, 4, &mut out);
        assert_approx_slice(&out, &[2.5, 6.5]);
    }

    #[test]
    fn avg_pool1d_kernel_5_remainder() {
        let input = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let mut out = vec![0.0f32; 1];
        avg_pool1d_neon(&input, 5, 5, 1, &mut out);
        assert!(approx_eq(out[0], 30.0));
    }

    #[test]
    fn avg_pool1d_large_input() {
        let n = 64;
        let input: Vec<f32> = (0..n).map(|x| x as f32).collect();
        let mut out = vec![0.0f32; 16];
        avg_pool1d_neon(&input, n, 4, 4, &mut out);
        for (i, v) in out.iter().enumerate() {
            let expected = (i * 4) as f32 + 1.5;
            assert!(approx_eq(*v, expected), "idx {i}: {v} != {expected}");
        }
    }

    #[test]
    fn avg_pool1d_input_len_less_than_slice() {
        let input = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let mut out = vec![0.0f32; 3]; // input_len=4, (4-2)/1+1=3
        avg_pool1d_neon(&input, 4, 2, 1, &mut out);
        assert_approx_slice(&out, &[15.0, 25.0, 35.0]);
    }

    #[test]
    #[should_panic(expected = "pool_size must be > 0")]
    fn avg_pool1d_zero_pool_panics() {
        let input = vec![1.0];
        let mut out = vec![0.0f32; 1];
        avg_pool1d_neon(&input, 1, 0, 1, &mut out);
    }

    #[test]
    #[should_panic(expected = "stride must be > 0")]
    fn avg_pool1d_zero_stride_panics() {
        let input = vec![1.0];
        let mut out = vec![0.0f32; 1];
        avg_pool1d_neon(&input, 1, 1, 0, &mut out);
    }

    #[test]
    #[should_panic(expected = "output too small")]
    fn avg_pool1d_output_too_small_panics() {
        let input = vec![1.0; 8];
        let mut out = vec![0.0f32; 1];
        avg_pool1d_neon(&input, 8, 3, 2, &mut out);
    }

    // ── global_avg_pool_neon ───────────────────────────────────────

    #[test]
    fn global_avg_pool_multi_channel() {
        let input: Vec<f32> = (0..15).map(|x| x as f32).collect();
        let mut out = vec![0.0f32; 3];
        global_avg_pool_neon(&input, 3, 5, &mut out);
        assert_approx_slice(&out, &[2.0, 7.0, 12.0]);
    }

    #[test]
    fn global_avg_pool_single_channel() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut out = vec![0.0f32; 1];
        global_avg_pool_neon(&input, 1, 8, &mut out);
        assert!(approx_eq(out[0], 4.5));
    }

    #[test]
    fn global_avg_pool_single_element_channels() {
        let input = vec![3.0, 7.0, 11.0];
        let mut out = vec![0.0f32; 3];
        global_avg_pool_neon(&input, 3, 1, &mut out);
        assert_eq!(out, vec![3.0, 7.0, 11.0]);
    }

    #[test]
    fn global_avg_pool_large_spatial() {
        let spatial = 64;
        let input: Vec<f32> = (0..spatial).map(|x| x as f32).collect();
        let mut out = vec![0.0f32; 1];
        global_avg_pool_neon(&input, 1, spatial, &mut out);
        let expected = (spatial - 1) as f32 / 2.0;
        assert!(approx_eq(out[0], expected));
    }

    #[test]
    fn global_avg_pool_negative_values() {
        let input = vec![-2.0, -4.0, -6.0, -8.0];
        let mut out = vec![0.0f32; 1];
        global_avg_pool_neon(&input, 1, 4, &mut out);
        assert!(approx_eq(out[0], -5.0));
    }

    #[test]
    fn global_avg_pool_two_channels() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let mut out = vec![0.0f32; 2];
        global_avg_pool_neon(&input, 2, 4, &mut out);
        assert_approx_slice(&out, &[2.5, 25.0]);
    }

    #[test]
    fn global_avg_pool_all_zeros() {
        let input = vec![0.0f32; 16];
        let mut out = vec![1.0f32; 2];
        global_avg_pool_neon(&input, 2, 8, &mut out);
        assert_eq!(out, vec![0.0, 0.0]);
    }

    #[test]
    fn global_avg_pool_spatial_5_remainder() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut out = vec![0.0f32; 1];
        global_avg_pool_neon(&input, 1, 5, &mut out);
        assert!(approx_eq(out[0], 3.0));
    }

    #[test]
    #[should_panic(expected = "channels must be > 0")]
    fn global_avg_pool_zero_channels_panics() {
        let input = vec![1.0];
        let mut out = vec![0.0f32; 1];
        global_avg_pool_neon(&input, 0, 1, &mut out);
    }

    #[test]
    #[should_panic(expected = "spatial_size must be > 0")]
    fn global_avg_pool_zero_spatial_panics() {
        let input = vec![1.0];
        let mut out = vec![0.0f32; 1];
        global_avg_pool_neon(&input, 1, 0, &mut out);
    }

    #[test]
    #[should_panic(expected = "input too short")]
    fn global_avg_pool_input_too_short_panics() {
        let input = vec![1.0, 2.0];
        let mut out = vec![0.0f32; 2];
        global_avg_pool_neon(&input, 2, 4, &mut out);
    }

    #[test]
    #[should_panic(expected = "output too small")]
    fn global_avg_pool_output_too_small_panics() {
        let input = vec![1.0; 8];
        let mut out = vec![0.0f32; 1];
        global_avg_pool_neon(&input, 2, 4, &mut out);
    }

    // ── global_max_pool_neon ───────────────────────────────────────

    #[test]
    fn global_max_pool_multi_channel() {
        let input: Vec<f32> = (0..15).map(|x| x as f32).collect();
        let mut out = vec![0.0f32; 3];
        global_max_pool_neon(&input, 3, 5, &mut out);
        assert_eq!(out, vec![4.0, 9.0, 14.0]);
    }

    #[test]
    fn global_max_pool_single_channel() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut out = vec![0.0f32; 1];
        global_max_pool_neon(&input, 1, 8, &mut out);
        assert_eq!(out[0], 8.0);
    }

    #[test]
    fn global_max_pool_single_element_channels() {
        let input = vec![3.0, 7.0, 11.0];
        let mut out = vec![0.0f32; 3];
        global_max_pool_neon(&input, 3, 1, &mut out);
        assert_eq!(out, vec![3.0, 7.0, 11.0]);
    }

    #[test]
    fn global_max_pool_negative_values() {
        let input = vec![-5.0, -3.0, -8.0, -1.0];
        let mut out = vec![0.0f32; 1];
        global_max_pool_neon(&input, 1, 4, &mut out);
        assert_eq!(out[0], -1.0);
    }

    #[test]
    fn global_max_pool_all_same() {
        let input = vec![7.0; 8];
        let mut out = vec![0.0f32; 2];
        global_max_pool_neon(&input, 2, 4, &mut out);
        assert_eq!(out, vec![7.0, 7.0]);
    }

    #[test]
    fn global_max_pool_with_inf() {
        let input = vec![1.0, f32::INFINITY, 3.0, 4.0];
        let mut out = vec![0.0f32; 1];
        global_max_pool_neon(&input, 1, 4, &mut out);
        assert_eq!(out[0], f32::INFINITY);
    }

    #[test]
    fn global_max_pool_large_spatial() {
        let spatial = 64;
        let input: Vec<f32> = (0..spatial).map(|x| x as f32).collect();
        let mut out = vec![0.0f32; 1];
        global_max_pool_neon(&input, 1, spatial, &mut out);
        assert_eq!(out[0], (spatial - 1) as f32);
    }

    #[test]
    fn global_max_pool_two_channels() {
        let input = vec![1.0, 9.0, 3.0, 4.0, 10.0, 2.0, 30.0, 5.0];
        let mut out = vec![0.0f32; 2];
        global_max_pool_neon(&input, 2, 4, &mut out);
        assert_eq!(out, vec![9.0, 30.0]);
    }

    #[test]
    fn global_max_pool_spatial_5_remainder() {
        let input = vec![1.0, 5.0, 3.0, 4.0, 2.0];
        let mut out = vec![0.0f32; 1];
        global_max_pool_neon(&input, 1, 5, &mut out);
        assert_eq!(out[0], 5.0);
    }

    #[test]
    fn global_max_pool_all_neg_inf() {
        let input = vec![f32::NEG_INFINITY; 4];
        let mut out = vec![0.0f32; 1];
        global_max_pool_neon(&input, 1, 4, &mut out);
        assert_eq!(out[0], f32::NEG_INFINITY);
    }

    #[test]
    #[should_panic(expected = "channels must be > 0")]
    fn global_max_pool_zero_channels_panics() {
        let input = vec![1.0];
        let mut out = vec![0.0f32; 1];
        global_max_pool_neon(&input, 0, 1, &mut out);
    }

    #[test]
    #[should_panic(expected = "spatial_size must be > 0")]
    fn global_max_pool_zero_spatial_panics() {
        let input = vec![1.0];
        let mut out = vec![0.0f32; 1];
        global_max_pool_neon(&input, 1, 0, &mut out);
    }

    #[test]
    #[should_panic(expected = "input too short")]
    fn global_max_pool_input_too_short_panics() {
        let input = vec![1.0, 2.0];
        let mut out = vec![0.0f32; 2];
        global_max_pool_neon(&input, 2, 4, &mut out);
    }

    #[test]
    #[should_panic(expected = "output too small")]
    fn global_max_pool_output_too_small_panics() {
        let input = vec![1.0; 8];
        let mut out = vec![0.0f32; 1];
        global_max_pool_neon(&input, 2, 4, &mut out);
    }

    // ── adaptive_avg_pool1d_neon ───────────────────────────────────

    #[test]
    fn adaptive_avg_pool1d_halve() {
        let input = vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0];
        let mut out = vec![0.0f32; 3];
        adaptive_avg_pool1d_neon(&input, 6, 3, &mut out);
        assert_approx_slice(&out, &[2.0, 6.0, 10.0]);
    }

    #[test]
    fn adaptive_avg_pool1d_identity() {
        let input = vec![2.0, 4.0, 6.0, 8.0];
        let mut out = vec![0.0f32; 4];
        adaptive_avg_pool1d_neon(&input, 4, 4, &mut out);
        assert_eq!(out, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn adaptive_avg_pool1d_to_one() {
        let input = vec![10.0, 20.0, 30.0, 40.0];
        let mut out = vec![0.0f32; 1];
        adaptive_avg_pool1d_neon(&input, 4, 1, &mut out);
        assert!(approx_eq(out[0], 25.0));
    }

    #[test]
    fn adaptive_avg_pool1d_single_input() {
        let input = vec![99.0];
        let mut out = vec![0.0f32; 1];
        adaptive_avg_pool1d_neon(&input, 1, 1, &mut out);
        assert!(approx_eq(out[0], 99.0));
    }

    #[test]
    fn adaptive_avg_pool1d_6_to_2() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = vec![0.0f32; 2];
        adaptive_avg_pool1d_neon(&input, 6, 2, &mut out);
        // bins: [0..3)=avg(1,2,3)=2, [3..6)=avg(4,5,6)=5
        assert_approx_slice(&out, &[2.0, 5.0]);
    }

    #[test]
    fn adaptive_avg_pool1d_8_to_4() {
        let input: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let mut out = vec![0.0f32; 4];
        adaptive_avg_pool1d_neon(&input, 8, 4, &mut out);
        // bins: [0,2)=1.5, [2,4)=3.5, [4,6)=5.5, [6,8)=7.5
        assert_approx_slice(&out, &[1.5, 3.5, 5.5, 7.5]);
    }

    #[test]
    fn adaptive_avg_pool1d_negative_values() {
        let input = vec![-4.0, -2.0, -6.0, -8.0];
        let mut out = vec![0.0f32; 2];
        adaptive_avg_pool1d_neon(&input, 4, 2, &mut out);
        // bins: [-4,-2]/2=-3, [-6,-8]/2=-7
        assert_approx_slice(&out, &[-3.0, -7.0]);
    }

    #[test]
    fn adaptive_avg_pool1d_all_same() {
        let input = vec![5.0; 16];
        let mut out = vec![0.0f32; 4];
        adaptive_avg_pool1d_neon(&input, 16, 4, &mut out);
        assert_approx_slice(&out, &[5.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn adaptive_avg_pool1d_large_input() {
        let n = 64;
        let input: Vec<f32> = (0..n).map(|x| x as f32).collect();
        let mut out = vec![0.0f32; 4];
        adaptive_avg_pool1d_neon(&input, n, 4, &mut out);
        // Each bin of 16 elements
        for (i, v) in out.iter().enumerate() {
            let start = i * 16;
            let expected: f32 = (start..start + 16).map(|x| x as f32).sum::<f32>() / 16.0;
            assert!(approx_eq(*v, expected), "idx {i}: {v} != {expected}");
        }
    }

    #[test]
    fn adaptive_avg_pool1d_input_len_limits_slice() {
        let input = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let mut out = vec![0.0f32; 2];
        // input_len=4 → only first 4 elements used
        adaptive_avg_pool1d_neon(&input, 4, 2, &mut out);
        assert_approx_slice(&out, &[15.0, 35.0]);
    }

    #[test]
    fn adaptive_avg_pool1d_uneven_bins() {
        // 7 elements → 3 bins: sizes 2, 2, 3 (PyTorch bin formula)
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut out = vec![0.0f32; 3];
        adaptive_avg_pool1d_neon(&input, 7, 3, &mut out);
        // bin0: [0..2)=avg(1,2)=1.5
        // bin1: [2..4)=avg(3,4)=3.5  (floor(2*7/3)=4)
        // bin2: [4..7)=avg(5,6,7)=6.0 (floor(2*7/3)=4)
        let b0_start = (0 * 7) / 3; // 0
        let b0_end = (1 * 7) / 3; // 2
        let b1_start = b0_end; // 2
        let b1_end = (2 * 7) / 3; // 4
        let b2_start = b1_end; // 4
        let b2_end = (3 * 7) / 3; // 7
        let e0: f32 = input[b0_start..b0_end].iter().sum::<f32>() / (b0_end - b0_start) as f32;
        let e1: f32 = input[b1_start..b1_end].iter().sum::<f32>() / (b1_end - b1_start) as f32;
        let e2: f32 = input[b2_start..b2_end].iter().sum::<f32>() / (b2_end - b2_start) as f32;
        assert_approx_slice(&out, &[e0, e1, e2]);
    }

    #[test]
    #[should_panic(expected = "output_len must be > 0")]
    fn adaptive_avg_pool1d_zero_output_panics() {
        let input = vec![1.0];
        let mut out = vec![0.0f32; 1];
        adaptive_avg_pool1d_neon(&input, 1, 0, &mut out);
    }

    #[test]
    #[should_panic(expected = "output_len")]
    fn adaptive_avg_pool1d_output_exceeds_input_panics() {
        let input = vec![1.0, 2.0];
        let mut out = vec![0.0f32; 5];
        adaptive_avg_pool1d_neon(&input, 2, 5, &mut out);
    }

    #[test]
    #[should_panic(expected = "output too small")]
    fn adaptive_avg_pool1d_output_too_small_panics() {
        let input = vec![1.0; 8];
        let mut out = vec![0.0f32; 1];
        adaptive_avg_pool1d_neon(&input, 8, 4, &mut out);
    }

    // ── Cross-function consistency ─────────────────────────────────

    #[test]
    fn global_avg_matches_avg_pool_full_window() {
        let input: Vec<f32> = (0..8).map(|x| x as f32).collect();
        let mut avg_out = vec![0.0f32; 1];
        let mut global_out = vec![0.0f32; 1];
        avg_pool1d_neon(&input, 8, 8, 1, &mut avg_out);
        global_avg_pool_neon(&input, 1, 8, &mut global_out);
        assert!(approx_eq(avg_out[0], global_out[0]));
    }

    #[test]
    fn global_max_matches_max_pool_full_window() {
        let input = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let mut max_out = vec![0.0f32; 1];
        let mut global_out = vec![0.0f32; 1];
        max_pool1d_neon(&input, 8, 8, 1, &mut max_out);
        global_max_pool_neon(&input, 1, 8, &mut global_out);
        assert_eq!(max_out[0], global_out[0]);
    }

    #[test]
    fn adaptive_identity_matches_input() {
        let input = vec![5.0, 10.0, 15.0, 20.0, 25.0];
        let mut out = vec![0.0f32; 5];
        adaptive_avg_pool1d_neon(&input, 5, 5, &mut out);
        assert_eq!(out, input);
    }

    #[test]
    fn adaptive_to_one_matches_global_avg() {
        let input: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let mut adaptive_out = vec![0.0f32; 1];
        let mut global_out = vec![0.0f32; 1];
        adaptive_avg_pool1d_neon(&input, 12, 1, &mut adaptive_out);
        global_avg_pool_neon(&input, 1, 12, &mut global_out);
        assert!(approx_eq(adaptive_out[0], global_out[0]));
    }

    #[test]
    fn max_pool_preserves_global_max() {
        let input = vec![3.0, 1.0, 9.0, 2.0, 7.0, 5.0];
        let global_max = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut out = vec![0.0f32; 2];
        max_pool1d_neon(&input, 6, 3, 3, &mut out);
        let pool_max = out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(pool_max, global_max);
    }

    #[test]
    fn avg_pool_sum_invariant() {
        // With non-overlapping windows of equal size, the weighted sum
        // of pool outputs equals the sum of the input.
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let pool_size = 3;
        let stride = 3;
        let mut out = vec![0.0f32; 2];
        avg_pool1d_neon(&input, 6, pool_size, stride, &mut out);
        let total: f32 = out.iter().map(|v| v * pool_size as f32).sum();
        let expected: f32 = input.iter().sum();
        assert!(approx_eq(total, expected));
    }

    // ── Edge case: pool1d_output_len ───────────────────────────────

    #[test]
    fn output_len_zero_when_pool_exceeds_input() {
        assert_eq!(pool1d_output_len(3, 5, 1), 0);
    }

    #[test]
    fn output_len_one_when_pool_equals_input() {
        assert_eq!(pool1d_output_len(5, 5, 1), 1);
    }

    #[test]
    fn output_len_standard() {
        assert_eq!(pool1d_output_len(10, 3, 2), 4);
    }

    #[test]
    fn output_len_stride_equals_pool() {
        assert_eq!(pool1d_output_len(12, 4, 4), 3);
    }

    #[test]
    fn output_len_stride_1() {
        assert_eq!(pool1d_output_len(8, 3, 1), 6);
    }
}
