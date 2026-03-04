//! ARM NEON pooling operations for Apple Silicon.
//!
//! Provides NEON-optimized 1-D max, average, global average, and adaptive
//! average pooling for `f32` slices.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// 1-D max pooling using NEON `vmaxq_f32`.
///
/// Slides a window of `kernel_size` over `input` with the given `stride`,
/// writing per-window maxima into `output`.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
///
/// # Panics
///
/// Panics if `kernel_size` or `stride` is zero, or if `output` is too small.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_max_pool_1d(
    input: &[f32],
    kernel_size: usize,
    stride: usize,
    output: &mut [f32],
) {
    assert!(kernel_size > 0, "kernel_size must be > 0");
    assert!(stride > 0, "stride must be > 0");

    let in_len = input.len();
    if in_len < kernel_size {
        return;
    }
    let out_len = (in_len - kernel_size) / stride + 1;
    assert!(output.len() >= out_len, "output too small: need {out_len}, got {}", output.len());

    let ptr = input.as_ptr();

    for (idx, out_val) in output.iter_mut().enumerate().take(out_len) {
        let base = idx * stride;
        let chunks = kernel_size / 4;
        let remainder = kernel_size % 4;

        let mut acc = vdupq_n_f32(f32::NEG_INFINITY);

        for c in 0..chunks {
            let v = unsafe { vld1q_f32(ptr.add(base + c * 4)) };
            acc = vmaxq_f32(acc, v);
        }

        let mut max_val = vmaxvq_f32(acc);

        for r in 0..remainder {
            let val = unsafe { *ptr.add(base + chunks * 4 + r) };
            if val > max_val {
                max_val = val;
            }
        }

        *out_val = max_val;
    }
}

/// 1-D average pooling using NEON `vaddq_f32`.
///
/// Slides a window of `kernel_size` over `input` with the given `stride`,
/// writing per-window means into `output`.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
///
/// # Panics
///
/// Panics if `kernel_size` or `stride` is zero, or if `output` is too small.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_avg_pool_1d(
    input: &[f32],
    kernel_size: usize,
    stride: usize,
    output: &mut [f32],
) {
    assert!(kernel_size > 0, "kernel_size must be > 0");
    assert!(stride > 0, "stride must be > 0");

    let in_len = input.len();
    if in_len < kernel_size {
        return;
    }
    let out_len = (in_len - kernel_size) / stride + 1;
    assert!(output.len() >= out_len, "output too small: need {out_len}, got {}", output.len());

    let ptr = input.as_ptr();
    let inv_k = 1.0 / kernel_size as f32;

    for (idx, out_val) in output.iter_mut().enumerate().take(out_len) {
        let base = idx * stride;
        let chunks = kernel_size / 4;
        let remainder = kernel_size % 4;

        let mut acc = vdupq_n_f32(0.0);

        for c in 0..chunks {
            let v = unsafe { vld1q_f32(ptr.add(base + c * 4)) };
            acc = vaddq_f32(acc, v);
        }

        let mut sum = vaddvq_f32(acc);

        for r in 0..remainder {
            sum += unsafe { *ptr.add(base + chunks * 4 + r) };
        }

        *out_val = sum * inv_k;
    }
}

/// Global average pooling per channel.
///
/// Given a flat buffer laid out as `[channels][spatial_size]`, computes the
/// mean of each spatial slice and writes one value per channel into `output`.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
///
/// # Panics
///
/// Panics if `channels` or `spatial_size` is zero, if `input.len()` is not
/// `channels * spatial_size`, or if `output` is too small.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_global_avg_pool(
    input: &[f32],
    channels: usize,
    spatial_size: usize,
    output: &mut [f32],
) {
    assert!(channels > 0, "channels must be > 0");
    assert!(spatial_size > 0, "spatial_size must be > 0");
    assert_eq!(
        input.len(),
        channels * spatial_size,
        "input length must equal channels * spatial_size"
    );
    assert!(output.len() >= channels, "output too small: need {channels}, got {}", output.len());

    let ptr = input.as_ptr();
    let inv_s = 1.0 / spatial_size as f32;

    for (ch, out_val) in output.iter_mut().enumerate().take(channels) {
        let base = ch * spatial_size;
        let chunks = spatial_size / 4;
        let remainder = spatial_size % 4;

        let mut acc = vdupq_n_f32(0.0);

        for c in 0..chunks {
            let v = unsafe { vld1q_f32(ptr.add(base + c * 4)) };
            acc = vaddq_f32(acc, v);
        }

        let mut sum = vaddvq_f32(acc);

        for r in 0..remainder {
            sum += unsafe { *ptr.add(base + chunks * 4 + r) };
        }

        *out_val = sum * inv_s;
    }
}

/// Adaptive 1-D average pooling.
///
/// Divides `input` into `output_size` bins (matching PyTorch
/// `AdaptiveAvgPool1d` bin boundaries) and writes the mean of each bin
/// into `output`.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
///
/// # Panics
///
/// Panics if `output_size` is zero, if `output_size > input.len()`, or if
/// `output` is too small.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_adaptive_avg_pool_1d(input: &[f32], output_size: usize, output: &mut [f32]) {
    let in_len = input.len();
    assert!(output_size > 0, "output_size must be > 0");
    assert!(
        output_size <= in_len,
        "output_size ({output_size}) must be <= input length ({in_len})"
    );
    assert!(
        output.len() >= output_size,
        "output too small: need {output_size}, got {}",
        output.len()
    );

    let ptr = input.as_ptr();

    for (i, out_val) in output.iter_mut().enumerate().take(output_size) {
        // PyTorch-compatible bin boundaries.
        let start = (i * in_len) / output_size;
        let end = ((i + 1) * in_len) / output_size;
        let bin_len = end - start;

        let chunks = bin_len / 4;
        let remainder = bin_len % 4;

        let mut acc = vdupq_n_f32(0.0);

        for c in 0..chunks {
            let v = unsafe { vld1q_f32(ptr.add(start + c * 4)) };
            acc = vaddq_f32(acc, v);
        }

        let mut sum = vaddvq_f32(acc);

        for r in 0..remainder {
            sum += unsafe { *ptr.add(start + chunks * 4 + r) };
        }

        *out_val = sum / bin_len as f32;
    }
}

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;

    // ── Max pooling ────────────────────────────────────────────────

    #[test]
    fn test_max_pool_1d_basic() {
        let input: Vec<f32> = (0..8).map(|x| x as f32).collect();
        let mut output = vec![0.0f32; 3]; // (8 - 3) / 2 + 1 = 3
        unsafe { neon_max_pool_1d(&input, 3, 2, &mut output) };
        // windows: [0,1,2]=2, [2,3,4]=4, [4,5,6]=6
        assert_eq!(output, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_max_pool_1d_stride_1() {
        let input = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let kernel_size = 3;
        let stride = 1;
        let out_len = (input.len() - kernel_size) / stride + 1; // 6
        let mut output = vec![0.0f32; out_len];
        unsafe { neon_max_pool_1d(&input, kernel_size, stride, &mut output) };
        assert_eq!(output, vec![4.0, 4.0, 5.0, 9.0, 9.0, 9.0]);
    }

    #[test]
    fn test_max_pool_1d_large_kernel() {
        // Kernel spans 5 elements to exercise the NEON 4-lane path + remainder.
        let input = vec![1.0, 8.0, 3.0, 7.0, 2.0, 9.0, 0.0, 4.0, 6.0, 5.0];
        let mut output = vec![0.0f32; 2]; // (10 - 5) / 3 + 1 = 2
        unsafe { neon_max_pool_1d(&input, 5, 3, &mut output) };
        // windows: [1,8,3,7,2]=8, [7,2,9,0,4]=9
        assert_eq!(output, vec![8.0, 9.0]);
    }

    // ── Avg pooling ────────────────────────────────────────────────

    #[test]
    fn test_avg_pool_1d_basic() {
        let input: Vec<f32> = (0..8).map(|x| x as f32).collect();
        let mut output = vec![0.0f32; 3]; // (8 - 3) / 2 + 1 = 3
        unsafe { neon_avg_pool_1d(&input, 3, 2, &mut output) };
        // windows: (0+1+2)/3=1, (2+3+4)/3=3, (4+5+6)/3=5
        assert_eq!(output, vec![1.0, 3.0, 5.0]);
    }

    #[test]
    fn test_avg_pool_1d_stride_1() {
        let input = vec![2.0, 4.0, 6.0, 8.0];
        let mut output = vec![0.0f32; 2]; // (4 - 3) / 1 + 1 = 2
        unsafe { neon_avg_pool_1d(&input, 3, 1, &mut output) };
        // windows: (2+4+6)/3=4, (4+6+8)/3=6
        assert_eq!(output, vec![4.0, 6.0]);
    }

    // ── Global avg pooling ─────────────────────────────────────────

    #[test]
    fn test_global_avg_pool_multi_channel() {
        // 3 channels, 5 spatial elements each.
        let input: Vec<f32> = (0..15).map(|x| x as f32).collect();
        let mut output = vec![0.0f32; 3];
        unsafe { neon_global_avg_pool(&input, 3, 5, &mut output) };
        // ch0: (0+1+2+3+4)/5=2, ch1: (5+6+7+8+9)/5=7, ch2: (10+11+12+13+14)/5=12
        assert_eq!(output, vec![2.0, 7.0, 12.0]);
    }

    #[test]
    fn test_global_avg_pool_single_channel() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut output = vec![0.0f32; 1];
        unsafe { neon_global_avg_pool(&input, 1, 8, &mut output) };
        assert!((output[0] - 4.5).abs() < 1e-6);
    }

    // ── Adaptive avg pooling ───────────────────────────────────────

    #[test]
    fn test_adaptive_avg_pool_1d_halve() {
        let input = vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0];
        let mut output = vec![0.0f32; 3]; // 6 -> 3
        unsafe { neon_adaptive_avg_pool_1d(&input, 3, &mut output) };
        // bins: [1,3]=2, [5,7]=6, [9,11]=10
        assert_eq!(output, vec![2.0, 6.0, 10.0]);
    }

    #[test]
    fn test_adaptive_avg_pool_1d_identity() {
        // output_size == input length → identity.
        let input = vec![2.0, 4.0, 6.0, 8.0];
        let mut output = vec![0.0f32; 4];
        unsafe { neon_adaptive_avg_pool_1d(&input, 4, &mut output) };
        assert_eq!(output, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_adaptive_avg_pool_1d_to_one() {
        let input = vec![10.0, 20.0, 30.0, 40.0];
        let mut output = vec![0.0f32; 1];
        unsafe { neon_adaptive_avg_pool_1d(&input, 1, &mut output) };
        assert!((output[0] - 25.0).abs() < 1e-6);
    }
}
