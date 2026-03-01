//! ARM NEON 1D convolution kernels for Apple Silicon.
//!
//! Provides vectorized 1D convolution, depthwise 1D convolution, and
//! cross-correlation using NEON SIMD intrinsics on AArch64. Processes
//! 4 × f32 lanes at a time with scalar fallback for remainder elements.

use std::arch::aarch64::*;

/// Computes the output length of a 1D convolution.
#[inline]
fn conv1d_output_len(input_len: usize, kernel_len: usize, stride: usize) -> usize {
    if input_len < kernel_len || stride == 0 {
        return 0;
    }
    (input_len - kernel_len) / stride + 1
}

/// 1D convolution with NEON dot-product acceleration.
///
/// For each output position `o`, computes:
///   `output[o] = Σ_k input[o * stride + k] * kernel[k]`
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
///
/// # Panics
///
/// Panics if `output.len()` is less than the required output size, if `stride`
/// is zero, or if `kernel` is empty.
#[target_feature(enable = "neon")]
pub unsafe fn neon_conv1d(input: &[f32], kernel: &[f32], stride: usize, output: &mut [f32]) {
    assert!(stride > 0, "stride must be positive");
    assert!(!kernel.is_empty(), "kernel must not be empty");
    let out_len = conv1d_output_len(input.len(), kernel.len(), stride);
    assert!(output.len() >= out_len, "output too small: need {out_len}, got {}", output.len());

    let k_len = kernel.len();
    let k_chunks = k_len / 4;
    let k_rem = k_len % 4;
    let k_ptr = kernel.as_ptr();

    for (o, out_val) in output.iter_mut().enumerate().take(out_len) {
        let base = o * stride;
        // SAFETY: base + k_len <= input.len() guaranteed by conv1d_output_len
        let in_ptr = unsafe { input.as_ptr().add(base) };

        let mut acc = vdupq_n_f32(0.0);

        // NEON 4-wide multiply-accumulate over the kernel
        for c in 0..k_chunks {
            let offset = c * 4;
            // SAFETY: offset + 4 <= k_len, in bounds for both pointers
            let vi = unsafe { vld1q_f32(in_ptr.add(offset)) };
            let vk = unsafe { vld1q_f32(k_ptr.add(offset)) };
            acc = vfmaq_f32(acc, vi, vk);
        }

        // Horizontal sum of the 4-lane accumulator
        let low = vget_low_f32(acc);
        let high = vget_high_f32(acc);
        let pair = vadd_f32(low, high);
        let mut sum = vget_lane_f32::<0>(pair) + vget_lane_f32::<1>(pair);

        // Scalar tail for remaining kernel elements
        for r in 0..k_rem {
            let idx = k_chunks * 4 + r;
            sum += input[base + idx] * kernel[idx];
        }

        *out_val = sum;
    }
}

/// Depthwise separable 1D convolution with NEON acceleration.
///
/// Each channel is convolved independently with its own kernel slice.
/// `input` is laid out as `[channels][samples_per_channel]` (contiguous per channel).
/// `kernel` is laid out as `[channels][kernel_len_per_channel]`.
/// `output` is laid out as `[channels][output_len_per_channel]`.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
///
/// # Panics
///
/// Panics if the input/kernel/output sizes are inconsistent with `channels`,
/// if `stride` is zero, or if `channels` is zero.
#[target_feature(enable = "neon")]
pub unsafe fn neon_conv1d_depthwise(
    input: &[f32],
    channels: usize,
    kernel: &[f32],
    stride: usize,
    output: &mut [f32],
) {
    assert!(stride > 0, "stride must be positive");
    assert!(channels > 0, "channels must be positive");
    assert_eq!(input.len() % channels, 0, "input length must be divisible by channels");
    assert_eq!(kernel.len() % channels, 0, "kernel length must be divisible by channels");

    let samples = input.len() / channels;
    let k_len = kernel.len() / channels;
    assert!(k_len > 0, "kernel must not be empty");
    let out_per_ch = conv1d_output_len(samples, k_len, stride);
    assert!(
        output.len() >= channels * out_per_ch,
        "output too small: need {}, got {}",
        channels * out_per_ch,
        output.len()
    );

    let k_chunks = k_len / 4;
    let k_rem = k_len % 4;

    for ch in 0..channels {
        let in_off = ch * samples;
        let k_off = ch * k_len;
        let out_off = ch * out_per_ch;

        for o in 0..out_per_ch {
            let base = in_off + o * stride;
            // SAFETY: base + k_len <= in_off + samples, in bounds
            let in_ptr = unsafe { input.as_ptr().add(base) };
            let k_ptr = unsafe { kernel.as_ptr().add(k_off) };

            let mut acc = vdupq_n_f32(0.0);

            for c in 0..k_chunks {
                let offset = c * 4;
                // SAFETY: offset + 4 <= k_len, in bounds for both pointers
                let vi = unsafe { vld1q_f32(in_ptr.add(offset)) };
                let vk = unsafe { vld1q_f32(k_ptr.add(offset)) };
                acc = vfmaq_f32(acc, vi, vk);
            }

            let low = vget_low_f32(acc);
            let high = vget_high_f32(acc);
            let pair = vadd_f32(low, high);
            let mut sum = vget_lane_f32::<0>(pair) + vget_lane_f32::<1>(pair);

            for r in 0..k_rem {
                let idx = k_chunks * 4 + r;
                sum += input[base + idx] * kernel[k_off + idx];
            }

            output[out_off + o] = sum;
        }
    }
}

/// Cross-correlation of two signals using NEON acceleration.
///
/// Computes `output[o] = Σ_k a[o + k] * b[k]` for `o` in `0..(a.len() - b.len() + 1)`.
/// This is equivalent to convolution without kernel reversal and is commonly used
/// in attention-pattern scoring.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
///
/// # Panics
///
/// Panics if `b` is empty, `a` is shorter than `b`, or `output` is too small.
#[target_feature(enable = "neon")]
pub unsafe fn neon_cross_correlation(a: &[f32], b: &[f32], output: &mut [f32]) {
    assert!(!b.is_empty(), "b must not be empty");
    assert!(a.len() >= b.len(), "a must be at least as long as b");
    let out_len = a.len() - b.len() + 1;
    assert!(output.len() >= out_len, "output too small: need {out_len}, got {}", output.len());

    let b_len = b.len();
    let b_chunks = b_len / 4;
    let b_rem = b_len % 4;
    let b_ptr = b.as_ptr();

    for o in 0..out_len {
        // SAFETY: o + b_len <= a.len(), in bounds
        let a_ptr = unsafe { a.as_ptr().add(o) };

        let mut acc = vdupq_n_f32(0.0);

        for c in 0..b_chunks {
            let offset = c * 4;
            // SAFETY: offset + 4 <= b_len, in bounds for both pointers
            let va = unsafe { vld1q_f32(a_ptr.add(offset)) };
            let vb = unsafe { vld1q_f32(b_ptr.add(offset)) };
            acc = vfmaq_f32(acc, va, vb);
        }

        let low = vget_low_f32(acc);
        let high = vget_high_f32(acc);
        let pair = vadd_f32(low, high);
        let mut sum = vget_lane_f32::<0>(pair) + vget_lane_f32::<1>(pair);

        for r in 0..b_rem {
            let idx = b_chunks * 4 + r;
            sum += a[o + idx] * b[idx];
        }

        output[o] = sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scalar reference for conv1d to verify NEON results.
    fn ref_conv1d(input: &[f32], kernel: &[f32], stride: usize) -> Vec<f32> {
        let out_len = conv1d_output_len(input.len(), kernel.len(), stride);
        let mut out = vec![0.0f32; out_len];
        for o in 0..out_len {
            let base = o * stride;
            let mut sum = 0.0f32;
            for k in 0..kernel.len() {
                sum += input[base + k] * kernel[k];
            }
            out[o] = sum;
        }
        out
    }

    #[test]
    fn test_conv1d_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let kernel = vec![1.0, 0.5, 0.25];
        let expected = ref_conv1d(&input, &kernel, 1);
        let mut output = vec![0.0f32; expected.len()];
        unsafe { neon_conv1d(&input, &kernel, 1, &mut output) };
        for (a, b) in output.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "got {a}, expected {b}");
        }
    }

    #[test]
    fn test_conv1d_stride2() {
        let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let kernel = vec![1.0, -1.0, 0.5, 0.25];
        let expected = ref_conv1d(&input, &kernel, 2);
        let mut output = vec![0.0f32; expected.len()];
        unsafe { neon_conv1d(&input, &kernel, 2, &mut output) };
        for (a, b) in output.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "got {a}, expected {b}");
        }
    }

    #[test]
    fn test_conv1d_large_kernel() {
        // Kernel larger than 4 elements to exercise both NEON and scalar tail
        let input: Vec<f32> = (0..20).map(|i| (i as f32) * 0.1).collect();
        let kernel = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
        let expected = ref_conv1d(&input, &kernel, 1);
        let mut output = vec![0.0f32; expected.len()];
        unsafe { neon_conv1d(&input, &kernel, 1, &mut output) };
        for (a, b) in output.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-4, "got {a}, expected {b}");
        }
    }

    #[test]
    fn test_conv1d_single_element_kernel() {
        let input = vec![2.0, 4.0, 6.0, 8.0];
        let kernel = vec![3.0];
        let expected = ref_conv1d(&input, &kernel, 1);
        let mut output = vec![0.0f32; expected.len()];
        unsafe { neon_conv1d(&input, &kernel, 1, &mut output) };
        for (a, b) in output.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "got {a}, expected {b}");
        }
    }

    #[test]
    fn test_conv1d_output_len() {
        assert_eq!(conv1d_output_len(8, 3, 1), 6);
        assert_eq!(conv1d_output_len(8, 3, 2), 3);
        assert_eq!(conv1d_output_len(10, 5, 3), 2);
        assert_eq!(conv1d_output_len(3, 5, 1), 0);
        assert_eq!(conv1d_output_len(5, 5, 1), 1);
        assert_eq!(conv1d_output_len(0, 3, 1), 0);
        assert_eq!(conv1d_output_len(5, 3, 0), 0);
    }

    #[test]
    fn test_depthwise_conv1d() {
        // 2 channels, 6 samples each
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // ch0
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // ch1
        ];
        // 2 channels, kernel size 3 each
        let kernel = vec![
            1.0, 0.0, -1.0, // ch0
            0.5, 0.5, 0.5, // ch1
        ];
        let out_per_ch = conv1d_output_len(6, 3, 1); // 4
        let mut output = vec![0.0f32; 2 * out_per_ch];
        unsafe { neon_conv1d_depthwise(&input, 2, &kernel, 1, &mut output) };

        // Verify ch0: kernel [1, 0, -1]
        let expected_ch0 = ref_conv1d(&input[0..6], &kernel[0..3], 1);
        for (a, b) in output[0..out_per_ch].iter().zip(expected_ch0.iter()) {
            assert!((a - b).abs() < 1e-5, "ch0: got {a}, expected {b}");
        }

        // Verify ch1: kernel [0.5, 0.5, 0.5]
        let expected_ch1 = ref_conv1d(&input[6..12], &kernel[3..6], 1);
        for (a, b) in output[out_per_ch..].iter().zip(expected_ch1.iter()) {
            assert!((a - b).abs() < 1e-5, "ch1: got {a}, expected {b}");
        }
    }

    #[test]
    fn test_depthwise_conv1d_stride2() {
        let input = vec![
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, // ch0
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // ch1
        ];
        let kernel = vec![
            1.0, 1.0, // ch0
            2.0, 2.0, // ch1
        ];
        let out_per_ch = conv1d_output_len(8, 2, 2); // 4
        let mut output = vec![0.0f32; 2 * out_per_ch];
        unsafe { neon_conv1d_depthwise(&input, 2, &kernel, 2, &mut output) };

        let expected_ch0 = ref_conv1d(&input[0..8], &kernel[0..2], 2);
        let expected_ch1 = ref_conv1d(&input[8..16], &kernel[2..4], 2);
        for (a, b) in output[..out_per_ch].iter().zip(expected_ch0.iter()) {
            assert!((a - b).abs() < 1e-5, "ch0: got {a}, expected {b}");
        }
        for (a, b) in output[out_per_ch..].iter().zip(expected_ch1.iter()) {
            assert!((a - b).abs() < 1e-5, "ch1: got {a}, expected {b}");
        }
    }

    #[test]
    fn test_cross_correlation_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![1.0, 0.5, 0.25];
        // Cross-correlation is the same as conv1d with stride=1 (no kernel flip)
        let expected = ref_conv1d(&a, &b, 1);
        let mut output = vec![0.0f32; expected.len()];
        unsafe { neon_cross_correlation(&a, &b, &mut output) };
        for (x, y) in output.iter().zip(expected.iter()) {
            assert!((x - y).abs() < 1e-5, "got {x}, expected {y}");
        }
    }

    #[test]
    fn test_cross_correlation_identical() {
        // Auto-correlation peak at position 0
        let signal = vec![1.0, 0.0, -1.0, 0.0, 1.0];
        let pattern = vec![1.0, 0.0, -1.0];
        let mut output = vec![0.0f32; 3];
        unsafe { neon_cross_correlation(&signal, &pattern, &mut output) };
        // Position 0: 1*1 + 0*0 + (-1)*(-1) = 2.0
        assert!((output[0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_cross_correlation_large() {
        // Exercise NEON lanes: b has 9 elements (2 NEON chunks + 1 scalar)
        let a: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..9).map(|i| (i as f32) * 0.2).collect();
        let expected = ref_conv1d(&a, &b, 1);
        let mut output = vec![0.0f32; expected.len()];
        unsafe { neon_cross_correlation(&a, &b, &mut output) };
        for (x, y) in output.iter().zip(expected.iter()) {
            assert!((x - y).abs() < 1e-3, "got {x}, expected {y}");
        }
    }
}
