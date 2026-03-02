//! ARM NEON-optimized activation function kernels for Apple Silicon.
//!
//! Provides ReLU, sigmoid, SiLU, and GELU activation functions using
//! NEON SIMD intrinsics on AArch64. ReLU is fully vectorized; sigmoid,
//! SiLU, and GELU use scalar paths for transcendentals with NEON for
//! vectorizable operations.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ── Helpers ─────────────────────────────────────────────────────────

/// Scalar sigmoid: 1 / (1 + exp(-x)).
#[inline(always)]
fn scalar_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ── ReLU ────────────────────────────────────────────────────────────

/// Compute ReLU (max(0, x)) using NEON intrinsics.
///
/// Processes 4 × f32 lanes at a time with scalar fallback for remainder.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
///
/// # Panics
///
/// Panics if `output.len() < input.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_relu_f32(input: &[f32], output: &mut [f32]) {
    assert!(output.len() >= input.len(), "output buffer too small");
    let n = input.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let zero = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let offset = i * 4;
        unsafe {
            let v = vld1q_f32(input.as_ptr().add(offset));
            let result = vmaxq_f32(v, zero);
            vst1q_f32(output.as_mut_ptr().add(offset), result);
        }
    }

    let tail_start = chunks * 4;
    for i in 0..remainder {
        let x = input[tail_start + i];
        output[tail_start + i] = if x > 0.0 { x } else { 0.0 };
    }
}

// ── Sigmoid ─────────────────────────────────────────────────────────

/// Compute sigmoid (1/(1+exp(-x))) element-wise.
///
/// Uses scalar computation since NEON lacks a native exp intrinsic.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
///
/// # Panics
///
/// Panics if `output.len() < input.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_sigmoid_f32(input: &[f32], output: &mut [f32]) {
    assert!(output.len() >= input.len(), "output buffer too small");
    for (x, o) in input.iter().zip(output.iter_mut()) {
        *o = scalar_sigmoid(*x);
    }
}

// ── SiLU ────────────────────────────────────────────────────────────

/// Compute SiLU (x * sigmoid(x)) — the activation used in BitNet.
///
/// Computes sigmoid in scalar, then uses NEON for the final multiply
/// of x * sigmoid(x) on 4-wide lanes.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
///
/// # Panics
///
/// Panics if `output.len() < input.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_silu_f32(input: &[f32], output: &mut [f32]) {
    assert!(output.len() >= input.len(), "output buffer too small");
    let n = input.len();

    // First compute sigmoid into output buffer (scalar).
    for (x, o) in input.iter().zip(output.iter_mut()) {
        *o = scalar_sigmoid(*x);
    }

    // Then multiply x * sigmoid(x) using NEON.
    let chunks = n / 4;
    let remainder = n % 4;

    for i in 0..chunks {
        let offset = i * 4;
        unsafe {
            let x_vec = vld1q_f32(input.as_ptr().add(offset));
            let sig_vec = vld1q_f32(output.as_ptr().add(offset));
            let result = vmulq_f32(x_vec, sig_vec);
            vst1q_f32(output.as_mut_ptr().add(offset), result);
        }
    }

    let tail_start = chunks * 4;
    for i in 0..remainder {
        let idx = tail_start + i;
        output[idx] *= input[idx];
    }
}

// ── GELU ────────────────────────────────────────────────────────────

/// Compute approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³))).
///
/// Uses scalar computation for tanh. This is the "fast" GELU approximation
/// commonly used in transformer models.
///
/// # Safety
///
/// Caller must ensure the target supports NEON (always true on AArch64).
///
/// # Panics
///
/// Panics if `output.len() < input.len()`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn neon_gelu_f32(input: &[f32], output: &mut [f32]) {
    assert!(output.len() >= input.len(), "output buffer too small");
    let sqrt_2_over_pi: f32 = (2.0_f32 / std::f32::consts::PI).sqrt();

    for (x, o) in input.iter().zip(output.iter_mut()) {
        let x3 = x * x * x;
        let inner = sqrt_2_over_pi * (x + 0.044715 * x3);
        *o = 0.5 * x * (1.0 + inner.tanh());
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    const EPS: f32 = 1e-5;

    // ── ReLU tests ──────────────────────────────────────────────────

    #[test]
    fn test_relu_basic() {
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let mut output = [0.0_f32; 4];
        unsafe { neon_relu_f32(&input, &mut output) };
        assert_eq!(output, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_relu_negative() {
        let input = [-1.0_f32, -2.0, -0.5, -100.0];
        let mut output = [0.0_f32; 4];
        unsafe { neon_relu_f32(&input, &mut output) };
        assert_eq!(output, [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_relu_mixed() {
        let input = [-1.0_f32, 0.0, 1.0, -0.5, 2.0, -3.0, 0.1];
        let mut output = [0.0_f32; 7];
        unsafe { neon_relu_f32(&input, &mut output) };
        assert_eq!(output, [0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.1]);
    }

    // ── Sigmoid tests ───────────────────────────────────────────────

    #[test]
    fn test_sigmoid_bounds() {
        let input = [-10.0_f32, -1.0, 0.0, 1.0, 10.0];
        let mut output = [0.0_f32; 5];
        unsafe { neon_sigmoid_f32(&input, &mut output) };
        for &v in &output {
            assert!(v >= 0.0 && v <= 1.0, "sigmoid out of [0,1]: {v}");
        }
    }

    #[test]
    fn test_sigmoid_zero() {
        let input = [0.0_f32];
        let mut output = [0.0_f32; 1];
        unsafe { neon_sigmoid_f32(&input, &mut output) };
        assert!(approx_eq(output[0], 0.5, EPS), "sigmoid(0) = {}", output[0]);
    }

    // ── SiLU tests ──────────────────────────────────────────────────

    #[test]
    fn test_silu_zero() {
        let input = [0.0_f32];
        let mut output = [0.0_f32; 1];
        unsafe { neon_silu_f32(&input, &mut output) };
        assert!(approx_eq(output[0], 0.0, EPS), "silu(0) = {}", output[0]);
    }

    #[test]
    fn test_silu_positive() {
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let mut output = [0.0_f32; 4];
        unsafe { neon_silu_f32(&input, &mut output) };
        for (i, (&x, &o)) in input.iter().zip(output.iter()).enumerate() {
            let expected = x * scalar_sigmoid(x);
            assert!(
                approx_eq(o, expected, EPS),
                "silu({x}) = {o}, expected {expected} at index {i}"
            );
        }
    }

    #[test]
    fn test_silu_negative() {
        let input = [-1.0_f32, -2.0, -3.0, -4.0];
        let mut output = [0.0_f32; 4];
        unsafe { neon_silu_f32(&input, &mut output) };
        for (&x, &o) in input.iter().zip(output.iter()) {
            let expected = x * scalar_sigmoid(x);
            assert!(approx_eq(o, expected, EPS), "silu({x}) = {o}, expected {expected}");
        }
    }

    // ── GELU tests ──────────────────────────────────────────────────

    #[test]
    fn test_gelu_zero() {
        let input = [0.0_f32];
        let mut output = [0.0_f32; 1];
        unsafe { neon_gelu_f32(&input, &mut output) };
        assert!(approx_eq(output[0], 0.0, EPS), "gelu(0) = {}", output[0]);
    }

    #[test]
    fn test_gelu_positive() {
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let mut output = [0.0_f32; 4];
        unsafe { neon_gelu_f32(&input, &mut output) };
        for (&x, &o) in input.iter().zip(output.iter()) {
            assert!(o > 0.0, "gelu({x}) should be positive, got {o}");
            assert!(o <= x, "gelu({x}) should be <= x, got {o}");
        }
    }

    // ── Edge cases ──────────────────────────────────────────────────

    #[test]
    fn test_all_empty_slices() {
        let input: [f32; 0] = [];
        let mut output: [f32; 0] = [];
        unsafe {
            neon_relu_f32(&input, &mut output);
            neon_sigmoid_f32(&input, &mut output);
            neon_silu_f32(&input, &mut output);
            neon_gelu_f32(&input, &mut output);
        }
    }

    #[test]
    fn test_large_vectors() {
        let n = 1024;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 - 512.0) * 0.01).collect();
        let mut relu_out = vec![0.0_f32; n];
        let mut sig_out = vec![0.0_f32; n];
        let mut silu_out = vec![0.0_f32; n];
        let mut gelu_out = vec![0.0_f32; n];

        unsafe {
            neon_relu_f32(&input, &mut relu_out);
            neon_sigmoid_f32(&input, &mut sig_out);
            neon_silu_f32(&input, &mut silu_out);
            neon_gelu_f32(&input, &mut gelu_out);
        }

        for i in 0..n {
            let x = input[i];

            // ReLU
            let expected_relu = if x > 0.0 { x } else { 0.0 };
            assert!(
                approx_eq(relu_out[i], expected_relu, EPS),
                "relu mismatch at {i}: {} vs {expected_relu}",
                relu_out[i]
            );

            // Sigmoid bounds
            assert!(
                sig_out[i] >= 0.0 && sig_out[i] <= 1.0,
                "sigmoid out of bounds at {i}: {}",
                sig_out[i]
            );

            // SiLU = x * sigmoid(x)
            let expected_silu = x * scalar_sigmoid(x);
            assert!(
                approx_eq(silu_out[i], expected_silu, EPS),
                "silu mismatch at {i}: {} vs {expected_silu}",
                silu_out[i]
            );

            // GELU should be finite
            assert!(gelu_out[i].is_finite(), "gelu not finite at {i}: {}", gelu_out[i]);
        }
    }
}
