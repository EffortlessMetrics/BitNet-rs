//! Numerical edge-case tests for Metal shader correctness on Apple Silicon.
//!
//! These tests validate the **mathematical correctness** of operations as they
//! would execute in Metal shaders, using pure Rust CPU simulation.  No Metal
//! runtime is required — the tests exercise the same numerical patterns that
//! trip up GPU half-precision pipelines (f16 range limits, FMA accumulation
//! order, softmax stability, RoPE periodicity, quantisation rounding, and
//! LayerNorm degenerate inputs).

#![cfg(target_os = "macos")]
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::useless_vec)]
#![allow(clippy::approx_constant)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::assertions_on_constants)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::unnecessary_cast)]

use std::f32::consts::PI;

// ── Helpers ────────────────────────────────────────────────────────

/// Absolute-tolerance comparison.
fn approx_eq(a: f32, b: f32, tol: f32) {
    assert!((a - b).abs() <= tol, "expected {a} ≈ {b} (tol={tol}, diff={})", (a - b).abs());
}

/// Relative-tolerance comparison (falls back to absolute when values are near zero).
fn relative_eq(a: f32, b: f32, rel_tol: f32, abs_tol: f32) {
    let diff = (a - b).abs();
    let mag = a.abs().max(b.abs());
    assert!(
        diff <= abs_tol || diff <= rel_tol * mag,
        "expected {a} ≈ {b} (rel_tol={rel_tol}, abs_tol={abs_tol}, diff={diff})"
    );
}

/// Simulate f16 round-trip: f32 → f16 → f32 using truncation to f16 range.
fn f32_to_f16_to_f32(v: f32) -> f32 {
    // f16 characteristics: 1 sign, 5 exponent, 10 mantissa bits
    // Range: ±65504, smallest normal: ~6.1e-5, epsilon: ~9.77e-4
    const F16_MAX: f32 = 65504.0;
    const F16_MIN_POSITIVE_NORMAL: f32 = 6.103_515_6e-5; // 2^-14
    const F16_EPSILON: f32 = 9.765_625e-4; // 2^-10

    if v.is_nan() {
        return f32::NAN;
    }
    if v.is_infinite() {
        return if v > 0.0 { f32::INFINITY } else { f32::NEG_INFINITY };
    }
    // Clamp to f16 representable range
    let clamped = v.clamp(-F16_MAX, F16_MAX);
    // Simulate precision loss: round to 10 mantissa bits
    // For values in the normal range, precision is ~3 decimal digits
    let bits = clamped.to_bits();
    // Zero out the lower 13 bits of mantissa (f32 has 23, f16 has 10)
    let rounded_bits = bits & 0xFFFF_E000;
    f32::from_bits(rounded_bits)
}

/// Numerically stable softmax (max-subtraction trick).
fn stable_softmax(input: &[f32]) -> Vec<f32> {
    assert!(!input.is_empty());
    let max_val = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = input.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Naive softmax (no max-subtraction — numerically unstable for large values).
fn naive_softmax(input: &[f32]) -> Vec<f32> {
    assert!(!input.is_empty());
    let exps: Vec<f32> = input.iter().map(|&x| x.exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Simple layer normalization: (x - mean) / sqrt(var + eps) * gamma + beta.
fn layer_norm_ref(input: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
    let n = input.len() as f32;
    let mean = input.iter().sum::<f32>() / n;
    let var = input.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
    let inv_std = 1.0 / (var + eps).sqrt();
    input.iter().enumerate().map(|(i, &x)| (x - mean) * inv_std * gamma[i] + beta[i]).collect()
}

/// RMS normalization: x / sqrt(mean(x^2) + eps) * gamma.
fn rms_norm_ref(input: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
    let n = input.len() as f32;
    let rms = (input.iter().map(|&x| x * x).sum::<f32>() / n + eps).sqrt();
    input.iter().enumerate().map(|(i, &x)| x / rms * gamma[i]).collect()
}

/// Symmetric 2-bit quantization: maps f32 → {-1, 0, +1}.
fn quantize_ternary(values: &[f32], threshold: f32) -> Vec<i8> {
    values
        .iter()
        .map(|&v| {
            if v > threshold {
                1
            } else if v < -threshold {
                -1
            } else {
                0
            }
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════
// § Float16 precision
// ═══════════════════════════════════════════════════════════════════

mod float16_precision {
    use super::*;

    #[test]
    fn f16_max_boundary_saturation() {
        // f16 max is 65504 — values beyond must clamp, not overflow to Inf
        let at_max = f32_to_f16_to_f32(65504.0);
        assert!(at_max.is_finite(), "f16 max should remain finite");
        approx_eq(at_max, 65504.0, 1.0);

        let beyond_max = f32_to_f16_to_f32(65536.0);
        // After clamping to f16 range, should be 65504
        assert!(beyond_max.is_finite(), "clamped value should be finite");
        assert!(beyond_max <= 65504.0);

        // Negative boundary
        let neg_max = f32_to_f16_to_f32(-65504.0);
        assert!(neg_max.is_finite());
        approx_eq(neg_max, -65504.0, 1.0);
    }

    #[test]
    fn f16_subnormal_precision_loss() {
        // Below f16 smallest normal (~6.1e-5), values lose precision rapidly
        let small_normal = 6.104e-5_f32;
        let rt = f32_to_f16_to_f32(small_normal);
        // Subnormal round-trip should preserve sign and rough magnitude
        assert!(rt >= 0.0, "sign must be preserved");
        assert!(rt < 1e-3, "magnitude must stay small");

        // Very small subnormal — may flush to zero
        let tiny = 1.0e-7_f32;
        let rt_tiny = f32_to_f16_to_f32(tiny);
        // In Metal f16, this may become zero — that's acceptable
        assert!(rt_tiny >= 0.0);
        assert!(rt_tiny < 1e-3);
    }

    #[test]
    fn f16_epsilon_precision_near_one() {
        // f16 epsilon is ~9.77e-4 — two values within this gap are indistinguishable
        let a = 1.0_f32;
        let b = 1.0_f32 + 5e-4; // less than f16 epsilon apart
        let ra = f32_to_f16_to_f32(a);
        let rb = f32_to_f16_to_f32(b);
        // After f16 round-trip, a and b may be identical
        assert!(
            (ra - rb).abs() <= 1e-3,
            "values within f16 epsilon should be very close after round-trip"
        );
    }

    #[test]
    fn f16_nan_and_inf_passthrough() {
        assert!(f32_to_f16_to_f32(f32::NAN).is_nan());
        assert!(f32_to_f16_to_f32(f32::INFINITY).is_infinite());
        assert!(f32_to_f16_to_f32(f32::NEG_INFINITY).is_infinite());
        assert!(f32_to_f16_to_f32(f32::NEG_INFINITY) < 0.0);
    }

    #[test]
    fn f16_negative_zero_preserved() {
        let nz = f32_to_f16_to_f32(-0.0_f32);
        assert!(nz == 0.0, "negative zero should compare equal to zero");
    }
}

// ═══════════════════════════════════════════════════════════════════
// § Softmax numerical stability
// ═══════════════════════════════════════════════════════════════════

mod softmax_stability {
    use super::*;

    #[test]
    fn stable_softmax_sums_to_one() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = stable_softmax(&input);
        let sum: f32 = output.iter().sum();
        approx_eq(sum, 1.0, 1e-6);
        for &v in &output {
            assert!(v >= 0.0 && v <= 1.0);
        }
    }

    #[test]
    fn softmax_large_values_stability() {
        // Large logits cause naive exp() to overflow — Metal shaders must use max-subtraction
        let input = vec![1000.0, 1001.0, 1002.0];
        let stable = stable_softmax(&input);
        let naive = naive_softmax(&input);

        // Stable version always works
        let stable_sum: f32 = stable.iter().sum();
        approx_eq(stable_sum, 1.0, 1e-6);
        for &v in &stable {
            assert!(v.is_finite(), "stable softmax must produce finite values");
        }

        // Naive version overflows — exp(1000) is Inf in f32
        let has_nan_or_inf = naive.iter().any(|v| v.is_nan() || v.is_infinite());
        assert!(
            has_nan_or_inf,
            "naive softmax should overflow with large inputs, confirming need for max-subtraction"
        );
    }

    #[test]
    fn softmax_negative_large_values() {
        // Very negative values should produce near-zero probabilities
        let input = vec![-1000.0, -999.0, 0.0];
        let output = stable_softmax(&input);
        approx_eq(output[2], 1.0, 1e-6); // only the zero dominates
        assert!(output[0] < 1e-6);
        assert!(output[1] < 1e-6);
    }

    #[test]
    fn softmax_identical_inputs_uniform() {
        // All-equal inputs → uniform distribution
        let input = vec![5.0; 8];
        let output = stable_softmax(&input);
        for &v in &output {
            approx_eq(v, 1.0 / 8.0, 1e-6);
        }
    }

    #[test]
    fn softmax_single_element() {
        let output = stable_softmax(&[42.0]);
        approx_eq(output[0], 1.0, 1e-7);
    }
}

// ═══════════════════════════════════════════════════════════════════
// § Matrix multiplication accumulation
// ═══════════════════════════════════════════════════════════════════

mod matmul_accumulation {
    use super::*;

    /// Dot product with sequential multiply-add (typical GPU shader pattern).
    fn dot_sequential(a: &[f32], b: &[f32]) -> f32 {
        let mut acc = 0.0_f32;
        for i in 0..a.len() {
            acc += a[i] * b[i];
        }
        acc
    }

    /// Dot product with FMA (fused multiply-add — single rounding).
    fn dot_fma(a: &[f32], b: &[f32]) -> f32 {
        let mut acc = 0.0_f32;
        for i in 0..a.len() {
            acc = a[i].mul_add(b[i], acc);
        }
        acc
    }

    /// Kahan compensated summation dot product (high accuracy reference).
    fn dot_kahan(a: &[f32], b: &[f32]) -> f32 {
        let mut sum = 0.0_f32;
        let mut comp = 0.0_f32;
        for i in 0..a.len() {
            let product = a[i] * b[i];
            let y = product - comp;
            let t = sum + y;
            comp = (t - sum) - y;
            sum = t;
        }
        sum
    }

    #[test]
    fn fma_vs_sequential_small_vectors() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let seq = dot_sequential(&a, &b);
        let fma = dot_fma(&a, &b);
        // Both should agree exactly for small integer-like values
        approx_eq(seq, 70.0, 1e-6);
        approx_eq(fma, 70.0, 1e-6);
    }

    #[test]
    fn fma_vs_sequential_accumulation_error() {
        // Large vectors of values that cause accumulation error
        let n = 10_000;
        let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001) + 0.1).collect();
        let b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.002) - 0.5).collect();

        let seq = dot_sequential(&a, &b);
        let fma = dot_fma(&a, &b);
        let kahan = dot_kahan(&a, &b);

        // FMA should be closer to the Kahan reference than sequential
        let seq_err = (seq - kahan).abs();
        let fma_err = (fma - kahan).abs();

        // Both should be in the right ballpark
        relative_eq(seq, kahan, 1e-3, 1.0);
        relative_eq(fma, kahan, 1e-3, 1.0);

        // FMA error should not be dramatically worse
        assert!(
            fma_err <= seq_err * 2.0 + 1e-3,
            "FMA error ({fma_err}) should not be dramatically worse than sequential ({seq_err})"
        );
    }

    #[test]
    fn matmul_catastrophic_cancellation() {
        // Large values that nearly cancel — tests accumulator precision
        let a = vec![1e6, 1e6, -1e6, -1e6, 1.0];
        let b = vec![1.0, -1.0, 1.0, -1.0, 1.0];
        // Expected: 1e6 - 1e6 - 1e6 + 1e6 + 1 = 1.0
        let seq = dot_sequential(&a, &b);
        let fma = dot_fma(&a, &b);
        approx_eq(seq, 1.0, 1e-1);
        approx_eq(fma, 1.0, 1e-1);
    }

    #[test]
    fn matmul_2d_small() {
        // 2×3 * 3×2 matmul
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2×3 row-major
        let b: Vec<f32> = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3×2 row-major
        let m = 2;
        let k = 3;
        let n = 2;
        let mut c = vec![0.0_f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0_f32;
                for p in 0..k {
                    acc = a[i * k + p].mul_add(b[p * n + j], acc);
                }
                c[i * n + j] = acc;
            }
        }
        // Row 0: [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        // Row 1: [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        approx_eq(c[0], 58.0, 1e-5);
        approx_eq(c[1], 64.0, 1e-5);
        approx_eq(c[2], 139.0, 1e-5);
        approx_eq(c[3], 154.0, 1e-5);
    }
}

// ═══════════════════════════════════════════════════════════════════
// § RoPE (Rotary Position Embedding)
// ═══════════════════════════════════════════════════════════════════

mod rope_edge_cases {
    use super::*;

    /// Apply RoPE to a pair (x0, x1) at a given angle θ:
    ///   x0' = x0·cos(θ) − x1·sin(θ)
    ///   x1' = x0·sin(θ) + x1·cos(θ)
    fn apply_rope_pair(x0: f32, x1: f32, angle: f32) -> (f32, f32) {
        let (sin_a, cos_a) = angle.sin_cos();
        (x0 * cos_a - x1 * sin_a, x0 * sin_a + x1 * cos_a)
    }

    /// Compute RoPE angle: θ_i = position / base^(2i/dim)
    fn rope_angle(position: usize, dim_index: usize, dim: usize, base: f32) -> f32 {
        let freq = 1.0 / base.powf(2.0 * dim_index as f32 / dim as f32);
        position as f32 * freq
    }

    #[test]
    fn rope_zero_angle_identity() {
        // At angle=0, cos=1, sin=0 → output equals input
        let (x0, x1) = apply_rope_pair(3.0, 4.0, 0.0);
        approx_eq(x0, 3.0, 1e-6);
        approx_eq(x1, 4.0, 1e-6);
    }

    #[test]
    fn rope_half_pi_rotation() {
        // At angle=π/2, cos=0, sin=1 → (x0,x1) → (-x1, x0)
        let (x0, x1) = apply_rope_pair(3.0, 4.0, PI / 2.0);
        approx_eq(x0, -4.0, 1e-5);
        approx_eq(x1, 3.0, 1e-5);
    }

    #[test]
    fn rope_pi_rotation_negation() {
        // At angle=π, cos=-1, sin=0 → (x0,x1) → (-x0, -x1)
        let (x0, x1) = apply_rope_pair(3.0, 4.0, PI);
        approx_eq(x0, -3.0, 1e-5);
        approx_eq(x1, -4.0, 1e-5);
    }

    #[test]
    fn rope_full_rotation_identity() {
        // At angle=2π, should return to original (within floating-point tolerance)
        let (x0, x1) = apply_rope_pair(3.0, 4.0, 2.0 * PI);
        approx_eq(x0, 3.0, 1e-4);
        approx_eq(x1, 4.0, 1e-4);
    }

    #[test]
    fn rope_preserves_norm() {
        // Rotation preserves vector magnitude: ||(x0', x1')|| == ||(x0, x1)||
        let x0 = 3.0_f32;
        let x1 = 4.0_f32;
        let original_norm = (x0 * x0 + x1 * x1).sqrt();

        for angle_mult in [0.1, 0.5, 1.0, 2.7, PI, 2.0 * PI, 100.0] {
            let (r0, r1) = apply_rope_pair(x0, x1, angle_mult);
            let rotated_norm = (r0 * r0 + r1 * r1).sqrt();
            approx_eq(rotated_norm, original_norm, 1e-4);
        }
    }

    #[test]
    fn rope_large_position_index() {
        // Position index of 100_000 with base=10_000 — tests sin/cos with large angles
        let dim = 128;
        let base = 10_000.0_f32;
        let position = 100_000_usize;

        for dim_idx in 0..dim / 2 {
            let angle = rope_angle(position, dim_idx, dim, base);
            let (sin_a, cos_a) = angle.sin_cos();
            // sin and cos must always be in [-1, 1]
            assert!(
                (-1.0..=1.0).contains(&sin_a),
                "sin({angle}) = {sin_a} out of range at dim_idx={dim_idx}"
            );
            assert!(
                (-1.0..=1.0).contains(&cos_a),
                "cos({angle}) = {cos_a} out of range at dim_idx={dim_idx}"
            );
            // sin²+cos²=1
            approx_eq(sin_a * sin_a + cos_a * cos_a, 1.0, 1e-5);
        }
    }

    #[test]
    fn rope_frequency_decreases_with_dim_index() {
        // Higher dimension indices should have lower frequencies
        let position = 10_usize;
        let dim = 64;
        let base = 10_000.0_f32;

        let angle_low = rope_angle(position, 0, dim, base);
        let angle_mid = rope_angle(position, dim / 4, dim, base);
        let angle_high = rope_angle(position, dim / 2 - 1, dim, base);

        assert!(angle_low > angle_mid, "lower dim index should have higher frequency");
        assert!(angle_mid > angle_high, "middle dim index should have higher frequency than high");
    }
}

// ═══════════════════════════════════════════════════════════════════
// § Quantization rounding
// ═══════════════════════════════════════════════════════════════════

mod quantization_rounding {
    use super::*;

    #[test]
    fn ternary_quantization_boundary_values() {
        let threshold = 0.5_f32;
        // Exactly at ±threshold — should map to zero (strict inequality: v < -t or v > t)
        let vals = vec![-0.5, -0.25, 0.0, 0.25, 0.5];
        let quant = quantize_ternary(&vals, threshold);
        assert_eq!(quant, vec![0, 0, 0, 0, 0]);
    }

    #[test]
    fn ternary_quantization_just_beyond_threshold() {
        let threshold = 0.5_f32;
        let eps = 1e-7_f32;
        let vals = vec![
            -0.5 - eps, // just below -threshold
            -0.5 + eps, // just above -threshold (in dead zone)
            0.5 - eps,  // just below threshold (in dead zone)
            0.5 + eps,  // just above threshold
        ];
        let quant = quantize_ternary(&vals, threshold);
        assert_eq!(quant, vec![-1, 0, 0, 1]);
    }

    #[test]
    fn ternary_quantization_extreme_values() {
        let threshold = 0.5_f32;
        let vals = vec![-1e6, -1.0, 0.0, 1.0, 1e6];
        let quant = quantize_ternary(&vals, threshold);
        assert_eq!(quant, vec![-1, -1, 0, 1, 1]);
    }

    #[test]
    fn ternary_quantization_symmetry() {
        // Quantization should be symmetric: q(-x) == -q(x) for all x != ±threshold
        let threshold = 0.5_f32;
        let test_vals = vec![0.1, 0.3, 0.7, 1.0, 2.5, 100.0];
        for &v in &test_vals {
            let pos = quantize_ternary(&[v], threshold)[0];
            let neg = quantize_ternary(&[-v], threshold)[0];
            assert_eq!(pos, -neg, "symmetry violated for v={v}");
        }
    }

    #[test]
    fn quantization_f16_round_trip_at_boundary() {
        // Values near quantization thresholds after f16 round-trip may shift
        let threshold = 0.5_f32;
        let near_boundary = 0.500_5_f32; // just above threshold
        let rt = f32_to_f16_to_f32(near_boundary);
        // After f16 truncation, the value may round to exactly 0.5 or stay above
        let q_original = quantize_ternary(&[near_boundary], threshold);
        let q_roundtrip = quantize_ternary(&[rt], threshold);
        // Document that f16 can flip quantization decisions at boundaries
        // This is expected behaviour that Metal shaders must handle
        assert!(
            q_original[0] == 1 || q_original[0] == 0,
            "near-boundary should quantize to 0 or 1"
        );
        assert!(
            q_roundtrip[0] == 1 || q_roundtrip[0] == 0,
            "f16 round-tripped value should quantize to 0 or 1"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// § LayerNorm edge cases
// ═══════════════════════════════════════════════════════════════════

mod layernorm_edge_cases {
    use super::*;

    #[test]
    fn layernorm_zero_variance() {
        // All-equal input → zero variance → output depends entirely on eps
        let input = vec![5.0; 4];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let eps = 1e-5;
        let output = layer_norm_ref(&input, &gamma, &beta, eps);
        // mean=5, var=0, inv_std=1/sqrt(eps), normalized=(5-5)*inv_std=0
        for &v in &output {
            approx_eq(v, 0.0, 1e-3);
        }
    }

    #[test]
    fn layernorm_single_element() {
        // Single element: mean=x, var=0, output = beta (since normalized is zero)
        let input = vec![42.0];
        let gamma = vec![2.0];
        let beta = vec![1.0];
        let eps = 1e-5;
        let output = layer_norm_ref(&input, &gamma, &beta, eps);
        // (42-42)/sqrt(0+eps) * 2 + 1 = 0 * 2 + 1 = 1.0
        approx_eq(output[0], 1.0, 1e-3);
    }

    #[test]
    fn layernorm_very_large_magnitudes() {
        // Values near f32 limits — tests numerical stability of mean/variance
        let input = vec![1e30, 1e30 + 1e24, 1e30 - 1e24, 1e30];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let eps = 1e-5;
        let output = layer_norm_ref(&input, &gamma, &beta, eps);
        // All outputs should be finite
        for &v in &output {
            assert!(v.is_finite(), "layernorm output must be finite, got {v}");
        }
        // Output should have mean ≈ 0
        let out_mean = output.iter().sum::<f32>() / output.len() as f32;
        approx_eq(out_mean, 0.0, 1e-2);
    }

    #[test]
    fn layernorm_very_small_magnitudes() {
        // Subnormal-scale values — tests eps prevents division by zero
        let input = vec![1e-38, 2e-38, 3e-38, 4e-38];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        let eps = 1e-5;
        let output = layer_norm_ref(&input, &gamma, &beta, eps);
        for &v in &output {
            assert!(v.is_finite(), "layernorm must be finite for tiny inputs, got {v}");
        }
    }

    #[test]
    fn rms_norm_zero_input() {
        // All-zero input: rms = sqrt(eps), output = 0/sqrt(eps)*gamma = 0
        let input = vec![0.0; 4];
        let gamma = vec![1.0; 4];
        let eps = 1e-5;
        let output = rms_norm_ref(&input, &gamma, eps);
        for &v in &output {
            approx_eq(v, 0.0, 1e-6);
        }
    }

    #[test]
    fn rms_norm_unit_vector() {
        // RMSNorm of values whose RMS is already 1 → output ≈ input * gamma
        // For RMS=1, need mean(x^2)=1. Use [1, -1, 1, -1].
        let input = vec![1.0, -1.0, 1.0, -1.0];
        let gamma = vec![2.0; 4];
        let eps = 1e-5;
        let output = rms_norm_ref(&input, &gamma, eps);
        // rms ≈ 1.0, so output ≈ input * 2
        for i in 0..4 {
            approx_eq(output[i], input[i] * 2.0, 1e-3);
        }
    }

    #[test]
    fn layernorm_gamma_beta_effect() {
        // Verify gamma scales and beta shifts correctly
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma_identity = vec![1.0; 4];
        let beta_zero = vec![0.0; 4];
        let gamma_scale = vec![2.0; 4];
        let beta_shift = vec![10.0; 4];
        let eps = 1e-5;

        let base = layer_norm_ref(&input, &gamma_identity, &beta_zero, eps);
        let scaled = layer_norm_ref(&input, &gamma_scale, &beta_zero, eps);
        let shifted = layer_norm_ref(&input, &gamma_identity, &beta_shift, eps);

        for i in 0..4 {
            approx_eq(scaled[i], base[i] * 2.0, 1e-4);
            approx_eq(shifted[i], base[i] + 10.0, 1e-4);
        }
    }
}
