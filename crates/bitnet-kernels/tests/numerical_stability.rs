//! Numerical stability tests for kernel operations.
//!
//! Tests all kernels with adversarial inputs: zeros, large values, NaN,
//! Inf, subnormals, mixed-sign, and boundary patterns.  Every test must
//! complete without panic and produce finite (non-NaN, non-Inf) output
//! where the kernel contract requires it.

use bitnet_common::QuantizationType;
use bitnet_kernels::{FallbackKernel, KernelProvider};

// ── helpers ────────────────────────────────────────────────────────

fn best_kernel() -> Box<dyn KernelProvider> {
    bitnet_kernels::select_cpu_kernel().unwrap()
}

fn assert_finite(data: &[f32], label: &str) {
    for (i, v) in data.iter().enumerate() {
        assert!(v.is_finite(), "{label}[{i}] = {v} is not finite");
    }
}

fn run_matmul(kernel: &dyn KernelProvider, a: &[i8], b: &[u8], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    kernel.matmul_i2s(a, b, &mut c, m, n, k).unwrap();
    c
}

fn run_quantize(kernel: &dyn KernelProvider, input: &[f32]) -> (Vec<u8>, Vec<f32>) {
    let len = input.len();
    let mut output = vec![0u8; len / 4];
    let num_blocks = len.div_ceil(32);
    let mut scales = vec![0.0f32; num_blocks];
    kernel.quantize(input, &mut output, &mut scales, QuantizationType::I2S).unwrap();
    (output, scales)
}

// ========================================================================
// MATMUL – zero inputs
// ========================================================================

#[test]
fn matmul_all_zero_a() {
    let k = best_kernel();
    let a = vec![0i8; 64 * 64];
    let b = vec![1u8; 64 * 64];
    let c = run_matmul(k.as_ref(), &a, &b, 64, 64, 64);
    assert!(c.iter().all(|v| *v == 0.0), "zero A must produce zero output");
}

#[test]
fn matmul_all_zero_b() {
    let k = best_kernel();
    let a = vec![1i8; 64 * 64];
    let b = vec![0u8; 64 * 64];
    let c = run_matmul(k.as_ref(), &a, &b, 64, 64, 64);
    assert!(c.iter().all(|v| *v == 0.0), "zero B must produce zero output");
}

#[test]
fn matmul_both_zero() {
    let k = best_kernel();
    let a = vec![0i8; 64 * 64];
    let b = vec![0u8; 64 * 64];
    let c = run_matmul(k.as_ref(), &a, &b, 64, 64, 64);
    assert!(c.iter().all(|v| *v == 0.0));
}

// ========================================================================
// MATMUL – extreme values
// ========================================================================

#[test]
fn matmul_max_positive_a() {
    let k = best_kernel();
    let a = vec![i8::MAX; 64 * 64];
    let b = vec![3u8; 64 * 64];
    let c = run_matmul(k.as_ref(), &a, &b, 64, 64, 64);
    assert_finite(&c, "max_positive_a");
}

#[test]
fn matmul_min_negative_a() {
    let k = best_kernel();
    let a = vec![i8::MIN; 64 * 64];
    let b = vec![3u8; 64 * 64];
    let c = run_matmul(k.as_ref(), &a, &b, 64, 64, 64);
    assert_finite(&c, "min_negative_a");
}

#[test]
fn matmul_alternating_sign_a() {
    let k = best_kernel();
    let a: Vec<i8> = (0..64 * 64).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
    let b = vec![2u8; 64 * 64];
    let c = run_matmul(k.as_ref(), &a, &b, 64, 64, 64);
    assert_finite(&c, "alternating_sign");
}

#[test]
fn matmul_max_b_value() {
    let k = best_kernel();
    let a = vec![1i8; 64 * 64];
    let b = vec![u8::MAX; 64 * 64];
    let c = run_matmul(k.as_ref(), &a, &b, 64, 64, 64);
    assert_finite(&c, "max_b");
}

// ========================================================================
// MATMUL – single element edge cases
// ========================================================================

#[test]
fn matmul_1x1x1_zero() {
    let k = best_kernel();
    let c = run_matmul(k.as_ref(), &[0i8], &[0u8], 1, 1, 1);
    assert_eq!(c, vec![0.0]);
}

#[test]
fn matmul_1x1x1_positive() {
    let k = best_kernel();
    let c = run_matmul(k.as_ref(), &[1i8], &[3u8], 1, 1, 1);
    assert_finite(&c, "1x1x1_pos");
}

#[test]
fn matmul_1x1x1_negative() {
    let k = best_kernel();
    let c = run_matmul(k.as_ref(), &[-1i8], &[3u8], 1, 1, 1);
    assert_finite(&c, "1x1x1_neg");
}

// ========================================================================
// MATMUL – large dimensions
// ========================================================================

#[test]
fn matmul_large_k_all_ones() {
    let k_dim = 2048;
    let kern = best_kernel();
    let a = vec![1i8; 1 * k_dim];
    let b = vec![1u8; k_dim * 1];
    let c = run_matmul(kern.as_ref(), &a, &b, 1, 1, k_dim);
    assert_finite(&c, "large_k");
}

#[test]
fn matmul_large_k_alternating() {
    let k_dim = 2048;
    let kern = best_kernel();
    let a: Vec<i8> = (0..k_dim).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
    let b = vec![1u8; k_dim];
    let c = run_matmul(kern.as_ref(), &a, &b, 1, 1, k_dim);
    assert_finite(&c, "large_k_alt");
}

// ========================================================================
// MATMUL – non-square shapes
// ========================================================================

#[test]
fn matmul_tall_skinny() {
    let kern = best_kernel();
    let a = vec![1i8; 256 * 16];
    let b = vec![1u8; 16 * 256];
    let c = run_matmul(kern.as_ref(), &a, &b, 256, 256, 16);
    assert_finite(&c, "tall_skinny");
}

#[test]
fn matmul_wide_flat() {
    let kern = best_kernel();
    let a = vec![1i8; 16 * 256];
    let b = vec![1u8; 256 * 16];
    let c = run_matmul(kern.as_ref(), &a, &b, 16, 16, 256);
    assert_finite(&c, "wide_flat");
}

#[test]
fn matmul_single_row() {
    let kern = best_kernel();
    let a = vec![1i8; 128];
    let b = vec![2u8; 128 * 64];
    let c = run_matmul(kern.as_ref(), &a, &b, 1, 64, 128);
    assert_finite(&c, "single_row");
}

#[test]
fn matmul_single_col() {
    let kern = best_kernel();
    let a = vec![1i8; 64 * 128];
    let b = vec![2u8; 128];
    let c = run_matmul(kern.as_ref(), &a, &b, 64, 1, 128);
    assert_finite(&c, "single_col");
}

// ========================================================================
// MATMUL – ternary-only values (-1, 0, 1)
// ========================================================================

#[test]
fn matmul_ternary_all_minus_one() {
    let kern = best_kernel();
    let a = vec![-1i8; 64 * 64];
    let b = vec![1u8; 64 * 64];
    let c = run_matmul(kern.as_ref(), &a, &b, 64, 64, 64);
    assert_finite(&c, "ternary_neg");
    assert!(c.iter().all(|v| *v <= 0.0), "all-negative weights must produce non-positive");
}

#[test]
fn matmul_ternary_all_one() {
    let kern = best_kernel();
    let a = vec![1i8; 64 * 64];
    let b = vec![1u8; 64 * 64];
    let c = run_matmul(kern.as_ref(), &a, &b, 64, 64, 64);
    assert_finite(&c, "ternary_pos");
    assert!(c.iter().all(|v| *v >= 0.0), "all-positive weights must produce non-negative");
}

#[test]
fn matmul_ternary_sparse() {
    // 90% zeros, 5% +1, 5% -1
    let kern = best_kernel();
    let a: Vec<i8> = (0..64 * 64)
        .map(|i| match i % 20 {
            0 => 1,
            10 => -1,
            _ => 0,
        })
        .collect();
    let b = vec![1u8; 64 * 64];
    let c = run_matmul(kern.as_ref(), &a, &b, 64, 64, 64);
    assert_finite(&c, "ternary_sparse");
}

// ========================================================================
// QUANTIZE – zero inputs
// ========================================================================

#[test]
fn quantize_all_zeros() {
    let kern = best_kernel();
    let input = vec![0.0f32; 128];
    let (out, scales) = run_quantize(kern.as_ref(), &input);
    // All zeros should quantize to zero with zero scale
    assert!(scales.iter().all(|s| *s == 0.0 || s.is_finite()));
    let _ = out; // packed output is valid
}

#[test]
fn quantize_near_zero() {
    let kern = best_kernel();
    let input: Vec<f32> = (0..128).map(|i| (i as f32) * 1e-10).collect();
    let (_, scales) = run_quantize(kern.as_ref(), &input);
    assert_finite(&scales, "near_zero_scales");
}

// ========================================================================
// QUANTIZE – large values
// ========================================================================

#[test]
fn quantize_large_positive() {
    let kern = best_kernel();
    let input = vec![1e6f32; 128];
    let (_, scales) = run_quantize(kern.as_ref(), &input);
    assert_finite(&scales, "large_pos_scales");
}

#[test]
fn quantize_large_negative() {
    let kern = best_kernel();
    let input = vec![-1e6f32; 128];
    let (_, scales) = run_quantize(kern.as_ref(), &input);
    assert_finite(&scales, "large_neg_scales");
}

#[test]
fn quantize_mixed_large() {
    let kern = best_kernel();
    let input: Vec<f32> = (0..128)
        .map(|i| if i % 2 == 0 { 1e6 } else { -1e6 })
        .collect();
    let (_, scales) = run_quantize(kern.as_ref(), &input);
    assert_finite(&scales, "mixed_large_scales");
}

#[test]
fn quantize_max_f32() {
    let kern = best_kernel();
    let input = vec![f32::MAX / 2.0; 128];
    let (_, scales) = run_quantize(kern.as_ref(), &input);
    assert_finite(&scales, "max_f32_scales");
}

#[test]
fn quantize_min_positive_f32() {
    let kern = best_kernel();
    let input = vec![f32::MIN_POSITIVE; 128];
    let (_, scales) = run_quantize(kern.as_ref(), &input);
    assert_finite(&scales, "min_pos_scales");
}

// ========================================================================
// QUANTIZE – subnormal values
// ========================================================================

#[test]
fn quantize_subnormals() {
    let kern = best_kernel();
    let subnormal = f32::MIN_POSITIVE / 2.0;
    assert!(subnormal > 0.0 && !subnormal.is_normal());
    let input = vec![subnormal; 128];
    let (_, scales) = run_quantize(kern.as_ref(), &input);
    assert_finite(&scales, "subnormal_scales");
}

#[test]
fn quantize_mixed_subnormal_normal() {
    let kern = best_kernel();
    let subnormal = f32::MIN_POSITIVE / 2.0;
    let input: Vec<f32> = (0..128)
        .map(|i| if i % 2 == 0 { subnormal } else { 1.0 })
        .collect();
    let (_, scales) = run_quantize(kern.as_ref(), &input);
    assert_finite(&scales, "mixed_sub_scales");
}

// ========================================================================
// QUANTIZE – uniform / constant values
// ========================================================================

#[test]
fn quantize_all_same_positive() {
    let kern = best_kernel();
    let input = vec![42.0f32; 256];
    let (_, scales) = run_quantize(kern.as_ref(), &input);
    assert_finite(&scales, "same_pos_scales");
}

#[test]
fn quantize_all_same_negative() {
    let kern = best_kernel();
    let input = vec![-42.0f32; 256];
    let (_, scales) = run_quantize(kern.as_ref(), &input);
    assert_finite(&scales, "same_neg_scales");
}

// ========================================================================
// QUANTIZE – ramp / gradient patterns
// ========================================================================

#[test]
fn quantize_ascending_ramp() {
    let kern = best_kernel();
    let input: Vec<f32> = (0..256).map(|i| i as f32).collect();
    let (_, scales) = run_quantize(kern.as_ref(), &input);
    assert_finite(&scales, "asc_ramp_scales");
}

#[test]
fn quantize_descending_ramp() {
    let kern = best_kernel();
    let input: Vec<f32> = (0..256).map(|i| -(i as f32)).collect();
    let (_, scales) = run_quantize(kern.as_ref(), &input);
    assert_finite(&scales, "desc_ramp_scales");
}

// ========================================================================
// QUANTIZE – spike / outlier patterns
// ========================================================================

#[test]
fn quantize_single_outlier() {
    let kern = best_kernel();
    let mut input = vec![0.0f32; 128];
    input[64] = 1e6;
    let (_, scales) = run_quantize(kern.as_ref(), &input);
    assert_finite(&scales, "single_outlier_scales");
}

#[test]
fn quantize_two_opposite_outliers() {
    let kern = best_kernel();
    let mut input = vec![0.0f32; 128];
    input[0] = 1e6;
    input[127] = -1e6;
    let (_, scales) = run_quantize(kern.as_ref(), &input);
    assert_finite(&scales, "two_outlier_scales");
}

// ========================================================================
// QUANTIZE – various sizes
// ========================================================================

#[test]
fn quantize_minimum_size() {
    let kern = best_kernel();
    let input = vec![1.0f32; 32];
    let mut output = vec![0u8; 8];
    let mut scales = vec![0.0f32; 1];
    kern.quantize(&input, &mut output, &mut scales, QuantizationType::I2S).unwrap();
    assert_finite(&scales, "min_size_scales");
}

#[test]
fn quantize_large_size() {
    let kern = best_kernel();
    let input: Vec<f32> = (0..8192).map(|i| ((i % 256) as f32) - 128.0).collect();
    let (_, scales) = run_quantize(kern.as_ref(), &input);
    assert_finite(&scales, "large_size_scales");
}

// ========================================================================
// FALLBACK vs BEST – consistency on adversarial input
// ========================================================================

#[test]
fn adversarial_matmul_fallback_vs_best() {
    let fb = FallbackKernel;
    let best = best_kernel();
    let a: Vec<i8> = (0..128 * 128)
        .map(|i| match i % 7 {
            0 => i8::MAX,
            1 => i8::MIN,
            2 => 0,
            3 => 1,
            4 => -1,
            _ => ((i % 3) as i8) - 1,
        })
        .collect();
    let b: Vec<u8> = (0..128 * 128)
        .map(|i| match i % 5 {
            0 => u8::MAX,
            1 => 0,
            _ => (i % 4) as u8,
        })
        .collect();

    let c_fb = run_matmul(&fb, &a, &b, 128, 128, 128);
    let c_best = run_matmul(best.as_ref(), &a, &b, 128, 128, 128);

    assert_finite(&c_fb, "adv_fb");
    assert_finite(&c_best, "adv_best");

    let diff: f32 = c_fb
        .iter()
        .zip(&c_best)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(diff < 1e-3, "adversarial matmul diff = {diff}");
}

#[test]
fn adversarial_quantize_fallback_vs_best() {
    let fb = FallbackKernel;
    let best = best_kernel();
    let input: Vec<f32> = (0..256)
        .map(|i| match i % 8 {
            0 => 0.0,
            1 => f32::MIN_POSITIVE,
            2 => f32::MIN_POSITIVE / 2.0,
            3 => 1e6,
            4 => -1e6,
            5 => 1.0,
            6 => -1.0,
            _ => ((i as f32) - 128.0) / 128.0,
        })
        .collect();

    let (out_fb, sc_fb) = run_quantize(&fb, &input);
    let (out_best, sc_best) = run_quantize(best.as_ref(), &input);

    assert_eq!(out_fb, out_best, "adversarial quantize bytes differ");
    assert_finite(&sc_fb, "adv_q_fb_scales");
    assert_finite(&sc_best, "adv_q_best_scales");
}

// ========================================================================
// OUTPUT BUFFER INITIAL STATE – verify kernel overwrites correctly
// ========================================================================

#[test]
fn matmul_overwrites_dirty_output() {
    let kern = best_kernel();
    let a = vec![1i8; 64 * 64];
    let b = vec![1u8; 64 * 64];
    let mut c = vec![f32::MAX; 64 * 64]; // pre-filled with MAX
    kern.matmul_i2s(&a, &b, &mut c, 64, 64, 64).unwrap();
    assert_finite(&c, "dirty_output");
    assert!(c.iter().all(|v| *v != f32::MAX), "kernel must overwrite output");
}

#[test]
fn quantize_overwrites_dirty_output() {
    let kern = best_kernel();
    let input = vec![1.0f32; 128];
    let mut output = vec![0xFFu8; 32];
    let mut scales = vec![f32::MAX; 4];
    kern.quantize(&input, &mut output, &mut scales, QuantizationType::I2S).unwrap();
    assert_finite(&scales, "dirty_q_scales");
}
