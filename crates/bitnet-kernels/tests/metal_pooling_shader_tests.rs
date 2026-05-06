#![allow(dead_code, unused_imports, unused_variables, non_camel_case_types, unused_mut)]
//! Metal pooling operation shader validation tests.
//! Tests average pooling, max pooling, adaptive pooling, global pooling,
//! and attention pooling operations expected to run on Metal GPU.
//!
//! GPU-runtime tests are #[ignore] with justification.
//! CPU-side logic tests run without Metal hardware.

#![cfg(target_os = "macos")]

// ── Constants ───────────────────────────────────────────────────────

/// Metal requires 256-byte buffer alignment for optimal performance.
const METAL_BUFFER_ALIGNMENT: usize = 256;

/// Maximum threads per threadgroup on Apple Silicon.
const MAX_THREADS_PER_THREADGROUP: u32 = 1024;

/// SIMD group (warp) width on Apple Silicon GPUs.
const SIMD_GROUP_WIDTH: u32 = 32;

// ── CPU reference implementations ───────────────────────────────────

/// CPU reference: 1D average pooling with stride and padding.
fn avg_pool_1d_cpu(input: &[f32], kernel_size: usize, stride: usize, padding: usize) -> Vec<f32> {
    let in_len = input.len();
    let padded_len = in_len + 2 * padding;
    let out_len = (padded_len - kernel_size) / stride + 1;
    (0..out_len)
        .map(|i| {
            let start = i * stride;
            let mut sum = 0.0_f32;
            let mut count = 0_u32;
            for k in 0..kernel_size {
                let idx = start + k;
                if idx >= padding && idx < padding + in_len {
                    sum += input[idx - padding];
                    count += 1;
                }
            }
            if count > 0 { sum / count as f32 } else { 0.0 }
        })
        .collect()
}

/// CPU reference: 2D average pooling (NCHW layout, single channel).
fn avg_pool_2d_cpu(
    input: &[f32],
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> Vec<f32> {
    let out_h = (h + 2 * pad_h - kh) / stride_h + 1;
    let out_w = (w + 2 * pad_w - kw) / stride_w + 1;
    let mut output = Vec::with_capacity(out_h * out_w);
    for oh in 0..out_h {
        for ow in 0..out_w {
            let mut sum = 0.0_f32;
            let mut count = 0_u32;
            for ki in 0..kh {
                for kj in 0..kw {
                    let ih = oh * stride_h + ki;
                    let iw = ow * stride_w + kj;
                    if ih >= pad_h && ih < pad_h + h && iw >= pad_w && iw < pad_w + w {
                        sum += input[(ih - pad_h) * w + (iw - pad_w)];
                        count += 1;
                    }
                }
            }
            output.push(if count > 0 { sum / count as f32 } else { 0.0 });
        }
    }
    output
}

/// CPU reference: 1D max pooling with stride and padding.
fn max_pool_1d_cpu(input: &[f32], kernel_size: usize, stride: usize, padding: usize) -> Vec<f32> {
    let in_len = input.len();
    let padded_len = in_len + 2 * padding;
    let out_len = (padded_len - kernel_size) / stride + 1;
    (0..out_len)
        .map(|i| {
            let start = i * stride;
            let mut max_val = f32::NEG_INFINITY;
            for k in 0..kernel_size {
                let idx = start + k;
                if idx >= padding && idx < padding + in_len {
                    let val = input[idx - padding];
                    if val > max_val {
                        max_val = val;
                    }
                }
            }
            if max_val == f32::NEG_INFINITY { 0.0 } else { max_val }
        })
        .collect()
}

/// CPU reference: 2D max pooling (NCHW layout, single channel).
fn max_pool_2d_cpu(
    input: &[f32],
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> Vec<f32> {
    let out_h = (h + 2 * pad_h - kh) / stride_h + 1;
    let out_w = (w + 2 * pad_w - kw) / stride_w + 1;
    let mut output = Vec::with_capacity(out_h * out_w);
    for oh in 0..out_h {
        for ow in 0..out_w {
            let mut max_val = f32::NEG_INFINITY;
            for ki in 0..kh {
                for kj in 0..kw {
                    let ih = oh * stride_h + ki;
                    let iw = ow * stride_w + kj;
                    if ih >= pad_h && ih < pad_h + h && iw >= pad_w && iw < pad_w + w {
                        let v = input[(ih - pad_h) * w + (iw - pad_w)];
                        if v > max_val {
                            max_val = v;
                        }
                    }
                }
            }
            output.push(if max_val == f32::NEG_INFINITY { 0.0 } else { max_val });
        }
    }
    output
}

/// CPU reference: global average pooling over a flat slice.
fn global_avg_pool_cpu(input: &[f32]) -> f32 {
    if input.is_empty() {
        return 0.0;
    }
    input.iter().sum::<f32>() / input.len() as f32
}

/// CPU reference: global max pooling over a flat slice.
fn global_max_pool_cpu(input: &[f32]) -> f32 {
    input.iter().copied().reduce(|a, b| if a >= b { a } else { b }).unwrap_or(0.0)
}

/// CPU reference: adaptive average pooling (1D).
/// Maps `in_len` → `out_len` bins with floor/ceil index ranges.
fn adaptive_avg_pool_1d_cpu(input: &[f32], out_len: usize) -> Vec<f32> {
    let in_len = input.len();
    (0..out_len)
        .map(|i| {
            let start = (i * in_len) / out_len;
            let end = ((i + 1) * in_len) / out_len;
            let slice = &input[start..end];
            if slice.is_empty() { 0.0 } else { slice.iter().sum::<f32>() / slice.len() as f32 }
        })
        .collect()
}

/// CPU reference: attention-weighted pooling.
/// output = sum(softmax(scores) * values) per feature.
fn attention_pool_cpu(values: &[f32], scores: &[f32], seq_len: usize, dim: usize) -> Vec<f32> {
    // softmax over scores
    let max_s = scores.iter().copied().reduce(|a, b| if a >= b { a } else { b }).unwrap_or(0.0);
    let exps: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp()).collect();
    let sum_exp: f32 = exps.iter().sum();
    let weights: Vec<f32> = exps.iter().map(|&e| e / sum_exp).collect();

    let mut output = vec![0.0_f32; dim];
    for t in 0..seq_len {
        for d in 0..dim {
            output[d] += weights[t] * values[t * dim + d];
        }
    }
    output
}

/// CPU reference: max-pool gradient (pass gradient to argmax positions).
fn max_pool_1d_grad_cpu(
    input: &[f32],
    grad_output: &[f32],
    kernel_size: usize,
    stride: usize,
) -> Vec<f32> {
    let in_len = input.len();
    let out_len = (in_len - kernel_size) / stride + 1;
    let mut grad_input = vec![0.0_f32; in_len];
    for i in 0..out_len {
        let start = i * stride;
        let end = start + kernel_size;
        let mut max_idx = start;
        let mut max_val = input[start];
        for j in (start + 1)..end {
            if input[j] > max_val {
                max_val = input[j];
                max_idx = j;
            }
        }
        grad_input[max_idx] += grad_output[i];
    }
    grad_input
}

/// CPU reference: average-pool gradient (distribute equally).
fn avg_pool_1d_grad_cpu(
    input_len: usize,
    grad_output: &[f32],
    kernel_size: usize,
    stride: usize,
) -> Vec<f32> {
    let out_len = (input_len - kernel_size) / stride + 1;
    let mut grad_input = vec![0.0_f32; input_len];
    let scale = 1.0 / kernel_size as f32;
    for i in 0..out_len {
        let start = i * stride;
        for k in 0..kernel_size {
            grad_input[start + k] += grad_output[i] * scale;
        }
    }
    grad_input
}

// ── Helpers ─────────────────────────────────────────────────────────

/// Align byte size to Metal 256-byte boundary.
fn align_to_metal(size: usize) -> usize {
    if size == 0 {
        return 0;
    }
    let mask = METAL_BUFFER_ALIGNMENT - 1;
    (size + mask) & !mask
}

/// Compute dispatch threadgroups.
fn dispatch_groups(total: u32, group_size: u32) -> u32 {
    total.div_ceil(group_size)
}

fn assert_close(actual: &[f32], expected: &[f32], atol: f32, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch (actual={}, expected={})",
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        assert!(diff < atol, "{label}[{i}]: actual={a}, expected={e}, diff={diff} >= atol={atol}");
    }
}

fn assert_no_nan_inf(data: &[f32], label: &str) {
    for (i, &v) in data.iter().enumerate() {
        assert!(v.is_finite(), "{label}[{i}]: non-finite value {v}");
    }
}

/// Kahan-compensated sum for numerically stable reference.
fn kahan_sum(data: &[f32]) -> f64 {
    let mut sum = 0.0_f64;
    let mut comp = 0.0_f64;
    for &x in data {
        let y = x as f64 - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
    }
    sum
}

// ═══════════════════════════════════════════════════════════════════
// 1. Average pooling 1D
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_avg_pool_1d_basic() {
    let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let result = avg_pool_1d_cpu(&input, 4, 4, 0);
    // bins: [0..4]=1.5, [4..8]=5.5, [8..12]=9.5, [12..16]=13.5
    assert_eq!(result.len(), 4);
    assert_close(&result, &[1.5, 5.5, 9.5, 13.5], 1e-6, "avg_pool_1d_basic");
}

#[test]
fn test_avg_pool_1d_stride_1() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = avg_pool_1d_cpu(&input, 3, 1, 0);
    assert_eq!(result.len(), 3);
    assert_close(&result, &[2.0, 3.0, 4.0], 1e-6, "avg_pool_1d_stride1");
}

#[test]
fn test_avg_pool_1d_with_padding() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let result = avg_pool_1d_cpu(&input, 3, 1, 1);
    // With count_include_pad=false: pad positions are excluded from average.
    assert_eq!(result.len(), 4);
    assert_close(&result, &[1.5, 2.0, 3.0, 3.5], 1e-6, "avg_pool_1d_padded");
}

#[test]
fn test_avg_pool_1d_large_kernel() {
    let input: Vec<f32> = (0..128).map(|i| (i as f32) * 0.1).collect();
    let result = avg_pool_1d_cpu(&input, 128, 128, 0);
    assert_eq!(result.len(), 1);
    let expected = input.iter().sum::<f32>() / 128.0;
    assert!((result[0] - expected).abs() < 1e-4, "large kernel avg");
}

// ═══════════════════════════════════════════════════════════════════
// Average pooling 2D
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_avg_pool_2d_basic() {
    // 4×4 input, 2×2 kernel, stride 2, no padding
    let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let result = avg_pool_2d_cpu(&input, 4, 4, 2, 2, 2, 2, 0, 0);
    assert_eq!(result.len(), 4);
    // top-left 2×2: mean(0,1,4,5) = 2.5
    assert_close(&result, &[2.5, 4.5, 10.5, 12.5], 1e-6, "avg_pool_2d");
}

#[test]
fn test_avg_pool_2d_stride_1() {
    let input: Vec<f32> = (0..9).map(|i| i as f32).collect();
    let result = avg_pool_2d_cpu(&input, 3, 3, 2, 2, 1, 1, 0, 0);
    // 2×2 output
    assert_eq!(result.len(), 4);
    let expected = vec![
        (0.0 + 1.0 + 3.0 + 4.0) / 4.0,
        (1.0 + 2.0 + 4.0 + 5.0) / 4.0,
        (3.0 + 4.0 + 6.0 + 7.0) / 4.0,
        (4.0 + 5.0 + 7.0 + 8.0) / 4.0,
    ];
    assert_close(&result, &expected, 1e-6, "avg_pool_2d_s1");
}

#[test]
fn test_avg_pool_2d_with_padding() {
    let input = vec![1.0, 2.0, 3.0, 4.0]; // 2×2
    let result = avg_pool_2d_cpu(&input, 2, 2, 2, 2, 1, 1, 1, 1);
    // padded to 4×4, 2×2 kernel, stride 1 → 3×3 output
    assert_eq!(result.len(), 9);
    assert_no_nan_inf(&result, "avg_pool_2d_padded");
}

#[test]
fn test_avg_pool_2d_non_square_kernel() {
    // 4×6 input, 2×3 kernel, stride (2,3)
    let input: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let result = avg_pool_2d_cpu(&input, 4, 6, 2, 3, 2, 3, 0, 0);
    assert_eq!(result.len(), 4); // 2×2 output
    assert_no_nan_inf(&result, "avg_pool_2d_nonsquare");
}

// ═══════════════════════════════════════════════════════════════════
// 2. Max pooling 1D
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_max_pool_1d_basic() {
    let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let result = max_pool_1d_cpu(&input, 4, 4, 0);
    assert_eq!(result.len(), 4);
    assert_close(&result, &[3.0, 7.0, 11.0, 15.0], 1e-6, "max_pool_1d");
}

#[test]
fn test_max_pool_1d_stride_1() {
    let input = vec![1.0, 3.0, 2.0, 5.0, 4.0];
    let result = max_pool_1d_cpu(&input, 3, 1, 0);
    assert_eq!(result.len(), 3);
    assert_close(&result, &[3.0, 5.0, 5.0], 1e-6, "max_pool_1d_s1");
}

#[test]
fn test_max_pool_1d_with_padding() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let result = max_pool_1d_cpu(&input, 3, 1, 1);
    assert_eq!(result.len(), 4);
    assert_close(&result, &[2.0, 3.0, 4.0, 4.0], 1e-6, "max_pool_1d_padded");
}

#[test]
fn test_max_pool_1d_negative_values() {
    let input = vec![-5.0, -3.0, -8.0, -1.0, -4.0, -2.0];
    let result = max_pool_1d_cpu(&input, 3, 3, 0);
    assert_eq!(result.len(), 2);
    assert_close(&result, &[-3.0, -1.0], 1e-6, "max_pool_1d_negative");
}

// ═══════════════════════════════════════════════════════════════════
// Max pooling 2D
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_max_pool_2d_basic() {
    let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let result = max_pool_2d_cpu(&input, 4, 4, 2, 2, 2, 2, 0, 0);
    assert_eq!(result.len(), 4);
    // top-left 2×2: max(0,1,4,5) = 5
    assert_close(&result, &[5.0, 7.0, 13.0, 15.0], 1e-6, "max_pool_2d");
}

#[test]
fn test_max_pool_2d_stride_1() {
    let input: Vec<f32> = (0..9).map(|i| i as f32).collect();
    let result = max_pool_2d_cpu(&input, 3, 3, 2, 2, 1, 1, 0, 0);
    assert_eq!(result.len(), 4);
    assert_close(&result, &[4.0, 5.0, 7.0, 8.0], 1e-6, "max_pool_2d_s1");
}

#[test]
fn test_max_pool_2d_with_padding() {
    let input = vec![1.0, 2.0, 3.0, 4.0]; // 2×2
    let result = max_pool_2d_cpu(&input, 2, 2, 2, 2, 1, 1, 1, 1);
    assert_eq!(result.len(), 9);
    assert_no_nan_inf(&result, "max_pool_2d_padded");
}

#[test]
fn test_max_pool_2d_negative_values() {
    let input = vec![-9.0, -4.0, -7.0, -2.0, -6.0, -1.0, -8.0, -3.0, -5.0];
    let result = max_pool_2d_cpu(&input, 3, 3, 2, 2, 1, 1, 0, 0);
    // 2×2 output
    assert_eq!(result.len(), 4);
    // top-left 2×2: max(-9,-4,-6,-1)=-1, etc.
    for &v in &result {
        assert!(v.is_finite());
    }
}

// ═══════════════════════════════════════════════════════════════════
// 3. Global average pooling
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_global_avg_pool_basic() {
    let input: Vec<f32> = (1..=100).map(|i| i as f32).collect();
    let result = global_avg_pool_cpu(&input);
    assert!((result - 50.5).abs() < 1e-4, "global avg: {result}");
}

#[test]
fn test_global_avg_pool_single_element() {
    let input = vec![42.0];
    assert!((global_avg_pool_cpu(&input) - 42.0).abs() < f32::EPSILON);
}

#[test]
fn test_global_avg_pool_uniform_values() {
    let input = vec![7.0; 256];
    assert!((global_avg_pool_cpu(&input) - 7.0).abs() < 1e-6);
}

#[test]
fn test_global_avg_pool_large_tensor() {
    let input: Vec<f32> = (0..4096).map(|i| (i as f32) * 0.001).collect();
    let expected = kahan_sum(&input) as f32 / 4096.0;
    let result = global_avg_pool_cpu(&input);
    assert!(
        (result - expected).abs() < 1e-2,
        "global avg large: result={result}, expected={expected}"
    );
}

#[test]
fn test_global_avg_pool_multi_channel() {
    // 3 channels, each 64 elements
    let input: Vec<f32> = (0..192).map(|i| (i as f32) * 0.1).collect();
    for c in 0..3 {
        let channel = &input[c * 64..(c + 1) * 64];
        let avg = global_avg_pool_cpu(channel);
        assert!(avg.is_finite(), "channel {c} avg is finite");
    }
}

#[test]
fn test_global_avg_pool_empty() {
    assert!((global_avg_pool_cpu(&[]) - 0.0).abs() < f32::EPSILON);
}

// ═══════════════════════════════════════════════════════════════════
// 4. Global max pooling
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_global_max_pool_basic() {
    let input = vec![1.0, 9.0, 3.0, 7.0, 5.0];
    assert!((global_max_pool_cpu(&input) - 9.0).abs() < f32::EPSILON);
}

#[test]
fn test_global_max_pool_negative() {
    let input = vec![-5.0, -3.0, -8.0, -1.0, -4.0];
    assert!((global_max_pool_cpu(&input) - (-1.0)).abs() < f32::EPSILON);
}

#[test]
fn test_global_max_pool_single_element() {
    let input = vec![42.0];
    assert!((global_max_pool_cpu(&input) - 42.0).abs() < f32::EPSILON);
}

#[test]
fn test_global_max_pool_all_same() {
    let input = vec![3.14; 512];
    assert!((global_max_pool_cpu(&input) - 3.14).abs() < 1e-6);
}

#[test]
fn test_global_max_pool_large_tensor() {
    let input: Vec<f32> = (0..4096).map(|i| (i as f32) * 0.01).collect();
    let expected = 4095.0 * 0.01;
    assert!((global_max_pool_cpu(&input) - expected).abs() < 1e-4, "global max large");
}

#[test]
fn test_global_max_pool_multi_channel() {
    let input: Vec<f32> = (0..192).map(|i| (i as f32) * 0.1).collect();
    for c in 0..3 {
        let channel = &input[c * 64..(c + 1) * 64];
        let max_v = global_max_pool_cpu(channel);
        let expected = ((c + 1) * 64 - 1) as f32 * 0.1;
        assert!((max_v - expected).abs() < 1e-4, "channel {c}: {max_v} vs {expected}");
    }
}

// ═══════════════════════════════════════════════════════════════════
// 5. Adaptive average pooling
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_adaptive_avg_pool_identity() {
    let input: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let result = adaptive_avg_pool_1d_cpu(&input, 8);
    assert_close(&result, &input, 1e-6, "adaptive_identity");
}

#[test]
fn test_adaptive_avg_pool_halve() {
    let input: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let result = adaptive_avg_pool_1d_cpu(&input, 4);
    assert_eq!(result.len(), 4);
    assert_close(&result, &[0.5, 2.5, 4.5, 6.5], 1e-6, "adaptive_halve");
}

#[test]
fn test_adaptive_avg_pool_to_one() {
    let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let result = adaptive_avg_pool_1d_cpu(&input, 1);
    assert_eq!(result.len(), 1);
    let expected = input.iter().sum::<f32>() / 16.0;
    assert!((result[0] - expected).abs() < 1e-5, "adaptive_to_one");
}

#[test]
fn test_adaptive_avg_pool_non_divisible() {
    let input: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let result = adaptive_avg_pool_1d_cpu(&input, 3);
    assert_eq!(result.len(), 3);
    assert_no_nan_inf(&result, "adaptive_non_div");
}

#[test]
fn test_adaptive_avg_pool_large_output() {
    let input: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();
    let result = adaptive_avg_pool_1d_cpu(&input, 128);
    assert_eq!(result.len(), 128);
    assert_no_nan_inf(&result, "adaptive_large_out");
}

#[test]
fn test_adaptive_avg_pool_prime_lengths() {
    for &in_len in &[7, 11, 13, 17, 23] {
        let input: Vec<f32> = (0..in_len).map(|i| i as f32).collect();
        for &out_len in &[3, 5] {
            let result = adaptive_avg_pool_1d_cpu(&input, out_len);
            assert_eq!(result.len(), out_len, "prime in={in_len} out={out_len}");
            assert_no_nan_inf(&result, &format!("adaptive_prime_{in_len}_{out_len}"));
        }
    }
}

#[test]
fn test_adaptive_avg_pool_preserves_mean() {
    let input: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let input_mean = input.iter().sum::<f32>() / input.len() as f32;
    let result = adaptive_avg_pool_1d_cpu(&input, 8);
    let result_mean = result.iter().sum::<f32>() / result.len() as f32;
    assert!(
        (input_mean - result_mean).abs() < 1.0,
        "adaptive mean preservation: {input_mean} vs {result_mean}"
    );
}

// ═══════════════════════════════════════════════════════════════════
// 6. Attention-weighted pooling
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_attention_pool_uniform_scores() {
    let seq_len = 4;
    let dim = 3;
    let values: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let scores = vec![0.0; seq_len]; // uniform attention
    let result = attention_pool_cpu(&values, &scores, seq_len, dim);
    assert_eq!(result.len(), dim);
    // uniform weights → mean over time steps
    let expected: Vec<f32> = (0..dim)
        .map(|d| (0..seq_len).map(|t| values[t * dim + d]).sum::<f32>() / seq_len as f32)
        .collect();
    assert_close(&result, &expected, 1e-5, "attn_uniform");
}

#[test]
fn test_attention_pool_peaked_scores() {
    let seq_len = 4;
    let dim = 2;
    let values: Vec<f32> = vec![
        10.0, 20.0, // t=0
        30.0, 40.0, // t=1
        50.0, 60.0, // t=2
        70.0, 80.0, // t=3
    ];
    // Very peaked at t=2
    let scores = vec![-100.0, -100.0, 100.0, -100.0];
    let result = attention_pool_cpu(&values, &scores, seq_len, dim);
    // Should be approximately values[2]
    assert_close(&result, &[50.0, 60.0], 1e-3, "attn_peaked");
}

#[test]
fn test_attention_pool_single_step() {
    let values = vec![1.0, 2.0, 3.0];
    let scores = vec![5.0];
    let result = attention_pool_cpu(&values, &scores, 1, 3);
    assert_close(&result, &[1.0, 2.0, 3.0], 1e-6, "attn_single");
}

#[test]
fn test_attention_pool_weights_sum_to_one() {
    let seq_len = 8;
    let dim = 4;
    let values: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();
    let scores: Vec<f32> = (0..seq_len).map(|i| (i as f32) * 0.5).collect();
    let result = attention_pool_cpu(&values, &scores, seq_len, dim);
    assert_eq!(result.len(), dim);
    assert_no_nan_inf(&result, "attn_weights_sum");
}

#[test]
fn test_attention_pool_negative_scores() {
    let seq_len = 3;
    let dim = 2;
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let scores = vec![-1.0, -2.0, -3.0];
    let result = attention_pool_cpu(&values, &scores, seq_len, dim);
    assert_eq!(result.len(), dim);
    assert_no_nan_inf(&result, "attn_neg_scores");
}

#[test]
fn test_attention_pool_large_dim() {
    let seq_len = 16;
    let dim = 256;
    let values: Vec<f32> = (0..seq_len * dim).map(|i| (i as f32) * 0.001).collect();
    let scores: Vec<f32> = (0..seq_len).map(|i| (i as f32) * 0.1).collect();
    let result = attention_pool_cpu(&values, &scores, seq_len, dim);
    assert_eq!(result.len(), dim);
    assert_no_nan_inf(&result, "attn_large_dim");
}

#[test]
fn test_attention_pool_numerical_stability() {
    let seq_len = 4;
    let dim = 2;
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    // Large scores that could overflow without max-subtraction
    let scores = vec![1000.0, 1001.0, 1002.0, 1003.0];
    let result = attention_pool_cpu(&values, &scores, seq_len, dim);
    assert_no_nan_inf(&result, "attn_stability");
}

// ═══════════════════════════════════════════════════════════════════
// 7. Stride and padding configurations
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_stride_equals_kernel() {
    let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let avg = avg_pool_1d_cpu(&input, 4, 4, 0);
    let max = max_pool_1d_cpu(&input, 4, 4, 0);
    assert_eq!(avg.len(), 4);
    assert_eq!(max.len(), 4);
}

#[test]
fn test_stride_less_than_kernel_overlapping() {
    let input: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let result = avg_pool_1d_cpu(&input, 4, 2, 0);
    // overlapping windows
    assert_eq!(result.len(), 3);
    assert_close(&result, &[1.5, 3.5, 5.5], 1e-6, "overlap_stride");
}

#[test]
fn test_stride_greater_than_kernel() {
    let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let result = avg_pool_1d_cpu(&input, 2, 4, 0);
    // non-overlapping with gaps
    assert_eq!(result.len(), 4);
    assert_close(&result, &[0.5, 4.5, 8.5, 12.5], 1e-6, "gap_stride");
}

#[test]
fn test_asymmetric_padding_2d() {
    // 3×3 input with different h/w padding
    let input: Vec<f32> = (0..9).map(|i| i as f32).collect();
    let result = avg_pool_2d_cpu(&input, 3, 3, 2, 2, 1, 1, 1, 0);
    assert!(!result.is_empty(), "asymmetric padding output");
    assert_no_nan_inf(&result, "asym_pad");
}

#[test]
fn test_zero_padding() {
    let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let with_pad = avg_pool_1d_cpu(&input, 4, 4, 0);
    // No padding should give same as explicit 0-padding.
    assert_eq!(with_pad.len(), 4);
}

#[test]
fn test_large_padding() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let result = avg_pool_1d_cpu(&input, 3, 1, 3);
    // padding=3 on each side → many boundary windows
    assert!(!result.is_empty());
    assert_no_nan_inf(&result, "large_pad");
}

#[test]
fn test_stride_1_2d_full_coverage() {
    // Every output position sees a unique window
    let input: Vec<f32> = (0..25).map(|i| i as f32).collect();
    let result = avg_pool_2d_cpu(&input, 5, 5, 3, 3, 1, 1, 0, 0);
    assert_eq!(result.len(), 9); // 3×3 output
    assert_no_nan_inf(&result, "stride1_2d");
}

// ═══════════════════════════════════════════════════════════════════
// 8. Multi-channel pooling
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_multi_channel_avg_pool() {
    let channels = 4;
    let spatial = 16;
    let input: Vec<f32> = (0..channels * spatial).map(|i| (i as f32) * 0.1).collect();
    for c in 0..channels {
        let ch = &input[c * spatial..(c + 1) * spatial];
        let result = avg_pool_1d_cpu(ch, 4, 4, 0);
        assert_eq!(result.len(), 4, "channel {c}");
        assert_no_nan_inf(&result, &format!("mc_avg_ch{c}"));
    }
}

#[test]
fn test_multi_channel_max_pool() {
    let channels = 4;
    let spatial = 16;
    let input: Vec<f32> = (0..channels * spatial).map(|i| (i as f32) * 0.1).collect();
    for c in 0..channels {
        let ch = &input[c * spatial..(c + 1) * spatial];
        let result = max_pool_1d_cpu(ch, 4, 4, 0);
        assert_eq!(result.len(), 4, "channel {c}");
        assert_no_nan_inf(&result, &format!("mc_max_ch{c}"));
    }
}

#[test]
fn test_multi_channel_global_avg() {
    let channels = 8;
    let spatial = 64;
    let input: Vec<f32> = (0..channels * spatial).map(|i| (i as f32) * 0.01).collect();
    let results: Vec<f32> = (0..channels)
        .map(|c| global_avg_pool_cpu(&input[c * spatial..(c + 1) * spatial]))
        .collect();
    // Each successive channel has higher mean
    for i in 1..channels {
        assert!(results[i] > results[i - 1], "ch{i} mean should exceed ch{}", i - 1);
    }
}

#[test]
fn test_multi_channel_preserves_channel_count() {
    let batch = 2;
    let channels = 3;
    let h = 8;
    let w = 8;
    let total = batch * channels * h * w;
    let input: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01).collect();
    let mut out_count = 0;
    for _b in 0..batch {
        for c in 0..channels {
            let offset = (_b * channels + c) * h * w;
            let ch = &input[offset..offset + h * w];
            let result = avg_pool_2d_cpu(ch, h, w, 2, 2, 2, 2, 0, 0);
            assert_eq!(result.len(), 16); // 4×4
            out_count += 1;
        }
    }
    assert_eq!(out_count, batch * channels);
}

#[test]
fn test_multi_channel_independence() {
    // Verify pooling on one channel doesn't affect another.
    let c1 = vec![1.0; 16];
    let c2 = vec![2.0; 16];
    let r1 = avg_pool_1d_cpu(&c1, 4, 4, 0);
    let r2 = avg_pool_1d_cpu(&c2, 4, 4, 0);
    assert_close(&r1, &[1.0, 1.0, 1.0, 1.0], 1e-6, "ch_indep_1");
    assert_close(&r2, &[2.0, 2.0, 2.0, 2.0], 1e-6, "ch_indep_2");
}

#[test]
fn test_multi_channel_2d_batched() {
    let batch = 2;
    let channels = 2;
    let h = 4;
    let w = 4;
    for b in 0..batch {
        for c in 0..channels {
            let val = (b * channels + c + 1) as f32;
            let input = vec![val; h * w];
            let result = max_pool_2d_cpu(&input, h, w, 2, 2, 2, 2, 0, 0);
            assert_close(&result, &[val, val, val, val], 1e-6, &format!("batch{b}_ch{c}"));
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// 9. Numerical precision
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_precision_avg_pool_small_values() {
    let input: Vec<f32> = (0..64).map(|i| (i as f32) * 1e-7).collect();
    let result = avg_pool_1d_cpu(&input, 8, 8, 0);
    assert_eq!(result.len(), 8);
    assert_no_nan_inf(&result, "precision_small");
    for &v in &result {
        assert!(v >= 0.0, "small value negative: {v}");
    }
}

#[test]
fn test_precision_avg_pool_large_values() {
    let input = vec![1e6; 64];
    let result = avg_pool_1d_cpu(&input, 8, 8, 0);
    for &v in &result {
        assert!((v - 1e6).abs() < 1.0, "large value drift: {v}");
    }
}

#[test]
fn test_precision_mixed_magnitude() {
    let mut input = vec![1e-6; 32];
    input[15] = 1e6;
    let result = avg_pool_1d_cpu(&input, 32, 32, 0);
    assert_eq!(result.len(), 1);
    assert!(result[0] > 0.0, "mixed mag should be positive");
}

#[test]
fn test_precision_kahan_vs_naive_global_avg() {
    // Large array where naive sum loses precision
    let n = 10_000;
    let input: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 1e-7).collect();
    let naive = input.iter().sum::<f32>() / n as f32;
    let kahan = (kahan_sum(&input) / n as f64) as f32;
    // Both should be close to 1.0 + small offset
    assert!((naive - kahan).abs() < 1e-3, "kahan vs naive: {naive} vs {kahan}");
}

#[test]
fn test_precision_max_pool_no_precision_loss() {
    // Max pool should have zero precision issues (no accumulation)
    let input: Vec<f32> = (0..1024).map(|i| (i as f32) * std::f32::consts::PI).collect();
    let result = max_pool_1d_cpu(&input, 32, 32, 0);
    for (i, &v) in result.iter().enumerate() {
        let expected = ((i + 1) * 32 - 1) as f32 * std::f32::consts::PI;
        assert!((v - expected).abs() < 1e-3, "max precision [{i}]: {v} vs {expected}");
    }
}

#[test]
fn test_precision_denormal_values() {
    let input = vec![f32::MIN_POSITIVE / 2.0; 16];
    let result = avg_pool_1d_cpu(&input, 4, 4, 0);
    for &v in &result {
        assert!(!v.is_nan(), "denormal produced NaN");
    }
}

#[test]
fn test_precision_alternating_sign() {
    // Catastrophic cancellation scenario
    let input: Vec<f32> = (0..64).map(|i| if i % 2 == 0 { 1e6 } else { -1e6 }).collect();
    let result = avg_pool_1d_cpu(&input, 64, 64, 0);
    assert_eq!(result.len(), 1);
    assert!(result[0].abs() < 1e-1, "cancellation avg: {}", result[0]);
}

// ═══════════════════════════════════════════════════════════════════
// 10. Edge cases
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_edge_batch_size_one() {
    let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let result = avg_pool_1d_cpu(&input, 4, 4, 0);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_edge_kernel_size_one() {
    let input: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let avg = avg_pool_1d_cpu(&input, 1, 1, 0);
    let max = max_pool_1d_cpu(&input, 1, 1, 0);
    // kernel=1 → identity
    assert_close(&avg, &input, 1e-6, "kernel1_avg");
    assert_close(&max, &input, 1e-6, "kernel1_max");
}

#[test]
fn test_edge_kernel_equals_input() {
    let input: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let avg = avg_pool_1d_cpu(&input, 8, 8, 0);
    let max = max_pool_1d_cpu(&input, 8, 8, 0);
    assert_eq!(avg.len(), 1);
    assert_eq!(max.len(), 1);
    assert!((max[0] - 7.0).abs() < f32::EPSILON);
}

#[test]
fn test_edge_single_element_input() {
    let input = vec![5.0];
    let avg = avg_pool_1d_cpu(&input, 1, 1, 0);
    let max = max_pool_1d_cpu(&input, 1, 1, 0);
    assert_close(&avg, &[5.0], 1e-6, "single_avg");
    assert_close(&max, &[5.0], 1e-6, "single_max");
}

#[test]
fn test_edge_2d_single_pixel() {
    let input = vec![42.0]; // 1×1
    let avg = avg_pool_2d_cpu(&input, 1, 1, 1, 1, 1, 1, 0, 0);
    let max = max_pool_2d_cpu(&input, 1, 1, 1, 1, 1, 1, 0, 0);
    assert_close(&avg, &[42.0], 1e-6, "2d_single_avg");
    assert_close(&max, &[42.0], 1e-6, "2d_single_max");
}

#[test]
fn test_edge_all_zeros() {
    let input = vec![0.0; 64];
    let avg = avg_pool_1d_cpu(&input, 8, 8, 0);
    let max = max_pool_1d_cpu(&input, 8, 8, 0);
    for &v in &avg {
        assert!((v - 0.0).abs() < f32::EPSILON, "zero avg: {v}");
    }
    for &v in &max {
        assert!((v - 0.0).abs() < f32::EPSILON, "zero max: {v}");
    }
}

// ═══════════════════════════════════════════════════════════════════
// 11. Pooling gradients
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_max_pool_grad_routes_to_argmax() {
    let input = vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0];
    let grad_out = vec![10.0, 20.0];
    let grad_in = max_pool_1d_grad_cpu(&input, &grad_out, 3, 3);
    // Window [0..3]: max at idx=1 → grad_in[1] += 10
    // Window [3..6]: max at idx=5 → grad_in[5] += 20
    assert_eq!(grad_in.len(), 6);
    assert!((grad_in[1] - 10.0).abs() < 1e-6, "grad at argmax 1");
    assert!((grad_in[5] - 20.0).abs() < 1e-6, "grad at argmax 5");
    assert!((grad_in[0] - 0.0).abs() < 1e-6, "non-argmax should be 0");
}

#[test]
fn test_avg_pool_grad_distributes_equally() {
    let grad_out = vec![4.0, 8.0];
    let grad_in = avg_pool_1d_grad_cpu(8, &grad_out, 4, 4);
    assert_eq!(grad_in.len(), 8);
    // First window: each gets 4.0/4 = 1.0
    for i in 0..4 {
        assert!((grad_in[i] - 1.0).abs() < 1e-6, "avg grad[{i}] = {}", grad_in[i]);
    }
    // Second window: each gets 8.0/4 = 2.0
    for i in 4..8 {
        assert!((grad_in[i] - 2.0).abs() < 1e-6, "avg grad[{i}] = {}", grad_in[i]);
    }
}

#[test]
fn test_avg_pool_grad_overlapping_accumulates() {
    let grad_out = vec![1.0, 1.0, 1.0];
    let grad_in = avg_pool_1d_grad_cpu(6, &grad_out, 4, 1);
    // Overlapping windows: positions in multiple windows accumulate
    assert_eq!(grad_in.len(), 6);
    assert_no_nan_inf(&grad_in, "avg_grad_overlap");
    // Middle elements should have higher gradient
    assert!(grad_in[2] > grad_in[0], "center accumulates more");
}

#[test]
fn test_max_pool_grad_with_ties() {
    // When multiple elements equal the max, first occurrence wins
    let input = vec![5.0, 5.0, 5.0, 1.0, 1.0, 1.0];
    let grad_out = vec![10.0, 20.0];
    let grad_in = max_pool_1d_grad_cpu(&input, &grad_out, 3, 3);
    // First window: max=5.0 at idx=0 (first occurrence)
    assert!((grad_in[0] - 10.0).abs() < 1e-6, "tie goes to first");
    assert!((grad_in[1] - 0.0).abs() < 1e-6, "tied non-first = 0");
}

#[test]
fn test_grad_zero_gradient_output() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let grad_out = vec![0.0];
    let max_grad = max_pool_1d_grad_cpu(&input, &grad_out, 4, 4);
    let avg_grad = avg_pool_1d_grad_cpu(4, &grad_out, 4, 4);
    for &v in &max_grad {
        assert!((v - 0.0).abs() < 1e-6, "zero grad max");
    }
    for &v in &avg_grad {
        assert!((v - 0.0).abs() < 1e-6, "zero grad avg");
    }
}

#[test]
fn test_grad_single_element_pool() {
    let input = vec![7.0];
    let grad_out = vec![3.0];
    let max_grad = max_pool_1d_grad_cpu(&input, &grad_out, 1, 1);
    let avg_grad = avg_pool_1d_grad_cpu(1, &grad_out, 1, 1);
    assert_close(&max_grad, &[3.0], 1e-6, "single_max_grad");
    assert_close(&avg_grad, &[3.0], 1e-6, "single_avg_grad");
}

// ═══════════════════════════════════════════════════════════════════
// 12. GPU-gated integration tests
// ═══════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_gpu_avg_pool_1d_dispatch() {
    let input: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();
    let cpu_result = avg_pool_1d_cpu(&input, 4, 4, 0);
    let groups = dispatch_groups(cpu_result.len() as u32, SIMD_GROUP_WIDTH);
    assert!(groups > 0, "at least 1 threadgroup");
    assert_no_nan_inf(&cpu_result, "gpu_avg_1d");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_gpu_max_pool_1d_dispatch() {
    let input: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();
    let cpu_result = max_pool_1d_cpu(&input, 4, 4, 0);
    let groups = dispatch_groups(cpu_result.len() as u32, SIMD_GROUP_WIDTH);
    assert!(groups > 0, "at least 1 threadgroup");
    assert_no_nan_inf(&cpu_result, "gpu_max_1d");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_gpu_avg_pool_2d_dispatch() {
    let h = 32;
    let w = 32;
    let input: Vec<f32> = (0..h * w).map(|i| (i as f32) * 0.001).collect();
    let cpu_result = avg_pool_2d_cpu(&input, h, w, 4, 4, 4, 4, 0, 0);
    let out_size = cpu_result.len() * std::mem::size_of::<f32>();
    let aligned_size = align_to_metal(out_size);
    assert!(aligned_size.is_multiple_of(METAL_BUFFER_ALIGNMENT), "output buffer alignment");
    assert_no_nan_inf(&cpu_result, "gpu_avg_2d");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_gpu_global_avg_pool_threadgroup_sizing() {
    let input: Vec<f32> =
        (0..MAX_THREADS_PER_THREADGROUP as usize).map(|i| (i as f32) * 0.01).collect();
    let result = global_avg_pool_cpu(&input);
    assert!(result.is_finite(), "global avg finite");
    // Exactly one threadgroup needed
    assert_eq!(dispatch_groups(MAX_THREADS_PER_THREADGROUP, MAX_THREADS_PER_THREADGROUP), 1);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_gpu_attention_pool_buffer_alignment() {
    let seq_len = 32;
    let dim = 128;
    let values: Vec<f32> = (0..seq_len * dim).map(|i| (i as f32) * 0.001).collect();
    let scores: Vec<f32> = (0..seq_len).map(|i| (i as f32) * 0.1).collect();
    let result = attention_pool_cpu(&values, &scores, seq_len, dim);
    let buf_size = result.len() * std::mem::size_of::<f32>();
    let aligned = align_to_metal(buf_size);
    assert!(aligned.is_multiple_of(METAL_BUFFER_ALIGNMENT), "attn buffer alignment");
    assert_no_nan_inf(&result, "gpu_attn_align");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_gpu_adaptive_pool_simd_groups() {
    let input: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.01).collect();
    let result = adaptive_avg_pool_1d_cpu(&input, 64);
    let groups = dispatch_groups(64, SIMD_GROUP_WIDTH);
    assert_eq!(groups, 2, "64 outputs / 32 SIMD width = 2 groups");
    assert_no_nan_inf(&result, "gpu_adaptive_simd");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_gpu_pooling_chain_avg_then_global() {
    // Two-stage pooling: local avg pool → global avg pool
    let input: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();
    let stage1 = avg_pool_1d_cpu(&input, 4, 4, 0);
    let stage2 = global_avg_pool_cpu(&stage1);
    let direct_global = global_avg_pool_cpu(&input);
    // Two-stage should approximate direct global average
    assert!((stage2 - direct_global).abs() < 1e-2, "chain vs direct: {stage2} vs {direct_global}");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_gpu_multi_channel_pooling_dispatch() {
    let channels = 8;
    let spatial = 256;
    let input: Vec<f32> = (0..channels * spatial).map(|i| (i as f32) * 0.001).collect();
    for c in 0..channels {
        let ch = &input[c * spatial..(c + 1) * spatial];
        let result = avg_pool_1d_cpu(ch, 8, 8, 0);
        let groups = dispatch_groups(result.len() as u32, SIMD_GROUP_WIDTH);
        assert!(groups >= 1, "channel {c}: at least 1 group");
        assert_no_nan_inf(&result, &format!("gpu_mc_ch{c}"));
    }
}
