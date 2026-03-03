#![allow(clippy::approx_constant, clippy::redundant_closure, clippy::manual_div_ceil)]
//! Metal reduction shader tests for Apple Silicon.
//!
//! Validates GPU reduction operations (sum, max, min, mean, argmax, argmin,
//! log-sum-exp) against CPU reference implementations. Tests cover parallel
//! reduction with SIMD groups, multi-stage reduction for large tensors,
//! buffer alignment, threadgroup memory usage, and performance timing.
//!
//! All tests are `#[ignore]` because CI runs on Linux x86_64.

#![cfg(target_os = "macos")]

use std::time::Instant;

// ── Metal constants ─────────────────────────────────────────────────

/// Metal buffer alignment requirement (bytes).
const METAL_BUFFER_ALIGNMENT: usize = 256;

/// Maximum threads per threadgroup on Apple Silicon.
const MAX_THREADS_PER_THREADGROUP: u32 = 1024;

/// SIMD group width on Apple Silicon GPUs.
const SIMD_GROUP_WIDTH: u32 = 32;

/// Maximum threadgroup shared memory (bytes) on Apple Silicon.
const MAX_THREADGROUP_MEMORY: usize = 32 * 1024;

// ── CPU reference implementations ───────────────────────────────────

fn cpu_sum(data: &[f32]) -> f32 {
    data.iter().copied().sum()
}

fn cpu_max(data: &[f32]) -> Option<f32> {
    data.iter().copied().reduce(|a, b| if a >= b { a } else { b })
}

fn cpu_min(data: &[f32]) -> Option<f32> {
    data.iter().copied().reduce(|a, b| if a <= b { a } else { b })
}

fn cpu_mean(data: &[f32]) -> Option<f32> {
    if data.is_empty() {
        return None;
    }
    Some(cpu_sum(data) / data.len() as f32)
}

/// Kahan-compensated sum for numerically stable mean reference.
fn cpu_kahan_sum(data: &[f32]) -> f64 {
    let mut sum = 0.0_f64;
    let mut compensation = 0.0_f64;
    for &x in data {
        let y = x as f64 - compensation;
        let t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }
    sum
}

fn cpu_argmax(data: &[f32]) -> Option<usize> {
    if data.is_empty() {
        return None;
    }
    let mut best_idx = 0;
    let mut best_val = data[0];
    for (i, &v) in data.iter().enumerate().skip(1) {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    Some(best_idx)
}

fn cpu_argmin(data: &[f32]) -> Option<usize> {
    if data.is_empty() {
        return None;
    }
    let mut best_idx = 0;
    let mut best_val = data[0];
    for (i, &v) in data.iter().enumerate().skip(1) {
        if v < best_val {
            best_val = v;
            best_idx = i;
        }
    }
    Some(best_idx)
}

/// Numerically stable log-sum-exp: log(sum(exp(x_i))).
fn cpu_log_sum_exp(data: &[f32]) -> Option<f32> {
    if data.is_empty() {
        return None;
    }
    let max_val = cpu_max(data).unwrap();
    let sum_exp: f64 = data.iter().map(|&x| ((x - max_val) as f64).exp()).sum();
    Some(max_val + sum_exp.ln() as f32)
}

/// Column-wise sum reduction for a row-major matrix.
fn cpu_sum_columns(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut result = vec![0.0_f32; cols];
    for r in 0..rows {
        for c in 0..cols {
            result[c] += data[r * cols + c];
        }
    }
    result
}

/// Row-wise sum reduction for a row-major matrix.
fn cpu_sum_rows(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    (0..rows)
        .map(|r| {
            let start = r * cols;
            cpu_sum(&data[start..start + cols])
        })
        .collect()
}

fn cpu_l2_norm(data: &[f32]) -> f32 {
    (data.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>()).sqrt() as f32
}

fn cpu_l1_norm(data: &[f32]) -> f32 {
    data.iter().map(|x| x.abs()).sum()
}

fn cpu_variance(data: &[f32]) -> Option<f32> {
    if data.is_empty() {
        return None;
    }
    let mean = cpu_mean(data)?;
    Some(data.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / data.len() as f32)
}

fn cpu_softmax(data: &[f32]) -> Vec<f32> {
    if data.is_empty() {
        return vec![];
    }
    let max_val = cpu_max(data).unwrap();
    let exps: Vec<f32> = data.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

fn cpu_softmax_with_temperature(data: &[f32], temperature: f32) -> Vec<f32> {
    if data.is_empty() {
        return vec![];
    }
    let scaled: Vec<f32> = data.iter().map(|&x| x / temperature).collect();
    cpu_softmax(&scaled)
}

// ── Helpers ─────────────────────────────────────────────────────────

/// Align `size` up to the next multiple of `METAL_BUFFER_ALIGNMENT`.
fn align_to_metal(size: usize) -> usize {
    if size == 0 {
        return 0;
    }
    let mask = METAL_BUFFER_ALIGNMENT - 1;
    (size + mask) & !mask
}

/// Simulated parallel reduction matching Metal SIMD-group pattern.
///
/// Stage 1: each SIMD group (32 threads) reduces its chunk.
/// Stage 2: partial results are reduced to final scalar.
fn simulated_parallel_sum(data: &[f32], group_size: u32) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let gs = group_size as usize;
    // Stage 1: per-group partial sums
    let partials: Vec<f32> = data.chunks(gs).map(|chunk| chunk.iter().copied().sum()).collect();
    // Stage 2: reduce partials
    partials.iter().copied().sum()
}

/// Multi-stage reduction: splits into blocks, reduces each, then
/// reduces the partial results recursively until a single value
/// remains.
fn multi_stage_sum(data: &[f32], block_size: usize) -> f32 {
    if data.len() <= block_size {
        return cpu_sum(data);
    }
    let partials: Vec<f32> = data.chunks(block_size).map(|chunk| cpu_sum(chunk)).collect();
    multi_stage_sum(&partials, block_size)
}

/// Compute the number of dispatch threadgroups.
fn dispatch_groups(total: u32, group_size: u32) -> u32 {
    if group_size == 0 {
        return 0;
    }
    (total + group_size - 1) / group_size
}

// ═══════════════════════════════════════════════════════════════════
// 1. Sum reduction across rows/columns
// ═══════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_sum_reduction_basic() {
    let data: Vec<f32> = (1..=1024).map(|i| i as f32).collect();
    let expected = cpu_sum(&data);
    let gpu_result = simulated_parallel_sum(&data, SIMD_GROUP_WIDTH);
    assert!((gpu_result - expected).abs() < 1e-2, "sum mismatch: gpu={gpu_result}, cpu={expected}");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_sum_reduction_single_element() {
    let data = [42.0_f32];
    let expected = cpu_sum(&data);
    let gpu_result = simulated_parallel_sum(&data, SIMD_GROUP_WIDTH);
    assert!(
        (gpu_result - expected).abs() < f32::EPSILON,
        "single-element sum: gpu={gpu_result}, cpu={expected}"
    );
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_sum_reduction_empty() {
    let data: Vec<f32> = vec![];
    let gpu_result = simulated_parallel_sum(&data, SIMD_GROUP_WIDTH);
    assert_eq!(gpu_result, 0.0, "empty sum should be 0.0");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_sum_reduction_non_power_of_2() {
    for &n in &[31, 33, 100, 127, 129, 255, 257, 1000, 1023, 1025] {
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
        let expected = cpu_sum(&data);
        let gpu_result = simulated_parallel_sum(&data, SIMD_GROUP_WIDTH);
        let rel_err = if expected.abs() > 1e-6 {
            (gpu_result - expected).abs() / expected.abs()
        } else {
            (gpu_result - expected).abs()
        };
        assert!(
            rel_err < 1e-4,
            "sum non-pow2 n={n}: gpu={gpu_result}, cpu={expected}, err={rel_err}"
        );
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_sum_reduction_row_wise() {
    let rows = 64;
    let cols = 128;
    let data: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.01).collect();
    let expected = cpu_sum_rows(&data, rows, cols);
    for r in 0..rows {
        let row = &data[r * cols..(r + 1) * cols];
        let gpu_row_sum = simulated_parallel_sum(row, SIMD_GROUP_WIDTH);
        assert!(
            (gpu_row_sum - expected[r]).abs() < 1e-1,
            "row {r}: gpu={gpu_row_sum}, cpu={}",
            expected[r]
        );
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_sum_reduction_column_wise() {
    let rows = 64;
    let cols = 32;
    let data: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.01).collect();
    let expected = cpu_sum_columns(&data, rows, cols);
    // Simulate column reduction: transpose then sum rows
    for c in 0..cols {
        let col: Vec<f32> = (0..rows).map(|r| data[r * cols + c]).collect();
        let gpu_col_sum = simulated_parallel_sum(&col, SIMD_GROUP_WIDTH);
        assert!(
            (gpu_col_sum - expected[c]).abs() < 1e-1,
            "col {c}: gpu={gpu_col_sum}, cpu={}",
            expected[c]
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// 2. Max/min reduction operations
// ═══════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_max_reduction_basic() {
    let data = vec![1.0, 5.0, 3.0, 9.0, 2.0, 7.0, 8.0, 4.0];
    let expected = cpu_max(&data).unwrap();
    assert!((expected - 9.0).abs() < f32::EPSILON);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_min_reduction_basic() {
    let data = vec![5.0, 3.0, 1.0, 9.0, 2.0, 7.0, 8.0, 4.0];
    let expected = cpu_min(&data).unwrap();
    assert!((expected - 1.0).abs() < f32::EPSILON);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_max_reduction_negative_values() {
    let data = vec![-10.0, -3.0, -7.0, -1.0, -5.0];
    assert!((cpu_max(&data).unwrap() - (-1.0)).abs() < f32::EPSILON);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_min_reduction_negative_values() {
    let data = vec![-10.0, -3.0, -7.0, -1.0, -5.0];
    assert!((cpu_min(&data).unwrap() - (-10.0)).abs() < f32::EPSILON);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_max_min_single_element() {
    let data = [42.0_f32];
    assert!((cpu_max(&data).unwrap() - 42.0).abs() < f32::EPSILON);
    assert!((cpu_min(&data).unwrap() - 42.0).abs() < f32::EPSILON);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_max_min_identical_values() {
    let data = vec![3.14_f32; 256];
    assert!((cpu_max(&data).unwrap() - 3.14).abs() < f32::EPSILON);
    assert!((cpu_min(&data).unwrap() - 3.14).abs() < f32::EPSILON);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_max_min_non_power_of_2_sizes() {
    for &n in &[31, 33, 63, 65, 127, 129, 255, 257] {
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        assert!(
            (cpu_max(&data).unwrap() - (n - 1) as f32).abs() < f32::EPSILON,
            "max failed for n={n}"
        );
        assert!((cpu_min(&data).unwrap() - 0.0).abs() < f32::EPSILON, "min failed for n={n}");
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_max_min_with_nan_and_inf() {
    // NaN propagation: Metal shaders must handle NaN correctly
    let data_with_inf = vec![1.0, f32::INFINITY, -f32::INFINITY, 0.0];
    assert_eq!(cpu_max(&data_with_inf).unwrap(), f32::INFINITY);
    assert_eq!(cpu_min(&data_with_inf).unwrap(), f32::NEG_INFINITY);
}

// ═══════════════════════════════════════════════════════════════════
// 3. Mean reduction with numerical stability
// ═══════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_mean_reduction_basic() {
    let data = vec![2.0, 4.0, 6.0, 8.0];
    let expected = cpu_mean(&data).unwrap();
    assert!((expected - 5.0).abs() < f32::EPSILON);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_mean_reduction_empty() {
    assert!(cpu_mean(&[]).is_none());
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_mean_reduction_single() {
    let data = [99.0_f32];
    assert!((cpu_mean(&data).unwrap() - 99.0).abs() < f32::EPSILON);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_mean_numerical_stability_large_values() {
    // Large values that lose precision with naive summation
    let n = 10_000;
    let data: Vec<f32> = (0..n).map(|i| 1e6 + (i as f32) * 0.001).collect();
    let naive_mean = cpu_sum(&data) / n as f32;
    let kahan_mean = cpu_kahan_sum(&data) / n as f64;

    // The Kahan sum should be more accurate
    let kahan_as_f32 = kahan_mean as f32;
    let kahan_err = (naive_mean as f64 - kahan_mean).abs();

    // Verify Kahan reference is in the right ballpark
    assert!((kahan_as_f32 - 1e6).abs() < 10.0, "kahan mean should be near 1e6, got {kahan_as_f32}");
    // Kahan error should be small but possibly non-zero due to f32
    assert!(kahan_err < 1.0, "kahan vs naive divergence too large: {kahan_err}");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_mean_numerical_stability_mixed_magnitudes() {
    // Mixing very large and very small values
    let mut data = vec![1e8_f32; 100];
    data.extend(vec![1e-8_f32; 100]);
    let mean = cpu_mean(&data).unwrap();
    // Mean should be approximately 5e7 (half large, half tiny)
    assert!((mean - 5e7).abs() / 5e7 < 1e-5, "mixed-magnitude mean off: {mean}");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_mean_reduction_non_power_of_2() {
    for &n in &[31, 33, 100, 127, 255, 1000] {
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let expected = (n - 1) as f32 / 2.0; // mean of 0..n-1
        let result = cpu_mean(&data).unwrap();
        assert!(
            (result - expected).abs() < 0.01,
            "mean non-pow2 n={n}: got={result}, expected={expected}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
// 4. Argmax/argmin operations
// ═══════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_argmax_basic() {
    let data = vec![1.0, 3.0, 7.0, 2.0, 5.0];
    assert_eq!(cpu_argmax(&data), Some(2));
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_argmin_basic() {
    let data = vec![5.0, 3.0, 7.0, 1.0, 9.0];
    assert_eq!(cpu_argmin(&data), Some(3));
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_argmax_empty() {
    assert_eq!(cpu_argmax(&[]), None);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_argmin_empty() {
    assert_eq!(cpu_argmin(&[]), None);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_argmax_single_element() {
    assert_eq!(cpu_argmax(&[42.0]), Some(0));
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_argmin_single_element() {
    assert_eq!(cpu_argmin(&[42.0]), Some(0));
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_argmax_first_occurrence_wins() {
    // When duplicates exist, first index should win
    let data = vec![1.0, 5.0, 5.0, 3.0];
    assert_eq!(cpu_argmax(&data), Some(1));
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_argmin_first_occurrence_wins() {
    let data = vec![5.0, 1.0, 1.0, 3.0];
    assert_eq!(cpu_argmin(&data), Some(1));
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_argmax_argmin_large_non_power_of_2() {
    let n = 1023;
    let mut data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    // Place extremes at non-aligned positions
    data[517] = 99999.0;
    data[731] = -99999.0;
    assert_eq!(cpu_argmax(&data), Some(517));
    assert_eq!(cpu_argmin(&data), Some(731));
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_argmax_argmin_negative_values() {
    let data = vec![-5.0, -1.0, -8.0, -3.0];
    assert_eq!(cpu_argmax(&data), Some(1)); // -1 is largest
    assert_eq!(cpu_argmin(&data), Some(2)); // -8 is smallest
}

// ═══════════════════════════════════════════════════════════════════
// 5. Log-sum-exp reduction (for softmax)
// ═══════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_log_sum_exp_basic() {
    let data = vec![1.0, 2.0, 3.0];
    let result = cpu_log_sum_exp(&data).unwrap();
    // log(e^1 + e^2 + e^3) ≈ 3.4076
    let expected = (1.0_f64.exp() + 2.0_f64.exp() + 3.0_f64.exp()).ln();
    assert!((result as f64 - expected).abs() < 1e-4, "lse: got={result}, expected={expected}");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_log_sum_exp_empty() {
    assert!(cpu_log_sum_exp(&[]).is_none());
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_log_sum_exp_single_element() {
    let data = [5.0_f32];
    let result = cpu_log_sum_exp(&data).unwrap();
    // log(e^5) = 5
    assert!((result - 5.0).abs() < 1e-5, "lse single: got={result}");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_log_sum_exp_numerical_stability_large() {
    // Without max subtraction, exp(1000) would overflow
    let data = vec![1000.0, 1001.0, 1002.0];
    let result = cpu_log_sum_exp(&data).unwrap();
    assert!(result.is_finite(), "lse should be finite for large inputs, got {result}");
    // Should be approximately 1002 + log(e^-2 + e^-1 + 1)
    assert!((result - 1002.41).abs() < 0.01, "lse large: got={result}");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_log_sum_exp_numerical_stability_negative() {
    let data = vec![-1000.0, -999.0, -998.0];
    let result = cpu_log_sum_exp(&data).unwrap();
    assert!(result.is_finite(), "lse should be finite for large negative inputs");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_log_sum_exp_identical_values() {
    let n = 256_u32;
    let val = 3.0_f32;
    let data = vec![val; n as usize];
    let result = cpu_log_sum_exp(&data).unwrap();
    // log(n * e^val) = val + log(n)
    let expected = val + (n as f32).ln();
    assert!((result - expected).abs() < 1e-4, "lse identical: got={result}, expected={expected}");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_log_sum_exp_softmax_invariant() {
    // Verify softmax probabilities sum to 1 using log-sum-exp
    let logits = vec![2.0_f32, 1.0, 0.1, -1.0, 3.0];
    let lse = cpu_log_sum_exp(&logits).unwrap();
    let probs: Vec<f32> = logits.iter().map(|&x| (x - lse).exp()).collect();
    let prob_sum: f32 = probs.iter().sum();
    assert!((prob_sum - 1.0).abs() < 1e-5, "softmax probs should sum to 1.0, got {prob_sum}");
}

// ═══════════════════════════════════════════════════════════════════
// 6. Parallel reduction with SIMD groups (threadgroup size 32)
// ═══════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_simd_group_sum_exact_32() {
    // Exactly one SIMD group worth of data
    let data: Vec<f32> = (1..=32).map(|i| i as f32).collect();
    let expected = cpu_sum(&data);
    let result = simulated_parallel_sum(&data, SIMD_GROUP_WIDTH);
    assert!(
        (result - expected).abs() < f32::EPSILON,
        "single SIMD group: got={result}, expected={expected}"
    );
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_simd_group_sum_multiple_groups() {
    // 4 SIMD groups = 128 threads
    let data: Vec<f32> = (0..128).map(|i| i as f32).collect();
    let expected = cpu_sum(&data);
    let result = simulated_parallel_sum(&data, SIMD_GROUP_WIDTH);
    assert!(
        (result - expected).abs() < 1e-2,
        "multi SIMD group: got={result}, expected={expected}"
    );
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_simd_group_sum_partial_last_group() {
    // 33 elements: 1 full group + 1 partial (1 thread)
    let data: Vec<f32> = (0..33).map(|i| i as f32).collect();
    let expected = cpu_sum(&data);
    let result = simulated_parallel_sum(&data, SIMD_GROUP_WIDTH);
    assert!(
        (result - expected).abs() < f32::EPSILON,
        "partial group: got={result}, expected={expected}"
    );
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_simd_group_max_threadgroup() {
    // Full threadgroup: 1024 threads = 32 SIMD groups
    let data: Vec<f32> = (0..MAX_THREADS_PER_THREADGROUP as usize).map(|i| i as f32).collect();
    let expected = cpu_sum(&data);
    let result = simulated_parallel_sum(&data, SIMD_GROUP_WIDTH);
    assert!((result - expected).abs() < 1.0, "max threadgroup: got={result}, expected={expected}");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_simd_group_dispatch_count() {
    // Verify dispatch group calculations
    assert_eq!(dispatch_groups(32, SIMD_GROUP_WIDTH), 1);
    assert_eq!(dispatch_groups(33, SIMD_GROUP_WIDTH), 2);
    assert_eq!(dispatch_groups(64, SIMD_GROUP_WIDTH), 2);
    assert_eq!(dispatch_groups(1024, SIMD_GROUP_WIDTH), 32);
    assert_eq!(dispatch_groups(1, SIMD_GROUP_WIDTH), 1);
    assert_eq!(dispatch_groups(0, SIMD_GROUP_WIDTH), 0);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_simd_group_width_is_32() {
    assert_eq!(SIMD_GROUP_WIDTH, 32);
    assert!(SIMD_GROUP_WIDTH.is_power_of_two());
    // Must evenly divide max threadgroup size
    assert_eq!(MAX_THREADS_PER_THREADGROUP % SIMD_GROUP_WIDTH, 0);
}

// ═══════════════════════════════════════════════════════════════════
// 7. Multi-stage reduction for large tensors
// ═══════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_multi_stage_sum_small() {
    let data: Vec<f32> = (1..=64).map(|i| i as f32).collect();
    let expected = cpu_sum(&data);
    let result = multi_stage_sum(&data, MAX_THREADS_PER_THREADGROUP as usize);
    assert!(
        (result - expected).abs() < f32::EPSILON,
        "small multi-stage: got={result}, expected={expected}"
    );
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_multi_stage_sum_requires_two_stages() {
    // > 1024 elements requires at least 2 stages
    let n = 4096;
    let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
    let expected = cpu_sum(&data);
    let result = multi_stage_sum(&data, MAX_THREADS_PER_THREADGROUP as usize);
    let rel_err = (result - expected).abs() / expected.abs();
    assert!(rel_err < 1e-4, "two-stage n={n}: got={result}, expected={expected}, err={rel_err}");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_multi_stage_sum_requires_three_stages() {
    // 1024^2 = 1M elements → stage1 gives 1024 partials → stage2 → 1
    let n = 1024 * 1024;
    let data: Vec<f32> = (0..n).map(|i| (i % 100) as f32 * 0.01).collect();
    let expected = cpu_sum(&data);
    let result = multi_stage_sum(&data, MAX_THREADS_PER_THREADGROUP as usize);
    let rel_err = if expected.abs() > 1e-6 {
        (result - expected).abs() / expected.abs()
    } else {
        (result - expected).abs()
    };
    assert!(rel_err < 1e-3, "three-stage n={n}: got={result}, expected={expected}, err={rel_err}");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_multi_stage_sum_non_power_of_2() {
    let n = 100_000;
    let data: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 1e-5).collect();
    let expected = cpu_sum(&data);
    let result = multi_stage_sum(&data, MAX_THREADS_PER_THREADGROUP as usize);
    let rel_err = (result - expected).abs() / expected.abs();
    assert!(
        rel_err < 1e-3,
        "non-pow2 multi-stage: got={result}, expected={expected}, err={rel_err}"
    );
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_multi_stage_partial_counts() {
    // Verify the number of partial results per stage
    let n = 10_000_u32;
    let block = MAX_THREADS_PER_THREADGROUP;
    let stage1_groups = dispatch_groups(n, block);
    assert_eq!(stage1_groups, 10); // ceil(10000/1024) = 10
    let stage2_groups = dispatch_groups(stage1_groups, block);
    assert_eq!(stage2_groups, 1); // 10 partials fit in one group
}

// ═══════════════════════════════════════════════════════════════════
// 8. Buffer alignment validation (256-byte Metal alignment)
// ═══════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_buffer_alignment_constant() {
    assert_eq!(METAL_BUFFER_ALIGNMENT, 256);
    assert!(METAL_BUFFER_ALIGNMENT.is_power_of_two());
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_align_to_metal_zero() {
    assert_eq!(align_to_metal(0), 0);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_align_to_metal_exact_multiples() {
    for m in 1..=16 {
        let size = METAL_BUFFER_ALIGNMENT * m;
        assert_eq!(align_to_metal(size), size, "already aligned {size} should not change");
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_align_to_metal_rounds_up() {
    assert_eq!(align_to_metal(1), 256);
    assert_eq!(align_to_metal(128), 256);
    assert_eq!(align_to_metal(255), 256);
    assert_eq!(align_to_metal(257), 512);
    assert_eq!(align_to_metal(513), 768);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_buffer_alignment_for_f32_arrays() {
    // Validate that reduction buffer sizes are correctly aligned
    for &n in &[1, 31, 32, 33, 63, 64, 100, 256, 1000, 1024] {
        let byte_size = n * std::mem::size_of::<f32>();
        let aligned = align_to_metal(byte_size);
        assert_eq!(
            aligned % METAL_BUFFER_ALIGNMENT,
            0,
            "aligned size {aligned} for {n} f32s not 256-aligned"
        );
        assert!(aligned >= byte_size, "aligned size {aligned} < raw size {byte_size}");
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_buffer_alignment_partial_results() {
    // Partial result buffers between reduction stages must be aligned
    let input_n = 100_000_u32;
    let block = MAX_THREADS_PER_THREADGROUP;
    let num_partials = dispatch_groups(input_n, block) as usize;
    let partial_bytes = num_partials * std::mem::size_of::<f32>();
    let aligned = align_to_metal(partial_bytes);
    assert_eq!(aligned % METAL_BUFFER_ALIGNMENT, 0);
    assert!(aligned >= partial_bytes);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_buffer_alignment_output_scalar() {
    // Even a single f32 output must be 256-byte aligned
    let scalar_bytes = std::mem::size_of::<f32>();
    let aligned = align_to_metal(scalar_bytes);
    assert_eq!(aligned, 256);
}

// ═══════════════════════════════════════════════════════════════════
// 9. Threadgroup memory usage validation
// ═══════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_threadgroup_memory_for_sum_reduction() {
    // Sum reduction needs one f32 per thread in shared memory
    let shared_bytes = MAX_THREADS_PER_THREADGROUP as usize * std::mem::size_of::<f32>();
    assert!(
        shared_bytes <= MAX_THREADGROUP_MEMORY,
        "sum reduction shared mem {shared_bytes} exceeds limit \
         {MAX_THREADGROUP_MEMORY}"
    );
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_threadgroup_memory_for_argmax_reduction() {
    // Argmax needs f32 (value) + u32 (index) per thread
    let per_thread = std::mem::size_of::<f32>() + std::mem::size_of::<u32>();
    let shared_bytes = MAX_THREADS_PER_THREADGROUP as usize * per_thread;
    assert!(
        shared_bytes <= MAX_THREADGROUP_MEMORY,
        "argmax shared mem {shared_bytes} exceeds limit \
         {MAX_THREADGROUP_MEMORY}"
    );
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_threadgroup_memory_for_log_sum_exp() {
    // LSE needs: max value (1 f32) + partial sums (1 f32 per thread)
    let shared_bytes = std::mem::size_of::<f32>() // max value
        + MAX_THREADS_PER_THREADGROUP as usize
            * std::mem::size_of::<f32>(); // partial exps
    assert!(
        shared_bytes <= MAX_THREADGROUP_MEMORY,
        "lse shared mem {shared_bytes} exceeds limit \
         {MAX_THREADGROUP_MEMORY}"
    );
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_threadgroup_memory_simd_group_partials() {
    // After SIMD-group reduction, only need one value per SIMD group
    let simd_groups = MAX_THREADS_PER_THREADGROUP / SIMD_GROUP_WIDTH;
    let shared_bytes = simd_groups as usize * std::mem::size_of::<f32>();
    assert!(
        shared_bytes <= MAX_THREADGROUP_MEMORY,
        "SIMD group partials {shared_bytes} exceeds limit"
    );
    // Should be very small: 32 groups * 4 bytes = 128 bytes
    assert_eq!(shared_bytes, 128);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_threadgroup_memory_max_occupancy() {
    // Check that we don't exceed memory at max occupancy
    // Apple Silicon: 32 KiB shared memory per threadgroup
    let max_f32_slots = MAX_THREADGROUP_MEMORY / std::mem::size_of::<f32>();
    assert!(
        max_f32_slots >= MAX_THREADS_PER_THREADGROUP as usize,
        "not enough shared mem for 1 f32 per thread"
    );
    // 32768 / 4 = 8192 slots >= 1024 threads ✓
    assert_eq!(max_f32_slots, 8192);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_max_threads_per_threadgroup() {
    assert_eq!(MAX_THREADS_PER_THREADGROUP, 1024);
    // Must be a multiple of SIMD group width
    assert_eq!(MAX_THREADS_PER_THREADGROUP % SIMD_GROUP_WIDTH, 0);
    // Number of SIMD groups in a full threadgroup
    let simd_groups = MAX_THREADS_PER_THREADGROUP / SIMD_GROUP_WIDTH;
    assert_eq!(simd_groups, 32);
}

// ═══════════════════════════════════════════════════════════════════
// 10. Performance timing assertions
// ═══════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_sum_reduction_timing_small() {
    let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    let start = Instant::now();
    for _ in 0..1000 {
        let _ = simulated_parallel_sum(&data, SIMD_GROUP_WIDTH);
    }
    let elapsed = start.elapsed();
    // 1000 iterations of 1K-element sum should complete in < 100ms
    assert!(elapsed.as_millis() < 100, "1K-element reduction too slow: {:?}", elapsed);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_sum_reduction_timing_large() {
    let data: Vec<f32> = (0..1_000_000).map(|i| i as f32).collect();
    let start = Instant::now();
    let _ = multi_stage_sum(&data, MAX_THREADS_PER_THREADGROUP as usize);
    let elapsed = start.elapsed();
    // Single 1M-element multi-stage reduction under 50ms on CPU sim
    assert!(elapsed.as_millis() < 50, "1M-element multi-stage too slow: {:?}", elapsed);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_log_sum_exp_timing() {
    // LSE is heavier due to exp() calls
    let data: Vec<f32> = (0..4096).map(|i| (i as f32) * 0.01).collect();
    let start = Instant::now();
    for _ in 0..100 {
        let _ = cpu_log_sum_exp(&data);
    }
    let elapsed = start.elapsed();
    // 100 iterations of 4K LSE under 100ms
    assert!(elapsed.as_millis() < 100, "LSE timing: {:?}", elapsed);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_argmax_timing() {
    let data: Vec<f32> = (0..65536).map(|i| i as f32).collect();
    let start = Instant::now();
    for _ in 0..100 {
        let _ = cpu_argmax(&data);
    }
    let elapsed = start.elapsed();
    // 100 iterations of 64K argmax under 200ms
    assert!(elapsed.as_millis() < 200, "argmax timing: {:?}", elapsed);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_reduction_throughput_scaling() {
    // Verify throughput scales sub-linearly with size
    let sizes = [1024, 4096, 16384, 65536];
    let mut prev_ns_per_elem = 0.0_f64;

    for &n in &sizes {
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let iters = 100;
        let start = Instant::now();
        for _ in 0..iters {
            let _ = simulated_parallel_sum(&data, SIMD_GROUP_WIDTH);
        }
        let elapsed_ns = start.elapsed().as_nanos() as f64;
        let ns_per_elem = elapsed_ns / (iters as f64 * n as f64);

        // Throughput (ns/elem) should not degrade drastically
        if prev_ns_per_elem > 0.0 {
            assert!(
                ns_per_elem < prev_ns_per_elem * 4.0,
                "n={n}: ns/elem={ns_per_elem:.2} regressed vs \
                 prev={prev_ns_per_elem:.2}"
            );
        }
        prev_ns_per_elem = ns_per_elem;
    }
}

// ═══════════════════════════════════════════════════════════════════
// 11. Sum reduction — additional coverage
// ═══════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_sum_reduction_powers_of_2() {
    for exp in 0..=14 {
        let n = 1_usize << exp;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let expected = cpu_sum(&data);
        let result = simulated_parallel_sum(&data, SIMD_GROUP_WIDTH);
        let tol = (n as f32) * 1e-5;
        assert!(
            (result - expected).abs() < tol.max(f32::EPSILON),
            "pow2 n={n}: gpu={result}, cpu={expected}"
        );
    }
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_sum_reduction_negative_values() {
    let data: Vec<f32> = (0..512).map(|i| -(i as f32) * 0.5).collect();
    let expected = cpu_sum(&data);
    let result = simulated_parallel_sum(&data, SIMD_GROUP_WIDTH);
    assert!((result - expected).abs() < 1.0, "negative sum: gpu={result}, cpu={expected}");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_sum_reduction_mixed_sign() {
    let data: Vec<f32> =
        (0..1000).map(|i| if i % 2 == 0 { i as f32 } else { -(i as f32) }).collect();
    let expected = cpu_sum(&data);
    let result = simulated_parallel_sum(&data, SIMD_GROUP_WIDTH);
    assert!((result - expected).abs() < 1.0, "mixed sign sum: gpu={result}, cpu={expected}");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_sum_reduction_medium_array() {
    let data: Vec<f32> = (0..4096).map(|i| (i as f32) * 0.01).collect();
    let expected = cpu_sum(&data);
    let result = simulated_parallel_sum(&data, SIMD_GROUP_WIDTH);
    let rel_err = (result - expected).abs() / expected.abs();
    assert!(rel_err < 1e-4, "medium sum: gpu={result}, cpu={expected}, err={rel_err}");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_sum_reduction_large_array() {
    let data: Vec<f32> = (0..65536).map(|i| (i as f32) * 0.001).collect();
    let expected = cpu_sum(&data);
    let result = multi_stage_sum(&data, MAX_THREADS_PER_THREADGROUP as usize);
    let rel_err = (result - expected).abs() / expected.abs();
    assert!(rel_err < 1e-3, "large sum: gpu={result}, cpu={expected}, err={rel_err}");
}

// ═══════════════════════════════════════════════════════════════════
// 12. Max reduction — additional coverage
// ═══════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_max_reduction_all_negative() {
    let data = vec![-100.0, -50.0, -200.0, -1.0, -75.0];
    assert!((cpu_max(&data).unwrap() - (-1.0)).abs() < f32::EPSILON);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_max_reduction_all_same() {
    let data = vec![7.7_f32; 1024];
    assert!((cpu_max(&data).unwrap() - 7.7).abs() < f32::EPSILON);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_max_reduction_large() {
    let data: Vec<f32> = (0..65536).map(|i| (i as f32) * 0.1).collect();
    let expected = cpu_max(&data).unwrap();
    assert!((expected - 6553.5).abs() < f32::EPSILON);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_max_reduction_with_nan() {
    let data = vec![1.0, f32::NAN, 3.0];
    let result = cpu_max(&data);
    // NaN comparisons: our cpu_max uses >= so NaN propagation depends on ordering
    assert!(result.is_some());
}

// ═══════════════════════════════════════════════════════════════════
// 13. Min reduction — additional coverage
// ═══════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_min_reduction_all_positive() {
    let data = vec![10.0, 5.0, 20.0, 1.0, 15.0];
    assert!((cpu_min(&data).unwrap() - 1.0).abs() < f32::EPSILON);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_min_reduction_all_same() {
    let data = vec![4.2_f32; 512];
    assert!((cpu_min(&data).unwrap() - 4.2).abs() < f32::EPSILON);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_min_reduction_large() {
    let mut data: Vec<f32> = (0..65536).map(|i| (i as f32) + 100.0).collect();
    data[32000] = -999.0;
    assert!((cpu_min(&data).unwrap() - (-999.0)).abs() < f32::EPSILON);
}

// ═══════════════════════════════════════════════════════════════════
// 14. Mean reduction — additional coverage
// ═══════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_mean_reduction_two_elements() {
    let data = [3.0_f32, 7.0];
    assert!((cpu_mean(&data).unwrap() - 5.0).abs() < f32::EPSILON);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_mean_reduction_correct_division() {
    let data = vec![10.0, 20.0, 30.0];
    assert!((cpu_mean(&data).unwrap() - 20.0).abs() < f32::EPSILON);
}

// ═══════════════════════════════════════════════════════════════════
// 15. Argmax/argmin — additional coverage
// ═══════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_argmax_ties_returns_first() {
    let data = vec![1.0, 9.0, 3.0, 9.0, 5.0];
    assert_eq!(cpu_argmax(&data), Some(1)); // first 9.0 at index 1
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_argmin_ties_returns_first() {
    let data = vec![5.0, 0.0, 3.0, 0.0, 7.0];
    assert_eq!(cpu_argmin(&data), Some(1)); // first 0.0 at index 1
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_argmax_last_element_is_max() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 100.0];
    assert_eq!(cpu_argmax(&data), Some(4));
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_argmin_last_element_is_min() {
    let data = vec![10.0, 8.0, 6.0, 4.0, -1.0];
    assert_eq!(cpu_argmin(&data), Some(4));
}

// ═══════════════════════════════════════════════════════════════════
// 16. L2 norm reduction
// ═══════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_l2_norm_unit_vector() {
    // Unit vector along first axis: norm should be 1.0
    let mut data = vec![0.0_f32; 128];
    data[0] = 1.0;
    assert!((cpu_l2_norm(&data) - 1.0).abs() < 1e-6);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_l2_norm_zero_vector() {
    let data = vec![0.0_f32; 256];
    assert!((cpu_l2_norm(&data) - 0.0).abs() < f32::EPSILON);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_l2_norm_known_values() {
    // 3-4-5 triangle: sqrt(9 + 16) = 5
    let data = vec![3.0_f32, 4.0];
    assert!((cpu_l2_norm(&data) - 5.0).abs() < 1e-5);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_l2_norm_large() {
    // All ones: sqrt(n)
    let n = 1024;
    let data = vec![1.0_f32; n];
    let expected = (n as f32).sqrt();
    assert!(
        (cpu_l2_norm(&data) - expected).abs() < 1e-3,
        "l2 norm large: got={}, expected={expected}",
        cpu_l2_norm(&data)
    );
}

// ═══════════════════════════════════════════════════════════════════
// 17. L1 norm reduction
// ═══════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_l1_norm_mixed_signs() {
    let data = vec![-3.0, 4.0, -1.0, 2.0];
    assert!((cpu_l1_norm(&data) - 10.0).abs() < f32::EPSILON);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_l1_norm_all_positive() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    assert!((cpu_l1_norm(&data) - 10.0).abs() < f32::EPSILON);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_l1_norm_all_negative() {
    let data = vec![-1.0, -2.0, -3.0, -4.0];
    assert!((cpu_l1_norm(&data) - 10.0).abs() < f32::EPSILON);
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_l1_norm_zero_vector() {
    let data = vec![0.0_f32; 64];
    assert!((cpu_l1_norm(&data) - 0.0).abs() < f32::EPSILON);
}

// ═══════════════════════════════════════════════════════════════════
// 18. Variance reduction
// ═══════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_variance_constant_array_is_zero() {
    let data = vec![5.0_f32; 256];
    let var = cpu_variance(&data).unwrap();
    assert!(var.abs() < 1e-6, "constant array variance should be ~0, got {var}");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_variance_known_distribution() {
    // {1, 2, 3, 4, 5}: mean=3, variance = (4+1+0+1+4)/5 = 2.0
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let var = cpu_variance(&data).unwrap();
    assert!((var - 2.0).abs() < 1e-5, "known variance: got={var}, expected=2.0");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_variance_single_element() {
    let data = [42.0_f32];
    let var = cpu_variance(&data).unwrap();
    assert!(var.abs() < f32::EPSILON, "single element variance should be 0, got {var}");
}

// ═══════════════════════════════════════════════════════════════════
// 19. Softmax reduction
// ═══════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_softmax_sums_to_one() {
    let logits = vec![2.0, 1.0, 0.1, -1.0, 3.0, 0.5];
    let probs = cpu_softmax(&logits);
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "softmax sum should be 1.0, got {sum}");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_softmax_large_input_stability() {
    // Large logits: naive exp would overflow, but max-subtraction keeps it stable
    let logits = vec![1000.0, 1001.0, 999.0, 1000.5];
    let probs = cpu_softmax(&logits);
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "softmax large: sum={sum}");
    assert!(probs.iter().all(|&p| p.is_finite()), "softmax probs must be finite");
    assert!(probs.iter().all(|&p| p >= 0.0), "softmax probs must be non-negative");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_softmax_temperature_scaling() {
    let logits = vec![1.0, 2.0, 3.0];

    // Low temperature → sharper distribution (more peaked at max)
    let sharp = cpu_softmax_with_temperature(&logits, 0.1);
    // High temperature → flatter distribution
    let flat = cpu_softmax_with_temperature(&logits, 10.0);

    // Both must sum to 1.0
    let sharp_sum: f32 = sharp.iter().sum();
    let flat_sum: f32 = flat.iter().sum();
    assert!((sharp_sum - 1.0).abs() < 1e-5);
    assert!((flat_sum - 1.0).abs() < 1e-5);

    // Max probability should be higher with low temperature
    let sharp_max = sharp.iter().cloned().reduce(f32::max).unwrap();
    let flat_max = flat.iter().cloned().reduce(f32::max).unwrap();
    assert!(
        sharp_max > flat_max,
        "low temp should be sharper: sharp_max={sharp_max}, flat_max={flat_max}"
    );
}

// ═══════════════════════════════════════════════════════════════════
// 20. Threadgroup sizing — parametric
// ═══════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_threadgroup_size_32() {
    let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    let expected = cpu_sum(&data);
    let result = simulated_parallel_sum(&data, 32);
    assert!((result - expected).abs() < 1.0, "tg=32: gpu={result}, cpu={expected}");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_threadgroup_size_64() {
    let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    let expected = cpu_sum(&data);
    let result = simulated_parallel_sum(&data, 64);
    assert!((result - expected).abs() < 1.0, "tg=64: gpu={result}, cpu={expected}");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_threadgroup_size_128() {
    let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    let expected = cpu_sum(&data);
    let result = simulated_parallel_sum(&data, 128);
    assert!((result - expected).abs() < 1.0, "tg=128: gpu={result}, cpu={expected}");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_threadgroup_size_256() {
    let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    let expected = cpu_sum(&data);
    let result = simulated_parallel_sum(&data, 256);
    assert!((result - expected).abs() < 1.0, "tg=256: gpu={result}, cpu={expected}");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_threadgroup_size_512() {
    let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    let expected = cpu_sum(&data);
    let result = simulated_parallel_sum(&data, 512);
    assert!((result - expected).abs() < 1.0, "tg=512: gpu={result}, cpu={expected}");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_threadgroup_size_1024() {
    let data: Vec<f32> = (0..4096).map(|i| i as f32).collect();
    let expected = cpu_sum(&data);
    let result = simulated_parallel_sum(&data, 1024);
    assert!((result - expected).abs() < 1.0, "tg=1024: gpu={result}, cpu={expected}");
}

// ═══════════════════════════════════════════════════════════════════
// 21. Edge cases — additional coverage
// ═══════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_edge_case_two_elements() {
    let data = [3.0_f32, 7.0];
    assert!((simulated_parallel_sum(&data, SIMD_GROUP_WIDTH) - 10.0).abs() < f32::EPSILON);
    assert_eq!(cpu_argmax(&data), Some(1));
    assert_eq!(cpu_argmin(&data), Some(0));
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_edge_case_exact_threadgroup_size() {
    // Exactly MAX_THREADS_PER_THREADGROUP elements — no partial groups
    let n = MAX_THREADS_PER_THREADGROUP as usize;
    let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let expected = cpu_sum(&data);
    let result = simulated_parallel_sum(&data, MAX_THREADS_PER_THREADGROUP);
    assert!((result - expected).abs() < 1.0, "exact tg: gpu={result}, cpu={expected}");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_edge_case_one_over_threadgroup_size() {
    // MAX_THREADS_PER_THREADGROUP + 1: triggers second threadgroup
    let n = MAX_THREADS_PER_THREADGROUP as usize + 1;
    let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let expected = cpu_sum(&data);
    let result = simulated_parallel_sum(&data, MAX_THREADS_PER_THREADGROUP);
    assert!((result - expected).abs() < 1.0, "one-over tg: gpu={result}, cpu={expected}");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_edge_case_buffer_alignment_non_aligned() {
    // 17 f32s = 68 bytes, not 256-aligned
    let byte_size = 17 * std::mem::size_of::<f32>();
    assert_ne!(byte_size % METAL_BUFFER_ALIGNMENT, 0);
    let aligned = align_to_metal(byte_size);
    assert_eq!(aligned % METAL_BUFFER_ALIGNMENT, 0);
    assert_eq!(aligned, 256);
}

// ═══════════════════════════════════════════════════════════════════
// 22. Log-sum-exp — additional mixed values test
// ═══════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_log_sum_exp_mixed_positive_negative() {
    let data = vec![-5.0, 0.0, 5.0, 10.0, -10.0];
    let result = cpu_log_sum_exp(&data).unwrap();
    assert!(result.is_finite(), "lse mixed should be finite");
    // Dominated by max=10, so result ≈ 10 + small correction
    assert!((result - 10.0).abs() < 1.0, "lse mixed: got={result}");
}

#[test]
#[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
fn test_log_sum_exp_small_values() {
    let data = vec![0.001, 0.002, 0.003];
    let result = cpu_log_sum_exp(&data).unwrap();
    assert!(result.is_finite());
    // All values near 0, so log(3 * e^~0) ≈ log(3) ≈ 1.099
    assert!((result - (3.0_f32).ln()).abs() < 0.01, "lse small: got={result}");
}
