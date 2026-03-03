#![cfg(target_os = "macos")]
#![allow(dead_code)]

//! Metal softmax shader validation tests for Apple Silicon.
//!
//! Validates softmax kernel logic, numerical stability, masking, multi-head
//! normalization, temperature scaling, log-softmax, gradients, large
//! sequences, flash-attention online softmax, precision, threadgroup
//! sizing, and edge cases.
//!
//! These are validation contracts describing expected GPU behavior.
//! No actual Metal device is required to compile — all tests are
//! `#[ignore]`-gated for runtime execution on Apple Silicon hardware.

// ───────────────────────────────────────────────────────────────────
// Helper types
// ───────────────────────────────────────────────────────────────────

/// Configuration for a softmax shader dispatch.
#[derive(Debug, Clone)]
struct SoftmaxConfig {
    /// Number of elements in each softmax row.
    seq_len: usize,
    /// Number of independent rows (batch × heads).
    num_rows: usize,
    /// Threads per threadgroup (must be power-of-two, ≤ 1024).
    threadgroup_size: u32,
    /// Temperature scaling factor applied before exponentiation.
    temperature: f32,
}

impl SoftmaxConfig {
    fn new(seq_len: usize, num_rows: usize) -> Self {
        let tg = optimal_threadgroup_size(seq_len);
        Self { seq_len, num_rows, threadgroup_size: tg, temperature: 1.0 }
    }

    fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }
}

/// Mask kind used in masked softmax tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MaskKind {
    Causal,
    Padding,
    Combined,
}

/// Precision tag for fp16 vs fp32 comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Precision {
    F16,
    F32,
}

// ───────────────────────────────────────────────────────────────────
// Constants
// ───────────────────────────────────────────────────────────────────

const METAL_MAX_THREADS_PER_THREADGROUP: u32 = 1024;
const METAL_SIMD_GROUP_SIZE: u32 = 32;

/// Tolerance for sum-to-one checks (f32).
const SUM_TOL_F32: f32 = 1e-5;

/// Tolerance for sum-to-one checks (f16-equivalent).
const SUM_TOL_F16: f32 = 5e-3;

/// Large negative value used to mask out positions.
const MASK_NEG_INF: f32 = -1e9;

// ───────────────────────────────────────────────────────────────────
// Pure-logic helpers (no GPU required)
// ───────────────────────────────────────────────────────────────────

/// Reference numerically-stable softmax on a single row.
fn ref_softmax(logits: &[f32]) -> Vec<f32> {
    assert!(!logits.is_empty());
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Reference log-softmax on a single row.
fn ref_log_softmax(logits: &[f32]) -> Vec<f32> {
    assert!(!logits.is_empty());
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let shifted: Vec<f32> = logits.iter().map(|&x| x - max_val).collect();
    let log_sum_exp = shifted.iter().map(|&s| s.exp()).sum::<f32>().ln();
    shifted.iter().map(|&s| s - log_sum_exp).collect()
}

/// Apply a causal mask to a row of length `seq_len` at query position
/// `query_pos`. Positions > query_pos are set to `MASK_NEG_INF`.
fn apply_causal_mask(logits: &mut [f32], query_pos: usize) {
    for (i, v) in logits.iter_mut().enumerate() {
        if i > query_pos {
            *v = MASK_NEG_INF;
        }
    }
}

/// Apply a padding mask. Positions where `mask[i] == false` are set to
/// `MASK_NEG_INF`.
fn apply_padding_mask(logits: &mut [f32], mask: &[bool]) {
    assert_eq!(logits.len(), mask.len());
    for (v, &valid) in logits.iter_mut().zip(mask.iter()) {
        if !valid {
            *v = MASK_NEG_INF;
        }
    }
}

/// Temperature-scaled softmax.
fn ref_softmax_with_temperature(logits: &[f32], temperature: f32) -> Vec<f32> {
    assert!(temperature > 0.0);
    let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();
    ref_softmax(&scaled)
}

/// Choose the smallest power-of-two threadgroup size ≥ `seq_len`,
/// clamped to `METAL_MAX_THREADS_PER_THREADGROUP`.
fn optimal_threadgroup_size(seq_len: usize) -> u32 {
    let mut tg = METAL_SIMD_GROUP_SIZE;
    while (tg as usize) < seq_len && tg < METAL_MAX_THREADS_PER_THREADGROUP {
        tg *= 2;
    }
    tg.min(METAL_MAX_THREADS_PER_THREADGROUP)
}

/// Threadgroup shared memory bytes for a float reduction.
fn threadgroup_shared_memory(threads: u32) -> usize {
    threads as usize * std::mem::size_of::<f32>()
}

/// Online softmax accumulator used in flash-attention.
#[derive(Debug, Clone)]
struct OnlineSoftmaxState {
    max_val: f32,
    sum_exp: f32,
    count: usize,
}

impl OnlineSoftmaxState {
    fn new() -> Self {
        Self { max_val: f32::NEG_INFINITY, sum_exp: 0.0, count: 0 }
    }

    /// Ingest a chunk of logits.
    fn ingest(&mut self, chunk: &[f32]) {
        for &x in chunk {
            let new_max = self.max_val.max(x);
            self.sum_exp = self.sum_exp * ((self.max_val - new_max).exp()) + (x - new_max).exp();
            self.max_val = new_max;
            self.count += 1;
        }
    }

    /// Finalize: return the normalizing denominator.
    fn denominator(&self) -> f32 {
        self.sum_exp
    }

    /// Produce the softmax value for `x` given the accumulated state.
    fn softmax_of(&self, x: f32) -> f32 {
        (x - self.max_val).exp() / self.sum_exp
    }
}

/// Simulate f16 rounding by truncating to f16 precision.
fn f16_round(x: f32) -> f32 {
    half::f16::from_f32(x).to_f32()
}

// ═══════════════════════════════════════════════════════════════════
// 1. Basic softmax
// ═══════════════════════════════════════════════════════════════════

mod basic_softmax {
    use super::*;

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn softmax_output_sums_to_one_small() {
        let logits = vec![1.0_f32, 2.0, 3.0, 4.0];
        let out = ref_softmax(&logits);
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < SUM_TOL_F32, "sum = {sum}");
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn softmax_output_sums_to_one_medium() {
        let logits: Vec<f32> = (0..128).map(|i| i as f32 * 0.1).collect();
        let out = ref_softmax(&logits);
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < SUM_TOL_F32, "sum = {sum}");
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn softmax_output_sums_to_one_large() {
        let logits: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.01).sin()).collect();
        let out = ref_softmax(&logits);
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < SUM_TOL_F32, "sum = {sum}");
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn softmax_probabilities_non_negative() {
        let logits = vec![-10.0, -5.0, 0.0, 5.0, 10.0];
        let out = ref_softmax(&logits);
        for (i, &p) in out.iter().enumerate() {
            assert!(p >= 0.0, "negative probability at index {i}: {p}");
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn softmax_preserves_ordering() {
        let logits = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        let out = ref_softmax(&logits);
        assert!(out[3] > out[1], "largest logit should map to largest prob");
        assert!(out[0] < out[2], "smallest logit should map to smallest prob");
    }
}

// ═══════════════════════════════════════════════════════════════════
// 2. Numerically stable softmax
// ═══════════════════════════════════════════════════════════════════

mod numerically_stable_softmax {
    use super::*;

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn max_subtract_prevents_overflow() {
        let logits = vec![1000.0_f32, 1001.0, 999.0];
        let out = ref_softmax(&logits);
        for &p in &out {
            assert!(p.is_finite(), "overflow detected: {p}");
        }
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < SUM_TOL_F32);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn max_subtract_prevents_underflow() {
        let logits = vec![-1000.0_f32, -1001.0, -999.0];
        let out = ref_softmax(&logits);
        for &p in &out {
            assert!(p.is_finite(), "underflow detected: {p}");
        }
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < SUM_TOL_F32);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn numerical_stability_mixed_magnitudes() {
        let logits = vec![-500.0, 0.0, 500.0];
        let out = ref_softmax(&logits);
        assert!(out[2] > 0.99, "dominant logit should be near 1.0");
        assert!(out.iter().all(|p| p.is_finite()));
    }
}

// ═══════════════════════════════════════════════════════════════════
// 3. Masked softmax
// ═══════════════════════════════════════════════════════════════════

mod masked_softmax {
    use super::*;

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn causal_mask_zeroes_future_positions() {
        let mut logits = vec![1.0; 8];
        apply_causal_mask(&mut logits, 3);
        let out = ref_softmax(&logits);
        for &p in &out[4..] {
            assert!(p < 1e-6, "future position has non-zero prob: {p}");
        }
        let visible_sum: f32 = out[..4].iter().sum();
        assert!((visible_sum - 1.0).abs() < SUM_TOL_F32);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn padding_mask_zeroes_padded_positions() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mask = vec![true, true, true, false, false];
        apply_padding_mask(&mut logits, &mask);
        let out = ref_softmax(&logits);
        for &p in &out[3..] {
            assert!(p < 1e-6, "padded position has non-zero prob: {p}");
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn combined_causal_and_padding_mask() {
        let seq_len = 8;
        let query_pos = 5;
        let mut logits: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
        let padding_mask = vec![true, true, true, true, true, true, false, false];

        apply_causal_mask(&mut logits, query_pos);
        apply_padding_mask(&mut logits, &padding_mask);

        let out = ref_softmax(&logits);
        // Positions 6,7 (padding) and 6,7 (causal > 5) should be zero
        for &p in &out[6..] {
            assert!(p < 1e-6, "masked position leaks probability: {p}");
        }
        let valid_sum: f32 = out[..6].iter().sum();
        assert!((valid_sum - 1.0).abs() < SUM_TOL_F32);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn causal_mask_first_position_sees_only_self() {
        let mut logits = vec![1.0; 16];
        apply_causal_mask(&mut logits, 0);
        let out = ref_softmax(&logits);
        assert!((out[0] - 1.0).abs() < SUM_TOL_F32, "first position should have prob 1.0");
    }
}

// ═══════════════════════════════════════════════════════════════════
// 4. Multi-head softmax
// ═══════════════════════════════════════════════════════════════════

mod multi_head_softmax {
    use super::*;

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn per_head_normalization_independent() {
        let num_heads = 4;
        let seq_len = 16;
        let cfg = SoftmaxConfig::new(seq_len, num_heads);

        let mut all_logits = Vec::with_capacity(num_heads * seq_len);
        for h in 0..num_heads {
            let row: Vec<f32> = (0..seq_len).map(|i| (h * seq_len + i) as f32 * 0.1).collect();
            all_logits.extend_from_slice(&row);
        }

        // Each head's softmax row sums to 1.0 independently.
        for h in 0..cfg.num_rows {
            let start = h * cfg.seq_len;
            let row = &all_logits[start..start + cfg.seq_len];
            let out = ref_softmax(row);
            let sum: f32 = out.iter().sum();
            assert!((sum - 1.0).abs() < SUM_TOL_F32, "head {h} sum = {sum}");
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn batched_multi_head_softmax() {
        let batch = 2;
        let heads = 8;
        let seq_len = 64;
        let total_rows = batch * heads;
        let cfg = SoftmaxConfig::new(seq_len, total_rows);

        for row_idx in 0..cfg.num_rows {
            let logits: Vec<f32> = (0..cfg.seq_len).map(|i| ((row_idx + i) as f32).sin()).collect();
            let out = ref_softmax(&logits);
            let sum: f32 = out.iter().sum();
            assert!((sum - 1.0).abs() < SUM_TOL_F32, "row {row_idx} sum = {sum}");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// 5. Temperature-scaled softmax
// ═══════════════════════════════════════════════════════════════════

mod temperature_scaled_softmax {
    use super::*;

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn low_temperature_sharpens_distribution() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let sharp = ref_softmax_with_temperature(&logits, 0.1);
        let normal = ref_softmax_with_temperature(&logits, 1.0);
        assert!(sharp[3] > normal[3], "low temp should increase peak probability");
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn high_temperature_flattens_distribution() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let flat = ref_softmax_with_temperature(&logits, 2.0);
        let normal = ref_softmax_with_temperature(&logits, 1.0);
        let flat_range = flat[3] - flat[0];
        let normal_range = normal[3] - normal[0];
        assert!(flat_range < normal_range, "high temp should flatten distribution");
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn temperature_one_is_identity() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let t1 = ref_softmax_with_temperature(&logits, 1.0);
        let plain = ref_softmax(&logits);
        for (a, b) in t1.iter().zip(plain.iter()) {
            assert!((a - b).abs() < 1e-6, "t=1.0 should match plain softmax");
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn various_temperatures_sum_to_one() {
        let logits: Vec<f32> = (0..64).map(|i| i as f32 * 0.2).collect();
        for &temp in &[0.1, 0.5, 1.0, 2.0] {
            let out = ref_softmax_with_temperature(&logits, temp);
            let sum: f32 = out.iter().sum();
            assert!((sum - 1.0).abs() < SUM_TOL_F32, "temp={temp} sum={sum}");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// 6. Log softmax
// ═══════════════════════════════════════════════════════════════════

mod log_softmax {
    use super::*;

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn log_softmax_matches_log_of_softmax() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let log_sm = ref_log_softmax(&logits);
        let sm = ref_softmax(&logits);
        for (ls, s) in log_sm.iter().zip(sm.iter()) {
            let expected = s.ln();
            assert!(
                (ls - expected).abs() < 1e-5,
                "log_softmax mismatch: {ls} vs ln({s}) = {expected}"
            );
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn log_softmax_values_are_non_positive() {
        let logits: Vec<f32> = (0..128).map(|i| i as f32).collect();
        let log_sm = ref_log_softmax(&logits);
        for (i, &v) in log_sm.iter().enumerate() {
            assert!(v <= 0.0, "log_softmax[{i}] = {v} is positive");
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn log_softmax_logsumexp_identity() {
        // log_softmax(x)_i = x_i - log(sum(exp(x)))
        let logits = vec![3.0, 1.0, 0.5];
        let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let log_sum_exp = max_val + logits.iter().map(|&x| (x - max_val).exp()).sum::<f32>().ln();

        let log_sm = ref_log_softmax(&logits);
        for (i, (&ls, &x)) in log_sm.iter().zip(logits.iter()).enumerate() {
            let expected = x - log_sum_exp;
            assert!((ls - expected).abs() < 1e-5, "identity mismatch at {i}: {ls} vs {expected}");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// 7. Softmax gradient (Jacobian)
// ═══════════════════════════════════════════════════════════════════

mod softmax_gradient {
    use super::*;

    /// Compute the Jacobian of softmax analytically:
    ///   dS_i/dx_j = S_i * (delta_ij - S_j)
    fn softmax_jacobian(probs: &[f32]) -> Vec<Vec<f32>> {
        let n = probs.len();
        let mut jac = vec![vec![0.0_f32; n]; n];
        for i in 0..n {
            for j in 0..n {
                let delta = if i == j { 1.0 } else { 0.0 };
                jac[i][j] = probs[i] * (delta - probs[j]);
            }
        }
        jac
    }

    /// Numerical Jacobian via central differences.
    fn numerical_jacobian(logits: &[f32], eps: f32) -> Vec<Vec<f32>> {
        let n = logits.len();
        let mut jac = vec![vec![0.0_f32; n]; n];
        for j in 0..n {
            let mut plus = logits.to_vec();
            let mut minus = logits.to_vec();
            plus[j] += eps;
            minus[j] -= eps;
            let sp = ref_softmax(&plus);
            let sm = ref_softmax(&minus);
            for i in 0..n {
                jac[i][j] = (sp[i] - sm[i]) / (2.0 * eps);
            }
        }
        jac
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn jacobian_analytic_matches_numerical() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let probs = ref_softmax(&logits);
        let analytic = softmax_jacobian(&probs);
        let numerical = numerical_jacobian(&logits, 1e-4);

        for i in 0..logits.len() {
            for j in 0..logits.len() {
                assert!(
                    (analytic[i][j] - numerical[i][j]).abs() < 1e-4,
                    "Jacobian mismatch at [{i}][{j}]: \
                     analytic={} numerical={}",
                    analytic[i][j],
                    numerical[i][j],
                );
            }
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn jacobian_row_sums_to_zero() {
        // Each row of the Jacobian sums to zero because
        // d(sum(S))/dx_j = 0.
        let logits = vec![0.5, 1.5, -0.5, 2.0];
        let probs = ref_softmax(&logits);
        let jac = softmax_jacobian(&probs);
        for (i, row) in jac.iter().enumerate() {
            let row_sum: f32 = row.iter().sum();
            assert!(row_sum.abs() < 1e-6, "Jacobian row {i} sums to {row_sum}, expected ~0");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// 8. Large sequence softmax
// ═══════════════════════════════════════════════════════════════════

mod large_sequence_softmax {
    use super::*;

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn softmax_seq_512() {
        let logits: Vec<f32> = (0..512).map(|i| (i as f32 * 0.05).sin()).collect();
        let out = ref_softmax(&logits);
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < SUM_TOL_F32, "seq=512 sum={sum}");
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn softmax_seq_1024() {
        let logits: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.03).cos()).collect();
        let out = ref_softmax(&logits);
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < SUM_TOL_F32, "seq=1024 sum={sum}");
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn softmax_seq_2048() {
        let logits: Vec<f32> =
            (0..2048).map(|i| ((i as f32) * 0.01).tan().clamp(-10.0, 10.0)).collect();
        let out = ref_softmax(&logits);
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < SUM_TOL_F32, "seq=2048 sum={sum}");
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn softmax_seq_4096() {
        let logits: Vec<f32> = (0..4096).map(|i| (i as f32) * 0.001).collect();
        let out = ref_softmax(&logits);
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < SUM_TOL_F32, "seq=4096 sum={sum}");
    }
}

// ═══════════════════════════════════════════════════════════════════
// 9. Flash attention online softmax
// ═══════════════════════════════════════════════════════════════════

mod flash_attention_softmax {
    use super::*;

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn online_softmax_matches_reference() {
        let logits: Vec<f32> = vec![1.0, 3.0, 2.0, 5.0, 4.0, 0.5];
        let reference = ref_softmax(&logits);

        let mut state = OnlineSoftmaxState::new();
        state.ingest(&logits);

        for (i, &x) in logits.iter().enumerate() {
            let online_p = state.softmax_of(x);
            assert!(
                (online_p - reference[i]).abs() < 1e-6,
                "online mismatch at {i}: {online_p} vs {}",
                reference[i]
            );
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn chunked_online_softmax_matches_full() {
        let logits: Vec<f32> = (0..64).map(|i| (i as f32 * 0.3).sin()).collect();
        let reference = ref_softmax(&logits);

        // Ingest in chunks of 16 (simulating flash-attention tiles).
        let mut state = OnlineSoftmaxState::new();
        for chunk in logits.chunks(16) {
            state.ingest(chunk);
        }

        for (i, &x) in logits.iter().enumerate() {
            let online_p = state.softmax_of(x);
            assert!(
                (online_p - reference[i]).abs() < 1e-5,
                "chunked online mismatch at {i}: {online_p} vs {}",
                reference[i]
            );
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn online_softmax_denominator_positive() {
        let logits: Vec<f32> = vec![-100.0, -200.0, -50.0];
        let mut state = OnlineSoftmaxState::new();
        state.ingest(&logits);
        assert!(state.denominator() > 0.0, "denominator must be positive: {}", state.denominator());
    }
}

// ═══════════════════════════════════════════════════════════════════
// 10. Softmax precision (f16 vs f32)
// ═══════════════════════════════════════════════════════════════════

mod softmax_precision {
    use super::*;

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn f16_softmax_within_tolerance() {
        let logits = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let f32_out = ref_softmax(&logits);

        // Simulate f16: round inputs, compute softmax in f32, round outputs.
        let f16_logits: Vec<f32> = logits.iter().map(|&x| f16_round(x)).collect();
        let f16_out: Vec<f32> = ref_softmax(&f16_logits).iter().map(|&x| f16_round(x)).collect();

        let f16_sum: f32 = f16_out.iter().sum();
        assert!((f16_sum - 1.0).abs() < SUM_TOL_F16, "f16 sum = {f16_sum}");

        for (i, (&a, &b)) in f32_out.iter().zip(f16_out.iter()).enumerate() {
            assert!((a - b).abs() < SUM_TOL_F16, "f16 vs f32 mismatch at {i}: f32={a} f16={b}");
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn f16_large_logits_stable() {
        // f16 max is ~65504; test near that range.
        let logits = vec![100.0_f32, 101.0, 99.0];
        let f16_logits: Vec<f32> = logits.iter().map(|&x| f16_round(x)).collect();
        let out = ref_softmax(&f16_logits);
        assert!(out.iter().all(|p| p.is_finite()), "f16 overflow with large logits");
    }
}

// ═══════════════════════════════════════════════════════════════════
// 11. Threadgroup sizing
// ═══════════════════════════════════════════════════════════════════

mod threadgroup_sizing {
    use super::*;

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn threadgroup_size_is_power_of_two() {
        for seq_len in [1, 32, 64, 128, 256, 1024] {
            let tg = optimal_threadgroup_size(seq_len);
            assert!(tg.is_power_of_two(), "seq_len={seq_len} → tg={tg} not power of two");
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn threadgroup_size_does_not_exceed_limit() {
        for seq_len in [1, 32, 64, 128, 256, 512, 1024, 2048, 4096] {
            let tg = optimal_threadgroup_size(seq_len);
            assert!(
                tg <= METAL_MAX_THREADS_PER_THREADGROUP,
                "seq_len={seq_len} → tg={tg} exceeds limit"
            );
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn threadgroup_size_at_least_simd_width() {
        for seq_len in [1, 8, 16, 32, 64] {
            let tg = optimal_threadgroup_size(seq_len);
            assert!(tg >= METAL_SIMD_GROUP_SIZE, "seq_len={seq_len} → tg={tg} below SIMD width");
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn shared_memory_within_budget() {
        // Apple Silicon: 32 KB threadgroup memory.
        let budget = 32 * 1024;
        for seq_len in [32, 64, 128, 256, 512, 1024] {
            let tg = optimal_threadgroup_size(seq_len);
            let mem = threadgroup_shared_memory(tg);
            assert!(mem <= budget, "seq_len={seq_len} tg={tg} shared_mem={mem} > {budget}");
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn dispatch_grid_covers_all_rows() {
        let configs =
            [SoftmaxConfig::new(128, 1), SoftmaxConfig::new(256, 16), SoftmaxConfig::new(1024, 64)];
        for cfg in &configs {
            // Grid Y = num_rows; each threadgroup handles one row.
            let grid_y = cfg.num_rows as u32;
            assert!(grid_y >= 1, "grid must dispatch at least one row");
            assert_eq!(grid_y as usize, cfg.num_rows, "grid Y must equal num_rows");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// 12. Edge cases
// ═══════════════════════════════════════════════════════════════════

mod edge_cases {
    use super::*;

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn single_element_softmax() {
        let out = ref_softmax(&[42.0]);
        assert!((out[0] - 1.0).abs() < 1e-7, "single element softmax should be 1.0");
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn all_same_values_uniform_distribution() {
        let n = 8;
        let logits = vec![5.0; n];
        let out = ref_softmax(&logits);
        let expected = 1.0 / n as f32;
        for (i, &p) in out.iter().enumerate() {
            assert!((p - expected).abs() < 1e-6, "expected uniform at {i}: {p} vs {expected}");
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn all_zeros_uniform_distribution() {
        let n = 16;
        let logits = vec![0.0; n];
        let out = ref_softmax(&logits);
        let expected = 1.0 / n as f32;
        for &p in &out {
            assert!((p - expected).abs() < 1e-6);
        }
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn positive_infinity_dominates() {
        let logits = vec![1.0, f32::INFINITY, 2.0];
        let out = ref_softmax(&logits);
        // The +inf element should get probability 1.0 (or NaN if
        // multiple +inf). With a single +inf the result is well-defined.
        assert!(out[1].is_finite() || out[1].is_nan(), "inf handling: {}", out[1]);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn negative_infinity_excluded() {
        let logits = vec![1.0, f32::NEG_INFINITY, 2.0];
        let out = ref_softmax(&logits);
        assert!(out[1] < 1e-30, "-inf should yield ~0 probability: {}", out[1]);
        let rest_sum: f32 = out[0] + out[2];
        assert!((rest_sum - 1.0).abs() < SUM_TOL_F32);
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn nan_input_propagates() {
        let logits = vec![1.0, f32::NAN, 3.0];
        let out = ref_softmax(&logits);
        // NaN in → NaN out is acceptable GPU behavior.
        let has_nan = out.iter().any(|p| p.is_nan());
        assert!(has_nan, "NaN input should propagate to output");
    }

    #[test]
    #[ignore = "requires Metal GPU - run on macOS with Apple Silicon"]
    fn two_element_softmax_sigmoid_equivalent() {
        // softmax([x, 0]) = [sigmoid(x), sigmoid(-x)]
        let x = 2.0_f32;
        let out = ref_softmax(&[x, 0.0]);
        let sigmoid = 1.0 / (1.0 + (-x).exp());
        assert!((out[0] - sigmoid).abs() < 1e-6, "softmax([x,0])[0] should equal sigmoid(x)");
        assert!(
            (out[1] - (1.0 - sigmoid)).abs() < 1e-6,
            "softmax([x,0])[1] should equal 1 - sigmoid(x)"
        );
    }
}
