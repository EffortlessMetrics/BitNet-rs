//! Metal softmax v2 shader validation tests for Apple Silicon.
//!
//! Validates softmax kernel logic patterns in pure Rust without requiring
//! Metal GPU hardware. Categories:
//! - Standard softmax (exp(x-max)/sum)
//! - Online softmax (single-pass numerically stable)
//! - Causal softmax (autoregressive masking)
//! - Multi-head softmax (per-head independent computation)
//! - Temperature scaling
//! - Top-k masking
//! - Numerical stability (overflow/underflow protection)
//! - Flash attention tile-based softmax
//! - Grouped query attention (GQA) softmax

#![allow(dead_code)]

// ──────────────────────────────────────────────────────────────
// Constants
// ──────────────────────────────────────────────────────────────

const EPS: f32 = 1e-5;
const EPS_ACCUMULATED: f32 = 1e-4;
const NEG_INF: f32 = f32::NEG_INFINITY;

// ──────────────────────────────────────────────────────────────
// Reference implementations
// ──────────────────────────────────────────────────────────────

/// Standard softmax: exp(x_i - max) / sum(exp(x_j - max))
fn softmax(input: &[f32]) -> Vec<f32> {
    if input.is_empty() {
        return vec![];
    }
    let max_val = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = input.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Softmax with temperature scaling: exp((x_i / T) - max) / sum(...)
fn softmax_with_temperature(input: &[f32], temperature: f32) -> Vec<f32> {
    assert!(temperature > 0.0, "temperature must be positive");
    let scaled: Vec<f32> = input.iter().map(|&x| x / temperature).collect();
    softmax(&scaled)
}

/// Online softmax: single-pass numerically stable algorithm.
/// Tracks running max and denominator in one sweep.
fn online_softmax(input: &[f32]) -> Vec<f32> {
    if input.is_empty() {
        return vec![];
    }
    let n = input.len();
    let mut max_val = f32::NEG_INFINITY;
    let mut denom = 0.0f32;

    // Forward pass: accumulate max and denominator
    for &x in input {
        if x > max_val {
            denom = denom * (max_val - x).exp() + (0.0f32).exp();
            max_val = x;
        } else {
            denom += (x - max_val).exp();
        }
    }

    // Normalize
    let mut output = vec![0.0f32; n];
    for (i, &x) in input.iter().enumerate() {
        output[i] = (x - max_val).exp() / denom;
    }
    output
}

/// Causal mask: positions j > i are masked to NEG_INF.
fn apply_causal_mask(input: &[f32], seq_len: usize, query_pos: usize) -> Vec<f32> {
    assert!(input.len() >= seq_len);
    let mut masked = input[..seq_len].to_vec();
    for j in (query_pos + 1)..seq_len {
        masked[j] = NEG_INF;
    }
    masked
}

/// Top-k masking: keep only top k values, rest set to NEG_INF.
fn top_k_mask(input: &[f32], k: usize) -> Vec<f32> {
    if k >= input.len() {
        return input.to_vec();
    }
    let mut indexed: Vec<(usize, f32)> = input.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut output = vec![NEG_INF; input.len()];
    for &(idx, val) in indexed.iter().take(k) {
        output[idx] = val;
    }
    output
}

/// Flash attention tile-based softmax rescaling.
/// Given local softmax over a tile and previous running stats,
/// returns rescaled output combining old and new tiles.
fn flash_attention_softmax_rescale(
    old_output: &[f32],
    old_max: f32,
    old_denom: f32,
    new_tile: &[f32],
) -> (Vec<f32>, f32, f32) {
    let new_max = new_tile.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let global_max = old_max.max(new_max);

    let old_scale = (old_max - global_max).exp();
    let new_denom_local: f32 = new_tile.iter().map(|&x| (x - global_max).exp()).sum();
    let new_denom = old_denom * old_scale + new_denom_local;

    let rescaled: Vec<f32> =
        old_output.iter().map(|&o| o * old_denom * old_scale / new_denom).collect();

    (rescaled, global_max, new_denom)
}

fn assert_close(a: f32, b: f32, eps: f32) {
    assert!((a - b).abs() < eps, "expected {a} ≈ {b} (diff={}, eps={eps})", (a - b).abs());
}

fn assert_vec_close(a: &[f32], b: &[f32], eps: f32) {
    assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
    for (i, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (va - vb).abs() < eps,
            "index {i}: {va} ≈ {vb} (diff={}, eps={eps})",
            (va - vb).abs()
        );
    }
}

fn assert_is_probability_distribution(p: &[f32], eps: f32) {
    let sum: f32 = p.iter().sum();
    assert_close(sum, 1.0, eps);
    for (i, &v) in p.iter().enumerate() {
        assert!(v >= 0.0, "negative probability at index {i}: {v}");
        assert!(v <= 1.0, "probability > 1 at index {i}: {v}");
    }
}

// ══════════════════════════════════════════════════════════════
// 1. Standard softmax tests
// ══════════════════════════════════════════════════════════════

#[test]
fn standard_softmax_uniform_input() {
    let input = vec![1.0; 8];
    let result = softmax(&input);
    assert_is_probability_distribution(&result, EPS);
    for &v in &result {
        assert_close(v, 0.125, EPS);
    }
}

#[test]
fn standard_softmax_known_values() {
    let input = vec![1.0, 2.0, 3.0];
    let result = softmax(&input);
    assert_is_probability_distribution(&result, EPS);
    // exp(1)+exp(2)+exp(3) ≈ 2.718+7.389+20.086 = 30.193
    assert_close(result[0], (1.0f32).exp() / 30.1929, EPS_ACCUMULATED);
    assert_close(result[2], (3.0f32).exp() / 30.1929, EPS_ACCUMULATED);
    // Monotonicity
    assert!(result[0] < result[1]);
    assert!(result[1] < result[2]);
}

#[test]
fn standard_softmax_single_element() {
    let result = softmax(&[42.0]);
    assert_eq!(result.len(), 1);
    assert_close(result[0], 1.0, EPS);
}

#[test]
fn standard_softmax_two_elements_symmetric() {
    let result = softmax(&[0.0, 0.0]);
    assert_close(result[0], 0.5, EPS);
    assert_close(result[1], 0.5, EPS);
}

#[test]
fn standard_softmax_large_difference() {
    // One dominant element should get ~1.0
    let input = vec![0.0, 0.0, 100.0, 0.0];
    let result = softmax(&input);
    assert_is_probability_distribution(&result, EPS);
    assert!(result[2] > 0.999);
}

#[test]
fn standard_softmax_negative_inputs() {
    let input = vec![-1.0, -2.0, -3.0, -4.0];
    let result = softmax(&input);
    assert_is_probability_distribution(&result, EPS);
    assert!(result[0] > result[1]);
    assert!(result[1] > result[2]);
    assert!(result[2] > result[3]);
}

#[test]
fn standard_softmax_zeros() {
    let input = vec![0.0; 16];
    let result = softmax(&input);
    assert_is_probability_distribution(&result, EPS);
    for &v in &result {
        assert_close(v, 1.0 / 16.0, EPS);
    }
}

#[test]
fn standard_softmax_preserves_ordering() {
    let input = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let result = softmax(&input);
    assert_is_probability_distribution(&result, EPS);
    // result[5] (input=9) should be largest
    let max_idx = result.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    assert_eq!(max_idx, 5);
}

#[test]
fn standard_softmax_empty_input() {
    let result = softmax(&[]);
    assert!(result.is_empty());
}

// ══════════════════════════════════════════════════════════════
// 2. Online softmax tests
// ══════════════════════════════════════════════════════════════

#[test]
fn online_softmax_matches_standard() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let standard = softmax(&input);
    let online = online_softmax(&input);
    assert_vec_close(&standard, &online, EPS);
}

#[test]
fn online_softmax_uniform() {
    let input = vec![7.0; 32];
    let result = online_softmax(&input);
    assert_is_probability_distribution(&result, EPS);
    for &v in &result {
        assert_close(v, 1.0 / 32.0, EPS);
    }
}

#[test]
fn online_softmax_large_range() {
    let input = vec![-100.0, 0.0, 50.0, 100.0];
    let standard = softmax(&input);
    let online = online_softmax(&input);
    assert_vec_close(&standard, &online, EPS);
}

#[test]
fn online_softmax_descending_order() {
    let input = vec![10.0, 8.0, 6.0, 4.0, 2.0, 0.0];
    let standard = softmax(&input);
    let online = online_softmax(&input);
    assert_vec_close(&standard, &online, EPS);
}

#[test]
fn online_softmax_ascending_order() {
    let input = vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0];
    let standard = softmax(&input);
    let online = online_softmax(&input);
    assert_vec_close(&standard, &online, EPS);
}

#[test]
fn online_softmax_single_element() {
    let result = online_softmax(&[3.14]);
    assert_eq!(result.len(), 1);
    assert_close(result[0], 1.0, EPS);
}

#[test]
fn online_softmax_negative_values() {
    let input = vec![-5.0, -3.0, -1.0, -7.0];
    let standard = softmax(&input);
    let online = online_softmax(&input);
    assert_vec_close(&standard, &online, EPS);
}

// ══════════════════════════════════════════════════════════════
// 3. Causal softmax tests
// ══════════════════════════════════════════════════════════════

#[test]
fn causal_softmax_first_position() {
    // At position 0, only index 0 is visible
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let masked = apply_causal_mask(&input, 4, 0);
    let result = softmax(&masked);
    assert_close(result[0], 1.0, EPS);
    assert_close(result[1], 0.0, EPS);
    assert_close(result[2], 0.0, EPS);
    assert_close(result[3], 0.0, EPS);
}

#[test]
fn causal_softmax_last_position() {
    // At last position, all tokens visible — same as standard softmax
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let masked = apply_causal_mask(&input, 4, 3);
    let standard = softmax(&input);
    let result = softmax(&masked);
    assert_vec_close(&standard, &result, EPS);
}

#[test]
fn causal_softmax_middle_position() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let masked = apply_causal_mask(&input, 6, 2);
    let result = softmax(&masked);
    assert_is_probability_distribution(&result, EPS);
    // Positions 3,4,5 should be zero
    assert_close(result[3], 0.0, EPS);
    assert_close(result[4], 0.0, EPS);
    assert_close(result[5], 0.0, EPS);
    // Positions 0,1,2 should sum to 1
    let visible_sum: f32 = result[0] + result[1] + result[2];
    assert_close(visible_sum, 1.0, EPS);
}

#[test]
fn causal_softmax_preserves_visible_ordering() {
    let input = vec![5.0, 3.0, 7.0, 1.0, 9.0];
    let masked = apply_causal_mask(&input, 5, 2);
    let result = softmax(&masked);
    // Among visible (0,1,2), index 2 (val=7) should be largest
    assert!(result[2] > result[0]);
    assert!(result[0] > result[1]);
}

#[test]
fn causal_softmax_single_token_sequence() {
    let input = vec![42.0];
    let masked = apply_causal_mask(&input, 1, 0);
    let result = softmax(&masked);
    assert_close(result[0], 1.0, EPS);
}

#[test]
fn causal_softmax_uniform_visible_tokens() {
    let input = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let masked = apply_causal_mask(&input, 8, 3);
    let result = softmax(&masked);
    // 4 visible tokens, each should be 0.25
    for &v in &result[..4] {
        assert_close(v, 0.25, EPS);
    }
    for &v in &result[4..] {
        assert_close(v, 0.0, EPS);
    }
}

#[test]
fn causal_mask_progressive_positions() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let seq_len = 4;
    // As query position advances, more context is visible
    for pos in 0..seq_len {
        let masked = apply_causal_mask(&input, seq_len, pos);
        let result = softmax(&masked);
        assert_is_probability_distribution(&result, EPS);
        let nonzero_count = result.iter().filter(|&&v| v > EPS).count();
        assert_eq!(nonzero_count, pos + 1);
    }
}

// ══════════════════════════════════════════════════════════════
// 4. Multi-head softmax tests
// ══════════════════════════════════════════════════════════════

#[test]
fn multi_head_softmax_independent_heads() {
    let head_0 = vec![1.0, 2.0, 3.0];
    let head_1 = vec![3.0, 2.0, 1.0];
    let result_0 = softmax(&head_0);
    let result_1 = softmax(&head_1);
    // Each head normalizes independently
    assert_is_probability_distribution(&result_0, EPS);
    assert_is_probability_distribution(&result_1, EPS);
    // Head 0 ascending, head 1 descending
    assert!(result_0[2] > result_0[0]);
    assert!(result_1[0] > result_1[2]);
}

#[test]
fn multi_head_softmax_batch_processing() {
    let num_heads = 4;
    let seq_len = 8;
    // Simulate batch of heads as a flat buffer
    let mut flat_input = vec![0.0f32; num_heads * seq_len];
    for h in 0..num_heads {
        for s in 0..seq_len {
            flat_input[h * seq_len + s] = (h as f32) * 0.5 + (s as f32) * 0.1;
        }
    }
    // Process each head independently
    let mut flat_output = vec![0.0f32; num_heads * seq_len];
    for h in 0..num_heads {
        let start = h * seq_len;
        let end = start + seq_len;
        let head_result = softmax(&flat_input[start..end]);
        flat_output[start..end].copy_from_slice(&head_result);
    }
    // Verify each head is a valid distribution
    for h in 0..num_heads {
        let start = h * seq_len;
        let end = start + seq_len;
        assert_is_probability_distribution(&flat_output[start..end], EPS);
    }
}

#[test]
fn multi_head_softmax_identical_heads_produce_same_output() {
    let seq_len = 6;
    let input = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
    let expected = softmax(&input);
    for _ in 0..8 {
        let result = softmax(&input);
        assert_vec_close(&expected, &result, EPS);
    }
}

#[test]
fn multi_head_softmax_varying_head_dimensions() {
    // Different head sizes (e.g., 64, 128)
    for &head_dim in &[4, 8, 16, 32, 64] {
        let input: Vec<f32> = (0..head_dim).map(|i| i as f32 * 0.1).collect();
        let result = softmax(&input);
        assert_eq!(result.len(), head_dim);
        assert_is_probability_distribution(&result, EPS);
    }
}

#[test]
fn multi_head_causal_softmax() {
    let num_heads = 2;
    let seq_len = 4;
    let query_pos = 2;
    let heads: Vec<Vec<f32>> = vec![vec![1.0, 2.0, 3.0, 4.0], vec![4.0, 3.0, 2.0, 1.0]];
    for h in 0..num_heads {
        let masked = apply_causal_mask(&heads[h], seq_len, query_pos);
        let result = softmax(&masked);
        assert_is_probability_distribution(&result, EPS);
        assert_close(result[3], 0.0, EPS); // masked out
    }
}

// ══════════════════════════════════════════════════════════════
// 5. Temperature scaling tests
// ══════════════════════════════════════════════════════════════

#[test]
fn temperature_scaling_high_temperature_uniform() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let result = softmax_with_temperature(&input, 100.0);
    assert_is_probability_distribution(&result, EPS);
    // High temperature → near-uniform
    for &v in &result {
        assert_close(v, 0.25, 0.01);
    }
}

#[test]
fn temperature_scaling_low_temperature_peaky() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let result = softmax_with_temperature(&input, 0.01);
    assert_is_probability_distribution(&result, EPS);
    // Low temperature → argmax dominates
    assert!(result[3] > 0.99);
}

#[test]
fn temperature_one_matches_standard() {
    let input = vec![1.0, 2.0, 3.0];
    let standard = softmax(&input);
    let with_temp = softmax_with_temperature(&input, 1.0);
    assert_vec_close(&standard, &with_temp, EPS);
}

#[test]
fn temperature_scaling_preserves_ordering() {
    let input = vec![3.0, 1.0, 4.0, 1.0, 5.0];
    for &temp in &[0.1, 0.5, 1.0, 2.0, 10.0] {
        let result = softmax_with_temperature(&input, temp);
        assert_is_probability_distribution(&result, EPS);
        // Index 4 (val=5) should always be largest
        let max_idx =
            result.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        assert_eq!(max_idx, 4, "ordering broken at temp={temp}");
    }
}

#[test]
fn temperature_scaling_entropy_increases_with_temperature() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mut prev_entropy = f32::NEG_INFINITY;
    for &temp in &[0.1, 0.5, 1.0, 2.0, 5.0, 10.0] {
        let result = softmax_with_temperature(&input, temp);
        let entropy: f32 = result.iter().filter(|&&p| p > 0.0).map(|&p| -p * p.ln()).sum();
        assert!(
            entropy >= prev_entropy - EPS,
            "entropy should increase with temperature: {entropy} < {prev_entropy} at T={temp}"
        );
        prev_entropy = entropy;
    }
}

#[test]
fn temperature_scaling_symmetric_input() {
    let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let result = softmax_with_temperature(&input, 0.5);
    assert_is_probability_distribution(&result, EPS);
    // Symmetric around center: result[0] ≈ result[4], result[1] ≈ result[3]
    // Not exactly due to nonlinearity, but check ordering
    assert!(result[4] > result[3]);
    assert!(result[3] > result[2]);
}

// ══════════════════════════════════════════════════════════════
// 6. Top-k masking tests
// ══════════════════════════════════════════════════════════════

#[test]
fn top_k_mask_basic() {
    let input = vec![1.0, 5.0, 3.0, 2.0, 4.0];
    let masked = top_k_mask(&input, 2);
    let result = softmax(&masked);
    assert_is_probability_distribution(&result, EPS);
    // Only indices 1 (val=5) and 4 (val=4) should be non-zero
    assert!(result[1] > 0.0);
    assert!(result[4] > 0.0);
    assert_close(result[0], 0.0, EPS);
    assert_close(result[2], 0.0, EPS);
    assert_close(result[3], 0.0, EPS);
}

#[test]
fn top_k_mask_k_equals_length() {
    let input = vec![1.0, 2.0, 3.0];
    let masked = top_k_mask(&input, 3);
    let standard = softmax(&input);
    let result = softmax(&masked);
    assert_vec_close(&standard, &result, EPS);
}

#[test]
fn top_k_mask_k_greater_than_length() {
    let input = vec![1.0, 2.0];
    let masked = top_k_mask(&input, 10);
    let standard = softmax(&input);
    let result = softmax(&masked);
    assert_vec_close(&standard, &result, EPS);
}

#[test]
fn top_k_mask_k_equals_one() {
    let input = vec![1.0, 5.0, 3.0, 2.0];
    let masked = top_k_mask(&input, 1);
    let result = softmax(&masked);
    assert_close(result[1], 1.0, EPS);
}

#[test]
fn top_k_then_temperature() {
    let input = vec![1.0, 5.0, 3.0, 2.0, 4.0];
    let masked = top_k_mask(&input, 3);
    let result = softmax_with_temperature(&masked, 0.5);
    assert_is_probability_distribution(&result, EPS);
    // Top 3: indices 1(5), 4(4), 2(3)
    let nonzero_count = result.iter().filter(|&&v| v > EPS).count();
    assert_eq!(nonzero_count, 3);
}

#[test]
fn top_k_mask_with_ties() {
    let input = vec![1.0, 3.0, 3.0, 2.0];
    let masked = top_k_mask(&input, 2);
    let result = softmax(&masked);
    assert_is_probability_distribution(&result, EPS);
    // Both 3.0 values should be kept
    assert!(result[1] > 0.0);
    assert!(result[2] > 0.0);
}

// ══════════════════════════════════════════════════════════════
// 7. Numerical stability tests
// ══════════════════════════════════════════════════════════════

#[test]
fn numerical_stability_very_large_inputs() {
    let input = vec![1000.0, 1001.0, 1002.0];
    let result = softmax(&input);
    assert_is_probability_distribution(&result, EPS);
    // Should not overflow — max subtraction prevents it
    assert!(result[2] > result[1]);
    assert!(result[1] > result[0]);
}

#[test]
fn numerical_stability_very_negative_inputs() {
    let input = vec![-1000.0, -999.0, -998.0];
    let result = softmax(&input);
    assert_is_probability_distribution(&result, EPS);
    assert!(result[2] > result[1]);
}

#[test]
fn numerical_stability_mixed_extreme_values() {
    let input = vec![-500.0, 0.0, 500.0];
    let result = softmax(&input);
    assert_is_probability_distribution(&result, EPS);
    // The largest value should dominate
    assert!(result[2] > 0.999);
}

#[test]
fn numerical_stability_all_negative_infinity() {
    // Edge case: all NEG_INF should produce NaN/0 — test that at least
    // one masked scenario handles this gracefully
    let input = vec![NEG_INF, NEG_INF, NEG_INF];
    let result = softmax(&input);
    // Result will be NaN due to 0/0 — this validates that the shader
    // must handle this case with a guard
    for &v in &result {
        assert!(v.is_nan(), "all-inf input should produce NaN without guard");
    }
}

#[test]
fn numerical_stability_single_non_inf() {
    let input = vec![NEG_INF, 5.0, NEG_INF];
    let result = softmax(&input);
    assert_close(result[1], 1.0, EPS);
    assert_close(result[0], 0.0, EPS);
    assert_close(result[2], 0.0, EPS);
}

#[test]
fn numerical_stability_online_vs_standard_extreme() {
    let input = vec![88.0, 89.0, 90.0, -88.0, -89.0];
    let standard = softmax(&input);
    let online = online_softmax(&input);
    assert_vec_close(&standard, &online, EPS);
}

#[test]
fn numerical_stability_subnormal_values() {
    // Very small but nonzero values
    let input = vec![1e-38, 2e-38, 3e-38];
    let result = softmax(&input);
    assert_is_probability_distribution(&result, EPS);
}

#[test]
fn numerical_stability_identical_large_values() {
    let input = vec![500.0; 128];
    let result = softmax(&input);
    assert_is_probability_distribution(&result, EPS);
    for &v in &result {
        assert_close(v, 1.0 / 128.0, EPS);
    }
}

#[test]
fn numerical_stability_alternating_extreme() {
    let input = vec![-1e6, 1e6, -1e6, 1e6];
    let result = softmax(&input);
    assert_is_probability_distribution(&result, EPS);
    assert_close(result[0], 0.0, EPS);
    assert_close(result[2], 0.0, EPS);
    assert_close(result[1], 0.5, EPS);
    assert_close(result[3], 0.5, EPS);
}

#[test]
fn numerical_stability_gradual_overflow_region() {
    // Values near f32 exp overflow boundary (~88.7)
    let input = vec![87.0, 88.0, 88.5, 88.7];
    let result = softmax(&input);
    assert_is_probability_distribution(&result, EPS);
    assert!(result[3] > result[2]);
}

// ══════════════════════════════════════════════════════════════
// 8. Flash attention tile-based softmax tests
// ══════════════════════════════════════════════════════════════

#[test]
fn flash_attention_single_tile_matches_standard() {
    let tile = vec![1.0, 2.0, 3.0, 4.0];
    let full_softmax = softmax(&tile);

    let max_val = tile.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let denom: f32 = tile.iter().map(|&x| (x - max_val).exp()).sum();
    let tile_result: Vec<f32> = tile.iter().map(|&x| (x - max_val).exp() / denom).collect();

    assert_vec_close(&full_softmax, &tile_result, EPS);
}

#[test]
fn flash_attention_two_tile_rescaling() {
    let full_input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let full_result = softmax(&full_input);

    // Process in two tiles of 4
    let tile_0 = &full_input[0..4];
    let tile_1 = &full_input[4..8];

    // Tile 0: initial softmax
    let max_0 = tile_0.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let denom_0: f32 = tile_0.iter().map(|&x| (x - max_0).exp()).sum();
    let out_0: Vec<f32> = tile_0.iter().map(|&x| (x - max_0).exp() / denom_0).collect();

    // Tile 1: rescale with new tile
    let (rescaled_0, global_max, global_denom) =
        flash_attention_softmax_rescale(&out_0, max_0, denom_0, tile_1);

    // Compute tile 1 output with global stats
    let out_1: Vec<f32> = tile_1.iter().map(|&x| (x - global_max).exp() / global_denom).collect();

    let mut combined = rescaled_0;
    combined.extend_from_slice(&out_1);
    assert_vec_close(&full_result, &combined, EPS_ACCUMULATED);
}

#[test]
fn flash_attention_four_tiles() {
    let full_input: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
    let full_result = softmax(&full_input);
    let tile_size = 4;

    let mut running_max = f32::NEG_INFINITY;
    let mut running_denom = 0.0f32;
    let mut all_outputs: Vec<f32> = Vec::new();

    for chunk_idx in 0..(full_input.len() / tile_size) {
        let start = chunk_idx * tile_size;
        let tile = &full_input[start..start + tile_size];

        if chunk_idx == 0 {
            running_max = tile.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            running_denom = tile.iter().map(|&x| (x - running_max).exp()).sum();
            let out: Vec<f32> =
                tile.iter().map(|&x| (x - running_max).exp() / running_denom).collect();
            all_outputs.extend_from_slice(&out);
        } else {
            let (rescaled, new_max, new_denom) =
                flash_attention_softmax_rescale(&all_outputs, running_max, running_denom, tile);
            let tile_out: Vec<f32> =
                tile.iter().map(|&x| (x - new_max).exp() / new_denom).collect();
            all_outputs = rescaled;
            all_outputs.extend_from_slice(&tile_out);
            running_max = new_max;
            running_denom = new_denom;
        }
    }

    assert_vec_close(&full_result, &all_outputs, EPS_ACCUMULATED);
}

#[test]
fn flash_attention_tile_boundary_values() {
    // All values in tile 0 are small, tile 1 has the dominant value
    let tile_0 = vec![0.1, 0.2, 0.3, 0.4];
    let tile_1 = vec![0.1, 0.2, 0.3, 100.0];
    let mut full = tile_0.clone();
    full.extend_from_slice(&tile_1);
    let full_result = softmax(&full);

    let max_0 = tile_0.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let denom_0: f32 = tile_0.iter().map(|&x| (x - max_0).exp()).sum();
    let out_0: Vec<f32> = tile_0.iter().map(|&x| (x - max_0).exp() / denom_0).collect();

    let (rescaled_0, gmax, gdenom) =
        flash_attention_softmax_rescale(&out_0, max_0, denom_0, &tile_1);
    let out_1: Vec<f32> = tile_1.iter().map(|&x| (x - gmax).exp() / gdenom).collect();

    let mut combined = rescaled_0;
    combined.extend_from_slice(&out_1);
    assert_vec_close(&full_result, &combined, EPS_ACCUMULATED);
}

#[test]
fn flash_attention_uniform_tiles() {
    let tile_0 = vec![1.0; 4];
    let tile_1 = vec![1.0; 4];
    let mut full = tile_0.clone();
    full.extend_from_slice(&tile_1);
    let full_result = softmax(&full);

    let max_0 = 1.0f32;
    let denom_0: f32 = 4.0 * (0.0f32).exp();
    let out_0: Vec<f32> = vec![0.25; 4];

    let (rescaled_0, _gmax, gdenom) =
        flash_attention_softmax_rescale(&out_0, max_0, denom_0, &tile_1);
    let out_1: Vec<f32> = tile_1.iter().map(|&x| (x - 1.0).exp() / gdenom).collect();

    let mut combined = rescaled_0;
    combined.extend_from_slice(&out_1);
    assert_vec_close(&full_result, &combined, EPS);
}

// ══════════════════════════════════════════════════════════════
// 9. Grouped query attention (GQA) softmax tests
// ══════════════════════════════════════════════════════════════

/// Simulate GQA where multiple query heads share a single KV head.
fn gqa_softmax(
    query_heads: &[Vec<f32>],
    kv_heads: &[Vec<f32>],
    num_query_heads: usize,
    num_kv_heads: usize,
) -> Vec<Vec<f32>> {
    assert_eq!(query_heads.len(), num_query_heads);
    assert_eq!(kv_heads.len(), num_kv_heads);
    assert!(num_query_heads % num_kv_heads == 0, "query heads must be divisible by kv heads");
    let group_size = num_query_heads / num_kv_heads;

    let mut results = Vec::with_capacity(num_query_heads);
    for q in 0..num_query_heads {
        let kv_idx = q / group_size;
        // Simulate attention score = dot(query, key) → softmax
        let scores: Vec<f32> = query_heads[q]
            .iter()
            .zip(kv_heads[kv_idx].iter())
            .map(|(&q_val, &k_val)| q_val * k_val)
            .collect();
        results.push(softmax(&scores));
    }
    results
}

#[test]
fn gqa_softmax_basic_4q_1kv() {
    let num_q = 4;
    let num_kv = 1;
    let seq_len = 4;
    let query_heads: Vec<Vec<f32>> = (0..num_q)
        .map(|h| (0..seq_len).map(|s| (h as f32) * 0.3 + (s as f32) * 0.1).collect())
        .collect();
    let kv_heads: Vec<Vec<f32>> = vec![vec![1.0, 0.5, 0.3, 0.1]];

    let results = gqa_softmax(&query_heads, &kv_heads, num_q, num_kv);
    assert_eq!(results.len(), num_q);
    for result in &results {
        assert_is_probability_distribution(result, EPS);
    }
}

#[test]
fn gqa_softmax_8q_2kv_groups() {
    let num_q = 8;
    let num_kv = 2;
    let seq_len = 4;
    let query_heads: Vec<Vec<f32>> = (0..num_q)
        .map(|h| (0..seq_len).map(|s| (h as f32) * 0.2 + (s as f32) * 0.1).collect())
        .collect();
    let kv_heads: Vec<Vec<f32>> = vec![vec![1.0, 0.8, 0.6, 0.4], vec![0.4, 0.6, 0.8, 1.0]];

    let results = gqa_softmax(&query_heads, &kv_heads, num_q, num_kv);
    assert_eq!(results.len(), num_q);
    // Heads 0-3 share kv_head 0, heads 4-7 share kv_head 1
    for result in &results {
        assert_is_probability_distribution(result, EPS);
    }
}

#[test]
fn gqa_softmax_same_group_shares_kv() {
    let num_q = 4;
    let num_kv = 1;
    let seq_len = 4;
    // Identical queries → identical outputs (since they share same KV)
    let query_heads: Vec<Vec<f32>> = vec![vec![1.0, 2.0, 3.0, 4.0]; num_q];
    let kv_heads: Vec<Vec<f32>> = vec![vec![0.5, 0.5, 0.5, 0.5]];

    let results = gqa_softmax(&query_heads, &kv_heads, num_q, num_kv);
    for i in 1..num_q {
        assert_vec_close(&results[0], &results[i], EPS);
    }
}

#[test]
fn gqa_softmax_mha_degenerate_case() {
    // When num_q == num_kv, GQA degenerates to standard MHA
    let num_heads = 4;
    let seq_len = 4;
    let query_heads: Vec<Vec<f32>> =
        (0..num_heads).map(|h| (0..seq_len).map(|s| (h + s) as f32).collect()).collect();
    let kv_heads = query_heads.clone();

    let results = gqa_softmax(&query_heads, &kv_heads, num_heads, num_heads);
    for h in 0..num_heads {
        let scores: Vec<f32> =
            query_heads[h].iter().zip(kv_heads[h].iter()).map(|(&q, &k)| q * k).collect();
        let expected = softmax(&scores);
        assert_vec_close(&expected, &results[h], EPS);
    }
}

#[test]
fn gqa_softmax_with_causal_mask() {
    let num_q = 2;
    let num_kv = 1;
    let seq_len = 4;
    let query_pos = 2;
    let query_heads: Vec<Vec<f32>> = vec![vec![1.0, 2.0, 3.0, 4.0], vec![4.0, 3.0, 2.0, 1.0]];
    let kv_heads: Vec<Vec<f32>> = vec![vec![1.0, 1.0, 1.0, 1.0]];

    for q in 0..num_q {
        let scores: Vec<f32> =
            query_heads[q].iter().zip(kv_heads[0].iter()).map(|(&qv, &kv)| qv * kv).collect();
        let masked = apply_causal_mask(&scores, seq_len, query_pos);
        let result = softmax(&masked);
        assert_is_probability_distribution(&result, EPS);
        assert_close(result[3], 0.0, EPS);
    }
}

// ══════════════════════════════════════════════════════════════
// 10. Additional edge case and integration tests
// ══════════════════════════════════════════════════════════════

#[test]
fn softmax_power_of_two_lengths() {
    for &len in &[1, 2, 4, 8, 16, 32, 64, 128, 256] {
        let input: Vec<f32> = (0..len).map(|i| (i as f32) * 0.01).collect();
        let result = softmax(&input);
        assert_eq!(result.len(), len);
        assert_is_probability_distribution(&result, EPS);
    }
}

#[test]
fn softmax_non_power_of_two_lengths() {
    for &len in &[3, 5, 7, 13, 17, 31, 65, 127, 255] {
        let input: Vec<f32> = (0..len).map(|i| (i as f32) * 0.01).collect();
        let result = softmax(&input);
        assert_eq!(result.len(), len);
        assert_is_probability_distribution(&result, EPS);
    }
}

#[test]
fn causal_softmax_combined_with_temperature() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let query_pos = 3;
    let temperature = 0.5;
    let masked = apply_causal_mask(&input, 6, query_pos);
    let result = softmax_with_temperature(&masked, temperature);
    assert_is_probability_distribution(&result, EPS);
    // Positions 4,5 should be zero
    assert_close(result[4], 0.0, EPS);
    assert_close(result[5], 0.0, EPS);
}

#[test]
fn top_k_combined_with_causal_mask() {
    let input = vec![1.0, 5.0, 3.0, 7.0, 2.0, 6.0];
    let query_pos = 3;
    let masked = apply_causal_mask(&input, 6, query_pos);
    let top_k_masked = top_k_mask(&masked, 2);
    let result = softmax(&top_k_masked);
    assert_is_probability_distribution(&result, EPS);
    // Only top-2 of visible tokens (0..=3) should be nonzero
    let nonzero_count = result.iter().filter(|&&v| v > EPS).count();
    assert_eq!(nonzero_count, 2);
}

#[test]
fn softmax_idempotent_double_application() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let once = softmax(&input);
    let twice = softmax(&once);
    // After double-softmax, distribution should be more uniform
    assert_is_probability_distribution(&twice, EPS);
    let max_diff: f32 = twice.iter().map(|&v| (v - 0.25).abs()).sum();
    let once_max_diff: f32 = once.iter().map(|&v| (v - 0.25).abs()).sum();
    assert!(max_diff < once_max_diff, "double softmax should be more uniform");
}

#[test]
fn softmax_translation_invariance() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let shifted: Vec<f32> = input.iter().map(|&x| x + 1000.0).collect();
    let result_orig = softmax(&input);
    let result_shifted = softmax(&shifted);
    assert_vec_close(&result_orig, &result_shifted, EPS);
}

#[test]
fn softmax_scale_sensitivity() {
    let input = vec![1.0, 2.0, 3.0];
    let scaled: Vec<f32> = input.iter().map(|&x| x * 2.0).collect();
    let result_orig = softmax(&input);
    let result_scaled = softmax(&scaled);
    // Scaling changes the distribution (more peaky)
    assert!(result_scaled[2] > result_orig[2]);
}

#[test]
fn online_softmax_large_sequence() {
    let input: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.01).collect();
    let standard = softmax(&input);
    let online = online_softmax(&input);
    assert_vec_close(&standard, &online, EPS_ACCUMULATED);
}

#[test]
fn softmax_argmax_stability() {
    // Verify argmax is preserved across different implementations
    let input = vec![0.1, 0.9, 0.5, 0.3, 0.7];
    let standard = softmax(&input);
    let online = online_softmax(&input);
    let with_temp = softmax_with_temperature(&input, 0.8);

    let argmax = |v: &[f32]| -> usize {
        v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
    };

    assert_eq!(argmax(&standard), 1);
    assert_eq!(argmax(&online), 1);
    assert_eq!(argmax(&with_temp), 1);
}

// ══════════════════════════════════════════════════════════════
// GPU runtime tests (ignored — require Metal hardware)
// ══════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires Metal GPU runtime"]
fn metal_softmax_dispatch_basic() {
    // Would dispatch softmax kernel on Metal GPU
    unimplemented!("Metal GPU dispatch not available in test environment");
}

#[test]
#[ignore = "requires Metal GPU runtime"]
fn metal_softmax_threadgroup_reduction() {
    // Would test threadgroup-level parallel reduction
    unimplemented!("Metal GPU dispatch not available in test environment");
}

#[test]
#[ignore = "requires Metal GPU runtime"]
fn metal_softmax_simdgroup_operations() {
    // Would test SIMD group shuffle for warp-level reduction
    unimplemented!("Metal GPU dispatch not available in test environment");
}

#[test]
#[ignore = "requires Metal GPU runtime"]
fn metal_flash_attention_softmax_kernel() {
    // Would test flash attention tiled softmax on GPU
    unimplemented!("Metal GPU dispatch not available in test environment");
}

#[test]
#[ignore = "requires Metal GPU runtime"]
fn metal_softmax_half_precision() {
    // Would test half-precision (float16) softmax on GPU
    unimplemented!("Metal GPU dispatch not available in test environment");
}

#[test]
#[ignore = "requires Metal GPU runtime"]
fn metal_gqa_softmax_kernel() {
    // Would test grouped query attention softmax dispatch
    unimplemented!("Metal GPU dispatch not available in test environment");
}

#[test]
#[ignore = "requires Metal GPU runtime"]
fn metal_causal_mask_softmax_kernel() {
    // Would test fused causal mask + softmax kernel
    unimplemented!("Metal GPU dispatch not available in test environment");
}
