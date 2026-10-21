//! Attention mask construction and manipulation utilities.
//!
//! Provides helpers to build causal, padding, and sliding-window masks
//! and to combine or apply them to pre-softmax attention scores.
//!
//! All masks use the **additive** convention: `0.0` means "attend" and
//! [`f32::NEG_INFINITY`] means "block".  Adding the mask to raw QK scores
//! before softmax drives blocked positions to zero probability.

/// Create a causal (lower-triangular) additive mask of shape
/// `[seq_len, seq_len]`.
///
/// `mask[i * seq_len + j]` is `0.0` when `j <= i` (attend) and
/// [`f32::NEG_INFINITY`] when `j > i` (block).
pub fn create_causal_mask(seq_len: usize) -> Vec<f32> {
    let mut mask = vec![0.0_f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask[i * seq_len + j] = f32::NEG_INFINITY;
        }
    }
    mask
}

/// Create a padding mask for a batch of sequences.
///
/// Returns a flat `[batch_size, max_len]` additive mask.  For each
/// sequence `b`, positions `0..lengths[b]` are `0.0` and the remaining
/// positions up to `max_len` are [`f32::NEG_INFINITY`].
///
/// If a length exceeds `max_len` it is clamped to `max_len`.
pub fn create_padding_mask(lengths: &[usize], max_len: usize) -> Vec<f32> {
    let batch = lengths.len();
    let mut mask = vec![0.0_f32; batch * max_len];
    for (b, &len) in lengths.iter().enumerate() {
        let valid = len.min(max_len);
        for j in valid..max_len {
            mask[b * max_len + j] = f32::NEG_INFINITY;
        }
    }
    mask
}

/// Apply an additive mask to pre-softmax attention scores (in-place).
///
/// Both `scores` and `mask` must contain at least `seq_len * seq_len`
/// elements.  Each element of `mask` is added to the corresponding
/// element of `scores`.
pub fn apply_mask(scores: &mut [f32], mask: &[f32], seq_len: usize) {
    let n = seq_len * seq_len;
    assert!(scores.len() >= n, "scores length {} too short for seq_len={seq_len}", scores.len());
    assert!(mask.len() >= n, "mask length {} too short for seq_len={seq_len}", mask.len());
    for i in 0..n {
        scores[i] += mask[i];
    }
}

/// Create a sliding-window causal mask of shape `[seq_len, seq_len]`.
///
/// Position `(i, j)` is `0.0` when
/// `i.saturating_sub(window - 1) <= j <= i`, i.e. the token at position
/// `i` can attend to at most `window` preceding tokens (including
/// itself).  All other positions are [`f32::NEG_INFINITY`].
///
/// When `window >= seq_len` this is equivalent to a standard causal mask.
/// A `window` of `0` blocks every position.
pub fn create_sliding_window_mask(seq_len: usize, window: usize) -> Vec<f32> {
    let mut mask = vec![f32::NEG_INFINITY; seq_len * seq_len];
    if window == 0 {
        return mask;
    }
    for i in 0..seq_len {
        let start = i.saturating_sub(window - 1);
        for j in start..=i {
            mask[i * seq_len + j] = 0.0;
        }
    }
    mask
}

/// Combine two additive masks element-wise.
///
/// Returns a new `[seq_len, seq_len]` mask whose element at index `k`
/// is `a[k] + b[k]`.  Because both inputs use `0.0` / `NEG_INFINITY`,
/// the result blocks a position whenever *either* input blocks it.
pub fn combine_masks(a: &[f32], b: &[f32], seq_len: usize) -> Vec<f32> {
    let n = seq_len * seq_len;
    assert!(a.len() >= n, "mask a length {} too short for seq_len={seq_len}", a.len());
    assert!(b.len() >= n, "mask b length {} too short for seq_len={seq_len}", b.len());
    (0..n).map(|i| a[i] + b[i]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const NEG_INF: f32 = f32::NEG_INFINITY;

    // ── helpers ────────────────────────────────────────────────────

    /// Row-wise softmax (numerically stable).
    fn softmax(row: &[f32]) -> Vec<f32> {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|&v| (v - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        if sum == 0.0 { vec![0.0; row.len()] } else { exps.iter().map(|&e| e / sum).collect() }
    }

    fn is_neg_inf(v: f32) -> bool {
        v == NEG_INF
    }

    // ── create_causal_mask ─────────────────────────────────────────

    #[test]
    fn causal_mask_3x3_known_values() {
        let m = create_causal_mask(3);
        #[rustfmt::skip]
        let expected = [
            0.0,     NEG_INF, NEG_INF,
            0.0,     0.0,     NEG_INF,
            0.0,     0.0,     0.0,
        ];
        assert_eq!(m.len(), 9);
        for (i, (&got, &want)) in m.iter().zip(expected.iter()).enumerate() {
            assert!(
                got == want || (got.is_nan() && want.is_nan()),
                "mismatch at index {i}: got {got}, want {want}",
            );
        }
    }

    #[test]
    fn causal_mask_seq_len_1() {
        let m = create_causal_mask(1);
        assert_eq!(m, vec![0.0]);
    }

    #[test]
    fn causal_mask_seq_len_0() {
        let m = create_causal_mask(0);
        assert!(m.is_empty());
    }

    #[test]
    fn causal_mask_diagonal_is_zero() {
        for n in 1..=8 {
            let m = create_causal_mask(n);
            for i in 0..n {
                assert_eq!(m[i * n + i], 0.0, "diagonal at ({i},{i}) should be 0.0");
            }
        }
    }

    #[test]
    fn causal_mask_lower_triangle_is_zero() {
        let n = 5;
        let m = create_causal_mask(n);
        for i in 0..n {
            for j in 0..=i {
                assert_eq!(m[i * n + j], 0.0, "({i},{j}) should be 0.0");
            }
        }
    }

    #[test]
    fn causal_mask_upper_triangle_is_neg_inf() {
        let n = 5;
        let m = create_causal_mask(n);
        for i in 0..n {
            for j in (i + 1)..n {
                assert!(is_neg_inf(m[i * n + j]), "({i},{j}) should be -inf");
            }
        }
    }

    // ── create_padding_mask ────────────────────────────────────────

    #[test]
    fn padding_mask_known_values() {
        let m = create_padding_mask(&[2, 3], 3);
        #[rustfmt::skip]
        let expected = [
            0.0, 0.0, NEG_INF,  // length 2 → positions 0,1 valid
            0.0, 0.0, 0.0,      // length 3 → all valid
        ];
        assert_eq!(m.len(), 6);
        for (i, (&got, &want)) in m.iter().zip(expected.iter()).enumerate() {
            assert!(got == want, "mismatch at index {i}: got {got}, want {want}");
        }
    }

    #[test]
    fn padding_mask_all_valid() {
        let m = create_padding_mask(&[5, 5], 3);
        assert!(m.iter().all(|&v| v == 0.0), "all positions should be valid");
    }

    #[test]
    fn padding_mask_all_masked() {
        let m = create_padding_mask(&[0, 0], 4);
        assert!(m.iter().all(|&v| is_neg_inf(v)), "all positions should be masked");
    }

    #[test]
    fn padding_mask_valid_position_count() {
        let lengths = [1, 3, 0, 5];
        let max_len = 4;
        let m = create_padding_mask(&lengths, max_len);
        let valid: usize = m.iter().filter(|&&v| v == 0.0).count();
        let expected: usize = lengths.iter().map(|&l| l.min(max_len)).sum();
        assert_eq!(valid, expected);
    }

    #[test]
    fn padding_mask_empty_batch() {
        let m = create_padding_mask(&[], 4);
        assert!(m.is_empty());
    }

    // ── apply_mask ─────────────────────────────────────────────────

    #[test]
    fn apply_mask_basic() {
        let mut scores = vec![1.0, 2.0, 3.0, 4.0];
        let mask = vec![0.0, NEG_INF, 0.0, NEG_INF];
        apply_mask(&mut scores, &mask, 2);
        assert_eq!(scores[0], 1.0);
        assert!(is_neg_inf(scores[1]));
        assert_eq!(scores[2], 3.0);
        assert!(is_neg_inf(scores[3]));
    }

    #[test]
    fn apply_mask_seq_len_1() {
        let mut scores = vec![5.0];
        let mask = vec![0.0];
        apply_mask(&mut scores, &mask, 1);
        assert_eq!(scores[0], 5.0);
    }

    #[test]
    #[should_panic(expected = "too short")]
    fn apply_mask_scores_too_short() {
        let mut scores = vec![1.0];
        let mask = vec![0.0; 4];
        apply_mask(&mut scores, &mask, 2);
    }

    #[test]
    fn apply_causal_then_softmax_masks_future() {
        let seq = 4;
        let mask = create_causal_mask(seq);
        // Uniform scores before masking.
        let mut scores = vec![1.0; seq * seq];
        apply_mask(&mut scores, &mask, seq);

        // After softmax, masked positions should be ~0.
        for i in 0..seq {
            let row = &scores[i * seq..(i + 1) * seq];
            let probs = softmax(row);
            for j in (i + 1)..seq {
                assert!(probs[j] < 1e-6, "row {i} col {j}: prob {:.8} should be ~0", probs[j]);
            }
        }
    }

    // ── create_sliding_window_mask ─────────────────────────────────

    #[test]
    fn sliding_window_3x3_window2() {
        let m = create_sliding_window_mask(3, 2);
        #[rustfmt::skip]
        let expected = [
            0.0,     NEG_INF, NEG_INF, // pos 0 attends to [0]
            0.0,     0.0,     NEG_INF, // pos 1 attends to [0,1]
            NEG_INF, 0.0,     0.0,     // pos 2 attends to [1,2]
        ];
        for (i, (&got, &want)) in m.iter().zip(expected.iter()).enumerate() {
            assert!(got == want, "index {i}: got {got}, want {want}");
        }
    }

    #[test]
    fn sliding_window_equals_causal_when_large() {
        let seq = 6;
        let causal = create_causal_mask(seq);
        let sliding = create_sliding_window_mask(seq, seq);
        assert_eq!(causal, sliding);
    }

    #[test]
    fn sliding_window_exceeds_seq_len() {
        let seq = 4;
        let causal = create_causal_mask(seq);
        let sliding = create_sliding_window_mask(seq, seq + 10);
        assert_eq!(causal, sliding);
    }

    #[test]
    fn sliding_window_size_1() {
        let seq = 4;
        let m = create_sliding_window_mask(seq, 1);
        // Only the diagonal should be 0.0.
        for i in 0..seq {
            for j in 0..seq {
                let val = m[i * seq + j];
                if i == j {
                    assert_eq!(val, 0.0, "diagonal ({i},{j}) should be 0.0");
                } else {
                    assert!(is_neg_inf(val), "off-diagonal ({i},{j}) should be -inf");
                }
            }
        }
    }

    #[test]
    fn sliding_window_size_0_all_masked() {
        let m = create_sliding_window_mask(3, 0);
        assert!(m.iter().all(|&v| is_neg_inf(v)));
    }

    #[test]
    fn sliding_window_seq_len_1() {
        let m = create_sliding_window_mask(1, 5);
        assert_eq!(m, vec![0.0]);
    }

    // ── combine_masks ──────────────────────────────────────────────

    #[test]
    fn combine_both_open() {
        let a = vec![0.0; 4];
        let b = vec![0.0; 4];
        let c = combine_masks(&a, &b, 2);
        assert!(c.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn combine_one_blocks() {
        let a = vec![0.0, NEG_INF, 0.0, 0.0];
        let b = vec![0.0, 0.0, NEG_INF, 0.0];
        let c = combine_masks(&a, &b, 2);
        assert_eq!(c[0], 0.0);
        assert!(is_neg_inf(c[1]));
        assert!(is_neg_inf(c[2]));
        assert_eq!(c[3], 0.0);
    }

    #[test]
    fn combine_both_block_same_position() {
        let a = vec![NEG_INF; 4];
        let b = vec![NEG_INF; 4];
        let c = combine_masks(&a, &b, 2);
        assert!(c.iter().all(|&v| is_neg_inf(v)));
    }

    #[test]
    fn combine_causal_with_padding_row() {
        // Simulate: seq_len=3, second row of padding blocks position 2.
        let causal = create_causal_mask(3);
        // "Broadcast" a per-row padding mask into [seq, seq] shape:
        // row 0: all open, row 1: all open, row 2: col 2 blocked.
        let mut pad = vec![0.0_f32; 9];
        pad[2 * 3 + 2] = NEG_INF;

        let combined = combine_masks(&causal, &pad, 3);
        // Position (2,2) should be blocked even though causal allows it.
        assert!(is_neg_inf(combined[2 * 3 + 2]));
        // Position (2,1) should remain open (causal OK, pad OK).
        assert_eq!(combined[2 * 3 + 1], 0.0);
    }

    #[test]
    #[should_panic(expected = "too short")]
    fn combine_masks_too_short() {
        let a = vec![0.0; 2];
        let b = vec![0.0; 4];
        combine_masks(&a, &b, 2);
    }

    // ── property-style tests ───────────────────────────────────────

    #[test]
    fn causal_mask_count_open_positions() {
        // A causal mask has n*(n+1)/2 open positions.
        for n in 0..=10 {
            let m = create_causal_mask(n);
            let open = m.iter().filter(|&&v| v == 0.0).count();
            assert_eq!(open, n * (n + 1) / 2, "seq_len={n}");
        }
    }

    #[test]
    fn sliding_window_open_count() {
        // For window w and seq n, open positions = sum_{i=0}^{n-1} min(i+1, w).
        for n in 1..=8 {
            for w in 1..=n + 2 {
                let m = create_sliding_window_mask(n, w);
                let open = m.iter().filter(|&&v| v == 0.0).count();
                let expected: usize = (0..n).map(|i| (i + 1).min(w)).sum();
                assert_eq!(open, expected, "seq={n} window={w}");
            }
        }
    }
}
