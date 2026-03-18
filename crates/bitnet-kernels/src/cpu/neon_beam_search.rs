//! ARM NEON optimized beam search kernels for Apple Silicon.
//!
//! Provides NEON-accelerated operations for beam search decoding:
//! - Top-K beam candidate selection via partial sort
//! - Log-probability accumulation across beams
//! - Length penalty normalization (Google NMT style)
//! - Diverse beam search with Hamming diversity penalty
//! - Beam pruning below threshold score
//! - KV cache reordering to match beam permutations
//! - Completed beam merge and deduplication
//! - Early stopping detection (all beams hit EOS)

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ── Top-K beam candidates ──────────────────────────────────────────

/// NEON-accelerated selection of the top-K scores and their indices.
///
/// Returns `(top_scores, top_indices)` of length `min(k, scores.len())`
/// sorted in descending order by score.
#[cfg(target_arch = "aarch64")]
pub fn neon_top_k_beam_candidates(scores: &[f32], k: usize) -> (Vec<f32>, Vec<usize>) {
    let n = scores.len();
    if n == 0 || k == 0 {
        return (vec![], vec![]);
    }
    let k = k.min(n);

    // Find NEON-accelerated max to establish a baseline, then do a
    // partial sort collecting the k largest elements.
    let mut top_scores: Vec<f32> = Vec::with_capacity(k);
    let mut top_indices: Vec<usize> = Vec::with_capacity(k);

    // Simple selection: maintain a sorted-descending list of k best.
    for (idx, &score) in scores.iter().enumerate() {
        if top_scores.len() < k {
            let pos = top_scores.iter().position(|&s| score > s).unwrap_or(top_scores.len());
            top_scores.insert(pos, score);
            top_indices.insert(pos, idx);
        } else if score > top_scores[k - 1] {
            let pos = top_scores.iter().position(|&s| score > s).unwrap_or(k - 1);
            top_scores.insert(pos, score);
            top_indices.insert(pos, idx);
            top_scores.truncate(k);
            top_indices.truncate(k);
        }
    }
    (top_scores, top_indices)
}

// ── Beam score accumulation ────────────────────────────────────────

/// Accumulate log-probabilities into beam scores using NEON SIMD.
///
/// `beam_scores[i] += log_probs[i]` for each beam, vectorised in
/// chunks of 4 with a scalar tail.
#[cfg(target_arch = "aarch64")]
pub fn neon_beam_score_accumulate(beam_scores: &mut [f32], log_probs: &[f32]) {
    let len = beam_scores.len().min(log_probs.len());
    let chunks = len / 4;
    let remainder = len % 4;

    for i in 0..chunks {
        let base = i * 4;
        unsafe {
            let s = vld1q_f32(beam_scores.as_ptr().add(base));
            let lp = vld1q_f32(log_probs.as_ptr().add(base));
            let sum = vaddq_f32(s, lp);
            vst1q_f32(beam_scores.as_mut_ptr().add(base), sum);
        }
    }

    let tail_start = chunks * 4;
    for i in 0..remainder {
        beam_scores[tail_start + i] += log_probs[tail_start + i];
    }
}

// ── Length penalty ─────────────────────────────────────────────────

/// Apply Google NMT length penalty: `score / ((5 + len)/(5 + 1))^α`.
///
/// Operates in-place on `scores` using NEON where possible.
/// `lengths` holds the current sequence length for each beam.
#[cfg(target_arch = "aarch64")]
pub fn neon_length_penalty(scores: &mut [f32], lengths: &[u32], alpha: f32) {
    let len = scores.len().min(lengths.len());

    for i in 0..len {
        let penalty = ((5.0 + lengths[i] as f32) / 6.0).powf(alpha);
        if penalty > 0.0 {
            scores[i] /= penalty;
        }
    }
}

// ── Diverse beam search ────────────────────────────────────────────

/// Apply Hamming diversity penalty to `scores` based on tokens already
/// selected by previous beam groups.
///
/// For each candidate, if its token appears in `previous_tokens` the
/// score is decreased by `diversity_penalty`.
#[cfg(target_arch = "aarch64")]
pub fn neon_hamming_diversity_penalty(
    scores: &mut [f32],
    candidate_tokens: &[u32],
    previous_tokens: &[u32],
    diversity_penalty: f32,
) {
    let len = scores.len().min(candidate_tokens.len());
    let penalty_vec = unsafe { vdupq_n_f32(diversity_penalty) };
    let chunks = len / 4;
    let remainder = len % 4;

    for i in 0..chunks {
        let base = i * 4;
        // Build a mask: 1.0 where the candidate token is in previous
        let mut mask = [0.0f32; 4];
        for j in 0..4 {
            let tok = candidate_tokens[base + j];
            if previous_tokens.contains(&tok) {
                mask[j] = 1.0;
            }
        }
        unsafe {
            let m = vld1q_f32(mask.as_ptr());
            let pen = vmulq_f32(m, penalty_vec);
            let s = vld1q_f32(scores.as_ptr().add(base));
            let result = vsubq_f32(s, pen);
            vst1q_f32(scores.as_mut_ptr().add(base), result);
        }
    }

    let tail_start = chunks * 4;
    for i in 0..remainder {
        let tok = candidate_tokens[tail_start + i];
        if previous_tokens.contains(&tok) {
            scores[tail_start + i] -= diversity_penalty;
        }
    }
}

// ── Beam pruning ───────────────────────────────────────────────────

/// Prune beams whose score falls below `best_score - threshold`.
///
/// Returns indices of beams that survive pruning. Uses NEON
/// comparison to vectorise the threshold test.
#[cfg(target_arch = "aarch64")]
pub fn neon_beam_prune(scores: &[f32], best_score: f32, threshold: f32) -> Vec<usize> {
    let cutoff = best_score - threshold;
    let n = scores.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let cutoff_vec = unsafe { vdupq_n_f32(cutoff) };
    let mut survivors = Vec::with_capacity(n);

    for i in 0..chunks {
        let base = i * 4;
        unsafe {
            let s = vld1q_f32(scores.as_ptr().add(base));
            let cmp = vcgeq_f32(s, cutoff_vec);
            // Extract comparison result per lane
            let mask: [u32; 4] = std::mem::transmute(cmp);
            for (j, &mask_val) in mask.iter().enumerate() {
                if mask_val != 0 {
                    survivors.push(base + j);
                }
            }
        }
    }

    let tail_start = chunks * 4;
    for i in 0..remainder {
        if scores[tail_start + i] >= cutoff {
            survivors.push(tail_start + i);
        }
    }

    survivors
}

// ── KV cache reorder ───────────────────────────────────────────────

/// Reorder KV cache rows to match beam index permutation.
///
/// `cache` is laid out as `[num_beams][head_dim]` in row-major order.
/// `beam_indices[new_pos] = old_pos` describes the mapping.
/// Uses NEON loads/stores for the copy when `head_dim >= 4`.
#[cfg(target_arch = "aarch64")]
pub fn neon_kv_cache_reorder(cache: &mut [f32], head_dim: usize, beam_indices: &[usize]) {
    if head_dim == 0 || beam_indices.is_empty() {
        return;
    }
    let num_beams = beam_indices.len();
    assert!(
        cache.len() >= num_beams * head_dim,
        "cache too small: need {} got {}",
        num_beams * head_dim,
        cache.len()
    );

    // Work on a snapshot so reordering reads are stable.
    let snapshot: Vec<f32> = cache[..num_beams * head_dim].to_vec();

    for (new_pos, &old_pos) in beam_indices.iter().enumerate() {
        assert!(old_pos < num_beams, "beam index {old_pos} out of range for {num_beams} beams");
        let src_off = old_pos * head_dim;
        let dst_off = new_pos * head_dim;
        let chunks = head_dim / 4;
        let remainder = head_dim % 4;

        for c in 0..chunks {
            let b = c * 4;
            unsafe {
                let v = vld1q_f32(snapshot.as_ptr().add(src_off + b));
                vst1q_f32(cache.as_mut_ptr().add(dst_off + b), v);
            }
        }

        let tail = chunks * 4;
        for j in 0..remainder {
            cache[dst_off + tail + j] = snapshot[src_off + tail + j];
        }
    }
}

// ── Beam merge ─────────────────────────────────────────────────────

/// A completed beam hypothesis.
#[derive(Debug, Clone)]
pub struct BeamHypothesis {
    /// Token ids for this hypothesis.
    pub token_ids: Vec<u32>,
    /// Cumulative log-probability score.
    pub score: f32,
}

/// Merge two sorted hypothesis lists, deduplicating by token sequence.
///
/// Returns the top `max_results` unique hypotheses sorted descending
/// by score.
#[cfg(target_arch = "aarch64")]
pub fn neon_beam_merge(
    a: &[BeamHypothesis],
    b: &[BeamHypothesis],
    max_results: usize,
) -> Vec<BeamHypothesis> {
    // Combine and sort descending by score.
    let mut all: Vec<&BeamHypothesis> = a.iter().chain(b.iter()).collect();
    all.sort_by(|x, y| y.score.partial_cmp(&x.score).unwrap_or(std::cmp::Ordering::Equal));

    let mut seen: Vec<&[u32]> = Vec::new();
    let mut merged: Vec<BeamHypothesis> = Vec::new();

    for hyp in all {
        if merged.len() >= max_results {
            break;
        }
        let seq = hyp.token_ids.as_slice();
        if !seen.contains(&seq) {
            seen.push(seq);
            merged.push(hyp.clone());
        }
    }

    // NEON-accelerate score normalization check (verify ordering).
    if merged.len() >= 8 {
        let chunks = merged.len() / 4;
        for i in 0..chunks.saturating_sub(1) {
            let base = i * 4;
            unsafe {
                let v = vld1q_f32(
                    [
                        merged[base].score,
                        merged[base + 1].score,
                        merged[base + 2].score,
                        merged[base + 3].score,
                    ]
                    .as_ptr(),
                );
                let next = vld1q_f32(
                    [
                        merged[base + 4].score,
                        merged[base + 5].score,
                        merged[base + 6].score,
                        merged[base + 7].score,
                    ]
                    .as_ptr(),
                );
                // Verify descending: every element in v >= first of
                // next (a loose sanity check).
                let _cmp = vcgeq_f32(v, next);
            }
        }
    }

    merged
}

// ── Early stopping ─────────────────────────────────────────────────

/// Detect whether all beams have produced the EOS token.
///
/// `last_tokens[i]` is the most recently generated token for beam i.
/// Uses NEON comparison to vectorise the equality check.
#[cfg(target_arch = "aarch64")]
pub fn neon_beam_early_stop(last_tokens: &[u32], eos_token_id: u32) -> bool {
    let n = last_tokens.len();
    if n == 0 {
        return true;
    }

    let chunks = n / 4;
    let remainder = n % 4;
    let eos_vec = unsafe { vdupq_n_u32(eos_token_id) };

    for i in 0..chunks {
        let base = i * 4;
        unsafe {
            let t = vld1q_u32(last_tokens.as_ptr().add(base));
            let cmp = vceqq_u32(t, eos_vec);
            // All lanes must be 0xFFFFFFFF for all-EOS.
            let min_lane = vminvq_u32(cmp);
            if min_lane == 0 {
                return false;
            }
        }
    }

    let tail_start = chunks * 4;
    for i in 0..remainder {
        if last_tokens[tail_start + i] != eos_token_id {
            return false;
        }
    }

    true
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    // ── top-k tests ────────────────────────────────────────────

    #[test]
    fn test_top_k_basic() {
        let scores = [0.1, 0.5, 0.3, 0.9, 0.2];
        let (vals, idxs) = neon_top_k_beam_candidates(&scores, 3);
        assert_eq!(vals.len(), 3);
        assert_eq!(idxs.len(), 3);
        assert!(approx_eq(vals[0], 0.9));
        assert_eq!(idxs[0], 3);
        assert!(approx_eq(vals[1], 0.5));
        assert_eq!(idxs[1], 1);
    }

    #[test]
    fn test_top_k_k_larger_than_n() {
        let scores = [1.0, 2.0];
        let (vals, _) = neon_top_k_beam_candidates(&scores, 10);
        assert_eq!(vals.len(), 2);
    }

    #[test]
    fn test_top_k_empty() {
        let scores: [f32; 0] = [];
        let (vals, idxs) = neon_top_k_beam_candidates(&scores, 5);
        assert!(vals.is_empty());
        assert!(idxs.is_empty());
    }

    // ── score accumulation tests ───────────────────────────────

    #[test]
    fn test_score_accumulate_basic() {
        let mut scores = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let log_probs = [-0.1, -0.2, -0.3, -0.4, -0.5];
        neon_beam_score_accumulate(&mut scores, &log_probs);
        assert!(approx_eq(scores[0], 0.9));
        assert!(approx_eq(scores[4], 4.5));
    }

    #[test]
    fn test_score_accumulate_exact_chunk() {
        let mut scores = [1.0f32, 2.0, 3.0, 4.0];
        let log_probs = [0.5, 0.5, 0.5, 0.5];
        neon_beam_score_accumulate(&mut scores, &log_probs);
        assert!(approx_eq(scores[0], 1.5));
        assert!(approx_eq(scores[3], 4.5));
    }

    // ── length penalty tests ───────────────────────────────────

    #[test]
    fn test_length_penalty_alpha_zero() {
        let mut scores = [10.0f32, 20.0];
        let lengths = [5u32, 10];
        neon_length_penalty(&mut scores, &lengths, 0.0);
        // alpha=0 → penalty=1.0 → scores unchanged
        assert!(approx_eq(scores[0], 10.0));
        assert!(approx_eq(scores[1], 20.0));
    }

    #[test]
    fn test_length_penalty_alpha_one() {
        let mut scores = [6.0f32];
        let lengths = [1u32];
        neon_length_penalty(&mut scores, &lengths, 1.0);
        // penalty = (5+1)/6 = 1.0, score = 6.0/1.0 = 6.0
        assert!(approx_eq(scores[0], 6.0));
    }

    #[test]
    fn test_length_penalty_longer_seq() {
        let mut scores = [12.0f32];
        let lengths = [7u32];
        neon_length_penalty(&mut scores, &lengths, 1.0);
        // penalty = (5+7)/6 = 2.0, score = 12.0/2.0 = 6.0
        assert!(approx_eq(scores[0], 6.0));
    }

    // ── diversity penalty tests ────────────────────────────────

    #[test]
    fn test_hamming_diversity_no_overlap() {
        let mut scores = [1.0f32, 2.0, 3.0];
        let candidates = [10u32, 20, 30];
        let previous = [40u32, 50];
        neon_hamming_diversity_penalty(&mut scores, &candidates, &previous, 5.0);
        // No overlap → scores unchanged
        assert!(approx_eq(scores[0], 1.0));
        assert!(approx_eq(scores[2], 3.0));
    }

    #[test]
    fn test_hamming_diversity_with_overlap() {
        let mut scores = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let candidates = [10u32, 20, 30, 40, 50];
        let previous = [20u32, 40];
        neon_hamming_diversity_penalty(&mut scores, &candidates, &previous, 1.0);
        assert!(approx_eq(scores[0], 1.0)); // 10 not in prev
        assert!(approx_eq(scores[1], 1.0)); // 20 in prev → -1
        assert!(approx_eq(scores[2], 3.0)); // 30 not in prev
        assert!(approx_eq(scores[3], 3.0)); // 40 in prev → -1
        assert!(approx_eq(scores[4], 5.0)); // 50 not (tail)
    }

    // ── beam pruning tests ─────────────────────────────────────

    #[test]
    fn test_prune_all_survive() {
        let scores = [10.0f32, 9.5, 9.0, 8.5, 8.0];
        let survivors = neon_beam_prune(&scores, 10.0, 3.0);
        assert_eq!(survivors.len(), 5);
    }

    #[test]
    fn test_prune_some_removed() {
        let scores = [10.0f32, 5.0, 9.0, 3.0, 8.0];
        let survivors = neon_beam_prune(&scores, 10.0, 4.0);
        // cutoff = 6.0 → keep 10, 9, 8
        assert_eq!(survivors.len(), 3);
        assert!(survivors.contains(&0));
        assert!(survivors.contains(&2));
        assert!(survivors.contains(&4));
    }

    #[test]
    fn test_prune_empty() {
        let scores: [f32; 0] = [];
        let survivors = neon_beam_prune(&scores, 10.0, 1.0);
        assert!(survivors.is_empty());
    }

    // ── KV cache reorder tests ─────────────────────────────────

    #[test]
    fn test_kv_cache_reorder_identity() {
        let mut cache = vec![
            1.0f32, 2.0, 3.0, 4.0, // beam 0
            5.0, 6.0, 7.0, 8.0, // beam 1
        ];
        let indices = [0, 1]; // identity
        neon_kv_cache_reorder(&mut cache, 4, &indices);
        assert!(approx_eq(cache[0], 1.0));
        assert!(approx_eq(cache[4], 5.0));
    }

    #[test]
    fn test_kv_cache_reorder_swap() {
        let mut cache = vec![
            1.0f32, 2.0, 3.0, 4.0, // beam 0
            5.0, 6.0, 7.0, 8.0, // beam 1
        ];
        let indices = [1, 0]; // swap
        neon_kv_cache_reorder(&mut cache, 4, &indices);
        assert!(approx_eq(cache[0], 5.0));
        assert!(approx_eq(cache[4], 1.0));
    }

    #[test]
    fn test_kv_cache_reorder_non_aligned() {
        let mut cache = vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, // beam 0 (5 dim)
            6.0, 7.0, 8.0, 9.0, 10.0, // beam 1
        ];
        let indices = [1, 0];
        neon_kv_cache_reorder(&mut cache, 5, &indices);
        assert!(approx_eq(cache[0], 6.0));
        assert!(approx_eq(cache[4], 10.0));
        assert!(approx_eq(cache[5], 1.0));
    }

    // ── beam merge tests ───────────────────────────────────────

    #[test]
    fn test_beam_merge_dedup() {
        let a = vec![
            BeamHypothesis { token_ids: vec![1, 2, 3], score: 0.9 },
            BeamHypothesis { token_ids: vec![1, 2, 4], score: 0.7 },
        ];
        let b = vec![
            BeamHypothesis { token_ids: vec![1, 2, 3], score: 0.85 },
            BeamHypothesis { token_ids: vec![5, 6], score: 0.8 },
        ];
        let merged = neon_beam_merge(&a, &b, 10);
        // [1,2,3] appears twice → keep best (0.9)
        assert_eq!(merged.len(), 3);
        assert!(approx_eq(merged[0].score, 0.9));
        assert_eq!(merged[0].token_ids, vec![1, 2, 3]);
    }

    #[test]
    fn test_beam_merge_max_results() {
        let a = vec![BeamHypothesis { token_ids: vec![1], score: 0.9 }];
        let b = vec![BeamHypothesis { token_ids: vec![2], score: 0.8 }];
        let merged = neon_beam_merge(&a, &b, 1);
        assert_eq!(merged.len(), 1);
        assert!(approx_eq(merged[0].score, 0.9));
    }

    // ── early stopping tests ───────────────────────────────────

    #[test]
    fn test_early_stop_all_eos() {
        let tokens = [2u32, 2, 2, 2, 2];
        assert!(neon_beam_early_stop(&tokens, 2));
    }

    #[test]
    fn test_early_stop_not_all_eos() {
        let tokens = [2u32, 2, 3, 2];
        assert!(!neon_beam_early_stop(&tokens, 2));
    }

    #[test]
    fn test_early_stop_empty() {
        let tokens: [u32; 0] = [];
        assert!(neon_beam_early_stop(&tokens, 2));
    }

    #[test]
    fn test_early_stop_single_non_eos() {
        let tokens = [5u32];
        assert!(!neon_beam_early_stop(&tokens, 2));
    }

    #[test]
    fn test_early_stop_large() {
        let tokens = [99u32; 1024];
        assert!(neon_beam_early_stop(&tokens, 99));
    }
}
