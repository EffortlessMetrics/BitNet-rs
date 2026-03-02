//! ARM NEON beam search v2 operations for Apple Silicon (aarch64).
//!
//! Provides NEON-optimized beam search kernels: score accumulation,
//! top-k pruning, length penalty, diversity grouping, hypothesis
//! merging, and early stopping checks.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// NEON lane count for `float32x4_t`.
const LANES: usize = 4;

// ── beam_score_update ───────────────────────────────────────────────────

/// NEON-accelerated beam score accumulation: `scores[i] += log_probs[i]`.
///
/// # Safety
/// Requires `aarch64` target with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn beam_score_update_neon(scores: &mut [f32], log_probs: &[f32]) {
    debug_assert_eq!(scores.len(), log_probs.len());
    let len = scores.len();
    let chunks = len / LANES;
    let remainder = len % LANES;
    let sp = scores.as_mut_ptr();
    let lp = log_probs.as_ptr();

    for i in 0..chunks {
        let off = i * LANES;
        let s = vld1q_f32(sp.add(off));
        let l = vld1q_f32(lp.add(off));
        let r = vaddq_f32(s, l);
        vst1q_f32(sp.add(off), r);
    }

    let base = chunks * LANES;
    for i in 0..remainder {
        *sp.add(base + i) += *lp.add(base + i);
    }
}

/// Scalar fallback for beam score accumulation.
fn beam_score_update_scalar(scores: &mut [f32], log_probs: &[f32]) {
    debug_assert_eq!(scores.len(), log_probs.len());
    for (s, &lp) in scores.iter_mut().zip(log_probs.iter()) {
        *s += lp;
    }
}

/// Accumulate `log_probs` into `scores` element-wise.
///
/// Uses NEON on aarch64, scalar fallback otherwise.
///
/// # Panics
/// Panics (debug) if `scores.len() != log_probs.len()`.
pub fn beam_score_update(scores: &mut [f32], log_probs: &[f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { beam_score_update_neon(scores, log_probs) };
        }
    }
    beam_score_update_scalar(scores, log_probs);
}

// ── beam_prune_topk ─────────────────────────────────────────────────────

/// NEON-accelerated top-k selection via partial sort.
///
/// Returns indices of the `k` highest-scoring beams, sorted descending.
///
/// # Safety
/// Requires `aarch64` target with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn beam_prune_topk_neon(scores: &[f32], k: usize) -> Vec<usize> {
    beam_prune_topk_scalar(scores, k)
}

/// Scalar fallback for top-k beam selection.
fn beam_prune_topk_scalar(scores: &[f32], k: usize) -> Vec<usize> {
    let k = k.min(scores.len());
    if k == 0 {
        return Vec::new();
    }
    let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    // Partial sort: move top-k to front.
    indexed.select_nth_unstable_by(k - 1, |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    indexed.truncate(k);
    indexed.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    indexed.into_iter().map(|(i, _)| i).collect()
}

/// Select the top-k beam indices by score (descending).
///
/// Returns at most `k` indices. Uses NEON-assisted partial sort on aarch64.
pub fn beam_prune_topk(scores: &[f32], k: usize) -> Vec<usize> {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { beam_prune_topk_neon(scores, k) };
        }
    }
    beam_prune_topk_scalar(scores, k)
}

// ── length_penalty_scale ────────────────────────────────────────────────

/// Compute length penalty: `((5 + len) / 6) ^ alpha`.
#[inline]
fn lp_factor(length: f32, alpha: f32) -> f32 {
    ((5.0 + length) / 6.0).powf(alpha)
}

/// NEON-accelerated length penalty scaling.
///
/// Divides each score by its corresponding length penalty factor.
///
/// # Safety
/// Requires `aarch64` target with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn length_penalty_scale_neon(
    scores: &mut [f32],
    lengths: &[f32],
    alpha: f32,
) {
    debug_assert_eq!(scores.len(), lengths.len());
    let len = scores.len();
    let sp = scores.as_mut_ptr();

    // Compute penalty factors, then vectorised divide.
    let mut penalties = vec![0.0f32; len];
    for i in 0..len {
        penalties[i] = lp_factor(*lengths.as_ptr().add(i), alpha);
    }

    let pp = penalties.as_ptr();
    let chunks = len / LANES;
    let remainder = len % LANES;

    for i in 0..chunks {
        let off = i * LANES;
        let s = vld1q_f32(sp.add(off));
        let p = vld1q_f32(pp.add(off));
        // NEON reciprocal + multiply for division.
        let rcp = vrecpeq_f32(p);
        let rcp = vmulq_f32(vrecpsq_f32(p, rcp), rcp); // one Newton-Raphson step
        let r = vmulq_f32(s, rcp);
        vst1q_f32(sp.add(off), r);
    }

    let base = chunks * LANES;
    for i in 0..remainder {
        *sp.add(base + i) /= penalties[base + i];
    }
}

/// Scalar fallback for length penalty scaling.
fn length_penalty_scale_scalar(scores: &mut [f32], lengths: &[f32], alpha: f32) {
    debug_assert_eq!(scores.len(), lengths.len());
    for (s, &l) in scores.iter_mut().zip(lengths.iter()) {
        *s /= lp_factor(l, alpha);
    }
}

/// Apply length penalty to beam scores: `scores[i] /= lp(lengths[i], alpha)`.
///
/// Length penalty = `((5 + length) / 6) ^ alpha` (Wu et al., 2016).
///
/// # Panics
/// Panics (debug) if `scores.len() != lengths.len()`.
pub fn length_penalty_scale(scores: &mut [f32], lengths: &[f32], alpha: f32) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { length_penalty_scale_neon(scores, lengths, alpha) };
        }
    }
    length_penalty_scale_scalar(scores, lengths, alpha);
}

// ── diverse_beam_groups ─────────────────────────────────────────────────

/// Hamming distance between two token-id slices.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn hamming_distance_neon(a: &[u32], b: &[u32]) -> u32 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    let chunks = len / LANES;
    let remainder = len % LANES;
    let ap = a.as_ptr();
    let bp = b.as_ptr();

    let mut count: u32 = 0;

    for i in 0..chunks {
        let off = i * LANES;
        let va = vld1q_u32(ap.add(off));
        let vb = vld1q_u32(bp.add(off));
        let neq = vmvnq_u32(vceqq_u32(va, vb)); // 0xFFFFFFFF where !=
        // Count non-zero lanes: shift right 31 to get 1/0, then horizontal add.
        let ones = vshrq_n_u32(neq, 31);
        count += vaddvq_u32(ones);
    }

    let base = chunks * LANES;
    for i in 0..remainder {
        if *ap.add(base + i) != *bp.add(base + i) {
            count += 1;
        }
    }

    count
}

/// Scalar hamming distance between two token-id slices.
fn hamming_distance_scalar(a: &[u32], b: &[u32]) -> u32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .filter(|(x, y)| x != y)
        .count() as u32
}

/// Compute hamming distance (number of differing positions) between two
/// token-id sequences.
pub fn hamming_distance(a: &[u32], b: &[u32]) -> u32 {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { hamming_distance_neon(a, b) };
        }
    }
    hamming_distance_scalar(a, b)
}

/// Partition `num_beams` beams into `num_groups` diversity groups.
///
/// Returns a `Vec<Vec<usize>>` where each inner vec holds beam indices
/// assigned to that group. Beams are assigned round-robin to encourage
/// diversity, then reranked within each group by hamming distance to the
/// group leader when `token_ids` is provided.
///
/// If `token_ids` is `None`, plain round-robin assignment is used.
pub fn diverse_beam_groups(
    num_beams: usize,
    num_groups: usize,
    token_ids: Option<&[Vec<u32>]>,
) -> Vec<Vec<usize>> {
    if num_groups == 0 || num_beams == 0 {
        return vec![Vec::new(); num_groups.max(1)];
    }
    let ng = num_groups.min(num_beams);
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); ng];

    // Round-robin assignment.
    for beam in 0..num_beams {
        groups[beam % ng].push(beam);
    }

    // Optionally rerank within groups by hamming distance to leader.
    if let Some(tids) = token_ids {
        for g in &mut groups {
            if g.len() <= 1 || tids.is_empty() {
                continue;
            }
            let leader = g[0];
            if leader >= tids.len() {
                continue;
            }
            let leader_toks = &tids[leader];
            // Sort remaining beams by descending hamming distance to leader.
            g[1..].sort_by(|&a, &b| {
                let da = if a < tids.len() {
                    let min_len = leader_toks.len().min(tids[a].len());
                    hamming_distance(&leader_toks[..min_len], &tids[a][..min_len])
                } else {
                    0
                };
                let db = if b < tids.len() {
                    let min_len = leader_toks.len().min(tids[b].len());
                    hamming_distance(&leader_toks[..min_len], &tids[b][..min_len])
                } else {
                    0
                };
                db.cmp(&da)
            });
        }
    }

    groups
}

// ── beam_hypothesis_merge ───────────────────────────────────────────────

/// A completed beam hypothesis with its score.
#[derive(Debug, Clone, PartialEq)]
pub struct BeamHypothesis {
    /// Normalised score (higher is better).
    pub score: f32,
    /// Generated token ids.
    pub token_ids: Vec<u32>,
}

/// NEON-accelerated merge: find insertion point via vectorised comparison.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn beam_hypothesis_merge_neon(
    completed: &mut Vec<BeamHypothesis>,
    new: BeamHypothesis,
    max_hyps: usize,
) {
    beam_hypothesis_merge_scalar(completed, new, max_hyps);
}

/// Scalar fallback for hypothesis merge.
fn beam_hypothesis_merge_scalar(
    completed: &mut Vec<BeamHypothesis>,
    new: BeamHypothesis,
    max_hyps: usize,
) {
    // Insert in sorted position (descending by score).
    let pos = completed
        .iter()
        .position(|h| new.score > h.score)
        .unwrap_or(completed.len());
    completed.insert(pos, new);
    if completed.len() > max_hyps {
        completed.truncate(max_hyps);
    }
}

/// Merge a new hypothesis into the completed set (kept sorted, capped at
/// `max_hyps`).
///
/// Hypotheses are ordered by descending score. If the set is full and the
/// new hypothesis has a lower score than all existing ones it is discarded.
pub fn beam_hypothesis_merge(
    completed: &mut Vec<BeamHypothesis>,
    new: BeamHypothesis,
    max_hyps: usize,
) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { beam_hypothesis_merge_neon(completed, new, max_hyps) };
        }
    }
    beam_hypothesis_merge_scalar(completed, new, max_hyps);
}

// ── early_stopping_check ────────────────────────────────────────────────

/// NEON-accelerated early stopping check.
///
/// Returns `true` if the best completed hypothesis score is better than the
/// best possible score of any active beam (accounting for the maximum
/// remaining log-probability contribution).
///
/// # Safety
/// Requires `aarch64` target with NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn early_stopping_check_neon(
    active_scores: &[f32],
    best_completed: f32,
    max_remaining_bonus: f32,
) -> bool {
    let len = active_scores.len();
    if len == 0 {
        return true;
    }

    let ptr = active_scores.as_ptr();
    let bonus = vdupq_n_f32(max_remaining_bonus);
    let threshold = vdupq_n_f32(best_completed);

    let chunks = len / LANES;
    let remainder = len % LANES;

    for i in 0..chunks {
        let s = vld1q_f32(ptr.add(i * LANES));
        let optimistic = vaddq_f32(s, bonus);
        // If any lane's optimistic score > best_completed, can't stop early.
        let cmp = vcgtq_f32(optimistic, threshold);
        if vmaxvq_u32(cmp) != 0 {
            return false;
        }
    }

    let base = chunks * LANES;
    for i in 0..remainder {
        let optimistic = *ptr.add(base + i) + max_remaining_bonus;
        if optimistic > best_completed {
            return false;
        }
    }

    true
}

/// Scalar fallback for early stopping check.
fn early_stopping_check_scalar(
    active_scores: &[f32],
    best_completed: f32,
    max_remaining_bonus: f32,
) -> bool {
    active_scores
        .iter()
        .all(|&s| s + max_remaining_bonus <= best_completed)
}

/// Check whether early stopping is warranted.
///
/// Returns `true` when no active beam can possibly exceed `best_completed`
/// even with the most optimistic remaining contribution
/// (`max_remaining_bonus`).
pub fn early_stopping_check(
    active_scores: &[f32],
    best_completed: f32,
    max_remaining_bonus: f32,
) -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe {
                early_stopping_check_neon(active_scores, best_completed, max_remaining_bonus)
            };
        }
    }
    early_stopping_check_scalar(active_scores, best_completed, max_remaining_bonus)
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── beam_score_update tests ─────────────────────────────────────────

    #[test]
    fn test_score_update_basic() {
        let mut scores = vec![1.0, 2.0, 3.0, 4.0];
        let log_probs = vec![0.1, 0.2, 0.3, 0.4];
        beam_score_update(&mut scores, &log_probs);
        let expected = vec![1.1, 2.2, 3.3, 4.4];
        for (a, b) in scores.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} != {b}");
        }
    }

    #[test]
    fn test_score_update_empty() {
        let mut scores: Vec<f32> = vec![];
        let log_probs: Vec<f32> = vec![];
        beam_score_update(&mut scores, &log_probs);
        assert!(scores.is_empty());
    }

    #[test]
    fn test_score_update_single() {
        let mut scores = vec![5.0];
        let log_probs = vec![-1.0];
        beam_score_update(&mut scores, &log_probs);
        assert!((scores[0] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_score_update_not_multiple_of_4() {
        let mut scores = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let log_probs = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
        beam_score_update(&mut scores, &log_probs);
        for (i, &s) in scores.iter().enumerate() {
            let expected = (i + 1) as f32 + (i + 1) as f32 * 0.1;
            assert!((s - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn test_score_update_negative_logprobs() {
        let mut scores = vec![0.0; 8];
        let log_probs = vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0];
        beam_score_update(&mut scores, &log_probs);
        for (i, &s) in scores.iter().enumerate() {
            assert!((s - (-(i as f32 + 1.0))).abs() < 1e-6);
        }
    }

    #[test]
    fn test_score_update_large_batch() {
        let n = 1024;
        let mut scores = vec![1.0f32; n];
        let log_probs = vec![0.5f32; n];
        beam_score_update(&mut scores, &log_probs);
        for &s in &scores {
            assert!((s - 1.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_score_update_neon_scalar_parity() {
        let mut s_neon = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut s_scalar = s_neon.clone();
        let lp = vec![0.5, -0.5, 1.0, -1.0, 0.0];
        beam_score_update(&mut s_neon, &lp);
        beam_score_update_scalar(&mut s_scalar, &lp);
        for (a, b) in s_neon.iter().zip(s_scalar.iter()) {
            assert!((a - b).abs() < 1e-6, "neon={a} scalar={b}");
        }
    }

    #[test]
    fn test_score_update_zeros() {
        let mut scores = vec![0.0; 4];
        let log_probs = vec![0.0; 4];
        beam_score_update(&mut scores, &log_probs);
        for &s in &scores {
            assert_eq!(s, 0.0);
        }
    }

    #[test]
    fn test_score_update_accumulates() {
        let mut scores = vec![1.0; 4];
        let lp = vec![1.0; 4];
        beam_score_update(&mut scores, &lp);
        beam_score_update(&mut scores, &lp);
        for &s in &scores {
            assert!((s - 3.0).abs() < 1e-6);
        }
    }

    // ── beam_prune_topk tests ───────────────────────────────────────────

    #[test]
    fn test_topk_basic() {
        let scores = vec![1.0, 4.0, 2.0, 5.0, 3.0];
        let top = beam_prune_topk(&scores, 3);
        assert_eq!(top.len(), 3);
        assert_eq!(top[0], 3); // score 5.0
        assert_eq!(top[1], 1); // score 4.0
        assert_eq!(top[2], 4); // score 3.0
    }

    #[test]
    fn test_topk_k_larger_than_len() {
        let scores = vec![1.0, 2.0];
        let top = beam_prune_topk(&scores, 10);
        assert_eq!(top.len(), 2);
    }

    #[test]
    fn test_topk_k_zero() {
        let scores = vec![1.0, 2.0, 3.0];
        let top = beam_prune_topk(&scores, 0);
        assert!(top.is_empty());
    }

    #[test]
    fn test_topk_empty_scores() {
        let scores: Vec<f32> = vec![];
        let top = beam_prune_topk(&scores, 3);
        assert!(top.is_empty());
    }

    #[test]
    fn test_topk_single_element() {
        let scores = vec![42.0];
        let top = beam_prune_topk(&scores, 1);
        assert_eq!(top, vec![0]);
    }

    #[test]
    fn test_topk_all_equal() {
        let scores = vec![3.0; 8];
        let top = beam_prune_topk(&scores, 4);
        assert_eq!(top.len(), 4);
    }

    #[test]
    fn test_topk_negative_scores() {
        let scores = vec![-5.0, -1.0, -3.0, -2.0, -4.0];
        let top = beam_prune_topk(&scores, 2);
        assert_eq!(top[0], 1); // -1.0
        assert_eq!(top[1], 3); // -2.0
    }

    #[test]
    fn test_topk_descending_order() {
        let scores = vec![10.0, 30.0, 20.0, 50.0, 40.0];
        let top = beam_prune_topk(&scores, 5);
        for w in top.windows(2) {
            assert!(scores[w[0]] >= scores[w[1]]);
        }
    }

    #[test]
    fn test_topk_neon_scalar_parity() {
        let scores = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let neon_result = beam_prune_topk(&scores, 4);
        let scalar_result = beam_prune_topk_scalar(&scores, 4);
        assert_eq!(neon_result, scalar_result);
    }

    #[test]
    fn test_topk_large() {
        let n = 256;
        let scores: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let top = beam_prune_topk(&scores, 8);
        assert_eq!(top.len(), 8);
        assert_eq!(top[0], n - 1);
    }

    // ── length_penalty_scale tests ──────────────────────────────────────

    #[test]
    fn test_lp_alpha_zero() {
        let mut scores = vec![10.0, 20.0, 30.0, 40.0];
        let original = scores.clone();
        let lengths = vec![1.0, 5.0, 10.0, 20.0];
        length_penalty_scale(&mut scores, &lengths, 0.0);
        // alpha=0 → penalty=1.0, scores unchanged (NEON reciprocal tolerance).
        for (a, b) in scores.iter().zip(original.iter()) {
            assert!((a - b).abs() < 0.01, "{a} != {b}");
        }
    }

    #[test]
    fn test_lp_alpha_one() {
        let mut scores = vec![6.0; 4];
        let lengths = vec![1.0, 1.0, 1.0, 1.0];
        length_penalty_scale(&mut scores, &lengths, 1.0);
        // penalty = (5+1)/6 = 1.0
        for &s in &scores {
            assert!((s - 6.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_lp_longer_sequences_penalised() {
        let mut s1 = vec![10.0];
        let mut s2 = vec![10.0];
        length_penalty_scale(&mut s1, &[1.0], 1.0);
        length_penalty_scale(&mut s2, &[100.0], 1.0);
        // Longer sequence → larger penalty → smaller penalised score.
        assert!(s1[0] > s2[0]);
    }

    #[test]
    fn test_lp_empty() {
        let mut scores: Vec<f32> = vec![];
        let lengths: Vec<f32> = vec![];
        length_penalty_scale(&mut scores, &lengths, 1.0);
        assert!(scores.is_empty());
    }

    #[test]
    fn test_lp_neon_scalar_parity() {
        let mut s_neon = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let mut s_scalar = s_neon.clone();
        let lengths = vec![1.0, 3.0, 5.0, 10.0, 20.0];
        let alpha = 0.6;
        length_penalty_scale(&mut s_neon, &lengths, alpha);
        length_penalty_scale_scalar(&mut s_scalar, &lengths, alpha);
        for (a, b) in s_neon.iter().zip(s_scalar.iter()) {
            assert!(
                (a - b).abs() < 0.05,
                "neon={a} scalar={b} diff={}",
                (a - b).abs()
            );
        }
    }

    #[test]
    fn test_lp_single() {
        let mut scores = vec![12.0];
        let lengths = vec![7.0];
        length_penalty_scale(&mut scores, &lengths, 1.0);
        let expected = 12.0 / ((5.0 + 7.0) / 6.0);
        assert!((scores[0] - expected).abs() < 1e-3);
    }

    #[test]
    fn test_lp_not_multiple_of_4() {
        let mut scores = vec![6.0; 7];
        let lengths = vec![1.0; 7];
        length_penalty_scale(&mut scores, &lengths, 1.0);
        for &s in &scores {
            assert!((s - 6.0).abs() < 1e-3);
        }
    }

    #[test]
    fn test_lp_high_alpha() {
        let mut scores = vec![100.0; 4];
        let lengths = vec![50.0; 4];
        length_penalty_scale(&mut scores, &lengths, 2.0);
        let penalty = ((5.0 + 50.0) / 6.0_f32).powf(2.0);
        for &s in &scores {
            assert!((s - 100.0 / penalty).abs() < 0.5);
        }
    }

    // ── diverse_beam_groups tests ───────────────────────────────────────

    #[test]
    fn test_groups_round_robin() {
        let groups = diverse_beam_groups(6, 3, None);
        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0], vec![0, 3]);
        assert_eq!(groups[1], vec![1, 4]);
        assert_eq!(groups[2], vec![2, 5]);
    }

    #[test]
    fn test_groups_more_groups_than_beams() {
        let groups = diverse_beam_groups(2, 5, None);
        assert_eq!(groups.len(), 2); // capped at num_beams
        assert_eq!(groups[0], vec![0]);
        assert_eq!(groups[1], vec![1]);
    }

    #[test]
    fn test_groups_single_group() {
        let groups = diverse_beam_groups(4, 1, None);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0], vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_groups_zero_beams() {
        let groups = diverse_beam_groups(0, 3, None);
        assert_eq!(groups.len(), 3);
        for g in &groups {
            assert!(g.is_empty());
        }
    }

    #[test]
    fn test_groups_zero_groups() {
        let groups = diverse_beam_groups(4, 0, None);
        assert_eq!(groups.len(), 1);
        assert!(groups[0].is_empty());
    }

    #[test]
    fn test_groups_with_token_ids() {
        let tids = vec![
            vec![1, 2, 3],
            vec![1, 2, 3], // same as leader
            vec![4, 5, 6], // maximally different
            vec![1, 5, 3], // partially different
        ];
        let groups = diverse_beam_groups(4, 2, Some(&tids));
        assert_eq!(groups.len(), 2);
        // group 0: beams 0, 2 — beam 2 is more diverse than beam 0's twin
        assert_eq!(groups[0][0], 0);
        assert_eq!(groups[0][1], 2);
    }

    #[test]
    fn test_groups_single_beam() {
        let groups = diverse_beam_groups(1, 1, None);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0], vec![0]);
    }

    #[test]
    fn test_groups_all_beams_covered() {
        let groups = diverse_beam_groups(10, 3, None);
        let all: Vec<usize> = groups.iter().flatten().copied().collect();
        for i in 0..10 {
            assert!(all.contains(&i), "beam {i} missing");
        }
    }

    // ── hamming_distance tests ──────────────────────────────────────────

    #[test]
    fn test_hamming_identical() {
        let a = vec![1, 2, 3, 4];
        assert_eq!(hamming_distance(&a, &a), 0);
    }

    #[test]
    fn test_hamming_all_different() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        assert_eq!(hamming_distance(&a, &b), 4);
    }

    #[test]
    fn test_hamming_partial() {
        let a = vec![1, 2, 3, 4, 5];
        let b = vec![1, 0, 3, 0, 5];
        assert_eq!(hamming_distance(&a, &b), 2);
    }

    #[test]
    fn test_hamming_empty() {
        let a: Vec<u32> = vec![];
        assert_eq!(hamming_distance(&a, &a), 0);
    }

    #[test]
    fn test_hamming_single() {
        assert_eq!(hamming_distance(&[1], &[1]), 0);
        assert_eq!(hamming_distance(&[1], &[2]), 1);
    }

    #[test]
    fn test_hamming_neon_scalar_parity() {
        let a: Vec<u32> = (0..17).collect();
        let b: Vec<u32> = (0..17).map(|i| if i % 3 == 0 { i + 1 } else { i }).collect();
        let neon_r = hamming_distance(&a, &b);
        let scalar_r = hamming_distance_scalar(&a, &b);
        assert_eq!(neon_r, scalar_r);
    }

    #[test]
    fn test_hamming_large() {
        let n = 512;
        let a: Vec<u32> = (0..n).collect();
        let b: Vec<u32> = (0..n).map(|i| i + 1).collect();
        assert_eq!(hamming_distance(&a, &b), n as u32);
    }

    // ── beam_hypothesis_merge tests ─────────────────────────────────────

    fn hyp(score: f32, toks: &[u32]) -> BeamHypothesis {
        BeamHypothesis {
            score,
            token_ids: toks.to_vec(),
        }
    }

    #[test]
    fn test_merge_into_empty() {
        let mut completed = Vec::new();
        beam_hypothesis_merge(&mut completed, hyp(5.0, &[1, 2]), 3);
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].score, 5.0);
    }

    #[test]
    fn test_merge_maintains_order() {
        let mut completed = Vec::new();
        beam_hypothesis_merge(&mut completed, hyp(3.0, &[1]), 5);
        beam_hypothesis_merge(&mut completed, hyp(5.0, &[2]), 5);
        beam_hypothesis_merge(&mut completed, hyp(1.0, &[3]), 5);
        beam_hypothesis_merge(&mut completed, hyp(4.0, &[4]), 5);
        let scores: Vec<f32> = completed.iter().map(|h| h.score).collect();
        assert_eq!(scores, vec![5.0, 4.0, 3.0, 1.0]);
    }

    #[test]
    fn test_merge_truncates() {
        let mut completed = Vec::new();
        for i in 0..10 {
            beam_hypothesis_merge(&mut completed, hyp(i as f32, &[i as u32]), 3);
        }
        assert_eq!(completed.len(), 3);
        assert_eq!(completed[0].score, 9.0);
        assert_eq!(completed[2].score, 7.0);
    }

    #[test]
    fn test_merge_discard_low() {
        let mut completed = vec![hyp(10.0, &[1]), hyp(8.0, &[2]), hyp(6.0, &[3])];
        beam_hypothesis_merge(&mut completed, hyp(1.0, &[4]), 3);
        assert_eq!(completed.len(), 3);
        // The worst is now the newly inserted at position 3, which got truncated
        assert_eq!(completed[2].score, 6.0);
    }

    #[test]
    fn test_merge_equal_scores() {
        let mut completed = Vec::new();
        beam_hypothesis_merge(&mut completed, hyp(5.0, &[1]), 5);
        beam_hypothesis_merge(&mut completed, hyp(5.0, &[2]), 5);
        assert_eq!(completed.len(), 2);
    }

    #[test]
    fn test_merge_max_hyps_one() {
        let mut completed = Vec::new();
        beam_hypothesis_merge(&mut completed, hyp(3.0, &[1]), 1);
        beam_hypothesis_merge(&mut completed, hyp(5.0, &[2]), 1);
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].score, 5.0);
    }

    #[test]
    fn test_merge_preserves_token_ids() {
        let mut completed = Vec::new();
        beam_hypothesis_merge(&mut completed, hyp(5.0, &[10, 20, 30]), 3);
        assert_eq!(completed[0].token_ids, vec![10, 20, 30]);
    }

    #[test]
    fn test_merge_neon_scalar_parity() {
        let mut c_neon = Vec::new();
        let mut c_scalar = Vec::new();
        let hyps = vec![
            hyp(3.0, &[1]),
            hyp(7.0, &[2]),
            hyp(1.0, &[3]),
            hyp(5.0, &[4]),
        ];
        for h in &hyps {
            beam_hypothesis_merge(&mut c_neon, h.clone(), 3);
            beam_hypothesis_merge_scalar(&mut c_scalar, h.clone(), 3);
        }
        assert_eq!(c_neon, c_scalar);
    }

    // ── early_stopping_check tests ──────────────────────────────────────

    #[test]
    fn test_early_stop_all_below() {
        let active = vec![1.0, 2.0, 3.0, 4.0];
        assert!(early_stopping_check(&active, 10.0, 1.0));
    }

    #[test]
    fn test_early_stop_one_can_beat() {
        let active = vec![1.0, 2.0, 3.0, 9.5];
        assert!(!early_stopping_check(&active, 10.0, 1.0));
    }

    #[test]
    fn test_early_stop_empty_active() {
        let active: Vec<f32> = vec![];
        assert!(early_stopping_check(&active, 5.0, 100.0));
    }

    #[test]
    fn test_early_stop_zero_bonus() {
        let active = vec![3.0, 4.0, 5.0];
        assert!(early_stopping_check(&active, 5.0, 0.0));
    }

    #[test]
    fn test_early_stop_negative_scores() {
        let active = vec![-10.0, -20.0, -5.0, -15.0];
        assert!(early_stopping_check(&active, 0.0, 2.0));
    }

    #[test]
    fn test_early_stop_exact_boundary() {
        let active = vec![5.0];
        // 5.0 + 5.0 = 10.0, equals best_completed → should stop
        assert!(early_stopping_check(&active, 10.0, 5.0));
    }

    #[test]
    fn test_early_stop_large_bonus_prevents() {
        let active = vec![1.0, 2.0, 3.0, 4.0];
        assert!(!early_stopping_check(&active, 10.0, 100.0));
    }

    #[test]
    fn test_early_stop_single_active() {
        assert!(early_stopping_check(&[1.0], 5.0, 1.0));
        assert!(!early_stopping_check(&[4.5], 5.0, 1.0));
    }

    #[test]
    fn test_early_stop_neon_scalar_parity() {
        let active = vec![1.0, 5.0, 3.0, 7.0, 2.0, 8.0, 4.0];
        let best = 10.0;
        let bonus = 1.5;
        let neon_r = early_stopping_check(&active, best, bonus);
        let scalar_r = early_stopping_check_scalar(&active, best, bonus);
        assert_eq!(neon_r, scalar_r);
    }

    #[test]
    fn test_early_stop_not_multiple_of_4() {
        let active = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        assert!(early_stopping_check(&active, 100.0, 0.0));
        assert!(!early_stopping_check(&active, 5.0, 5.0));
    }

    // ── integration / cross-operation tests ─────────────────────────────

    #[test]
    fn test_beam_search_pipeline() {
        // Simulate a mini beam search step.
        let mut scores = vec![0.0; 4];
        let log_probs = vec![-1.0, -0.5, -2.0, -0.1];
        beam_score_update(&mut scores, &log_probs);

        let top2 = beam_prune_topk(&scores, 2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0], 3); // best: -0.1
        assert_eq!(top2[1], 1); // second: -0.5
    }

    #[test]
    fn test_penalty_then_topk() {
        let mut scores = vec![10.0, 10.0, 10.0, 10.0];
        let lengths = vec![1.0, 5.0, 10.0, 20.0];
        length_penalty_scale(&mut scores, &lengths, 1.0);
        let top = beam_prune_topk(&scores, 2);
        // Shortest sequence should rank first after penalty.
        assert_eq!(top[0], 0);
    }

    #[test]
    fn test_merge_then_early_stop() {
        let mut completed = Vec::new();
        beam_hypothesis_merge(&mut completed, hyp(8.0, &[1, 2]), 5);
        beam_hypothesis_merge(&mut completed, hyp(10.0, &[3, 4]), 5);
        let best = completed[0].score; // 10.0
        let active = vec![3.0, 4.0, 5.0, 6.0];
        assert!(early_stopping_check(&active, best, 2.0));
    }

    #[test]
    fn test_diversity_groups_then_merge() {
        let groups = diverse_beam_groups(6, 2, None);
        assert_eq!(groups.len(), 2);
        let mut completed = Vec::new();
        for &beam_idx in &groups[0] {
            beam_hypothesis_merge(
                &mut completed,
                hyp(beam_idx as f32, &[beam_idx as u32]),
                3,
            );
        }
        assert!(!completed.is_empty());
    }

    #[test]
    fn test_score_update_inf_nan() {
        let mut scores = vec![f32::NEG_INFINITY, 0.0, f32::INFINITY, 1.0];
        let lp = vec![1.0, f32::NEG_INFINITY, 0.0, -1.0];
        beam_score_update(&mut scores, &lp);
        assert!(scores[0] == f32::NEG_INFINITY);
        assert!(scores[1] == f32::NEG_INFINITY);
        assert!(scores[2] == f32::INFINITY);
        assert!((scores[3] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_topk_two_elements_k_two() {
        let scores = vec![3.0, 7.0];
        let top = beam_prune_topk(&scores, 2);
        assert_eq!(top, vec![1, 0]);
    }

    #[test]
    fn test_early_stop_all_active_negative_large_bonus() {
        let active = vec![-100.0, -200.0, -300.0, -400.0];
        // With enough bonus, active beams can still beat completed.
        assert!(!early_stopping_check(&active, -50.0, 500.0));
        // Without enough bonus, they cannot.
        assert!(early_stopping_check(&active, -50.0, 10.0));
    }

    #[test]
    fn test_full_beam_search_flow() {
        let num_beams = 4;
        let vocab_size = 8;
        let max_steps = 3;
        let k = 2;

        let mut scores = vec![0.0f32; num_beams];
        let mut completed: Vec<BeamHypothesis> = Vec::new();

        for step in 0..max_steps {
            // Simulate log probs.
            let log_probs: Vec<f32> = (0..num_beams)
                .map(|i| -((i + step) as f32) * 0.1 - 0.1)
                .collect();
            beam_score_update(&mut scores, &log_probs);

            let lengths: Vec<f32> = vec![(step + 1) as f32; num_beams];
            let mut penalised = scores.clone();
            length_penalty_scale(&mut penalised, &lengths, 0.6);

            let top = beam_prune_topk(&penalised, k);
            assert_eq!(top.len(), k);

            // Best beam completes.
            beam_hypothesis_merge(
                &mut completed,
                hyp(penalised[top[0]], &[top[0] as u32]),
                vocab_size,
            );

            if early_stopping_check(&penalised, completed[0].score, 0.0) {
                break;
            }
        }

        assert!(!completed.is_empty());
        // Completed hypotheses are sorted descending.
        for w in completed.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }
}
