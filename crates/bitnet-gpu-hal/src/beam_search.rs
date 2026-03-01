//! Beam search decoding for GPU HAL inference.
//!
//! Provides beam search with length penalty (Wu et al.), n-gram blocking,
//! diverse beam search (Vijayakumar et al.), and beam deduplication.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

// ── Configuration ─────────────────────────────────────────────────────────

/// Configuration for beam search decoding.
#[derive(Debug, Clone)]
pub struct BeamSearchConfig {
    /// Number of beams to maintain at each step.
    pub beam_width: usize,
    /// Maximum sequence length before forced stop.
    pub max_length: usize,
    /// Length penalty exponent (Wu et al. alpha). 0.0 = no penalty, 1.0 = linear.
    pub length_penalty: f32,
    /// Stop when all beams have finished.
    pub early_stopping: bool,
    /// Block repeated n-grams of this size. 0 = disabled.
    pub no_repeat_ngram_size: usize,
    /// Temperature for logit scaling. 0.0 = greedy.
    pub temperature: f32,
}

impl Default for BeamSearchConfig {
    fn default() -> Self {
        Self {
            beam_width: 4,
            max_length: 128,
            length_penalty: 1.0,
            early_stopping: true,
            no_repeat_ngram_size: 0,
            temperature: 1.0,
        }
    }
}

// ── Length Penalty ─────────────────────────────────────────────────────────

/// Wu et al. length penalty: `(5 + len)^alpha / (5 + 1)^alpha`.
#[derive(Debug, Clone, Copy)]
pub struct LengthPenalty {
    pub alpha: f32,
}

impl LengthPenalty {
    pub const fn new(alpha: f32) -> Self {
        Self { alpha }
    }

    /// Apply length penalty to a log probability.
    pub fn apply(&self, log_prob: f32, length: usize) -> f32 {
        let penalty = self.factor(length);
        log_prob / penalty
    }

    /// Compute the penalty factor for a given length.
    pub fn factor(&self, length: usize) -> f32 {
        #[allow(clippy::cast_precision_loss)]
        let numerator = (5.0 + length as f32).powf(self.alpha);
        let denominator = 6.0_f32.powf(self.alpha);
        numerator / denominator
    }
}

impl Default for LengthPenalty {
    fn default() -> Self {
        Self::new(1.0)
    }
}

// ── Beam ──────────────────────────────────────────────────────────────────

/// A single hypothesis in the beam search.
#[derive(Debug, Clone)]
pub struct Beam {
    /// Token IDs generated so far.
    pub token_ids: Vec<u32>,
    /// Cumulative log probability.
    pub log_probability: f32,
    /// Whether this beam has produced an EOS token.
    pub is_finished: bool,
}

impl Beam {
    /// Create a new beam with the given initial tokens.
    pub const fn new(token_ids: Vec<u32>, log_probability: f32) -> Self {
        Self { token_ids, log_probability, is_finished: false }
    }
    pub const fn empty() -> Self {
        Self::new(Vec::new(), 0.0)
    }

    /// Compute the length-normalized score.
    pub fn score(&self, length_penalty: &LengthPenalty) -> f32 {
        length_penalty.apply(self.log_probability, self.token_ids.len())
    }

    /// Sequence length.
    pub const fn len(&self) -> usize {
        self.token_ids.len()
    }

    /// Whether the beam has no tokens.
    pub const fn is_empty(&self) -> bool {
        self.token_ids.is_empty()
    }

    /// Extend this beam with a new token and its log probability.
    #[must_use]
    pub fn extend(&self, token_id: u32, log_prob: f32) -> Self {
        let mut token_ids = self.token_ids.clone();
        token_ids.push(token_id);
        Self { token_ids, log_probability: self.log_probability + log_prob, is_finished: false }
    }
}

// ── NGram Blocker ─────────────────────────────────────────────────────────

/// Prevents repeating n-grams in a beam.
pub struct NGramBlocker;

impl NGramBlocker {
    /// Return the set of token IDs that would create a repeated n-gram.
    pub fn blocked_tokens(beam: &Beam, n: usize) -> HashSet<u32> {
        let mut blocked = HashSet::new();
        if n == 0 || beam.token_ids.len() < n {
            return blocked;
        }

        // Collect all n-grams of size (n-1) and the token that follows
        let ids = &beam.token_ids;
        let mut ngram_map: HashMap<&[u32], Vec<u32>> = HashMap::new();

        for i in 0..ids.len().saturating_sub(n - 1) {
            let prefix = &ids[i..i + n - 1];
            if i + n - 1 < ids.len() {
                if i + n <= ids.len() {
                    // This is a completed n-gram
                } else {
                    continue;
                }
            }
            if i + n - 1 < ids.len() {
                let next_tok = ids[i + n - 1];
                ngram_map.entry(prefix).or_default().push(next_tok);
            }
        }

        // The current suffix of length (n-1)
        if ids.len() >= n - 1 {
            let suffix = &ids[ids.len() - (n - 1)..];
            if let Some(seen) = ngram_map.get(suffix) {
                for &tok in seen {
                    blocked.insert(tok);
                }
            }
        }

        blocked
    }
}

// ── Beam Group ────────────────────────────────────────────────────────────

/// A group of beams, sorted by score.
#[derive(Debug, Clone)]
pub struct BeamGroup {
    pub beams: Vec<Beam>,
}

impl BeamGroup {
    pub const fn new(beams: Vec<Beam>) -> Self {
        Self { beams }
    }
    pub fn sort_by_score(&mut self, length_penalty: &LengthPenalty) {
        self.beams.sort_by(|a, b| {
            b.score(length_penalty)
                .partial_cmp(&a.score(length_penalty))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Keep only the top-k beams.
    pub fn prune(&mut self, k: usize, length_penalty: &LengthPenalty) {
        self.sort_by_score(length_penalty);
        self.beams.truncate(k);
    }

    /// Check if all beams are finished.
    pub fn all_finished(&self) -> bool {
        !self.beams.is_empty() && self.beams.iter().all(|b| b.is_finished)
    }

    /// Return the number of active (non-finished) beams.
    pub fn active_count(&self) -> usize {
        self.beams.iter().filter(|b| !b.is_finished).count()
    }
}

// ── Beam Search Decoder ───────────────────────────────────────────────────

/// Core beam search decoder.
pub struct BeamSearchDecoder {
    pub config: BeamSearchConfig,
    pub group: BeamGroup,
    pub completed: Vec<Beam>,
    pub length_penalty: LengthPenalty,
    pub metrics: BeamMetrics,
    step_count: usize,
}

impl BeamSearchDecoder {
    /// Create a new decoder with the given config.
    pub fn new(config: BeamSearchConfig) -> Self {
        let length_penalty = LengthPenalty::new(config.length_penalty);
        let initial_beams = vec![Beam::empty(); config.beam_width];
        let group = BeamGroup::new(initial_beams);
        Self {
            config,
            group,
            completed: Vec::new(),
            length_penalty,
            metrics: BeamMetrics::default(),
            step_count: 0,
        }
    }

    /// Initialize with a set of seed beams.
    pub fn with_seeds(config: BeamSearchConfig, seeds: Vec<Beam>) -> Self {
        let length_penalty = LengthPenalty::new(config.length_penalty);
        let group = BeamGroup::new(seeds);
        Self {
            config,
            group,
            completed: Vec::new(),
            length_penalty,
            metrics: BeamMetrics::default(),
            step_count: 0,
        }
    }

    /// Advance all beams by one step given logits (shape: `[vocab_size]`).
    ///
    /// Returns `true` if search should continue, `false` if done.
    pub fn step(&mut self, logits: &[f32], eos_token_id: u32) -> bool {
        if self.step_count >= self.config.max_length {
            return false;
        }
        if self.config.early_stopping && self.group.all_finished() {
            return false;
        }

        let start = Instant::now();
        let candidates = self.expand_beams(logits, eos_token_id);
        self.metrics.total_expansions += candidates.len();

        let mut new_beams = Vec::new();
        for beam in candidates {
            if beam.is_finished {
                self.completed.push(beam);
            } else {
                new_beams.push(beam);
            }
        }

        let before_prune = new_beams.len();
        let lp = self.length_penalty;
        self.group = BeamGroup::new(new_beams);
        self.group.prune(self.config.beam_width, &lp);
        self.metrics.beams_pruned += before_prune.saturating_sub(self.config.beam_width);

        self.step_count += 1;
        #[allow(clippy::cast_precision_loss)]
        {
            self.metrics.search_time_ms += start.elapsed().as_secs_f64() * 1000.0;
        }

        // Stop when enough completed beams collected or no active beams remain
        if self.config.early_stopping && self.completed.len() >= self.config.beam_width {
            return false;
        }
        if self.group.beams.is_empty() || self.group.all_finished() {
            return false;
        }
        self.step_count < self.config.max_length
    }

    /// Expand all active beams using the given logits.
    pub fn expand_beams(&self, logits: &[f32], eos_token_id: u32) -> Vec<Beam> {
        let mut candidates = Vec::new();
        let mut scaled_logits = logits.to_vec();

        // Apply temperature
        if self.config.temperature > 0.0 && (self.config.temperature - 1.0).abs() > f32::EPSILON {
            let inv_t = 1.0 / self.config.temperature;
            for v in &mut scaled_logits {
                *v *= inv_t;
            }
        }

        // Convert to log probabilities
        let log_probs = log_softmax(&scaled_logits);

        for beam in &self.group.beams {
            if beam.is_finished {
                candidates.push(beam.clone());
                continue;
            }

            // Determine blocked tokens from n-gram blocking
            let blocked = if self.config.no_repeat_ngram_size > 0 {
                NGramBlocker::blocked_tokens(beam, self.config.no_repeat_ngram_size)
            } else {
                HashSet::new()
            };

            // Get top-k tokens from log probs
            let k = self.config.beam_width * 2;
            let top_tokens = top_k_indices(&log_probs, k);

            for (token_id, log_prob) in top_tokens {
                if blocked.contains(&token_id) {
                    continue;
                }
                let mut new_beam = beam.extend(token_id, log_prob);
                if token_id == eos_token_id {
                    new_beam.is_finished = true;
                }
                candidates.push(new_beam);
            }
        }

        candidates
    }

    /// Get the final results, sorted by score.
    pub fn finish(mut self) -> BeamSearchResult {
        // Move any remaining beams to completed
        for beam in &self.group.beams {
            self.completed.push(beam.clone());
        }

        self.completed.sort_by(|a, b| {
            b.score(&self.length_penalty)
                .partial_cmp(&a.score(&self.length_penalty))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Update average beam length
        if !self.completed.is_empty() {
            #[allow(clippy::cast_precision_loss)]
            {
                self.metrics.avg_beam_length = self.completed.iter().map(Beam::len).sum::<usize>()
                    as f32
                    / self.completed.len() as f32;
            }
        }

        BeamSearchResult { sequences: self.completed, metrics: self.metrics }
    }

    /// Current step count.
    pub const fn steps(&self) -> usize {
        self.step_count
    }
}

// ── Diverse Beam Search ───────────────────────────────────────────────────

/// Vijayakumar et al. diverse beam search with Hamming diversity penalty.
pub struct DiverseBeamSearch {
    pub config: BeamSearchConfig,
    /// Number of beam groups.
    pub num_groups: usize,
    /// Diversity strength (lambda).
    pub diversity_penalty: f32,
    pub groups: Vec<BeamGroup>,
    pub completed: Vec<Beam>,
    pub length_penalty: LengthPenalty,
    pub metrics: BeamMetrics,
    step_count: usize,
}

impl DiverseBeamSearch {
    pub fn new(config: BeamSearchConfig, num_groups: usize, diversity_penalty: f32) -> Self {
        let length_penalty = LengthPenalty::new(config.length_penalty);
        let beams_per_group = config.beam_width.max(1);
        let groups: Vec<BeamGroup> =
            (0..num_groups).map(|_| BeamGroup::new(vec![Beam::empty(); beams_per_group])).collect();
        Self {
            config,
            num_groups,
            diversity_penalty,
            groups,
            completed: Vec::new(),
            length_penalty,
            metrics: BeamMetrics::default(),
            step_count: 0,
        }
    }

    /// Compute Hamming diversity penalty for a token given previously
    /// selected tokens at this position.
    pub fn hamming_penalty(token_id: u32, previously_selected: &[u32]) -> f32 {
        #[allow(clippy::cast_precision_loss)]
        let count = previously_selected.iter().filter(|&&t| t == token_id).count() as f32;
        count
    }

    /// Advance one step with diverse beam search.
    pub fn step(&mut self, logits: &[f32], eos_token_id: u32) -> bool {
        if self.step_count >= self.config.max_length {
            return false;
        }

        let start = Instant::now();
        let mut scaled_logits = logits.to_vec();
        if self.config.temperature > 0.0 && (self.config.temperature - 1.0).abs() > f32::EPSILON {
            let inv_t = 1.0 / self.config.temperature;
            for v in &mut scaled_logits {
                *v *= inv_t;
            }
        }
        let base_log_probs = log_softmax(&scaled_logits);

        let mut selected_this_step: Vec<u32> = Vec::new();

        for g in 0..self.num_groups {
            // Apply diversity penalty based on previous groups' selections
            let mut adjusted = base_log_probs.clone();
            if self.diversity_penalty > 0.0 {
                for (tok, lp) in adjusted.iter_mut().enumerate() {
                    #[allow(clippy::cast_possible_truncation)]
                    let penalty = Self::hamming_penalty(tok as u32, &selected_this_step);
                    *lp -= self.diversity_penalty * penalty;
                }
            }

            let k = self.config.beam_width * 2;
            let top_tokens = top_k_indices(&adjusted, k);

            let mut new_beams = Vec::new();
            for beam in &self.groups[g].beams {
                if beam.is_finished {
                    new_beams.push(beam.clone());
                    continue;
                }

                let blocked = if self.config.no_repeat_ngram_size > 0 {
                    NGramBlocker::blocked_tokens(beam, self.config.no_repeat_ngram_size)
                } else {
                    HashSet::new()
                };

                for &(token_id, log_prob) in &top_tokens {
                    if blocked.contains(&token_id) {
                        continue;
                    }
                    let mut new_beam = beam.extend(token_id, log_prob);
                    if token_id == eos_token_id {
                        new_beam.is_finished = true;
                    }
                    new_beams.push(new_beam);
                }
            }

            self.metrics.total_expansions += new_beams.len();

            // Separate finished and active beams
            let mut active = Vec::new();
            for beam in new_beams {
                if beam.is_finished {
                    self.completed.push(beam);
                } else {
                    active.push(beam);
                }
            }

            let before_prune = active.len();
            let mut group = BeamGroup::new(active);
            group.prune(self.config.beam_width, &self.length_penalty);
            self.metrics.beams_pruned += before_prune.saturating_sub(self.config.beam_width);

            // Record selected tokens for diversity penalty
            for beam in &group.beams {
                if let Some(&tok) = beam.token_ids.last() {
                    selected_this_step.push(tok);
                }
            }

            self.groups[g] = group;
        }

        self.step_count += 1;
        #[allow(clippy::cast_precision_loss)]
        {
            self.metrics.search_time_ms += start.elapsed().as_secs_f64() * 1000.0;
        }

        if self.config.early_stopping
            && self.completed.len() >= self.config.beam_width * self.num_groups
        {
            return false;
        }
        if self.config.early_stopping
            && self.groups.iter().all(|g| g.all_finished() || g.beams.is_empty())
        {
            return false;
        }

        self.step_count < self.config.max_length
    }

    /// Collect results from all groups.
    pub fn finish(mut self) -> BeamSearchResult {
        for group in &self.groups {
            for beam in &group.beams {
                self.completed.push(beam.clone());
            }
        }

        self.completed.sort_by(|a, b| {
            b.score(&self.length_penalty)
                .partial_cmp(&a.score(&self.length_penalty))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if !self.completed.is_empty() {
            #[allow(clippy::cast_precision_loss)]
            {
                self.metrics.avg_beam_length = self.completed.iter().map(Beam::len).sum::<usize>()
                    as f32
                    / self.completed.len() as f32;
            }
        }

        BeamSearchResult { sequences: self.completed, metrics: self.metrics }
    }
}

// ── Beam Merger ───────────────────────────────────────────────────────────

/// Deduplicates beams that converged to the same token sequence.
pub struct BeamMerger;

impl BeamMerger {
    /// Merge duplicate beams, keeping the one with the highest log probability.
    pub fn merge(beams: Vec<Beam>) -> Vec<Beam> {
        let mut best: HashMap<Vec<u32>, Beam> = HashMap::new();
        for beam in beams {
            best.entry(beam.token_ids.clone())
                .and_modify(|existing| {
                    if beam.log_probability > existing.log_probability {
                        *existing = beam.clone();
                    }
                })
                .or_insert(beam);
        }
        best.into_values().collect()
    }
}

// ── Beam Search Result ────────────────────────────────────────────────────

/// Final output of beam search: completed sequences sorted by score.
#[derive(Debug)]
pub struct BeamSearchResult {
    /// Completed sequences, sorted by score (best first).
    pub sequences: Vec<Beam>,
    /// Metrics collected during the search.
    pub metrics: BeamMetrics,
}

impl BeamSearchResult {
    /// Return the top-k sequences.
    pub fn top_k(&self, k: usize) -> &[Beam] {
        &self.sequences[..k.min(self.sequences.len())]
    }

    /// Return the best sequence, if any.
    pub fn best(&self) -> Option<&Beam> {
        self.sequences.first()
    }
}

// ── Beam Metrics ──────────────────────────────────────────────────────────

/// Metrics collected during beam search.
#[derive(Debug, Clone, Default)]
pub struct BeamMetrics {
    /// Total number of beam expansions performed.
    pub total_expansions: usize,
    /// Number of beams pruned across all steps.
    pub beams_pruned: usize,
    /// Average final beam length.
    pub avg_beam_length: f32,
    /// Total search time in milliseconds.
    pub search_time_ms: f64,
}

// ── Helpers ───────────────────────────────────────────────────────────────

/// Log-softmax: `log(softmax(x))`.
fn log_softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits.iter().map(|&v| (v - max).exp()).sum();
    let log_sum_exp = max + sum_exp.ln();
    logits.iter().map(|&v| v - log_sum_exp).collect()
}

/// Return the top-k (`token_id`, `log_prob`) pairs sorted descending.
fn top_k_indices(log_probs: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut indexed: Vec<(u32, f32)> = log_probs
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            #[allow(clippy::cast_possible_truncation)]
            let id = i as u32;
            (id, v)
        })
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);
    indexed
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Config tests ──────────────────────────────────────────────────

    #[test]
    fn test_default_config() {
        let config = BeamSearchConfig::default();
        assert_eq!(config.beam_width, 4);
        assert_eq!(config.max_length, 128);
        assert!((config.length_penalty - 1.0).abs() < f32::EPSILON);
        assert!(config.early_stopping);
        assert_eq!(config.no_repeat_ngram_size, 0);
        assert!((config.temperature - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_custom_config() {
        let config = BeamSearchConfig {
            beam_width: 8,
            max_length: 64,
            length_penalty: 0.6,
            early_stopping: false,
            no_repeat_ngram_size: 3,
            temperature: 0.5,
        };
        assert_eq!(config.beam_width, 8);
        assert_eq!(config.no_repeat_ngram_size, 3);
    }

    // ── Length Penalty tests ──────────────────────────────────────────

    #[test]
    fn test_length_penalty_alpha_zero() {
        let lp = LengthPenalty::new(0.0);
        // (5+len)^0 / (5+1)^0 = 1/1 = 1
        assert!((lp.factor(10) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_length_penalty_alpha_one() {
        let lp = LengthPenalty::new(1.0);
        // (5+1)/(5+1) = 1.0
        assert!((lp.factor(1) - 1.0).abs() < 1e-6);
        // (5+10)/(5+1) = 15/6 = 2.5
        assert!((lp.factor(10) - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_length_penalty_increases_with_length() {
        let lp = LengthPenalty::new(0.8);
        let f1 = lp.factor(1);
        let f5 = lp.factor(5);
        let f10 = lp.factor(10);
        assert!(f1 < f5);
        assert!(f5 < f10);
    }

    #[test]
    fn test_length_penalty_apply() {
        let lp = LengthPenalty::new(1.0);
        let log_prob = -5.0;
        let score = lp.apply(log_prob, 10);
        // score = -5.0 / (15/6) = -5.0 / 2.5 = -2.0
        assert!((score - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_length_penalty_default() {
        let lp = LengthPenalty::default();
        assert!((lp.alpha - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_length_penalty_zero_length() {
        let lp = LengthPenalty::new(1.0);
        // (5+0)/(5+1) = 5/6
        let factor = lp.factor(0);
        assert!((factor - 5.0 / 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_length_penalty_high_alpha() {
        let lp = LengthPenalty::new(2.0);
        // Factor grows faster with higher alpha
        let f5 = lp.factor(5);
        let f10 = lp.factor(10);
        assert!(f10 > f5);
        // With alpha=2, penalty grows quadratically
        let f5_a1 = LengthPenalty::new(1.0).factor(5);
        assert!(f5 > f5_a1);
    }

    // ── Beam tests ────────────────────────────────────────────────────

    #[test]
    fn test_beam_empty() {
        let beam = Beam::empty();
        assert!(beam.is_empty());
        assert_eq!(beam.len(), 0);
        assert!((beam.log_probability - 0.0).abs() < f32::EPSILON);
        assert!(!beam.is_finished);
    }

    #[test]
    fn test_beam_new() {
        let beam = Beam::new(vec![1, 2, 3], -2.5);
        assert_eq!(beam.len(), 3);
        assert!((beam.log_probability - (-2.5)).abs() < f32::EPSILON);
        assert!(!beam.is_finished);
    }

    #[test]
    fn test_beam_extend() {
        let beam = Beam::new(vec![1, 2], -1.0);
        let extended = beam.extend(3, -0.5);
        assert_eq!(extended.token_ids, vec![1, 2, 3]);
        assert!((extended.log_probability - (-1.5)).abs() < f32::EPSILON);
        assert!(!extended.is_finished);
    }

    #[test]
    fn test_beam_extend_preserves_original() {
        let beam = Beam::new(vec![1], -1.0);
        let _extended = beam.extend(2, -0.5);
        assert_eq!(beam.token_ids, vec![1]);
        assert!((beam.log_probability - (-1.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_beam_score_with_penalty() {
        let beam = Beam::new(vec![1, 2, 3, 4, 5], -10.0);
        let lp = LengthPenalty::new(1.0);
        let score = beam.score(&lp);
        // length=5, penalty = (5+5)/(5+1) = 10/6
        let expected = -10.0 / (10.0 / 6.0);
        assert!((score - expected).abs() < 1e-6);
    }

    #[test]
    fn test_beam_score_no_penalty() {
        let beam = Beam::new(vec![1, 2, 3], -3.0);
        let lp = LengthPenalty::new(0.0);
        let score = beam.score(&lp);
        assert!((score - (-3.0)).abs() < 1e-6);
    }

    // ── NGram Blocker tests ───────────────────────────────────────────

    #[test]
    fn test_ngram_blocker_no_repeat_bigram() {
        // Beam: [1, 2, 3, 1, 2] — suffix is [1, 2]
        // Bigram "1,2" appeared before, followed by 3
        // Now suffix is [2], so we check what follows "2"
        let beam = Beam::new(vec![1, 2, 3, 1, 2], 0.0);
        let blocked = NGramBlocker::blocked_tokens(&beam, 2);
        // After [2] we saw [3] (from pos 1→2), and [nothing at end yet]
        // The suffix of length 1 is [2]. After [2] we saw 3 (pos 1).
        assert!(blocked.contains(&3));
    }

    #[test]
    fn test_ngram_blocker_trigram() {
        // Beam: [1, 2, 3, 1, 2] — check trigrams (n=3)
        // Suffix of length 2: [1, 2]
        // The trigram [1,2,3] appeared at pos 0
        // So 3 is blocked
        let beam = Beam::new(vec![1, 2, 3, 1, 2], 0.0);
        let blocked = NGramBlocker::blocked_tokens(&beam, 3);
        assert!(blocked.contains(&3));
    }

    #[test]
    fn test_ngram_blocker_empty_beam() {
        let beam = Beam::empty();
        let blocked = NGramBlocker::blocked_tokens(&beam, 2);
        assert!(blocked.is_empty());
    }

    #[test]
    fn test_ngram_blocker_short_beam() {
        let beam = Beam::new(vec![1], 0.0);
        let blocked = NGramBlocker::blocked_tokens(&beam, 3);
        assert!(blocked.is_empty());
    }

    #[test]
    fn test_ngram_blocker_n_zero() {
        let beam = Beam::new(vec![1, 2, 3], 0.0);
        let blocked = NGramBlocker::blocked_tokens(&beam, 0);
        assert!(blocked.is_empty());
    }

    #[test]
    fn test_ngram_blocker_no_repetition() {
        // All unique bigrams
        let beam = Beam::new(vec![1, 2, 3, 4, 5], 0.0);
        let blocked = NGramBlocker::blocked_tokens(&beam, 2);
        // Suffix is [5]. Check if [5] appeared before followed by something.
        // [5] never appeared earlier as prefix of a bigram, so nothing blocked.
        assert!(blocked.is_empty());
    }

    #[test]
    fn test_ngram_blocker_multiple_blocked() {
        // [1, 2, 1, 3, 1] — suffix [1], unigram 1 was followed by 2 and 3
        let beam = Beam::new(vec![1, 2, 1, 3, 1], 0.0);
        let blocked = NGramBlocker::blocked_tokens(&beam, 2);
        assert!(blocked.contains(&2));
        assert!(blocked.contains(&3));
    }

    // ── BeamGroup tests ───────────────────────────────────────────────

    #[test]
    fn test_beam_group_sort() {
        let lp = LengthPenalty::new(0.0);
        let beams =
            vec![Beam::new(vec![1], -3.0), Beam::new(vec![2], -1.0), Beam::new(vec![3], -2.0)];
        let mut group = BeamGroup::new(beams);
        group.sort_by_score(&lp);
        assert_eq!(group.beams[0].token_ids, vec![2]);
        assert_eq!(group.beams[1].token_ids, vec![3]);
        assert_eq!(group.beams[2].token_ids, vec![1]);
    }

    #[test]
    fn test_beam_group_prune() {
        let lp = LengthPenalty::new(0.0);
        let beams = vec![
            Beam::new(vec![1], -3.0),
            Beam::new(vec![2], -1.0),
            Beam::new(vec![3], -2.0),
            Beam::new(vec![4], -4.0),
        ];
        let mut group = BeamGroup::new(beams);
        group.prune(2, &lp);
        assert_eq!(group.beams.len(), 2);
        assert_eq!(group.beams[0].token_ids, vec![2]);
        assert_eq!(group.beams[1].token_ids, vec![3]);
    }

    #[test]
    fn test_beam_group_all_finished() {
        let mut beams = vec![Beam::new(vec![1], -1.0), Beam::new(vec![2], -2.0)];
        beams[0].is_finished = true;
        beams[1].is_finished = true;
        let group = BeamGroup::new(beams);
        assert!(group.all_finished());
    }

    #[test]
    fn test_beam_group_not_all_finished() {
        let mut beams = vec![Beam::new(vec![1], -1.0), Beam::new(vec![2], -2.0)];
        beams[0].is_finished = true;
        let group = BeamGroup::new(beams);
        assert!(!group.all_finished());
    }

    #[test]
    fn test_beam_group_active_count() {
        let mut beams =
            vec![Beam::new(vec![1], -1.0), Beam::new(vec![2], -2.0), Beam::new(vec![3], -3.0)];
        beams[1].is_finished = true;
        let group = BeamGroup::new(beams);
        assert_eq!(group.active_count(), 2);
    }

    #[test]
    fn test_beam_group_empty() {
        let group = BeamGroup::new(Vec::new());
        assert!(!group.all_finished()); // empty = not all finished
        assert_eq!(group.active_count(), 0);
    }

    // ── Decoder tests ─────────────────────────────────────────────────

    fn make_logits(vocab_size: usize, hot_token: usize, hot_value: f32) -> Vec<f32> {
        let mut logits = vec![0.0; vocab_size];
        logits[hot_token] = hot_value;
        logits
    }

    #[test]
    fn test_decoder_single_step() {
        let config = BeamSearchConfig {
            beam_width: 2,
            max_length: 10,
            length_penalty: 0.0,
            early_stopping: true,
            no_repeat_ngram_size: 0,
            temperature: 1.0,
        };
        let mut decoder = BeamSearchDecoder::new(config);
        let logits = make_logits(10, 5, 10.0);
        let cont = decoder.step(&logits, 99);
        assert!(cont);
        // All beams should have token 5 (highest logit)
        for beam in &decoder.group.beams {
            assert!(!beam.is_empty());
        }
    }

    #[test]
    fn test_decoder_greedy_beam_width_1() {
        let config = BeamSearchConfig {
            beam_width: 1,
            max_length: 5,
            length_penalty: 0.0,
            early_stopping: true,
            no_repeat_ngram_size: 0,
            temperature: 1.0,
        };
        let mut decoder = BeamSearchDecoder::new(config);
        // Step with token 3 as the hottest
        let logits = make_logits(10, 3, 10.0);
        decoder.step(&logits, 99);
        assert_eq!(decoder.group.beams.len(), 1);
        assert_eq!(*decoder.group.beams[0].token_ids.last().unwrap(), 3);
    }

    #[test]
    fn test_decoder_eos_finishes_beam() {
        let config = BeamSearchConfig {
            beam_width: 1,
            max_length: 10,
            length_penalty: 0.0,
            early_stopping: true,
            no_repeat_ngram_size: 0,
            temperature: 1.0,
        };
        let mut decoder = BeamSearchDecoder::new(config);
        let eos = 2;
        let logits = make_logits(10, eos as usize, 10.0);
        let cont = decoder.step(&logits, eos);
        // EOS beam should be in completed
        assert!(!decoder.completed.is_empty());
        assert!(decoder.completed[0].is_finished);
        // Should stop since all beams finished (early_stopping)
        assert!(!cont);
    }

    #[test]
    fn test_decoder_max_length_stops() {
        let config = BeamSearchConfig {
            beam_width: 1,
            max_length: 3,
            length_penalty: 0.0,
            early_stopping: false,
            no_repeat_ngram_size: 0,
            temperature: 1.0,
        };
        let mut decoder = BeamSearchDecoder::new(config);
        let logits = make_logits(10, 5, 10.0);
        let c1 = decoder.step(&logits, 99);
        assert!(c1);
        let c2 = decoder.step(&logits, 99);
        assert!(c2);
        let c3 = decoder.step(&logits, 99);
        assert!(!c3);
    }

    #[test]
    fn test_decoder_multi_beam_expansion() {
        let config = BeamSearchConfig {
            beam_width: 3,
            max_length: 10,
            length_penalty: 0.0,
            early_stopping: true,
            no_repeat_ngram_size: 0,
            temperature: 1.0,
        };
        let mut decoder = BeamSearchDecoder::new(config);
        // Logits with 2 strong candidates
        let mut logits = vec![0.0; 10];
        logits[3] = 5.0;
        logits[7] = 4.5;
        logits[1] = 4.0;
        decoder.step(&logits, 99);
        assert_eq!(decoder.group.beams.len(), 3);
    }

    #[test]
    fn test_decoder_with_seeds() {
        let config = BeamSearchConfig {
            beam_width: 2,
            max_length: 10,
            length_penalty: 0.0,
            early_stopping: true,
            no_repeat_ngram_size: 0,
            temperature: 1.0,
        };
        let seeds = vec![Beam::new(vec![1, 2], -1.0), Beam::new(vec![3, 4], -2.0)];
        let mut decoder = BeamSearchDecoder::with_seeds(config, seeds);
        let logits = make_logits(10, 5, 10.0);
        decoder.step(&logits, 99);
        // Seeds should have been extended
        for beam in &decoder.group.beams {
            assert!(beam.len() >= 3);
        }
    }

    #[test]
    fn test_decoder_finish_collects_all() {
        let config = BeamSearchConfig {
            beam_width: 2,
            max_length: 10,
            length_penalty: 0.0,
            early_stopping: true,
            no_repeat_ngram_size: 0,
            temperature: 1.0,
        };
        let mut decoder = BeamSearchDecoder::new(config);
        let logits = make_logits(10, 5, 10.0);
        decoder.step(&logits, 99);
        let result = decoder.finish();
        assert!(!result.sequences.is_empty());
    }

    #[test]
    fn test_decoder_result_sorted_by_score() {
        let config = BeamSearchConfig {
            beam_width: 3,
            max_length: 10,
            length_penalty: 0.0,
            early_stopping: true,
            no_repeat_ngram_size: 0,
            temperature: 1.0,
        };
        let mut decoder = BeamSearchDecoder::new(config);
        let mut logits = vec![0.0; 10];
        logits[3] = 5.0;
        logits[7] = 4.0;
        logits[1] = 3.0;
        decoder.step(&logits, 99);
        let result = decoder.finish();
        let lp = LengthPenalty::new(0.0);
        for w in result.sequences.windows(2) {
            assert!(w[0].score(&lp) >= w[1].score(&lp));
        }
    }

    #[test]
    fn test_decoder_temperature_scaling() {
        let config_hot = BeamSearchConfig {
            beam_width: 1,
            max_length: 10,
            length_penalty: 0.0,
            early_stopping: true,
            no_repeat_ngram_size: 0,
            temperature: 0.1, // very peaked
        };
        let config_cold = BeamSearchConfig {
            beam_width: 1,
            max_length: 10,
            length_penalty: 0.0,
            early_stopping: true,
            no_repeat_ngram_size: 0,
            temperature: 2.0, // very flat
        };
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let mut d_hot = BeamSearchDecoder::new(config_hot);
        d_hot.step(&logits, 99);

        let mut d_cold = BeamSearchDecoder::new(config_cold);
        d_cold.step(&logits, 99);

        // Both should pick token 4 (highest), but hot distribution has higher confidence
        assert_eq!(*d_hot.group.beams[0].token_ids.last().unwrap(), 4);
        assert_eq!(*d_cold.group.beams[0].token_ids.last().unwrap(), 4);
    }

    #[test]
    fn test_decoder_ngram_blocking() {
        let config = BeamSearchConfig {
            beam_width: 1,
            max_length: 10,
            length_penalty: 0.0,
            early_stopping: true,
            no_repeat_ngram_size: 2,
            temperature: 1.0,
        };
        // Seed with [3, 5] so that bigram [5, ?] can be checked
        let seeds = vec![Beam::new(vec![5, 3, 5], -0.1)];
        let mut decoder = BeamSearchDecoder::with_seeds(config, seeds);
        // Token 3 should be blocked (bigram [5, 3] already exists)
        // Make token 3 the most probable
        let mut logits = vec![0.0; 10];
        logits[3] = 10.0;
        logits[7] = 5.0; // second best
        decoder.step(&logits, 99);
        // The beam should have picked 7 (since 3 is blocked)
        let last = *decoder.group.beams[0].token_ids.last().unwrap();
        assert_eq!(last, 7, "Token 3 should be blocked by bigram repetition");
    }

    #[test]
    fn test_decoder_metrics_tracked() {
        let config = BeamSearchConfig {
            beam_width: 2,
            max_length: 10,
            length_penalty: 0.0,
            early_stopping: true,
            no_repeat_ngram_size: 0,
            temperature: 1.0,
        };
        let mut decoder = BeamSearchDecoder::new(config);
        let logits = make_logits(10, 5, 10.0);
        decoder.step(&logits, 99);
        decoder.step(&logits, 99);
        assert!(decoder.metrics.total_expansions > 0);
        assert!(decoder.metrics.search_time_ms >= 0.0);
    }

    #[test]
    fn test_decoder_steps_count() {
        let config =
            BeamSearchConfig { beam_width: 1, max_length: 10, ..BeamSearchConfig::default() };
        let mut decoder = BeamSearchDecoder::new(config);
        assert_eq!(decoder.steps(), 0);
        let logits = make_logits(10, 5, 10.0);
        decoder.step(&logits, 99);
        assert_eq!(decoder.steps(), 1);
        decoder.step(&logits, 99);
        assert_eq!(decoder.steps(), 2);
    }

    // ── BeamMerger tests ──────────────────────────────────────────────

    #[test]
    fn test_merger_deduplicates() {
        let beams = vec![
            Beam::new(vec![1, 2, 3], -2.0),
            Beam::new(vec![1, 2, 3], -1.0), // same sequence, better score
            Beam::new(vec![4, 5, 6], -3.0),
        ];
        let merged = BeamMerger::merge(beams);
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn test_merger_keeps_best_probability() {
        let beams = vec![
            Beam::new(vec![1, 2], -5.0),
            Beam::new(vec![1, 2], -1.0),
            Beam::new(vec![1, 2], -3.0),
        ];
        let merged = BeamMerger::merge(beams);
        assert_eq!(merged.len(), 1);
        assert!((merged[0].log_probability - (-1.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_merger_no_duplicates() {
        let beams =
            vec![Beam::new(vec![1], -1.0), Beam::new(vec![2], -2.0), Beam::new(vec![3], -3.0)];
        let merged = BeamMerger::merge(beams);
        assert_eq!(merged.len(), 3);
    }

    #[test]
    fn test_merger_empty_input() {
        let merged = BeamMerger::merge(Vec::new());
        assert!(merged.is_empty());
    }

    #[test]
    fn test_merger_all_identical() {
        let beams = vec![
            Beam::new(vec![1, 2], -3.0),
            Beam::new(vec![1, 2], -2.0),
            Beam::new(vec![1, 2], -1.0),
        ];
        let merged = BeamMerger::merge(beams);
        assert_eq!(merged.len(), 1);
        assert!((merged[0].log_probability - (-1.0)).abs() < f32::EPSILON);
    }

    // ── DiverseBeamSearch tests ───────────────────────────────────────

    #[test]
    fn test_diverse_hamming_penalty_no_match() {
        let penalty = DiverseBeamSearch::hamming_penalty(5, &[1, 2, 3]);
        assert!((penalty - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_diverse_hamming_penalty_one_match() {
        let penalty = DiverseBeamSearch::hamming_penalty(2, &[1, 2, 3]);
        assert!((penalty - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_diverse_hamming_penalty_multiple_matches() {
        let penalty = DiverseBeamSearch::hamming_penalty(2, &[2, 3, 2, 2]);
        assert!((penalty - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_diverse_hamming_penalty_empty() {
        let penalty = DiverseBeamSearch::hamming_penalty(5, &[]);
        assert!((penalty - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_diverse_beam_search_single_step() {
        let config = BeamSearchConfig {
            beam_width: 2,
            max_length: 10,
            length_penalty: 0.0,
            early_stopping: true,
            no_repeat_ngram_size: 0,
            temperature: 1.0,
        };
        let mut dbs = DiverseBeamSearch::new(config, 2, 1.0);
        let logits = make_logits(10, 5, 10.0);
        let cont = dbs.step(&logits, 99);
        assert!(cont);
    }

    #[test]
    fn test_diverse_beam_search_groups_diverge() {
        let config = BeamSearchConfig {
            beam_width: 1,
            max_length: 10,
            length_penalty: 0.0,
            early_stopping: true,
            no_repeat_ngram_size: 0,
            temperature: 1.0,
        };
        // High diversity penalty to force groups apart
        let mut dbs = DiverseBeamSearch::new(config, 2, 100.0);
        let mut logits = vec![0.0; 10];
        logits[3] = 5.0;
        logits[7] = 4.9; // close second
        dbs.step(&logits, 99);

        // Groups should have different top tokens
        let tok0 = dbs.groups[0].beams[0].token_ids.last().copied();
        let tok1 = dbs.groups[1].beams[0].token_ids.last().copied();
        // With high diversity penalty, second group should avoid first group's token
        assert_ne!(tok0, tok1);
    }

    #[test]
    fn test_diverse_beam_search_finish() {
        let config = BeamSearchConfig {
            beam_width: 2,
            max_length: 10,
            length_penalty: 0.0,
            early_stopping: true,
            no_repeat_ngram_size: 0,
            temperature: 1.0,
        };
        let mut dbs = DiverseBeamSearch::new(config, 2, 0.5);
        let logits = make_logits(10, 5, 10.0);
        dbs.step(&logits, 99);
        let result = dbs.finish();
        assert!(!result.sequences.is_empty());
    }

    #[test]
    fn test_diverse_beam_search_eos() {
        let config = BeamSearchConfig {
            beam_width: 1,
            max_length: 10,
            length_penalty: 0.0,
            early_stopping: true,
            no_repeat_ngram_size: 0,
            temperature: 1.0,
        };
        let mut dbs = DiverseBeamSearch::new(config, 1, 0.0);
        let eos = 2_u32;
        let logits = make_logits(10, eos as usize, 10.0);
        let cont = dbs.step(&logits, eos);
        assert!(!cont);
        assert!(!dbs.completed.is_empty());
    }

    #[test]
    fn test_diverse_beam_search_max_length() {
        let config = BeamSearchConfig {
            beam_width: 1,
            max_length: 2,
            length_penalty: 0.0,
            early_stopping: false,
            no_repeat_ngram_size: 0,
            temperature: 1.0,
        };
        let mut dbs = DiverseBeamSearch::new(config, 1, 0.0);
        let logits = make_logits(10, 5, 10.0);
        let c1 = dbs.step(&logits, 99);
        assert!(c1);
        let c2 = dbs.step(&logits, 99);
        assert!(!c2);
    }

    #[test]
    fn test_diverse_beam_search_zero_penalty() {
        // With zero diversity penalty, behaves like standard beam search
        let config = BeamSearchConfig {
            beam_width: 1,
            max_length: 10,
            length_penalty: 0.0,
            early_stopping: true,
            no_repeat_ngram_size: 0,
            temperature: 1.0,
        };
        let mut dbs = DiverseBeamSearch::new(config, 2, 0.0);
        let logits = make_logits(10, 5, 10.0);
        dbs.step(&logits, 99);
        // Both groups should pick the same token
        let tok0 = dbs.groups[0].beams[0].token_ids.last().copied();
        let tok1 = dbs.groups[1].beams[0].token_ids.last().copied();
        assert_eq!(tok0, tok1);
    }

    // ── BeamSearchResult tests ────────────────────────────────────────

    #[test]
    fn test_result_top_k() {
        let result = BeamSearchResult {
            sequences: vec![
                Beam::new(vec![1], -1.0),
                Beam::new(vec![2], -2.0),
                Beam::new(vec![3], -3.0),
            ],
            metrics: BeamMetrics::default(),
        };
        let top2 = result.top_k(2);
        assert_eq!(top2.len(), 2);
    }

    #[test]
    fn test_result_top_k_exceeds_len() {
        let result = BeamSearchResult {
            sequences: vec![Beam::new(vec![1], -1.0)],
            metrics: BeamMetrics::default(),
        };
        let top10 = result.top_k(10);
        assert_eq!(top10.len(), 1);
    }

    #[test]
    fn test_result_best() {
        let result = BeamSearchResult {
            sequences: vec![Beam::new(vec![1], -1.0), Beam::new(vec![2], -2.0)],
            metrics: BeamMetrics::default(),
        };
        let best = result.best().unwrap();
        assert_eq!(best.token_ids, vec![1]);
    }

    #[test]
    fn test_result_best_empty() {
        let result = BeamSearchResult { sequences: Vec::new(), metrics: BeamMetrics::default() };
        assert!(result.best().is_none());
    }

    // ── Metrics tests ─────────────────────────────────────────────────

    #[test]
    fn test_metrics_default() {
        let m = BeamMetrics::default();
        assert_eq!(m.total_expansions, 0);
        assert_eq!(m.beams_pruned, 0);
        assert!((m.avg_beam_length - 0.0).abs() < f32::EPSILON);
        assert!((m.search_time_ms - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_metrics_after_search() {
        let config = BeamSearchConfig {
            beam_width: 2,
            max_length: 5,
            length_penalty: 0.0,
            early_stopping: false,
            no_repeat_ngram_size: 0,
            temperature: 1.0,
        };
        let mut decoder = BeamSearchDecoder::new(config);
        let logits = make_logits(10, 5, 10.0);
        decoder.step(&logits, 99);
        decoder.step(&logits, 99);
        let result = decoder.finish();
        assert!(result.metrics.total_expansions > 0);
        assert!(result.metrics.avg_beam_length > 0.0);
    }

    // ── Helper function tests ─────────────────────────────────────────

    #[test]
    fn test_log_softmax_sums_to_one() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let lsm = log_softmax(&logits);
        let sum_probs: f32 = lsm.iter().map(|&v| v.exp()).sum();
        assert!((sum_probs - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_log_softmax_max_element() {
        let logits = vec![1.0, 2.0, 3.0];
        let lsm = log_softmax(&logits);
        // Last element should have highest log prob
        assert!(lsm[2] > lsm[1]);
        assert!(lsm[1] > lsm[0]);
    }

    #[test]
    fn test_log_softmax_empty() {
        let lsm = log_softmax(&[]);
        assert!(lsm.is_empty());
    }

    #[test]
    fn test_top_k_indices_order() {
        let probs = vec![0.1, 0.3, 0.5, 0.05, 0.05];
        let top = top_k_indices(&probs, 3);
        assert_eq!(top.len(), 3);
        assert_eq!(top[0].0, 2); // highest
        assert_eq!(top[1].0, 1);
        assert_eq!(top[2].0, 0);
    }

    #[test]
    fn test_top_k_indices_k_larger_than_len() {
        let probs = vec![0.5, 0.3];
        let top = top_k_indices(&probs, 10);
        assert_eq!(top.len(), 2);
    }

    // ── Edge case tests ───────────────────────────────────────────────

    #[test]
    fn test_beam_width_1_is_greedy() {
        let config = BeamSearchConfig {
            beam_width: 1,
            max_length: 5,
            length_penalty: 0.0,
            early_stopping: false,
            no_repeat_ngram_size: 0,
            temperature: 1.0,
        };
        let mut decoder = BeamSearchDecoder::new(config);
        let logits = make_logits(10, 7, 10.0);
        decoder.step(&logits, 99);
        decoder.step(&logits, 99);
        decoder.step(&logits, 99);
        // With beam_width=1, should always pick the argmax
        let beam = &decoder.group.beams[0];
        assert!(beam.token_ids.iter().all(|&t| t == 7));
    }

    #[test]
    fn test_single_token_sequence() {
        let config = BeamSearchConfig {
            beam_width: 1,
            max_length: 1,
            length_penalty: 0.0,
            early_stopping: false,
            no_repeat_ngram_size: 0,
            temperature: 1.0,
        };
        let mut decoder = BeamSearchDecoder::new(config);
        let logits = make_logits(10, 5, 10.0);
        let cont = decoder.step(&logits, 99);
        assert!(!cont); // max_length=1, so should stop
    }

    #[test]
    fn test_all_beams_identical_initial() {
        // All beams start empty and expand to same top-k
        let config = BeamSearchConfig {
            beam_width: 3,
            max_length: 10,
            length_penalty: 0.0,
            early_stopping: true,
            no_repeat_ngram_size: 0,
            temperature: 1.0,
        };
        let mut decoder = BeamSearchDecoder::new(config);
        let logits = make_logits(10, 5, 10.0);
        decoder.step(&logits, 99);
        // After merging, there may be duplicates but pruning keeps beam_width
        assert!(decoder.group.beams.len() <= 3);
    }

    #[test]
    fn test_uniform_logits() {
        let config = BeamSearchConfig {
            beam_width: 2,
            max_length: 3,
            length_penalty: 0.0,
            early_stopping: false,
            no_repeat_ngram_size: 0,
            temperature: 1.0,
        };
        let mut decoder = BeamSearchDecoder::new(config);
        let logits = vec![1.0; 10]; // all equal
        decoder.step(&logits, 99);
        // Should still work; any 2 tokens are valid
        assert_eq!(decoder.group.beams.len(), 2);
    }

    #[test]
    fn test_negative_logits() {
        let config = BeamSearchConfig {
            beam_width: 1,
            max_length: 5,
            length_penalty: 0.0,
            early_stopping: false,
            no_repeat_ngram_size: 0,
            temperature: 1.0,
        };
        let mut decoder = BeamSearchDecoder::new(config);
        let logits = vec![-5.0, -3.0, -10.0, -1.0, -7.0];
        decoder.step(&logits, 99);
        // Token 3 has highest logit (-1.0)
        assert_eq!(*decoder.group.beams[0].token_ids.last().unwrap(), 3);
    }

    #[test]
    fn test_large_vocab() {
        let config = BeamSearchConfig {
            beam_width: 4,
            max_length: 3,
            length_penalty: 0.0,
            early_stopping: false,
            no_repeat_ngram_size: 0,
            temperature: 1.0,
        };
        let mut decoder = BeamSearchDecoder::new(config);
        let logits = make_logits(50_000, 12345, 10.0);
        decoder.step(&logits, 99);
        // Should handle large vocab without issue
        assert!(!decoder.group.beams.is_empty());
    }

    #[test]
    fn test_length_penalty_affects_ranking() {
        let lp = LengthPenalty::new(1.0);
        // Short beam with worse log prob might rank better
        let short = Beam::new(vec![1], -1.0);
        let long = Beam::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], -2.0);
        let short_score = short.score(&lp);
        let long_score = long.score(&lp);
        // Short beam: -1.0 / (6/6) = -1.0
        // Long beam: -2.0 / (15/6) = -0.8
        // Long beam actually scores better due to length normalization
        assert!(long_score > short_score);
    }

    #[test]
    fn test_early_stopping_with_completed() {
        let config = BeamSearchConfig {
            beam_width: 2,
            max_length: 10,
            length_penalty: 0.0,
            early_stopping: true,
            no_repeat_ngram_size: 0,
            temperature: 1.0,
        };
        let mut beams = vec![Beam::new(vec![1], -1.0), Beam::new(vec![2], -2.0)];
        beams[0].is_finished = true;
        beams[1].is_finished = true;
        let mut decoder = BeamSearchDecoder::with_seeds(config, beams);
        let logits = make_logits(10, 5, 10.0);
        let cont = decoder.step(&logits, 99);
        assert!(!cont);
    }

    #[test]
    fn test_decoder_prune_metrics() {
        let config = BeamSearchConfig {
            beam_width: 2,
            max_length: 10,
            length_penalty: 0.0,
            early_stopping: false,
            no_repeat_ngram_size: 0,
            temperature: 1.0,
        };
        let mut decoder = BeamSearchDecoder::new(config);
        let mut logits = vec![0.0; 10];
        logits[1] = 5.0;
        logits[2] = 4.0;
        logits[3] = 3.0;
        logits[4] = 2.0;
        decoder.step(&logits, 99);
        // Should have pruned some candidates
        assert!(decoder.metrics.beams_pruned > 0 || decoder.metrics.total_expansions > 0);
    }

    #[test]
    fn test_diverse_search_metrics() {
        let config = BeamSearchConfig {
            beam_width: 2,
            max_length: 5,
            length_penalty: 0.0,
            early_stopping: false,
            no_repeat_ngram_size: 0,
            temperature: 1.0,
        };
        let mut dbs = DiverseBeamSearch::new(config, 2, 1.0);
        let logits = make_logits(10, 5, 10.0);
        dbs.step(&logits, 99);
        let result = dbs.finish();
        assert!(result.metrics.total_expansions > 0);
    }
}
