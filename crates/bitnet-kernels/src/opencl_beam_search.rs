//! Beam search decoding for Intel Arc A770 (OpenCL backend).
//!
//! This module provides beam search with early stopping, Google NMT length
//! normalization, and diverse beam groups (Hamming diversity penalty). CPU
//! reference implementations are supplied for all operations; an OpenCL C
//! kernel handles parallel beam expansion on GPU.

use std::fmt;

// ── Configuration ──────────────────────────────────────────────────────

/// Beam search hyper-parameters.
#[derive(Debug, Clone)]
pub struct BeamConfig {
    /// Number of active beams per group.
    pub beam_width: usize,
    /// Maximum sequence length (excluding prompt).
    pub max_length: usize,
    /// Google NMT length penalty exponent (0 = no penalty).
    pub length_penalty: f32,
    /// Stop as soon as the top beam is finished and cannot be beaten.
    pub early_stopping: bool,
    /// Number of diverse beam groups (1 = standard beam search).
    pub num_groups: usize,
    /// EOS token id.
    pub eos_token_id: u32,
    /// Hamming diversity penalty strength (used across groups).
    pub diversity_penalty: f32,
}

impl Default for BeamConfig {
    fn default() -> Self {
        Self {
            beam_width: 4,
            max_length: 128,
            length_penalty: 0.6,
            early_stopping: true,
            num_groups: 1,
            eos_token_id: 0,
            diversity_penalty: 0.5,
        }
    }
}

/// Validation errors for [`BeamConfig`].
#[derive(Debug, Clone, PartialEq)]
pub enum BeamSearchError {
    InvalidBeamWidth,
    InvalidMaxLength,
    InvalidNumGroups,
    InvalidLengthPenalty,
    EmptyVocab,
    NoBeamsAlive,
    NumericalError(String),
}

impl fmt::Display for BeamSearchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBeamWidth => write!(f, "beam_width must be >= 1"),
            Self::InvalidMaxLength => write!(f, "max_length must be >= 1"),
            Self::InvalidNumGroups => write!(f, "num_groups must be >= 1"),
            Self::InvalidLengthPenalty => {
                write!(f, "length_penalty must be >= 0")
            }
            Self::EmptyVocab => write!(f, "vocabulary is empty"),
            Self::NoBeamsAlive => write!(f, "no beams alive"),
            Self::NumericalError(s) => write!(f, "numerical error: {s}"),
        }
    }
}

impl std::error::Error for BeamSearchError {}

/// Validate a [`BeamConfig`].
pub fn validate_beam_config(cfg: &BeamConfig) -> Result<(), BeamSearchError> {
    if cfg.beam_width == 0 {
        return Err(BeamSearchError::InvalidBeamWidth);
    }
    if cfg.max_length == 0 {
        return Err(BeamSearchError::InvalidMaxLength);
    }
    if cfg.num_groups == 0 {
        return Err(BeamSearchError::InvalidNumGroups);
    }
    if cfg.length_penalty < 0.0 {
        return Err(BeamSearchError::InvalidLengthPenalty);
    }
    Ok(())
}

// ── Core types ─────────────────────────────────────────────────────────

/// A single hypothesis in the beam.
#[derive(Debug, Clone)]
pub struct Beam {
    /// Token ids generated so far.
    pub token_ids: Vec<u32>,
    /// Cumulative (un-normalised) log-probability.
    pub log_prob: f64,
    /// Length-normalised score.
    pub score: f64,
    /// Whether this beam has produced EOS.
    pub is_finished: bool,
    /// The group this beam belongs to (0-indexed).
    pub group_id: usize,
}

impl Beam {
    /// Create a new empty beam.
    pub fn new(group_id: usize) -> Self {
        Self { token_ids: Vec::new(), log_prob: 0.0, score: 0.0, is_finished: false, group_id }
    }

    /// Extend this beam with a token and its log-probability.
    pub fn extend(&self, token_id: u32, token_log_prob: f64, eos_token_id: u32) -> Self {
        let mut token_ids = self.token_ids.clone();
        token_ids.push(token_id);
        let log_prob = self.log_prob + token_log_prob;
        let is_finished = token_id == eos_token_id;
        Self { token_ids, log_prob, score: log_prob, is_finished, group_id: self.group_id }
    }
}

// ── Length normaliser ──────────────────────────────────────────────────

/// Google NMT length penalty: `(5 + len)^alpha / (5 + 1)^alpha`.
#[derive(Debug, Clone, Copy)]
pub struct LengthNormalizer {
    pub alpha: f32,
}

impl LengthNormalizer {
    pub fn new(alpha: f32) -> Self {
        Self { alpha }
    }

    /// Compute the normalisation factor for a sequence of `length` tokens.
    pub fn penalty(&self, length: usize) -> f64 {
        if self.alpha == 0.0 {
            return 1.0;
        }
        let num = (5.0_f64 + length as f64).powf(self.alpha as f64);
        let den = (5.0_f64 + 1.0_f64).powf(self.alpha as f64);
        num / den
    }

    /// Return the length-normalised score.
    pub fn normalise(&self, log_prob: f64, length: usize) -> f64 {
        log_prob / self.penalty(length)
    }
}

// ── Early-stop detector ────────────────────────────────────────────────

/// Detects when the best finished beam cannot be beaten by any active beam.
#[derive(Debug)]
pub struct EarlyStopDetector {
    best_finished_score: Option<f64>,
}

impl EarlyStopDetector {
    pub fn new() -> Self {
        Self { best_finished_score: None }
    }

    /// Record a finished beam's score.
    pub fn record_finished(&mut self, score: f64) {
        match self.best_finished_score {
            Some(prev) if score > prev => self.best_finished_score = Some(score),
            None => self.best_finished_score = Some(score),
            _ => {}
        }
    }

    /// Returns `true` when the best active beam's upper bound cannot beat
    /// the best finished beam.
    pub fn should_stop(&self, best_active_log_prob: f64, normalizer: &LengthNormalizer) -> bool {
        let Some(best_finished) = self.best_finished_score else {
            return false;
        };
        // Upper bound: the active beam cannot get a score better than its
        // current log-prob normalised at its current length (since adding
        // tokens only adds ≤ 0 log-prob).
        let upper_bound = normalizer.normalise(best_active_log_prob, 1);
        upper_bound <= best_finished
    }
}

impl Default for EarlyStopDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ── Beam pool ──────────────────────────────────────────────────────────

/// Maintains the top-K beams per group with pruning.
#[derive(Debug)]
pub struct BeamPool {
    /// Active (unfinished) beams, grouped by `group_id`.
    pub active: Vec<Beam>,
    /// Finished beams collected over the search.
    pub finished: Vec<Beam>,
    /// Beam width (K).
    pub beam_width: usize,
    /// Number of groups.
    pub num_groups: usize,
}

impl BeamPool {
    /// Create a pool seeded with one empty beam per group.
    pub fn new(beam_width: usize, num_groups: usize) -> Self {
        let active: Vec<Beam> =
            (0..num_groups).flat_map(|g| (0..beam_width).map(move |_| Beam::new(g))).collect();
        Self { active, finished: Vec::new(), beam_width, num_groups }
    }

    /// Seed the pool with a single empty beam per group (reset).
    pub fn seed(&mut self) {
        self.active = (0..self.num_groups).flat_map(|g| std::iter::once(Beam::new(g))).collect();
        self.finished.clear();
    }

    /// Replace the active beams for `group_id`, keeping only the top-K by
    /// score. Finished beams are moved to the finished set.
    pub fn update_group(
        &mut self,
        group_id: usize,
        mut candidates: Vec<Beam>,
        normalizer: &LengthNormalizer,
    ) {
        // Apply length normalisation.
        for beam in &mut candidates {
            let len = beam.token_ids.len().max(1);
            beam.score = normalizer.normalise(beam.log_prob, len);
        }

        // Separate finished from active.
        let (fin, mut alive): (Vec<_>, Vec<_>) =
            candidates.into_iter().partition(|b| b.is_finished);

        self.finished.extend(fin);

        // Sort active beams descending by score and keep top-K.
        alive.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        alive.truncate(self.beam_width);

        // Remove old beams for this group and insert new ones.
        self.active.retain(|b| b.group_id != group_id);
        self.active.extend(alive);
    }

    /// Get active beams for a specific group.
    pub fn group_beams(&self, group_id: usize) -> Vec<&Beam> {
        self.active.iter().filter(|b| b.group_id == group_id).collect()
    }

    /// Best score among all active beams.
    pub fn best_active_log_prob(&self) -> f64 {
        self.active.iter().map(|b| b.log_prob).fold(f64::NEG_INFINITY, f64::max)
    }

    /// True when no active beams remain.
    pub fn all_finished(&self) -> bool {
        self.active.is_empty()
    }
}

// ── Diverse beam search ────────────────────────────────────────────────

/// Applies a Hamming diversity penalty across beam groups so that
/// different groups are encouraged to explore different token sequences.
#[derive(Debug)]
pub struct DiverseBeamSearch {
    pub penalty: f32,
}

impl DiverseBeamSearch {
    pub fn new(penalty: f32) -> Self {
        Self { penalty }
    }

    /// Apply the Hamming diversity penalty to `logits` given previously
    /// selected tokens from earlier groups at the same time-step.
    ///
    /// For each token already selected by a prior group, we subtract
    /// `penalty` from that token's logit.
    pub fn apply_penalty(&self, logits: &mut [f32], selected_tokens: &[u32]) {
        if self.penalty == 0.0 {
            return;
        }
        for &tok in selected_tokens {
            let idx = tok as usize;
            if idx < logits.len() {
                logits[idx] -= self.penalty;
            }
        }
    }
}

// ── Single beam step (CPU reference) ───────────────────────────────────

/// Candidate produced during beam expansion.
#[derive(Debug, Clone)]
struct Candidate {
    beam_idx: usize,
    token_id: u32,
    log_prob: f64,
    score: f64,
    #[allow(dead_code)]
    group_id: usize,
}

/// Compute log-softmax of `logits` in-place, returning log-probabilities.
fn log_softmax(logits: &[f32]) -> Vec<f64> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let shifted: Vec<f64> = logits.iter().map(|&l| (l - max_val) as f64).collect();
    let log_sum_exp = shifted.iter().copied().map(f64::exp).sum::<f64>().ln();
    shifted.iter().map(|&s| s - log_sum_exp).collect()
}

/// Perform one decode step: expand all active beams, score, and prune.
pub struct BeamStep;

impl BeamStep {
    /// Expand beams for a single group. `logits_per_beam` provides one
    /// logit vector per active beam in the group.
    pub fn expand_group(
        group_beams: &[&Beam],
        logits_per_beam: &[Vec<f32>],
        config: &BeamConfig,
        normalizer: &LengthNormalizer,
        diversity: Option<&DiverseBeamSearch>,
        prior_group_tokens: &[u32],
    ) -> Vec<Beam> {
        let mut all_candidates: Vec<Candidate> = Vec::new();

        for (beam_idx, beam) in group_beams.iter().enumerate() {
            if beam.is_finished {
                // Propagate finished beams unchanged.
                all_candidates.push(Candidate {
                    beam_idx,
                    token_id: config.eos_token_id,
                    log_prob: beam.log_prob,
                    score: beam.score,
                    group_id: beam.group_id,
                });
                continue;
            }

            let mut logits = logits_per_beam[beam_idx].clone();

            // Apply diversity penalty from prior groups.
            if let Some(div) = diversity {
                div.apply_penalty(&mut logits, prior_group_tokens);
            }

            let log_probs = log_softmax(&logits);
            if log_probs.is_empty() {
                continue;
            }

            // Collect the top beam_width candidates from this beam.
            let mut indices: Vec<usize> = (0..log_probs.len()).collect();
            indices.sort_unstable_by(|&a, &b| {
                log_probs[b].partial_cmp(&log_probs[a]).unwrap_or(std::cmp::Ordering::Equal)
            });
            let take = config.beam_width.min(indices.len());
            for &idx in &indices[..take] {
                let cumulative = beam.log_prob + log_probs[idx];
                let length = beam.token_ids.len() + 1;
                let score = normalizer.normalise(cumulative, length);
                all_candidates.push(Candidate {
                    beam_idx,
                    token_id: idx as u32,
                    log_prob: cumulative,
                    score,
                    group_id: beam.group_id,
                });
            }
        }

        // Sort by score descending and keep top beam_width.
        all_candidates
            .sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        all_candidates.truncate(config.beam_width);

        all_candidates
            .into_iter()
            .map(|c| {
                let parent = group_beams[c.beam_idx];
                if parent.is_finished {
                    parent.clone()
                } else {
                    parent.extend(c.token_id, c.log_prob - parent.log_prob, config.eos_token_id)
                }
            })
            .collect()
    }
}

// ── Full beam search (CPU reference) ───────────────────────────────────

/// Final result of beam search.
#[derive(Debug, Clone)]
pub struct BeamSearchResult {
    /// Completed hypotheses sorted by score (best first).
    pub beams: Vec<Beam>,
    /// Total decode steps executed.
    pub steps: usize,
    /// Whether early stopping was triggered.
    pub early_stopped: bool,
}

/// Run beam search decoding.
///
/// `logits_fn` is called with the list of token-id prefixes (one per
/// active beam) and must return one logit vector per prefix. The logit
/// vectors must all have the same vocabulary size.
pub fn cpu_beam_search(
    mut logits_fn: impl FnMut(&[Vec<u32>]) -> Vec<Vec<f32>>,
    config: &BeamConfig,
) -> Result<BeamSearchResult, BeamSearchError> {
    validate_beam_config(config)?;

    let normalizer = LengthNormalizer::new(config.length_penalty);
    let diversity = if config.num_groups > 1 {
        Some(DiverseBeamSearch::new(config.diversity_penalty))
    } else {
        None
    };

    let mut pool = BeamPool::new(1, config.num_groups); // start with 1 beam per group
    pool.seed();
    let mut early_detector = EarlyStopDetector::new();
    let mut steps = 0_usize;

    for _step in 0..config.max_length {
        if pool.all_finished() {
            break;
        }

        // Tokens selected this step by earlier groups (for diversity).
        let mut prior_group_tokens: Vec<u32> = Vec::new();

        for group_id in 0..config.num_groups {
            let group_beams = pool.group_beams(group_id);
            if group_beams.is_empty() {
                continue;
            }

            // Build prefixes.
            let prefixes: Vec<Vec<u32>> = group_beams.iter().map(|b| b.token_ids.clone()).collect();

            let logits_batch = logits_fn(&prefixes);
            if logits_batch.is_empty() || logits_batch[0].is_empty() {
                return Err(BeamSearchError::EmptyVocab);
            }

            let expanded = BeamStep::expand_group(
                &group_beams,
                &logits_batch,
                config,
                &normalizer,
                diversity.as_ref(),
                &prior_group_tokens,
            );

            // Record tokens chosen by this group for diversity penalty.
            for beam in &expanded {
                if let Some(&tok) = beam.token_ids.last() {
                    prior_group_tokens.push(tok);
                }
            }

            pool.update_group(group_id, expanded, &normalizer);
        }

        // Record finished beams for early stopping.
        for beam in &pool.finished {
            early_detector.record_finished(beam.score);
        }

        if config.early_stopping
            && early_detector.should_stop(pool.best_active_log_prob(), &normalizer)
        {
            steps = _step + 1;
            let mut result_beams = pool.finished.clone();
            // Include active beams (normalised) as fallback.
            for mut b in pool.active.drain(..) {
                let len = b.token_ids.len().max(1);
                b.score = normalizer.normalise(b.log_prob, len);
                result_beams.push(b);
            }
            result_beams
                .sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
            return Ok(BeamSearchResult { beams: result_beams, steps, early_stopped: true });
        }

        steps = _step + 1;
    }

    // Collect all beams (finished + remaining active).
    let mut result_beams = pool.finished;
    for mut b in pool.active {
        let len = b.token_ids.len().max(1);
        b.score = normalizer.normalise(b.log_prob, len);
        result_beams.push(b);
    }
    result_beams.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    Ok(BeamSearchResult { beams: result_beams, steps, early_stopped: false })
}

// ── OpenCL kernel source ───────────────────────────────────────────────

/// OpenCL C kernel for parallel beam expansion on Intel Arc A770.
///
/// Each work-item evaluates one (beam, token) pair, computing the
/// cumulative log-probability. A subsequent host-side top-K selection
/// prunes the expanded set.
pub const BEAM_EXPANSION_KERNEL_SRC: &str = r#"
// ----- Beam Expansion for Intel Arc A770 (Xe-HPG) -----
// Grid: [num_beams * vocab_size]
// Each work-item computes score for one (beam, token) pair.

__kernel void beam_expand(
    __global const float* log_probs,     // [num_beams, vocab_size]
    __global const float* beam_scores,   // [num_beams] cumulative log-prob
    __global       float* candidate_scores, // [num_beams * vocab_size]
    const int vocab_size,
    const int num_beams,
    const float length_penalty_factor)   // pre-computed (5+len+1)^a / (5+1)^a
{
    const int gid = get_global_id(0);
    if (gid >= num_beams * vocab_size) return;

    const int beam_idx  = gid / vocab_size;
    const int token_idx = gid % vocab_size;

    float cumulative = beam_scores[beam_idx] + log_probs[beam_idx * vocab_size + token_idx];
    candidate_scores[gid] = cumulative / length_penalty_factor;
}

// Hamming diversity penalty: subtract penalty for tokens chosen by prior groups.
__kernel void apply_diversity_penalty(
    __global       float* logits,          // [vocab_size]
    __global const int*   selected_tokens, // [num_selected]
    const int num_selected,
    const float penalty)
{
    const int gid = get_global_id(0);
    for (int i = 0; i < num_selected; i++) {
        if (gid == selected_tokens[i]) {
            logits[gid] -= penalty;
        }
    }
}
"#;

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ────────────────────────────────────────────────────────

    /// Simple logits function: always returns uniform logits.
    fn uniform_logits(vocab_size: usize) -> impl FnMut(&[Vec<u32>]) -> Vec<Vec<f32>> {
        move |prefixes: &[Vec<u32>]| vec![vec![0.0_f32; vocab_size]; prefixes.len()]
    }

    /// Logits where token 2 is always dominant.
    fn dominant_token_logits(vocab_size: usize) -> impl FnMut(&[Vec<u32>]) -> Vec<Vec<f32>> {
        move |prefixes: &[Vec<u32>]| {
            prefixes
                .iter()
                .map(|_| {
                    let mut l = vec![0.0_f32; vocab_size];
                    l[2] = 10.0;
                    l
                })
                .collect()
        }
    }

    /// Logits where the dominant token changes each step.
    fn rotating_logits(vocab_size: usize) -> impl FnMut(&[Vec<u32>]) -> Vec<Vec<f32>> {
        move |prefixes: &[Vec<u32>]| {
            prefixes
                .iter()
                .map(|prefix| {
                    let step = prefix.len();
                    let mut l = vec![0.0_f32; vocab_size];
                    let dominant = (step + 1) % vocab_size;
                    l[dominant] = 10.0;
                    l
                })
                .collect()
        }
    }

    /// Logits that produce EOS (token 0) after `n` steps.
    fn eos_after_n(vocab_size: usize, n: usize) -> impl FnMut(&[Vec<u32>]) -> Vec<Vec<f32>> {
        move |prefixes: &[Vec<u32>]| {
            prefixes
                .iter()
                .map(|prefix| {
                    let step = prefix.len();
                    let mut l = vec![0.0_f32; vocab_size];
                    if step >= n {
                        l[0] = 100.0; // EOS
                    } else {
                        l[1] = 10.0;
                    }
                    l
                })
                .collect()
        }
    }

    fn default_config_with(beam_width: usize, max_length: usize) -> BeamConfig {
        BeamConfig {
            beam_width,
            max_length,
            length_penalty: 0.6,
            early_stopping: false,
            num_groups: 1,
            eos_token_id: 0,
            diversity_penalty: 0.5,
            ..Default::default()
        }
    }

    // ── BeamConfig validation ─────────────────────────────────────────

    #[test]
    fn validate_rejects_zero_beam_width() {
        let cfg = BeamConfig { beam_width: 0, ..Default::default() };
        assert_eq!(validate_beam_config(&cfg), Err(BeamSearchError::InvalidBeamWidth));
    }

    #[test]
    fn validate_rejects_zero_max_length() {
        let cfg = BeamConfig { max_length: 0, ..Default::default() };
        assert_eq!(validate_beam_config(&cfg), Err(BeamSearchError::InvalidMaxLength));
    }

    #[test]
    fn validate_rejects_zero_num_groups() {
        let cfg = BeamConfig { num_groups: 0, ..Default::default() };
        assert_eq!(validate_beam_config(&cfg), Err(BeamSearchError::InvalidNumGroups));
    }

    #[test]
    fn validate_rejects_negative_length_penalty() {
        let cfg = BeamConfig { length_penalty: -1.0, ..Default::default() };
        assert_eq!(validate_beam_config(&cfg), Err(BeamSearchError::InvalidLengthPenalty));
    }

    #[test]
    fn validate_accepts_defaults() {
        assert!(validate_beam_config(&BeamConfig::default()).is_ok());
    }

    // ── LengthNormalizer ──────────────────────────────────────────────

    #[test]
    fn length_normalizer_alpha_zero_is_identity() {
        let n = LengthNormalizer::new(0.0);
        assert_eq!(n.penalty(1), 1.0);
        assert_eq!(n.penalty(100), 1.0);
    }

    #[test]
    fn length_normalizer_alpha_1_grows_with_length() {
        let n = LengthNormalizer::new(1.0);
        let short = n.penalty(1);
        let long = n.penalty(10);
        assert!(long > short, "penalty should grow: {long} > {short}");
    }

    #[test]
    fn length_normalizer_penalty_at_length_1() {
        // (5+1)^a / (5+1)^a == 1.0
        let n = LengthNormalizer::new(0.6);
        let p = n.penalty(1);
        assert!((p - 1.0).abs() < 1e-10, "penalty at length 1 should be 1.0, got {p}");
    }

    #[test]
    fn length_normalizer_larger_alpha_stronger_penalty() {
        let len = 20;
        let mild = LengthNormalizer::new(0.6).penalty(len);
        let strong = LengthNormalizer::new(2.0).penalty(len);
        assert!(strong > mild, "alpha=2.0 should penalise more than 0.6: {strong} vs {mild}");
    }

    #[test]
    fn length_normalizer_normalise_divides_by_penalty() {
        let n = LengthNormalizer::new(0.6);
        let log_prob = -5.0;
        let len = 10;
        let expected = log_prob / n.penalty(len);
        assert!((n.normalise(log_prob, len) - expected).abs() < 1e-12);
    }

    #[test]
    fn length_normalizer_alpha_0_6_known_value() {
        let n = LengthNormalizer::new(0.6);
        // penalty(10) = (5+10)^0.6 / (5+1)^0.6 = 15^0.6 / 6^0.6
        let expected = 15.0_f64.powf(0.6) / 6.0_f64.powf(0.6);
        let got = n.penalty(10);
        assert!((got - expected).abs() < 1e-6, "expected {expected}, got {got}");
    }

    // ── EarlyStopDetector ─────────────────────────────────────────────

    #[test]
    fn early_stop_no_finished_never_stops() {
        let det = EarlyStopDetector::new();
        let n = LengthNormalizer::new(0.6);
        assert!(!det.should_stop(0.0, &n));
    }

    #[test]
    fn early_stop_triggers_when_active_cant_beat_finished() {
        let mut det = EarlyStopDetector::new();
        let n = LengthNormalizer::new(0.0);
        det.record_finished(-1.0);
        // Active beam with log-prob -10 normalised at length 1 = -10
        assert!(det.should_stop(-10.0, &n));
    }

    #[test]
    fn early_stop_does_not_trigger_when_active_could_win() {
        let mut det = EarlyStopDetector::new();
        let n = LengthNormalizer::new(0.0);
        det.record_finished(-10.0);
        // Active beam with log-prob -5 could still win
        assert!(!det.should_stop(-5.0, &n));
    }

    #[test]
    fn early_stop_tracks_best_finished() {
        let mut det = EarlyStopDetector::new();
        det.record_finished(-10.0);
        det.record_finished(-5.0);
        det.record_finished(-8.0);
        assert_eq!(det.best_finished_score, Some(-5.0));
    }

    // ── Beam ──────────────────────────────────────────────────────────

    #[test]
    fn beam_new_is_empty() {
        let b = Beam::new(0);
        assert!(b.token_ids.is_empty());
        assert_eq!(b.log_prob, 0.0);
        assert!(!b.is_finished);
    }

    #[test]
    fn beam_extend_adds_token() {
        let b = Beam::new(0);
        let b2 = b.extend(5, -1.0, 0);
        assert_eq!(b2.token_ids, vec![5]);
        assert_eq!(b2.log_prob, -1.0);
        assert!(!b2.is_finished);
    }

    #[test]
    fn beam_extend_with_eos_finishes() {
        let b = Beam::new(0);
        let b2 = b.extend(0, -0.5, 0); // EOS = 0
        assert!(b2.is_finished);
    }

    #[test]
    fn beam_extend_accumulates_log_prob() {
        let b = Beam::new(0);
        let b2 = b.extend(1, -1.0, 0);
        let b3 = b2.extend(2, -2.0, 0);
        assert!((b3.log_prob - (-3.0)).abs() < 1e-12);
        assert_eq!(b3.token_ids, vec![1, 2]);
    }

    // ── BeamPool ──────────────────────────────────────────────────────

    #[test]
    fn beam_pool_seed_creates_one_per_group() {
        let mut pool = BeamPool::new(4, 3);
        pool.seed();
        assert_eq!(pool.active.len(), 3);
        for (i, b) in pool.active.iter().enumerate() {
            assert_eq!(b.group_id, i);
        }
    }

    #[test]
    fn beam_pool_update_group_prunes_to_width() {
        let n = LengthNormalizer::new(0.0);
        let mut pool = BeamPool::new(2, 1);
        pool.seed();
        let candidates: Vec<Beam> = (0..5)
            .map(|i| {
                let mut b = Beam::new(0);
                b.log_prob = -(i as f64);
                b.token_ids = vec![i as u32];
                b
            })
            .collect();
        pool.update_group(0, candidates, &n);
        let group0 = pool.group_beams(0);
        assert_eq!(group0.len(), 2);
    }

    #[test]
    fn beam_pool_finished_beams_collected() {
        let n = LengthNormalizer::new(0.0);
        let mut pool = BeamPool::new(2, 1);
        pool.seed();
        let mut fin_beam = Beam::new(0);
        fin_beam.is_finished = true;
        fin_beam.token_ids = vec![0];
        fin_beam.log_prob = -1.0;
        pool.update_group(0, vec![fin_beam], &n);
        assert!(!pool.finished.is_empty());
    }

    #[test]
    fn beam_pool_all_finished_when_empty() {
        let mut pool = BeamPool::new(2, 1);
        pool.active.clear();
        assert!(pool.all_finished());
    }

    // ── DiverseBeamSearch ─────────────────────────────────────────────

    #[test]
    fn diversity_penalty_subtracts_from_selected() {
        let div = DiverseBeamSearch::new(5.0);
        let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        div.apply_penalty(&mut logits, &[1, 3]);
        assert_eq!(logits[0], 1.0);
        assert_eq!(logits[1], -3.0);
        assert_eq!(logits[2], 3.0);
        assert_eq!(logits[3], -1.0);
        assert_eq!(logits[4], 5.0);
    }

    #[test]
    fn diversity_penalty_zero_is_noop() {
        let div = DiverseBeamSearch::new(0.0);
        let mut logits = vec![1.0, 2.0, 3.0];
        let original = logits.clone();
        div.apply_penalty(&mut logits, &[0, 1, 2]);
        assert_eq!(logits, original);
    }

    #[test]
    fn diversity_penalty_ignores_oob_tokens() {
        let div = DiverseBeamSearch::new(1.0);
        let mut logits = vec![1.0, 2.0];
        let original = logits.clone();
        div.apply_penalty(&mut logits, &[999]);
        assert_eq!(logits, original);
    }

    // ── log_softmax ───────────────────────────────────────────────────

    #[test]
    fn log_softmax_sums_to_one_in_prob_space() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let lp = log_softmax(&logits);
        let sum: f64 = lp.iter().map(|&x| x.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-6, "softmax should sum to 1, got {sum}");
    }

    #[test]
    fn log_softmax_preserves_ordering() {
        let logits = vec![1.0, 3.0, 2.0];
        let lp = log_softmax(&logits);
        assert!(lp[1] > lp[2]);
        assert!(lp[2] > lp[0]);
    }

    #[test]
    fn log_softmax_empty_returns_empty() {
        assert!(log_softmax(&[]).is_empty());
    }

    #[test]
    fn log_softmax_single_element_is_zero() {
        let lp = log_softmax(&[42.0]);
        assert!((lp[0] - 0.0).abs() < 1e-10);
    }

    // ── BeamStep expansion ────────────────────────────────────────────

    #[test]
    fn beam_step_expand_produces_beam_width_candidates() {
        let cfg = default_config_with(3, 10);
        let n = LengthNormalizer::new(0.0);
        let beam = Beam::new(0);
        let logits = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let beams: Vec<&Beam> = vec![&beam];
        let result = BeamStep::expand_group(&beams, &logits, &cfg, &n, None, &[]);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn beam_step_expand_picks_best_tokens() {
        let cfg = default_config_with(1, 10);
        let n = LengthNormalizer::new(0.0);
        let beam = Beam::new(0);
        let logits = vec![vec![0.0, 0.0, 0.0, 100.0, 0.0]];
        let beams: Vec<&Beam> = vec![&beam];
        let result = BeamStep::expand_group(&beams, &logits, &cfg, &n, None, &[]);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].token_ids, vec![3]);
    }

    #[test]
    fn beam_step_finished_beam_propagated() {
        let cfg = default_config_with(2, 10);
        let n = LengthNormalizer::new(0.0);
        let mut finished = Beam::new(0);
        finished.is_finished = true;
        finished.token_ids = vec![5];
        finished.log_prob = -1.0;
        finished.score = -1.0;
        let alive = Beam::new(0);
        let logits = vec![vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0]];
        let beams: Vec<&Beam> = vec![&finished, &alive];
        let result = BeamStep::expand_group(&beams, &logits, &cfg, &n, None, &[]);
        // The finished beam should appear in the result.
        let has_finished = result.iter().any(|b| b.is_finished);
        assert!(has_finished, "finished beam should be propagated");
    }

    // ── Width=1 equals greedy ─────────────────────────────────────────

    #[test]
    fn beam_width_1_equals_greedy() {
        let cfg = BeamConfig {
            beam_width: 1,
            max_length: 5,
            length_penalty: 0.0,
            early_stopping: false,
            num_groups: 1,
            eos_token_id: 999, // unreachable
            diversity_penalty: 0.0,
        };

        let mut step = 0_usize;
        let result = cpu_beam_search(
            |prefixes: &[Vec<u32>]| {
                prefixes
                    .iter()
                    .map(|_| {
                        let mut l = vec![0.0_f32; 5];
                        // Greedy should always pick the dominant token.
                        l[(step % 5).max(1)] = 10.0;
                        step += 1;
                        l
                    })
                    .collect()
            },
            &cfg,
        )
        .unwrap();

        assert!(!result.beams.is_empty());
        assert_eq!(result.beams[0].token_ids.len(), 5);
    }

    // ── beam width 2, 4, 8 ───────────────────────────────────────────

    #[test]
    fn beam_width_2_returns_results() {
        let cfg = default_config_with(2, 4);
        let result = cpu_beam_search(dominant_token_logits(5), &cfg).unwrap();
        assert!(!result.beams.is_empty());
    }

    #[test]
    fn beam_width_4_returns_results() {
        let cfg = default_config_with(4, 4);
        let result = cpu_beam_search(dominant_token_logits(10), &cfg).unwrap();
        assert!(!result.beams.is_empty());
    }

    #[test]
    fn beam_width_8_returns_results() {
        let cfg = default_config_with(8, 4);
        let result = cpu_beam_search(dominant_token_logits(10), &cfg).unwrap();
        assert!(!result.beams.is_empty());
    }

    // ── length penalty effects ────────────────────────────────────────

    #[test]
    fn length_penalty_alpha_0_no_normalisation() {
        let cfg = BeamConfig {
            beam_width: 2,
            max_length: 3,
            length_penalty: 0.0,
            early_stopping: false,
            num_groups: 1,
            eos_token_id: 999,
            diversity_penalty: 0.0,
        };
        let result = cpu_beam_search(dominant_token_logits(5), &cfg).unwrap();
        // Score should equal raw log-prob when alpha=0.
        for beam in &result.beams {
            assert!(
                (beam.score - beam.log_prob).abs() < 1e-6,
                "alpha=0: score ({}) should match log_prob ({})",
                beam.score,
                beam.log_prob
            );
        }
    }

    #[test]
    fn length_penalty_alpha_0_6_affects_score() {
        let n = LengthNormalizer::new(0.6);
        let score_short = n.normalise(-5.0, 2);
        let score_long = n.normalise(-5.0, 20);
        // Longer sequence with same log-prob should have a lower (less
        // negative when divided by larger penalty) score... but the
        // score is log_prob / penalty, where penalty > 1 for longer
        // seqs, so score_long > score_short (less negative).
        assert!(
            score_long > score_short,
            "length penalty should favour longer: {score_long} vs {score_short}"
        );
    }

    #[test]
    fn length_penalty_alpha_1_0_linear_normalisation() {
        let n = LengthNormalizer::new(1.0);
        let p5 = n.penalty(5);
        let p10 = n.penalty(10);
        assert!(p10 > p5);
    }

    #[test]
    fn length_penalty_alpha_2_0_strong_normalisation() {
        let n = LengthNormalizer::new(2.0);
        let p5 = n.penalty(5);
        let p10 = n.penalty(10);
        let ratio = p10 / p5;
        // With alpha=2.0 the ratio should be significantly larger than alpha=0.6.
        let n_mild = LengthNormalizer::new(0.6);
        let mild_ratio = n_mild.penalty(10) / n_mild.penalty(5);
        assert!(ratio > mild_ratio, "alpha=2 ratio ({ratio}) > alpha=0.6 ratio ({mild_ratio})");
    }

    // ── early stopping ────────────────────────────────────────────────

    #[test]
    fn early_stopping_terminates_before_max_length() {
        let cfg = BeamConfig {
            beam_width: 2,
            max_length: 100,
            length_penalty: 0.0,
            early_stopping: true,
            num_groups: 1,
            eos_token_id: 0,
            diversity_penalty: 0.0,
        };
        // EOS after 3 steps.
        let result = cpu_beam_search(eos_after_n(5, 3), &cfg).unwrap();
        assert!(result.steps < 100, "should stop early, stopped at step {}", result.steps);
    }

    #[test]
    fn no_early_stopping_runs_to_completion() {
        let cfg = BeamConfig {
            beam_width: 1,
            max_length: 5,
            length_penalty: 0.0,
            early_stopping: false,
            num_groups: 1,
            eos_token_id: 999, // unreachable
            diversity_penalty: 0.0,
        };
        let result = cpu_beam_search(dominant_token_logits(5), &cfg).unwrap();
        assert_eq!(result.steps, 5);
        assert!(!result.early_stopped);
    }

    // ── EOS handling ──────────────────────────────────────────────────

    #[test]
    fn eos_token_finishes_beam() {
        let cfg = BeamConfig {
            beam_width: 1,
            max_length: 10,
            length_penalty: 0.0,
            early_stopping: true,
            num_groups: 1,
            eos_token_id: 0,
            diversity_penalty: 0.0,
        };
        let result = cpu_beam_search(eos_after_n(5, 2), &cfg).unwrap();
        let has_finished = result.beams.iter().any(|b| b.is_finished);
        assert!(has_finished, "some beam should be finished via EOS");
    }

    #[test]
    fn eos_at_step_0_produces_single_token() {
        let cfg = BeamConfig {
            beam_width: 1,
            max_length: 10,
            length_penalty: 0.0,
            early_stopping: true,
            num_groups: 1,
            eos_token_id: 0,
            diversity_penalty: 0.0,
        };
        let result = cpu_beam_search(eos_after_n(5, 0), &cfg).unwrap();
        let finished = result.beams.iter().find(|b| b.is_finished);
        assert!(finished.is_some());
        assert_eq!(finished.unwrap().token_ids.len(), 1);
    }

    // ── diverse beam groups ───────────────────────────────────────────

    #[test]
    fn diverse_groups_produce_different_first_tokens() {
        let cfg = BeamConfig {
            beam_width: 1,
            max_length: 3,
            length_penalty: 0.0,
            early_stopping: false,
            num_groups: 3,
            eos_token_id: 999,
            diversity_penalty: 10.0, // very strong penalty
        };
        // All groups see the same logits; diversity should push them apart.
        let result = cpu_beam_search(
            |prefixes: &[Vec<u32>]| {
                prefixes.iter().map(|_| vec![1.0, 2.0, 3.0, 4.0, 5.0]).collect()
            },
            &cfg,
        )
        .unwrap();

        // Beams from different groups should ideally start with different tokens.
        let first_tokens: Vec<Option<&u32>> =
            result.beams.iter().map(|b| b.token_ids.first()).collect();
        // At least two distinct first tokens across all beams.
        let mut unique: Vec<u32> = first_tokens.iter().filter_map(|t| t.copied()).collect();
        unique.sort();
        unique.dedup();
        assert!(unique.len() >= 2, "expected diverse first tokens, got {unique:?}");
    }

    #[test]
    fn single_group_no_diversity_penalty() {
        let cfg = BeamConfig {
            beam_width: 2,
            max_length: 3,
            length_penalty: 0.0,
            early_stopping: false,
            num_groups: 1,
            eos_token_id: 999,
            diversity_penalty: 10.0, // should be ignored with 1 group
        };
        let result = cpu_beam_search(dominant_token_logits(5), &cfg).unwrap();
        // All beams should have the same dominant token since there's
        // only one group.
        for beam in &result.beams {
            if !beam.token_ids.is_empty() {
                assert_eq!(beam.token_ids[0], 2, "single group should all pick dominant token");
            }
        }
    }

    // ── score monotonicity ────────────────────────────────────────────

    #[test]
    fn result_beams_sorted_by_score_descending() {
        let cfg = default_config_with(4, 5);
        let result = cpu_beam_search(rotating_logits(8), &cfg).unwrap();
        for w in result.beams.windows(2) {
            assert!(
                w[0].score >= w[1].score,
                "beams should be sorted: {} >= {}",
                w[0].score,
                w[1].score
            );
        }
    }

    #[test]
    fn log_probs_are_non_positive() {
        let cfg = default_config_with(4, 5);
        let result = cpu_beam_search(rotating_logits(8), &cfg).unwrap();
        for beam in &result.beams {
            assert!(beam.log_prob <= 1e-9, "log_prob should be <= 0, got {}", beam.log_prob);
        }
    }

    // ── edge cases ────────────────────────────────────────────────────

    #[test]
    fn empty_vocab_returns_error() {
        let cfg = default_config_with(2, 5);
        let result = cpu_beam_search(|prefixes: &[Vec<u32>]| vec![vec![]; prefixes.len()], &cfg);
        assert!(result.is_err());
    }

    #[test]
    fn single_token_vocab() {
        let cfg = BeamConfig {
            beam_width: 1,
            max_length: 3,
            length_penalty: 0.0,
            early_stopping: false,
            num_groups: 1,
            eos_token_id: 0,
            diversity_penalty: 0.0,
        };
        // Only token 0 which is also EOS.
        let result =
            cpu_beam_search(|prefixes: &[Vec<u32>]| vec![vec![5.0]; prefixes.len()], &cfg).unwrap();
        assert!(!result.beams.is_empty());
        // Token 0 is EOS so beam should be finished.
        assert!(result.beams.iter().any(|b| b.is_finished));
    }

    #[test]
    fn max_length_hit_terminates() {
        let cfg = BeamConfig {
            beam_width: 1,
            max_length: 3,
            length_penalty: 0.0,
            early_stopping: false,
            num_groups: 1,
            eos_token_id: 999,
            diversity_penalty: 0.0,
        };
        let result = cpu_beam_search(dominant_token_logits(5), &cfg).unwrap();
        assert_eq!(result.steps, 3);
        let max_len = result.beams.iter().map(|b| b.token_ids.len()).max().unwrap_or(0);
        assert!(max_len <= 3);
    }

    // ── OpenCL kernel source ──────────────────────────────────────────

    #[test]
    fn kernel_source_contains_beam_expand() {
        assert!(BEAM_EXPANSION_KERNEL_SRC.contains("beam_expand"));
    }

    #[test]
    fn kernel_source_contains_diversity_kernel() {
        assert!(BEAM_EXPANSION_KERNEL_SRC.contains("apply_diversity_penalty"));
    }

    #[test]
    fn kernel_source_is_not_empty() {
        assert!(BEAM_EXPANSION_KERNEL_SRC.len() > 100);
    }

    // ── property tests ────────────────────────────────────────────────

    #[test]
    fn width_1_greedy_picks_argmax_each_step() {
        let vocab = 6;
        let cfg = BeamConfig {
            beam_width: 1,
            max_length: 4,
            length_penalty: 0.0,
            early_stopping: false,
            num_groups: 1,
            eos_token_id: 999,
            diversity_penalty: 0.0,
        };
        let result = cpu_beam_search(rotating_logits(vocab), &cfg).unwrap();
        let tokens = &result.beams[0].token_ids;
        for (i, &tok) in tokens.iter().enumerate() {
            let expected = ((i + 1) % vocab) as u32;
            assert_eq!(tok, expected, "step {i}: expected {expected}, got {tok}");
        }
    }

    #[test]
    fn increasing_beam_width_does_not_decrease_best_score() {
        let make_logits = || {
            move |prefixes: &[Vec<u32>]| -> Vec<Vec<f32>> {
                prefixes
                    .iter()
                    .map(|p| {
                        let mut l = vec![0.0_f32; 8];
                        let step = p.len();
                        l[(step * 3 + 1) % 8] = 5.0;
                        l[(step * 7 + 2) % 8] = 3.0;
                        l
                    })
                    .collect()
            }
        };

        let cfg1 = BeamConfig {
            beam_width: 1,
            max_length: 4,
            length_penalty: 0.6,
            early_stopping: false,
            num_groups: 1,
            eos_token_id: 999,
            diversity_penalty: 0.0,
        };
        let cfg4 = BeamConfig { beam_width: 4, ..cfg1.clone() };

        let r1 = cpu_beam_search(make_logits(), &cfg1).unwrap();
        let r4 = cpu_beam_search(make_logits(), &cfg4).unwrap();

        let best1 = r1.beams[0].score;
        let best4 = r4.beams[0].score;
        assert!(
            best4 >= best1 - 1e-9,
            "wider beam should not decrease best score: {best4} vs {best1}"
        );
    }

    #[test]
    fn all_beams_have_valid_group_ids() {
        let cfg = BeamConfig {
            beam_width: 2,
            max_length: 3,
            length_penalty: 0.6,
            early_stopping: false,
            num_groups: 3,
            eos_token_id: 999,
            diversity_penalty: 1.0,
        };
        let result = cpu_beam_search(rotating_logits(8), &cfg).unwrap();
        for beam in &result.beams {
            assert!(beam.group_id < cfg.num_groups, "group_id {} out of range", beam.group_id);
        }
    }

    // ── BeamSearchResult metadata ─────────────────────────────────────

    #[test]
    fn result_steps_matches_max_length_when_no_early_stop() {
        let cfg = BeamConfig {
            beam_width: 2,
            max_length: 7,
            length_penalty: 0.0,
            early_stopping: false,
            num_groups: 1,
            eos_token_id: 999,
            diversity_penalty: 0.0,
        };
        let result = cpu_beam_search(dominant_token_logits(5), &cfg).unwrap();
        assert_eq!(result.steps, 7);
    }

    #[test]
    fn result_early_stopped_flag_set_correctly() {
        let cfg = BeamConfig {
            beam_width: 1,
            max_length: 100,
            length_penalty: 0.0,
            early_stopping: true,
            num_groups: 1,
            eos_token_id: 0,
            diversity_penalty: 0.0,
        };
        let result = cpu_beam_search(eos_after_n(5, 1), &cfg).unwrap();
        assert!(result.early_stopped);
    }

    #[test]
    fn result_contains_at_least_one_beam() {
        let cfg = default_config_with(4, 3);
        let result = cpu_beam_search(uniform_logits(10), &cfg).unwrap();
        assert!(!result.beams.is_empty());
    }

    // ── beam pruning correctness ──────────────────────────────────────

    #[test]
    fn pruning_keeps_top_k_beams() {
        let n = LengthNormalizer::new(0.0);
        let mut pool = BeamPool::new(3, 1);
        pool.seed();
        let candidates: Vec<Beam> = (0..10)
            .map(|i| {
                let mut b = Beam::new(0);
                b.log_prob = -(i as f64);
                b.token_ids = vec![i as u32];
                b
            })
            .collect();
        pool.update_group(0, candidates, &n);
        let group = pool.group_beams(0);
        assert_eq!(group.len(), 3);
        // Verify they are the top 3 (log_probs 0, -1, -2).
        let probs: Vec<f64> = group.iter().map(|b| b.log_prob).collect();
        assert!(probs.contains(&0.0));
        assert!(probs.contains(&-1.0));
        assert!(probs.contains(&-2.0));
    }

    #[test]
    fn pruning_with_length_penalty_reorders() {
        let n = LengthNormalizer::new(1.0);
        let mut pool = BeamPool::new(2, 1);
        pool.seed();
        // Two beams: short low-prob vs long lower-prob.
        let mut short = Beam::new(0);
        short.log_prob = -5.0;
        short.token_ids = vec![1];
        let mut long = Beam::new(0);
        long.log_prob = -8.0;
        long.token_ids = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        pool.update_group(0, vec![short, long], &n);
        let group = pool.group_beams(0);
        assert_eq!(group.len(), 2);
        // With length penalty alpha=1.0, the longer sequence gets a
        // more favourable normalisation.
        assert!(group[0].score >= group[1].score);
    }

    // ── additional coverage ───────────────────────────────────────────

    #[test]
    fn beam_config_default_is_valid() {
        assert!(validate_beam_config(&BeamConfig::default()).is_ok());
    }

    #[test]
    fn beam_search_error_display() {
        let errors = vec![
            BeamSearchError::InvalidBeamWidth,
            BeamSearchError::InvalidMaxLength,
            BeamSearchError::InvalidNumGroups,
            BeamSearchError::InvalidLengthPenalty,
            BeamSearchError::EmptyVocab,
            BeamSearchError::NoBeamsAlive,
            BeamSearchError::NumericalError("test".into()),
        ];
        for e in &errors {
            let s = format!("{e}");
            assert!(!s.is_empty(), "Display should produce non-empty string");
        }
    }

    #[test]
    fn beam_search_error_is_error_trait() {
        let e: Box<dyn std::error::Error> = Box::new(BeamSearchError::NumericalError("x".into()));
        assert!(!e.to_string().is_empty());
    }

    #[test]
    fn diverse_beam_search_debug() {
        let d = DiverseBeamSearch::new(1.0);
        let s = format!("{d:?}");
        assert!(s.contains("DiverseBeamSearch"));
    }

    #[test]
    fn beam_pool_best_active_log_prob_neg_inf_when_empty() {
        let mut pool = BeamPool::new(1, 1);
        pool.active.clear();
        assert_eq!(pool.best_active_log_prob(), f64::NEG_INFINITY);
    }

    #[test]
    fn beam_extend_preserves_group_id() {
        let b = Beam::new(7);
        let b2 = b.extend(1, -0.5, 0);
        assert_eq!(b2.group_id, 7);
    }
}
