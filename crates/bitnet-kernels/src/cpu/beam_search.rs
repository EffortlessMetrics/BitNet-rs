//! CPU beam search kernel with diversity and length penalty support.
//!
//! Provides a configurable beam search implementation for autoregressive
//! token generation, including diverse beam search via Hamming diversity
//! penalties and length-normalized scoring.

use std::fmt;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during beam search.
#[derive(Debug, Clone, PartialEq)]
pub enum BeamSearchError {
    /// Beam width must be at least 1.
    InvalidBeamWidth(usize),
    /// Max length must be at least 1.
    InvalidMaxLength(usize),
    /// Logits slice was empty.
    EmptyLogits,
    /// Logits length does not match the expected vocabulary size.
    VocabMismatch { expected: usize, got: usize },
    /// No active beams remain to expand.
    NoActiveBeams,
}

impl fmt::Display for BeamSearchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBeamWidth(w) => write!(f, "invalid beam width: {w} (must be >= 1)"),
            Self::InvalidMaxLength(l) => write!(f, "invalid max length: {l} (must be >= 1)"),
            Self::EmptyLogits => write!(f, "logits slice is empty"),
            Self::VocabMismatch { expected, got } => {
                write!(f, "vocab size mismatch: expected {expected}, got {got}")
            }
            Self::NoActiveBeams => write!(f, "no active beams remain"),
        }
    }
}

impl std::error::Error for BeamSearchError {}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for beam search decoding.
#[derive(Debug, Clone)]
pub struct BeamSearchConfig {
    /// Number of beams to keep at each step.
    pub beam_width: usize,
    /// Length penalty (α): score = log_prob / len^α.  0 = no penalty, 1 = full.
    pub length_penalty: f32,
    /// Stop as soon as `beam_width` finished hypotheses exist.
    pub early_stopping: bool,
    /// Absolute maximum sequence length (including prompt tokens).
    pub max_length: usize,
    /// Hamming diversity penalty (λ) applied across beam groups.
    pub diversity_penalty: f32,
}

impl Default for BeamSearchConfig {
    fn default() -> Self {
        Self {
            beam_width: 5,
            length_penalty: 1.0,
            early_stopping: false,
            max_length: 512,
            diversity_penalty: 0.0,
        }
    }
}

impl BeamSearchConfig {
    /// Validate the configuration, returning an error on invalid values.
    pub fn validate(&self) -> Result<(), BeamSearchError> {
        if self.beam_width == 0 {
            return Err(BeamSearchError::InvalidBeamWidth(self.beam_width));
        }
        if self.max_length == 0 {
            return Err(BeamSearchError::InvalidMaxLength(self.max_length));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Hypothesis
// ---------------------------------------------------------------------------

/// A single beam hypothesis.
#[derive(Debug, Clone)]
pub struct BeamHypothesis {
    /// Token ids generated so far.
    pub tokens: Vec<u32>,
    /// Cumulative log-probability (un-normalised).
    pub score: f64,
    /// Whether an end-of-sequence token was emitted.
    pub is_finished: bool,
}

impl BeamHypothesis {
    /// Create a new hypothesis seeded with a single token.
    pub fn new(token: u32, score: f64) -> Self {
        Self { tokens: vec![token], score, is_finished: false }
    }

    /// Create an empty hypothesis (start-of-sequence).
    pub fn empty() -> Self {
        Self { tokens: Vec::new(), score: 0.0, is_finished: false }
    }

    /// Extend the hypothesis with a new token and its log-probability.
    pub fn extend(&self, token: u32, log_prob: f64) -> Self {
        let mut tokens = self.tokens.clone();
        tokens.push(token);
        Self { tokens, score: self.score + log_prob, is_finished: false }
    }
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// Mutable state tracked across beam search steps.
#[derive(Debug, Clone)]
pub struct BeamSearchState {
    /// Currently active (unfinished) beams.
    pub active_beams: Vec<BeamHypothesis>,
    /// Hypotheses that reached EOS.
    pub finished_beams: Vec<BeamHypothesis>,
    /// Current decoding step (0-indexed).
    pub step: usize,
}

impl BeamSearchState {
    /// Initialise state with a single empty beam.
    pub fn new() -> Self {
        Self { active_beams: vec![BeamHypothesis::empty()], finished_beams: Vec::new(), step: 0 }
    }

    /// Initialise state with the given seed beams.
    pub fn with_beams(beams: Vec<BeamHypothesis>) -> Self {
        Self { active_beams: beams, finished_beams: Vec::new(), step: 0 }
    }
}

impl Default for BeamSearchState {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------

/// Compute a length-normalised beam score.
///
/// `score = raw_log_prob / length_penalty_divisor`
///
/// where `length_penalty_divisor = ((5 + len) / 6)^alpha` (Wu et al., 2016).
pub fn beam_search_score(raw_score: f64, length: usize, length_penalty: f32) -> f64 {
    if length == 0 {
        return raw_score;
    }
    let lp = ((5.0 + length as f64) / 6.0).powf(length_penalty as f64);
    raw_score / lp
}

// ---------------------------------------------------------------------------
// Core step
// ---------------------------------------------------------------------------

/// Perform one beam search expansion step.
///
/// * `state`  – current beam search state (mutated in place).
/// * `logits` – `[beam_width × vocab_size]` flattened log-probabilities, one
///   row per active beam.  If only a single row is provided it is broadcast to
///   all active beams.
/// * `vocab_size` – vocabulary size.
/// * `eos_token_id` – end-of-sequence token id.
/// * `config` – beam search configuration.
///
/// Returns the updated state.
pub fn beam_search_step(
    state: &mut BeamSearchState,
    logits: &[f32],
    vocab_size: usize,
    eos_token_id: u32,
    config: &BeamSearchConfig,
) -> Result<(), BeamSearchError> {
    config.validate()?;

    if vocab_size == 0 {
        return Err(BeamSearchError::EmptyLogits);
    }
    if logits.is_empty() {
        return Err(BeamSearchError::EmptyLogits);
    }
    if state.active_beams.is_empty() {
        return Err(BeamSearchError::NoActiveBeams);
    }

    let num_beams = state.active_beams.len();
    let broadcast = logits.len() == vocab_size;

    if !broadcast && logits.len() != num_beams * vocab_size {
        return Err(BeamSearchError::VocabMismatch {
            expected: num_beams * vocab_size,
            got: logits.len(),
        });
    }

    // Collect (score, beam_idx, token) candidates.
    let mut candidates: Vec<(f64, usize, u32)> =
        Vec::with_capacity(num_beams * vocab_size.min(config.beam_width * 2));

    for (beam_idx, beam) in state.active_beams.iter().enumerate() {
        let row = if broadcast {
            &logits[..vocab_size]
        } else {
            &logits[beam_idx * vocab_size..(beam_idx + 1) * vocab_size]
        };
        for (tok, &lp) in row.iter().enumerate() {
            candidates.push((beam.score + lp as f64, beam_idx, tok as u32));
        }
    }

    // Sort descending by score.
    candidates.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut next_beams: Vec<BeamHypothesis> = Vec::with_capacity(config.beam_width);

    for &(score, beam_idx, token) in &candidates {
        if next_beams.len() >= config.beam_width {
            break;
        }
        let mut hyp =
            state.active_beams[beam_idx].extend(token, score - state.active_beams[beam_idx].score);
        if token == eos_token_id {
            hyp.is_finished = true;
            state.finished_beams.push(hyp);
        } else {
            next_beams.push(hyp);
        }
    }

    state.active_beams = next_beams;
    state.step += 1;
    Ok(())
}

// ---------------------------------------------------------------------------
// Pruning
// ---------------------------------------------------------------------------

/// Prune the active beams, keeping only the top `beam_width` by normalised
/// score.
pub fn beam_search_prune(state: &mut BeamSearchState, config: &BeamSearchConfig) {
    if state.active_beams.len() <= config.beam_width {
        return;
    }
    state.active_beams.sort_unstable_by(|a, b| {
        let sa = beam_search_score(a.score, a.tokens.len(), config.length_penalty);
        let sb = beam_search_score(b.score, b.tokens.len(), config.length_penalty);
        sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
    });
    state.active_beams.truncate(config.beam_width);
}

// ---------------------------------------------------------------------------
// Diversity
// ---------------------------------------------------------------------------

/// Apply Hamming diversity penalty to `logits` for a given beam group.
///
/// For each token already selected by a *previous* beam in this step, subtract
/// `diversity_penalty` from its logit.  This encourages different beams to
/// explore different tokens (Vijayakumar et al., 2016).
///
/// * `logits` – mutable logit row for the current beam (length = vocab_size).
/// * `selected_tokens` – tokens already chosen by earlier beams in this step.
/// * `diversity_penalty` – penalty λ subtracted per prior selection.
pub fn beam_search_diverse(logits: &mut [f32], selected_tokens: &[u32], diversity_penalty: f32) {
    for &tok in selected_tokens {
        if (tok as usize) < logits.len() {
            logits[tok as usize] -= diversity_penalty;
        }
    }
}

// ---------------------------------------------------------------------------
// Completion check
// ---------------------------------------------------------------------------

/// Return `true` when beam search should stop.
///
/// Stopping conditions:
/// 1. All active beams are exhausted.
/// 2. `early_stopping` is set and at least `beam_width` finished hypotheses
///    exist.
/// 3. The current step has reached `max_length`.
pub fn beam_search_complete(state: &BeamSearchState, config: &BeamSearchConfig) -> bool {
    if state.active_beams.is_empty() {
        return true;
    }
    if config.early_stopping && state.finished_beams.len() >= config.beam_width {
        return true;
    }
    if state.step >= config.max_length {
        return true;
    }
    false
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers -----------------------------------------------------------

    fn default_config(beam_width: usize) -> BeamSearchConfig {
        BeamSearchConfig { beam_width, ..BeamSearchConfig::default() }
    }

    /// Build uniform logits of the given vocab size.
    fn uniform_logits(vocab_size: usize, value: f32) -> Vec<f32> {
        vec![value; vocab_size]
    }

    /// Build logits where `hot` token has `hot_val` and rest have `cold_val`.
    fn one_hot_logits(vocab_size: usize, hot: usize, hot_val: f32, cold_val: f32) -> Vec<f32> {
        let mut v = vec![cold_val; vocab_size];
        if hot < vocab_size {
            v[hot] = hot_val;
        }
        v
    }

    // =====================================================================
    // Config validation
    // =====================================================================

    #[test]
    fn config_default_is_valid() {
        assert!(BeamSearchConfig::default().validate().is_ok());
    }

    #[test]
    fn config_zero_beam_width_rejected() {
        let c = BeamSearchConfig { beam_width: 0, ..Default::default() };
        assert_eq!(c.validate(), Err(BeamSearchError::InvalidBeamWidth(0)));
    }

    #[test]
    fn config_zero_max_length_rejected() {
        let c = BeamSearchConfig { max_length: 0, ..Default::default() };
        assert_eq!(c.validate(), Err(BeamSearchError::InvalidMaxLength(0)));
    }

    #[test]
    fn config_beam_width_one_valid() {
        let c = BeamSearchConfig { beam_width: 1, ..Default::default() };
        assert!(c.validate().is_ok());
    }

    #[test]
    fn config_large_beam_width_valid() {
        let c = BeamSearchConfig { beam_width: 1000, ..Default::default() };
        assert!(c.validate().is_ok());
    }

    // =====================================================================
    // Error display
    // =====================================================================

    #[test]
    fn error_display_invalid_beam_width() {
        let e = BeamSearchError::InvalidBeamWidth(0);
        assert_eq!(e.to_string(), "invalid beam width: 0 (must be >= 1)");
    }

    #[test]
    fn error_display_invalid_max_length() {
        let e = BeamSearchError::InvalidMaxLength(0);
        assert_eq!(e.to_string(), "invalid max length: 0 (must be >= 1)");
    }

    #[test]
    fn error_display_empty_logits() {
        assert_eq!(BeamSearchError::EmptyLogits.to_string(), "logits slice is empty");
    }

    #[test]
    fn error_display_vocab_mismatch() {
        let e = BeamSearchError::VocabMismatch { expected: 100, got: 50 };
        assert!(e.to_string().contains("100"));
    }

    #[test]
    fn error_display_no_active_beams() {
        assert_eq!(BeamSearchError::NoActiveBeams.to_string(), "no active beams remain");
    }

    #[test]
    fn error_is_std_error() {
        let e: &dyn std::error::Error = &BeamSearchError::EmptyLogits;
        let _ = e.to_string();
    }

    // =====================================================================
    // Hypothesis
    // =====================================================================

    #[test]
    fn hypothesis_new() {
        let h = BeamHypothesis::new(42, -1.0);
        assert_eq!(h.tokens, vec![42]);
        assert!((h.score - (-1.0)).abs() < f64::EPSILON);
        assert!(!h.is_finished);
    }

    #[test]
    fn hypothesis_empty() {
        let h = BeamHypothesis::empty();
        assert!(h.tokens.is_empty());
        assert!((h.score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn hypothesis_extend() {
        let h = BeamHypothesis::empty();
        let h2 = h.extend(10, -0.5);
        assert_eq!(h2.tokens, vec![10]);
        assert!((h2.score - (-0.5)).abs() < f64::EPSILON);
        let h3 = h2.extend(20, -0.3);
        assert_eq!(h3.tokens, vec![10, 20]);
        assert!((h3.score - (-0.8)).abs() < 1e-9);
    }

    #[test]
    fn hypothesis_extend_preserves_original() {
        let h = BeamHypothesis::new(1, -1.0);
        let _ = h.extend(2, -0.5);
        assert_eq!(h.tokens.len(), 1);
    }

    // =====================================================================
    // State
    // =====================================================================

    #[test]
    fn state_new_has_one_beam() {
        let s = BeamSearchState::new();
        assert_eq!(s.active_beams.len(), 1);
        assert!(s.finished_beams.is_empty());
        assert_eq!(s.step, 0);
    }

    #[test]
    fn state_with_beams() {
        let beams = vec![BeamHypothesis::new(1, 0.0), BeamHypothesis::new(2, 0.0)];
        let s = BeamSearchState::with_beams(beams);
        assert_eq!(s.active_beams.len(), 2);
    }

    #[test]
    fn state_default_equals_new() {
        let a = BeamSearchState::new();
        let b = BeamSearchState::default();
        assert_eq!(a.active_beams.len(), b.active_beams.len());
        assert_eq!(a.step, b.step);
    }

    // =====================================================================
    // Scoring
    // =====================================================================

    #[test]
    fn score_no_penalty() {
        let s = beam_search_score(-5.0, 10, 0.0);
        assert!((s - (-5.0)).abs() < 1e-9);
    }

    #[test]
    fn score_full_penalty() {
        let s = beam_search_score(-6.0, 6, 1.0);
        // divisor = ((5+6)/6)^1 = 11/6
        let expected = -6.0 / (11.0 / 6.0);
        assert!((s - expected).abs() < 1e-9);
    }

    #[test]
    fn score_zero_length() {
        assert!((beam_search_score(-3.0, 0, 1.0) - (-3.0)).abs() < 1e-9);
    }

    #[test]
    fn score_longer_seq_penalised_more() {
        let short = beam_search_score(-10.0, 5, 1.0);
        let long = beam_search_score(-10.0, 50, 1.0);
        // Longer sequence has larger divisor, so normalised score is less
        // negative (closer to 0).  For negative raw scores the *shorter*
        // sequence therefore has the more negative (worse) normalised score.
        assert!(short < long);
    }

    #[test]
    fn score_high_alpha_stronger_penalty() {
        let mild = beam_search_score(-10.0, 10, 0.5);
        let strong = beam_search_score(-10.0, 10, 2.0);
        assert!(mild < strong); // stronger penalty → larger divisor → less negative
    }

    #[test]
    fn score_positive_raw_score() {
        let s = beam_search_score(10.0, 5, 1.0);
        assert!(s > 0.0);
        assert!(s < 10.0); // normalised down
    }

    // =====================================================================
    // Beam search step – basic
    // =====================================================================

    #[test]
    fn step_single_beam_picks_best_token() {
        let mut state = BeamSearchState::new();
        let config = default_config(1);
        let logits = one_hot_logits(4, 2, 0.0, -10.0);
        beam_search_step(&mut state, &logits, 4, 99, &config).unwrap();
        assert_eq!(state.active_beams.len(), 1);
        assert_eq!(state.active_beams[0].tokens, vec![2]);
    }

    #[test]
    fn step_two_beams_pick_top_two() {
        let mut state = BeamSearchState::new();
        let config = default_config(2);
        let mut logits = [-10.0f32; 5];
        logits[1] = 0.0;
        logits[3] = -1.0;
        beam_search_step(&mut state, &logits, 5, 99, &config).unwrap();
        assert_eq!(state.active_beams.len(), 2);
        let toks: Vec<u32> = state.active_beams.iter().map(|b| *b.tokens.last().unwrap()).collect();
        assert!(toks.contains(&1));
        assert!(toks.contains(&3));
    }

    #[test]
    fn step_increments_step_counter() {
        let mut state = BeamSearchState::new();
        let config = default_config(2);
        let logits = uniform_logits(3, -1.0);
        beam_search_step(&mut state, &logits, 3, 99, &config).unwrap();
        assert_eq!(state.step, 1);
        beam_search_step(&mut state, &logits, 3, 99, &config).unwrap();
        assert_eq!(state.step, 2);
    }

    #[test]
    fn step_eos_moves_to_finished() {
        let mut state = BeamSearchState::new();
        let config = default_config(2);
        let eos: u32 = 0;
        let logits = one_hot_logits(4, eos as usize, 0.0, -10.0);
        beam_search_step(&mut state, &logits, 4, eos, &config).unwrap();
        assert_eq!(state.finished_beams.len(), 1);
        assert!(state.finished_beams[0].is_finished);
    }

    #[test]
    fn step_broadcast_logits() {
        let mut state = BeamSearchState::with_beams(vec![
            BeamHypothesis::new(0, 0.0),
            BeamHypothesis::new(1, 0.0),
        ]);
        let config = default_config(2);
        let logits = one_hot_logits(3, 2, 0.0, -5.0);
        // single row → broadcast
        beam_search_step(&mut state, &logits, 3, 99, &config).unwrap();
        for b in &state.active_beams {
            assert_eq!(*b.tokens.last().unwrap(), 2);
        }
    }

    #[test]
    fn step_multi_row_logits() {
        let mut state = BeamSearchState::with_beams(vec![
            BeamHypothesis::new(0, 0.0),
            BeamHypothesis::new(1, 0.0),
        ]);
        let config = default_config(2);
        // beam 0 prefers token 1; beam 1 prefers token 2
        let mut logits = [-10.0f32; 6]; // 2×3
        logits[1] = 0.0; // beam 0, token 1
        logits[5] = 0.0; // beam 1, token 2
        beam_search_step(&mut state, &logits, 3, 99, &config).unwrap();
        assert_eq!(state.active_beams.len(), 2);
    }

    // =====================================================================
    // Beam search step – errors
    // =====================================================================

    #[test]
    fn step_empty_logits_error() {
        let mut state = BeamSearchState::new();
        let config = default_config(2);
        let r = beam_search_step(&mut state, &[], 4, 0, &config);
        assert_eq!(r, Err(BeamSearchError::EmptyLogits));
    }

    #[test]
    fn step_zero_vocab_error() {
        let mut state = BeamSearchState::new();
        let config = default_config(2);
        let r = beam_search_step(&mut state, &[1.0], 0, 0, &config);
        assert_eq!(r, Err(BeamSearchError::EmptyLogits));
    }

    #[test]
    fn step_no_active_beams_error() {
        let mut state = BeamSearchState { active_beams: vec![], finished_beams: vec![], step: 0 };
        let config = default_config(2);
        let r = beam_search_step(&mut state, &[1.0, 2.0], 2, 99, &config);
        assert_eq!(r, Err(BeamSearchError::NoActiveBeams));
    }

    #[test]
    fn step_vocab_mismatch_error() {
        let mut state = BeamSearchState::with_beams(vec![
            BeamHypothesis::new(0, 0.0),
            BeamHypothesis::new(1, 0.0),
        ]);
        let config = default_config(2);
        // 2 beams × 3 vocab = 6 expected, but we give 5
        let r = beam_search_step(&mut state, &[0.0; 5], 3, 99, &config);
        assert!(matches!(r, Err(BeamSearchError::VocabMismatch { .. })));
    }

    #[test]
    fn step_invalid_config_propagates() {
        let mut state = BeamSearchState::new();
        let config = BeamSearchConfig { beam_width: 0, ..Default::default() };
        let r = beam_search_step(&mut state, &[1.0], 1, 99, &config);
        assert!(matches!(r, Err(BeamSearchError::InvalidBeamWidth(_))));
    }

    // =====================================================================
    // Pruning
    // =====================================================================

    #[test]
    fn prune_no_op_when_within_budget() {
        let mut state = BeamSearchState::with_beams(vec![
            BeamHypothesis::new(1, -1.0),
            BeamHypothesis::new(2, -2.0),
        ]);
        let config = default_config(5);
        beam_search_prune(&mut state, &config);
        assert_eq!(state.active_beams.len(), 2);
    }

    #[test]
    fn prune_truncates_to_beam_width() {
        let mut state = BeamSearchState::with_beams(vec![
            BeamHypothesis::new(1, -1.0),
            BeamHypothesis::new(2, -2.0),
            BeamHypothesis::new(3, -0.5),
            BeamHypothesis::new(4, -3.0),
        ]);
        let config = default_config(2);
        beam_search_prune(&mut state, &config);
        assert_eq!(state.active_beams.len(), 2);
    }

    #[test]
    fn prune_keeps_highest_scores() {
        let mut state = BeamSearchState::with_beams(vec![
            BeamHypothesis::new(1, -5.0),
            BeamHypothesis::new(2, -1.0),
            BeamHypothesis::new(3, -3.0),
        ]);
        let config = BeamSearchConfig { beam_width: 1, length_penalty: 0.0, ..Default::default() };
        beam_search_prune(&mut state, &config);
        assert_eq!(state.active_beams.len(), 1);
        assert_eq!(state.active_beams[0].tokens, vec![2]); // best score
    }

    #[test]
    fn prune_respects_length_penalty() {
        // With length penalty the normalised score divides by a length
        // factor.  For negative raw scores, the longer hypothesis gets a
        // *less* negative normalised score (closer to 0) and therefore
        // ranks higher.  We verify pruning picks the longer hypothesis
        // when it has a better normalised score.
        let short = BeamHypothesis { tokens: vec![1], score: -2.0, is_finished: false };
        let long = BeamHypothesis {
            tokens: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            score: -2.5,
            is_finished: false,
        };
        let mut state = BeamSearchState::with_beams(vec![short.clone(), long.clone()]);
        let config = BeamSearchConfig { beam_width: 1, length_penalty: 1.0, ..Default::default() };
        beam_search_prune(&mut state, &config);
        // Long sequence wins because -2.5 / large_divisor > -2.0 / small_divisor.
        assert_eq!(state.active_beams[0].tokens.len(), 10);
    }

    #[test]
    fn prune_empty_beams_is_noop() {
        let mut state = BeamSearchState { active_beams: vec![], finished_beams: vec![], step: 0 };
        let config = default_config(2);
        beam_search_prune(&mut state, &config);
        assert!(state.active_beams.is_empty());
    }

    // =====================================================================
    // Diversity
    // =====================================================================

    #[test]
    fn diverse_penalises_selected_tokens() {
        let mut logits = [0.0f32; 5];
        beam_search_diverse(&mut logits, &[1, 3], 2.0);
        assert!((logits[0] - 0.0).abs() < f32::EPSILON);
        assert!((logits[1] - (-2.0)).abs() < f32::EPSILON);
        assert!((logits[2] - 0.0).abs() < f32::EPSILON);
        assert!((logits[3] - (-2.0)).abs() < f32::EPSILON);
        assert!((logits[4] - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn diverse_no_tokens_no_change() {
        let mut logits = [1.0f32; 4];
        beam_search_diverse(&mut logits, &[], 5.0);
        assert!(logits.iter().all(|&v| (v - 1.0).abs() < f32::EPSILON));
    }

    #[test]
    fn diverse_out_of_bounds_token_ignored() {
        let mut logits = [0.0f32; 3];
        beam_search_diverse(&mut logits, &[10], 1.0);
        assert!(logits.iter().all(|&v| v.abs() < f32::EPSILON));
    }

    #[test]
    fn diverse_zero_penalty_no_change() {
        let mut logits = vec![1.0, 2.0, 3.0];
        beam_search_diverse(&mut logits, &[0, 1, 2], 0.0);
        assert!((logits[0] - 1.0).abs() < f32::EPSILON);
        assert!((logits[1] - 2.0).abs() < f32::EPSILON);
        assert!((logits[2] - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn diverse_cumulative_penalty() {
        let mut logits = [0.0f32; 3];
        beam_search_diverse(&mut logits, &[1, 1], 1.0);
        // Token 1 penalised twice.
        assert!((logits[1] - (-2.0)).abs() < f32::EPSILON);
    }

    // =====================================================================
    // Completion
    // =====================================================================

    #[test]
    fn complete_fresh_state_not_done() {
        let state = BeamSearchState::new();
        let config = default_config(2);
        assert!(!beam_search_complete(&state, &config));
    }

    #[test]
    fn complete_no_active_beams() {
        let state = BeamSearchState { active_beams: vec![], finished_beams: vec![], step: 0 };
        let config = default_config(2);
        assert!(beam_search_complete(&state, &config));
    }

    #[test]
    fn complete_early_stopping_enough_finished() {
        let mut state = BeamSearchState::new();
        state.finished_beams = vec![
            BeamHypothesis { tokens: vec![1], score: -1.0, is_finished: true },
            BeamHypothesis { tokens: vec![2], score: -2.0, is_finished: true },
        ];
        let config = BeamSearchConfig { beam_width: 2, early_stopping: true, ..Default::default() };
        assert!(beam_search_complete(&state, &config));
    }

    #[test]
    fn complete_early_stopping_not_enough_finished() {
        let mut state = BeamSearchState::new();
        state.finished_beams =
            vec![BeamHypothesis { tokens: vec![1], score: -1.0, is_finished: true }];
        let config = BeamSearchConfig { beam_width: 3, early_stopping: true, ..Default::default() };
        assert!(!beam_search_complete(&state, &config));
    }

    #[test]
    fn complete_max_length_reached() {
        let mut state = BeamSearchState::new();
        state.step = 100;
        let config = BeamSearchConfig { max_length: 100, ..Default::default() };
        assert!(beam_search_complete(&state, &config));
    }

    #[test]
    fn complete_max_length_not_reached() {
        let mut state = BeamSearchState::new();
        state.step = 99;
        let config = BeamSearchConfig { max_length: 100, ..Default::default() };
        assert!(!beam_search_complete(&state, &config));
    }

    #[test]
    fn complete_early_stopping_off_ignores_finished() {
        let mut state = BeamSearchState::new();
        state.finished_beams = vec![
            BeamHypothesis { tokens: vec![1], score: -1.0, is_finished: true },
            BeamHypothesis { tokens: vec![2], score: -2.0, is_finished: true },
            BeamHypothesis { tokens: vec![3], score: -3.0, is_finished: true },
        ];
        let config =
            BeamSearchConfig { beam_width: 2, early_stopping: false, ..Default::default() };
        assert!(!beam_search_complete(&state, &config));
    }

    // =====================================================================
    // Multi-step integration
    // =====================================================================

    #[test]
    fn multi_step_greedy_beam1() {
        let mut state = BeamSearchState::new();
        let config = default_config(1);
        let vocab = 3;
        let eos = 99u32;

        // Step 1: pick token 0
        let logits = one_hot_logits(vocab, 0, 0.0, -10.0);
        beam_search_step(&mut state, &logits, vocab, eos, &config).unwrap();
        assert_eq!(state.active_beams[0].tokens, vec![0]);

        // Step 2: pick token 2
        let logits = one_hot_logits(vocab, 2, 0.0, -10.0);
        beam_search_step(&mut state, &logits, vocab, eos, &config).unwrap();
        assert_eq!(state.active_beams[0].tokens, vec![0, 2]);
    }

    #[test]
    fn multi_step_beam2_diverges() {
        let mut state = BeamSearchState::new();
        let config = default_config(2);
        let vocab = 4;
        let eos = 99u32;

        // Step 1: tokens 1 and 3 are the top-2
        let mut logits = vec![-10.0f32; vocab];
        logits[1] = 0.0;
        logits[3] = -1.0;
        beam_search_step(&mut state, &logits, vocab, eos, &config).unwrap();
        assert_eq!(state.active_beams.len(), 2);

        // Step 2: all beams see token 0 as best
        let logits2 = one_hot_logits(vocab, 0, 0.0, -10.0);
        // broadcast
        beam_search_step(&mut state, &logits2, vocab, eos, &config).unwrap();
        for b in &state.active_beams {
            assert_eq!(*b.tokens.last().unwrap(), 0);
        }
    }

    #[test]
    fn multi_step_with_eos_collects_finished() {
        let mut state = BeamSearchState::new();
        let config = default_config(3);
        let vocab = 3;
        let eos = 0u32;

        // 3 candidates: token 0 (=eos), token 1, token 2
        let logits = vec![0.0f32, -1.0, -2.0];
        beam_search_step(&mut state, &logits, vocab, eos, &config).unwrap();

        assert_eq!(state.finished_beams.len(), 1);
        assert_eq!(state.active_beams.len(), 2);
    }

    // =====================================================================
    // Edge cases
    // =====================================================================

    #[test]
    fn single_token_vocab() {
        let mut state = BeamSearchState::new();
        let config = default_config(1);
        beam_search_step(&mut state, &[0.5], 1, 99, &config).unwrap();
        assert_eq!(state.active_beams[0].tokens, vec![0]);
    }

    #[test]
    fn large_vocab_top1() {
        let mut state = BeamSearchState::new();
        let config = default_config(1);
        let vocab = 50_000;
        let logits = one_hot_logits(vocab, 12345, 0.0, -10.0);
        beam_search_step(&mut state, &logits, vocab, 99, &config).unwrap();
        assert_eq!(state.active_beams[0].tokens, vec![12345]);
    }

    #[test]
    fn all_eos_empties_active() {
        let mut state = BeamSearchState::new();
        let config = default_config(1);
        // Only token is eos.
        beam_search_step(&mut state, &[0.0], 1, 0, &config).unwrap();
        assert!(state.active_beams.is_empty());
        assert_eq!(state.finished_beams.len(), 1);
    }

    #[test]
    fn negative_logits_still_rank_correctly() {
        let mut state = BeamSearchState::new();
        let config = default_config(1);
        let logits = vec![-5.0f32, -3.0, -8.0, -1.0];
        beam_search_step(&mut state, &logits, 4, 99, &config).unwrap();
        assert_eq!(state.active_beams[0].tokens, vec![3]); // -1.0 is best
    }

    #[test]
    fn identical_scores_deterministic_order() {
        let mut state = BeamSearchState::new();
        let config = default_config(2);
        let logits = [0.0f32; 4];
        beam_search_step(&mut state, &logits, 4, 99, &config).unwrap();
        // Must pick exactly 2, not crash.
        assert_eq!(state.active_beams.len(), 2);
    }

    // =====================================================================
    // Property tests
    // =====================================================================

    #[cfg(test)]
    mod prop {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn score_finite(raw in -1e6f64..1e6, len in 0usize..1000, alpha in 0.0f32..5.0) {
                let s = beam_search_score(raw, len, alpha);
                prop_assert!(s.is_finite(), "score must be finite: got {s}");
            }

            #[test]
            fn score_zero_length_identity(raw in -1e6f64..1e6, alpha in 0.0f32..5.0) {
                let s = beam_search_score(raw, 0, alpha);
                prop_assert!((s - raw).abs() < 1e-9, "zero-length score should equal raw");
            }

            #[test]
            fn step_preserves_beam_count(
                beam_width in 1usize..6,
                vocab in 2usize..20,
            ) {
                let mut state = BeamSearchState::new();
                let config = BeamSearchConfig { beam_width, ..Default::default() };
                let logits: Vec<f32> = (0..vocab).map(|i| -(i as f32)).collect();
                let _ = beam_search_step(&mut state, &logits, vocab, 9999, &config);
                prop_assert!(
                    state.active_beams.len() + state.finished_beams.len() <= beam_width,
                    "total beams should not exceed beam_width"
                );
            }

            #[test]
            fn prune_never_exceeds_width(n in 1usize..20, width in 1usize..10) {
                let beams: Vec<BeamHypothesis> = (0..n)
                    .map(|i| BeamHypothesis::new(i as u32, -(i as f64)))
                    .collect();
                let mut state = BeamSearchState::with_beams(beams);
                let config = BeamSearchConfig { beam_width: width, length_penalty: 0.0, ..Default::default() };
                beam_search_prune(&mut state, &config);
                prop_assert!(state.active_beams.len() <= width);
            }

            #[test]
            fn diverse_does_not_increase_logits(
                vocab in 2usize..50,
                penalty in 0.0f32..10.0,
            ) {
                let mut logits = vec![0.0f32; vocab];
                let tokens: Vec<u32> = (0..vocab.min(5)).map(|i| i as u32).collect();
                let orig = logits.clone();
                beam_search_diverse(&mut logits, &tokens, penalty);
                for i in 0..vocab {
                    prop_assert!(logits[i] <= orig[i] + f32::EPSILON);
                }
            }

            #[test]
            fn complete_once_max_length(steps in 0usize..1000, max_len in 1usize..500) {
                let mut state = BeamSearchState::new();
                state.step = steps;
                let config = BeamSearchConfig { max_length: max_len, ..Default::default() };
                if steps >= max_len {
                    prop_assert!(beam_search_complete(&state, &config));
                }
            }
        }
    }
}
