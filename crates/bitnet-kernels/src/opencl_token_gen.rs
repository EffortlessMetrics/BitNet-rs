//! Autoregressive token generation for Intel Arc A770 (OpenCL backend).
//!
//! This module provides CPU reference implementations for the forward-pass
//! sampling pipeline used during autoregressive text generation. The kernels
//! cover temperature scaling, repetition penalty, top-k / top-p filtering,
//! softmax sampling, and the outer generation loop with stop-condition
//! checking.

use std::fmt;
use std::time::Instant;

// ── Types ──────────────────────────────────────────────────────────────

/// Configuration for autoregressive token generation.
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_k: Option<u32>,
    pub top_p: Option<f32>,
    pub repetition_penalty: f32,
    pub stop_tokens: Vec<u32>,
    pub seed: Option<u64>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 128,
            temperature: 1.0,
            top_k: None,
            top_p: None,
            repetition_penalty: 1.0,
            stop_tokens: Vec::new(),
            seed: None,
        }
    }
}

/// Mutable state tracked across generation steps.
#[derive(Debug, Clone, Default)]
pub struct GenerationState {
    pub generated_tokens: Vec<u32>,
    pub kv_cache_positions: Vec<usize>,
    pub current_position: usize,
    pub total_time_us: u64,
    pub token_times_us: Vec<u64>,
}

/// Sampling strategy selector.
#[derive(Debug, Clone, PartialEq)]
pub enum SamplingMethod {
    Greedy,
    Temperature(f32),
    TopK(u32),
    TopP(f32),
    TopKP(u32, f32),
}

/// Reason generation was stopped.
#[derive(Debug, Clone, PartialEq)]
pub enum StopReason {
    MaxTokens,
    StopToken(u32),
    EndOfSequence,
    Error(String),
}

impl fmt::Display for StopReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MaxTokens => write!(f, "max_tokens reached"),
            Self::StopToken(t) => write!(f, "stop token {t}"),
            Self::EndOfSequence => write!(f, "end of sequence"),
            Self::Error(e) => write!(f, "error: {e}"),
        }
    }
}

/// Result of a completed generation run.
#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub tokens: Vec<u32>,
    pub stop_reason: StopReason,
    pub stats: GenerationStats,
}

/// Timing / throughput statistics for a generation run.
#[derive(Debug, Clone)]
pub struct GenerationStats {
    pub total_tokens: usize,
    pub prefill_time_us: u64,
    pub decode_time_us: u64,
    pub tokens_per_second: f32,
}

/// Errors arising from invalid generation parameters.
#[derive(Debug, Clone, PartialEq)]
pub enum GenerationError {
    InvalidConfig(String),
    EmptyPrompt,
    VocabTooSmall,
    NumericalError(String),
    MaxTokensExceeded,
}

impl fmt::Display for GenerationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(s) => write!(f, "invalid config: {s}"),
            Self::EmptyPrompt => write!(f, "empty prompt"),
            Self::VocabTooSmall => write!(f, "vocab too small"),
            Self::NumericalError(s) => write!(f, "numerical error: {s}"),
            Self::MaxTokensExceeded => write!(f, "max tokens exceeded"),
        }
    }
}

impl std::error::Error for GenerationError {}

// ── Deterministic RNG ──────────────────────────────────────────────────

/// Xorshift64 PRNG – returns a value in [0, 1).
fn xorshift64(state: &mut u64) -> f32 {
    if *state == 0 {
        *state = 0x5851_F42D_4C95_7F2D; // default non-zero seed
    }
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f32) / (u64::MAX as f32)
}

// ── CPU reference implementations ──────────────────────────────────────

/// Divide every logit by `temperature` (in-place).
pub fn cpu_apply_temperature(logits: &mut [f32], temperature: f32) {
    if temperature == 1.0 || logits.is_empty() {
        return;
    }
    let inv = 1.0 / temperature;
    for l in logits.iter_mut() {
        *l *= inv;
    }
}

/// Penalise previously generated tokens.
///
/// For each token in `generated`, the corresponding logit is divided by
/// `penalty` when positive and multiplied by `penalty` when negative.
pub fn cpu_apply_repetition_penalty(logits: &mut [f32], generated: &[u32], penalty: f32) {
    if penalty == 1.0 {
        return;
    }
    for &tok in generated {
        let idx = tok as usize;
        if idx < logits.len() {
            if logits[idx] > 0.0 {
                logits[idx] /= penalty;
            } else {
                logits[idx] *= penalty;
            }
        }
    }
}

/// Keep only the top-`k` logits; set all others to `-f32::INFINITY`.
pub fn cpu_top_k_filter(logits: &mut [f32], k: u32) {
    let k = k as usize;
    if k == 0 || k >= logits.len() {
        return;
    }
    // Find the k-th largest value.
    let mut sorted: Vec<f32> = logits.to_vec();
    sorted.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = sorted[k - 1];
    // Count how many values equal the threshold we should keep.
    let mut keep = k;
    for &v in &sorted[..k] {
        if v > threshold {
            keep -= 1;
        }
    }
    // keep now holds how many copies of `threshold` we still need.
    for l in logits.iter_mut() {
        if *l > threshold {
            // always kept
        } else if *l == threshold && keep > 0 {
            keep -= 1;
        } else {
            *l = f32::NEG_INFINITY;
        }
    }
}

/// Nucleus (top-p) sampling filter.
///
/// After sorting by descending probability, tokens whose cumulative
/// softmax mass exceeds `p` are set to `-f32::INFINITY`.
pub fn cpu_top_p_filter(logits: &mut [f32], p: f32) {
    if p >= 1.0 || logits.is_empty() {
        return;
    }
    // Stable softmax for ordering.
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();

    // Indices sorted by descending probability.
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_unstable_by(|&a, &b| {
        probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut cumulative = 0.0_f32;
    let mut keep = vec![false; logits.len()];
    for &idx in &indices {
        if cumulative >= p && cumulative > 0.0 {
            break;
        }
        keep[idx] = true;
        cumulative += probs[idx];
    }
    for (i, l) in logits.iter_mut().enumerate() {
        if !keep[i] {
            *l = f32::NEG_INFINITY;
        }
    }
}

/// Sample from a probability distribution with a deterministic xorshift
/// RNG. Returns the selected token index.
pub fn cpu_softmax_sample(logits: &[f32], seed: u64) -> u32 {
    if logits.is_empty() {
        return 0;
    }
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        return 0;
    }
    let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();

    let mut rng = seed;
    let r = xorshift64(&mut rng);
    let mut cumulative = 0.0_f32;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return i as u32;
        }
    }
    (logits.len() - 1) as u32
}

/// Greedy (argmax) decode.
pub fn cpu_greedy_decode(logits: &[f32]) -> u32 {
    if logits.is_empty() {
        return 0;
    }
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Full single-step sampling pipeline:
/// temperature → repetition penalty → top-k → top-p → sample/greedy.
pub fn cpu_generate_token(
    logits: &[f32],
    config: &GenerationConfig,
    state: &GenerationState,
) -> u32 {
    let mut buf = logits.to_vec();

    if config.temperature > 0.0 && config.temperature != 1.0 {
        cpu_apply_temperature(&mut buf, config.temperature);
    }

    if config.repetition_penalty != 1.0 {
        cpu_apply_repetition_penalty(&mut buf, &state.generated_tokens, config.repetition_penalty);
    }

    if let Some(k) = config.top_k {
        cpu_top_k_filter(&mut buf, k);
    }

    if let Some(p) = config.top_p {
        cpu_top_p_filter(&mut buf, p);
    }

    // If temperature ≈ 0 use greedy; otherwise sample.
    if config.temperature < 1e-6 {
        cpu_greedy_decode(&buf)
    } else {
        let seed = config.seed.unwrap_or(42).wrapping_add(state.generated_tokens.len() as u64);
        cpu_softmax_sample(&buf, seed)
    }
}

/// Check whether generation should stop after producing `token` at
/// `step`.
pub fn cpu_check_stop_condition(
    token: u32,
    step: usize,
    config: &GenerationConfig,
) -> Option<StopReason> {
    if config.stop_tokens.contains(&token) {
        return Some(StopReason::StopToken(token));
    }
    // EOS is token 0 by convention when not otherwise specified.
    if token == 0 && !config.stop_tokens.contains(&0) {
        return Some(StopReason::EndOfSequence);
    }
    if step + 1 >= config.max_tokens as usize {
        return Some(StopReason::MaxTokens);
    }
    None
}

/// Validate a [`GenerationConfig`], returning an error for obviously
/// wrong parameters.
pub fn validate_config(config: &GenerationConfig) -> Result<(), GenerationError> {
    if config.max_tokens == 0 {
        return Err(GenerationError::InvalidConfig("max_tokens must be > 0".into()));
    }
    if config.temperature < 0.0 {
        return Err(GenerationError::InvalidConfig("temperature must be >= 0".into()));
    }
    if config.repetition_penalty <= 0.0 {
        return Err(GenerationError::InvalidConfig("repetition_penalty must be > 0".into()));
    }
    if let Some(k) = config.top_k
        && k == 0
    {
        return Err(GenerationError::InvalidConfig("top_k must be > 0".into()));
    }
    if let Some(p) = config.top_p
        && !(0.0..=1.0).contains(&p)
    {
        return Err(GenerationError::InvalidConfig("top_p must be in [0, 1]".into()));
    }
    Ok(())
}

/// Run the full autoregressive generation loop.
///
/// `initial_logits_fn` is called with the previously generated token and
/// must return a logits vector of fixed vocabulary size.
pub fn cpu_generate_sequence(
    mut initial_logits_fn: impl FnMut(u32) -> Vec<f32>,
    config: &GenerationConfig,
) -> GenerationResult {
    if let Err(e) = validate_config(config) {
        return GenerationResult {
            tokens: Vec::new(),
            stop_reason: StopReason::Error(e.to_string()),
            stats: GenerationStats {
                total_tokens: 0,
                prefill_time_us: 0,
                decode_time_us: 0,
                tokens_per_second: 0.0,
            },
        };
    }

    let mut state = GenerationState::default();
    let start = Instant::now();

    // Prefill: obtain first logits from a dummy BOS token.
    let prefill_start = Instant::now();
    let first_logits = initial_logits_fn(0);
    let prefill_us = prefill_start.elapsed().as_micros() as u64;

    if first_logits.is_empty() {
        return GenerationResult {
            tokens: Vec::new(),
            stop_reason: StopReason::Error("empty logits".into()),
            stats: GenerationStats {
                total_tokens: 0,
                prefill_time_us: prefill_us,
                decode_time_us: 0,
                tokens_per_second: 0.0,
            },
        };
    }

    let decode_start = Instant::now();
    let mut logits = first_logits;

    for step in 0..config.max_tokens as usize {
        let tok_start = Instant::now();
        let token = cpu_generate_token(&logits, config, &state);
        let tok_us = tok_start.elapsed().as_micros() as u64;

        state.generated_tokens.push(token);
        state.token_times_us.push(tok_us);
        state.current_position += 1;
        state.kv_cache_positions.push(state.current_position);

        if let Some(reason) = cpu_check_stop_condition(token, step, config) {
            let decode_us = decode_start.elapsed().as_micros() as u64;
            let total_us = start.elapsed().as_micros() as u64;
            state.total_time_us = total_us;
            let n = state.generated_tokens.len();
            let tps = if decode_us > 0 { n as f32 / (decode_us as f32 / 1_000_000.0) } else { 0.0 };
            return GenerationResult {
                tokens: state.generated_tokens,
                stop_reason: reason,
                stats: GenerationStats {
                    total_tokens: n,
                    prefill_time_us: prefill_us,
                    decode_time_us: decode_us,
                    tokens_per_second: tps,
                },
            };
        }

        // Fetch next logits.
        logits = initial_logits_fn(token);
    }

    let decode_us = decode_start.elapsed().as_micros() as u64;
    let total_us = start.elapsed().as_micros() as u64;
    state.total_time_us = total_us;
    let n = state.generated_tokens.len();
    let tps = if decode_us > 0 { n as f32 / (decode_us as f32 / 1_000_000.0) } else { 0.0 };

    GenerationResult {
        tokens: state.generated_tokens,
        stop_reason: StopReason::MaxTokens,
        stats: GenerationStats {
            total_tokens: n,
            prefill_time_us: prefill_us,
            decode_time_us: decode_us,
            tokens_per_second: tps,
        },
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ────────────────────────────────────────────────────────

    fn sample_logits() -> Vec<f32> {
        vec![1.0, 2.0, 3.0, 4.0, 5.0]
    }

    fn uniform_logits(n: usize) -> Vec<f32> {
        vec![1.0; n]
    }

    fn dominated_logits() -> Vec<f32> {
        vec![0.0, 0.0, 100.0, 0.0, 0.0]
    }

    // ── temperature tests ─────────────────────────────────────────────

    #[test]
    fn temperature_1_0_is_identity() {
        let mut logits = sample_logits();
        let original = logits.clone();
        cpu_apply_temperature(&mut logits, 1.0);
        assert_eq!(logits, original);
    }

    #[test]
    fn temperature_approaching_zero_sharpens() {
        let mut logits = sample_logits();
        cpu_apply_temperature(&mut logits, 0.01);
        // The gap between adjacent logits should have grown.
        let gap = logits[4] - logits[3];
        assert!(gap > 50.0, "expected large gap, got {gap}");
    }

    #[test]
    fn temperature_2_0_flattens_distribution() {
        let mut logits = sample_logits();
        cpu_apply_temperature(&mut logits, 2.0);
        let range = logits[4] - logits[0];
        assert!(range < 4.0, "expected narrower range, got {range}");
    }

    #[test]
    fn temperature_on_empty_is_noop() {
        let mut logits: Vec<f32> = vec![];
        cpu_apply_temperature(&mut logits, 0.5);
        assert!(logits.is_empty());
    }

    // ── repetition penalty tests ──────────────────────────────────────

    #[test]
    fn repetition_penalty_1_0_is_identity() {
        let mut logits = sample_logits();
        let original = logits.clone();
        cpu_apply_repetition_penalty(&mut logits, &[0, 1], 1.0);
        assert_eq!(logits, original);
    }

    #[test]
    fn repetition_penalty_reduces_positive_logits() {
        let mut logits = sample_logits();
        let before = logits[2];
        cpu_apply_repetition_penalty(&mut logits, &[2], 2.0);
        assert!(logits[2] < before);
    }

    #[test]
    fn repetition_penalty_amplifies_negative_logits() {
        let mut logits = vec![-1.0, -2.0, -3.0];
        let before = logits[1];
        cpu_apply_repetition_penalty(&mut logits, &[1], 2.0);
        assert!(logits[1] < before, "negative logit should become more negative");
    }

    #[test]
    fn repetition_penalty_ignores_out_of_range() {
        let mut logits = sample_logits();
        let original = logits.clone();
        cpu_apply_repetition_penalty(&mut logits, &[999], 2.0);
        assert_eq!(logits, original);
    }

    #[test]
    fn repetition_penalty_increases_with_more_repetitions() {
        // Penalising the same token twice has the same effect as a
        // single penalty since the penalty is multiplicative per
        // occurrence, but applying it to more unique tokens reduces
        // more entries.
        let mut a = sample_logits();
        cpu_apply_repetition_penalty(&mut a, &[4], 1.5);
        let single = a[4];

        let mut b = sample_logits();
        cpu_apply_repetition_penalty(&mut b, &[4, 4], 1.5);
        let double = b[4];
        assert!(double < single);
    }

    // ── top-k tests ───────────────────────────────────────────────────

    #[test]
    fn top_k_1_is_greedy() {
        let mut logits = sample_logits();
        cpu_top_k_filter(&mut logits, 1);
        let finite: Vec<f32> = logits.iter().copied().filter(|v| v.is_finite()).collect();
        assert_eq!(finite.len(), 1);
        assert_eq!(finite[0], 5.0);
    }

    #[test]
    fn top_k_equals_vocab_keeps_all() {
        let mut logits = sample_logits();
        let len = logits.len() as u32;
        cpu_top_k_filter(&mut logits, len);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn top_k_larger_than_vocab_keeps_all() {
        let mut logits = sample_logits();
        cpu_top_k_filter(&mut logits, 100);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn top_k_3_keeps_three() {
        let mut logits = sample_logits();
        cpu_top_k_filter(&mut logits, 3);
        let finite: usize = logits.iter().filter(|v| v.is_finite()).count();
        assert_eq!(finite, 3);
    }

    #[test]
    fn top_k_zero_is_noop() {
        let mut logits = sample_logits();
        let original = logits.clone();
        cpu_top_k_filter(&mut logits, 0);
        assert_eq!(logits, original);
    }

    // ── top-p tests ───────────────────────────────────────────────────

    #[test]
    fn top_p_1_0_keeps_all() {
        let mut logits = sample_logits();
        cpu_top_p_filter(&mut logits, 1.0);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn top_p_small_keeps_dominant() {
        let mut logits = dominated_logits();
        cpu_top_p_filter(&mut logits, 0.01);
        // Only the dominant token at index 2 should remain.
        assert!(logits[2].is_finite());
        for (i, v) in logits.iter().enumerate() {
            if i != 2 {
                assert!(!v.is_finite() || *v == 0.0, "expected -inf at {i}, got {v}");
            }
        }
    }

    #[test]
    fn top_p_on_empty_is_noop() {
        let mut logits: Vec<f32> = vec![];
        cpu_top_p_filter(&mut logits, 0.5);
        assert!(logits.is_empty());
    }

    // ── combined top-k + top-p ────────────────────────────────────────

    #[test]
    fn top_k_then_top_p() {
        let mut logits = sample_logits();
        cpu_top_k_filter(&mut logits, 3);
        cpu_top_p_filter(&mut logits, 0.5);
        let finite: usize = logits.iter().filter(|v| v.is_finite()).count();
        assert!(finite >= 1 && finite <= 3);
    }

    // ── greedy decode tests ───────────────────────────────────────────

    #[test]
    fn greedy_returns_argmax() {
        assert_eq!(cpu_greedy_decode(&sample_logits()), 4);
    }

    #[test]
    fn greedy_returns_argmax_dominated() {
        assert_eq!(cpu_greedy_decode(&dominated_logits()), 2);
    }

    #[test]
    fn greedy_on_empty_returns_zero() {
        assert_eq!(cpu_greedy_decode(&[]), 0);
    }

    #[test]
    fn greedy_on_uniform_returns_first_max() {
        // With equal values argmax should return some valid index.
        let idx = cpu_greedy_decode(&uniform_logits(5));
        assert!(idx < 5);
    }

    #[test]
    fn greedy_returns_valid_vocab_index() {
        for size in [1, 10, 100, 1000] {
            let logits: Vec<f32> = (0..size).map(|i| (i as f32).sin()).collect();
            let idx = cpu_greedy_decode(&logits);
            assert!((idx as usize) < size);
        }
    }

    // ── softmax sample tests ──────────────────────────────────────────

    #[test]
    fn softmax_sample_deterministic_same_seed() {
        let a = cpu_softmax_sample(&sample_logits(), 42);
        let b = cpu_softmax_sample(&sample_logits(), 42);
        assert_eq!(a, b);
    }

    #[test]
    fn softmax_sample_valid_index() {
        let idx = cpu_softmax_sample(&sample_logits(), 123);
        assert!((idx as usize) < sample_logits().len());
    }

    #[test]
    fn softmax_sample_dominated_picks_dominant() {
        // With one overwhelmingly large logit, sampling should almost
        // certainly pick it.
        let logits = dominated_logits();
        let counts: usize =
            (0..100).map(|s| cpu_softmax_sample(&logits, s)).filter(|&t| t == 2).count();
        assert!(counts > 95, "expected dominant pick, got {counts}/100");
    }

    #[test]
    fn softmax_sample_on_empty_returns_zero() {
        assert_eq!(cpu_softmax_sample(&[], 1), 0);
    }

    // ── stop condition tests ──────────────────────────────────────────

    #[test]
    fn stop_on_stop_token() {
        let config =
            GenerationConfig { max_tokens: 100, stop_tokens: vec![5], ..Default::default() };
        assert_eq!(cpu_check_stop_condition(5, 0, &config), Some(StopReason::StopToken(5)));
    }

    #[test]
    fn stop_on_eos_token_zero() {
        let config = GenerationConfig::default();
        assert_eq!(cpu_check_stop_condition(0, 0, &config), Some(StopReason::EndOfSequence));
    }

    #[test]
    fn stop_on_max_tokens() {
        let config = GenerationConfig { max_tokens: 5, ..Default::default() };
        assert_eq!(cpu_check_stop_condition(1, 4, &config), Some(StopReason::MaxTokens));
    }

    #[test]
    fn no_stop_mid_generation() {
        let config = GenerationConfig { max_tokens: 100, ..Default::default() };
        assert_eq!(cpu_check_stop_condition(1, 0, &config), None);
    }

    #[test]
    fn stop_token_zero_in_stop_list_prevents_eos() {
        let config = GenerationConfig { stop_tokens: vec![0], ..Default::default() };
        // Token 0 is in the stop list, so it should be StopToken, not
        // EOS.
        assert_eq!(cpu_check_stop_condition(0, 0, &config), Some(StopReason::StopToken(0)));
    }

    // ── config validation tests ───────────────────────────────────────

    #[test]
    fn valid_default_config() {
        assert!(validate_config(&GenerationConfig::default()).is_ok());
    }

    #[test]
    fn invalid_max_tokens_zero() {
        let cfg = GenerationConfig { max_tokens: 0, ..Default::default() };
        assert!(matches!(validate_config(&cfg), Err(GenerationError::InvalidConfig(_))));
    }

    #[test]
    fn invalid_negative_temperature() {
        let cfg = GenerationConfig { temperature: -1.0, ..Default::default() };
        assert!(matches!(validate_config(&cfg), Err(GenerationError::InvalidConfig(_))));
    }

    #[test]
    fn invalid_repetition_penalty_zero() {
        let cfg = GenerationConfig { repetition_penalty: 0.0, ..Default::default() };
        assert!(matches!(validate_config(&cfg), Err(GenerationError::InvalidConfig(_))));
    }

    #[test]
    fn invalid_top_k_zero() {
        let cfg = GenerationConfig { top_k: Some(0), ..Default::default() };
        assert!(matches!(validate_config(&cfg), Err(GenerationError::InvalidConfig(_))));
    }

    #[test]
    fn invalid_top_p_out_of_range() {
        let cfg = GenerationConfig { top_p: Some(1.5), ..Default::default() };
        assert!(matches!(validate_config(&cfg), Err(GenerationError::InvalidConfig(_))));
    }

    // ── cpu_generate_token tests ──────────────────────────────────────

    #[test]
    fn generate_token_greedy_near_zero_temp() {
        let config = GenerationConfig { temperature: 0.0, ..Default::default() };
        let state = GenerationState::default();
        let token = cpu_generate_token(&sample_logits(), &config, &state);
        assert_eq!(token, 4);
    }

    #[test]
    fn generate_token_with_top_k() {
        let config = GenerationConfig { top_k: Some(2), seed: Some(42), ..Default::default() };
        let state = GenerationState::default();
        let token = cpu_generate_token(&sample_logits(), &config, &state);
        // Must be one of the top-2 indices (3 or 4).
        assert!(token == 3 || token == 4, "got {token}");
    }

    #[test]
    fn generate_token_dominated_always_picks_dominant() {
        let config = GenerationConfig::default();
        let state = GenerationState::default();
        for seed in 0..50u64 {
            let cfg = GenerationConfig { seed: Some(seed), ..config.clone() };
            let t = cpu_generate_token(&dominated_logits(), &cfg, &state);
            assert_eq!(t, 2, "seed {seed} produced {t}");
        }
    }

    // ── full generation loop tests ────────────────────────────────────

    #[test]
    fn generate_sequence_greedy_deterministic() {
        // Logits fn always returns [0,1,2,3,4] → greedy always picks 4.
        let config = GenerationConfig {
            max_tokens: 5,
            temperature: 0.0,
            stop_tokens: vec![],
            ..Default::default()
        };
        let result = cpu_generate_sequence(|_| vec![0.0, 1.0, 2.0, 3.0, 4.0], &config);
        // Token 0 causes EOS; so the logits fn never returns token 0.
        // Greedy picks index 4 every time; stop at max_tokens.
        assert_eq!(result.tokens, vec![4, 4, 4, 4, 4]);
        assert_eq!(result.stop_reason, StopReason::MaxTokens);
    }

    #[test]
    fn generate_sequence_stops_on_stop_token() {
        // Return stop token (7) after 3 steps.
        let config = GenerationConfig {
            max_tokens: 100,
            temperature: 0.0,
            stop_tokens: vec![7],
            ..Default::default()
        };
        let mut call = 0u32;
        let result = cpu_generate_sequence(
            |_| {
                call += 1;
                if call <= 3 {
                    // Logit 5 at index 5 → greedy picks 5.
                    vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0]
                } else {
                    // Logit 10 at index 7 → greedy picks 7 (stop).
                    vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0]
                }
            },
            &config,
        );
        assert_eq!(result.stop_reason, StopReason::StopToken(7));
        assert!(result.tokens.len() <= 5);
    }

    #[test]
    fn generate_sequence_stats_positive() {
        let config = GenerationConfig { max_tokens: 3, temperature: 0.0, ..Default::default() };
        let result = cpu_generate_sequence(|_| vec![0.0, 1.0, 2.0, 3.0, 4.0], &config);
        assert!(result.stats.total_tokens > 0);
        assert!(result.stats.tokens_per_second >= 0.0);
    }

    #[test]
    fn generate_sequence_empty_logits_returns_error() {
        let config = GenerationConfig { max_tokens: 5, ..Default::default() };
        let result = cpu_generate_sequence(|_| vec![], &config);
        assert!(matches!(result.stop_reason, StopReason::Error(_)));
    }

    #[test]
    fn generate_sequence_invalid_config_returns_error() {
        let config = GenerationConfig { max_tokens: 0, ..Default::default() };
        let result = cpu_generate_sequence(|_| vec![1.0, 2.0], &config);
        assert!(matches!(result.stop_reason, StopReason::Error(_)));
    }

    // ── xorshift determinism ──────────────────────────────────────────

    #[test]
    fn xorshift_deterministic() {
        let mut s1 = 42u64;
        let mut s2 = 42u64;
        let a: Vec<f32> = (0..10).map(|_| xorshift64(&mut s1)).collect();
        let b: Vec<f32> = (0..10).map(|_| xorshift64(&mut s2)).collect();
        assert_eq!(a, b);
    }

    #[test]
    fn xorshift_in_range() {
        let mut s = 12345u64;
        for _ in 0..1000 {
            let v = xorshift64(&mut s);
            assert!((0.0..=1.0).contains(&v), "out of range: {v}");
        }
    }

    // ── property-style tests ──────────────────────────────────────────

    #[test]
    fn property_temperature_preserves_argmax_when_dominated() {
        let logits = dominated_logits();
        for temp_x10 in 1..=30u32 {
            let temp = temp_x10 as f32 / 10.0;
            let mut buf = logits.clone();
            cpu_apply_temperature(&mut buf, temp);
            assert_eq!(cpu_greedy_decode(&buf), 2, "argmax changed at temp={temp}");
        }
    }

    #[test]
    fn property_top_k_monotonically_reduces_finite() {
        let logits = sample_logits();
        let mut prev_finite = logits.len();
        for k in (1..=5).rev() {
            let mut buf = logits.clone();
            cpu_top_k_filter(&mut buf, k);
            let finite = buf.iter().filter(|v| v.is_finite()).count();
            assert!(finite <= prev_finite);
            prev_finite = finite;
        }
    }

    // ── SamplingMethod / StopReason / GenerationError Display ─────────

    #[test]
    fn sampling_method_variants() {
        let _g = SamplingMethod::Greedy;
        let _t = SamplingMethod::Temperature(0.8);
        let _k = SamplingMethod::TopK(10);
        let _p = SamplingMethod::TopP(0.9);
        let _kp = SamplingMethod::TopKP(10, 0.9);
        assert_ne!(_g, _t);
    }

    #[test]
    fn stop_reason_display() {
        let s = StopReason::MaxTokens.to_string();
        assert!(s.contains("max_tokens"));
    }

    #[test]
    fn generation_error_display() {
        let e = GenerationError::EmptyPrompt;
        assert!(e.to_string().contains("empty"));
    }

    #[test]
    fn generation_error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(GenerationError::VocabTooSmall);
        assert!(e.to_string().contains("vocab"));
    }

    // ── edge cases ────────────────────────────────────────────────────

    #[test]
    fn single_element_logits_greedy() {
        assert_eq!(cpu_greedy_decode(&[42.0]), 0);
    }

    #[test]
    fn single_element_logits_sample() {
        assert_eq!(cpu_softmax_sample(&[42.0], 7), 0);
    }

    #[test]
    fn top_p_negative_range_error() {
        let cfg = GenerationConfig { top_p: Some(-0.1), ..Default::default() };
        assert!(validate_config(&cfg).is_err());
    }

    #[test]
    fn generation_state_default() {
        let s = GenerationState::default();
        assert!(s.generated_tokens.is_empty());
        assert_eq!(s.current_position, 0);
        assert_eq!(s.total_time_us, 0);
    }

    #[test]
    fn generation_config_default_is_valid() {
        assert!(validate_config(&GenerationConfig::default()).is_ok());
    }

    #[test]
    fn numerical_error_variant() {
        let e = GenerationError::NumericalError("nan".into());
        assert!(e.to_string().contains("nan"));
    }

    #[test]
    fn max_tokens_exceeded_variant() {
        let e = GenerationError::MaxTokensExceeded;
        assert!(e.to_string().contains("max tokens"));
    }
}
