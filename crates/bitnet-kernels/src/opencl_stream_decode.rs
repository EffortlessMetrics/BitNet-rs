//! Streaming token decoder for OpenCL-accelerated inference on Intel Arc A770.
//!
//! Processes logits incrementally and yields tokens one at a time with
//! backpressure support. Provides CPU reference implementations of temperature
//! scaling, top-k, top-p (nucleus), and repetition penalty sampling.

use std::fmt;
use std::time::Instant;

// ── StreamConfig ───────────────────────────────────────────────────

/// Configuration for a streaming decode session.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Maximum number of tokens to generate.
    pub max_tokens: usize,
    /// Softmax temperature (0 → argmax, higher → more random).
    pub temperature: f32,
    /// Keep only the top-k highest-probability tokens.
    pub top_k: usize,
    /// Nucleus sampling: keep smallest set whose cumulative prob ≥ p.
    pub top_p: f32,
    /// Multiplicative penalty applied to previously generated tokens.
    pub repetition_penalty: f32,
    /// Token IDs that signal end of generation.
    pub stop_tokens: Vec<u32>,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            max_tokens: 128,
            temperature: 1.0,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.0,
            stop_tokens: Vec::new(),
        }
    }
}

// ── TokenEvent ─────────────────────────────────────────────────────

/// A single token emitted by the decoder.
#[derive(Debug, Clone)]
pub struct TokenEvent {
    /// Sampled token ID.
    pub token_id: u32,
    /// Probability of this token after sampling transforms.
    pub token_prob: f32,
    /// Running cumulative probability across the sequence.
    pub cumulative_prob: f32,
    /// Decode latency for this token in microseconds.
    pub latency_us: u64,
    /// Zero-based position in the generated sequence.
    pub position: usize,
}

// ── StreamState ────────────────────────────────────────────────────

/// Current state of the streaming decoder.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamState {
    Idle,
    Prefilling,
    Generating,
    Paused,
    Finished(FinishReason),
    Error(String),
}

impl fmt::Display for StreamState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Idle => write!(f, "Idle"),
            Self::Prefilling => write!(f, "Prefilling"),
            Self::Generating => write!(f, "Generating"),
            Self::Paused => write!(f, "Paused"),
            Self::Finished(reason) => write!(f, "Finished({reason})"),
            Self::Error(msg) => write!(f, "Error({msg})"),
        }
    }
}

// ── FinishReason ───────────────────────────────────────────────────

/// Why the stream finished generating tokens.
#[derive(Debug, Clone, PartialEq)]
pub enum FinishReason {
    MaxTokens,
    StopToken(u32),
    EosToken,
    UserAbort,
}

impl fmt::Display for FinishReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MaxTokens => write!(f, "max_tokens"),
            Self::StopToken(id) => write!(f, "stop_token({id})"),
            Self::EosToken => write!(f, "eos_token"),
            Self::UserAbort => write!(f, "user_abort"),
        }
    }
}

// ── StreamStats ────────────────────────────────────────────────────

/// Aggregate statistics for a streaming decode session.
#[derive(Debug, Clone, Default)]
pub struct StreamStats {
    pub total_tokens: usize,
    pub prefill_time_us: u64,
    pub total_decode_time_us: u64,
    pub tokens_per_second: f64,
    pub time_to_first_token_us: u64,
}

// ── BackpressurePolicy ─────────────────────────────────────────────

/// Policy when the consumer cannot keep up with token production.
#[derive(Debug, Clone, PartialEq)]
pub enum BackpressurePolicy {
    /// Drop excess tokens silently.
    Drop,
    /// Buffer up to `n` tokens before applying back-pressure.
    Buffer(usize),
    /// Block the producer until the consumer is ready.
    Block,
}

// ── StreamError ────────────────────────────────────────────────────

/// Errors returned by streaming decode operations.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamError {
    NotStarted,
    AlreadyFinished,
    InvalidLogits,
    Aborted,
    BufferOverflow,
}

impl fmt::Display for StreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotStarted => write!(f, "stream not started"),
            Self::AlreadyFinished => write!(f, "stream already finished"),
            Self::InvalidLogits => write!(f, "invalid logits (empty or non-finite)"),
            Self::Aborted => write!(f, "stream aborted"),
            Self::BufferOverflow => write!(f, "backpressure buffer overflow"),
        }
    }
}

impl std::error::Error for StreamError {}

// ── StreamDecoder ──────────────────────────────────────────────────

/// Streaming token decoder with backpressure support.
pub struct StreamDecoder {
    pub config: StreamConfig,
    pub state: StreamState,
    pub tokens: Vec<TokenEvent>,
    pub stats: StreamStats,
    pub rng_state: u64,
    /// Wall-clock anchor used for latency measurement.
    decode_start: Option<Instant>,
    /// Timestamp of the last token emission.
    last_token_time: Option<Instant>,
}

// ── Public API (CPU reference implementations) ─────────────────────

/// Create a new streaming decoder with the given configuration.
pub fn create_stream_decoder(config: StreamConfig) -> StreamDecoder {
    StreamDecoder {
        config,
        state: StreamState::Idle,
        tokens: Vec::new(),
        stats: StreamStats::default(),
        rng_state: 0,
        decode_start: None,
        last_token_time: None,
    }
}

/// Transition the decoder into the prefilling state.
pub fn cpu_start_stream(decoder: &mut StreamDecoder) {
    decoder.state = StreamState::Prefilling;
    decoder.decode_start = Some(Instant::now());
    decoder.last_token_time = decoder.decode_start;
}

/// Process a logits vector and return the next sampled token.
pub fn cpu_feed_logits(
    decoder: &mut StreamDecoder,
    logits: &[f32],
) -> Result<TokenEvent, StreamError> {
    // State validation
    match &decoder.state {
        StreamState::Idle => return Err(StreamError::NotStarted),
        StreamState::Finished(_) => return Err(StreamError::AlreadyFinished),
        StreamState::Error(_) => return Err(StreamError::Aborted),
        _ => {}
    }

    if logits.is_empty() || logits.iter().any(|v| v.is_nan()) {
        return Err(StreamError::InvalidLogits);
    }

    let token_start = Instant::now();

    // Transition prefilling → generating on first logits
    if decoder.state == StreamState::Prefilling {
        let prefill_elapsed = decoder
            .decode_start
            .map(|t| t.elapsed().as_micros() as u64)
            .unwrap_or(0);
        decoder.stats.prefill_time_us = prefill_elapsed;
        decoder.state = StreamState::Generating;
    }

    // Work on a mutable copy of logits
    let mut work = logits.to_vec();

    // Apply sampling transforms
    cpu_apply_repetition_penalty(
        &mut work,
        &decoder.tokens.iter().map(|t| t.token_id).collect::<Vec<_>>(),
        decoder.config.repetition_penalty,
    );
    cpu_apply_temperature(&mut work, decoder.config.temperature);
    cpu_apply_top_k(&mut work, decoder.config.top_k);
    cpu_apply_top_p(&mut work, decoder.config.top_p);

    // Sample
    let (token_id, token_prob) = cpu_sample_token(&work, &mut decoder.rng_state);

    let cumulative_prob = decoder
        .tokens
        .last()
        .map(|t| t.cumulative_prob)
        .unwrap_or(0.0)
        + token_prob;

    let latency_us = token_start.elapsed().as_micros() as u64;
    let position = decoder.tokens.len();

    let event = TokenEvent { token_id, token_prob, cumulative_prob, latency_us, position };

    // Record time-to-first-token
    if decoder.tokens.is_empty() {
        decoder.stats.time_to_first_token_us = decoder
            .decode_start
            .map(|t| t.elapsed().as_micros() as u64)
            .unwrap_or(0);
    }

    decoder.tokens.push(event.clone());
    decoder.last_token_time = Some(Instant::now());

    // Update running stats
    decoder.stats.total_tokens = decoder.tokens.len();
    let total_us = decoder
        .decode_start
        .map(|t| t.elapsed().as_micros() as u64)
        .unwrap_or(1);
    decoder.stats.total_decode_time_us = total_us;
    decoder.stats.tokens_per_second = if total_us > 0 {
        decoder.stats.total_tokens as f64 / (total_us as f64 / 1_000_000.0)
    } else {
        0.0
    };

    // Check stop conditions
    if let Some(reason) = cpu_check_stop_condition(decoder, token_id) {
        decoder.state = StreamState::Finished(reason);
    }

    Ok(event)
}

/// Scale logits by `1/temperature`. Temperature 0 converts to argmax (sets
/// the maximum logit to a large value and zeroes out the rest).
pub fn cpu_apply_temperature(logits: &mut [f32], temperature: f32) {
    if logits.is_empty() {
        return;
    }
    if temperature <= 0.0 {
        // Argmax: keep only the maximum value
        let max_idx = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        for (i, v) in logits.iter_mut().enumerate() {
            *v = if i == max_idx { 1e9 } else { f32::NEG_INFINITY };
        }
        return;
    }
    for v in logits.iter_mut() {
        *v /= temperature;
    }
}

/// Zero out all but the top-k highest logits. If k ≥ vocab_size this is a no-op.
pub fn cpu_apply_top_k(logits: &mut [f32], k: usize) {
    if k == 0 || k >= logits.len() {
        return;
    }

    // Find the k-th largest value via a partial sort (selection)
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_unstable_by(|&a, &b| {
        logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal)
    });

    for &idx in &indices[k..] {
        logits[idx] = f32::NEG_INFINITY;
    }
}

/// Nucleus sampling: keep the smallest set of tokens whose cumulative
/// probability mass ≥ `p`, zeroing out the rest.
pub fn cpu_apply_top_p(logits: &mut [f32], p: f32) {
    if p >= 1.0 {
        return;
    }

    // Convert to probabilities via softmax
    let max_val = logits
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    if max_val == f32::NEG_INFINITY {
        return;
    }

    let mut probs: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let exp = (v - max_val).exp();
            (i, if exp.is_finite() { exp } else { 0.0 })
        })
        .collect();

    let sum: f32 = probs.iter().map(|(_, v)| v).sum();
    if sum <= 0.0 {
        return;
    }
    for item in &mut probs {
        item.1 /= sum;
    }

    // Sort descending by probability
    probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumulative = 0.0f32;
    let mut keep = vec![false; logits.len()];
    for &(idx, prob) in &probs {
        keep[idx] = true;
        cumulative += prob;
        if cumulative >= p {
            break;
        }
    }

    for (i, v) in logits.iter_mut().enumerate() {
        if !keep[i] {
            *v = f32::NEG_INFINITY;
        }
    }
}

/// Apply a multiplicative repetition penalty to previously generated token IDs.
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

/// Sample a token from the (possibly transformed) logits using a simple
/// xorshift64 PRNG. Returns `(token_id, probability)`.
pub fn cpu_sample_token(logits: &[f32], rng_state: &mut u64) -> (u32, f32) {
    if logits.is_empty() {
        return (0, 0.0);
    }

    // Softmax
    let max_val = logits
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);

    let mut probs: Vec<f32> = logits
        .iter()
        .map(|&v| {
            let e = (v - max_val).exp();
            if e.is_finite() { e } else { 0.0 }
        })
        .collect();

    let sum: f32 = probs.iter().sum();
    if sum <= 0.0 {
        // Degenerate: return the first finite logit position
        let idx = logits
            .iter()
            .position(|v| v.is_finite())
            .unwrap_or(0);
        return (idx as u32, 1.0);
    }
    for p in &mut probs {
        *p /= sum;
    }

    // xorshift64 PRNG
    let r = xorshift64(rng_state);
    let threshold = (r as f64) / (u64::MAX as f64);

    let mut cumulative = 0.0f64;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p as f64;
        if cumulative >= threshold {
            return (i as u32, p);
        }
    }

    // Fallback to last token
    let last = probs.len() - 1;
    (last as u32, probs[last])
}

/// Check whether the decoder should stop generating.
pub fn cpu_check_stop_condition(
    decoder: &StreamDecoder,
    token: u32,
) -> Option<FinishReason> {
    if decoder.config.stop_tokens.contains(&token) {
        return Some(FinishReason::StopToken(token));
    }
    if decoder.tokens.len() >= decoder.config.max_tokens {
        return Some(FinishReason::MaxTokens);
    }
    None
}

/// Pause the stream (only valid while generating).
pub fn cpu_pause_stream(decoder: &mut StreamDecoder) -> Result<(), StreamError> {
    match &decoder.state {
        StreamState::Generating => {
            decoder.state = StreamState::Paused;
            Ok(())
        }
        StreamState::Idle | StreamState::Prefilling => Err(StreamError::NotStarted),
        StreamState::Finished(_) => Err(StreamError::AlreadyFinished),
        StreamState::Error(_) | StreamState::Paused => Err(StreamError::Aborted),
    }
}

/// Resume a paused stream.
pub fn cpu_resume_stream(decoder: &mut StreamDecoder) -> Result<(), StreamError> {
    match &decoder.state {
        StreamState::Paused => {
            decoder.state = StreamState::Generating;
            Ok(())
        }
        StreamState::Idle | StreamState::Prefilling => Err(StreamError::NotStarted),
        StreamState::Finished(_) => Err(StreamError::AlreadyFinished),
        _ => Err(StreamError::Aborted),
    }
}

/// Abort the stream and return all tokens generated so far.
pub fn cpu_abort_stream(
    decoder: &mut StreamDecoder,
) -> Result<Vec<TokenEvent>, StreamError> {
    match &decoder.state {
        StreamState::Idle => return Err(StreamError::NotStarted),
        StreamState::Finished(_) => return Err(StreamError::AlreadyFinished),
        _ => {}
    }
    decoder.state = StreamState::Finished(FinishReason::UserAbort);
    Ok(decoder.tokens.clone())
}

/// Return a snapshot of the current stream statistics.
pub fn cpu_get_stream_stats(decoder: &StreamDecoder) -> StreamStats {
    decoder.stats.clone()
}

/// Human-readable status string for the decoder.
pub fn format_stream_status(decoder: &StreamDecoder) -> String {
    format!(
        "state={} tokens={} tps={:.1}",
        decoder.state, decoder.stats.total_tokens, decoder.stats.tokens_per_second,
    )
}

// ── Internal helpers ───────────────────────────────────────────────

/// xorshift64 PRNG — simple, fast, deterministic.
fn xorshift64(state: &mut u64) -> u64 {
    if *state == 0 {
        *state = 0x5EED_CAFE_BABE_D00D;
    }
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Convert logits to a probability distribution (softmax).
#[cfg(test)]
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum <= 0.0 {
        return vec![0.0; logits.len()];
    }
    exps.iter().map(|e| e / sum).collect()
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> StreamConfig {
        StreamConfig { max_tokens: 64, ..StreamConfig::default() }
    }

    fn uniform_logits(n: usize) -> Vec<f32> {
        vec![1.0; n]
    }

    #[allow(dead_code)]
    fn peaked_logits(n: usize, peak: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; n];
        if peak < n {
            v[peak] = 10.0;
        }
        v
    }

    // ── Construction / config ──────────────────────────────────────

    #[test]
    fn test_create_decoder_default() {
        let dec = create_stream_decoder(StreamConfig::default());
        assert_eq!(dec.state, StreamState::Idle);
        assert!(dec.tokens.is_empty());
    }

    #[test]
    fn test_create_decoder_custom_config() {
        let cfg = StreamConfig { max_tokens: 32, temperature: 0.5, ..default_config() };
        let dec = create_stream_decoder(cfg);
        assert_eq!(dec.config.max_tokens, 32);
        assert!((dec.config.temperature - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_create_decoder_with_stop_tokens() {
        let cfg = StreamConfig { stop_tokens: vec![2, 50256], ..default_config() };
        let dec = create_stream_decoder(cfg);
        assert_eq!(dec.config.stop_tokens, vec![2, 50256]);
    }

    // ── State transitions ──────────────────────────────────────────

    #[test]
    fn test_start_stream_transitions_to_prefilling() {
        let mut dec = create_stream_decoder(default_config());
        cpu_start_stream(&mut dec);
        assert_eq!(dec.state, StreamState::Prefilling);
    }

    #[test]
    fn test_first_feed_transitions_to_generating() {
        let mut dec = create_stream_decoder(default_config());
        cpu_start_stream(&mut dec);
        let _ = cpu_feed_logits(&mut dec, &uniform_logits(100));
        assert_eq!(dec.state, StreamState::Generating);
    }

    #[test]
    fn test_feed_without_start_returns_not_started() {
        let mut dec = create_stream_decoder(default_config());
        let result = cpu_feed_logits(&mut dec, &uniform_logits(10));
        assert_eq!(result.unwrap_err(), StreamError::NotStarted);
    }

    #[test]
    fn test_feed_after_finish_returns_already_finished() {
        let cfg = StreamConfig { max_tokens: 1, ..default_config() };
        let mut dec = create_stream_decoder(cfg);
        cpu_start_stream(&mut dec);
        let _ = cpu_feed_logits(&mut dec, &uniform_logits(10));
        let result = cpu_feed_logits(&mut dec, &uniform_logits(10));
        assert_eq!(result.unwrap_err(), StreamError::AlreadyFinished);
    }

    // ── Feed logits / token production ─────────────────────────────

    #[test]
    fn test_feed_logits_produces_token() {
        let mut dec = create_stream_decoder(default_config());
        cpu_start_stream(&mut dec);
        let event = cpu_feed_logits(&mut dec, &uniform_logits(100)).unwrap();
        assert!(event.token_id < 100);
        assert!(event.token_prob > 0.0);
    }

    #[test]
    fn test_feed_logits_increments_position() {
        let mut dec = create_stream_decoder(default_config());
        cpu_start_stream(&mut dec);
        for expected_pos in 0..5 {
            let event = cpu_feed_logits(&mut dec, &uniform_logits(10)).unwrap();
            assert_eq!(event.position, expected_pos);
        }
    }

    #[test]
    fn test_feed_logits_records_latency() {
        let mut dec = create_stream_decoder(default_config());
        cpu_start_stream(&mut dec);
        let event = cpu_feed_logits(&mut dec, &uniform_logits(10)).unwrap();
        // Latency should be non-negative (could be 0 on fast machines)
        assert!(event.latency_us < 10_000_000); // sanity: < 10s
    }

    #[test]
    fn test_feed_empty_logits_returns_error() {
        let mut dec = create_stream_decoder(default_config());
        cpu_start_stream(&mut dec);
        assert_eq!(
            cpu_feed_logits(&mut dec, &[]).unwrap_err(),
            StreamError::InvalidLogits
        );
    }

    #[test]
    fn test_feed_nan_logits_returns_error() {
        let mut dec = create_stream_decoder(default_config());
        cpu_start_stream(&mut dec);
        assert_eq!(
            cpu_feed_logits(&mut dec, &[1.0, f32::NAN, 2.0]).unwrap_err(),
            StreamError::InvalidLogits
        );
    }

    // ── Temperature ────────────────────────────────────────────────

    #[test]
    fn test_temperature_scaling() {
        let mut low_temp = vec![1.0, 2.0, 3.0];
        let mut high_temp = low_temp.clone();
        cpu_apply_temperature(&mut low_temp, 0.5);
        cpu_apply_temperature(&mut high_temp, 2.0);

        // Low temp amplifies differences → higher max prob
        let low_probs = softmax(&low_temp);
        let high_probs = softmax(&high_temp);
        assert!(low_probs.iter().copied().fold(0.0f32, f32::max)
            > high_probs.iter().copied().fold(0.0f32, f32::max));
    }

    #[test]
    fn test_temperature_high_more_uniform() {
        let base = vec![1.0, 2.0, 3.0, 4.0];
        let mut t1 = base.clone();
        let mut t10 = base.clone();
        cpu_apply_temperature(&mut t1, 1.0);
        cpu_apply_temperature(&mut t10, 10.0);

        let p1 = softmax(&t1);
        let p10 = softmax(&t10);

        // Higher temp → std-dev of probs should be smaller (more uniform)
        let mean1: f32 = p1.iter().sum::<f32>() / p1.len() as f32;
        let mean10: f32 = p10.iter().sum::<f32>() / p10.len() as f32;
        let var1: f32 = p1.iter().map(|x| (x - mean1).powi(2)).sum::<f32>();
        let var10: f32 = p10.iter().map(|x| (x - mean10).powi(2)).sum::<f32>();
        assert!(var10 < var1);
    }

    #[test]
    fn test_temperature_zero_gives_argmax() {
        let mut logits = vec![1.0, 5.0, 3.0, 2.0];
        cpu_apply_temperature(&mut logits, 0.0);
        let probs = softmax(&logits);
        // Token 1 (value 5.0) should dominate
        assert!(probs[1] > 0.99);
    }

    // ── Top-k ──────────────────────────────────────────────────────

    #[test]
    fn test_top_k_filters() {
        let mut logits = vec![1.0, 5.0, 3.0, 4.0, 2.0];
        cpu_apply_top_k(&mut logits, 2);
        let non_neg_inf = logits.iter().filter(|v| v.is_finite()).count();
        assert_eq!(non_neg_inf, 2);
    }

    #[test]
    fn test_top_k_keeps_highest() {
        let mut logits = vec![1.0, 5.0, 3.0, 4.0, 2.0];
        cpu_apply_top_k(&mut logits, 2);
        assert!(logits[1].is_finite()); // 5.0
        assert!(logits[3].is_finite()); // 4.0
    }

    #[test]
    fn test_top_k_larger_than_vocab_is_noop() {
        let original = vec![1.0, 2.0, 3.0];
        let mut logits = original.clone();
        cpu_apply_top_k(&mut logits, 100);
        assert_eq!(logits, original);
    }

    #[test]
    fn test_top_k_one() {
        let mut logits = vec![1.0, 5.0, 3.0];
        cpu_apply_top_k(&mut logits, 1);
        let finite_count = logits.iter().filter(|v| v.is_finite()).count();
        assert_eq!(finite_count, 1);
        assert!(logits[1].is_finite()); // max was at idx 1
    }

    // ── Top-p (nucleus) ────────────────────────────────────────────

    #[test]
    fn test_top_p_filters_low_prob() {
        // Token 1 dominates (exp(10) >> exp(0))
        let mut logits = vec![0.0, 10.0, 0.0, 0.0];
        cpu_apply_top_p(&mut logits, 0.5);
        // Token 1 alone exceeds 0.5 cumulative prob
        assert!(logits[1].is_finite());
    }

    #[test]
    fn test_top_p_one_is_noop() {
        let original = vec![1.0, 2.0, 3.0];
        let mut logits = original.clone();
        cpu_apply_top_p(&mut logits, 1.0);
        assert_eq!(logits, original);
    }

    #[test]
    fn test_top_p_very_small_keeps_at_least_one() {
        let mut logits = vec![1.0, 1.0, 1.0, 1.0];
        cpu_apply_top_p(&mut logits, 0.01);
        let finite = logits.iter().filter(|v| v.is_finite()).count();
        assert!(finite >= 1);
    }

    // ── Repetition penalty ─────────────────────────────────────────

    #[test]
    fn test_repetition_penalty_reduces_positive_logit() {
        let mut logits = vec![5.0, 5.0, 5.0];
        cpu_apply_repetition_penalty(&mut logits, &[0, 1], 2.0);
        assert!(logits[0] < 5.0);
        assert!(logits[1] < 5.0);
        assert!((logits[2] - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_repetition_penalty_amplifies_negative_logit() {
        let mut logits = vec![-2.0, 1.0];
        cpu_apply_repetition_penalty(&mut logits, &[0], 2.0);
        assert!(logits[0] < -2.0); // -2 * 2 = -4
    }

    #[test]
    fn test_repetition_penalty_one_is_noop() {
        let original = vec![3.0, 4.0, 5.0];
        let mut logits = original.clone();
        cpu_apply_repetition_penalty(&mut logits, &[0, 1, 2], 1.0);
        assert_eq!(logits, original);
    }

    #[test]
    fn test_repetition_penalty_out_of_range_token_ignored() {
        let mut logits = vec![1.0, 2.0];
        cpu_apply_repetition_penalty(&mut logits, &[999], 2.0);
        assert!((logits[0] - 1.0).abs() < f32::EPSILON);
        assert!((logits[1] - 2.0).abs() < f32::EPSILON);
    }

    // ── Stop conditions ────────────────────────────────────────────

    #[test]
    fn test_stop_token_finishes_stream() {
        let cfg = StreamConfig { stop_tokens: vec![42], ..default_config() };
        let mut dec = create_stream_decoder(cfg);
        cpu_start_stream(&mut dec);

        // Feed logits where token 42 will be sampled (peaked)
        let mut logits = vec![0.0; 100];
        logits[42] = 100.0;
        let _ = cpu_feed_logits(&mut dec, &logits);
        assert_eq!(dec.state, StreamState::Finished(FinishReason::StopToken(42)));
    }

    #[test]
    fn test_max_tokens_finishes_stream() {
        let cfg = StreamConfig { max_tokens: 3, ..default_config() };
        let mut dec = create_stream_decoder(cfg);
        cpu_start_stream(&mut dec);
        for _ in 0..3 {
            let _ = cpu_feed_logits(&mut dec, &uniform_logits(10));
        }
        assert!(matches!(dec.state, StreamState::Finished(FinishReason::MaxTokens)));
    }

    #[test]
    fn test_stop_token_finish_reason_carries_token_id() {
        let cfg = StreamConfig { stop_tokens: vec![7], max_tokens: 100, ..default_config() };
        let mut dec = create_stream_decoder(cfg);
        cpu_start_stream(&mut dec);
        let mut logits = vec![0.0; 20];
        logits[7] = 100.0;
        let _ = cpu_feed_logits(&mut dec, &logits);
        assert_eq!(dec.state, StreamState::Finished(FinishReason::StopToken(7)));
    }

    // ── Pause / resume ─────────────────────────────────────────────

    #[test]
    fn test_pause_resume_roundtrip() {
        let mut dec = create_stream_decoder(default_config());
        cpu_start_stream(&mut dec);
        let _ = cpu_feed_logits(&mut dec, &uniform_logits(10));
        assert!(cpu_pause_stream(&mut dec).is_ok());
        assert_eq!(dec.state, StreamState::Paused);
        assert!(cpu_resume_stream(&mut dec).is_ok());
        assert_eq!(dec.state, StreamState::Generating);
    }

    #[test]
    fn test_pause_idle_returns_error() {
        let mut dec = create_stream_decoder(default_config());
        assert!(cpu_pause_stream(&mut dec).is_err());
    }

    #[test]
    fn test_resume_without_pause_returns_error() {
        let mut dec = create_stream_decoder(default_config());
        cpu_start_stream(&mut dec);
        let _ = cpu_feed_logits(&mut dec, &uniform_logits(10));
        // Not paused — resume should fail
        assert!(cpu_resume_stream(&mut dec).is_err());
    }

    // ── Abort ──────────────────────────────────────────────────────

    #[test]
    fn test_abort_returns_generated_tokens() {
        let mut dec = create_stream_decoder(default_config());
        cpu_start_stream(&mut dec);
        for _ in 0..5 {
            let _ = cpu_feed_logits(&mut dec, &uniform_logits(10));
        }
        let tokens = cpu_abort_stream(&mut dec).unwrap();
        assert_eq!(tokens.len(), 5);
        assert_eq!(dec.state, StreamState::Finished(FinishReason::UserAbort));
    }

    #[test]
    fn test_abort_idle_returns_error() {
        let mut dec = create_stream_decoder(default_config());
        assert_eq!(cpu_abort_stream(&mut dec).unwrap_err(), StreamError::NotStarted);
    }

    #[test]
    fn test_abort_finished_returns_error() {
        let cfg = StreamConfig { max_tokens: 1, ..default_config() };
        let mut dec = create_stream_decoder(cfg);
        cpu_start_stream(&mut dec);
        let _ = cpu_feed_logits(&mut dec, &uniform_logits(10));
        assert_eq!(
            cpu_abort_stream(&mut dec).unwrap_err(),
            StreamError::AlreadyFinished,
        );
    }

    // ── Stats ──────────────────────────────────────────────────────

    #[test]
    fn test_stats_token_count() {
        let mut dec = create_stream_decoder(default_config());
        cpu_start_stream(&mut dec);
        for _ in 0..10 {
            let _ = cpu_feed_logits(&mut dec, &uniform_logits(20));
        }
        let stats = cpu_get_stream_stats(&dec);
        assert_eq!(stats.total_tokens, 10);
    }

    #[test]
    fn test_stats_tps_positive() {
        let mut dec = create_stream_decoder(default_config());
        cpu_start_stream(&mut dec);
        for _ in 0..5 {
            let _ = cpu_feed_logits(&mut dec, &uniform_logits(20));
        }
        let stats = cpu_get_stream_stats(&dec);
        assert!(stats.tokens_per_second > 0.0);
    }

    #[test]
    fn test_stats_time_to_first_token() {
        let mut dec = create_stream_decoder(default_config());
        cpu_start_stream(&mut dec);
        let _ = cpu_feed_logits(&mut dec, &uniform_logits(10));
        let stats = cpu_get_stream_stats(&dec);
        // TTFT should be recorded (≥ 0)
        assert!(stats.time_to_first_token_us < 10_000_000);
    }

    // ── Determinism ────────────────────────────────────────────────

    #[test]
    fn test_deterministic_same_seed() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let make_seq = |seed: u64| -> Vec<u32> {
            let cfg = StreamConfig { max_tokens: 20, ..default_config() };
            let mut dec = create_stream_decoder(cfg);
            dec.rng_state = seed;
            cpu_start_stream(&mut dec);
            (0..20)
                .map(|_| cpu_feed_logits(&mut dec, &logits).unwrap().token_id)
                .collect()
        };
        assert_eq!(make_seq(42), make_seq(42));
    }

    #[test]
    fn test_different_seeds_differ() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let make_seq = |seed: u64| -> Vec<u32> {
            let cfg = StreamConfig { max_tokens: 20, ..default_config() };
            let mut dec = create_stream_decoder(cfg);
            dec.rng_state = seed;
            cpu_start_stream(&mut dec);
            (0..20)
                .map(|_| cpu_feed_logits(&mut dec, &logits).unwrap().token_id)
                .collect()
        };
        // With high probability, different seeds yield different sequences
        assert_ne!(make_seq(1), make_seq(999));
    }

    // ── Edge cases ─────────────────────────────────────────────────

    #[test]
    fn test_vocab_size_one() {
        let mut dec = create_stream_decoder(default_config());
        cpu_start_stream(&mut dec);
        let event = cpu_feed_logits(&mut dec, &[5.0]).unwrap();
        assert_eq!(event.token_id, 0);
        assert!((event.token_prob - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_temperature_zero_selects_argmax() {
        let cfg = StreamConfig { temperature: 0.0, ..default_config() };
        let mut dec = create_stream_decoder(cfg);
        cpu_start_stream(&mut dec);
        let logits = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        let event = cpu_feed_logits(&mut dec, &logits).unwrap();
        assert_eq!(event.token_id, 3); // idx of max logit (5.0)
    }

    #[test]
    fn test_all_logits_equal() {
        let mut dec = create_stream_decoder(default_config());
        cpu_start_stream(&mut dec);
        let event = cpu_feed_logits(&mut dec, &[2.0; 50]).unwrap();
        assert!(event.token_id < 50);
    }

    #[test]
    fn test_top_k_larger_than_vocab_edge() {
        let mut logits = vec![1.0, 2.0, 3.0];
        cpu_apply_top_k(&mut logits, 1000);
        // All should remain finite
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_very_large_logits_no_overflow() {
        let mut dec = create_stream_decoder(default_config());
        cpu_start_stream(&mut dec);
        let logits = vec![1e30, 1e30, 1e30];
        let event = cpu_feed_logits(&mut dec, &logits).unwrap();
        assert!(event.token_id < 3);
    }

    #[test]
    fn test_very_negative_logits() {
        let mut dec = create_stream_decoder(default_config());
        cpu_start_stream(&mut dec);
        let logits = vec![-1e30, -1e30, 0.0];
        let event = cpu_feed_logits(&mut dec, &logits).unwrap();
        assert_eq!(event.token_id, 2); // only non-extreme value
    }

    // ── Property-based ─────────────────────────────────────────────

    #[test]
    fn test_probs_sum_to_one() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "probs sum = {sum}");
    }

    #[test]
    fn test_cumulative_prob_monotonically_increases() {
        let mut dec = create_stream_decoder(default_config());
        cpu_start_stream(&mut dec);
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut prev_cum = 0.0f32;
        for _ in 0..10 {
            let event = cpu_feed_logits(&mut dec, &logits).unwrap();
            assert!(
                event.cumulative_prob >= prev_cum,
                "cumulative prob decreased: {} < {}",
                event.cumulative_prob,
                prev_cum,
            );
            prev_cum = event.cumulative_prob;
        }
    }

    // ── Backpressure / format ──────────────────────────────────────

    #[test]
    fn test_backpressure_policy_variants() {
        let _drop = BackpressurePolicy::Drop;
        let _buf = BackpressurePolicy::Buffer(64);
        let _block = BackpressurePolicy::Block;
        assert_ne!(BackpressurePolicy::Drop, BackpressurePolicy::Block);
    }

    #[test]
    fn test_format_stream_status_idle() {
        let dec = create_stream_decoder(default_config());
        let status = format_stream_status(&dec);
        assert!(status.contains("Idle"));
    }

    #[test]
    fn test_format_stream_status_generating() {
        let mut dec = create_stream_decoder(default_config());
        cpu_start_stream(&mut dec);
        let _ = cpu_feed_logits(&mut dec, &uniform_logits(10));
        let status = format_stream_status(&dec);
        assert!(status.contains("Generating"));
        assert!(status.contains("tokens=1"));
    }

    #[test]
    fn test_stream_error_display() {
        assert_eq!(format!("{}", StreamError::NotStarted), "stream not started");
        assert_eq!(format!("{}", StreamError::BufferOverflow), "backpressure buffer overflow");
    }

    #[test]
    fn test_finish_reason_display() {
        assert_eq!(format!("{}", FinishReason::MaxTokens), "max_tokens");
        assert_eq!(format!("{}", FinishReason::StopToken(42)), "stop_token(42)");
    }

    #[test]
    fn test_xorshift_deterministic() {
        let mut s1 = 42u64;
        let mut s2 = 42u64;
        let a: Vec<u64> = (0..10).map(|_| xorshift64(&mut s1)).collect();
        let b: Vec<u64> = (0..10).map(|_| xorshift64(&mut s2)).collect();
        assert_eq!(a, b);
    }

    #[test]
    fn test_xorshift_zero_seed_reseeds() {
        let mut state = 0u64;
        let val = xorshift64(&mut state);
        assert_ne!(val, 0);
        assert_ne!(state, 0);
    }

    #[test]
    fn test_sample_token_peaked_distribution() {
        let mut logits = vec![0.0; 100];
        logits[42] = 100.0;
        let mut rng = 123u64;
        let (id, _) = cpu_sample_token(&logits, &mut rng);
        assert_eq!(id, 42);
    }

    #[test]
    fn test_multiple_stop_tokens() {
        let cfg = StreamConfig {
            stop_tokens: vec![3, 7, 11],
            max_tokens: 100,
            ..default_config()
        };
        let mut dec = create_stream_decoder(cfg);
        cpu_start_stream(&mut dec);
        let mut logits = vec![0.0; 20];
        logits[7] = 100.0;
        let _ = cpu_feed_logits(&mut dec, &logits);
        assert_eq!(dec.state, StreamState::Finished(FinishReason::StopToken(7)));
    }

    #[test]
    fn test_feed_logits_after_pause_works() {
        let mut dec = create_stream_decoder(default_config());
        cpu_start_stream(&mut dec);
        let _ = cpu_feed_logits(&mut dec, &uniform_logits(10));
        cpu_pause_stream(&mut dec).unwrap();
        cpu_resume_stream(&mut dec).unwrap();
        let event = cpu_feed_logits(&mut dec, &uniform_logits(10)).unwrap();
        assert_eq!(event.position, 1);
    }
}
