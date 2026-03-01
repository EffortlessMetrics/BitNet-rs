//! Text generation engine.
//!
//! Orchestrates the autoregressive token generation loop
//! with streaming output and stop condition handling.

use std::time::Duration;

// ---------------------------------------------------------------------------
// Stop conditions
// ---------------------------------------------------------------------------

/// Stop conditions for generation.
#[derive(Debug, Clone)]
pub enum StopCondition {
    /// Stop after generating this many tokens.
    MaxTokens(usize),
    /// Stop when this token ID is produced.
    EosToken(u32),
    /// Stop when any of these token-ID sequences appear at the tail.
    StopSequence(Vec<Vec<u32>>),
    /// Stop after this wall-clock duration.
    MaxTime(Duration),
}

// ---------------------------------------------------------------------------
// Generation configuration
// ---------------------------------------------------------------------------

/// Generation configuration.
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Maximum number of tokens to generate.
    pub max_tokens: usize,
    /// Active stop conditions.
    pub stop_conditions: Vec<StopCondition>,
    /// Whether to stream tokens as they are produced.
    pub stream: bool,
    /// Whether to echo the prompt tokens in the output.
    pub echo_prompt: bool,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            stop_conditions: vec![StopCondition::MaxTokens(256)],
            stream: false,
            echo_prompt: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Generated token
// ---------------------------------------------------------------------------

/// Token output from the generation loop.
#[derive(Debug, Clone)]
pub struct GeneratedToken {
    /// Vocabulary index.
    pub token_id: u32,
    /// Decoded text fragment.
    pub token_text: String,
    /// Log-probability of this token.
    pub logprob: f32,
    /// Whether this is a special/control token.
    pub is_special: bool,
    /// Running total of tokens generated so far (including this one).
    pub cumulative_tokens: usize,
}

// ---------------------------------------------------------------------------
// Generation statistics
// ---------------------------------------------------------------------------

/// Generation statistics.
#[derive(Debug, Clone)]
pub struct GenerationStats {
    /// Number of prompt tokens processed.
    pub prompt_tokens: usize,
    /// Number of tokens generated.
    pub generated_tokens: usize,
    /// Total wall-clock time in milliseconds.
    pub total_time_ms: u64,
    /// Time spent evaluating the prompt in milliseconds.
    pub prompt_eval_time_ms: u64,
    /// Time spent generating tokens in milliseconds.
    pub token_gen_time_ms: u64,
    /// Generation throughput (tokens / second).
    pub tokens_per_second: f64,
    /// Prompt evaluation throughput (prompt tokens / second).
    pub prompt_tokens_per_second: f64,
}

impl GenerationStats {
    /// Create zeroed-out stats.
    pub const fn new() -> Self {
        Self {
            prompt_tokens: 0,
            generated_tokens: 0,
            total_time_ms: 0,
            prompt_eval_time_ms: 0,
            token_gen_time_ms: 0,
            tokens_per_second: 0.0,
            prompt_tokens_per_second: 0.0,
        }
    }

    /// Compute derived metrics after generation completes.
    pub fn finalize(
        &mut self,
        prompt_tokens: usize,
        gen_tokens: usize,
        total_ms: u64,
        prompt_ms: u64,
    ) {
        self.prompt_tokens = prompt_tokens;
        self.generated_tokens = gen_tokens;
        self.total_time_ms = total_ms;
        self.prompt_eval_time_ms = prompt_ms;
        self.token_gen_time_ms = total_ms.saturating_sub(prompt_ms);

        self.tokens_per_second = if self.token_gen_time_ms > 0 {
            #[allow(clippy::cast_precision_loss)]
            let tps = (gen_tokens as f64) / (self.token_gen_time_ms as f64 / 1000.0);
            tps
        } else {
            0.0
        };

        self.prompt_tokens_per_second = if self.prompt_eval_time_ms > 0 {
            #[allow(clippy::cast_precision_loss)]
            let ptps = (prompt_tokens as f64) / (self.prompt_eval_time_ms as f64 / 1000.0);
            ptps
        } else {
            0.0
        };
    }
}

impl Default for GenerationStats {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Stop reason
// ---------------------------------------------------------------------------

/// Reason why generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// Reached the maximum token budget.
    MaxTokens,
    /// The model produced an EOS token.
    EosToken,
    /// A stop sequence was matched.
    StopSequence,
    /// Wall-clock timeout expired.
    Timeout,
    /// The user or caller aborted generation.
    UserAbort,
}

// ---------------------------------------------------------------------------
// Generation result
// ---------------------------------------------------------------------------

/// Complete result of a generation run.
#[derive(Debug)]
pub struct GenerationResult {
    /// All generated tokens.
    pub tokens: Vec<GeneratedToken>,
    /// Why generation stopped.
    pub stop_reason: StopReason,
    /// Performance statistics.
    pub stats: GenerationStats,
}

// ---------------------------------------------------------------------------
// Stop condition checking
// ---------------------------------------------------------------------------

/// Check whether any stop condition is satisfied.
///
/// Returns the first matching [`StopReason`], or `None` if generation
/// should continue.
pub fn check_stop_conditions(
    tokens: &[u32],
    config: &GenerationConfig,
    elapsed: Duration,
) -> Option<StopReason> {
    for condition in &config.stop_conditions {
        match condition {
            StopCondition::MaxTokens(max) => {
                if tokens.len() >= *max {
                    return Some(StopReason::MaxTokens);
                }
            }
            StopCondition::EosToken(eos_id) => {
                if let Some(&last) = tokens.last()
                    && last == *eos_id
                {
                    return Some(StopReason::EosToken);
                }
            }
            StopCondition::StopSequence(sequences) => {
                if check_stop_sequence(tokens, sequences) {
                    return Some(StopReason::StopSequence);
                }
            }
            StopCondition::MaxTime(max_dur) => {
                if elapsed >= *max_dur {
                    return Some(StopReason::Timeout);
                }
            }
        }
    }
    None
}

/// Check whether `generated` ends with any of the `stop_sequences`.
pub fn check_stop_sequence(generated: &[u32], stop_sequences: &[Vec<u32>]) -> bool {
    for seq in stop_sequences {
        if seq.is_empty() {
            continue;
        }
        if generated.len() >= seq.len() && generated[generated.len() - seq.len()..] == seq[..] {
            return true;
        }
    }
    false
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helper builders
    // -----------------------------------------------------------------------

    fn config_with(conditions: Vec<StopCondition>) -> GenerationConfig {
        GenerationConfig {
            max_tokens: 256,
            stop_conditions: conditions,
            stream: false,
            echo_prompt: false,
        }
    }

    fn token(id: u32, text: &str, logprob: f32, cum: usize) -> GeneratedToken {
        GeneratedToken {
            token_id: id,
            token_text: text.to_string(),
            logprob,
            is_special: false,
            cumulative_tokens: cum,
        }
    }

    // -----------------------------------------------------------------------
    // GenerationConfig defaults
    // -----------------------------------------------------------------------

    #[test]
    fn default_config_max_tokens() {
        let cfg = GenerationConfig::default();
        assert_eq!(cfg.max_tokens, 256);
    }

    #[test]
    fn default_config_stream_off() {
        let cfg = GenerationConfig::default();
        assert!(!cfg.stream);
    }

    #[test]
    fn default_config_echo_prompt_off() {
        let cfg = GenerationConfig::default();
        assert!(!cfg.echo_prompt);
    }

    #[test]
    fn default_config_has_one_stop_condition() {
        let cfg = GenerationConfig::default();
        assert_eq!(cfg.stop_conditions.len(), 1);
    }

    #[test]
    fn default_config_stop_is_max_tokens_256() {
        let cfg = GenerationConfig::default();
        assert!(matches!(cfg.stop_conditions[0], StopCondition::MaxTokens(256)));
    }

    // -----------------------------------------------------------------------
    // StopCondition::MaxTokens
    // -----------------------------------------------------------------------

    #[test]
    fn max_tokens_fires_at_limit() {
        let cfg = config_with(vec![StopCondition::MaxTokens(3)]);
        let tokens = vec![1, 2, 3];
        assert_eq!(
            check_stop_conditions(&tokens, &cfg, Duration::ZERO),
            Some(StopReason::MaxTokens)
        );
    }

    #[test]
    fn max_tokens_does_not_fire_below_limit() {
        let cfg = config_with(vec![StopCondition::MaxTokens(5)]);
        let tokens = vec![1, 2];
        assert!(check_stop_conditions(&tokens, &cfg, Duration::ZERO).is_none());
    }

    #[test]
    fn max_tokens_fires_above_limit() {
        let cfg = config_with(vec![StopCondition::MaxTokens(2)]);
        let tokens = vec![1, 2, 3, 4];
        assert_eq!(
            check_stop_conditions(&tokens, &cfg, Duration::ZERO),
            Some(StopReason::MaxTokens)
        );
    }

    #[test]
    fn max_tokens_zero_never_fires_on_empty() {
        let cfg = config_with(vec![StopCondition::MaxTokens(0)]);
        let tokens: Vec<u32> = vec![];
        assert_eq!(
            check_stop_conditions(&tokens, &cfg, Duration::ZERO),
            Some(StopReason::MaxTokens)
        );
    }

    // -----------------------------------------------------------------------
    // StopCondition::EosToken
    // -----------------------------------------------------------------------

    #[test]
    fn eos_token_fires_when_last_matches() {
        let cfg = config_with(vec![StopCondition::EosToken(2)]);
        let tokens = vec![10, 20, 2];
        assert_eq!(
            check_stop_conditions(&tokens, &cfg, Duration::ZERO),
            Some(StopReason::EosToken)
        );
    }

    #[test]
    fn eos_token_does_not_fire_when_last_differs() {
        let cfg = config_with(vec![StopCondition::EosToken(2)]);
        let tokens = vec![10, 20, 30];
        assert!(check_stop_conditions(&tokens, &cfg, Duration::ZERO).is_none());
    }

    #[test]
    fn eos_token_does_not_fire_on_empty() {
        let cfg = config_with(vec![StopCondition::EosToken(2)]);
        let tokens: Vec<u32> = vec![];
        assert!(check_stop_conditions(&tokens, &cfg, Duration::ZERO).is_none());
    }

    #[test]
    fn eos_token_fires_single_token() {
        let cfg = config_with(vec![StopCondition::EosToken(42)]);
        let tokens = vec![42];
        assert_eq!(
            check_stop_conditions(&tokens, &cfg, Duration::ZERO),
            Some(StopReason::EosToken)
        );
    }

    #[test]
    fn eos_token_not_in_middle() {
        let cfg = config_with(vec![StopCondition::EosToken(2)]);
        let tokens = vec![2, 10, 20];
        assert!(check_stop_conditions(&tokens, &cfg, Duration::ZERO).is_none());
    }

    // -----------------------------------------------------------------------
    // StopCondition::StopSequence
    // -----------------------------------------------------------------------

    #[test]
    fn stop_sequence_exact_match_at_tail() {
        let cfg = config_with(vec![StopCondition::StopSequence(vec![vec![10, 20]])]);
        let tokens = vec![1, 2, 10, 20];
        assert_eq!(
            check_stop_conditions(&tokens, &cfg, Duration::ZERO),
            Some(StopReason::StopSequence)
        );
    }

    #[test]
    fn stop_sequence_no_match() {
        let cfg = config_with(vec![StopCondition::StopSequence(vec![vec![10, 20]])]);
        let tokens = vec![1, 2, 30, 40];
        assert!(check_stop_conditions(&tokens, &cfg, Duration::ZERO).is_none());
    }

    #[test]
    fn stop_sequence_partial_match() {
        let cfg = config_with(vec![StopCondition::StopSequence(vec![vec![10, 20, 30]])]);
        let tokens = vec![1, 10, 20];
        assert!(check_stop_conditions(&tokens, &cfg, Duration::ZERO).is_none());
    }

    #[test]
    fn stop_sequence_multiple_candidates() {
        let cfg = config_with(vec![StopCondition::StopSequence(vec![vec![99, 98], vec![5, 6]])]);
        let tokens = vec![1, 2, 5, 6];
        assert_eq!(
            check_stop_conditions(&tokens, &cfg, Duration::ZERO),
            Some(StopReason::StopSequence)
        );
    }

    #[test]
    fn stop_sequence_first_of_multiple_matches() {
        let cfg =
            config_with(vec![StopCondition::StopSequence(vec![vec![5, 6], vec![1, 2, 5, 6]])]);
        let tokens = vec![1, 2, 5, 6];
        assert_eq!(
            check_stop_conditions(&tokens, &cfg, Duration::ZERO),
            Some(StopReason::StopSequence)
        );
    }

    #[test]
    fn stop_sequence_single_token_seq() {
        let cfg = config_with(vec![StopCondition::StopSequence(vec![vec![42]])]);
        let tokens = vec![1, 2, 42];
        assert_eq!(
            check_stop_conditions(&tokens, &cfg, Duration::ZERO),
            Some(StopReason::StopSequence)
        );
    }

    #[test]
    fn stop_sequence_empty_sequence_skipped() {
        let cfg = config_with(vec![StopCondition::StopSequence(vec![vec![]])]);
        let tokens = vec![1, 2, 3];
        assert!(check_stop_conditions(&tokens, &cfg, Duration::ZERO).is_none());
    }

    #[test]
    fn stop_sequence_empty_generated() {
        let cfg = config_with(vec![StopCondition::StopSequence(vec![vec![1, 2]])]);
        let tokens: Vec<u32> = vec![];
        assert!(check_stop_conditions(&tokens, &cfg, Duration::ZERO).is_none());
    }

    #[test]
    fn stop_sequence_generated_shorter_than_seq() {
        let cfg = config_with(vec![StopCondition::StopSequence(vec![vec![1, 2, 3, 4, 5]])]);
        let tokens = vec![4, 5];
        assert!(check_stop_conditions(&tokens, &cfg, Duration::ZERO).is_none());
    }

    // -----------------------------------------------------------------------
    // StopCondition::MaxTime
    // -----------------------------------------------------------------------

    #[test]
    fn max_time_fires_when_exceeded() {
        let cfg = config_with(vec![StopCondition::MaxTime(Duration::from_secs(5))]);
        let elapsed = Duration::from_secs(6);
        assert_eq!(check_stop_conditions(&[1], &cfg, elapsed), Some(StopReason::Timeout));
    }

    #[test]
    fn max_time_fires_at_exact_boundary() {
        let cfg = config_with(vec![StopCondition::MaxTime(Duration::from_secs(5))]);
        let elapsed = Duration::from_secs(5);
        assert_eq!(check_stop_conditions(&[1], &cfg, elapsed), Some(StopReason::Timeout));
    }

    #[test]
    fn max_time_does_not_fire_below() {
        let cfg = config_with(vec![StopCondition::MaxTime(Duration::from_secs(5))]);
        let elapsed = Duration::from_secs(4);
        assert!(check_stop_conditions(&[1], &cfg, elapsed).is_none());
    }

    #[test]
    fn max_time_sub_second_precision() {
        let cfg = config_with(vec![StopCondition::MaxTime(Duration::from_millis(500))]);
        let elapsed = Duration::from_millis(501);
        assert_eq!(check_stop_conditions(&[1], &cfg, elapsed), Some(StopReason::Timeout));
    }

    // -----------------------------------------------------------------------
    // GeneratedToken
    // -----------------------------------------------------------------------

    #[test]
    fn generated_token_creation() {
        let t = token(42, "hello", -0.5, 1);
        assert_eq!(t.token_id, 42);
        assert_eq!(t.token_text, "hello");
        assert!((t.logprob - (-0.5)).abs() < f32::EPSILON);
        assert!(!t.is_special);
        assert_eq!(t.cumulative_tokens, 1);
    }

    #[test]
    fn generated_token_special() {
        let t = GeneratedToken {
            token_id: 0,
            token_text: "<bos>".to_string(),
            logprob: 0.0,
            is_special: true,
            cumulative_tokens: 0,
        };
        assert!(t.is_special);
    }

    #[test]
    fn generated_token_clone() {
        let t = token(1, "a", -1.0, 5);
        let t2 = t.clone();
        assert_eq!(t.token_id, t2.token_id);
        assert_eq!(t.token_text, t2.token_text);
    }

    // -----------------------------------------------------------------------
    // GenerationStats
    // -----------------------------------------------------------------------

    #[test]
    fn stats_new_is_zeroed() {
        let s = GenerationStats::new();
        assert_eq!(s.prompt_tokens, 0);
        assert_eq!(s.generated_tokens, 0);
        assert_eq!(s.total_time_ms, 0);
        assert_eq!(s.tokens_per_second, 0.0);
    }

    #[test]
    fn stats_default_equals_new() {
        let a = GenerationStats::new();
        let b = GenerationStats::default();
        assert_eq!(a.prompt_tokens, b.prompt_tokens);
        assert_eq!(a.generated_tokens, b.generated_tokens);
    }

    #[test]
    fn stats_finalize_tokens_per_second() {
        let mut s = GenerationStats::new();
        // 10 tokens in 1000ms of gen time (total 1200ms, prompt 200ms)
        s.finalize(5, 10, 1200, 200);
        assert_eq!(s.generated_tokens, 10);
        assert_eq!(s.token_gen_time_ms, 1000);
        assert!((s.tokens_per_second - 10.0).abs() < 0.001);
    }

    #[test]
    fn stats_finalize_prompt_tokens_per_second() {
        let mut s = GenerationStats::new();
        // 100 prompt tokens in 500ms
        s.finalize(100, 10, 1500, 500);
        assert_eq!(s.prompt_tokens, 100);
        assert_eq!(s.prompt_eval_time_ms, 500);
        assert!((s.prompt_tokens_per_second - 200.0).abs() < 0.001);
    }

    #[test]
    fn stats_finalize_zero_gen_time() {
        let mut s = GenerationStats::new();
        // total == prompt → gen time is 0
        s.finalize(5, 10, 100, 100);
        assert_eq!(s.token_gen_time_ms, 0);
        assert_eq!(s.tokens_per_second, 0.0);
    }

    #[test]
    fn stats_finalize_zero_prompt_time() {
        let mut s = GenerationStats::new();
        s.finalize(5, 10, 1000, 0);
        assert_eq!(s.prompt_eval_time_ms, 0);
        assert_eq!(s.prompt_tokens_per_second, 0.0);
    }

    #[test]
    fn stats_finalize_saturating_sub() {
        let mut s = GenerationStats::new();
        // prompt_ms > total_ms → gen time saturates to 0
        s.finalize(5, 10, 100, 200);
        assert_eq!(s.token_gen_time_ms, 0);
        assert_eq!(s.tokens_per_second, 0.0);
    }

    #[test]
    fn stats_finalize_large_throughput() {
        let mut s = GenerationStats::new();
        // 1000 tokens in 100ms gen time → 10_000 tok/s
        s.finalize(0, 1000, 200, 100);
        assert!((s.tokens_per_second - 10_000.0).abs() < 0.1);
    }

    // -----------------------------------------------------------------------
    // StopReason exhaustive matching
    // -----------------------------------------------------------------------

    #[test]
    fn stop_reason_max_tokens() {
        assert_eq!(StopReason::MaxTokens, StopReason::MaxTokens);
    }

    #[test]
    fn stop_reason_eos_token() {
        assert_eq!(StopReason::EosToken, StopReason::EosToken);
    }

    #[test]
    fn stop_reason_stop_sequence() {
        assert_eq!(StopReason::StopSequence, StopReason::StopSequence);
    }

    #[test]
    fn stop_reason_timeout() {
        assert_eq!(StopReason::Timeout, StopReason::Timeout);
    }

    #[test]
    fn stop_reason_user_abort() {
        assert_eq!(StopReason::UserAbort, StopReason::UserAbort);
    }

    #[test]
    fn stop_reason_inequality() {
        assert_ne!(StopReason::MaxTokens, StopReason::EosToken);
        assert_ne!(StopReason::Timeout, StopReason::UserAbort);
    }

    #[test]
    fn stop_reason_copy() {
        let a = StopReason::EosToken;
        let b = a; // Copy
        assert_eq!(a, b);
    }

    // -----------------------------------------------------------------------
    // GenerationResult
    // -----------------------------------------------------------------------

    #[test]
    fn generation_result_empty() {
        let result = GenerationResult {
            tokens: vec![],
            stop_reason: StopReason::MaxTokens,
            stats: GenerationStats::new(),
        };
        assert!(result.tokens.is_empty());
        assert_eq!(result.stop_reason, StopReason::MaxTokens);
    }

    #[test]
    fn generation_result_with_tokens() {
        let result = GenerationResult {
            tokens: vec![token(1, "a", -0.1, 1), token(2, "b", -0.2, 2)],
            stop_reason: StopReason::EosToken,
            stats: GenerationStats::new(),
        };
        assert_eq!(result.tokens.len(), 2);
        assert_eq!(result.stop_reason, StopReason::EosToken);
    }

    #[test]
    fn generation_result_debug() {
        let result = GenerationResult {
            tokens: vec![],
            stop_reason: StopReason::Timeout,
            stats: GenerationStats::new(),
        };
        let dbg = format!("{result:?}");
        assert!(dbg.contains("Timeout"));
    }

    // -----------------------------------------------------------------------
    // check_stop_conditions — combined / priority
    // -----------------------------------------------------------------------

    #[test]
    fn no_conditions_never_stops() {
        let cfg = config_with(vec![]);
        assert!(check_stop_conditions(&[1, 2, 3], &cfg, Duration::ZERO).is_none());
    }

    #[test]
    fn first_matching_condition_wins() {
        // MaxTokens(2) is listed before EosToken(3).
        let cfg = config_with(vec![StopCondition::MaxTokens(2), StopCondition::EosToken(3)]);
        let tokens = vec![1, 3]; // len==2 matches MaxTokens; last==3 matches Eos
        assert_eq!(
            check_stop_conditions(&tokens, &cfg, Duration::ZERO),
            Some(StopReason::MaxTokens)
        );
    }

    #[test]
    fn eos_before_max_tokens_when_ordered() {
        let cfg = config_with(vec![StopCondition::EosToken(3), StopCondition::MaxTokens(2)]);
        let tokens = vec![1, 3];
        assert_eq!(
            check_stop_conditions(&tokens, &cfg, Duration::ZERO),
            Some(StopReason::EosToken)
        );
    }

    #[test]
    fn timeout_with_no_tokens() {
        let cfg = config_with(vec![StopCondition::MaxTime(Duration::from_secs(1))]);
        let elapsed = Duration::from_secs(2);
        assert_eq!(check_stop_conditions(&[], &cfg, elapsed), Some(StopReason::Timeout));
    }

    #[test]
    fn multiple_conditions_all_unmet() {
        let cfg = config_with(vec![
            StopCondition::MaxTokens(100),
            StopCondition::EosToken(999),
            StopCondition::StopSequence(vec![vec![50, 60]]),
            StopCondition::MaxTime(Duration::from_secs(60)),
        ]);
        let tokens = vec![1, 2, 3];
        assert!(check_stop_conditions(&tokens, &cfg, Duration::from_secs(1)).is_none());
    }

    // -----------------------------------------------------------------------
    // check_stop_sequence — standalone
    // -----------------------------------------------------------------------

    #[test]
    fn check_stop_sequence_exact() {
        assert!(check_stop_sequence(&[1, 2, 3], &[vec![2, 3]]));
    }

    #[test]
    fn check_stop_sequence_full_match() {
        assert!(check_stop_sequence(&[1, 2, 3], &[vec![1, 2, 3]]));
    }

    #[test]
    fn check_stop_sequence_no_match() {
        assert!(!check_stop_sequence(&[1, 2, 3], &[vec![4, 5]]));
    }

    #[test]
    fn check_stop_sequence_empty_seqs() {
        assert!(!check_stop_sequence(&[1, 2, 3], &[]));
    }

    #[test]
    fn check_stop_sequence_empty_generated() {
        assert!(!check_stop_sequence(&[], &[vec![1]]));
    }

    #[test]
    fn check_stop_sequence_empty_individual_seq() {
        assert!(!check_stop_sequence(&[1, 2], &[vec![]]));
    }

    #[test]
    fn check_stop_sequence_longer_than_generated() {
        assert!(!check_stop_sequence(&[1], &[vec![1, 2, 3]]));
    }

    #[test]
    fn check_stop_sequence_second_candidate_matches() {
        assert!(check_stop_sequence(&[1, 2, 3], &[vec![9, 9], vec![2, 3]]));
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn empty_generation_no_stop() {
        let cfg = config_with(vec![
            StopCondition::EosToken(2),
            StopCondition::StopSequence(vec![vec![5, 6]]),
        ]);
        assert!(check_stop_conditions(&[], &cfg, Duration::ZERO).is_none());
    }

    #[test]
    fn single_token_eos() {
        let cfg = config_with(vec![StopCondition::EosToken(7)]);
        assert_eq!(check_stop_conditions(&[7], &cfg, Duration::ZERO), Some(StopReason::EosToken));
    }

    #[test]
    fn single_token_max_tokens_one() {
        let cfg = config_with(vec![StopCondition::MaxTokens(1)]);
        assert_eq!(check_stop_conditions(&[42], &cfg, Duration::ZERO), Some(StopReason::MaxTokens));
    }

    #[test]
    fn stop_sequence_at_very_end() {
        let tokens: Vec<u32> = (0..100).collect();
        let seq = vec![98, 99];
        let cfg = config_with(vec![StopCondition::StopSequence(vec![seq])]);
        assert_eq!(
            check_stop_conditions(&tokens, &cfg, Duration::ZERO),
            Some(StopReason::StopSequence)
        );
    }

    // -----------------------------------------------------------------------
    // proptest
    // -----------------------------------------------------------------------

    proptest::proptest! {
        #[test]
        fn max_tokens_always_fires_at_or_above(
            limit in 1usize..50,
            extra in 0usize..10,
        ) {
            let cfg = config_with(vec![StopCondition::MaxTokens(limit)]);
            let tokens: Vec<u32> = (0..u32::try_from(limit + extra).unwrap())
                .collect();
            let result = check_stop_conditions(
                &tokens, &cfg, Duration::ZERO,
            );
            proptest::prop_assert_eq!(result, Some(StopReason::MaxTokens));
        }

        #[test]
        fn eos_fires_when_last_token_matches(
            eos_id in 0u32..1000,
            prefix_len in 0usize..20,
        ) {
            let cfg = config_with(vec![StopCondition::EosToken(eos_id)]);
            let mut tokens: Vec<u32> =
                (1000..1000 + u32::try_from(prefix_len).unwrap()).collect();
            tokens.push(eos_id);
            let result = check_stop_conditions(
                &tokens, &cfg, Duration::ZERO,
            );
            proptest::prop_assert_eq!(result, Some(StopReason::EosToken));
        }

        #[test]
        fn no_false_positives_without_conditions(
            len in 0usize..50,
        ) {
            let cfg = config_with(vec![]);
            let tokens: Vec<u32> =
                (0..u32::try_from(len).unwrap()).collect();
            let result = check_stop_conditions(
                &tokens, &cfg, Duration::from_secs(999),
            );
            proptest::prop_assert!(result.is_none());
        }

        #[test]
        fn stop_sequence_matches_tail(
            seq_len in 1usize..5,
            prefix_len in 0usize..10,
        ) {
            let seq: Vec<u32> =
                (100..100 + u32::try_from(seq_len).unwrap()).collect();
            let mut tokens: Vec<u32> =
                (0..u32::try_from(prefix_len).unwrap()).collect();
            tokens.extend_from_slice(&seq);
            proptest::prop_assert!(check_stop_sequence(&tokens, &[seq]));
        }

        #[test]
        fn finalize_gen_time_never_negative(
            total in 0u64..10000,
            prompt in 0u64..20000,
        ) {
            let mut s = GenerationStats::new();
            s.finalize(5, 10, total, prompt);
            // saturating_sub means gen time is always >= 0
            proptest::prop_assert!(s.token_gen_time_ms <= total);
        }
    }
}
