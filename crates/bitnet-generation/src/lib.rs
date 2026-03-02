//! Decode-loop stopping logic and generation event types for `BitNet` inference.
//!
//! This crate provides the **contracts** for generation control: stopping
//! criteria, streaming events, and generation statistics.  It does not
//! contain a decode loop implementation – `bitnet-inference` owns that.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Stopping
// ---------------------------------------------------------------------------

/// Criteria used to decide when to stop token generation.
///
/// All fields are additive: any satisfied condition terminates generation.
/// Build with struct-literal syntax or [`Default::default`] and then
/// override the fields you need.
///
/// # Examples
///
/// ```
/// use bitnet_generation::StopCriteria;
///
/// let criteria = StopCriteria {
///     stop_token_ids: vec![128009],          // <|eot_id|> for LLaMA-3
///     stop_strings: vec!["</s>".to_string()],
///     max_tokens: 256,
///     eos_token_id: Some(2),
/// };
/// assert_eq!(criteria.max_tokens, 256);
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StopCriteria {
    /// Token IDs that immediately terminate generation when produced.
    pub stop_token_ids: Vec<u32>,
    /// String sequences that terminate generation when they appear in the
    /// rolling decoded-text tail.
    pub stop_strings: Vec<String>,
    /// Hard cap on the number of tokens to generate (0 = no limit).
    pub max_tokens: usize,
    /// The model's EOS token id, if known.
    pub eos_token_id: Option<u32>,
}

/// Reason why generation stopped.
///
/// Returned by [`check_stop`] when a stopping condition is met.
///
/// # Examples
///
/// ```
/// use bitnet_generation::StopReason;
///
/// let reason = StopReason::StopTokenId(128009);
/// assert_eq!(reason, StopReason::StopTokenId(128009));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StopReason {
    /// Reached the maximum token budget.
    MaxTokens,
    /// A stop token was produced.
    StopTokenId(u32),
    /// A stop string was found in the decoded output.
    StopString(String),
    /// The model produced its EOS token.
    EosToken,
}

/// Check whether any stop condition is satisfied after producing `token_id`.
///
/// Returns `Some(StopReason)` on the **first** matching condition, in this
/// priority order:
/// 1. Token-ID stop list (O(n) scan; typically very small).
/// 2. EOS token.
/// 3. `max_tokens` budget (checked against `generated.len()`).
/// 4. Stop strings (substring search inside `decoded_tail`).
///
/// Returns `None` if generation should continue.
///
/// # Examples
///
/// ```
/// use bitnet_generation::{check_stop, StopCriteria, StopReason};
///
/// let criteria = StopCriteria {
///     stop_token_ids: vec![128009],
///     stop_strings: vec!["</s>".to_string()],
///     max_tokens: 4,
///     eos_token_id: Some(2),
/// };
///
/// // 1. Explicit stop token ID has highest priority.
/// assert_eq!(
///     check_stop(&criteria, 128009, &[], ""),
///     Some(StopReason::StopTokenId(128009))
/// );
///
/// // 2. EOS token.
/// assert_eq!(
///     check_stop(&criteria, 2, &[], ""),
///     Some(StopReason::EosToken)
/// );
///
/// // 3. Token budget exhausted.
/// assert_eq!(
///     check_stop(&criteria, 5, &[0, 1, 2, 3], ""),
///     Some(StopReason::MaxTokens)
/// );
///
/// // 4. Stop string found in decoded tail.
/// assert_eq!(
///     check_stop(&criteria, 5, &[], "hello</s>"),
///     Some(StopReason::StopString("</s>".to_string()))
/// );
///
/// // No condition met → generation continues.
/// assert!(check_stop(&criteria, 7, &[1], "hello world").is_none());
/// ```
pub fn check_stop(
    criteria: &StopCriteria,
    token_id: u32,
    generated: &[u32],
    decoded_tail: &str,
) -> Option<StopReason> {
    // 1. Explicit stop token IDs.
    if criteria.stop_token_ids.contains(&token_id) {
        return Some(StopReason::StopTokenId(token_id));
    }
    // 2. EOS token.
    if criteria.eos_token_id == Some(token_id) {
        return Some(StopReason::EosToken);
    }
    // 3. Token budget.
    if criteria.max_tokens > 0 && generated.len() >= criteria.max_tokens {
        return Some(StopReason::MaxTokens);
    }
    // 4. Stop strings.
    for stop in &criteria.stop_strings {
        if decoded_tail.contains(stop.as_str()) {
            return Some(StopReason::StopString(stop.clone()));
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Generation config
// ---------------------------------------------------------------------------

/// Configuration for a single generation request.
///
/// Note: This is the *orchestration-level* config used by `bitnet-engine-core`
/// traits.  The per-engine `GenerationConfig` in `bitnet-inference` carries
/// additional implementation details.
///
/// # Examples
///
/// ```
/// use bitnet_generation::{GenerationConfig, StopCriteria};
///
/// let config = GenerationConfig {
///     max_new_tokens: 64,
///     seed: Some(42),
///     stop_criteria: StopCriteria {
///         eos_token_id: Some(2),
///         max_tokens: 64,
///         ..Default::default()
///     },
/// };
/// assert_eq!(config.max_new_tokens, 64);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Maximum number of new tokens to generate.
    pub max_new_tokens: usize,
    /// Random seed for reproducible generation (None = random).
    pub seed: Option<u64>,
    /// Stopping criteria.
    pub stop_criteria: StopCriteria,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self { max_new_tokens: 128, seed: None, stop_criteria: StopCriteria::default() }
    }
}

// ---------------------------------------------------------------------------
// Streaming events
// ---------------------------------------------------------------------------

/// A token produced during streaming generation.
///
/// Emitted as [`StreamEvent::Token`] during streaming generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenEvent {
    /// Vocabulary index of the token.
    pub id: u32,
    /// Decoded text fragment for this token.
    pub text: String,
}

/// Summary statistics after generation completes.
///
/// Carried by [`StreamEvent::Done`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GenerationStats {
    /// Number of tokens generated.
    pub tokens_generated: usize,
    /// Throughput in tokens/second.
    pub tokens_per_second: f64,
}

/// Events emitted during streaming generation.
///
/// A complete generation run produces zero or more [`StreamEvent::Token`]
/// events followed by exactly one [`StreamEvent::Done`].
///
/// # Examples
///
/// ```
/// use bitnet_generation::{StreamEvent, StopReason, GenerationStats, TokenEvent};
///
/// let events: Vec<StreamEvent> = vec![
///     StreamEvent::Token(TokenEvent { id: 42, text: "Hello".to_string() }),
///     StreamEvent::Done {
///         reason: StopReason::MaxTokens,
///         stats: GenerationStats { tokens_generated: 1, tokens_per_second: 10.0 },
///     },
/// ];
///
/// let done_count = events.iter().filter(|e| matches!(e, StreamEvent::Done { .. })).count();
/// assert_eq!(done_count, 1);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamEvent {
    /// A single token was generated.
    Token(TokenEvent),
    /// Generation is complete.
    Done {
        /// Why generation stopped.
        reason: StopReason,
        /// Performance summary.
        stats: GenerationStats,
    },
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_criteria(
        stop_ids: &[u32],
        stop_strings: &[&str],
        max: usize,
        eos: Option<u32>,
    ) -> StopCriteria {
        StopCriteria {
            stop_token_ids: stop_ids.to_vec(),
            stop_strings: stop_strings.iter().map(ToString::to_string).collect(),
            max_tokens: max,
            eos_token_id: eos,
        }
    }

    #[test]
    fn no_stop_when_conditions_not_met() {
        let criteria = make_criteria(&[999], &[], 10, Some(0));
        let result = check_stop(&criteria, 42, &[1, 2, 3], "hello");
        assert!(result.is_none());
    }

    #[test]
    fn stop_on_stop_token_id() {
        let criteria = make_criteria(&[42], &[], 100, None);
        let result = check_stop(&criteria, 42, &[], "");
        assert_eq!(result, Some(StopReason::StopTokenId(42)));
    }

    #[test]
    fn stop_on_eos_token() {
        let criteria = make_criteria(&[], &[], 100, Some(2));
        let result = check_stop(&criteria, 2, &[], "");
        assert_eq!(result, Some(StopReason::EosToken));
    }

    #[test]
    fn stop_token_id_takes_priority_over_eos() {
        // token_id == 2 is in both stop_token_ids and eos_token_id.
        let criteria = make_criteria(&[2], &[], 100, Some(2));
        let result = check_stop(&criteria, 2, &[], "");
        assert_eq!(result, Some(StopReason::StopTokenId(2)));
    }

    #[test]
    fn stop_on_max_tokens() {
        let criteria = make_criteria(&[], &[], 3, None);
        // After producing token at index 3, generated has 3 entries.
        let result = check_stop(&criteria, 5, &[1, 2, 3], "");
        assert_eq!(result, Some(StopReason::MaxTokens));
    }

    #[test]
    fn no_stop_when_below_max_tokens() {
        let criteria = make_criteria(&[], &[], 5, None);
        let result = check_stop(&criteria, 5, &[1, 2], "");
        assert!(result.is_none());
    }

    #[test]
    fn stop_on_stop_string() {
        let criteria = make_criteria(&[], &["</s>"], 100, None);
        let result = check_stop(&criteria, 5, &[], "some text</s>extra");
        assert_eq!(result, Some(StopReason::StopString("</s>".to_string())));
    }

    #[test]
    fn no_stop_when_string_absent() {
        let criteria = make_criteria(&[], &["</s>"], 100, None);
        let result = check_stop(&criteria, 5, &[], "some text");
        assert!(result.is_none());
    }

    #[test]
    fn max_tokens_zero_means_no_budget_limit() {
        let criteria = make_criteria(&[], &[], 0, None);
        // Even with 1 million "generated" tokens the budget check is skipped.
        let result = check_stop(&criteria, 5, &vec![0u32; 1_000_000], "");
        assert!(result.is_none());
    }

    #[test]
    fn generation_config_defaults_are_sensible() {
        let cfg = GenerationConfig::default();
        assert_eq!(cfg.max_new_tokens, 128);
        assert!(cfg.seed.is_none());
    }

    #[test]
    fn stream_event_done_carries_reason() {
        let ev = StreamEvent::Done {
            reason: StopReason::EosToken,
            stats: GenerationStats { tokens_generated: 10, tokens_per_second: 5.0 },
        };
        match ev {
            StreamEvent::Done { reason, stats } => {
                assert_eq!(reason, StopReason::EosToken);
                assert_eq!(stats.tokens_generated, 10);
            }
            StreamEvent::Token { .. } => panic!("expected Done event"),
        }
    }

    // --- proptest -----------------------------------------------------------

    proptest::proptest! {
        #[test]
        fn stop_on_token_id_always_fires_when_present(
            id in 0u32..1000,
        ) {
            let criteria = make_criteria(&[id], &[], 1000, None);
            let result = check_stop(&criteria, id, &[1, 2, 3], "hello");
            proptest::prop_assert_eq!(result, Some(StopReason::StopTokenId(id)));
        }

        #[test]
        fn no_stop_without_triggers(
            id in 1000u32..2000,
            generated_len in 1usize..50,
        ) {
            // Use a different id than stop_token_ids and max_tokens > generated_len.
            let criteria = make_criteria(&[9999], &[], 100, Some(9998));
            let generated: Vec<u32> = (0..u32::try_from(generated_len).unwrap()).collect();
            let result = check_stop(&criteria, id, &generated, "no-stop-string-here");
            proptest::prop_assert!(result.is_none());
        }

        #[test]
        fn max_tokens_fires_exactly_at_budget(
            budget in 1usize..100,
        ) {
            let criteria = make_criteria(&[], &[], budget, None);
            let generated: Vec<u32> = vec![1u32; budget];
            // At exactly budget, max_tokens should fire.
            let result = check_stop(&criteria, 5, &generated, "");
            proptest::prop_assert_eq!(result, Some(StopReason::MaxTokens));
        }

        #[test]
        fn stop_string_match_fires(
            prefix in "[a-z]{0,20}",
        ) {
            let criteria = make_criteria(&[], &["STOP"], 1000, None);
            let tail = format!("{prefix}STOP");
            let result = check_stop(&criteria, 5, &[], &tail);
            proptest::prop_assert_eq!(result, Some(StopReason::StopString("STOP".to_string())));
        }
    }
}
