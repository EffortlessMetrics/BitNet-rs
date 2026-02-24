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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenEvent {
    /// Vocabulary index of the token.
    pub id: u32,
    /// Decoded text fragment for this token.
    pub text: String,
}

/// Summary statistics after generation completes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GenerationStats {
    /// Number of tokens generated.
    pub tokens_generated: usize,
    /// Throughput in tokens/second.
    pub tokens_per_second: f64,
}

/// Events emitted during streaming generation.
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
}
