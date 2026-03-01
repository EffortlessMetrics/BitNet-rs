//! Core stop criteria contracts and stop checking logic for `BitNet` generation.
//!
//! This crate focuses purely on decode-loop stop decisions. Generation
//! orchestration types (stream events, stats, generation config) live in
//! `bitnet-generation`.

use serde::{Deserialize, Serialize};

/// Criteria used to decide when to stop token generation.
///
/// All fields are additive: any satisfied condition terminates generation.
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
/// Returns `Some(StopReason)` on the first matching condition, in this
/// priority order:
/// 1. Explicit stop token IDs.
/// 2. EOS token.
/// 3. `max_tokens` budget.
/// 4. Stop strings.
pub fn check_stop(
    criteria: &StopCriteria,
    token_id: u32,
    generated: &[u32],
    decoded_tail: &str,
) -> Option<StopReason> {
    if criteria.stop_token_ids.contains(&token_id) {
        return Some(StopReason::StopTokenId(token_id));
    }
    if criteria.eos_token_id == Some(token_id) {
        return Some(StopReason::EosToken);
    }
    if criteria.max_tokens > 0 && generated.len() >= criteria.max_tokens {
        return Some(StopReason::MaxTokens);
    }
    for stop in &criteria.stop_strings {
        if decoded_tail.contains(stop.as_str()) {
            return Some(StopReason::StopString(stop.clone()));
        }
    }
    None
}

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
    fn stop_on_stop_token_id() {
        let criteria = make_criteria(&[42], &[], 100, None);
        assert_eq!(check_stop(&criteria, 42, &[], ""), Some(StopReason::StopTokenId(42)));
    }

    #[test]
    fn stop_token_id_takes_priority_over_eos() {
        let criteria = make_criteria(&[2], &[], 100, Some(2));
        assert_eq!(check_stop(&criteria, 2, &[], ""), Some(StopReason::StopTokenId(2)));
    }

    #[test]
    fn stop_on_max_tokens() {
        let criteria = make_criteria(&[], &[], 3, None);
        assert_eq!(check_stop(&criteria, 5, &[1, 2, 3], ""), Some(StopReason::MaxTokens));
    }

    #[test]
    fn stop_on_stop_string() {
        let criteria = make_criteria(&[], &["</s>"], 100, None);
        assert_eq!(
            check_stop(&criteria, 5, &[], "some text</s>extra"),
            Some(StopReason::StopString("</s>".to_string()))
        );
    }

    proptest::proptest! {
        #[test]
        fn no_stop_without_triggers(id in 1000u32..2000, generated_len in 1usize..50) {
            let criteria = make_criteria(&[9999], &[], 100, Some(9998));
            let generated: Vec<u32> = (0..u32::try_from(generated_len).unwrap()).collect();
            let result = check_stop(&criteria, id, &generated, "no-stop-string-here");
            proptest::prop_assert!(result.is_none());
        }
    }
}
