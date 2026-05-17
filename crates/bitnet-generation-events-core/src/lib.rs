//! Reusable streaming generation event contracts for `BitNet` inference.

use bitnet_generation_stop_core::StopReason;
use serde::{Deserialize, Serialize};

/// Top-k logit evidence for a candidate token at one decode step.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TopLogit {
    /// Vocabulary index of the candidate token.
    pub token_id: u32,
    /// Raw logit value for the candidate token.
    pub logit: f32,
}

/// Logit evidence captured for a single generation step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GreedyLogitStep {
    /// Zero-based generation step index.
    pub step: usize,
    /// Top-k candidate logits captured for this step.
    pub top_logits: Vec<TopLogit>,
    /// Token selected by the sampler for this step.
    pub chosen_id: Option<u32>,
}

/// A single greedy argmax invariant violation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GreedyArgmaxViolation {
    /// Zero-based generation step index.
    pub step: usize,
    /// Token with the maximum finite logit among captured candidates.
    pub argmax_token_id: u32,
    /// Maximum finite logit among captured candidates.
    pub argmax_logit: f32,
    /// Token selected by the sampler.
    pub chosen_id: u32,
}

/// Missing or unusable evidence for a generation step.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GreedyArgmaxMissingEvidence {
    /// Zero-based generation step index.
    pub step: usize,
    /// Human-readable reason the step could not be checked.
    pub reason: String,
}

/// Result of checking greedy argmax evidence.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GreedyArgmaxReport {
    /// Number of logit-dump steps inspected.
    pub checked_steps: usize,
    /// Steps where the selected token did not match the captured argmax.
    pub violations: Vec<GreedyArgmaxViolation>,
    /// Steps that lacked sufficient finite top-logit or chosen-token evidence.
    pub missing_evidence: Vec<GreedyArgmaxMissingEvidence>,
}

impl GreedyArgmaxReport {
    /// Returns true when every inspected step had enough evidence and no mismatch.
    pub fn is_valid(&self) -> bool {
        self.checked_steps > 0 && self.violations.is_empty() && self.missing_evidence.is_empty()
    }
}

/// Check that every captured greedy decode step selected the maximum finite logit.
///
/// The check intentionally ignores non-finite and implausibly huge sentinel values
/// so JSON receipts containing `NaN`/`inf`-like placeholders do not produce a false
/// argmax. Ties are resolved in favor of the first maximum in the captured top-k
/// order, matching the historical maintenance script this core helper replaces.
pub fn check_greedy_argmax(steps: &[GreedyLogitStep]) -> GreedyArgmaxReport {
    let mut report = GreedyArgmaxReport { checked_steps: steps.len(), ..Default::default() };

    for (fallback_step, step) in steps.iter().enumerate() {
        let step_idx = step.step;
        let Some(chosen_id) = step.chosen_id else {
            report.missing_evidence.push(GreedyArgmaxMissingEvidence {
                step: step_idx,
                reason: "missing chosen_id".to_string(),
            });
            continue;
        };

        let mut best: Option<TopLogit> = None;
        for candidate in &step.top_logits {
            if !is_checkable_logit(candidate.logit) {
                continue;
            }
            if best.is_none_or(|current| candidate.logit > current.logit) {
                best = Some(*candidate);
            }
        }

        let Some(best) = best else {
            report.missing_evidence.push(GreedyArgmaxMissingEvidence {
                step: step_idx,
                reason: format!("no finite top_logits for step {fallback_step}"),
            });
            continue;
        };

        if best.token_id != chosen_id {
            report.violations.push(GreedyArgmaxViolation {
                step: step_idx,
                argmax_token_id: best.token_id,
                argmax_logit: best.logit,
                chosen_id,
            });
        }
    }

    report
}

#[inline]
fn is_checkable_logit(value: f32) -> bool {
    value.is_finite() && (-1.0e10..1.0e10).contains(&value)
}

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

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn greedy_argmax_report_accepts_matching_steps() {
        let report = check_greedy_argmax(&[GreedyLogitStep {
            step: 0,
            top_logits: vec![
                TopLogit { token_id: 7, logit: 1.5 },
                TopLogit { token_id: 9, logit: 3.25 },
            ],
            chosen_id: Some(9),
        }]);

        assert!(report.is_valid());
        assert_eq!(report.checked_steps, 1);
    }

    #[test]
    fn greedy_argmax_report_records_mismatch_and_missing_evidence() {
        let report = check_greedy_argmax(&[
            GreedyLogitStep {
                step: 2,
                top_logits: vec![
                    TopLogit { token_id: 4, logit: 2.0 },
                    TopLogit { token_id: 5, logit: 1.0 },
                ],
                chosen_id: Some(5),
            },
            GreedyLogitStep { step: 3, top_logits: vec![], chosen_id: Some(6) },
            GreedyLogitStep {
                step: 4,
                top_logits: vec![TopLogit { token_id: 8, logit: 9.0 }],
                chosen_id: None,
            },
        ]);

        assert!(!report.is_valid());
        assert_eq!(report.violations[0].step, 2);
        assert_eq!(report.violations[0].argmax_token_id, 4);
        assert_eq!(report.violations[0].chosen_id, 5);
        assert_eq!(report.missing_evidence.len(), 2);
    }
}
