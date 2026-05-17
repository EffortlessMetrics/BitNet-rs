//! Greedy argmax invariant checks for BitNet logits receipt JSON.
//!
//! This crate owns the reusable, testable validation logic that used to live in
//! `scripts/check_greedy_argmax.py`.  Tooling and shell compatibility wrappers
//! should call this crate instead of reimplementing JSON traversal themselves.

use serde_json::Value;
use std::fmt;
use thiserror::Error;

/// Exit code historically used by the Python checker for greedy argmax mismatches.
pub const EXIT_ARGMAX_MISMATCH: i32 = 7;

/// Finite-logit range accepted by the legacy checker.
const LEGACY_VALID_LOGIT_BOUND: f64 = 1.0e10;

/// Result of checking a complete receipt.
#[derive(Debug, Clone, PartialEq)]
pub struct GreedyArgmaxReport {
    /// Number of `logits_dump` steps inspected.
    pub steps: usize,
    /// Non-fatal malformed steps that were skipped by the legacy checker.
    pub missing_steps: Vec<MissingStep>,
    /// Actual greedy argmax violations.
    pub violations: Vec<GreedyArgmaxViolation>,
}

impl GreedyArgmaxReport {
    /// Returns `true` when no selected token disagrees with the top-logit argmax.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.violations.is_empty()
    }
}

/// A step that cannot be checked because required data is absent or malformed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissingStep {
    /// Zero-based step index in `logits_dump`.
    pub step_index: usize,
    /// Human-readable reason.
    pub reason: MissingStepReason,
}

/// Why a step could not be checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissingStepReason {
    /// `top_logits` is absent, not an array, or empty.
    MissingTopLogits,
    /// `chosen_id` is absent or not an unsigned integer.
    MissingChosenId,
    /// No finite logit remained after filtering NaN/inf/out-of-range values.
    NoFiniteTopLogit,
}

impl fmt::Display for MissingStepReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingTopLogits => f.write_str("missing top_logits"),
            Self::MissingChosenId => f.write_str("missing chosen_id"),
            Self::NoFiniteTopLogit => f.write_str("no finite top logit"),
        }
    }
}

/// Greedy selection did not match the argmax token among valid top logits.
#[derive(Debug, Clone, PartialEq)]
pub struct GreedyArgmaxViolation {
    /// Zero-based step index in `logits_dump`.
    pub step_index: usize,
    /// Token id with the maximum valid logit.
    pub argmax_token: u64,
    /// Maximum valid logit value.
    pub argmax_logit: f64,
    /// Token id reported as selected by the sampler.
    pub chosen_id: u64,
    /// First entries from `top_logits`, retained for diagnostics.
    pub top_logits: Vec<TopLogit>,
}

/// A token/logit pair from a step's `top_logits` list.
#[derive(Debug, Clone, PartialEq)]
pub struct TopLogit {
    /// Token id from the JSON entry.
    pub token_id: u64,
    /// Logit value from the JSON entry.
    pub logit: f64,
}

/// JSON-level validation errors that prevent checking the receipt at all.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum GreedyArgmaxError {
    /// The root JSON object does not contain a non-empty `logits_dump` array.
    #[error("no logits dump found in JSON")]
    MissingLogitsDump,
}

/// Check a parsed JSON receipt for the greedy argmax invariant.
///
/// The checker preserves the legacy Python behavior: malformed individual steps
/// are reported as missing data and do not fail the invariant by themselves;
/// mismatched argmax/chosen pairs do fail.
pub fn check_greedy_argmax(value: &Value) -> Result<GreedyArgmaxReport, GreedyArgmaxError> {
    let logits_dump = value
        .get("logits_dump")
        .and_then(Value::as_array)
        .filter(|steps| !steps.is_empty())
        .ok_or(GreedyArgmaxError::MissingLogitsDump)?;

    let mut missing_steps = Vec::new();
    let mut violations = Vec::new();

    for (step_index, step) in logits_dump.iter().enumerate() {
        let Some(top_logits) = step.get("top_logits").and_then(Value::as_array) else {
            missing_steps
                .push(MissingStep { step_index, reason: MissingStepReason::MissingTopLogits });
            continue;
        };
        if top_logits.is_empty() {
            missing_steps
                .push(MissingStep { step_index, reason: MissingStepReason::MissingTopLogits });
            continue;
        }

        let Some(chosen_id) = step.get("chosen_id").and_then(Value::as_u64) else {
            missing_steps
                .push(MissingStep { step_index, reason: MissingStepReason::MissingChosenId });
            continue;
        };

        let valid_top_logits = parse_top_logits(top_logits);
        let Some(argmax) =
            valid_top_logits.iter().max_by(|left, right| left.logit.total_cmp(&right.logit))
        else {
            missing_steps
                .push(MissingStep { step_index, reason: MissingStepReason::NoFiniteTopLogit });
            continue;
        };

        if argmax.token_id != chosen_id {
            violations.push(GreedyArgmaxViolation {
                step_index,
                argmax_token: argmax.token_id,
                argmax_logit: argmax.logit,
                chosen_id,
                top_logits: valid_top_logits.into_iter().take(5).collect(),
            });
        }
    }

    Ok(GreedyArgmaxReport { steps: logits_dump.len(), missing_steps, violations })
}

fn parse_top_logits(top_logits: &[Value]) -> Vec<TopLogit> {
    top_logits
        .iter()
        .filter_map(|entry| {
            let token_id = entry.get("token_id")?.as_u64()?;
            let logit = entry.get("logit")?.as_f64()?;
            valid_legacy_logit(logit).then_some(TopLogit { token_id, logit })
        })
        .collect()
}

fn valid_legacy_logit(logit: f64) -> bool {
    logit.is_finite() && (-LEGACY_VALID_LOGIT_BOUND..LEGACY_VALID_LOGIT_BOUND).contains(&logit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn accepts_matching_argmax_steps() {
        let report = check_greedy_argmax(&json!({
            "logits_dump": [
                {"chosen_id": 2, "top_logits": [
                    {"token_id": 1, "logit": 0.5},
                    {"token_id": 2, "logit": 1.25}
                ]}
            ]
        }))
        .unwrap();

        assert!(report.is_valid());
        assert_eq!(report.steps, 1);
        assert!(report.missing_steps.is_empty());
    }

    #[test]
    fn reports_argmax_violation_with_top_five_context() {
        let report = check_greedy_argmax(&json!({
            "logits_dump": [
                {"chosen_id": 3, "top_logits": [
                    {"token_id": 1, "logit": 9.0},
                    {"token_id": 2, "logit": 8.0},
                    {"token_id": 3, "logit": 7.0},
                    {"token_id": 4, "logit": 6.0},
                    {"token_id": 5, "logit": 5.0},
                    {"token_id": 6, "logit": 4.0}
                ]}
            ]
        }))
        .unwrap();

        assert!(!report.is_valid());
        assert_eq!(report.violations.len(), 1);
        let violation = &report.violations[0];
        assert_eq!(violation.step_index, 0);
        assert_eq!(violation.argmax_token, 1);
        assert_eq!(violation.chosen_id, 3);
        assert_eq!(violation.top_logits.len(), 5);
    }

    #[test]
    fn skips_invalid_logits_like_legacy_script() {
        let report = check_greedy_argmax(&json!({
            "logits_dump": [
                {"chosen_id": 7, "top_logits": [
                    {"token_id": 1, "logit": 1.0e20},
                    {"token_id": 7, "logit": 0.5}
                ]}
            ]
        }))
        .unwrap();

        assert!(report.is_valid());
    }

    #[test]
    fn reports_missing_data_without_failing_invariant() {
        let report = check_greedy_argmax(&json!({
            "logits_dump": [
                {"top_logits": []},
                {"top_logits": [{"token_id": 1, "logit": 1.0}]}
            ]
        }))
        .unwrap();

        assert!(report.is_valid());
        assert_eq!(report.missing_steps.len(), 2);
        assert_eq!(report.missing_steps[0].reason, MissingStepReason::MissingTopLogits);
        assert_eq!(report.missing_steps[1].reason, MissingStepReason::MissingChosenId);
    }

    #[test]
    fn rejects_missing_logits_dump() {
        let error = check_greedy_argmax(&json!({})).unwrap_err();
        assert_eq!(error, GreedyArgmaxError::MissingLogitsDump);
    }
}
