use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::{fs, path::Path};

/// Exit code used by the historical Python checker for greedy argmax violations.
pub const EXIT_ARGMAX_MISMATCH: i32 = 7;

#[derive(Debug, Deserialize)]
struct RunOutput {
    #[serde(default)]
    logits_dump: Vec<LogitStep>,
}

#[derive(Debug, Deserialize)]
struct LogitStep {
    #[serde(default)]
    top_logits: Vec<TopLogit>,
    chosen_id: Option<u64>,
}

#[derive(Clone, Debug, Deserialize)]
struct TopLogit {
    token_id: u64,
    logit: f64,
}

#[derive(Debug)]
struct Violation {
    step: usize,
    chosen_id: Option<u64>,
    argmax_token: Option<u64>,
    max_logit: f64,
    top_logits: Vec<TopLogit>,
}

/// Check that every dumped greedy decoding step chose the top finite logit.
///
/// This is the Rust replacement for `scripts/check_greedy_argmax.py`, keeping
/// the validation in xtask with the rest of the developer task surface.
pub fn run(json_path: &Path, verbose: bool) -> Result<()> {
    let raw = fs::read_to_string(json_path)
        .with_context(|| format!("failed to read JSON output from {}", json_path.display()))?;
    let data: RunOutput = serde_json::from_str(&raw)
        .with_context(|| format!("failed to parse CLI JSON output from {}", json_path.display()))?;

    let report = check_greedy_invariant(&data.logits_dump);

    if !report.violations.is_empty() {
        for violation in &report.violations {
            print_violation(violation);
        }
        bail!("greedy argmax mismatch: {} invalid step(s)", report.violations.len());
    }

    println!("✓ Greedy invariant holds for all {} steps", report.step_count);
    if verbose {
        eprintln!("checked {} top-logit entries", report.top_logit_count);
    }
    Ok(())
}

#[derive(Debug, PartialEq)]
struct CheckReport {
    step_count: usize,
    top_logit_count: usize,
    violations: Vec<Violation>,
}

fn check_greedy_invariant(logits_dump: &[LogitStep]) -> CheckReport {
    let mut violations = Vec::new();
    let mut top_logit_count = 0;

    if logits_dump.is_empty() {
        violations.push(Violation {
            step: 0,
            chosen_id: None,
            argmax_token: None,
            max_logit: f64::NEG_INFINITY,
            top_logits: Vec::new(),
        });
        return CheckReport { step_count: 0, top_logit_count, violations };
    }

    for (step, record) in logits_dump.iter().enumerate() {
        top_logit_count += record.top_logits.len();
        let (argmax_token, max_logit) = record
            .top_logits
            .iter()
            .filter(|entry| {
                entry.logit.is_finite() && entry.logit > -1.0e10 && entry.logit < 1.0e10
            })
            .max_by(|left, right| left.logit.total_cmp(&right.logit))
            .map_or((None, f64::NEG_INFINITY), |entry| (Some(entry.token_id), entry.logit));

        if record.top_logits.is_empty()
            || record.chosen_id.is_none()
            || argmax_token != record.chosen_id
        {
            violations.push(Violation {
                step,
                chosen_id: record.chosen_id,
                argmax_token,
                max_logit,
                top_logits: record.top_logits.iter().take(5).cloned().collect(),
            });
        }
    }

    CheckReport { step_count: logits_dump.len(), top_logit_count, violations }
}

fn print_violation(violation: &Violation) {
    match violation.argmax_token {
        Some(argmax) => eprintln!(
            "Step {}: Greedy violation! argmax={} (logit={:.4}) but chosen={}",
            violation.step,
            argmax,
            violation.max_logit,
            violation.chosen_id.map_or_else(|| "<missing>".to_string(), |id| id.to_string())
        ),
        None => eprintln!(
            "Step {}: Missing or invalid greedy logits; chosen={}",
            violation.step,
            violation.chosen_id.map_or_else(|| "<missing>".to_string(), |id| id.to_string())
        ),
    }

    if !violation.top_logits.is_empty() {
        eprintln!("  Top logits at step {}:", violation.step);
        for (idx, entry) in violation.top_logits.iter().enumerate() {
            let marker =
                if Some(entry.token_id) == violation.chosen_id { " <-- CHOSEN" } else { "" };
            eprintln!(
                "    {}. token={} logit={:.4}{}",
                idx + 1,
                entry.token_id,
                entry.logit,
                marker
            );
        }
    }
}

impl PartialEq for Violation {
    fn eq(&self, other: &Self) -> bool {
        self.step == other.step
            && self.chosen_id == other.chosen_id
            && self.argmax_token == other.argmax_token
            && self.max_logit.to_bits() == other.max_logit.to_bits()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_chosen_argmax() {
        let report = check_greedy_invariant(&[LogitStep {
            chosen_id: Some(9),
            top_logits: vec![
                TopLogit { token_id: 1, logit: 0.1 },
                TopLogit { token_id: 9, logit: 2.0 },
            ],
        }]);

        assert_eq!(report.step_count, 1);
        assert_eq!(report.top_logit_count, 2);
        assert!(report.violations.is_empty());
    }

    #[test]
    fn reports_mismatched_chosen_token() {
        let report = check_greedy_invariant(&[LogitStep {
            chosen_id: Some(1),
            top_logits: vec![
                TopLogit { token_id: 1, logit: 0.1 },
                TopLogit { token_id: 9, logit: 2.0 },
            ],
        }]);

        assert_eq!(report.violations.len(), 1);
        assert_eq!(report.violations[0].argmax_token, Some(9));
        assert_eq!(report.violations[0].chosen_id, Some(1));
    }

    #[test]
    fn reports_missing_dump() {
        let report = check_greedy_invariant(&[]);

        assert_eq!(report.step_count, 0);
        assert_eq!(report.violations.len(), 1);
        assert_eq!(report.violations[0].argmax_token, None);
    }
}
