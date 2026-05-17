use anyhow::{Context, Result, bail};
use serde_json::Value;
use std::fs;
use std::path::Path;

pub const EXIT_ARGMAX_MISMATCH: i32 = 7;

#[derive(Debug, Default, PartialEq)]
pub struct GreedyArgmaxReport {
    pub steps: usize,
    pub violations: Vec<GreedyViolation>,
    pub missing_steps: Vec<usize>,
}

#[derive(Debug, PartialEq)]
pub struct GreedyViolation {
    pub step: usize,
    pub argmax_token: i64,
    pub chosen_id: i64,
    pub max_logit: f64,
    pub top_logits: Vec<(i64, f64)>,
}

impl GreedyArgmaxReport {
    pub fn is_valid(&self) -> bool {
        self.violations.is_empty() && self.missing_steps.is_empty() && self.steps > 0
    }
}

pub fn check_greedy_argmax(path: &Path) -> Result<GreedyArgmaxReport> {
    let text = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let data: Value =
        serde_json::from_str(&text).with_context(|| format!("parsing JSON {}", path.display()))?;
    let Some(logits_dump) = data.get("logits_dump").and_then(Value::as_array) else {
        bail!("No logits dump found in JSON");
    };
    if logits_dump.is_empty() {
        bail!("No logits dump found in JSON");
    }

    let mut report =
        GreedyArgmaxReport { steps: logits_dump.len(), ..GreedyArgmaxReport::default() };
    for (step_idx, step) in logits_dump.iter().enumerate() {
        let Some(top_logits) = step.get("top_logits").and_then(Value::as_array) else {
            report.missing_steps.push(step_idx);
            continue;
        };
        let Some(chosen_id) = step.get("chosen_id").and_then(Value::as_i64) else {
            report.missing_steps.push(step_idx);
            continue;
        };
        if top_logits.is_empty() {
            report.missing_steps.push(step_idx);
            continue;
        }

        let mut argmax_token = None;
        let mut max_logit = f64::NEG_INFINITY;
        let mut top = Vec::new();
        for entry in top_logits {
            let Some(token_id) = entry.get("token_id").and_then(Value::as_i64) else {
                continue;
            };
            let Some(logit) = entry.get("logit").and_then(Value::as_f64) else {
                continue;
            };
            top.push((token_id, logit));
            if !logit.is_finite() || !(-1.0e10..1.0e10).contains(&logit) {
                continue;
            }
            if logit > max_logit {
                max_logit = logit;
                argmax_token = Some(token_id);
            }
        }

        let Some(argmax_token) = argmax_token else {
            report.missing_steps.push(step_idx);
            continue;
        };
        if argmax_token != chosen_id {
            report.violations.push(GreedyViolation {
                step: step_idx,
                argmax_token,
                chosen_id,
                max_logit,
                top_logits: top.into_iter().take(5).collect(),
            });
        }
    }

    Ok(report)
}

pub fn print_greedy_argmax_report(report: &GreedyArgmaxReport) {
    for step in &report.missing_steps {
        eprintln!("Step {step}: Missing data");
    }
    for violation in &report.violations {
        eprintln!(
            "Step {}: Greedy violation! argmax={} (logit={:.4}) but chosen={}",
            violation.step, violation.argmax_token, violation.max_logit, violation.chosen_id,
        );
        eprintln!("  Top logits at step {}:", violation.step);
        for (index, (token_id, logit)) in violation.top_logits.iter().enumerate() {
            let marker = if *token_id == violation.chosen_id { " <-- CHOSEN" } else { "" };
            eprintln!("    {}. token={} logit={:.4}{}", index + 1, token_id, logit, marker);
        }
    }
    if report.is_valid() {
        println!("✓ Greedy invariant holds for all {} steps", report.steps);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn accepts_matching_argmax() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("ok.json");
        fs::write(&path, r#"{"logits_dump":[{"chosen_id":2,"top_logits":[{"token_id":1,"logit":0.1},{"token_id":2,"logit":0.3}]}]}"#).unwrap();
        let report = check_greedy_argmax(&path).unwrap();
        assert!(report.is_valid());
        assert_eq!(report.steps, 1);
    }

    #[test]
    fn reports_mismatch() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("bad.json");
        fs::write(&path, r#"{"logits_dump":[{"chosen_id":1,"top_logits":[{"token_id":1,"logit":0.1},{"token_id":2,"logit":0.3}]}]}"#).unwrap();
        let report = check_greedy_argmax(&path).unwrap();
        assert!(!report.is_valid());
        assert_eq!(report.violations[0].argmax_token, 2);
    }
}
