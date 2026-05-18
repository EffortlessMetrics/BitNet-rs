//! Greedy argmax invariant checker.
//!
//! Native Rust port of `scripts/check_greedy_argmax.py`. Reads a JSON file
//! produced by `bitnet run --json-out ...` together with `--dump-logit-steps`
//! and `--logits-topk`, and verifies that at every recorded step the chosen
//! token id matches the argmax of the recorded top logits after filtering
//! NaN/inf and out-of-range values.
//!
//! # Exit codes
//!
//! - `0` — invariant holds for every step
//! - `7` — invariant violated for at least one step (`EXIT_ARGMAX_MISMATCH`)
//! - non-zero anyhow error — JSON file missing or malformed
//!
//! The exit code `7` matches the contract of the previous Python script and
//! the `scripts/validate_all.sh` invocation that propagates it.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

/// One entry of the `top_logits` array recorded for a step.
#[derive(Debug, Deserialize)]
struct LogitEntry {
    token_id: i64,
    logit: f64,
}

/// One step in the `logits_dump` array.
#[derive(Debug, Deserialize)]
struct LogitStep {
    #[serde(default)]
    top_logits: Vec<LogitEntry>,
    #[serde(default)]
    chosen_id: Option<i64>,
}

/// Wrapper for the `logits_dump` array in the JSON document.
#[derive(Debug, Deserialize)]
struct LogitsDump {
    #[serde(default)]
    logits_dump: Vec<LogitStep>,
}

/// Match the Python filter `-1e10 < x < 1e10` after the `isinstance` check.
///
/// `serde_json` already coerces the value to `f64`, so we only need to reject
/// NaN/inf and values outside the sane numeric range that the Python check
/// imposed.
fn is_usable_logit(value: f64) -> bool {
    value.is_finite() && value > -1e10 && value < 1e10
}

/// Check that every step's `chosen_id` matches the argmax over its
/// `top_logits` (filtered to finite, in-range values).
///
/// Diagnostic output mirrors the Python reference: success line goes to
/// stdout, violations and missing-data warnings go to stderr.
fn check_steps(steps: &[LogitStep]) -> bool {
    if steps.is_empty() {
        eprintln!("No logits dump found in JSON");
        return false;
    }

    let mut all_valid = true;
    for (step_idx, step) in steps.iter().enumerate() {
        if step.top_logits.is_empty() || step.chosen_id.is_none() {
            eprintln!("Step {step_idx}: Missing data");
            continue;
        }
        let Some(chosen_id) = step.chosen_id else {
            continue;
        };

        let mut argmax_token: Option<i64> = None;
        let mut max_logit = f64::NEG_INFINITY;
        for entry in &step.top_logits {
            if !is_usable_logit(entry.logit) {
                continue;
            }
            if entry.logit > max_logit {
                max_logit = entry.logit;
                argmax_token = Some(entry.token_id);
            }
        }

        if argmax_token != Some(chosen_id) {
            let arg_display =
                argmax_token.map(|t| t.to_string()).unwrap_or_else(|| "None".to_string());
            eprintln!(
                "Step {step_idx}: Greedy violation! argmax={arg_display} (logit={max_logit:.4}) but chosen={chosen_id}"
            );
            eprintln!("  Top logits at step {step_idx}:");
            for (i, entry) in step.top_logits.iter().take(5).enumerate() {
                let marker = if entry.token_id == chosen_id { " <-- CHOSEN" } else { "" };
                eprintln!(
                    "    {idx}. token={tok} logit={lg:.4}{marker}",
                    idx = i + 1,
                    tok = entry.token_id,
                    lg = entry.logit,
                );
            }
            all_valid = false;
        }
    }

    if all_valid {
        println!("\u{2713} Greedy invariant holds for all {} steps", steps.len());
    }
    all_valid
}

/// Load and check a JSON file at `path`.
///
/// Returns `Ok(true)` if the invariant holds for every step, `Ok(false)` if
/// any step is a violation, and `Err` if the file cannot be read or parsed.
fn check_file(path: &Path) -> Result<bool> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let dump: LogitsDump = serde_json::from_str(&text)
        .with_context(|| format!("failed to parse JSON from {}", path.display()))?;
    Ok(check_steps(&dump.logits_dump))
}

/// Entry point used by the xtask subcommand.
///
/// Exits the process with code `7` when the invariant is violated so the
/// behavior matches the previous Python script. Returns a normal `Result`
/// for I/O / parse failures so xtask's standard error reporting handles them.
pub fn run(path: &Path) -> Result<()> {
    if check_file(path)? {
        Ok(())
    } else {
        std::process::exit(7);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn step(chosen: Option<i64>, entries: &[(i64, f64)]) -> LogitStep {
        LogitStep {
            top_logits: entries
                .iter()
                .map(|&(token_id, logit)| LogitEntry { token_id, logit })
                .collect(),
            chosen_id: chosen,
        }
    }

    #[test]
    fn empty_dump_is_invalid() {
        assert!(!check_steps(&[]));
    }

    #[test]
    fn chosen_matches_argmax_passes() {
        let steps = vec![
            step(Some(42), &[(7, 0.1), (42, 5.0), (3, 2.3)]),
            step(Some(11), &[(11, 9.9), (12, 9.8)]),
        ];
        assert!(check_steps(&steps));
    }

    #[test]
    fn chosen_does_not_match_argmax_fails() {
        let steps = vec![step(Some(7), &[(7, 1.0), (42, 5.0)])];
        assert!(!check_steps(&steps));
    }

    #[test]
    fn nan_and_inf_logits_are_filtered() {
        let steps = vec![step(Some(42), &[(99, f64::NAN), (100, f64::INFINITY), (42, 5.0)])];
        assert!(check_steps(&steps));
    }

    #[test]
    fn out_of_range_logits_are_filtered() {
        let steps = vec![step(Some(42), &[(99, 1e20), (42, 5.0)])];
        assert!(check_steps(&steps));
    }

    #[test]
    fn missing_chosen_id_is_skipped_not_failed() {
        let steps = vec![step(None, &[(1, 1.0), (2, 2.0)]), step(Some(3), &[(3, 9.0), (4, 1.0)])];
        assert!(check_steps(&steps));
    }

    #[test]
    fn missing_top_logits_is_skipped_not_failed() {
        let steps = vec![step(Some(5), &[]), step(Some(7), &[(7, 1.0)])];
        assert!(check_steps(&steps));
    }

    #[test]
    fn all_logits_filtered_yields_violation() {
        // No usable logit means argmax_token stays None; chosen is Some(_),
        // so the invariant is violated (matches the Python reference).
        let steps = vec![step(Some(1), &[(1, f64::NAN), (2, 1e20)])];
        assert!(!check_steps(&steps));
    }

    #[test]
    fn check_file_parses_well_formed_json() -> Result<()> {
        let mut f = tempfile::NamedTempFile::new()?;
        writeln!(
            f,
            r#"{{
                "logits_dump": [
                    {{"step": 0, "top_logits": [{{"token_id": 7, "logit": 1.0}}, {{"token_id": 42, "logit": 5.0}}], "chosen_id": 42}}
                ]
            }}"#
        )?;
        let ok = check_file(f.path())?;
        assert!(ok);
        Ok(())
    }

    #[test]
    fn check_file_reports_violation() -> Result<()> {
        let mut f = tempfile::NamedTempFile::new()?;
        writeln!(
            f,
            r#"{{
                "logits_dump": [
                    {{"step": 0, "top_logits": [{{"token_id": 7, "logit": 1.0}}, {{"token_id": 42, "logit": 5.0}}], "chosen_id": 7}}
                ]
            }}"#
        )?;
        let ok = check_file(f.path())?;
        assert!(!ok);
        Ok(())
    }

    #[test]
    fn check_file_missing_file_is_error() {
        let result = check_file(Path::new("/nonexistent/check_greedy_argmax_input.json"));
        assert!(result.is_err());
    }

    #[test]
    fn check_file_malformed_json_is_error() -> Result<()> {
        let mut f = tempfile::NamedTempFile::new()?;
        writeln!(f, "{{not valid json")?;
        let result = check_file(f.path());
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn check_file_missing_logits_dump_field_is_invalid() -> Result<()> {
        // `logits_dump` defaults to empty when absent → check returns false
        // (matching the Python script's "No logits dump found" branch).
        let mut f = tempfile::NamedTempFile::new()?;
        writeln!(f, r#"{{"unrelated_key": 1}}"#)?;
        let ok = check_file(f.path())?;
        assert!(!ok);
        Ok(())
    }
}
