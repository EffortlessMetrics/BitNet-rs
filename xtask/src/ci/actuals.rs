//! `xtask ci actuals` — emit a normalised LEM-actuals artefact from a
//! GitHub Actions workflow run.
//!
//! The artefact format is `target/ci/ci-actuals.json`:
//!
//! ```json
//! {
//!   "schema_version": 1,
//!   "repo": "BitNet-rs",
//!   "sha": "<HEAD SHA>",
//!   "pr": <PR number or null>,
//!   "workflow": "<workflow name>",
//!   "jobs": [
//!     {
//!       "name": "Build & Test",
//!       "runner": "ubuntu-22.04",
//!       "estimated_lem": 22,
//!       "actual_seconds": 840,
//!       "actual_lem": 14,
//!       "conclusion": "success",
//!       "cache_hit": true,
//!       "risk_packs": ["qk256"]
//!     }
//!   ]
//! }
//! ```
//!
//! PR 16 ships this scaffold so future PRs (notably PR 20 — learned
//! estimates) have a concrete consumer surface. The current
//! implementation only knows how to emit a synthetic record from
//! command-line arguments; production runs will swap in a GitHub
//! API call.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Actuals {
    pub schema_version: u32,
    pub repo: String,
    pub sha: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pr: Option<u64>,
    pub workflow: String,
    pub jobs: Vec<JobActual>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobActual {
    pub name: String,
    pub runner: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimated_lem: Option<f64>,
    pub actual_seconds: u64,
    pub actual_lem: f64,
    pub conclusion: String,
    #[serde(default)]
    pub cache_hit: bool,
    #[serde(default)]
    pub risk_packs: Vec<String>,
}

/// Compute LEM from a runner's wall-clock seconds and a multiplier
/// table. Multipliers default to the values in
/// `policy/ci-budget.toml` when not overridden.
pub fn lem_from_seconds(seconds: u64, runner: &str) -> f64 {
    let multiplier = match runner {
        "ubuntu-22.04" | "ubuntu-latest" | "ubuntu_22_04" | "ubuntu_latest" => 1.0,
        "windows-latest" | "windows_latest" => 2.0,
        "macos-14" | "macos-latest" | "macos_14" | "macos_latest" => 10.0,
        "gpu-docker" | "gpu_docker" => 6.0,
        _ => 1.0,
    };
    let minutes = seconds as f64 / 60.0;
    (minutes * multiplier * 100.0).round() / 100.0
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    repo: String,
    sha: String,
    pr: Option<u64>,
    workflow: String,
    job_name: Option<String>,
    runner: Option<String>,
    actual_seconds: Option<u64>,
    estimated_lem: Option<f64>,
    conclusion: Option<String>,
    cache_hit: bool,
    json_out: PathBuf,
) -> Result<()> {
    let mut jobs = Vec::new();
    if let (Some(name), Some(runner_id), Some(seconds)) =
        (job_name.clone(), runner.clone(), actual_seconds)
    {
        let actual_lem = lem_from_seconds(seconds, &runner_id);
        jobs.push(JobActual {
            name,
            runner: runner_id,
            estimated_lem,
            actual_seconds: seconds,
            actual_lem,
            conclusion: conclusion.unwrap_or_else(|| "success".into()),
            cache_hit,
            risk_packs: vec![],
        });
    }

    let actuals = Actuals {
        schema_version: 1,
        repo,
        sha,
        pr,
        workflow,
        jobs,
    };

    if let Some(parent) = json_out.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    let body = serde_json::to_string_pretty(&actuals)?;
    fs::write(&json_out, &body)
        .with_context(|| format!("writing {}", json_out.display()))?;
    println!("ci-actuals: wrote {} ({} jobs)", json_out.display(), actuals.jobs.len());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lem_for_ubuntu_is_seconds_div_60() {
        assert_eq!(lem_from_seconds(60, "ubuntu-22.04"), 1.0);
        assert_eq!(lem_from_seconds(120, "ubuntu-latest"), 2.0);
    }

    #[test]
    fn lem_for_macos_is_10x() {
        assert_eq!(lem_from_seconds(60, "macos-14"), 10.0);
    }

    #[test]
    fn lem_for_windows_is_2x() {
        assert_eq!(lem_from_seconds(60, "windows-latest"), 2.0);
    }

    #[test]
    fn lem_for_unknown_runner_is_1x() {
        assert_eq!(lem_from_seconds(60, "novel-runner"), 1.0);
    }
}
