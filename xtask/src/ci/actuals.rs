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
//! The command supports two inputs:
//!
//! * a synthetic single-job record from command-line arguments for
//!   local smoke tests and fixtures;
//! * a GitHub Actions jobs API response, used by the `CI Actuals`
//!   workflow to capture completed workflow-run timing.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Actuals {
    pub schema_version: u32,
    pub repo: String,
    pub sha: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pr: Option<u64>,
    pub workflow: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workflow_run_id: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub head_branch: Option<String>,
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

#[derive(Debug, Clone)]
pub struct ActualsOptions {
    pub repo: String,
    pub sha: String,
    pub pr: Option<u64>,
    pub workflow: String,
    pub workflow_run_id: Option<u64>,
    pub event: Option<String>,
    pub head_branch: Option<String>,
    pub job_name: Option<String>,
    pub runner: Option<String>,
    pub actual_seconds: Option<u64>,
    pub estimated_lem: Option<f64>,
    pub conclusion: Option<String>,
    pub cache_hit: bool,
    pub github_jobs_json: Option<PathBuf>,
    pub json_out: PathBuf,
    pub summary_out: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct ActionsJobsResponse {
    #[serde(default)]
    jobs: Vec<ActionsJob>,
}

#[derive(Debug, Deserialize)]
struct ActionsJob {
    name: String,
    #[serde(default)]
    conclusion: Option<String>,
    #[serde(default)]
    started_at: Option<String>,
    #[serde(default)]
    completed_at: Option<String>,
    #[serde(default)]
    labels: Vec<String>,
    #[serde(default)]
    runner_name: Option<String>,
}

/// Compute LEM from a runner's wall-clock seconds and a multiplier
/// table. Multipliers default to the values in
/// `policy/ci-budget.toml` when not overridden.
pub fn lem_from_seconds(seconds: u64, runner: &str) -> f64 {
    let runner = runner.to_ascii_lowercase().replace('_', "-");
    let multiplier = if runner.contains("macos") || runner.contains("darwin") {
        10.0
    } else if runner.contains("windows") {
        2.0
    } else if runner.contains("gpu-docker") {
        6.0
    } else {
        1.0
    };
    let minutes = seconds as f64 / 60.0;
    (minutes * multiplier * 100.0).round() / 100.0
}

pub fn run(options: ActualsOptions) -> Result<()> {
    let jobs = if let Some(path) = &options.github_jobs_json {
        jobs_from_actions_api(path)?
    } else {
        synthetic_job(&options)
    };

    let actuals = Actuals {
        schema_version: 1,
        repo: options.repo,
        sha: options.sha,
        pr: options.pr,
        workflow: options.workflow,
        workflow_run_id: options.workflow_run_id,
        event: options.event,
        head_branch: options.head_branch,
        jobs,
    };

    if let Some(parent) = options.json_out.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    let body = serde_json::to_string_pretty(&actuals)?;
    fs::write(&options.json_out, &body)
        .with_context(|| format!("writing {}", options.json_out.display()))?;
    if let Some(summary_out) = &options.summary_out {
        write_summary(&actuals, summary_out)?;
    }
    println!("ci-actuals: wrote {} ({} jobs)", options.json_out.display(), actuals.jobs.len());
    Ok(())
}

fn synthetic_job(options: &ActualsOptions) -> Vec<JobActual> {
    let mut jobs = Vec::new();
    if let (Some(name), Some(runner_id), Some(seconds)) =
        (options.job_name.clone(), options.runner.clone(), options.actual_seconds)
    {
        let actual_lem = lem_from_seconds(seconds, &runner_id);
        jobs.push(JobActual {
            name,
            runner: runner_id,
            estimated_lem: options.estimated_lem,
            actual_seconds: seconds,
            actual_lem,
            conclusion: options.conclusion.clone().unwrap_or_else(|| "success".into()),
            cache_hit: options.cache_hit,
            risk_packs: vec![],
        });
    }
    jobs
}

fn jobs_from_actions_api(path: &Path) -> Result<Vec<JobActual>> {
    let body = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let response: ActionsJobsResponse =
        serde_json::from_str(&body).with_context(|| format!("parsing {}", path.display()))?;
    Ok(response.jobs.into_iter().filter_map(actions_job_to_actual).collect())
}

fn actions_job_to_actual(job: ActionsJob) -> Option<JobActual> {
    let started = parse_github_timestamp(job.started_at.as_deref())?;
    let completed = parse_github_timestamp(job.completed_at.as_deref())?;
    let seconds = completed.signed_duration_since(started).num_seconds().max(0) as u64;
    let runner = runner_label(&job);
    Some(JobActual {
        name: job.name,
        runner: runner.clone(),
        estimated_lem: None,
        actual_seconds: seconds,
        actual_lem: lem_from_seconds(seconds, &runner),
        conclusion: job.conclusion.unwrap_or_else(|| "unknown".into()),
        cache_hit: false,
        risk_packs: vec![],
    })
}

fn parse_github_timestamp(value: Option<&str>) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(value?).ok().map(|dt| dt.with_timezone(&Utc))
}

fn runner_label(job: &ActionsJob) -> String {
    job.labels
        .iter()
        .find(|label| {
            let label = label.to_ascii_lowercase();
            label.contains("ubuntu")
                || label.contains("linux")
                || label.contains("macos")
                || label.contains("windows")
                || label.contains("gpu")
        })
        .cloned()
        .or_else(|| job.runner_name.clone())
        .unwrap_or_else(|| "unknown".into())
}

fn write_summary(actuals: &Actuals, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    let mut jobs = actuals.jobs.clone();
    jobs.sort_by(|a, b| b.actual_lem.total_cmp(&a.actual_lem));
    let total_lem: f64 = jobs
        .iter()
        .filter(|job| !matches!(job.conclusion.as_str(), "skipped" | "cancelled"))
        .map(|job| job.actual_lem)
        .sum();
    let total_seconds: u64 = jobs
        .iter()
        .filter(|job| !matches!(job.conclusion.as_str(), "skipped" | "cancelled"))
        .map(|job| job.actual_seconds)
        .sum();
    let mut summary = format!(
        "## CI Actuals - `{}`\n\n- Jobs: {}\n- Wall time sum: {:.1} min\n- LEM actual: {:.1}\n",
        actuals.workflow,
        jobs.len(),
        total_seconds as f64 / 60.0,
        total_lem
    );
    if !jobs.is_empty() {
        summary.push_str("\n| Job | Runner | Wall min | LEM | Conclusion |\n");
        summary.push_str("|---|---|---:|---:|---|\n");
        for job in jobs.iter().take(15) {
            summary.push_str(&format!(
                "| `{}` | `{}` | {:.2} | {:.2} | {} |\n",
                job.name,
                job.runner,
                job.actual_seconds as f64 / 60.0,
                job.actual_lem,
                job.conclusion
            ));
        }
    }
    fs::write(path, summary).with_context(|| format!("writing {}", path.display()))
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

    #[test]
    fn parses_actions_jobs_into_actuals() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("jobs.json");
        fs::write(
            &input,
            r#"{
              "jobs": [
                {
                  "name": "Build",
                  "conclusion": "success",
                  "started_at": "2026-05-09T09:00:00Z",
                  "completed_at": "2026-05-09T09:02:00Z",
                  "labels": ["ubuntu-22.04"]
                },
                {
                  "name": "macOS",
                  "conclusion": "success",
                  "started_at": "2026-05-09T09:00:00Z",
                  "completed_at": "2026-05-09T09:01:00Z",
                  "labels": ["macos-14"]
                }
              ]
            }"#,
        )
        .unwrap();

        let jobs = jobs_from_actions_api(&input).unwrap();
        assert_eq!(jobs.len(), 2);
        assert_eq!(jobs[0].actual_seconds, 120);
        assert_eq!(jobs[0].actual_lem, 2.0);
        assert_eq!(jobs[1].actual_lem, 10.0);
    }

    #[test]
    fn skips_actions_jobs_without_complete_timestamps() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("jobs.json");
        fs::write(
            &input,
            r#"{
              "jobs": [
                {
                  "name": "Pending",
                  "conclusion": null,
                  "started_at": "2026-05-09T09:00:00Z",
                  "completed_at": null,
                  "labels": ["ubuntu-latest"]
                }
              ]
            }"#,
        )
        .unwrap();

        let jobs = jobs_from_actions_api(&input).unwrap();
        assert!(jobs.is_empty());
    }

    #[test]
    fn summary_excludes_skipped_and_cancelled_totals() {
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("summary.md");
        let actuals = Actuals {
            schema_version: 1,
            repo: "repo".into(),
            sha: "sha".into(),
            pr: None,
            workflow: "CI".into(),
            workflow_run_id: Some(42),
            event: Some("pull_request".into()),
            head_branch: Some("branch".into()),
            jobs: vec![
                JobActual {
                    name: "Build".into(),
                    runner: "ubuntu-latest".into(),
                    estimated_lem: None,
                    actual_seconds: 60,
                    actual_lem: 1.0,
                    conclusion: "success".into(),
                    cache_hit: false,
                    risk_packs: vec![],
                },
                JobActual {
                    name: "Skipped".into(),
                    runner: "ubuntu-latest".into(),
                    estimated_lem: None,
                    actual_seconds: 600,
                    actual_lem: 10.0,
                    conclusion: "skipped".into(),
                    cache_hit: false,
                    risk_packs: vec![],
                },
            ],
        };

        write_summary(&actuals, &output).unwrap();
        let summary = fs::read_to_string(output).unwrap();
        assert!(summary.contains("LEM actual: 1.0"));
        assert!(summary.contains("Jobs: 2"));
    }
}
