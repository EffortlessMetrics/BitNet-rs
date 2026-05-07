//! CI lane whitelist checker.
//!
//! Validates that:
//!
//! * `policy/ci-lane-whitelist.toml` parses and every lane has the
//!   required fields (intent, failure_mode, proof_obligation, owner,
//!   evidence, allowed_triggers).
//! * Every workflow job referenced has a backing lane entry.
//! * Lanes that are `default_pr = true` and `expensive = true` are
//!   covered by an explicit, unexpired exception in
//!   `policy/ci-whitelist-exceptions.toml`.
//! * `duplicate_of` references resolve to known lane IDs (or are
//!   `future:` placeholders, which are allowed).
//! * Non-Linux runners declare a multiplier in `[runner_multipliers]`.
//! * Exceptions have not expired.
//!
//! The checker is advisory in PR 02 of the rollout: it always emits a
//! report, and the exit status is controlled by `--fail-on-error`.

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

const REQUIRED_LANE_FIELDS: &[&str] = &[
    "id",
    "workflow",
    "job",
    "kind",
    "tier",
    "owner",
    "intent",
    "failure_mode",
    "proof_obligation",
    "evidence",
    "allowed_triggers",
];

#[derive(Debug, Deserialize)]
struct Whitelist {
    #[serde(default)]
    runner_multipliers: toml::Table,
    #[serde(default, rename = "lane")]
    lanes: Vec<Lane>,
}

#[derive(Debug, Deserialize, Clone)]
struct Lane {
    id: String,
    workflow: String,
    job: String,
    kind: String,
    tier: String,
    #[serde(default)]
    default_pr: bool,
    #[serde(default)]
    blocking: bool,
    runner: String,
    #[serde(default)]
    base_lem: Option<f64>,
    #[serde(default)]
    base_minutes: Option<f64>,
    owner: String,
    intent: String,
    failure_mode: String,
    proof_obligation: String,
    evidence: Vec<String>,
    allowed_triggers: Vec<String>,
    #[serde(default)]
    labels: Vec<String>,
    duplicate_of: Vec<String>,
    #[serde(default)]
    expensive: bool,
    review_after: String,
    expires: String,
}

#[derive(Debug, Deserialize)]
struct Exceptions {
    #[serde(default, rename = "exception")]
    exceptions: Vec<Exception>,
}

#[derive(Debug, Deserialize, Clone)]
struct Exception {
    id: String,
    kind: String,
    lane: String,
    #[serde(default)]
    allowed: bool,
    owner: String,
    #[serde(default)]
    reason: String,
    #[serde(default)]
    created: String,
    #[serde(default)]
    review_after: String,
    expires: String,
}

#[derive(Debug, Default)]
pub struct Report {
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub lane_count: usize,
    pub exception_count: usize,
}

impl Report {
    pub fn is_clean(&self) -> bool {
        self.errors.is_empty()
    }
}

/// Run the whitelist check. Returns `Ok(report)` regardless of findings;
/// the CLI handler decides what exit code to use.
pub fn check(
    whitelist_path: &Path,
    exceptions_path: &Path,
    workflows_dir: &Path,
    report_dir: Option<&Path>,
) -> Result<Report> {
    let mut report = Report::default();

    let whitelist_text = fs::read_to_string(whitelist_path)
        .with_context(|| format!("reading whitelist {}", whitelist_path.display()))?;
    let whitelist: Whitelist = toml::from_str(&whitelist_text)
        .with_context(|| format!("parsing whitelist {}", whitelist_path.display()))?;

    let exceptions_text = fs::read_to_string(exceptions_path)
        .with_context(|| format!("reading exceptions {}", exceptions_path.display()))?;
    let exceptions: Exceptions = toml::from_str(&exceptions_text)
        .with_context(|| format!("parsing exceptions {}", exceptions_path.display()))?;

    report.lane_count = whitelist.lanes.len();
    report.exception_count = exceptions.exceptions.len();

    let lane_ids: BTreeSet<&str> = whitelist.lanes.iter().map(|l| l.id.as_str()).collect();
    let runner_keys: BTreeSet<String> = whitelist.runner_multipliers.keys().cloned().collect();

    let mut lane_dups: BTreeMap<&str, usize> = BTreeMap::new();
    for lane in &whitelist.lanes {
        *lane_dups.entry(lane.id.as_str()).or_default() += 1;
    }
    for (id, count) in &lane_dups {
        if *count > 1 {
            report.errors.push(format!("lane id `{id}` declared {count} times"));
        }
    }

    let today = chrono::Utc::now().date_naive();

    for lane in &whitelist.lanes {
        for field in REQUIRED_LANE_FIELDS {
            let missing = match *field {
                "id" => lane.id.is_empty(),
                "workflow" => lane.workflow.is_empty(),
                "job" => lane.job.is_empty(),
                "kind" => lane.kind.is_empty(),
                "tier" => lane.tier.is_empty(),
                "owner" => lane.owner.is_empty(),
                "intent" => lane.intent.is_empty(),
                "failure_mode" => lane.failure_mode.is_empty(),
                "proof_obligation" => lane.proof_obligation.is_empty(),
                "evidence" => lane.evidence.is_empty(),
                "allowed_triggers" => lane.allowed_triggers.is_empty(),
                _ => false,
            };
            if missing {
                report.errors.push(format!("lane `{}` missing field `{field}`", lane.id));
            }
        }

        if lane.base_lem.is_none() && lane.base_minutes.is_none() {
            report.errors.push(format!(
                "lane `{}` must declare either `base_lem` or `base_minutes`",
                lane.id
            ));
        }

        if lane.blocking && lane.evidence.is_empty() {
            report.errors.push(format!("blocking lane `{}` has no evidence", lane.id));
        }

        if !runner_keys.contains(&lane.runner) {
            report.errors.push(format!(
                "lane `{}` uses runner `{}` not present in [runner_multipliers]",
                lane.id, lane.runner
            ));
        }

        if lane.default_pr && lane.expensive {
            // Must have an exception.
            let covered = exceptions.exceptions.iter().any(|e| {
                e.allowed
                    && e.lane == lane.id
                    && (e.kind.contains("default_pr") || e.kind.contains("expensive"))
            });
            if !covered {
                report.errors.push(format!(
                    "lane `{}` is default_pr=true and expensive=true with no whitelist exception",
                    lane.id
                ));
            }
        }

        for dep in &lane.duplicate_of {
            if dep.starts_with("future:") || dep.is_empty() {
                continue;
            }
            // Allow free-form "duplicate_of" prose like "ci-core-clippy where overlap"
            let head = dep.split_whitespace().next().unwrap_or(dep);
            if !lane_ids.contains(head) {
                report.warnings.push(format!(
                    "lane `{}` duplicate_of references unknown lane `{}`",
                    lane.id, dep
                ));
            }
        }

        if let Ok(d) = chrono::NaiveDate::parse_from_str(&lane.expires, "%Y-%m-%d")
            && d < today
        {
            report.errors.push(format!("lane `{}` expired on {}", lane.id, lane.expires));
        }

        if !lane.review_after.is_empty()
            && let Ok(d) = chrono::NaiveDate::parse_from_str(&lane.review_after, "%Y-%m-%d")
            && d < today
        {
            report.warnings.push(format!(
                "lane `{}` past review_after {} (still within expiry)",
                lane.id, lane.review_after
            ));
        }

        // Workflow file should exist.
        let wf = PathBuf::from(&lane.workflow);
        if !wf.exists() {
            report.warnings.push(format!(
                "lane `{}` references missing workflow `{}`",
                lane.id, lane.workflow
            ));
        }
    }

    // Validate exceptions independently.
    for ex in &exceptions.exceptions {
        if !lane_ids.contains(ex.lane.as_str()) {
            report
                .errors
                .push(format!("exception `{}` references unknown lane `{}`", ex.id, ex.lane));
        }
        if let Ok(d) = chrono::NaiveDate::parse_from_str(&ex.expires, "%Y-%m-%d")
            && d < today
        {
            report.errors.push(format!(
                "exception `{}` expired on {} (owner: {})",
                ex.id, ex.expires, ex.owner
            ));
        }
    }

    // Light cross-check between the whitelist and the workflows directory:
    // for every workflow the whitelist mentions, confirm it exists; for now
    // we deliberately do not require every workflow on disk to map to a
    // lane (we are still rolling lanes in).
    if workflows_dir.exists() {
        let entries = fs::read_dir(workflows_dir)
            .with_context(|| format!("reading {}", workflows_dir.display()))?;
        let mut on_disk: BTreeSet<String> = BTreeSet::new();
        for e in entries.flatten() {
            let p = e.path();
            if p.extension().and_then(|s| s.to_str()) == Some("yml") {
                on_disk.insert(format!(".github/workflows/{}", e.file_name().to_string_lossy()));
            }
        }
        let referenced: BTreeSet<String> =
            whitelist.lanes.iter().map(|l| l.workflow.clone()).collect();
        for wf in referenced.difference(&on_disk) {
            report.warnings.push(format!("whitelist references workflow `{wf}` not on disk"));
        }
    }

    if let Some(dir) = report_dir {
        fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        let json = serde_json::json!({
            "schema_version": 1,
            "errors": report.errors,
            "warnings": report.warnings,
            "lane_count": report.lane_count,
            "exception_count": report.exception_count,
        });
        fs::write(dir.join("ci-lane-whitelist.json"), serde_json::to_string_pretty(&json)?)?;

        let mut md = String::new();
        md.push_str("# CI Lane Whitelist Report\n\n");
        md.push_str(&format!("- lanes: {}\n", report.lane_count));
        md.push_str(&format!("- exceptions: {}\n", report.exception_count));
        md.push_str(&format!("- errors: {}\n", report.errors.len()));
        md.push_str(&format!("- warnings: {}\n", report.warnings.len()));
        if !report.errors.is_empty() {
            md.push_str("\n## Errors\n");
            for e in &report.errors {
                md.push_str(&format!("- {e}\n"));
            }
        }
        if !report.warnings.is_empty() {
            md.push_str("\n## Warnings\n");
            for w in &report.warnings {
                md.push_str(&format!("- {w}\n"));
            }
        }
        fs::write(dir.join("ci-lane-whitelist.md"), md)?;
    }

    Ok(report)
}

/// CLI entry point invoked from `xtask`.
pub fn run(
    workflows: PathBuf,
    whitelist: PathBuf,
    exceptions: PathBuf,
    report_dir: Option<PathBuf>,
    fail_on_error: bool,
) -> Result<()> {
    let report = check(&whitelist, &exceptions, &workflows, report_dir.as_deref())?;

    println!(
        "ci-lane-whitelist: {} lanes, {} exceptions",
        report.lane_count, report.exception_count
    );
    for w in &report.warnings {
        println!("warning: {w}");
    }
    for e in &report.errors {
        println!("error: {e}");
    }

    if fail_on_error && !report.is_clean() {
        bail!("ci-lane-whitelist check failed: {} errors", report.errors.len());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_tmp(name: &str, body: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ci-lanes-{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        let p = dir.join(name);
        let mut f = fs::File::create(&p).unwrap();
        f.write_all(body.as_bytes()).unwrap();
        p
    }

    fn minimal_whitelist() -> &'static str {
        r#"
schema_version = "1.0"
[runner_multipliers]
ubuntu_22_04 = 1.0
[[lane]]
id = "x"
workflow = ".github/workflows/none.yml"
job = "X"
kind = "rust"
tier = "frontdoor"
default_pr = true
blocking = true
runner = "ubuntu_22_04"
base_lem = 1
owner = "team"
intent = "intent"
failure_mode = "mode"
proof_obligation = "obligation"
evidence = ["log"]
allowed_triggers = ["pull_request"]
duplicate_of = []
review_after = "3000-01-01"
expires = "3000-01-01"
"#
    }

    #[test]
    fn parses_minimal_whitelist() {
        let wl = write_tmp("wl.toml", minimal_whitelist());
        let ex = write_tmp("ex.toml", "schema_version = \"1.0\"\n");
        let r = check(&wl, &ex, Path::new("nope"), None).unwrap();
        // workflow file is missing, so we expect a warning but no errors.
        assert!(r.errors.is_empty(), "errors: {:?}", r.errors);
        assert_eq!(r.lane_count, 1);
    }

    #[test]
    fn detects_unknown_runner() {
        let body = r#"
schema_version = "1.0"
[runner_multipliers]
ubuntu_22_04 = 1.0
[[lane]]
id = "x"
workflow = ".github/workflows/none.yml"
job = "X"
kind = "rust"
tier = "frontdoor"
default_pr = true
blocking = true
runner = "moon_runner"
base_lem = 1
owner = "team"
intent = "intent"
failure_mode = "mode"
proof_obligation = "obligation"
evidence = ["log"]
allowed_triggers = ["pull_request"]
duplicate_of = []
review_after = "3000-01-01"
expires = "3000-01-01"
"#;
        let wl = write_tmp("wl_runner.toml", body);
        let ex = write_tmp("ex_runner.toml", "schema_version = \"1.0\"\n");
        let r = check(&wl, &ex, Path::new("nope"), None).unwrap();
        assert!(r.errors.iter().any(|e| e.contains("moon_runner")));
    }

    #[test]
    fn detects_expired_exception() {
        let wl = write_tmp("wl_exp.toml", minimal_whitelist());
        let ex_body = r#"
schema_version = "1.0"
[[exception]]
id = "e"
kind = "default_pr_compile_lane"
lane = "x"
allowed = true
owner = "o"
expires = "1999-01-01"
"#;
        let ex = write_tmp("ex_exp.toml", ex_body);
        let r = check(&wl, &ex, Path::new("nope"), None).unwrap();
        assert!(r.errors.iter().any(|e| e.contains("expired")));
    }
}
