use anyhow::{Context, Result};
use clap::{Args, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Subcommand)]
pub enum RepoBoundaryCmd {
    /// Report source/swarm boundary status without changing files.
    Status(StatusArgs),
}

#[derive(Args)]
pub struct StatusArgs {
    /// Source/swarm boundary policy ledger.
    #[arg(long, default_value = "policy/repo-boundary.toml")]
    policy: PathBuf,
    /// Ref that represents source main in this checkout.
    #[arg(long, default_value = "source/main")]
    source_ref: String,
    /// Ref that represents the swarm side being checked.
    #[arg(long, default_value = "HEAD")]
    swarm_ref: String,
    /// Output format.
    #[arg(long, value_enum, default_value = "text")]
    format: OutputFormat,
    /// Exit non-zero if source history is missing or release guards are unsafe.
    #[arg(long, default_value_t = false)]
    fail_on_drift: bool,
}

#[derive(Clone, Debug, ValueEnum)]
pub enum OutputFormat {
    Text,
    Json,
}

#[derive(Debug, Deserialize)]
struct RepoBoundaryPolicy {
    source: RepoRole,
    swarm: RepoRole,
    release_workflows: ReleaseWorkflowPolicy,
    ci: CiPolicy,
}

#[derive(Debug, Deserialize)]
struct RepoRole {
    repo: String,
    branch: String,
}

#[derive(Debug, Deserialize)]
struct ReleaseWorkflowPolicy {
    source_repo_guard: String,
}

#[derive(Debug, Deserialize)]
struct CiPolicy {
    normalized_result_check: String,
}

#[derive(Debug, Serialize)]
struct RepoBoundaryStatus {
    current_repo: RepoKind,
    policy_path: String,
    source_repo: String,
    source_branch: String,
    swarm_repo: String,
    swarm_branch: String,
    origin_remote_url: Option<String>,
    source_remote_url: Option<String>,
    source_remote_configured: bool,
    source_ref: String,
    swarm_ref: String,
    source_ref_reachable_from_swarm_ref: Option<bool>,
    commits_source_has_that_swarm_lacks: Option<u64>,
    commits_swarm_has_that_source_lacks: Option<u64>,
    release_workflow_guard: ReleaseWorkflowGuardReport,
    normalized_result_check: String,
    status: BoundaryVerdict,
    warnings: Vec<String>,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum RepoKind {
    Source,
    Swarm,
    Unknown,
}

#[derive(Debug, Serialize)]
struct ReleaseWorkflowGuardReport {
    guarded: bool,
    guard: String,
    checked_files: Vec<String>,
    unguarded_files: Vec<String>,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum BoundaryVerdict {
    Ok,
    Warn,
    Drift,
}

pub fn run(cmd: RepoBoundaryCmd) -> Result<()> {
    match cmd {
        RepoBoundaryCmd::Status(args) => status(args),
    }
}

fn status(args: StatusArgs) -> Result<()> {
    let policy = load_policy(&args.policy)?;
    let origin_remote_url = git_output(["remote", "get-url", "origin"]).ok();
    let source_remote_url = git_output(["remote", "get-url", "source"]).ok();
    let current_repo =
        classify_repo(origin_remote_url.as_deref(), &policy.source.repo, &policy.swarm.repo);

    let source_ref_reachable = git_status([
        "merge-base",
        "--is-ancestor",
        args.source_ref.as_str(),
        args.swarm_ref.as_str(),
    ]);
    let source_missing_commits = git_count([
        "rev-list".to_string(),
        "--count".to_string(),
        args.source_ref.clone(),
        format!("^{}", args.swarm_ref),
    ]);
    let swarm_only_commits = git_count([
        "rev-list".to_string(),
        "--count".to_string(),
        args.swarm_ref.clone(),
        format!("^{}", args.source_ref),
    ]);
    let release_workflow_guard = scan_release_workflow_guards(
        Path::new(".github/workflows"),
        &policy.release_workflows.source_repo_guard,
    )?;

    let mut warnings = Vec::new();
    if current_repo == RepoKind::Unknown {
        warnings.push("origin remote does not match source or swarm policy repo".to_string());
    }
    if source_remote_url.is_none() {
        warnings.push(
            "source remote is not configured; source reachability is less reliable".to_string(),
        );
    }
    if source_ref_reachable == Some(false) {
        warnings.push(format!("{} is not an ancestor of {}", args.source_ref, args.swarm_ref));
    }
    if source_missing_commits.is_some_and(|count| count > 0) {
        warnings.push(format!("{} has commits missing from {}", args.source_ref, args.swarm_ref));
    }
    if !release_workflow_guard.guarded {
        warnings.push(
            "one or more release-sensitive workflow files lack the source-repo guard".to_string(),
        );
    }

    let status = if source_ref_reachable == Some(false)
        || source_missing_commits.is_some_and(|count| count > 0)
        || !release_workflow_guard.guarded
    {
        BoundaryVerdict::Drift
    } else if !warnings.is_empty()
        || source_ref_reachable.is_none()
        || source_missing_commits.is_none()
        || swarm_only_commits.is_none()
    {
        BoundaryVerdict::Warn
    } else {
        BoundaryVerdict::Ok
    };

    let source_remote_configured = source_remote_url.is_some();
    let report = RepoBoundaryStatus {
        current_repo,
        policy_path: args.policy.display().to_string(),
        source_repo: policy.source.repo,
        source_branch: policy.source.branch,
        swarm_repo: policy.swarm.repo,
        swarm_branch: policy.swarm.branch,
        origin_remote_url,
        source_remote_url,
        source_remote_configured,
        source_ref: args.source_ref,
        swarm_ref: args.swarm_ref,
        source_ref_reachable_from_swarm_ref: source_ref_reachable,
        commits_source_has_that_swarm_lacks: source_missing_commits,
        commits_swarm_has_that_source_lacks: swarm_only_commits,
        release_workflow_guard,
        normalized_result_check: policy.ci.normalized_result_check,
        status,
        warnings,
    };

    match args.format {
        OutputFormat::Text => print_text(&report),
        OutputFormat::Json => println!("{}", serde_json::to_string_pretty(&report)?),
    }

    if args.fail_on_drift && report.status == BoundaryVerdict::Drift {
        anyhow::bail!("repo-boundary drift detected");
    }
    Ok(())
}

fn load_policy(path: &Path) -> Result<RepoBoundaryPolicy> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("failed to read repo-boundary policy: {}", path.display()))?;
    toml::from_str(&raw).context("failed to parse repo-boundary policy")
}

fn classify_repo(origin: Option<&str>, source_repo: &str, swarm_repo: &str) -> RepoKind {
    let Some(origin) = origin else {
        return RepoKind::Unknown;
    };
    let normalized = normalize_repo_url(origin);
    if normalized.ends_with(&normalize_repo_name(source_repo)) {
        RepoKind::Source
    } else if normalized.ends_with(&normalize_repo_name(swarm_repo)) {
        RepoKind::Swarm
    } else {
        RepoKind::Unknown
    }
}

fn normalize_repo_url(value: &str) -> String {
    value.trim().trim_end_matches(".git").replace(':', "/").replace('\\', "/").to_ascii_lowercase()
}

fn normalize_repo_name(value: &str) -> String {
    value.trim().trim_end_matches(".git").replace('\\', "/").to_ascii_lowercase()
}

fn git_output<const N: usize>(args: [&str; N]) -> Result<String> {
    let output = Command::new("git").args(args).output().context("failed to run git")?;
    if !output.status.success() {
        anyhow::bail!("git command exited with {}", output.status);
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn git_status<const N: usize>(args: [&str; N]) -> Option<bool> {
    let output = Command::new("git").args(args).output().ok()?;
    match output.status.code() {
        Some(0) => Some(true),
        Some(1) => Some(false),
        _ => None,
    }
}

fn git_count<I, S>(args: I) -> Option<u64>
where
    I: IntoIterator<Item = S>,
    S: AsRef<std::ffi::OsStr>,
{
    let output = Command::new("git").args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout).trim().parse().ok()
}

fn scan_release_workflow_guards(dir: &Path, guard: &str) -> Result<ReleaseWorkflowGuardReport> {
    let mut checked_files = Vec::new();
    let mut unguarded_files = Vec::new();
    if !dir.exists() {
        return Ok(ReleaseWorkflowGuardReport {
            guarded: true,
            guard: guard.to_string(),
            checked_files,
            unguarded_files,
        });
    }

    for entry in fs::read_dir(dir).with_context(|| format!("failed to read {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !(name.ends_with(".yml") || name.ends_with(".yaml")) {
            continue;
        }
        let raw = fs::read_to_string(&path)
            .with_context(|| format!("failed to read workflow: {}", path.display()))?;
        if !is_release_sensitive_workflow(name, &raw) {
            continue;
        }
        let path_string = path.to_string_lossy().replace('\\', "/");
        checked_files.push(path_string.clone());
        if !raw.contains(guard) {
            unguarded_files.push(path_string);
        }
    }
    checked_files.sort();
    unguarded_files.sort();

    Ok(ReleaseWorkflowGuardReport {
        guarded: unguarded_files.is_empty(),
        guard: guard.to_string(),
        checked_files,
        unguarded_files,
    })
}

fn is_release_sensitive_workflow(name: &str, raw: &str) -> bool {
    let name = name.to_ascii_lowercase();
    if name.contains("release")
        || name.contains("publish")
        || name.contains("sign")
        || name.contains("deploy")
    {
        return true;
    }

    let raw = raw.to_ascii_lowercase();
    [
        "cargo publish",
        "crates.io",
        "gh release",
        "action-gh-release",
        "docker push",
        "cosign",
        "publish_image",
    ]
    .iter()
    .any(|needle| raw.contains(needle))
}

fn print_text(report: &RepoBoundaryStatus) {
    println!("repo-boundary status: {:?}", report.status);
    println!("current_repo: {:?}", report.current_repo);
    println!("source_repo: {} ({})", report.source_repo, report.source_branch);
    println!("swarm_repo: {} ({})", report.swarm_repo, report.swarm_branch);
    println!("source_remote_configured: {}", report.source_remote_configured);
    println!(
        "source_ref_reachable_from_swarm_ref: {}",
        option_bool(report.source_ref_reachable_from_swarm_ref)
    );
    println!(
        "commits_source_has_that_swarm_lacks: {}",
        option_count(report.commits_source_has_that_swarm_lacks)
    );
    println!(
        "commits_swarm_has_that_source_lacks: {}",
        option_count(report.commits_swarm_has_that_source_lacks)
    );
    println!("release_workflows_guarded: {}", report.release_workflow_guard.guarded);
    if !report.release_workflow_guard.checked_files.is_empty() {
        println!(
            "release_workflow_files_checked: {}",
            report.release_workflow_guard.checked_files.join(", ")
        );
    }
    if !report.release_workflow_guard.unguarded_files.is_empty() {
        println!(
            "release_workflow_files_unguarded: {}",
            report.release_workflow_guard.unguarded_files.join(", ")
        );
    }
    println!("normalized_result_check: {}", report.normalized_result_check);
    for warning in &report.warnings {
        println!("warning: {warning}");
    }
}

fn option_bool(value: Option<bool>) -> &'static str {
    match value {
        Some(true) => "true",
        Some(false) => "false",
        None => "unknown",
    }
}

fn option_count(value: Option<u64>) -> String {
    value.map_or_else(|| "unknown".to_string(), |value| value.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_scp_syntax_swarm_repo() {
        assert_eq!(
            classify_repo(
                Some("git@github.com:EffortlessMetrics/bitnet-rs-swarm.git"),
                "EffortlessMetrics/BitNet-rs",
                "EffortlessMetrics/bitnet-rs-swarm",
            ),
            RepoKind::Swarm
        );
    }

    #[test]
    fn release_sensitive_workflow_detection_ignores_release_builds() {
        assert!(!is_release_sensitive_workflow(
            "gpu-smoke.yml",
            "run: cargo build --release --locked"
        ));
        assert!(is_release_sensitive_workflow("release.yml", "name: Release"));
        assert!(is_release_sensitive_workflow("image.yml", "env:\n  PUBLISH_IMAGE: true"));
    }
}
