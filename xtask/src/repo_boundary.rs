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
    #[arg(long)]
    source_ref: Option<String>,
    /// Ref that represents the swarm side being checked.
    #[arg(long)]
    swarm_ref: Option<String>,
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
    swarm_remote_url: Option<String>,
    source_remote_configured: bool,
    swarm_remote_configured: bool,
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
    let named_source_remote_url = git_output(["remote", "get-url", "source"]).ok();
    let swarm_remote_url = git_output(["remote", "get-url", "swarm"]).ok();
    let current_repo =
        classify_repo(origin_remote_url.as_deref(), &policy.source.repo, &policy.swarm.repo);
    let source_remote_url = named_source_remote_url.or_else(|| match current_repo {
        RepoKind::Source => origin_remote_url.clone(),
        RepoKind::Swarm | RepoKind::Unknown => None,
    });
    let source_ref = args.source_ref.unwrap_or_else(|| default_source_ref(&current_repo));
    let swarm_ref = args.swarm_ref.unwrap_or_else(|| "HEAD".to_string());

    let source_ref_reachable =
        git_status(["merge-base", "--is-ancestor", source_ref.as_str(), swarm_ref.as_str()]);
    let source_missing_commits = git_count([
        "rev-list".to_string(),
        "--count".to_string(),
        source_ref.clone(),
        format!("^{}", swarm_ref),
    ]);
    let swarm_only_commits = git_count([
        "rev-list".to_string(),
        "--count".to_string(),
        swarm_ref.clone(),
        format!("^{}", source_ref),
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
            "source repo remote is not configured; source reachability is less reliable"
                .to_string(),
        );
    }
    if source_ref_reachable == Some(false) {
        warnings.push(format!("{} is not an ancestor of {}", source_ref, swarm_ref));
    } else if source_ref_reachable.is_none() {
        warnings.push(format!(
            "could not verify whether {} is an ancestor of {}",
            source_ref, swarm_ref
        ));
    }
    if source_missing_commits.is_some_and(|count| count > 0) {
        warnings.push(format!("{} has commits missing from {}", source_ref, swarm_ref));
    } else if source_missing_commits.is_none() {
        warnings
            .push(format!("could not count commits in {} missing from {}", source_ref, swarm_ref));
    }
    if swarm_only_commits.is_none() {
        warnings
            .push(format!("could not count commits in {} missing from {}", swarm_ref, source_ref));
    }
    if !release_workflow_guard.guarded {
        warnings.push(
            "one or more release-sensitive workflow files lack the source-repo guard".to_string(),
        );
    }

    let status = boundary_verdict(
        source_ref_reachable,
        source_missing_commits,
        swarm_only_commits,
        release_workflow_guard.guarded,
        !warnings.is_empty(),
    );

    let source_remote_configured = source_remote_url.is_some();
    let swarm_remote_configured = swarm_remote_url.is_some();
    let report = RepoBoundaryStatus {
        current_repo,
        policy_path: args.policy.display().to_string(),
        source_repo: policy.source.repo,
        source_branch: policy.source.branch,
        swarm_repo: policy.swarm.repo,
        swarm_branch: policy.swarm.branch,
        origin_remote_url,
        source_remote_url,
        swarm_remote_url,
        source_remote_configured,
        swarm_remote_configured,
        source_ref,
        swarm_ref,
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

fn default_source_ref(current_repo: &RepoKind) -> String {
    if git_ref_exists("source/main") {
        "source/main".to_string()
    } else if *current_repo == RepoKind::Source && git_ref_exists("origin/main") {
        "origin/main".to_string()
    } else {
        "source/main".to_string()
    }
}

fn normalize_repo_url(value: &str) -> String {
    value.trim().trim_end_matches(".git").replace(':', "/").replace('\\', "/").to_ascii_lowercase()
}

fn normalize_repo_name(value: &str) -> String {
    value.trim().trim_end_matches(".git").replace('\\', "/").to_ascii_lowercase()
}

fn git_ref_exists(ref_name: &str) -> bool {
    Command::new("git")
        .args(["rev-parse", "--verify", "--quiet", ref_name])
        .output()
        .is_ok_and(|output| output.status.success())
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

fn boundary_verdict(
    source_ref_reachable: Option<bool>,
    source_missing_commits: Option<u64>,
    swarm_only_commits: Option<u64>,
    release_workflows_guarded: bool,
    has_warnings: bool,
) -> BoundaryVerdict {
    let source_history_missing_or_stale = source_ref_reachable != Some(true)
        || source_missing_commits.map_or(true, |count| count > 0)
        || swarm_only_commits.is_none();

    if source_history_missing_or_stale || !release_workflows_guarded {
        BoundaryVerdict::Drift
    } else if has_warnings {
        BoundaryVerdict::Warn
    } else {
        BoundaryVerdict::Ok
    }
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
        "gh release",
        "action-gh-release",
        "softprops/action-gh-release",
        "docker push",
        "cosign",
        "publish_image",
        "cargo_registry_token",
        "npm publish",
        "twine upload",
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
    println!("swarm_remote_configured: {}", report.swarm_remote_configured);
    if let Some(source_remote_url) = &report.source_remote_url {
        println!("source_remote_url: {source_remote_url}");
    }
    if let Some(swarm_remote_url) = &report.swarm_remote_url {
        println!("swarm_remote_url: {swarm_remote_url}");
    }
    println!("source_ref: {}", report.source_ref);
    println!("swarm_ref: {}", report.swarm_ref);
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
        assert!(!is_release_sensitive_workflow(
            "security.yml",
            "# Check that all dependencies come from crates.io\nrun: cargo audit"
        ));
        assert!(is_release_sensitive_workflow("release.yml", "name: Release"));
        assert!(is_release_sensitive_workflow("security.yml", "run: cargo publish --dry-run"));
        assert!(is_release_sensitive_workflow("image.yml", "env:\n  PUBLISH_IMAGE: true"));
    }

    #[test]
    fn missing_source_history_is_drift() {
        assert_eq!(boundary_verdict(None, None, None, true, false), BoundaryVerdict::Drift);
        assert_eq!(
            boundary_verdict(Some(true), None, Some(0), true, false),
            BoundaryVerdict::Drift
        );
        assert_eq!(
            boundary_verdict(Some(true), Some(0), None, true, false),
            BoundaryVerdict::Drift
        );
    }

    #[test]
    fn verified_swarm_only_commits_are_not_drift() {
        assert_eq!(
            boundary_verdict(Some(true), Some(0), Some(42), true, false),
            BoundaryVerdict::Ok
        );
    }

    #[test]
    fn warnings_without_drift_remain_warn() {
        assert_eq!(
            boundary_verdict(Some(true), Some(0), Some(42), true, true),
            BoundaryVerdict::Warn
        );
    }
}
