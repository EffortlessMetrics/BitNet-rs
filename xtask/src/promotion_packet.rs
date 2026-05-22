use anyhow::{Context, Result, bail};
use clap::Args;
use serde::Deserialize;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Args)]
pub struct PromotionPacketArgs {
    /// Source-side baseline ref for the promotion range.
    #[arg(long, value_name = "REF")]
    from: String,
    /// Swarm-side tip ref for the promotion range.
    #[arg(long, value_name = "REF", default_value = "HEAD")]
    to: String,
    /// Source/swarm boundary policy ledger.
    #[arg(long, default_value = "policy/repo-boundary.toml")]
    policy: PathBuf,
    /// Markdown packet output path.
    #[arg(long, default_value = "target/promotion/packet.md")]
    out: PathBuf,
    /// Also print the generated packet to stdout.
    #[arg(long, default_value_t = false)]
    print: bool,
}

#[derive(Debug, Deserialize)]
struct BoundaryPolicy {
    source: RepoRole,
    swarm: RepoRole,
}

#[derive(Debug, Deserialize)]
struct RepoRole {
    repo: String,
    branch: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CommitSummary {
    sha: String,
    subject: String,
}

#[derive(Debug, PartialEq, Eq)]
struct PromotionAnalysis {
    changed_files: Vec<String>,
    included_prs: Vec<String>,
    included_shas: Vec<String>,
    source_impact: Vec<String>,
    touched_crates: Vec<String>,
    campaigns_touched: Vec<String>,
    policy_files_touched: Vec<String>,
    generated_dashboard_paths: Vec<String>,
    release_sensitive_workflows: Vec<String>,
}

pub fn run(args: PromotionPacketArgs) -> Result<()> {
    let policy = load_policy(&args.policy)?;
    let from_sha = rev_parse(&args.from)?;
    let to_sha = rev_parse(&args.to)?;
    let changed_files = changed_files(&args.from, &args.to)?;
    let commits = commit_summaries(&args.from, &args.to)?;
    let analysis = analyze(&changed_files, &commits);
    let packet = render_packet(&policy, &args.from, &from_sha, &args.to, &to_sha, &analysis);

    if let Some(parent) = args.out.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
    }
    fs::write(&args.out, &packet)
        .with_context(|| format!("failed to write promotion packet: {}", args.out.display()))?;
    if args.print {
        print!("{packet}");
    }
    println!("wrote promotion packet: {}", args.out.display());
    Ok(())
}

fn load_policy(path: &Path) -> Result<BoundaryPolicy> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("failed to read repo-boundary policy: {}", path.display()))?;
    toml::from_str(&raw).context("failed to parse repo-boundary policy")
}

fn rev_parse(ref_name: &str) -> Result<String> {
    git_output(["rev-parse", ref_name])
        .with_context(|| format!("failed to resolve git ref `{ref_name}`"))
}

fn changed_files(from: &str, to: &str) -> Result<Vec<String>> {
    let range = format!("{from}..{to}");
    let output = git_output(["diff", "--name-only", &range])
        .with_context(|| format!("failed to list changed files for `{range}`"))?;
    Ok(split_lines(&output))
}

fn commit_summaries(from: &str, to: &str) -> Result<Vec<CommitSummary>> {
    let range = format!("{from}..{to}");
    let output = git_output(["log", "--format=%H%x09%s", &range])
        .with_context(|| format!("failed to list commits for `{range}`"))?;
    Ok(split_lines(&output)
        .into_iter()
        .filter_map(|line| {
            let (sha, subject) = line.split_once('\t')?;
            Some(CommitSummary { sha: sha.to_string(), subject: subject.to_string() })
        })
        .collect())
}

fn git_output<const N: usize>(args: [&str; N]) -> Result<String> {
    let output = Command::new("git").args(args).output().context("failed to run git")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("git command exited with {}: {}", output.status, stderr.trim());
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn split_lines(value: &str) -> Vec<String> {
    value.lines().map(str::trim).filter(|line| !line.is_empty()).map(ToOwned::to_owned).collect()
}

fn analyze(changed_files: &[String], commits: &[CommitSummary]) -> PromotionAnalysis {
    let mut included_prs = BTreeSet::new();
    let mut included_shas = BTreeSet::new();
    let mut touched_crates = BTreeSet::new();
    let mut campaigns_touched = BTreeSet::new();
    let mut policy_files_touched = BTreeSet::new();
    let mut generated_dashboard_paths = BTreeSet::new();
    let mut release_sensitive_workflows = BTreeSet::new();
    let mut source_impact = BTreeSet::new();

    for commit in commits {
        included_shas.insert(format!("{} {}", short_sha(&commit.sha), commit.subject));
        for pr in extract_pr_numbers(&commit.subject) {
            included_prs.insert(format!("#{pr}"));
        }
    }

    for path in changed_files {
        let normalized = normalize_path(path);
        let parts: Vec<&str> = normalized.split('/').collect();

        if parts.first() == Some(&"crates") {
            if let Some(crate_name) = parts.get(1) {
                touched_crates.insert((*crate_name).to_string());
            }
            source_impact.insert("runtime_or_product_code".to_string());
        } else if parts.first() == Some(&"xtask") {
            touched_crates.insert("xtask".to_string());
            source_impact.insert("xtask_or_policy_control_plane".to_string());
        } else if normalized == "Cargo.toml" || normalized == "Cargo.lock" {
            source_impact.insert("workspace_manifest_or_lockfile".to_string());
        }

        if parts.first() == Some(&"policy") {
            policy_files_touched.insert(normalized.clone());
            source_impact.insert("xtask_or_policy_control_plane".to_string());
        }

        if normalized.starts_with("docs/tracking/campaigns/") {
            if let Some(campaign) = parts.get(3) {
                campaigns_touched.insert((*campaign).to_string());
            }
            source_impact.insert("campaign_or_generated_tracking".to_string());
        }

        if normalized.starts_with("docs/tracking/generated/")
            || (normalized.starts_with("docs/tracking/campaigns/")
                && normalized.contains("/generated/"))
        {
            generated_dashboard_paths.insert(normalized.clone());
        }

        if normalized.starts_with(".github/workflows/") {
            source_impact.insert("workflow_change".to_string());
            if is_release_sensitive_workflow(&normalized) {
                release_sensitive_workflows.insert(normalized.clone());
                source_impact.insert("release_review_required".to_string());
            }
        }
    }

    if changed_files.is_empty() {
        source_impact.insert("no_changes_detected".to_string());
    } else if source_impact.is_empty() && changed_files.iter().all(|path| is_docs_like(path)) {
        source_impact.insert("docs_only".to_string());
    }
    if source_impact.is_empty() {
        source_impact.insert("unknown_review_required".to_string());
    }

    PromotionAnalysis {
        changed_files: sorted(changed_files.iter().map(|path| normalize_path(path))),
        included_prs: included_prs.into_iter().collect(),
        included_shas: included_shas.into_iter().collect(),
        source_impact: source_impact.into_iter().collect(),
        touched_crates: touched_crates.into_iter().collect(),
        campaigns_touched: campaigns_touched.into_iter().collect(),
        policy_files_touched: policy_files_touched.into_iter().collect(),
        generated_dashboard_paths: generated_dashboard_paths.into_iter().collect(),
        release_sensitive_workflows: release_sensitive_workflows.into_iter().collect(),
    }
}

fn normalize_path(path: &str) -> String {
    path.replace('\\', "/")
}

fn sorted<I>(values: I) -> Vec<String>
where
    I: IntoIterator<Item = String>,
{
    let set: BTreeSet<String> = values.into_iter().collect();
    set.into_iter().collect()
}

fn short_sha(sha: &str) -> String {
    sha.chars().take(12).collect()
}

fn extract_pr_numbers(subject: &str) -> Vec<String> {
    let mut numbers = BTreeSet::new();
    let bytes = subject.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'#' {
            let start = index + 1;
            let mut end = start;
            while end < bytes.len() && bytes[end].is_ascii_digit() {
                end += 1;
            }
            if end > start {
                numbers.insert(subject[start..end].to_string());
            }
            index = end;
        } else {
            index += 1;
        }
    }
    numbers.into_iter().collect()
}

fn is_docs_like(path: &str) -> bool {
    let path = normalize_path(path);
    path.starts_with("docs/")
        || path == "README.md"
        || path == "AGENTS.md"
        || path.ends_with(".md")
        || path.ends_with(".toml")
}

fn is_release_sensitive_workflow(path: &str) -> bool {
    let name = Path::new(path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if name.contains("release")
        || name.contains("publish")
        || name.contains("sign")
        || name.contains("deploy")
    {
        return true;
    }
    let Ok(raw) = fs::read_to_string(path) else {
        return false;
    };
    let raw = raw.to_ascii_lowercase();
    [
        "cargo publish",
        "gh release",
        "action-gh-release",
        "softprops/action-gh-release",
        "docker push",
        "cosign",
        "cargo_registry_token",
        "npm publish",
        "twine upload",
    ]
    .iter()
    .any(|needle| raw.contains(needle))
}

fn render_packet(
    policy: &BoundaryPolicy,
    from_ref: &str,
    from_sha: &str,
    to_ref: &str,
    to_sha: &str,
    analysis: &PromotionAnalysis,
) -> String {
    let id = format!("{}-to-{}", short_sha(from_sha), short_sha(to_sha));
    let dashboard_status = if analysis.generated_dashboard_paths.is_empty() {
        "No generated dashboard paths detected in this range.".to_string()
    } else {
        format!(
            "Generated dashboard paths changed; run `cargo run --locked -p xtask --no-default-features -- campaign generate --check` before promotion.\n{}",
            markdown_list(&analysis.generated_dashboard_paths)
        )
    };
    let release_impact = if analysis.release_sensitive_workflows.is_empty() {
        "No release-sensitive workflow paths detected by the packet generator.".to_string()
    } else {
        format!(
            "Release-sensitive workflow review required. Do not promote publish/signing behavior unless explicitly approved.\n{}",
            markdown_list(&analysis.release_sensitive_workflows)
        )
    };

    format!(
        r#"# Swarm To Source Promotion Packet

Promotion id: {id}
Source repo: {swarm_repo} ({swarm_branch})
Target repo: {source_repo} ({source_branch})
Swarm range: `{from_ref}..{to_ref}`
Resolved range: `{from_sha}`..`{to_sha}`

## Included Swarm PRs

{included_prs}

## Included Swarm SHAs

{included_shas}

## Source Impact

{source_impact}

## Changed Files

{changed_files}

## Touched Crates

{touched_crates}

## Campaigns Touched

{campaigns_touched}

## Policy Files Touched

{policy_files}

## Generated Dashboard Status

{dashboard_status}

## Proof Commands

- `git diff --check {from_ref}..{to_ref}`
- TODO: add validation commands from the included swarm PR bodies and receipts.

## Receipts

- TODO: add receipt, report, or campaign event paths that prove the promoted claims.

## Claim Boundary

This generated packet does not promote hardware, model, quality, speed, residency, server-readiness, release, publish, or signing claims by itself. Promote only claims with explicit receipts and source-review acceptance.

## What This Does Not Claim

- No dense CUDA proof is treated as BitNet QK256 CUDA proof.
- No BitNet QK256 CUDA proof is treated as dense SLM CUDA proof.
- No Qwen2.5 proof is treated as Qwen3 proof.
- No generic CUDA proof is treated as strict device proof.
- No AVX detection is treated as execution, parity, or speedup proof.
- No server smoke is treated as broad server readiness.
- No speedup is claimed outside an exact profile.

## Release/Publish/Signing Impact

{release_impact}

## Excluded Swarm Work

- TODO: list swarm PRs, commits, diagnostics, or drafts intentionally excluded from this source promotion.

## Rollback

- Revert the source promotion merge commit, or revert the explicit promoted commits if an approved fast-forward/direct update was used.
- Preserve the swarm history; do not hard-reset or squash the history import/promotion path.
"#,
        swarm_repo = policy.swarm.repo,
        swarm_branch = policy.swarm.branch,
        source_repo = policy.source.repo,
        source_branch = policy.source.branch,
        included_prs = markdown_list_or_none(&analysis.included_prs),
        included_shas = markdown_list_or_none(&analysis.included_shas),
        source_impact = markdown_list_or_none(&analysis.source_impact),
        changed_files = markdown_list_or_none(&analysis.changed_files),
        touched_crates = markdown_list_or_none(&analysis.touched_crates),
        campaigns_touched = markdown_list_or_none(&analysis.campaigns_touched),
        policy_files = markdown_list_or_none(&analysis.policy_files_touched),
    )
}

fn markdown_list(values: &[String]) -> String {
    values.iter().map(|value| format!("- `{value}`")).collect::<Vec<_>>().join("\n")
}

fn markdown_list_or_none(values: &[String]) -> String {
    if values.is_empty() { "- none detected".to_string() } else { markdown_list(values) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_pr_numbers_from_commit_subjects() {
        assert_eq!(
            extract_pr_numbers("merge pull request #351 from branch (#352)"),
            vec!["351".to_string(), "352".to_string()]
        );
        assert!(extract_pr_numbers("docs: no pr marker").is_empty());
    }

    #[test]
    fn classifies_shared_surfaces() {
        let changed_files = vec![
            "crates/bitnet-cli/src/main.rs".to_string(),
            "docs/tracking/campaigns/nvidia-5070ti/generated/status.md".to_string(),
            "policy/repo-boundary.toml".to_string(),
            ".github/workflows/publish.yml".to_string(),
        ];
        let commits = vec![CommitSummary {
            sha: "1234567890abcdef".to_string(),
            subject: "docs: update packet (#42)".to_string(),
        }];
        let analysis = analyze(&changed_files, &commits);

        assert_eq!(analysis.included_prs, vec!["#42".to_string()]);
        assert!(analysis.touched_crates.contains(&"bitnet-cli".to_string()));
        assert!(analysis.campaigns_touched.contains(&"nvidia-5070ti".to_string()));
        assert!(analysis.policy_files_touched.contains(&"policy/repo-boundary.toml".to_string()));
        assert!(
            analysis
                .generated_dashboard_paths
                .contains(&"docs/tracking/campaigns/nvidia-5070ti/generated/status.md".to_string())
        );
        assert!(
            analysis
                .release_sensitive_workflows
                .contains(&".github/workflows/publish.yml".to_string())
        );
        assert!(analysis.source_impact.contains(&"release_review_required".to_string()));
    }

    #[test]
    fn renders_required_packet_sections() {
        let policy = BoundaryPolicy {
            source: RepoRole {
                repo: "EffortlessMetrics/BitNet-rs".to_string(),
                branch: "main".to_string(),
            },
            swarm: RepoRole {
                repo: "EffortlessMetrics/bitnet-rs-swarm".to_string(),
                branch: "main".to_string(),
            },
        };
        let analysis = PromotionAnalysis {
            changed_files: vec!["docs/release/PROMOTE_TO_BITNET_RS.md".to_string()],
            included_prs: vec!["#1".to_string()],
            included_shas: vec!["abcdef123456 docs: example".to_string()],
            source_impact: vec!["docs_only".to_string()],
            touched_crates: Vec::new(),
            campaigns_touched: Vec::new(),
            policy_files_touched: Vec::new(),
            generated_dashboard_paths: Vec::new(),
            release_sensitive_workflows: Vec::new(),
        };
        let packet = render_packet(
            &policy,
            "origin/main",
            "abcdef1234567890",
            "HEAD",
            "fedcba9876543210",
            &analysis,
        );

        assert!(packet.contains("## Proof Commands"));
        assert!(packet.contains("## Claim Boundary"));
        assert!(packet.contains("## Release/Publish/Signing Impact"));
        assert!(packet.contains("EffortlessMetrics/bitnet-rs-swarm"));
        assert!(packet.contains("EffortlessMetrics/BitNet-rs"));
    }
}
