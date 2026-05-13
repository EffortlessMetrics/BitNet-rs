use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

const BADGE_ENDPOINT_DIR: &str = "badges";
const BADGE_ENDPOINT_TARGET_DIR: &str = "target/xtask/badges";
const RIPR_PR_DIR: &str = "target/ripr/pr";
const RIPR_REVIEW_DIR: &str = "target/ripr/review";

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub(crate) struct ShieldsEndpointBadge {
    #[serde(rename = "schemaVersion")]
    schema_version: u8,
    label: String,
    message: String,
    color: String,
}

pub(crate) fn workspace_root_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask manifest has workspace parent")
        .to_path_buf()
}

pub(crate) fn badges(check: bool) -> Result<()> {
    let workspace_root = workspace_root_path();
    let target_dir = workspace_root.join(BADGE_ENDPOINT_TARGET_DIR);
    fs::create_dir_all(&target_dir)
        .with_context(|| format!("creating {}", target_dir.display()))?;

    let ripr_plus = ripr_plus_badge(&workspace_root)?;
    validate_shields_badge(&ripr_plus, Some("ripr+"))?;
    write_json_pretty(&target_dir.join("ripr-plus.json"), &ripr_plus)?;

    if check {
        let committed_dir = workspace_root.join(BADGE_ENDPOINT_DIR);
        compare_files(&committed_dir.join("ripr-plus.json"), &target_dir.join("ripr-plus.json"))?;
        println!("badges: committed endpoints are current");
        return Ok(());
    }

    let committed_dir = workspace_root.join(BADGE_ENDPOINT_DIR);
    fs::create_dir_all(&committed_dir)
        .with_context(|| format!("creating {}", committed_dir.display()))?;
    fs::copy(target_dir.join("ripr-plus.json"), committed_dir.join("ripr-plus.json"))
        .with_context(|| "copying generated ripr+ endpoint into badges/")?;
    println!("badges: refreshed public endpoint JSON under badges/");
    Ok(())
}

fn ripr_plus_badge(workspace_root: &Path) -> Result<ShieldsEndpointBadge> {
    let ripr_bin = std::env::var("RIPR_BIN").unwrap_or_else(|_| "ripr".to_string());
    let output = Command::new(&ripr_bin)
        .arg("check")
        .arg("--root")
        .arg(workspace_root)
        .arg("--format")
        .arg("repo-badge-plus-shields")
        .current_dir(workspace_root)
        .output()
        .with_context(|| format!("running {ripr_bin} for repo-scoped ripr+ badge"))?;

    if !output.status.success() {
        bail!(
            "{ripr_bin} repo-badge-plus-shields failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    serde_json::from_slice(&output.stdout)
        .with_context(|| format!("{ripr_bin} emitted invalid Shields endpoint JSON"))
}

pub(crate) fn ripr_pr(check: bool) -> Result<()> {
    let workspace_root = workspace_root_path();
    let out_dir = workspace_root.join(RIPR_PR_DIR);
    let json = out_dir.join("repo-exposure.json");
    let markdown = out_dir.join("repo-exposure.md");

    if check {
        check_json_file(&json)?;
        check_nonempty_file(&markdown)?;
        println!("ripr-pr: output contract is intact");
        return Ok(());
    }

    fs::create_dir_all(&out_dir).with_context(|| format!("creating {}", out_dir.display()))?;
    let ripr_bin = std::env::var("RIPR_BIN").unwrap_or_else(|_| "ripr".to_string());
    run_ripr_check_to_file(
        &ripr_bin,
        &workspace_root,
        "repo-exposure-json",
        Some(ripr_base_ref()),
        &json,
    )?;
    run_ripr_check_to_file(
        &ripr_bin,
        &workspace_root,
        "repo-exposure-md",
        Some(ripr_base_ref()),
        &markdown,
    )?;
    check_json_file(&json)?;
    check_nonempty_file(&markdown)?;
    println!("ripr-pr: wrote {}", out_dir.display());
    Ok(())
}

pub(crate) fn ripr_review_comments(check: bool) -> Result<()> {
    let workspace_root = workspace_root_path();
    let out_dir = workspace_root.join(RIPR_REVIEW_DIR);
    let json = out_dir.join("comments.json");
    let markdown = out_dir.join("comments.md");

    if check {
        check_json_file(&json)?;
        check_nonempty_file(&markdown)?;
        println!("ripr-review-comments: output contract is intact");
        return Ok(());
    }

    fs::create_dir_all(&out_dir).with_context(|| format!("creating {}", out_dir.display()))?;
    let ripr_bin = std::env::var("RIPR_BIN").unwrap_or_else(|_| "ripr".to_string());
    let status = Command::new(&ripr_bin)
        .arg("review-comments")
        .arg("--root")
        .arg(&workspace_root)
        .arg("--base")
        .arg(ripr_base_ref())
        .arg("--head")
        .arg(ripr_head_ref())
        .arg("--out")
        .arg(&json)
        .current_dir(&workspace_root)
        .status()
        .with_context(|| format!("running {ripr_bin} review-comments"))?;
    if !status.success() {
        bail!("{ripr_bin} review-comments failed with status {status}");
    }
    check_json_file(&json)?;
    check_nonempty_file(&markdown)?;
    println!("ripr-review-comments: wrote {}", out_dir.display());
    Ok(())
}

fn run_ripr_check_to_file(
    ripr_bin: &str,
    workspace_root: &Path,
    format: &str,
    base: Option<String>,
    output_path: &Path,
) -> Result<()> {
    let mut command = Command::new(ripr_bin);
    command
        .arg("check")
        .arg("--root")
        .arg(workspace_root)
        .arg("--format")
        .arg(format)
        .current_dir(workspace_root);
    if let Some(base) = base {
        command.arg("--base").arg(base);
    }
    let output =
        command.output().with_context(|| format!("running {ripr_bin} check --format {format}"))?;
    if !output.status.success() {
        bail!(
            "{ripr_bin} check --format {format} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    fs::write(output_path, output.stdout)
        .with_context(|| format!("writing {}", output_path.display()))
}

fn ripr_base_ref() -> String {
    std::env::var("RIPR_BASE").unwrap_or_else(|_| "origin/main".to_string())
}

fn ripr_head_ref() -> String {
    std::env::var("RIPR_HEAD").unwrap_or_else(|_| "HEAD".to_string())
}

pub(crate) fn validate_shields_badge(
    badge: &ShieldsEndpointBadge,
    expected_label: Option<&str>,
) -> Result<()> {
    if badge.schema_version != 1 {
        bail!("badge `{}` has unsupported schemaVersion", badge.label);
    }
    if let Some(expected_label) = expected_label {
        if badge.label != expected_label {
            bail!("badge label drifted: got `{}`, expected `{expected_label}`", badge.label);
        }
    }
    if badge.message.trim().is_empty() {
        bail!("badge `{}` has empty message", badge.label);
    }
    if badge.color.trim().is_empty() {
        bail!("badge `{}` has empty color", badge.label);
    }
    Ok(())
}

fn write_json_pretty(path: &Path, badge: &ShieldsEndpointBadge) -> Result<()> {
    let json = serde_json::to_string_pretty(badge)?;
    fs::write(path, format!("{json}\n")).with_context(|| format!("writing {}", path.display()))
}

fn compare_files(committed: &Path, generated: &Path) -> Result<()> {
    let committed_bytes = fs::read(committed)
        .with_context(|| format!("reading committed badge {}", committed.display()))?;
    let generated_bytes = fs::read(generated)
        .with_context(|| format!("reading generated badge {}", generated.display()))?;
    if committed_bytes != generated_bytes {
        bail!(
            "badge endpoint drift: {} differs from {}; run `cargo xtask badges`",
            committed.display(),
            generated.display()
        );
    }
    Ok(())
}

fn check_json_file(path: &Path) -> Result<()> {
    let bytes =
        fs::read(path).with_context(|| format!("missing required JSON file {}", path.display()))?;
    if bytes.is_empty() {
        bail!("required JSON file {} is empty", path.display());
    }
    let _: serde_json::Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("invalid JSON in {}", path.display()))?;
    Ok(())
}

fn check_nonempty_file(path: &Path) -> Result<()> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("missing required report file {}", path.display()))?;
    if content.trim().is_empty() {
        bail!("required report file {} is empty", path.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ripr_plus_badge_shape_is_stable() {
        let badge = ShieldsEndpointBadge {
            schema_version: 1,
            label: "ripr+".to_string(),
            message: "0".to_string(),
            color: "brightgreen".to_string(),
        };

        validate_shields_badge(&badge, Some("ripr+")).unwrap();
    }

    #[test]
    fn scanner_safe_badge_shape_is_stable() {
        let badge = ShieldsEndpointBadge {
            schema_version: 1,
            label: "fixtures".to_string(),
            message: "scanner-safe".to_string(),
            color: "brightgreen".to_string(),
        };

        validate_shields_badge(&badge, Some("fixtures")).unwrap();
    }
}
