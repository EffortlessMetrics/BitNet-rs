//! Policy governance commands.
//!
//! These commands enforce the Effortless Metrics shared strict policy in
//! BitNet-rs. See `docs/development/STRICT_CLIPPY_POLICY.md` and
//! `docs/development/POLICY_ALLOWLISTS.md` for the model.
//!
//! Outputs are written under `target/bitnet/reports/` so CI can pick them
//! up as artifacts.

use std::path::{Path, PathBuf};

pub mod non_rust;
pub mod report;

/// Standard report directory for all policy commands.
pub fn report_dir(repo_root: &Path) -> PathBuf {
    repo_root.join("target").join("bitnet").join("reports")
}

/// Resolve the repository root from the current working directory.
pub fn repo_root() -> anyhow::Result<PathBuf> {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .map_err(|e| anyhow::anyhow!("failed to invoke git: {e}"))?;
    if !output.status.success() {
        anyhow::bail!("git rev-parse --show-toplevel failed");
    }
    let root = String::from_utf8(output.stdout)
        .map_err(|e| anyhow::anyhow!("git output not utf-8: {e}"))?;
    Ok(PathBuf::from(root.trim()))
}

/// Enumerate files tracked by git under `repo_root`.
pub fn tracked_files(repo_root: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let output = std::process::Command::new("git")
        .args(["ls-files", "-z"])
        .current_dir(repo_root)
        .output()
        .map_err(|e| anyhow::anyhow!("failed to invoke git ls-files: {e}"))?;
    if !output.status.success() {
        anyhow::bail!("git ls-files failed");
    }
    let mut files = Vec::with_capacity(output.stdout.len() / 32);
    for chunk in output.stdout.split(|b| *b == 0) {
        if chunk.is_empty() {
            continue;
        }
        let path = std::str::from_utf8(chunk)
            .map_err(|e| anyhow::anyhow!("non-utf8 path in git ls-files: {e}"))?;
        files.push(PathBuf::from(path));
    }
    Ok(files)
}
