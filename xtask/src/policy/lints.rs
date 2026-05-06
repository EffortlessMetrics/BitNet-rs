//! Workspace lint inheritance check.
//!
//! Cargo workspace lints (`[workspace.lints]` in the root manifest) are
//! only applied to a workspace member if that member's manifest contains:
//!
//! ```toml
//! [lints]
//! workspace = true
//! ```
//!
//! This module enumerates workspace members via `cargo metadata` and
//! verifies that every member opts in to workspace lint inheritance.
//! Without it, lint policy silently rots as new crates are added.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct CargoMetadata {
    packages: Vec<Package>,
    workspace_members: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct Package {
    id: String,
    manifest_path: PathBuf,
}

#[derive(Debug, Default)]
pub struct CheckOutcome {
    pub members: usize,
    pub inherited: usize,
    pub missing: Vec<PathBuf>,
}

impl CheckOutcome {
    pub fn ok(&self) -> bool {
        self.missing.is_empty()
    }
}

pub fn run_check(repo_root: &Path) -> Result<CheckOutcome> {
    let output = std::process::Command::new("cargo")
        .args(["metadata", "--format-version=1", "--no-deps", "--offline"])
        .current_dir(repo_root)
        .output()
        .context("cargo metadata")?;
    if !output.status.success() {
        anyhow::bail!(
            "cargo metadata failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let metadata: CargoMetadata =
        serde_json::from_slice(&output.stdout).context("parse cargo metadata")?;

    let member_ids: std::collections::HashSet<&str> =
        metadata.workspace_members.iter().map(String::as_str).collect();

    let mut outcome = CheckOutcome::default();
    for pkg in &metadata.packages {
        if !member_ids.contains(pkg.id.as_str()) {
            continue;
        }
        outcome.members += 1;
        if has_lints_workspace(&pkg.manifest_path)? {
            outcome.inherited += 1;
        } else {
            outcome.missing.push(pkg.manifest_path.clone());
        }
    }

    println!(
        "lint-inheritance: {} | members={}, inherited={}, missing={}",
        if outcome.ok() { "OK" } else { "ERROR" },
        outcome.members,
        outcome.inherited,
        outcome.missing.len()
    );
    Ok(outcome)
}

fn has_lints_workspace(manifest_path: &Path) -> Result<bool> {
    let raw = std::fs::read_to_string(manifest_path)
        .with_context(|| format!("read {}", manifest_path.display()))?;
    let parsed: toml::Value = toml::from_str(&raw)
        .with_context(|| format!("parse {}", manifest_path.display()))?;
    let Some(lints) = parsed.get("lints") else {
        return Ok(false);
    };
    let Some(workspace) = lints.get("workspace") else {
        return Ok(false);
    };
    Ok(workspace.as_bool().unwrap_or(false))
}
