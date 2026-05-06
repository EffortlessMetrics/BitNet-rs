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

// ===========================================================================
// `cargo xtask check-lint-policy`
// ===========================================================================

#[derive(Debug, Deserialize)]
struct PolicyLedger {
    schema_version: String,
    msrv: String,
    #[serde(default)]
    policy: PolicyMeta,
    #[serde(default)]
    planned: Vec<PlannedLint>,
}

#[derive(Debug, Default, Deserialize)]
struct PolicyMeta {
    #[serde(default)]
    panic_free_tests: bool,
    #[serde(default)]
    allow_test_carveouts: bool,
    #[serde(default)]
    suppression_style: String,
    #[serde(default)]
    blanket_categories: bool,
    #[serde(default)]
    #[expect(dead_code, reason = "Reserved for future policy enforcement")]
    rust_default_implementation: bool,
}

#[derive(Debug, Deserialize)]
struct PlannedLint {
    name: String,
    level: String,
    activate_when_msrv: String,
    #[serde(default)]
    #[expect(dead_code, reason = "Reserved for future policy enforcement")]
    reason: String,
}

#[derive(Debug, Default)]
pub struct LintPolicyOutcome {
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub planned_due: Vec<String>,
    pub policy_msrv: String,
    pub manifest_msrv: String,
}

impl LintPolicyOutcome {
    pub fn ok(&self) -> bool {
        self.errors.is_empty()
    }
}

const FORBIDDEN_CARVEOUTS: &[&str] = &[
    "allow-unwrap-in-tests",
    "allow-expect-in-tests",
    "allow-panic-in-tests",
    "allow-indexing-slicing-in-tests",
    "allow-dbg-in-tests",
];

pub fn run_lint_policy(repo_root: &Path) -> Result<LintPolicyOutcome> {
    let mut outcome = LintPolicyOutcome::default();

    // 1. Parse policy/clippy-lints.toml.
    let ledger_path = repo_root.join("policy/clippy-lints.toml");
    let ledger_raw = std::fs::read_to_string(&ledger_path)
        .with_context(|| format!("read {}", ledger_path.display()))?;
    let ledger: PolicyLedger = toml::from_str(&ledger_raw)
        .with_context(|| format!("parse {}", ledger_path.display()))?;

    if ledger.schema_version != "1.0" {
        outcome.errors.push(format!(
            "policy/clippy-lints.toml: unsupported schema_version `{}` (expected 1.0)",
            ledger.schema_version
        ));
    }

    // 2. Compare ledger MSRV with root manifest.
    let root_manifest = repo_root.join("Cargo.toml");
    let root_raw = std::fs::read_to_string(&root_manifest)
        .with_context(|| format!("read {}", root_manifest.display()))?;
    let root: toml::Value = toml::from_str(&root_raw)
        .with_context(|| format!("parse {}", root_manifest.display()))?;
    let manifest_msrv = root
        .get("workspace")
        .and_then(|w| w.get("package"))
        .and_then(|p| p.get("rust-version"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    outcome.policy_msrv = ledger.msrv.clone();
    outcome.manifest_msrv = manifest_msrv.clone();
    if !manifest_msrv.is_empty() && manifest_msrv != ledger.msrv {
        outcome.errors.push(format!(
            "MSRV mismatch: policy/clippy-lints.toml = {}, Cargo.toml workspace.package.rust-version = {}",
            ledger.msrv, manifest_msrv
        ));
    }

    // 3. Verify policy invariants.
    if !ledger.policy.panic_free_tests {
        outcome.errors.push("policy/clippy-lints.toml: policy.panic_free_tests must be true".into());
    }
    if ledger.policy.allow_test_carveouts {
        outcome.errors.push("policy/clippy-lints.toml: policy.allow_test_carveouts must be false".into());
    }
    if ledger.policy.suppression_style != "expect-with-reason" {
        outcome.errors.push(format!(
            "policy/clippy-lints.toml: policy.suppression_style must be \"expect-with-reason\" (got `{}`)",
            ledger.policy.suppression_style
        ));
    }
    if ledger.policy.blanket_categories {
        outcome.errors.push("policy/clippy-lints.toml: policy.blanket_categories must be false".into());
    }

    // 4. clippy.toml must not declare panic-family test carveouts.
    let clippy_toml_path = repo_root.join("clippy.toml");
    if clippy_toml_path.exists() {
        let clippy_raw = std::fs::read_to_string(&clippy_toml_path)
            .with_context(|| format!("read {}", clippy_toml_path.display()))?;
        for needle in FORBIDDEN_CARVEOUTS {
            if clippy_raw
                .lines()
                .any(|l| l.trim_start().starts_with(needle))
            {
                outcome.errors.push(format!(
                    "clippy.toml: forbidden test carve-out `{needle}`; tests are workspace surface"
                ));
            }
        }
    }

    // 5. Verify that root manifest declares an explicit lints block (no
    //    blanket category at deny, and at least the dbg_macro deny rule).
    let root_lints = root
        .get("workspace")
        .and_then(|w| w.get("lints"))
        .cloned();
    match root_lints {
        Some(toml::Value::Table(t)) => {
            for (cat, val) in &t {
                if let toml::Value::Table(inner) = val {
                    for (name, level) in inner {
                        // Reject category-level deny: e.g. `all = "deny"` or `all = { level = "deny", priority = -1 }`.
                        if matches!(name.as_str(), "all" | "pedantic" | "nursery" | "cargo" | "restriction") {
                            let level_str = match level {
                                toml::Value::String(s) => s.clone(),
                                toml::Value::Table(tt) => tt
                                    .get("level")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string(),
                                _ => String::new(),
                            };
                            if level_str == "deny" {
                                outcome.errors.push(format!(
                                    "Cargo.toml: blanket clippy category `{cat}.{name}` must not be deny"
                                ));
                            }
                        }
                    }
                }
            }
        }
        Some(_) => outcome
            .errors
            .push("Cargo.toml: [workspace.lints] is not a table".into()),
        None => outcome
            .errors
            .push("Cargo.toml: [workspace.lints] is missing — strict policy requires an explicit lint block".into()),
    }

    // 6. Planned-lint guard: anything whose `activate_when_msrv` is at or
    //    below the workspace MSRV is due for activation in the next flip.
    for plan in &ledger.planned {
        if msrv_compare(&plan.activate_when_msrv, &ledger.msrv).is_le() {
            outcome.planned_due.push(format!(
                "{} ({}): due since MSRV {}",
                plan.name, plan.level, plan.activate_when_msrv
            ));
        }
    }
    if !outcome.planned_due.is_empty() {
        outcome.warnings.push(format!(
            "{} planned lint(s) are due for activation; reconcile with the active block.",
            outcome.planned_due.len()
        ));
    }

    Ok(outcome)
}

fn msrv_compare(a: &str, b: &str) -> std::cmp::Ordering {
    let parse = |s: &str| -> (u32, u32, u32) {
        let mut it = s.split('.');
        let major = it.next().and_then(|x| x.parse().ok()).unwrap_or(0);
        let minor = it.next().and_then(|x| x.parse().ok()).unwrap_or(0);
        let patch = it.next().and_then(|x| x.parse().ok()).unwrap_or(0);
        (major, minor, patch)
    };
    parse(a).cmp(&parse(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn msrv_compare_orders_minor_versions() {
        assert!(msrv_compare("1.93", "1.94").is_lt());
        assert!(msrv_compare("1.94.0", "1.94").is_eq());
        assert!(msrv_compare("1.95", "1.93").is_gt());
    }
}
