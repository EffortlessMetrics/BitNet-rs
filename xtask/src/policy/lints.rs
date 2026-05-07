//! Workspace lint inheritance checker.
//!
//! Cargo workspace lints are not inherited automatically — each crate
//! manifest must declare:
//!
//! ```toml
//! [lints]
//! workspace = true
//! ```
//!
//! Without this opt-in, `[workspace.lints.*]` does nothing for that
//! crate. This checker walks every member listed in the workspace
//! root `Cargo.toml` and reports any crate that is missing the
//! `[lints]` section.

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
struct WorkspaceManifest {
    workspace: Workspace,
}

#[derive(Debug, Deserialize)]
struct Workspace {
    #[serde(default)]
    members: Vec<String>,
    #[serde(default)]
    exclude: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct CrateManifest {
    #[serde(default)]
    lints: Option<LintsSection>,
    #[serde(default)]
    workspace: Option<toml::Value>,
}

#[derive(Debug, Deserialize)]
struct LintsSection {
    #[serde(default)]
    workspace: Option<bool>,
}

#[derive(Debug, Default)]
pub struct Report {
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub crate_count: usize,
}

pub fn run(manifest: PathBuf, report_dir: PathBuf, fail_on_error: bool) -> Result<()> {
    let report = check(&manifest, &report_dir)?;
    println!(
        "lint-inheritance: {} crates checked, {} missing",
        report.crate_count,
        report.errors.len()
    );
    for w in &report.warnings {
        println!("warning: {w}");
    }
    for e in &report.errors {
        println!("error: {e}");
    }
    if fail_on_error && !report.errors.is_empty() {
        bail!("lint-inheritance check failed: {} crates", report.errors.len());
    }
    Ok(())
}

fn check(manifest_path: &Path, report_dir: &Path) -> Result<Report> {
    let mut report = Report::default();

    let text = fs::read_to_string(manifest_path)
        .with_context(|| format!("reading {}", manifest_path.display()))?;
    let ws: WorkspaceManifest =
        toml::from_str(&text).with_context(|| format!("parsing {}", manifest_path.display()))?;

    let root = manifest_path.parent().map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from("."));

    let exclude: BTreeSet<&str> = ws.workspace.exclude.iter().map(String::as_str).collect();
    for member in &ws.workspace.members {
        if exclude.contains(member.as_str()) {
            continue;
        }
        // Expand simple "*" globs (Cargo workspace member globs).
        let candidates = expand_member(&root, member);
        for crate_dir in candidates {
            let crate_manifest = crate_dir.join("Cargo.toml");
            if !crate_manifest.exists() {
                report.warnings.push(format!(
                    "workspace member `{}` missing Cargo.toml",
                    crate_manifest.display()
                ));
                continue;
            }
            report.crate_count += 1;
            let body = match fs::read_to_string(&crate_manifest) {
                Ok(b) => b,
                Err(e) => {
                    report
                        .warnings
                        .push(format!("could not read {}: {e}", crate_manifest.display()));
                    continue;
                }
            };
            let parsed: CrateManifest = match toml::from_str(&body) {
                Ok(p) => p,
                Err(e) => {
                    report
                        .warnings
                        .push(format!("could not parse {}: {e}", crate_manifest.display()));
                    continue;
                }
            };
            let inherits = parsed.lints.as_ref().and_then(|l| l.workspace).unwrap_or(false);
            if !inherits {
                report.errors.push(format!(
                    "{} missing `[lints] workspace = true`",
                    crate_manifest.display()
                ));
            }
        }
    }

    fs::create_dir_all(report_dir)?;
    let json = serde_json::json!({
        "schema_version": 1,
        "errors": report.errors,
        "warnings": report.warnings,
        "crate_count": report.crate_count,
    });
    fs::write(report_dir.join("lint-inheritance.json"), serde_json::to_string_pretty(&json)?)?;

    Ok(report)
}

fn expand_member(root: &Path, member: &str) -> Vec<PathBuf> {
    if !member.contains('*') {
        if member == "." {
            return vec![root.to_path_buf()];
        }
        return vec![root.join(member)];
    }
    // Handle "crates/*" and similar one-segment globs.
    let mut out = Vec::new();
    let parts: Vec<&str> = member.split('/').collect();
    if let Some((star_idx, _)) = parts.iter().enumerate().find(|(_, p)| p.contains('*')) {
        let prefix = parts[..star_idx].join("/");
        let suffix = parts[star_idx + 1..].join("/");
        let parent = if prefix.is_empty() { root.to_path_buf() } else { root.join(prefix) };
        if let Ok(entries) = fs::read_dir(&parent) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_dir() {
                    let candidate = if suffix.is_empty() { p } else { p.join(&suffix) };
                    out.push(candidate);
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_missing_lints_section() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(
            root.join("Cargo.toml"),
            r#"
[workspace]
members = ["a"]
"#,
        )
        .unwrap();
        std::fs::create_dir_all(root.join("a")).unwrap();
        std::fs::write(
            root.join("a/Cargo.toml"),
            r#"
[package]
name = "a"
version = "0.1.0"
edition = "2024"
"#,
        )
        .unwrap();
        let r = check(&root.join("Cargo.toml"), root).unwrap();
        assert_eq!(r.crate_count, 1);
        assert!(!r.errors.is_empty());
    }

    #[test]
    fn accepts_inheriting_crate() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(
            root.join("Cargo.toml"),
            r#"
[workspace]
members = ["a"]
"#,
        )
        .unwrap();
        std::fs::create_dir_all(root.join("a")).unwrap();
        std::fs::write(
            root.join("a/Cargo.toml"),
            r#"
[package]
name = "a"
version = "0.1.0"
edition = "2024"
[lints]
workspace = true
"#,
        )
        .unwrap();
        let r = check(&root.join("Cargo.toml"), root).unwrap();
        assert!(r.errors.is_empty(), "errors: {:?}", r.errors);
    }
}
