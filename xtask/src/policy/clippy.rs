//! Clippy exception checker.
//!
//! Enforces the suppression governance rule: no bare
//! `#[allow(clippy::...)]`; every Clippy suppression must be
//! `#[expect(clippy::..., reason = "policy:clippy-XXXX")]` and must
//! reference an entry in `policy/clippy-exceptions.toml`.
//!
//! This is a regex-shaped scan rather than a full parse — it matches
//! the `allow_attributes`/`allow_attributes_without_reason` Clippy
//! posture from PR 08+ at the repository level so the receipt is
//! enforced even on crates that have not yet flipped to the strict
//! profile.

use anyhow::{Context, Result, bail};
use regex::Regex;
use serde::Deserialize;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize, Default)]
struct ExceptionsFile {
    #[serde(default, rename = "exception")]
    exceptions: Vec<ExceptionEntry>,
}

#[derive(Debug, Deserialize, Clone)]
struct ExceptionEntry {
    id: String,
    #[serde(default)]
    lint: String,
    #[serde(default)]
    path: String,
    #[serde(default)]
    classification: String,
    #[serde(default)]
    owner: String,
    #[serde(default)]
    reason: String,
    #[serde(default)]
    expires: String,
}

#[derive(Debug, Default)]
pub struct Report {
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub allow_count: usize,
}

pub fn run(exceptions_path: PathBuf, report_dir: PathBuf, fail_on_error: bool) -> Result<()> {
    let report = check(&exceptions_path, &report_dir)?;
    println!(
        "clippy-exceptions: {} entries, {} errors, {} warnings",
        report.allow_count,
        report.errors.len(),
        report.warnings.len()
    );
    for w in &report.warnings {
        println!("warning: {w}");
    }
    for e in &report.errors {
        if fail_on_error {
            println!("error: {e}");
        } else {
            println!("advisory: {e}");
        }
    }
    if fail_on_error && !report.errors.is_empty() {
        bail!("clippy-exceptions check failed: {} errors", report.errors.len());
    }
    Ok(())
}

fn check(exceptions_path: &Path, report_dir: &Path) -> Result<Report> {
    let mut report = Report::default();

    let exceptions: ExceptionsFile = if exceptions_path.exists() {
        let text = fs::read_to_string(exceptions_path)
            .with_context(|| format!("reading {}", exceptions_path.display()))?;
        toml::from_str(&text).with_context(|| format!("parsing {}", exceptions_path.display()))?
    } else {
        report.warnings.push(format!(
            "clippy-exceptions file `{}` does not exist; running advisory only",
            exceptions_path.display()
        ));
        ExceptionsFile::default()
    };
    report.allow_count = exceptions.exceptions.len();

    let known_ids: BTreeSet<&str> = exceptions.exceptions.iter().map(|e| e.id.as_str()).collect();

    let today = chrono::Utc::now().date_naive();
    for ex in &exceptions.exceptions {
        if ex.id.is_empty() {
            report.errors.push("exception missing `id`".into());
        }
        if ex.lint.is_empty() {
            report.errors.push(format!("exception `{}` missing `lint`", ex.id));
        }
        if ex.owner.is_empty() {
            report.errors.push(format!("exception `{}` missing `owner`", ex.id));
        }
        if ex.reason.is_empty() {
            report.errors.push(format!("exception `{}` missing `reason`", ex.id));
        }
        if !ex.expires.is_empty()
            && let Ok(d) = chrono::NaiveDate::parse_from_str(&ex.expires, "%Y-%m-%d")
            && d < today
        {
            report.errors.push(format!("exception `{}` expired on {}", ex.id, ex.expires));
        }
    }

    let bare_allow =
        Regex::new(r"#\s*\[\s*allow\s*\(\s*clippy::").context("compile bare clippy allow regex")?;
    let expect_re =
        Regex::new(r#"#\s*\[\s*expect\s*\(\s*clippy::([A-Za-z0-9_]+)[^\]]*reason\s*=\s*"([^"]*)""#)
            .context("compile clippy expect-with-reason regex")?;
    let plain_expect =
        Regex::new(r"#\s*\[\s*expect\s*\(\s*clippy::").context("compile clippy expect regex")?;

    for entry in walkdir::WalkDir::new(".")
        .into_iter()
        .filter_entry(should_scan_entry)
        .filter_map(std::result::Result::ok)
    {
        let p = entry.path();
        if !p.is_file() {
            continue;
        }
        if p.extension().and_then(|s| s.to_str()) != Some("rs") {
            continue;
        }
        let path_str = p.strip_prefix(".").unwrap_or(p).to_string_lossy().replace('\\', "/");
        if path_str.contains("/target/") || path_str.starts_with("target/") {
            continue;
        }
        let body = match fs::read_to_string(p) {
            Ok(b) => b,
            Err(_) => continue,
        };
        for (i, raw) in body.lines().enumerate() {
            let line_no = i + 1;
            let trimmed = raw.trim_start();
            if trimmed.starts_with("//") {
                continue;
            }
            // Only fire on lines that look like an attribute (start with `#[`
            // or `#![`). This prevents matching regex literals or quoted
            // example strings.
            if !(trimmed.starts_with("#[") || trimmed.starts_with("#![")) {
                continue;
            }
            if bare_allow.is_match(raw) {
                report.errors.push(format!(
                    "{path_str}:{line_no} bare `#[allow(clippy::...)]` (use `#[expect(..., reason = \"policy:clippy-XXXX\")]`)"
                ));
                continue;
            }
            if plain_expect.is_match(raw) {
                if let Some(cap) = expect_re.captures(raw) {
                    let reason = cap.get(2).map(|m| m.as_str()).unwrap_or("");
                    if reason.is_empty() {
                        report.errors.push(format!(
                            "{path_str}:{line_no} `#[expect(clippy::...)]` empty reason"
                        ));
                    } else if let Some(rest) = reason.strip_prefix("policy:") {
                        if !known_ids.contains(rest) {
                            report.errors.push(format!(
                                "{path_str}:{line_no} `#[expect]` references unknown policy id `{rest}`"
                            ));
                        }
                    } else {
                        report.warnings.push(format!(
                            "{path_str}:{line_no} `#[expect(clippy::...)]` reason does not use `policy:` prefix"
                        ));
                    }
                } else {
                    report.errors.push(format!(
                        "{path_str}:{line_no} `#[expect(clippy::...)]` missing `reason = \"policy:...\"`"
                    ));
                }
            }
        }
    }

    fs::create_dir_all(report_dir)?;
    let json = serde_json::json!({
        "schema_version": 1,
        "errors": report.errors,
        "warnings": report.warnings,
        "allow_count": report.allow_count,
    });
    fs::write(report_dir.join("clippy-exceptions.json"), serde_json::to_string_pretty(&json)?)?;

    Ok(report)
}

fn should_scan_entry(entry: &walkdir::DirEntry) -> bool {
    if !entry.file_type().is_dir() {
        return true;
    }
    !matches!(entry.file_name().to_str(), Some(".git" | "target"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn parses_exceptions_with_expired() {
        let dir = std::env::temp_dir().join(format!("clippy-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let p = dir.join("clippy-exceptions.toml");
        writeln!(
            fs::File::create(&p).unwrap(),
            r#"
schema_version = "1.0"
[[exception]]
id = "clippy-0001"
lint = "clippy::indexing_slicing"
path = "x"
classification = "test"
owner = "team"
reason = "yes"
expires = "1999-01-01"
"#
        )
        .unwrap();
        let r = check(&p, &dir).unwrap();
        assert!(r.errors.iter().any(|e| e.contains("expired")), "errors: {:?}", r.errors);
    }

    #[test]
    fn prunes_generated_and_vcs_directories() -> Result<()> {
        let dir = std::env::temp_dir().join(format!("clippy-prune-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("src"))?;
        fs::create_dir_all(dir.join("target/debug"))?;
        fs::create_dir_all(dir.join(".git/objects"))?;
        fs::write(dir.join("src/lib.rs"), "")?;
        fs::write(dir.join("target/debug/generated.rs"), "")?;
        fs::write(dir.join(".git/objects/blob.rs"), "")?;

        let files: BTreeSet<String> = walkdir::WalkDir::new(&dir)
            .into_iter()
            .filter_entry(should_scan_entry)
            .filter_map(std::result::Result::ok)
            .filter(|entry| entry.file_type().is_file())
            .filter_map(|entry| {
                entry
                    .path()
                    .strip_prefix(&dir)
                    .ok()
                    .map(|path| path.to_string_lossy().replace('\\', "/"))
            })
            .collect();

        assert!(files.contains("src/lib.rs"), "files: {files:?}");
        assert!(!files.contains("target/debug/generated.rs"), "files: {files:?}");
        assert!(!files.contains(".git/objects/blob.rs"), "files: {files:?}");

        let _ = fs::remove_dir_all(&dir);
        Ok(())
    }
}
