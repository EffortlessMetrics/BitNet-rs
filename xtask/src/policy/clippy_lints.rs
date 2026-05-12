//! Clippy lint policy ledger checker.
//!
//! This validates the staged lint ledger and the debt entries that justify
//! keeping an MSRV-ready lint at `allow` until the cleanup PR lands.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize, Default)]
struct LintsFile {
    #[serde(default)]
    schema_version: String,
    #[serde(default)]
    msrv: String,
    #[serde(default, rename = "planned")]
    planned: Vec<PlannedLint>,
}

#[derive(Debug, Deserialize, Clone)]
struct PlannedLint {
    #[serde(default)]
    name: String,
    #[serde(default)]
    level: String,
    #[serde(default)]
    activate_when_msrv: String,
    #[serde(default)]
    reason: String,
}

#[derive(Debug, Deserialize, Default)]
struct DebtFile {
    #[serde(default)]
    schema_version: String,
    #[serde(default, rename = "debt")]
    debt: Vec<DebtEntry>,
}

#[derive(Debug, Deserialize, Clone)]
struct DebtEntry {
    #[serde(default)]
    lint: String,
    #[serde(default)]
    path: String,
    #[serde(default)]
    owner: String,
    #[serde(default)]
    reason: String,
    #[serde(default)]
    expires: String,
    #[serde(default)]
    status: String,
}

#[derive(Debug, Default, Serialize)]
pub struct Report {
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub active: Vec<LintStatus>,
    pub deferred: Vec<LintStatus>,
    pub missing: Vec<LintStatus>,
    pub debt_count: usize,
}

#[derive(Debug, Serialize)]
pub struct LintStatus {
    pub lint: String,
    pub planned_level: String,
    pub cargo_level: Option<String>,
    pub activate_when_msrv: String,
    pub debt_entries: usize,
}

pub fn run(
    lints_path: PathBuf,
    debt_path: PathBuf,
    manifest: PathBuf,
    report_dir: PathBuf,
    fail_on_error: bool,
) -> Result<()> {
    let report = check(&lints_path, &debt_path, &manifest, &report_dir)?;
    println!(
        "clippy-lint-policy: {} active, {} deferred, {} missing, {} errors, {} warnings",
        report.active.len(),
        report.deferred.len(),
        report.missing.len(),
        report.errors.len(),
        report.warnings.len()
    );
    for w in &report.warnings {
        println!("warning: {w}");
    }
    for e in &report.errors {
        println!("error: {e}");
    }
    if fail_on_error && !report.errors.is_empty() {
        bail!("clippy-lint-policy check failed: {} errors", report.errors.len());
    }
    Ok(())
}

fn check(
    lints_path: &Path,
    debt_path: &Path,
    manifest: &Path,
    report_dir: &Path,
) -> Result<Report> {
    let mut report = Report::default();

    let lints_text = fs::read_to_string(lints_path)
        .with_context(|| format!("reading {}", lints_path.display()))?;
    let lints: LintsFile =
        toml::from_str(&lints_text).with_context(|| format!("parsing {}", lints_path.display()))?;
    if lints.schema_version.is_empty() {
        report.errors.push(format!("{} missing `schema_version`", lints_path.display()));
    }
    if lints.msrv.is_empty() {
        report.errors.push(format!("{} missing `msrv`", lints_path.display()));
    }

    let debt: DebtFile = if debt_path.exists() {
        let debt_text = fs::read_to_string(debt_path)
            .with_context(|| format!("reading {}", debt_path.display()))?;
        toml::from_str(&debt_text).with_context(|| format!("parsing {}", debt_path.display()))?
    } else {
        report.warnings.push(format!(
            "clippy debt file `{}` does not exist; staged lint deferrals cannot be receipted",
            debt_path.display()
        ));
        DebtFile::default()
    };
    if debt.schema_version.is_empty() {
        report.errors.push(format!("{} missing `schema_version`", debt_path.display()));
    }
    report.debt_count = debt.debt.len();

    let debt_by_lint = validate_debt(&debt, &mut report);
    validate_planned(&lints, &debt_by_lint, &mut report);

    let manifest_text =
        fs::read_to_string(manifest).with_context(|| format!("reading {}", manifest.display()))?;
    let manifest_value: toml::Value = toml::from_str(&manifest_text)
        .with_context(|| format!("parsing {}", manifest.display()))?;
    let cargo_lints = clippy_lints_table(&manifest_value);

    for planned in &lints.planned {
        if !version_le(&planned.activate_when_msrv, &lints.msrv) {
            continue;
        }
        let key = planned.name.strip_prefix("clippy::").unwrap_or(&planned.name);
        let cargo_level = cargo_lints.and_then(|table| table.get(key)).and_then(lint_level);
        let debt_entries = debt_by_lint.get(planned.name.as_str()).map_or(0, Vec::len);
        let status = LintStatus {
            lint: planned.name.clone(),
            planned_level: planned.level.clone(),
            cargo_level: cargo_level.clone(),
            activate_when_msrv: planned.activate_when_msrv.clone(),
            debt_entries,
        };
        match cargo_level.as_deref() {
            Some(level) if level == planned.level => report.active.push(status),
            Some("allow") if debt_entries > 0 => report.deferred.push(status),
            None if debt_entries > 0 => report.deferred.push(status),
            Some("allow") => {
                report.warnings.push(format!(
                    "{} is due at MSRV {} but remains `allow` without debt",
                    planned.name, planned.activate_when_msrv
                ));
                report.deferred.push(status);
            }
            None => {
                report.warnings.push(format!(
                    "{} is due at MSRV {} but is absent from workspace Clippy lints and has no debt",
                    planned.name, planned.activate_when_msrv
                ));
                report.missing.push(status);
            }
            Some(level) => {
                report.warnings.push(format!(
                    "{} has workspace level `{level}` but policy level `{}`",
                    planned.name, planned.level
                ));
                report.deferred.push(status);
            }
        }
    }

    fs::create_dir_all(report_dir)?;
    let json = serde_json::to_string_pretty(&report)?;
    fs::write(report_dir.join("clippy-lint-policy.json"), json)?;
    Ok(report)
}

fn validate_planned(
    lints: &LintsFile,
    debt_by_lint: &BTreeMap<&str, Vec<&DebtEntry>>,
    report: &mut Report,
) {
    let mut seen = BTreeSet::new();
    for planned in &lints.planned {
        if planned.name.is_empty() {
            report.errors.push("planned lint missing `name`".into());
            continue;
        }
        if !seen.insert(planned.name.as_str()) {
            report.errors.push(format!("duplicate planned lint `{}`", planned.name));
        }
        if !planned.name.starts_with("clippy::") {
            report
                .errors
                .push(format!("planned lint `{}` must start with `clippy::`", planned.name));
        }
        if !matches!(planned.level.as_str(), "allow" | "warn" | "deny") {
            report.errors.push(format!(
                "planned lint `{}` has invalid level `{}`",
                planned.name, planned.level
            ));
        }
        if planned.activate_when_msrv.is_empty() {
            report
                .errors
                .push(format!("planned lint `{}` missing `activate_when_msrv`", planned.name));
        }
        if planned.reason.is_empty() {
            report.errors.push(format!("planned lint `{}` missing `reason`", planned.name));
        }
    }
    for lint in debt_by_lint.keys() {
        if !seen.contains(*lint) {
            report
                .warnings
                .push(format!("debt references lint `{lint}` not present in planned ledger"));
        }
    }
}

fn validate_debt<'a>(
    debt: &'a DebtFile,
    report: &mut Report,
) -> BTreeMap<&'a str, Vec<&'a DebtEntry>> {
    let mut by_lint: BTreeMap<&str, Vec<&DebtEntry>> = BTreeMap::new();
    let today = chrono::Utc::now().date_naive();
    for entry in &debt.debt {
        if entry.lint.is_empty() {
            report.errors.push("debt entry missing `lint`".into());
        } else {
            by_lint.entry(entry.lint.as_str()).or_default().push(entry);
            if !entry.lint.starts_with("clippy::") {
                report
                    .errors
                    .push(format!("debt lint `{}` must start with `clippy::`", entry.lint));
            }
        }
        if entry.path.is_empty() {
            report.errors.push(format!("debt `{}` missing `path`", entry.lint));
        }
        if entry.owner.is_empty() {
            report.errors.push(format!("debt `{}` missing `owner`", entry.lint));
        }
        if entry.reason.is_empty() {
            report.errors.push(format!("debt `{}` missing `reason`", entry.lint));
        }
        if entry.status.is_empty() {
            report.errors.push(format!("debt `{}` missing `status`", entry.lint));
        }
        if entry.expires.is_empty() {
            report.errors.push(format!("debt `{}` missing `expires`", entry.lint));
        } else if let Ok(d) = chrono::NaiveDate::parse_from_str(&entry.expires, "%Y-%m-%d") {
            if d < today {
                report.errors.push(format!("debt `{}` expired on {}", entry.lint, entry.expires));
            }
        } else {
            report
                .errors
                .push(format!("debt `{}` has invalid `expires` `{}`", entry.lint, entry.expires));
        }
    }
    by_lint
}

fn clippy_lints_table(value: &toml::Value) -> Option<&toml::map::Map<String, toml::Value>> {
    value.get("workspace")?.get("lints")?.get("clippy")?.as_table()
}

fn lint_level(value: &toml::Value) -> Option<String> {
    match value {
        toml::Value::String(s) => Some(s.clone()),
        toml::Value::Table(t) => t.get("level").and_then(toml::Value::as_str).map(str::to_string),
        _ => None,
    }
}

fn version_le(lhs: &str, rhs: &str) -> bool {
    let parse = |s: &str| -> Vec<u64> {
        s.split('.').map(|part| part.parse::<u64>().unwrap_or(0)).collect::<Vec<_>>()
    };
    let mut left = parse(lhs);
    let mut right = parse(rhs);
    let len = left.len().max(right.len());
    left.resize(len, 0);
    right.resize(len, 0);
    left <= right
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn validates_debt_expiry() {
        let debt = DebtFile {
            schema_version: "1.0".into(),
            debt: vec![DebtEntry {
                lint: "clippy::manual_checked_ops".into(),
                path: "crates/x/**".into(),
                owner: "team".into(),
                reason: "reason".into(),
                expires: "1999-01-01".into(),
                status: "active".into(),
            }],
        };
        let mut report = Report::default();
        let by_lint = validate_debt(&debt, &mut report);
        assert!(by_lint.contains_key("clippy::manual_checked_ops"));
        assert!(report.errors.iter().any(|e| e.contains("expired")), "{:?}", report.errors);
    }

    #[test]
    fn reports_active_and_deferred_lints() {
        let dir = std::env::temp_dir().join(format!(
            "clippy-lints-{}-{}",
            std::process::id(),
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ));
        fs::create_dir_all(&dir).unwrap();
        let lints = dir.join("clippy-lints.toml");
        let debt = dir.join("clippy-debt.toml");
        let manifest = dir.join("Cargo.toml");
        writeln!(
            fs::File::create(&lints).unwrap(),
            r#"
schema_version = "1.0"
msrv = "1.95"

[[planned]]
name = "clippy::same_length_and_capacity"
level = "deny"
activate_when_msrv = "1.94"
reason = "reason"

[[planned]]
name = "clippy::manual_checked_ops"
level = "warn"
activate_when_msrv = "1.95"
reason = "reason"
"#
        )
        .unwrap();
        writeln!(
            fs::File::create(&debt).unwrap(),
            r#"
schema_version = "1.0"

[[debt]]
lint = "clippy::manual_checked_ops"
path = "crates/x/**"
owner = "team"
reason = "reason"
expires = "2099-01-01"
status = "active"
"#
        )
        .unwrap();
        writeln!(
            fs::File::create(&manifest).unwrap(),
            r#"
[workspace.lints.clippy]
same_length_and_capacity = "deny"
manual_checked_ops = "allow"
"#
        )
        .unwrap();
        let report = check(&lints, &debt, &manifest, &dir).unwrap();
        assert_eq!(report.active.len(), 1);
        assert_eq!(report.deferred.len(), 1);
        assert!(report.errors.is_empty(), "{:?}", report.errors);
    }
}
