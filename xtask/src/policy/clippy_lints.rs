//! Clippy lint policy ledger checker.
//!
//! This validates the staged lint ledger and the debt entries that justify
//! keeping an MSRV-ready lint at `allow` until the cleanup PR lands.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

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
    activate_when_available: bool,
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
    pub unavailable: Vec<LintStatus>,
    pub toolchain: Option<ToolchainLintInventory>,
    pub debt_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolchainLintInventory {
    release: String,
    compiler_version: String,
    lints: BTreeSet<String>,
}

fn clippy_driver_output(args: &[&str]) -> Result<String> {
    let output = Command::new("clippy-driver")
        .args(args)
        .output()
        .context("querying the installed Clippy toolchain; install its clippy component")?;
    if !output.status.success() {
        bail!(
            "clippy-driver {args:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    String::from_utf8(output.stdout).context("Clippy toolchain output was not UTF-8")
}

fn parse_lint_inventory(help: &str) -> Result<BTreeSet<String>> {
    let lints: BTreeSet<_> = help
        .lines()
        .filter_map(|line| {
            let mut columns = line.split_whitespace();
            let name = columns.next()?;
            let level = columns.next()?;
            (name.starts_with("clippy::") && matches!(level, "allow" | "warn" | "deny" | "forbid"))
                .then(|| name.replace('-', "_"))
        })
        .collect();
    if lints.is_empty() {
        bail!("clippy-driver -W help did not report any Clippy lints; availability is unproven");
    }
    Ok(lints)
}

fn discover_toolchain_lints() -> Result<ToolchainLintInventory> {
    let compiler_version = clippy_driver_output(&["-Vv"])?;
    let release = compiler_version
        .lines()
        .find_map(|line| line.strip_prefix("release: "))
        .context("Clippy verbose version omitted its compiler release")?
        .trim()
        .to_string();
    let lints = parse_lint_inventory(&clippy_driver_output(&["-W", "help"])?)?;
    Ok(ToolchainLintInventory {
        release,
        compiler_version,
        lints,
    })
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
    let inventory = discover_toolchain_lints()?;
    let report = check(&lints_path, &debt_path, &manifest, &report_dir, &inventory)?;
    println!(
        "clippy-lint-policy: {} active, {} deferred, {} missing, {} unavailable, {} errors, {} warnings",
        report.active.len(),
        report.deferred.len(),
        report.missing.len(),
        report.unavailable.len(),
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
        bail!(
            "clippy-lint-policy check failed: {} errors",
            report.errors.len()
        );
    }
    Ok(())
}

fn check(
    lints_path: &Path,
    debt_path: &Path,
    manifest: &Path,
    report_dir: &Path,
    inventory: &ToolchainLintInventory,
) -> Result<Report> {
    let mut report = Report {
        toolchain: Some(inventory.clone()),
        ..Report::default()
    };

    let lints_text = fs::read_to_string(lints_path)
        .with_context(|| format!("reading {}", lints_path.display()))?;
    let lints: LintsFile =
        toml::from_str(&lints_text).with_context(|| format!("parsing {}", lints_path.display()))?;
    if lints.schema_version.is_empty() {
        report
            .errors
            .push(format!("{} missing `schema_version`", lints_path.display()));
    }
    if lints.msrv.is_empty() {
        report
            .errors
            .push(format!("{} missing `msrv`", lints_path.display()));
    }
    if inventory
        .release
        .split('.')
        .take(2)
        .ne(lints.msrv.split('.').take(2))
    {
        report.errors.push(format!(
            "Clippy compiler release {} does not match policy MSRV {}; lint availability must be measured on the pinned release",
            inventory.release, lints.msrv
        ));
    }

    let toolchain_path = manifest.with_file_name("rust-toolchain.toml");
    let toolchain_text = fs::read_to_string(&toolchain_path)
        .with_context(|| format!("reading pinned toolchain {}", toolchain_path.display()))?;
    let toolchain: toml::Value = toml::from_str(&toolchain_text)?;
    let pinned_release = toolchain
        .get("toolchain")
        .and_then(|value| value.get("channel"))
        .and_then(toml::Value::as_str)
        .context("rust-toolchain.toml omitted toolchain.channel")?;
    if inventory.release != pinned_release {
        report.errors.push(format!(
            "Clippy compiler release {} does not match pinned toolchain {pinned_release}",
            inventory.release
        ));
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
        report
            .errors
            .push(format!("{} missing `schema_version`", debt_path.display()));
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
        let key = planned
            .name
            .strip_prefix("clippy::")
            .unwrap_or(&planned.name);
        let cargo_level = cargo_lints
            .and_then(|table| table.get(key))
            .and_then(lint_level);
        let debt_entries = debt_by_lint.get(planned.name.as_str()).map_or(0, Vec::len);
        let status = LintStatus {
            lint: planned.name.clone(),
            planned_level: planned.level.clone(),
            cargo_level: cargo_level.clone(),
            activate_when_msrv: planned.activate_when_msrv.clone(),
            debt_entries,
        };
        if !inventory.lints.contains(&planned.name) {
            if !planned.activate_when_available {
                report.errors.push(format!(
                    "{} is not available in Clippy {}; its activation contract needs an explicit availability gate",
                    planned.name, inventory.release
                ));
            }
            if cargo_level.is_some() {
                report.errors.push(format!(
                    "unavailable lint {} must be absent from workspace Clippy lints, not declared at any level",
                    planned.name
                ));
            }
            if debt_entries > 0 {
                report.errors.push(format!(
                    "unavailable lint {} cannot be represented by source-code debt",
                    planned.name
                ));
            }
            report.unavailable.push(status);
            continue;
        }
        match cargo_level.as_deref() {
            Some(level) if level == planned.level => report.active.push(status),
            Some("allow") if debt_entries > 0 => report.deferred.push(status),
            None if debt_entries > 0 => report.deferred.push(status),
            Some("allow") => {
                report.errors.push(format!(
                    "{} is due at MSRV {} but remains `allow` without debt",
                    planned.name, planned.activate_when_msrv
                ));
                report.deferred.push(status);
            }
            None => {
                report.errors.push(format!(
                    "{} is due at MSRV {} but is absent from workspace Clippy lints and has no debt",
                    planned.name, planned.activate_when_msrv
                ));
                report.missing.push(status);
            }
            Some(level) => {
                report.errors.push(format!(
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
            report
                .errors
                .push(format!("duplicate planned lint `{}`", planned.name));
        }
        if !planned.name.starts_with("clippy::") {
            report.errors.push(format!(
                "planned lint `{}` must start with `clippy::`",
                planned.name
            ));
        }
        if !matches!(planned.level.as_str(), "allow" | "warn" | "deny") {
            report.errors.push(format!(
                "planned lint `{}` has invalid level `{}`",
                planned.name, planned.level
            ));
        }
        if planned.activate_when_msrv.is_empty() {
            report.errors.push(format!(
                "planned lint `{}` missing `activate_when_msrv`",
                planned.name
            ));
        }
        if planned.reason.is_empty() {
            report
                .errors
                .push(format!("planned lint `{}` missing `reason`", planned.name));
        }
    }
    for lint in debt_by_lint.keys() {
        if !seen.contains(*lint) {
            report.warnings.push(format!(
                "debt references lint `{lint}` not present in planned ledger"
            ));
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
                report.errors.push(format!(
                    "debt lint `{}` must start with `clippy::`",
                    entry.lint
                ));
            }
        }
        if entry.path.is_empty() {
            report
                .errors
                .push(format!("debt `{}` missing `path`", entry.lint));
        }
        if entry.owner.is_empty() {
            report
                .errors
                .push(format!("debt `{}` missing `owner`", entry.lint));
        }
        if entry.reason.is_empty() {
            report
                .errors
                .push(format!("debt `{}` missing `reason`", entry.lint));
        }
        if entry.status.is_empty() {
            report
                .errors
                .push(format!("debt `{}` missing `status`", entry.lint));
        }
        if entry.expires.is_empty() {
            report
                .errors
                .push(format!("debt `{}` missing `expires`", entry.lint));
        } else if let Ok(d) = chrono::NaiveDate::parse_from_str(&entry.expires, "%Y-%m-%d") {
            if d < today {
                report.errors.push(format!(
                    "debt `{}` expired on {}",
                    entry.lint, entry.expires
                ));
            }
        } else {
            report.errors.push(format!(
                "debt `{}` has invalid `expires` `{}`",
                entry.lint, entry.expires
            ));
        }
    }
    by_lint
}

fn clippy_lints_table(value: &toml::Value) -> Option<&toml::map::Map<String, toml::Value>> {
    value
        .get("workspace")?
        .get("lints")?
        .get("clippy")?
        .as_table()
}

fn lint_level(value: &toml::Value) -> Option<String> {
    match value {
        toml::Value::String(s) => Some(s.clone()),
        toml::Value::Table(t) => t
            .get("level")
            .and_then(toml::Value::as_str)
            .map(str::to_string),
        _ => None,
    }
}

fn version_le(lhs: &str, rhs: &str) -> bool {
    let parse = |s: &str| -> Vec<u64> {
        s.split('.')
            .map(|part| part.parse::<u64>().unwrap_or(0))
            .collect::<Vec<_>>()
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

    fn inventory(names: &[&str]) -> ToolchainLintInventory {
        ToolchainLintInventory {
            release: "1.95.0".into(),
            compiler_version: "test fixture: rustc 1.95.0".into(),
            lints: names.iter().map(|name| (*name).into()).collect(),
        }
    }

    fn write_pinned_toolchain(dir: &Path) -> Result<()> {
        fs::write(
            dir.join("rust-toolchain.toml"),
            "[toolchain]\nchannel = \"1.95.0\"\n",
        )?;
        Ok(())
    }

    fn availability_report(
        available: bool,
        gated: bool,
        level: Option<&str>,
        with_debt: bool,
        release: &str,
    ) -> Result<Report> {
        let dir = tempfile::tempdir()?;
        write_pinned_toolchain(dir.path())?;
        let lints = dir.path().join("clippy-lints.toml");
        let debt = dir.path().join("clippy-debt.toml");
        let manifest = dir.path().join("Cargo.toml");
        fs::write(
            &lints,
            format!(
                r#"
schema_version = "1.0"
msrv = "1.95"
[[planned]]
name = "clippy::manual_pop_if"
level = "warn"
activate_when_msrv = "1.95"
activate_when_available = {gated}
reason = "Use the compiler-provided predicate-and-pop lint."
"#
            ),
        )?;
        let debt_entry = if with_debt {
            r#"
[[debt]]
lint = "clippy::manual_pop_if"
path = "crates/x/**"
owner = "test"
reason = "Synthetic debt used to test rejection."
expires = "2099-01-01"
status = "active"
"#
        } else {
            ""
        };
        fs::write(&debt, format!("schema_version = \"1.0\"\n{debt_entry}"))?;
        let declaration =
            level.map_or_else(String::new, |level| format!("manual_pop_if = \"{level}\""));
        fs::write(
            &manifest,
            format!("[workspace.lints.clippy]\n{declaration}\n"),
        )?;
        let mut toolchain = inventory(if available {
            &["clippy::manual_pop_if"]
        } else {
            &[]
        });
        toolchain.release = release.into();
        check(&lints, &debt, &manifest, dir.path(), &toolchain)
    }

    #[test]
    fn parses_individual_lints_without_confusing_group_membership() -> Result<()> {
        let lints = parse_lint_inventory(
            "clippy::manual-checked-ops warn use checked arithmetic\n\
             clippy::disallowed-fields allow protect fields\n\
             clippy::all clippy::manual-checked-ops, clippy::disallowed-fields\n\
             unused warn unused code\n",
        )?;
        assert_eq!(
            lints,
            inventory(&["clippy::manual_checked_ops", "clippy::disallowed_fields"]).lints
        );
        assert!(parse_lint_inventory("clippy::all clippy::manual-checked-ops\n").is_err());
        Ok(())
    }

    #[test]
    fn unavailable_lint_requires_explicit_gate_and_no_declaration_or_debt() -> Result<()> {
        let report = availability_report(false, true, None, false, "1.95.0")?;
        assert!(report.errors.is_empty(), "{:?}", report.errors);
        assert_eq!(report.unavailable.len(), 1);
        assert!(report.active.is_empty());
        for level in [Some("warn"), Some("allow")] {
            let report = availability_report(false, true, level, false, "1.95.0")?;
            assert!(
                report
                    .errors
                    .iter()
                    .any(|error| error.contains("must be absent")),
                "{:?}",
                report.errors
            );
        }
        let ungated = availability_report(false, false, None, false, "1.95.0")?;
        assert!(
            ungated
                .errors
                .iter()
                .any(|error| error.contains("explicit availability gate"))
        );
        let debt = availability_report(false, true, None, true, "1.95.0")?;
        assert!(
            debt.errors
                .iter()
                .any(|error| error.contains("cannot be represented by source-code debt"))
        );
        Ok(())
    }

    #[test]
    fn availability_gate_requires_activation_when_lint_arrives() -> Result<()> {
        let missing = availability_report(true, true, None, false, "1.95.0")?;
        assert!(missing.errors.iter().any(|error| error.contains("absent")));
        let allowed = availability_report(true, true, Some("allow"), false, "1.95.0")?;
        assert!(
            allowed
                .errors
                .iter()
                .any(|error| error.contains("without debt"))
        );
        let active = availability_report(true, true, Some("warn"), false, "1.95.0")?;
        assert!(active.errors.is_empty(), "{:?}", active.errors);
        assert_eq!(active.active.len(), 1);
        assert!(active.unavailable.is_empty());
        Ok(())
    }

    #[test]
    fn rejects_inventory_from_another_compiler_release() -> Result<()> {
        for release in ["1.96.0", "1.95.1", "1.95.0-nightly"] {
            let report = availability_report(false, true, None, false, release)?;
            assert!(
                report
                    .errors
                    .iter()
                    .any(|error| error.contains("does not match pinned toolchain")),
                "{release}: {:?}",
                report.errors
            );
        }
        Ok(())
    }

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
        assert!(
            report.errors.iter().any(|e| e.contains("expired")),
            "{:?}",
            report.errors
        );
    }

    #[test]
    fn reports_active_and_deferred_lints() -> Result<()> {
        let dir = std::env::temp_dir().join(format!(
            "clippy-lints-{}-{}",
            std::process::id(),
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ));
        fs::create_dir_all(&dir).unwrap();
        write_pinned_toolchain(&dir)?;
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
        let toolchain = inventory(&[
            "clippy::same_length_and_capacity",
            "clippy::manual_checked_ops",
        ]);
        let report = check(&lints, &debt, &manifest, &dir, &toolchain).unwrap();
        assert_eq!(report.active.len(), 1);
        assert_eq!(report.deferred.len(), 1);
        assert!(report.errors.is_empty(), "{:?}", report.errors);
        Ok(())
    }

    #[test]
    fn due_lints_without_activation_or_debt_are_errors() -> Result<()> {
        let dir = std::env::temp_dir().join(format!(
            "clippy-lints-errors-{}-{}",
            std::process::id(),
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ));
        fs::create_dir_all(&dir).unwrap();
        write_pinned_toolchain(&dir)?;
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
name = "clippy::manual_take"
level = "warn"
activate_when_msrv = "1.95"
reason = "reason"

[[planned]]
name = "clippy::needless_type_cast"
level = "warn"
activate_when_msrv = "1.95"
reason = "reason"
"#
        )
        .unwrap();
        writeln!(fs::File::create(&debt).unwrap(), "schema_version = \"1.0\"").unwrap();
        writeln!(
            fs::File::create(&manifest).unwrap(),
            r#"
[workspace.lints.clippy]
same_length_and_capacity = "allow"
manual_take = "deny"
"#
        )
        .unwrap();
        let toolchain = inventory(&[
            "clippy::same_length_and_capacity",
            "clippy::manual_take",
            "clippy::needless_type_cast",
        ]);
        let report = check(&lints, &debt, &manifest, &dir, &toolchain).unwrap();
        assert_eq!(report.errors.len(), 3, "{:?}", report.errors);
        assert!(
            report
                .errors
                .iter()
                .any(|e| e.contains("same_length_and_capacity") && e.contains("without debt")),
            "{:?}",
            report.errors
        );
        assert!(
            report
                .errors
                .iter()
                .any(|e| e.contains("manual_take") && e.contains("workspace level")),
            "{:?}",
            report.errors
        );
        assert!(
            report
                .errors
                .iter()
                .any(|e| e.contains("needless_type_cast") && e.contains("absent")),
            "{:?}",
            report.errors
        );
        Ok(())
    }
}
