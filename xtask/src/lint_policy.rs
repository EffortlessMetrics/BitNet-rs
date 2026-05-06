use anyhow::{Context, Result, bail};
use chrono::{NaiveDate, Utc};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use toml::Value;
use walkdir::WalkDir;

const REQUIRED_POLICY_FILES: &[&str] = &[
    "policy/clippy-lints.toml",
    "policy/clippy-debt.toml",
    "policy/no-panic-allowlist.toml",
    "policy/non-rust-allowlist.toml",
    "docs/CLIPPY_POLICY.md",
    "clippy.toml",
];

const TEST_CARVEOUTS: &[&str] = &[
    "allow-unwrap-in-tests",
    "allow-expect-in-tests",
    "allow-panic-in-tests",
    "allow-indexing-slicing-in-tests",
    "allow-dbg-in-tests",
];

pub fn run() -> Result<()> {
    let root = repo_root()?;
    let mut failures = Vec::new();

    check_required_files(&root, &mut failures);

    let cargo_path = root.join("Cargo.toml");
    let cargo = fs::read_to_string(&cargo_path)
        .with_context(|| format!("failed to read {}", cargo_path.display()))?;
    let policy = read_toml(&root.join("policy/clippy-lints.toml"), &mut failures)?;
    let debt = read_toml(&root.join("policy/clippy-debt.toml"), &mut failures)?;
    let clippy = read_toml(&root.join("clippy.toml"), &mut failures)?;

    check_msrv(&cargo, &policy, &clippy, &mut failures);
    check_clippy_carveouts(&clippy, &mut failures);
    check_policy_posture(&policy, &mut failures);
    check_active_lints(&cargo, &policy, &mut failures);
    check_planned_lints(&cargo, &policy, &mut failures);
    check_lint_inheritance(&root, &mut failures);
    check_debt(&debt, &mut failures);

    if !failures.is_empty() {
        failures.sort();
        failures.dedup();
        for failure in &failures {
            eprintln!("lint policy violation: {failure}");
        }
        bail!("lint policy check failed with {} violation(s)", failures.len());
    }

    println!("✅ lint policy check passed");
    Ok(())
}

fn repo_root() -> Result<PathBuf> {
    let mut dir = std::env::current_dir().context("failed to read current directory")?;
    loop {
        if dir.join("Cargo.toml").is_file() && dir.join("xtask").is_dir() {
            return Ok(dir);
        }
        if !dir.pop() {
            bail!("could not find repository root containing Cargo.toml and xtask/");
        }
    }
}

fn read_toml(path: &Path, failures: &mut Vec<String>) -> Result<Value> {
    let raw = match fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(err) => {
            failures.push(format!("{} must be readable: {err}", path.display()));
            String::new()
        }
    };
    raw.parse::<toml::Table>()
        .map(Value::Table)
        .with_context(|| format!("failed to parse {} as TOML", path.display()))
}

fn check_required_files(root: &Path, failures: &mut Vec<String>) {
    for file in REQUIRED_POLICY_FILES {
        if !root.join(file).is_file() {
            failures.push(format!("required policy file {file} is missing"));
        }
    }
}

fn check_msrv(cargo: &str, policy: &Value, clippy: &Value, failures: &mut Vec<String>) {
    let workspace_msrv = cargo_workspace_msrv(cargo);
    let policy_msrv = policy.get("msrv").and_then(Value::as_str);
    let clippy_msrv = clippy.get("msrv").and_then(Value::as_str);

    if workspace_msrv != Some("1.93") {
        failures.push(format!(
            "workspace.package.rust-version must be 1.93, found {:?}",
            workspace_msrv
        ));
    }
    if policy_msrv != workspace_msrv {
        failures.push(format!(
            "policy/clippy-lints.toml msrv {:?} must match workspace MSRV {:?}",
            policy_msrv, workspace_msrv
        ));
    }
    if clippy_msrv != workspace_msrv {
        failures.push(format!(
            "clippy.toml msrv {:?} must match workspace MSRV {:?}",
            clippy_msrv, workspace_msrv
        ));
    }
}

fn check_clippy_carveouts(clippy: &Value, failures: &mut Vec<String>) {
    for carveout in TEST_CARVEOUTS {
        if clippy.get(*carveout).is_some() {
            failures.push(format!("clippy.toml must not configure test carveout {carveout}"));
        }
    }
}

fn check_policy_posture(policy: &Value, failures: &mut Vec<String>) {
    let posture = policy.get("policy");
    require_bool(posture, "panic_free_tests", true, failures);
    require_bool(posture, "allow_test_carveouts", false, failures);
    require_bool(posture, "blanket_categories", false, failures);
    let suppression =
        posture.and_then(|value| value.get("suppression_style")).and_then(Value::as_str);
    if suppression != Some("expect-with-reason") {
        failures.push(format!(
            "policy.suppression_style must be expect-with-reason, found {:?}",
            suppression
        ));
    }
}

fn require_bool(posture: Option<&Value>, key: &str, expected: bool, failures: &mut Vec<String>) {
    let actual = posture.and_then(|value| value.get(key)).and_then(Value::as_bool);
    if actual != Some(expected) {
        failures.push(format!("policy.{key} must be {expected}, found {actual:?}"));
    }
}

fn check_active_lints(cargo: &str, policy: &Value, failures: &mut Vec<String>) {
    let active = policy.get("lint").and_then(Value::as_array).cloned().unwrap_or_default();
    if active.is_empty() {
        failures.push("policy/clippy-lints.toml must define active [[lint]] entries".to_string());
    }

    let rust_lints = workspace_lint_table(cargo, "rust");
    let clippy_lints = workspace_lint_table(cargo, "clippy");

    for lint in active {
        let Some(name) = lint.get("name").and_then(Value::as_str) else {
            failures.push("active lint entry missing name".to_string());
            continue;
        };
        let Some(level) = lint.get("level").and_then(Value::as_str) else {
            failures.push(format!("active lint {name} missing level"));
            continue;
        };
        if lint.get("status").and_then(Value::as_str) != Some("active") {
            failures.push(format!("active lint {name} must have status = active"));
        }
        if lint.get("reason").and_then(Value::as_str).is_none_or(str::is_empty) {
            failures.push(format!("active lint {name} missing reason"));
        }
        if lint.get("class").and_then(Value::as_str).is_none_or(str::is_empty) {
            failures.push(format!("active lint {name} missing class"));
        }

        let (table, local_name) = if let Some(local) = name.strip_prefix("rust::") {
            (&rust_lints, local)
        } else if let Some(local) = name.strip_prefix("clippy::") {
            (&clippy_lints, local)
        } else {
            failures.push(format!("active lint {name} must start with rust:: or clippy::"));
            continue;
        };
        match table.get(local_name) {
            Some(actual) if actual == level => {}
            Some(actual) => {
                failures.push(format!("active lint {name} level is {actual}, expected {level}"))
            }
            None => failures.push(format!("active lint {name} is missing from Cargo.toml")),
        }
    }
}

fn check_planned_lints(cargo: &str, policy: &Value, failures: &mut Vec<String>) {
    let planned = policy.get("planned").and_then(Value::as_array).cloned().unwrap_or_default();
    let clippy_lints = workspace_lint_table(cargo, "clippy");
    let msrv = cargo_workspace_msrv(cargo).unwrap_or_default();

    for lint in planned {
        let Some(name) = lint.get("name").and_then(Value::as_str) else {
            failures.push("planned lint entry missing name".to_string());
            continue;
        };
        let Some(activate) = lint.get("activate_when_msrv").and_then(Value::as_str) else {
            failures.push(format!("planned lint {name} missing activate_when_msrv"));
            continue;
        };
        if lint.get("level").and_then(Value::as_str).is_none() {
            failures.push(format!("planned lint {name} missing level"));
        }
        if lint.get("reason").and_then(Value::as_str).is_none_or(str::is_empty) {
            failures.push(format!("planned lint {name} missing reason"));
        }
        if lint.get("class").and_then(Value::as_str).is_none_or(str::is_empty) {
            failures.push(format!("planned lint {name} missing class"));
        }
        if version_lt(msrv, activate) {
            if let Some(local) = name.strip_prefix("clippy::") {
                if clippy_lints.contains_key(local) {
                    failures.push(format!(
                        "planned lint {name} must not be active before MSRV {activate}"
                    ));
                }
            }
        }
    }
}

fn workspace_lint_table(cargo: &str, tool: &str) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    let header = format!("[workspace.lints.{tool}]");
    let mut in_table = false;
    for line in cargo.lines() {
        let trimmed = line.trim();
        if trimmed == header {
            in_table = true;
            continue;
        }
        if in_table && trimmed.starts_with('[') {
            break;
        }
        if !in_table || trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let Some((key, rest)) = trimmed.split_once('=') else {
            continue;
        };
        let key = key.trim().to_string();
        let rest = rest.split('#').next().unwrap_or(rest).trim();
        let level = if rest.starts_with('{') {
            rest.split("level").nth(1).and_then(|tail| tail.split('"').nth(1)).map(str::to_string)
        } else {
            rest.trim_matches('"').split('"').next().map(str::to_string)
        };
        if let Some(level) = level.filter(|level| !level.is_empty()) {
            out.insert(key, level);
        }
    }
    out
}

fn cargo_workspace_msrv(cargo: &str) -> Option<&str> {
    let mut in_table = false;
    for line in cargo.lines() {
        let trimmed = line.trim();
        if trimmed == "[workspace.package]" {
            in_table = true;
            continue;
        }
        if in_table && trimmed.starts_with('[') {
            return None;
        }
        if in_table {
            let Some((key, value)) = trimmed.split_once('=') else {
                continue;
            };
            if key.trim() == "rust-version" {
                return value.trim().trim_matches('"').split('"').next();
            }
        }
    }
    None
}

fn check_lint_inheritance(root: &Path, failures: &mut Vec<String>) {
    for entry in WalkDir::new(root)
        .into_iter()
        .filter_entry(|entry| !is_ignored_dir(entry.path()))
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_file() && entry.file_name() == "Cargo.toml")
    {
        let path = entry.path();
        let Ok(raw) = fs::read_to_string(path) else {
            failures.push(format!("{} must be readable", relative(root, path)));
            continue;
        };
        if !raw.contains("[package]") || path == root.join("Cargo.toml") {
            continue;
        }
        if !has_lints_workspace_true(&raw) {
            failures
                .push(format!("{} must include [lints] workspace = true", relative(root, path)));
        }
    }
}

fn has_lints_workspace_true(raw: &str) -> bool {
    let mut in_table = false;
    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed == "[lints]" {
            in_table = true;
            continue;
        }
        if in_table && trimmed.starts_with('[') {
            return false;
        }
        if in_table {
            let Some((key, value)) = trimmed.split_once('=') else {
                continue;
            };
            if key.trim() == "workspace" && value.trim() == "true" {
                return true;
            }
        }
    }
    false
}

fn check_debt(debt: &Value, failures: &mut Vec<String>) {
    if debt.get("schema").and_then(Value::as_integer) != Some(1) {
        failures.push("policy/clippy-debt.toml must declare schema = 1".to_string());
    }
    let Some(entries) = debt.get("debt").and_then(Value::as_array) else {
        return;
    };
    let today = Utc::now().date_naive();
    let mut seen = BTreeSet::new();
    for entry in entries {
        let lint = required_str(entry, "lint", failures, "debt entry");
        let path = required_str(entry, "path", failures, "debt entry");
        required_str(entry, "owner", failures, "debt entry");
        required_str(entry, "reason", failures, "debt entry");
        let expires = required_str(entry, "expires", failures, "debt entry");
        if let Some(expires) = expires {
            match NaiveDate::parse_from_str(expires, "%Y-%m-%d") {
                Ok(date) if date < today => failures.push(format!(
                    "debt entry for {} at {} expired on {expires}",
                    lint.unwrap_or("<missing lint>"),
                    path.unwrap_or("<missing path>")
                )),
                Ok(_) => {}
                Err(err) => failures.push(format!(
                    "debt entry for {} at {} has invalid expires date {expires}: {err}",
                    lint.unwrap_or("<missing lint>"),
                    path.unwrap_or("<missing path>")
                )),
            }
        }
        if let (Some(lint), Some(path)) = (lint, path) {
            let key = format!("{lint}|{path}");
            if !seen.insert(key.clone()) {
                failures.push(format!("duplicate debt entry {key}"));
            }
        }
    }
}

fn required_str<'a>(
    entry: &'a Value,
    key: &str,
    failures: &mut Vec<String>,
    context: &str,
) -> Option<&'a str> {
    let value = entry.get(key).and_then(Value::as_str);
    if value.is_none_or(str::is_empty) {
        failures.push(format!("{context} missing {key}"));
    }
    value
}

fn is_ignored_dir(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| matches!(name, ".git" | "target" | ".cargo"))
}

fn relative(root: &Path, path: &Path) -> String {
    path.strip_prefix(root).unwrap_or(path).display().to_string()
}

fn version_lt(left: &str, right: &str) -> bool {
    parse_version(left) < parse_version(right)
}

fn parse_version(version: &str) -> Vec<u32> {
    version.split('.').map(|part| part.parse::<u32>().unwrap_or(0)).collect()
}
