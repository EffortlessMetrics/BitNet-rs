use anyhow::{Context, Result, anyhow, bail};
use chrono::NaiveDate;
use std::{fs, path::Path};
use toml::Value;
use walkdir::WalkDir;

const TEST_CARVEOUTS: &[&str] = &[
    "allow-unwrap-in-tests",
    "allow-expect-in-tests",
    "allow-panic-in-tests",
    "allow-indexing-slicing-in-tests",
    "allow-dbg-in-tests",
];

const REQUIRED_PLANNED: &[(&str, &str)] = &[
    ("clippy::same_length_and_capacity", "1.94"),
    ("clippy::manual_ilog2", "1.94"),
    ("clippy::decimal_bitwise_operands", "1.94"),
    ("clippy::needless_type_cast", "1.94"),
    ("clippy::disallowed_fields", "1.95"),
    ("clippy::manual_checked_ops", "1.95"),
    ("clippy::manual_take", "1.95"),
    ("clippy::manual_pop_if", "1.95"),
    ("clippy::duration_suboptimal_units", "1.95"),
    ("clippy::unnecessary_trailing_comma", "1.95"),
];

pub fn check_lint_policy(root: &Path) -> Result<()> {
    let cargo = read_toml(&root.join("Cargo.toml"))?;
    let policy = read_toml(&root.join("policy/clippy-lints.toml"))?;

    let msrv = required_str(&policy, &["msrv"])?;
    let workspace_msrv = required_str(&cargo, &["workspace", "package", "rust-version"])?;
    ensure_eq("workspace.package.rust-version", workspace_msrv, "policy msrv", msrv)?;

    let clippy_toml = fs::read_to_string(root.join("clippy.toml")).context("read clippy.toml")?;
    ensure_no_test_carveouts(&clippy_toml)?;
    let clippy_msrv = parse_scalar_assignment(&clippy_toml, "msrv")
        .ok_or_else(|| anyhow!("clippy.toml must set msrv"))?;
    ensure_eq("clippy.toml msrv", &clippy_msrv, "policy msrv", msrv)?;

    validate_policy_flags(&policy)?;
    validate_active_lints(&cargo, &policy, msrv)?;
    validate_required_planned(&cargo, &policy, msrv)?;
    validate_workspace_lint_inheritance(root)?;
    validate_debt(root, "policy/clippy-debt.toml")?;
    validate_no_panic_allowlist(root, "policy/no-panic-allowlist.toml")?;
    validate_non_rust_allowlist(root, "policy/non-rust-allowlist.toml")?;

    println!("lint policy OK: MSRV {msrv}, active lints and planned flips are coherent");
    Ok(())
}

pub fn policy_report(root: &Path) -> Result<()> {
    let policy = read_toml(&root.join("policy/clippy-lints.toml"))?;
    let lints = required_array(&policy, &["lint"])?;
    let mut active = 0;
    let mut planned = 0;
    for lint in lints {
        let status = table_str(lint, "status")?;
        if status == "active" {
            active += 1;
        } else if status == "planned" {
            planned += 1;
        }
    }

    let debt = optional_array(root, "policy/clippy-debt.toml", "debt")?;
    let panic_allow = optional_array(root, "policy/no-panic-allowlist.toml", "allow")?;
    let non_rust_allow = optional_array(root, "policy/non-rust-allowlist.toml", "allow")?;

    println!("lint policy report");
    println!("  active lints: {active}");
    println!("  planned lints: {planned}");
    println!("  clippy debt entries: {}", debt.len());
    println!("  no-panic allowlist entries: {}", panic_allow.len());
    println!("  non-rust allowlist entries: {}", non_rust_allow.len());
    Ok(())
}

fn validate_policy_flags(policy: &Value) -> Result<()> {
    ensure_bool(policy, &["policy", "panic_free_tests"], true)?;
    ensure_bool(policy, &["policy", "allow_test_carveouts"], false)?;
    ensure_bool(policy, &["policy", "blanket_categories"], false)?;
    let suppression_style = required_str(policy, &["policy", "suppression_style"])?;
    ensure_eq(
        "policy.suppression_style",
        suppression_style,
        "required suppression style",
        "expect-with-reason",
    )
}

fn validate_active_lints(cargo: &Value, policy: &Value, policy_msrv: &str) -> Result<()> {
    for lint in required_array(policy, &["lint"])? {
        if table_str(lint, "status")? != "active" {
            continue;
        }
        let name = table_str(lint, "name")?;
        let expected_level = table_str(lint, "level")?;
        let actual_level = cargo_lint_level(cargo, name)?;
        ensure_eq(name, &actual_level, "policy level", expected_level)?;
        if let Some(activate_when) = table_optional_str(lint, "activate_when_msrv")? {
            bail!("active lint {name} must not retain activate_when_msrv={activate_when}");
        }
        let _ = policy_msrv;
    }
    Ok(())
}

fn validate_required_planned(cargo: &Value, policy: &Value, policy_msrv: &str) -> Result<()> {
    for (name, msrv) in REQUIRED_PLANNED {
        let lint = find_lint(policy, name, "planned")?
            .ok_or_else(|| anyhow!("missing planned lint {name}"))?;
        ensure_eq(
            &format!("planned {name} activate_when_msrv"),
            table_str(lint, "activate_when_msrv")?,
            "required planned MSRV",
            msrv,
        )?;
        if version_lt(policy_msrv, msrv) && cargo_lint_level(cargo, name).is_ok() {
            bail!("planned lint {name} must not be active before MSRV {msrv}");
        }
    }
    Ok(())
}

fn validate_workspace_lint_inheritance(root: &Path) -> Result<()> {
    let mut missing = Vec::new();
    for entry in WalkDir::new(root).into_iter().filter_entry(|entry| !is_ignored(entry.path())) {
        let entry = entry?;
        if !entry.file_type().is_file() || entry.file_name() != "Cargo.toml" {
            continue;
        }
        let path = entry.path();
        if path.starts_with(root.join("examples")) {
            continue;
        }
        let manifest = read_toml(path)?;
        if manifest.get("package").is_none() {
            continue;
        }
        let inherits = manifest
            .get("lints")
            .and_then(|lints| lints.get("workspace"))
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if !inherits {
            missing.push(path.strip_prefix(root).unwrap_or(path).display().to_string());
        }
    }
    if missing.is_empty() {
        Ok(())
    } else {
        bail!("workspace members must inherit [lints] workspace = true: {}", missing.join(", "))
    }
}

fn validate_debt(root: &Path, relative: &str) -> Result<()> {
    let value = read_toml(&root.join(relative))?;
    for debt in value.get("debt").and_then(Value::as_array).into_iter().flatten() {
        require_fields(debt, &["lint", "path", "owner", "reason", "expires"])?;
        ensure_future_date(table_str(debt, "expires")?, relative)?;
    }
    Ok(())
}

fn validate_no_panic_allowlist(root: &Path, relative: &str) -> Result<()> {
    let value = read_toml(&root.join(relative))?;
    ensure_eq("no-panic schema", required_str(&value, &["schema_version"])?, "required", "0.3")?;
    for allow in value.get("allow").and_then(Value::as_array).into_iter().flatten() {
        require_fields(
            allow,
            &["path", "family", "classification", "owner", "explanation", "selector"],
        )?;
        let selector = allow
            .get("selector")
            .ok_or_else(|| anyhow!("no-panic allow entry missing selector"))?;
        require_fields(selector, &["kind"])?;
        if let Some(expires) = table_optional_str(allow, "expires")? {
            ensure_future_date(expires, relative)?;
        }
    }
    Ok(())
}

fn validate_non_rust_allowlist(root: &Path, relative: &str) -> Result<()> {
    let value = read_toml(&root.join(relative))?;
    ensure_eq("non-rust schema", required_str(&value, &["schema_version"])?, "required", "1.0")?;
    for allow in value.get("allow").and_then(Value::as_array).into_iter().flatten() {
        require_fields(allow, &["kind", "owner", "reason", "surface", "classification"])?;
        if allow.get("path").is_none() && allow.get("glob").is_none() {
            bail!("non-rust allow entry must set path or glob");
        }
        let covered_by = allow
            .get("covered_by")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow!("non-rust allow entry must set covered_by"))?;
        if covered_by.is_empty() {
            bail!("non-rust allow entry covered_by must not be empty");
        }
        if let Some(expires) = table_optional_str(allow, "expires")? {
            ensure_future_date(expires, relative)?;
        }
    }
    Ok(())
}

fn cargo_lint_level(cargo: &Value, lint_name: &str) -> Result<String> {
    let (tool, lint) = lint_name
        .split_once("::")
        .ok_or_else(|| anyhow!("lint name must be tool::name: {lint_name}"))?;
    let value = required_value(cargo, &["workspace", "lints", tool, lint])?;
    if let Some(level) = value.as_str() {
        return Ok(level.to_owned());
    }
    value
        .get("level")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| anyhow!("workspace lint {lint_name} must be a string or {{ level = ... }}"))
}

fn read_toml(path: &Path) -> Result<Value> {
    let contents = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    toml::from_str::<Value>(&contents).with_context(|| format!("parse {}", path.display()))
}

fn optional_array(root: &Path, relative: &str, key: &str) -> Result<Vec<Value>> {
    let value = read_toml(&root.join(relative))?;
    Ok(value.get(key).and_then(Value::as_array).cloned().unwrap_or_default())
}

fn find_lint<'a>(policy: &'a Value, name: &str, status: &str) -> Result<Option<&'a Value>> {
    for lint in required_array(policy, &["lint"])? {
        if table_str(lint, "name")? == name && table_str(lint, "status")? == status {
            return Ok(Some(lint));
        }
    }
    Ok(None)
}

fn required_array<'a>(value: &'a Value, path: &[&str]) -> Result<&'a Vec<Value>> {
    required_value(value, path)?
        .as_array()
        .ok_or_else(|| anyhow!("{} must be an array", path.join(".")))
}

fn required_str<'a>(value: &'a Value, path: &[&str]) -> Result<&'a str> {
    required_value(value, path)?
        .as_str()
        .ok_or_else(|| anyhow!("{} must be a string", path.join(".")))
}

fn required_value<'a>(value: &'a Value, path: &[&str]) -> Result<&'a Value> {
    let mut current = value;
    for part in path {
        current =
            current.get(*part).ok_or_else(|| anyhow!("missing required key {}", path.join(".")))?;
    }
    Ok(current)
}

fn ensure_bool(value: &Value, path: &[&str], expected: bool) -> Result<()> {
    let actual = required_value(value, path)?
        .as_bool()
        .ok_or_else(|| anyhow!("{} must be a boolean", path.join(".")))?;
    if actual == expected { Ok(()) } else { bail!("{} must be {expected}", path.join(".")) }
}

fn table_str<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("policy table missing string field {key}"))
}

fn table_optional_str<'a>(value: &'a Value, key: &str) -> Result<Option<&'a str>> {
    value
        .get(key)
        .map(|field| field.as_str().ok_or_else(|| anyhow!("field {key} must be a string")))
        .transpose()
}

fn require_fields(value: &Value, fields: &[&str]) -> Result<()> {
    for field in fields {
        if value.get(*field).is_none() {
            bail!("policy entry missing required field {field}");
        }
    }
    Ok(())
}

fn ensure_eq(left_name: &str, left: &str, right_name: &str, right: &str) -> Result<()> {
    if left == right {
        Ok(())
    } else {
        bail!("{left_name} ({left}) must match {right_name} ({right})")
    }
}

fn ensure_no_test_carveouts(clippy_toml: &str) -> Result<()> {
    for carveout in TEST_CARVEOUTS {
        if parse_scalar_assignment(clippy_toml, carveout).is_some() {
            bail!("clippy.toml must not set test carveout {carveout}");
        }
    }
    Ok(())
}

fn parse_scalar_assignment(contents: &str, key: &str) -> Option<String> {
    contents.lines().find_map(|line| {
        let trimmed = line.split('#').next()?.trim();
        let (left, right) = trimmed.split_once('=')?;
        if left.trim() != key {
            return None;
        }
        let value = right.trim().trim_matches('"').to_owned();
        Some(value)
    })
}

fn ensure_future_date(date: &str, source: &str) -> Result<()> {
    let expires = NaiveDate::parse_from_str(date, "%Y-%m-%d")
        .with_context(|| format!("parse expiry {date} in {source}"))?;
    let today = chrono::Utc::now().date_naive();
    if expires < today {
        bail!("expired policy entry in {source}: {date}");
    }
    Ok(())
}

fn version_lt(left: &str, right: &str) -> bool {
    version_parts(left) < version_parts(right)
}

fn version_parts(version: &str) -> (u32, u32, u32) {
    let mut parts = version.split('.').map(|part| part.parse::<u32>().unwrap_or(0));
    let major = parts.next().unwrap_or(0);
    let minor = parts.next().unwrap_or(0);
    let patch = parts.next().unwrap_or(0);
    (major, minor, patch)
}

fn is_ignored(path: &Path) -> bool {
    path.components().any(|component| {
        let name = component.as_os_str();
        name == ".git" || name == "target" || name == "vendor"
    })
}
