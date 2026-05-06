use anyhow::{Context, Result, bail};
use chrono::{NaiveDate, Utc};
use std::fs;
use toml::Value;
use walkdir::WalkDir;

const ROOT_CARGO: &str = "Cargo.toml";
const CLIPPY_TOML: &str = "clippy.toml";
const LINT_LEDGER: &str = "policy/clippy-lints.toml";
const DEBT_LEDGER: &str = "policy/clippy-debt.toml";
const NO_PANIC_ALLOWLIST: &str = "policy/no-panic-allowlist.toml";
const NON_RUST_ALLOWLIST: &str = "policy/non-rust-allowlist.toml";

const FORBIDDEN_TEST_CARVEOUTS: &[&str] = &[
    "allow-unwrap-in-tests",
    "allow-expect-in-tests",
    "allow-panic-in-tests",
    "allow-indexing-slicing-in-tests",
    "allow-dbg-in-tests",
];

const REQUIRED_ACTIVE_LINTS: &[&str] = &[
    "unsafe_code",
    "unsafe_op_in_unsafe_fn",
    "unused_must_use",
    "dbg_macro",
    "todo",
    "unimplemented",
    "panic",
    "unreachable",
    "unwrap_used",
    "expect_used",
    "indexing_slicing",
    "let_underscore_future",
    "let_underscore_must_use",
    "map_err_ignore",
    "allow_attributes",
    "allow_attributes_without_reason",
    "arithmetic_side_effects",
];

const REQUIRED_PLANNED_LINTS: &[&str] = &[
    "clippy::same_length_and_capacity",
    "clippy::manual_ilog2",
    "clippy::decimal_bitwise_operands",
    "clippy::needless_type_cast",
    "clippy::disallowed_fields",
    "clippy::manual_checked_ops",
    "clippy::manual_take",
    "clippy::manual_pop_if",
    "clippy::duration_suboptimal_units",
    "clippy::unnecessary_trailing_comma",
];

pub fn check() -> Result<()> {
    let root = read_toml(ROOT_CARGO)?;
    let lint_ledger = read_toml(LINT_LEDGER)?;
    let debt_ledger = read_toml(DEBT_LEDGER)?;
    read_toml(NO_PANIC_ALLOWLIST)?;
    read_toml(NON_RUST_ALLOWLIST)?;

    check_msrv(&root, &lint_ledger)?;
    check_workspace_lints(&root)?;
    check_clippy_toml()?;
    check_lint_ledger(&lint_ledger, &root)?;
    check_debt_ledger(&debt_ledger)?;
    let inherited = count_workspace_lint_inheritance()?;

    println!("lint policy check passed ({inherited} manifests inherit workspace lints)");
    Ok(())
}

fn read_toml(path: &str) -> Result<Value> {
    let text = fs::read_to_string(path).with_context(|| format!("read {path}"))?;
    text.parse::<Value>().with_context(|| format!("parse {path}"))
}

fn check_msrv(root: &Value, ledger: &Value) -> Result<()> {
    let cargo_msrv = root
        .get("workspace")
        .and_then(|workspace| workspace.get("package"))
        .and_then(|package| package.get("rust-version"))
        .and_then(Value::as_str)
        .context("missing workspace.package.rust-version")?;
    let ledger_msrv = ledger.get("msrv").and_then(Value::as_str).context("missing policy msrv")?;
    if cargo_msrv != ledger_msrv {
        bail!(
            "workspace.package.rust-version ({cargo_msrv}) must match {LINT_LEDGER} msrv ({ledger_msrv})"
        );
    }
    Ok(())
}

fn check_workspace_lints(root: &Value) -> Result<()> {
    let lints = root
        .get("workspace")
        .and_then(|workspace| workspace.get("lints"))
        .context("missing workspace.lints")?;
    let rust = lints.get("rust").context("missing workspace.lints.rust")?;
    let clippy = lints.get("clippy").context("missing workspace.lints.clippy")?;

    require_level(rust, "unsafe_code", "warn")?;
    require_level(rust, "unsafe_op_in_unsafe_fn", "deny")?;
    require_level(rust, "unused_must_use", "deny")?;

    for lint in REQUIRED_ACTIVE_LINTS {
        if rust.get(*lint).is_none() && clippy.get(*lint).is_none() {
            bail!("missing active workspace lint `{lint}`");
        }
    }
    Ok(())
}

fn require_level(table: &Value, key: &str, expected: &str) -> Result<()> {
    let level =
        table.get(key).and_then(Value::as_str).with_context(|| format!("missing lint `{key}`"))?;
    if level != expected {
        bail!("lint `{key}` is `{level}`, expected `{expected}`");
    }
    Ok(())
}

fn check_clippy_toml() -> Result<()> {
    let text = fs::read_to_string(CLIPPY_TOML).with_context(|| format!("read {CLIPPY_TOML}"))?;
    for carveout in FORBIDDEN_TEST_CARVEOUTS {
        if text.lines().any(|line| line.trim_start().starts_with(carveout)) {
            bail!("{CLIPPY_TOML} contains forbidden test carveout `{carveout}`");
        }
    }
    Ok(())
}

fn check_lint_ledger(ledger: &Value, root: &Value) -> Result<()> {
    let policy = ledger.get("policy").context("missing [policy] in lint ledger")?;
    require_bool(policy, "panic_free_tests", true)?;
    require_bool(policy, "allow_test_carveouts", false)?;
    require_bool(policy, "blanket_categories", false)?;
    require_string(policy, "suppression_style")?;

    let planned = ledger
        .get("planned")
        .and_then(Value::as_array)
        .context("missing [[planned]] lint ledger entries")?;
    for name in REQUIRED_PLANNED_LINTS {
        if !planned.iter().any(|entry| entry.get("name").and_then(Value::as_str) == Some(*name)) {
            bail!("missing planned lint `{name}` in {LINT_LEDGER}");
        }
    }

    let clippy = root
        .get("workspace")
        .and_then(|workspace| workspace.get("lints"))
        .and_then(|lints| lints.get("clippy"))
        .context("missing workspace.lints.clippy")?;
    for entry in planned {
        let name =
            entry.get("name").and_then(Value::as_str).context("planned lint missing name")?;
        let active_key = name.strip_prefix("clippy::").unwrap_or(name);
        if clippy.get(active_key).is_some() {
            bail!("planned lint `{name}` is already active before its MSRV flip");
        }
        require_string(entry, "level")?;
        require_string(entry, "activate_when_msrv")?;
        require_string(entry, "reason")?;
    }
    Ok(())
}

fn require_bool(table: &Value, key: &str, expected: bool) -> Result<()> {
    let value = table
        .get(key)
        .and_then(Value::as_bool)
        .with_context(|| format!("missing boolean `{key}`"))?;
    if value != expected {
        bail!("policy `{key}` is `{value}`, expected `{expected}`");
    }
    Ok(())
}

fn require_string(table: &Value, key: &str) -> Result<()> {
    let value = table
        .get(key)
        .and_then(Value::as_str)
        .with_context(|| format!("missing string `{key}`"))?;
    if value.trim().is_empty() {
        bail!("policy field `{key}` must not be empty");
    }
    Ok(())
}

fn check_debt_ledger(ledger: &Value) -> Result<()> {
    let Some(debts) = ledger.get("debt").and_then(Value::as_array) else {
        return Ok(());
    };
    let today = Utc::now().date_naive();
    for debt in debts {
        for key in ["lint", "path", "owner", "reason", "expires"] {
            require_string(debt, key)?;
        }
        let expires =
            debt.get("expires").and_then(Value::as_str).context("debt missing expires")?;
        let expires = NaiveDate::parse_from_str(expires, "%Y-%m-%d")
            .with_context(|| format!("invalid debt expiry `{expires}`"))?;
        if expires < today {
            bail!("expired clippy debt entry for {} at {}", debt["lint"], debt["path"]);
        }
    }
    Ok(())
}

fn count_workspace_lint_inheritance() -> Result<usize> {
    let mut inherited = 0;
    for entry in WalkDir::new(".").into_iter().filter_entry(include_entry) {
        let entry = entry?;
        if entry.file_name() != "Cargo.toml" {
            continue;
        }
        let path = entry.path();
        let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        if text.contains("[package]") && has_workspace_lints(&text) {
            inherited += 1;
        }
    }
    Ok(inherited)
}

fn include_entry(entry: &walkdir::DirEntry) -> bool {
    let path = entry.path();
    !path.components().any(|component| {
        let name = component.as_os_str();
        name == std::ffi::OsStr::new("target")
            || name == std::ffi::OsStr::new(".git")
            || name == std::ffi::OsStr::new(".cache")
    })
}

fn has_workspace_lints(text: &str) -> bool {
    let mut in_lints = false;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            in_lints = trimmed == "[lints]";
            continue;
        }
        if in_lints && trimmed == "workspace = true" {
            return true;
        }
    }
    false
}
