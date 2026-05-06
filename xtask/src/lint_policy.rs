//! `cargo xtask check-lint-policy` — verify the workspace lint governance
//! ledger (`policy/clippy-lints.toml`), the lint debt ledger
//! (`policy/clippy-debt.toml`), and the no-panic / non-Rust allowlists are
//! self-consistent and consistent with `Cargo.toml` and
//! `rust-toolchain.toml`.
//!
//! This is the PR 1 (governance scaffolding) implementation. It performs the
//! checks that make sense before the strict lint baseline lands:
//!
//! 1. `policy/clippy-lints.toml` parses and has the expected schema version.
//! 2. `policy.msrv` matches `workspace.package.rust-version` in `Cargo.toml`
//!    and the channel pinned by `rust-toolchain.toml`.
//! 3. Every `[[planned]]` entry has a sane `activate_when_msrv` — either the
//!    current MSRV (already-active baseline) or a future Rust release.
//! 4. Every `[[active]]` entry references a real lint name shape
//!    (`<root>::<lint>`) and a recognised level.
//! 5. `policy/clippy-debt.toml` parses; every entry has owner / reason / lint /
//!    expiry; expired debt fails (warn-only in PR 1, gate in PR 2).
//! 6. `policy/no-panic-allowlist.toml` and `policy/non-rust-allowlist.toml`
//!    parse and have the expected schema versions.
//!
//! 7. (PR 2) Every `[[active]]` entry in `policy/clippy-lints.toml` is
//!    reflected in `Cargo.toml` `[workspace.lints.<root>]` at the same
//!    level, and every Cargo-active lint is either in the ledger or is a
//!    blanket category (`all`, `pedantic`, `nursery`, ...).
//!
//! It does *not* yet:
//!   - Walk the AST to enforce the no-panic allowlist (planned: `xtask
//!     check-no-panic-family`).
//!   - Walk the file tree to enforce the non-Rust allowlist (planned: `xtask
//!     check-file-policy`).

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::Deserialize;

const POLICY_DIR: &str = "policy";
const LINTS_TOML: &str = "clippy-lints.toml";
const DEBT_TOML: &str = "clippy-debt.toml";
const NO_PANIC_TOML: &str = "no-panic-allowlist.toml";
const NON_RUST_TOML: &str = "non-rust-allowlist.toml";

const LINTS_SCHEMA: u32 = 1;
const DEBT_SCHEMA: u32 = 1;
const NO_PANIC_SCHEMA: &str = "0.3";
const NON_RUST_SCHEMA: &str = "1.0";

/// Mode in which the checker runs. `Advisory` reports findings but never
/// returns a non-zero exit code; `Strict` fails on any finding. PR 1 wires
/// this up as advisory; PR 2 promotes it to strict.
#[derive(Clone, Copy, Debug)]
pub enum Mode {
    Advisory,
    Strict,
}

#[derive(Debug, Deserialize)]
struct LintsLedger {
    schema: u32,
    msrv: String,
    #[serde(default)]
    policy: Policy,
    #[serde(default)]
    active: Vec<ActiveLint>,
    #[serde(default)]
    planned: Vec<PlannedLint>,
}

#[derive(Debug, Default, Deserialize)]
#[allow(dead_code)] // fields are documented in policy/clippy-lints.toml; checker only reads a subset today
struct Policy {
    #[serde(default)]
    panic_free_tests: bool,
    #[serde(default)]
    allow_test_carveouts: bool,
    #[serde(default)]
    suppression_style: Option<String>,
    #[serde(default)]
    blanket_categories: bool,
    #[serde(default)]
    target_panic_free_tests: bool,
    #[serde(default)]
    target_msrv: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ActiveLint {
    name: String,
    level: String,
    #[allow(dead_code)]
    #[serde(default)]
    class: Option<String>,
    #[allow(dead_code)]
    #[serde(default)]
    reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PlannedLint {
    name: String,
    level: String,
    activate_when_msrv: String,
    #[allow(dead_code)]
    #[serde(default)]
    class: Option<String>,
    reason: Option<String>,
    #[allow(dead_code)]
    #[serde(default)]
    overlay_exception: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct DebtLedger {
    schema: u32,
    #[serde(default)]
    debt: Vec<DebtEntry>,
}

#[derive(Debug, Deserialize)]
struct DebtEntry {
    lint: String,
    path: String,
    owner: String,
    reason: String,
    expires: String,
}

#[derive(Debug, Deserialize)]
struct NoPanicLedger {
    schema_version: String,
    #[serde(default)]
    #[allow(dead_code)]
    allow: Vec<toml::Value>,
}

#[derive(Debug, Deserialize)]
struct NonRustLedger {
    schema_version: String,
    #[serde(default)]
    #[allow(dead_code)]
    allow: Vec<toml::Value>,
}

const VALID_LEVELS: &[&str] = &["allow", "warn", "deny", "forbid"];
const VALID_LINT_PREFIXES: &[&str] = &["clippy::", "rust::", "rustc::", "rustdoc::"];

pub fn run(mode: Mode) -> Result<()> {
    let repo_root = repo_root()?;
    let policy_dir = repo_root.join(POLICY_DIR);
    if !policy_dir.is_dir() {
        bail!(
            "policy directory not found at {} — run from the workspace root or check that PR 1 has landed",
            policy_dir.display()
        );
    }

    let mut findings = Findings::default();

    let lints = load_lints(&policy_dir, &mut findings);
    load_debt(&policy_dir, lints.as_ref(), &mut findings);
    load_no_panic(&policy_dir, &mut findings);
    load_non_rust(&policy_dir, &mut findings);

    if let Some(lints) = lints.as_ref() {
        check_msrv_consistency(&repo_root, lints, &mut findings);
        check_active_lints_match_cargo(&repo_root, lints, &mut findings);
    }

    findings.report(mode)
}

#[derive(Default)]
struct Findings {
    info: Vec<String>,
    warnings: Vec<String>,
    errors: Vec<String>,
}

impl Findings {
    fn info(&mut self, s: impl Into<String>) {
        self.info.push(s.into());
    }
    fn warn(&mut self, s: impl Into<String>) {
        self.warnings.push(s.into());
    }
    fn error(&mut self, s: impl Into<String>) {
        self.errors.push(s.into());
    }

    fn report(self, mode: Mode) -> Result<()> {
        for line in &self.info {
            println!("info: {line}");
        }
        for line in &self.warnings {
            println!("warn: {line}");
        }
        for line in &self.errors {
            println!("error: {line}");
        }

        let summary = format!(
            "lint policy: {} info, {} warnings, {} errors",
            self.info.len(),
            self.warnings.len(),
            self.errors.len()
        );

        match mode {
            Mode::Advisory => {
                println!("{summary} (advisory mode — not failing build)");
                Ok(())
            }
            Mode::Strict => {
                println!("{summary}");
                if self.errors.is_empty() {
                    Ok(())
                } else {
                    bail!("lint policy check failed: {} error(s)", self.errors.len())
                }
            }
        }
    }
}

fn repo_root() -> Result<PathBuf> {
    // The xtask binary is always invoked from the workspace root in CI and in
    // the standard developer workflow, but be defensive: walk upward looking
    // for a Cargo.toml that has [workspace].
    let cwd = std::env::current_dir().context("getting current directory")?;
    for ancestor in cwd.ancestors() {
        let cargo = ancestor.join("Cargo.toml");
        if cargo.is_file()
            && fs::read_to_string(&cargo).map(|s| s.contains("[workspace]")).unwrap_or(false)
        {
            return Ok(ancestor.to_path_buf());
        }
    }
    Ok(cwd)
}

fn load_lints(policy_dir: &Path, findings: &mut Findings) -> Option<LintsLedger> {
    let path = policy_dir.join(LINTS_TOML);
    let text = match fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => {
            findings.error(format!("cannot read {}: {e}", path.display()));
            return None;
        }
    };
    let ledger: LintsLedger = match toml::from_str(&text) {
        Ok(l) => l,
        Err(e) => {
            findings.error(format!("parsing {}: {e}", path.display()));
            return None;
        }
    };

    if ledger.schema != LINTS_SCHEMA {
        findings.error(format!(
            "{}: schema = {}, expected {LINTS_SCHEMA}",
            path.display(),
            ledger.schema
        ));
    }

    let mut seen: BTreeSet<String> = BTreeSet::new();
    for lint in &ledger.active {
        check_lint_name(&lint.name, &path, "active", findings);
        check_level(&lint.level, &lint.name, &path, findings);
        if !seen.insert(format!("active:{}", lint.name)) {
            findings.error(format!(
                "{}: duplicate [[active]] entry for {}",
                path.display(),
                lint.name
            ));
        }
    }
    for lint in &ledger.planned {
        check_lint_name(&lint.name, &path, "planned", findings);
        check_level(&lint.level, &lint.name, &path, findings);
        check_planned_msrv(&lint.activate_when_msrv, &lint.name, &ledger.msrv, &path, findings);
        if lint.reason.as_deref().unwrap_or("").trim().is_empty() {
            findings.error(format!(
                "{}: planned lint {} is missing `reason`",
                path.display(),
                lint.name
            ));
        }
        if !seen.insert(format!("planned:{}:{}", lint.name, lint.activate_when_msrv)) {
            findings.error(format!(
                "{}: duplicate [[planned]] entry for {} @ {}",
                path.display(),
                lint.name,
                lint.activate_when_msrv
            ));
        }
    }

    if ledger.policy.target_panic_free_tests && ledger.policy.allow_test_carveouts {
        findings.info(
            "policy.target_panic_free_tests=true with allow_test_carveouts=true — \
             test carveouts are scheduled for removal in a follow-up PR"
                .to_string(),
        );
    }
    if let Some(style) = &ledger.policy.suppression_style
        && style != "expect-with-reason"
    {
        findings.warn(format!(
            "policy.suppression_style = {style:?}; the workspace standard is \"expect-with-reason\""
        ));
    }

    Some(ledger)
}

fn load_debt(policy_dir: &Path, lints: Option<&LintsLedger>, findings: &mut Findings) {
    let path = policy_dir.join(DEBT_TOML);
    let text = match fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => {
            findings.error(format!("cannot read {}: {e}", path.display()));
            return;
        }
    };
    let ledger: DebtLedger = match toml::from_str(&text) {
        Ok(l) => l,
        Err(e) => {
            findings.error(format!("parsing {}: {e}", path.display()));
            return;
        }
    };
    if ledger.schema != DEBT_SCHEMA {
        findings.error(format!(
            "{}: schema = {}, expected {DEBT_SCHEMA}",
            path.display(),
            ledger.schema
        ));
    }

    let known_lints: BTreeSet<&str> = lints
        .map(|l| {
            l.active
                .iter()
                .map(|a| a.name.as_str())
                .chain(l.planned.iter().map(|p| p.name.as_str()))
                .collect()
        })
        .unwrap_or_default();

    let today = chrono::Utc::now().date_naive();
    for entry in &ledger.debt {
        for (name, value) in [
            ("lint", &entry.lint),
            ("path", &entry.path),
            ("owner", &entry.owner),
            ("reason", &entry.reason),
            ("expires", &entry.expires),
        ] {
            if value.trim().is_empty() {
                findings.error(format!(
                    "{}: debt entry for {} has empty `{}`",
                    path.display(),
                    entry.lint,
                    name
                ));
            }
        }
        if !known_lints.is_empty() && !known_lints.contains(entry.lint.as_str()) {
            findings.error(format!(
                "{}: debt entry references unknown lint {}",
                path.display(),
                entry.lint
            ));
        }
        match chrono::NaiveDate::parse_from_str(&entry.expires, "%Y-%m-%d") {
            Ok(exp) if exp < today => {
                findings.error(format!(
                    "{}: debt entry for {} ({}) expired on {}",
                    path.display(),
                    entry.lint,
                    entry.path,
                    entry.expires
                ));
            }
            Ok(_) => {}
            Err(_) => {
                findings.error(format!(
                    "{}: debt entry for {} has unparseable `expires = {:?}` (want YYYY-MM-DD)",
                    path.display(),
                    entry.lint,
                    entry.expires
                ));
            }
        }
    }
}

fn load_no_panic(policy_dir: &Path, findings: &mut Findings) {
    let path = policy_dir.join(NO_PANIC_TOML);
    let text = match fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => {
            findings.error(format!("cannot read {}: {e}", path.display()));
            return;
        }
    };
    let ledger: NoPanicLedger = match toml::from_str(&text) {
        Ok(l) => l,
        Err(e) => {
            findings.error(format!("parsing {}: {e}", path.display()));
            return;
        }
    };
    if ledger.schema_version != NO_PANIC_SCHEMA {
        findings.error(format!(
            "{}: schema_version = {:?}, expected {:?}",
            path.display(),
            ledger.schema_version,
            NO_PANIC_SCHEMA
        ));
    }
}

fn load_non_rust(policy_dir: &Path, findings: &mut Findings) {
    let path = policy_dir.join(NON_RUST_TOML);
    let text = match fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => {
            findings.error(format!("cannot read {}: {e}", path.display()));
            return;
        }
    };
    let ledger: NonRustLedger = match toml::from_str(&text) {
        Ok(l) => l,
        Err(e) => {
            findings.error(format!("parsing {}: {e}", path.display()));
            return;
        }
    };
    if ledger.schema_version != NON_RUST_SCHEMA {
        findings.error(format!(
            "{}: schema_version = {:?}, expected {:?}",
            path.display(),
            ledger.schema_version,
            NON_RUST_SCHEMA
        ));
    }
}

fn check_lint_name(name: &str, path: &Path, kind: &str, findings: &mut Findings) {
    if !VALID_LINT_PREFIXES.iter().any(|p| name.starts_with(p)) {
        findings.error(format!(
            "{}: {kind} lint {name:?} must start with one of {VALID_LINT_PREFIXES:?}",
            path.display()
        ));
    }
}

fn check_level(level: &str, lint: &str, path: &Path, findings: &mut Findings) {
    if !VALID_LEVELS.contains(&level) {
        findings.error(format!(
            "{}: lint {lint} has invalid level {level:?} (want one of {VALID_LEVELS:?})",
            path.display()
        ));
    }
}

fn check_planned_msrv(
    target: &str,
    lint: &str,
    current: &str,
    path: &Path,
    findings: &mut Findings,
) {
    let target_v = parse_msrv(target);
    let current_v = parse_msrv(current);
    let (Some(target_v), Some(current_v)) = (target_v, current_v) else {
        findings.error(format!(
            "{}: planned lint {lint} has unparseable `activate_when_msrv = {target:?}`",
            path.display()
        ));
        return;
    };
    if target_v < current_v {
        findings.error(format!(
            "{}: planned lint {lint} activates at {target} but workspace MSRV is already {current}",
            path.display()
        ));
    }
}

fn parse_msrv(s: &str) -> Option<(u32, u32)> {
    let mut parts = s.split('.').take(2);
    let major: u32 = parts.next()?.parse().ok()?;
    let minor: u32 = parts.next()?.parse().ok()?;
    Some((major, minor))
}

fn check_msrv_consistency(repo_root: &Path, lints: &LintsLedger, findings: &mut Findings) {
    // workspace.package.rust-version
    let cargo_path = repo_root.join("Cargo.toml");
    if let Ok(cargo) = fs::read_to_string(&cargo_path) {
        match extract_rust_version(&cargo) {
            Some(rust_version) if !msrv_matches(&rust_version, &lints.msrv) => {
                findings.error(format!(
                    "MSRV mismatch: Cargo.toml workspace.package.rust-version = {rust_version:?}, \
                     policy/clippy-lints.toml msrv = {:?}",
                    lints.msrv
                ));
            }
            Some(_) => {}
            None => findings.warn(format!(
                "{}: could not locate workspace.package.rust-version",
                cargo_path.display()
            )),
        }
    }

    // rust-toolchain.toml channel
    let toolchain_path = repo_root.join("rust-toolchain.toml");
    if let Ok(text) = fs::read_to_string(&toolchain_path)
        && let Some(channel) = extract_toolchain_channel(&text)
        && !msrv_matches(&channel, &lints.msrv)
    {
        findings.error(format!(
            "MSRV mismatch: rust-toolchain.toml channel = {channel:?}, \
             policy/clippy-lints.toml msrv = {:?}",
            lints.msrv
        ));
    }
}

fn msrv_matches(left: &str, right: &str) -> bool {
    parse_msrv(left) == parse_msrv(right)
}

fn extract_rust_version(cargo_toml: &str) -> Option<String> {
    // Look for the workspace.package table and pull rust-version. We do this
    // by string scanning rather than a typed parse so we don't have to model
    // the full workspace manifest schema.
    let value: toml::Value = toml::from_str(cargo_toml).ok()?;
    let workspace = value.get("workspace")?.as_table()?;
    let package = workspace.get("package")?.as_table()?;
    let rust_version = package.get("rust-version")?.as_str()?;
    Some(rust_version.to_string())
}

fn extract_toolchain_channel(text: &str) -> Option<String> {
    let value: toml::Value = toml::from_str(text).ok()?;
    let toolchain = value.get("toolchain")?.as_table()?;
    let channel = toolchain.get("channel")?.as_str()?;
    Some(channel.to_string())
}

/// Verify that every `[[active]]` lint in `policy/clippy-lints.toml` is also
/// present in `Cargo.toml` `[workspace.lints.<root>]` at the declared level.
/// The reverse direction (lints active in Cargo.toml without a ledger entry)
/// is also reported. We accept either bare-string levels (`level = "warn"`)
/// or table form with `priority`.
fn check_active_lints_match_cargo(repo_root: &Path, lints: &LintsLedger, findings: &mut Findings) {
    let cargo_path = repo_root.join("Cargo.toml");
    let cargo_text = match fs::read_to_string(&cargo_path) {
        Ok(t) => t,
        Err(_) => return,
    };
    let cargo: toml::Value = match toml::from_str(&cargo_text) {
        Ok(v) => v,
        Err(_) => return,
    };

    let workspace_lints =
        cargo.get("workspace").and_then(|w| w.get("lints")).and_then(|l| l.as_table());
    let Some(workspace_lints) = workspace_lints else {
        findings.warn(format!(
            "{}: workspace.lints table is missing — strict baseline cannot be verified",
            cargo_path.display()
        ));
        return;
    };

    // Build map: ("clippy" | "rust" | ...) -> { lint_name -> level_string }
    let mut cargo_lints: std::collections::BTreeMap<
        String,
        std::collections::BTreeMap<String, String>,
    > = std::collections::BTreeMap::new();
    for (root, body) in workspace_lints {
        let Some(body) = body.as_table() else { continue };
        let bucket = cargo_lints.entry(root.clone()).or_default();
        for (name, value) in body {
            let level = match value {
                toml::Value::String(s) => Some(s.clone()),
                toml::Value::Table(t) => t.get("level").and_then(|v| v.as_str()).map(str::to_owned),
                _ => None,
            };
            if let Some(level) = level {
                bucket.insert(name.clone(), level);
            }
        }
    }

    // Forward direction: every [[active]] entry must be reflected in cargo.
    for active in &lints.active {
        let Some((root, name)) = split_lint_name(&active.name) else { continue };
        let cargo_level = cargo_lints.get(root).and_then(|b| b.get(name));
        match cargo_level {
            None => findings.error(format!(
                "policy/clippy-lints.toml declares [[active]] {} but Cargo.toml has no entry",
                active.name
            )),
            Some(level) if level != &active.level => findings.error(format!(
                "lint {} level mismatch: Cargo.toml = {level:?}, policy/clippy-lints.toml = {:?}",
                active.name, active.level
            )),
            Some(_) => {}
        }
    }

    // Reverse direction: report any cargo-active lint not in the ledger as a
    // warning. Category lints (`all`, `pedantic`, `nursery`, `cargo`,
    // `complexity`, `correctness`, `perf`, `style`, `restriction`,
    // `suspicious`) are accepted without ledger entries until the explicit
    // baseline replaces them in a later PR.
    let known: std::collections::BTreeSet<&str> =
        lints.active.iter().map(|a| a.name.as_str()).collect();
    for (root, bucket) in &cargo_lints {
        for name in bucket.keys() {
            let qualified = format!("{root}::{name}");
            if known.contains(qualified.as_str()) {
                continue;
            }
            if root == "clippy" && is_clippy_category(name) {
                continue;
            }
            findings.warn(format!(
                "Cargo.toml declares {qualified} but policy/clippy-lints.toml has no [[active]] entry"
            ));
        }
    }
}

fn split_lint_name(qualified: &str) -> Option<(&str, &str)> {
    qualified.split_once("::")
}

fn is_clippy_category(name: &str) -> bool {
    matches!(
        name,
        "all"
            | "cargo"
            | "complexity"
            | "correctness"
            | "nursery"
            | "pedantic"
            | "perf"
            | "restriction"
            | "style"
            | "suspicious"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_msrv() {
        assert_eq!(parse_msrv("1.92"), Some((1, 92)));
        assert_eq!(parse_msrv("1.92.0"), Some((1, 92)));
        assert_eq!(parse_msrv("1.93"), Some((1, 93)));
        assert!(parse_msrv("nightly").is_none());
        assert!(parse_msrv("").is_none());
    }

    #[test]
    fn msrv_matching_is_minor_level() {
        assert!(msrv_matches("1.92", "1.92.0"));
        assert!(msrv_matches("1.92.0", "1.92"));
        assert!(!msrv_matches("1.92", "1.93"));
    }
}
