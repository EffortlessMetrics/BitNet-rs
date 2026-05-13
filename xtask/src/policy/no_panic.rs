//! Semantic no-panic-family checker.
//!
//! Walks Rust sources and reports occurrences of panic-family API
//! shapes (`unwrap()`, `expect()`, `panic!`, `todo!`, `unimplemented!`,
//! `unreachable!`) that are not covered by an exception receipt in
//! `policy/no-panic-allowlist.toml`.
//!
//! Identity for an exception is exact and counted:
//! `path + family + selector_kind + selector_callee + snippet + count`.
//! The `last_seen` line/column is advisory only.
//!
//! This is a lightweight regex-shaped detector, not a full AST parse.
//! It is deliberately conservative: false positives are tolerable in
//! the `propose` workflow but the receipt-shaped ones in the real
//! allowlist are reviewed by humans.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize, Serialize, Clone, Default)]
struct Allowlist {
    #[serde(default = "schema_default", skip_serializing_if = "String::is_empty")]
    schema_version: String,
    #[serde(default, skip_serializing_if = "PolicyConfig::is_default")]
    policy: PolicyConfig,
    #[serde(default, rename = "allow")]
    entries: Vec<Entry>,
}

fn schema_default() -> String {
    "0.4".into()
}

fn baseline_schema_default() -> String {
    "0.1".into()
}

fn default_count() -> usize {
    1
}

#[derive(Debug, Deserialize, Serialize, Clone, Default)]
struct PolicyConfig {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    mode: String,
}

impl PolicyConfig {
    fn is_default(&self) -> bool {
        self.mode.trim().is_empty()
    }

    fn no_new_debt(&self) -> bool {
        self.mode.trim() == "no-new-debt"
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct Entry {
    id: String,
    path: String,
    family: String,
    #[serde(default)]
    snippet: String,
    #[serde(default = "default_count")]
    count: usize,
    #[serde(default)]
    classification: String,
    #[serde(default)]
    owner: String,
    #[serde(default)]
    explanation: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    expires: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    selector: Option<Selector>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    last_seen: Option<LastSeen>,
}

#[derive(Debug, Deserialize, Serialize, Clone, Default)]
struct Baseline {
    #[serde(default = "baseline_schema_default", skip_serializing_if = "String::is_empty")]
    schema_version: String,
    #[serde(default, rename = "baseline")]
    entries: Vec<BaselineEntry>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct BaselineEntry {
    path: String,
    family: String,
    snippet: String,
    #[serde(default = "default_count")]
    count: usize,
    selector: Selector,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct Selector {
    #[serde(default)]
    kind: String,
    #[serde(default)]
    container: String,
    #[serde(default)]
    callee: String,
    #[serde(default)]
    receiver_fingerprint: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct LastSeen {
    line: usize,
    #[serde(default)]
    column: usize,
}

#[derive(Debug)]
struct Finding {
    path: String,
    family: &'static str,
    line: usize,
    snippet: String,
}

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd)]
struct FindingKey {
    path: String,
    family: String,
    selector_kind: String,
    selector_callee: String,
    snippet: String,
}

#[derive(Debug, Clone)]
struct FindingBucket {
    key: FindingKey,
    count: usize,
    first_line: usize,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum MatchingMode {
    NoNewDebt,
    Blocking,
}

#[derive(Debug, Default)]
struct MatchOutcome {
    errors: Vec<String>,
    allow_consumed: usize,
    baseline_consumed: usize,
    new_debt: usize,
}

#[derive(Debug, Default)]
pub struct Report {
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub findings: usize,
    pub allow_count: usize,
    pub baseline_count: usize,
    pub policy_mode: String,
}

pub fn run(
    allowlist_path: PathBuf,
    baseline_path: PathBuf,
    report_dir: PathBuf,
    fail_on_error: bool,
    blocking_mode: bool,
) -> Result<()> {
    let mode = if blocking_mode { MatchingMode::Blocking } else { MatchingMode::NoNewDebt };
    let report = check(&allowlist_path, &baseline_path, &report_dir, mode)?;
    let policy_enforces_no_new_debt = report.policy_mode.trim() == "no-new-debt";
    println!(
        "no-panic: {} findings vs {} allowlist entries and {} baseline entries; {} unallowlisted ({})",
        report.findings,
        report.allow_count,
        report.baseline_count,
        report.errors.len(),
        report.policy_mode
    );
    for w in &report.warnings {
        println!("warning: {w}");
    }
    for e in &report.errors {
        println!("error: {e}");
    }
    if (fail_on_error || policy_enforces_no_new_debt) && !report.errors.is_empty() {
        bail!("no-panic check failed: {} unallowlisted findings", report.errors.len());
    }
    Ok(())
}

pub fn baseline(
    allowlist_path: PathBuf,
    baseline_path: PathBuf,
    report_dir: PathBuf,
    reset: bool,
) -> Result<()> {
    let allowlist = load_allowlist(&allowlist_path)?;
    let allow_counts = validate_allowlist(&allowlist)?;
    let findings = scan_workspace()?;
    let next = synthesize_baseline(&findings, &allow_counts);

    if baseline_path.exists() && !reset {
        let existing = load_baseline(&baseline_path)?;
        let existing_counts = validate_baseline(&existing)?;
        let new_debt = baseline_new_debt(&next, &existing_counts)?;
        if !new_debt.is_empty() {
            bail!(
                "no-panic baseline refresh refused to absorb {} new finding(s); rerun with --reset only in the dedicated baseline reset PR\n{}",
                new_debt.len(),
                new_debt.join("\n")
            );
        }
    } else if !baseline_path.exists() && !reset {
        bail!(
            "no-panic baseline `{}` is missing; rerun with --reset in the dedicated baseline PR",
            baseline_path.display()
        );
    }

    if let Some(parent) = baseline_path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    let text = toml::to_string_pretty(&next)?;
    fs::write(&baseline_path, text)
        .with_context(|| format!("writing {}", baseline_path.display()))?;

    fs::create_dir_all(&report_dir)
        .with_context(|| format!("creating {}", report_dir.display()))?;
    fs::write(
        report_dir.join("no-panic-baseline-refresh.json"),
        serde_json::to_string_pretty(&serde_json::json!({
            "schema_version": 1,
            "reset": reset,
            "findings": findings.len(),
            "baseline_entries": next.entries.len(),
            "baseline_path": baseline_path.display().to_string(),
        }))?,
    )?;

    println!(
        "no-panic baseline: wrote {} entries to {}",
        next.entries.len(),
        baseline_path.display()
    );
    Ok(())
}

fn check(
    allowlist_path: &Path,
    baseline_path: &Path,
    report_dir: &Path,
    mode: MatchingMode,
) -> Result<Report> {
    let mut report = Report::default();

    // Allowlist may not exist yet on first run.
    let allowlist: Allowlist = if allowlist_path.exists() {
        load_allowlist(allowlist_path)?
    } else {
        report.warnings.push(format!(
            "no-panic allowlist `{}` does not exist; running advisory only",
            allowlist_path.display()
        ));
        Allowlist::default()
    };
    report.policy_mode =
        if allowlist.policy.no_new_debt() { "no-new-debt".into() } else { "advisory".into() };
    report.allow_count = allowlist.entries.len();
    let allow_counts = validate_allowlist(&allowlist)?;

    let baseline: Baseline =
        if baseline_path.exists() { load_baseline(baseline_path)? } else { Baseline::default() };
    report.baseline_count = baseline.entries.len();
    let baseline_counts = validate_baseline(&baseline)?;

    let findings = scan_workspace()?;
    report.findings = findings.len();

    let outcome = match_findings(&findings, &allow_counts, &baseline_counts, mode);
    report.errors = outcome.errors;

    fs::create_dir_all(report_dir).with_context(|| format!("creating {}", report_dir.display()))?;
    let json = serde_json::json!({
        "schema_version": 1,
        "errors": report.errors,
        "warnings": report.warnings,
        "findings": report.findings,
        "allow_count": report.allow_count,
        "baseline_count": report.baseline_count,
        "allow_consumed": outcome.allow_consumed,
        "baseline_consumed": outcome.baseline_consumed,
        "new_debt": outcome.new_debt,
        "matching_mode": match mode {
            MatchingMode::NoNewDebt => "no-new-debt",
            MatchingMode::Blocking => "blocking",
        },
        "policy_mode": report.policy_mode,
    });
    fs::write(report_dir.join("no-panic.json"), serde_json::to_string_pretty(&json)?)?;

    // Always emit a proposed allowlist.
    let proposed = synthesize_allowlist(&findings);
    let proposed_text = toml::to_string_pretty(&proposed)?;
    fs::write(report_dir.join("no-panic-proposed-allowlist.toml"), proposed_text)?;

    let proposed_baseline = synthesize_baseline(&findings, &allow_counts);
    let proposed_baseline_text = toml::to_string_pretty(&proposed_baseline)?;
    fs::write(report_dir.join("no-panic-proposed-baseline.toml"), proposed_baseline_text)?;

    Ok(report)
}

fn load_allowlist(path: &Path) -> Result<Allowlist> {
    let text = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

fn load_baseline(path: &Path) -> Result<Baseline> {
    let text = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

fn normalise_path(p: &str) -> String {
    p.replace('\\', "/")
}

fn normalise_snippet(snippet: &str) -> String {
    snippet.trim().to_string()
}

fn selector_for_family(family: &str) -> (&'static str, &str) {
    match family {
        "unwrap" | "expect" => ("method_call", family),
        "panic_macro" => ("macro_call", "panic"),
        "todo" => ("macro_call", "todo"),
        "unimplemented" => ("macro_call", "unimplemented"),
        "unreachable" => ("macro_call", "unreachable"),
        _ => ("call", family),
    }
}

impl Finding {
    fn key(&self) -> FindingKey {
        let (selector_kind, selector_callee) = selector_for_family(self.family);
        FindingKey {
            path: normalise_path(&self.path),
            family: self.family.to_string(),
            selector_kind: selector_kind.to_string(),
            selector_callee: selector_callee.to_string(),
            snippet: normalise_snippet(&self.snippet),
        }
    }
}

impl Entry {
    fn key(&self) -> Result<FindingKey> {
        if normalise_snippet(&self.snippet).is_empty() {
            bail!("no-panic allowlist entry `{}` must set non-empty `snippet`", self.id);
        }
        if self.count == 0 {
            bail!("no-panic allowlist entry `{}` must set positive `count`", self.id);
        }
        let selector = self.selector.as_ref().with_context(|| {
            format!("no-panic allowlist entry `{}` must set `selector`", self.id)
        })?;
        if selector.kind.trim().is_empty() {
            bail!("no-panic allowlist entry `{}` must set selector.kind", self.id);
        }
        if selector.callee.trim().is_empty() {
            bail!("no-panic allowlist entry `{}` must set selector.callee", self.id);
        }
        Ok(FindingKey {
            path: normalise_path(&self.path),
            family: self.family.clone(),
            selector_kind: selector.kind.trim().to_string(),
            selector_callee: selector.callee.trim().to_string(),
            snippet: normalise_snippet(&self.snippet),
        })
    }
}

impl BaselineEntry {
    fn key(&self) -> Result<FindingKey> {
        if normalise_snippet(&self.snippet).is_empty() {
            bail!("no-panic baseline entry for `{}` must set non-empty `snippet`", self.path);
        }
        if self.count == 0 {
            bail!("no-panic baseline entry for `{}` must set positive `count`", self.path);
        }
        if self.selector.kind.trim().is_empty() {
            bail!("no-panic baseline entry for `{}` must set selector.kind", self.path);
        }
        if self.selector.callee.trim().is_empty() {
            bail!("no-panic baseline entry for `{}` must set selector.callee", self.path);
        }
        Ok(FindingKey {
            path: normalise_path(&self.path),
            family: self.family.clone(),
            selector_kind: self.selector.kind.trim().to_string(),
            selector_callee: self.selector.callee.trim().to_string(),
            snippet: normalise_snippet(&self.snippet),
        })
    }
}

fn validate_allowlist(allowlist: &Allowlist) -> Result<BTreeMap<FindingKey, usize>> {
    let mut counts = BTreeMap::new();
    for entry in &allowlist.entries {
        let key = entry.key()?;
        if counts.insert(key, entry.count).is_some() {
            bail!("duplicate no-panic allowlist key for entry `{}`", entry.id);
        }
    }
    Ok(counts)
}

fn validate_baseline(baseline: &Baseline) -> Result<BTreeMap<FindingKey, usize>> {
    let mut counts = BTreeMap::new();
    for entry in &baseline.entries {
        let key = entry.key()?;
        if counts.insert(key, entry.count).is_some() {
            bail!("duplicate no-panic baseline key for `{}` `{}`", entry.path, entry.snippet);
        }
    }
    Ok(counts)
}

fn bucket_findings(findings: &[Finding]) -> BTreeMap<FindingKey, FindingBucket> {
    let mut buckets = BTreeMap::new();
    for finding in findings {
        let key = finding.key();
        buckets
            .entry(key.clone())
            .and_modify(|bucket: &mut FindingBucket| bucket.count += 1)
            .or_insert(FindingBucket { key, count: 1, first_line: finding.line });
    }
    buckets
}

fn match_findings(
    findings: &[Finding],
    allow_counts: &BTreeMap<FindingKey, usize>,
    baseline_counts: &BTreeMap<FindingKey, usize>,
    mode: MatchingMode,
) -> MatchOutcome {
    let mut outcome = MatchOutcome::default();

    for bucket in bucket_findings(findings).values() {
        let mut remaining = bucket.count;

        let allow_slots = allow_counts.get(&bucket.key).copied().unwrap_or(0);
        let allow_used = remaining.min(allow_slots);
        remaining -= allow_used;
        outcome.allow_consumed += allow_used;

        if mode == MatchingMode::NoNewDebt {
            let baseline_slots = baseline_counts.get(&bucket.key).copied().unwrap_or(0);
            let baseline_used = remaining.min(baseline_slots);
            remaining -= baseline_used;
            outcome.baseline_consumed += baseline_used;
        }

        if remaining > 0 {
            outcome.new_debt += remaining;
            outcome.errors.push(format!(
                "{}:{} {} not covered by exact no-panic identity ({} occurrence{} new): `{}`",
                bucket.key.path,
                bucket.first_line,
                bucket.key.family,
                remaining,
                if remaining == 1 { "" } else { "s" },
                bucket.key.snippet
            ));
        }
    }

    outcome
}

fn synthesize_allowlist(findings: &[Finding]) -> Allowlist {
    let buckets = bucket_findings(findings);
    let mut al = Allowlist {
        schema_version: "0.4".into(),
        policy: PolicyConfig::default(),
        entries: Vec::with_capacity(buckets.len()),
    };
    for (i, bucket) in buckets.values().enumerate() {
        let f = &bucket.key;
        al.entries.push(Entry {
            id: format!("panic-proposed-{:04}", i + 1),
            path: f.path.clone(),
            family: f.family.clone(),
            snippet: f.snippet.clone(),
            count: bucket.count,
            classification: "uncategorised".into(),
            owner: "TODO".into(),
            explanation: "auto-proposed; reviewer must classify and justify".into(),
            expires: String::new(),
            selector: Some(Selector {
                kind: f.selector_kind.clone(),
                container: String::new(),
                callee: f.selector_callee.clone(),
                receiver_fingerprint: f.snippet.clone(),
            }),
            last_seen: Some(LastSeen { line: bucket.first_line, column: 0 }),
        });
    }
    al
}

fn synthesize_baseline(
    findings: &[Finding],
    allow_counts: &BTreeMap<FindingKey, usize>,
) -> Baseline {
    let mut baseline = Baseline { schema_version: "0.1".into(), entries: Vec::new() };

    for bucket in bucket_findings(findings).values() {
        let allow_slots = allow_counts.get(&bucket.key).copied().unwrap_or(0);
        let remaining = bucket.count.saturating_sub(allow_slots);
        if remaining == 0 {
            continue;
        }
        baseline.entries.push(BaselineEntry {
            path: bucket.key.path.clone(),
            family: bucket.key.family.clone(),
            snippet: bucket.key.snippet.clone(),
            count: remaining,
            selector: Selector {
                kind: bucket.key.selector_kind.clone(),
                container: String::new(),
                callee: bucket.key.selector_callee.clone(),
                receiver_fingerprint: bucket.key.snippet.clone(),
            },
        });
    }

    baseline
}

fn baseline_new_debt(
    next: &Baseline,
    existing_counts: &BTreeMap<FindingKey, usize>,
) -> Result<Vec<String>> {
    let mut debt = Vec::new();
    for entry in &next.entries {
        let key = entry.key()?;
        let existing = existing_counts.get(&key).copied().unwrap_or(0);
        if entry.count > existing {
            debt.push(format!(
                "{} {} gained {} occurrence(s): `{}`",
                entry.path,
                entry.family,
                entry.count - existing,
                entry.snippet
            ));
        }
    }
    Ok(debt)
}

fn scan_workspace() -> Result<Vec<Finding>> {
    let mut out = Vec::new();
    for entry in walkdir::WalkDir::new(".").into_iter().filter_map(std::result::Result::ok) {
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
        scan_file(&path_str, &body, &mut out);
    }
    Ok(out)
}

fn scan_file(path: &str, body: &str, out: &mut Vec<Finding>) {
    for (i, raw) in body.lines().enumerate() {
        let line_no = i + 1;
        let trimmed = raw.trim_start();
        if trimmed.starts_with("//") || trimmed.starts_with("///") || trimmed.starts_with("//!") {
            continue;
        }
        // Skip cfg(test) modules cheaply: this is best-effort. The
        // full rule (test code is part of the contract) is enforced
        // by promoting the lints in PR 12; here we are only producing
        // semantic receipts.
        if let Some(family) = classify(raw) {
            out.push(Finding {
                path: path.to_string(),
                family,
                line: line_no,
                snippet: raw.trim().chars().take(160).collect(),
            });
        }
    }
}

fn classify(line: &str) -> Option<&'static str> {
    if line.contains(".unwrap()") {
        return Some("unwrap");
    }
    if line.contains(".expect(") {
        return Some("expect");
    }
    if line.contains("panic!(") {
        return Some("panic_macro");
    }
    if line.contains("todo!(") || line.trim_end() == "todo!()" {
        return Some("todo");
    }
    if line.contains("unimplemented!(") || line.trim_end() == "unimplemented!()" {
        return Some("unimplemented");
    }
    if line.contains("unreachable!(") || line.trim_end() == "unreachable!()" {
        return Some("unreachable");
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_panic_family() {
        assert_eq!(classify("    let x = foo.unwrap();"), Some("unwrap"));
        assert_eq!(classify("    let x = foo.expect(\"y\");"), Some("expect"));
        assert_eq!(classify("    panic!(\"x\");"), Some("panic_macro"));
        assert_eq!(classify("    todo!();"), Some("todo"));
        assert_eq!(classify("    unimplemented!();"), Some("unimplemented"));
        assert_eq!(classify("    unreachable!();"), Some("unreachable"));
        assert_eq!(classify("    let x = foo;"), None);
        assert_eq!(classify("    // foo.unwrap();"), Some("unwrap")); // line-level, ok
    }

    #[test]
    fn synthesizes_allowlist() {
        let findings = vec![Finding {
            path: "crates/x/src/lib.rs".into(),
            family: "unwrap",
            line: 10,
            snippet: "foo.unwrap()".into(),
        }];
        let al = synthesize_allowlist(&findings);
        assert_eq!(al.entries.len(), 1);
        assert_eq!(al.entries[0].family, "unwrap");
        assert_eq!(al.entries[0].snippet, "foo.unwrap()");
        assert_eq!(al.entries[0].count, 1);
    }

    #[test]
    fn allowlist_entry_requires_exact_snippet() -> Result<()> {
        let allowlist = Allowlist {
            schema_version: "0.4".into(),
            policy: PolicyConfig::default(),
            entries: vec![allow_entry("panic-0001", "crates/x/src/lib.rs", "unwrap", "", 1)],
        };

        let err = match validate_allowlist(&allowlist) {
            Ok(_) => anyhow::bail!("empty snippet was accepted"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("snippet"));
        Ok(())
    }

    #[test]
    fn allowlist_count_is_consumed_per_occurrence() -> Result<()> {
        let findings = vec![
            finding("crates/x/src/lib.rs", "unwrap", 10, "foo.unwrap()"),
            finding("crates/x/src/lib.rs", "unwrap", 11, "foo.unwrap()"),
        ];
        let allowlist = Allowlist {
            schema_version: "0.4".into(),
            policy: PolicyConfig::default(),
            entries: vec![allow_entry(
                "panic-0001",
                "crates/x/src/lib.rs",
                "unwrap",
                "foo.unwrap()",
                1,
            )],
        };
        let allow_counts = validate_allowlist(&allowlist)?;
        let baseline_counts = BTreeMap::new();

        let outcome =
            match_findings(&findings, &allow_counts, &baseline_counts, MatchingMode::NoNewDebt);

        assert_eq!(outcome.allow_consumed, 1);
        assert_eq!(outcome.new_debt, 1);
        assert_eq!(outcome.errors.len(), 1);
        Ok(())
    }

    #[test]
    fn allowlist_does_not_cover_same_file_same_callee_different_snippet() -> Result<()> {
        let findings = vec![finding("crates/x/src/lib.rs", "unwrap", 10, "right.unwrap()")];
        let allowlist = Allowlist {
            schema_version: "0.4".into(),
            policy: PolicyConfig::default(),
            entries: vec![allow_entry(
                "panic-0001",
                "crates/x/src/lib.rs",
                "unwrap",
                "left.unwrap()",
                1,
            )],
        };
        let allow_counts = validate_allowlist(&allowlist)?;
        let baseline_counts = BTreeMap::new();

        let outcome =
            match_findings(&findings, &allow_counts, &baseline_counts, MatchingMode::NoNewDebt);

        assert_eq!(outcome.allow_consumed, 0);
        assert_eq!(outcome.new_debt, 1);
        assert!(outcome.errors[0].contains("right.unwrap()"));
        Ok(())
    }

    #[test]
    fn baseline_generation_subtracts_allowlisted_counts() -> Result<()> {
        let findings = vec![
            finding("crates/x/src/lib.rs", "unwrap", 10, "foo.unwrap()"),
            finding("crates/x/src/lib.rs", "unwrap", 11, "foo.unwrap()"),
            finding("crates/x/src/lib.rs", "unwrap", 12, "foo.unwrap()"),
        ];
        let allowlist = Allowlist {
            schema_version: "0.4".into(),
            policy: PolicyConfig::default(),
            entries: vec![allow_entry(
                "panic-0001",
                "crates/x/src/lib.rs",
                "unwrap",
                "foo.unwrap()",
                2,
            )],
        };
        let allow_counts = validate_allowlist(&allowlist)?;

        let baseline = synthesize_baseline(&findings, &allow_counts);

        assert_eq!(baseline.entries.len(), 1);
        assert_eq!(baseline.entries[0].count, 1);
        assert_eq!(baseline.entries[0].snippet, "foo.unwrap()");
        Ok(())
    }

    #[test]
    fn baseline_refresh_detects_new_debt() -> Result<()> {
        let next = Baseline {
            schema_version: "0.1".into(),
            entries: vec![baseline_entry("crates/x/src/lib.rs", "unwrap", "foo.unwrap()", 2)],
        };
        let existing = Baseline {
            schema_version: "0.1".into(),
            entries: vec![baseline_entry("crates/x/src/lib.rs", "unwrap", "foo.unwrap()", 1)],
        };
        let existing_counts = validate_baseline(&existing)?;

        let debt = baseline_new_debt(&next, &existing_counts)?;

        assert_eq!(debt.len(), 1);
        assert!(debt[0].contains("gained 1 occurrence"));
        Ok(())
    }

    #[test]
    fn blocking_mode_ignores_baseline_but_honors_counted_allowlist() -> Result<()> {
        let findings = vec![
            finding("crates/x/src/lib.rs", "unwrap", 10, "foo.unwrap()"),
            finding("crates/x/src/lib.rs", "unwrap", 11, "foo.unwrap()"),
        ];
        let allowlist = Allowlist {
            schema_version: "0.4".into(),
            policy: PolicyConfig::default(),
            entries: vec![allow_entry(
                "panic-0001",
                "crates/x/src/lib.rs",
                "unwrap",
                "foo.unwrap()",
                1,
            )],
        };
        let baseline = Baseline {
            schema_version: "0.1".into(),
            entries: vec![baseline_entry("crates/x/src/lib.rs", "unwrap", "foo.unwrap()", 1)],
        };
        let allow_counts = validate_allowlist(&allowlist)?;
        let baseline_counts = validate_baseline(&baseline)?;

        let advisory =
            match_findings(&findings, &allow_counts, &baseline_counts, MatchingMode::NoNewDebt);
        let blocking =
            match_findings(&findings, &allow_counts, &baseline_counts, MatchingMode::Blocking);

        assert_eq!(advisory.errors.len(), 0);
        assert_eq!(advisory.baseline_consumed, 1);
        assert_eq!(blocking.allow_consumed, 1);
        assert_eq!(blocking.baseline_consumed, 0);
        assert_eq!(blocking.new_debt, 1);
        Ok(())
    }

    #[test]
    fn duplicate_allowlist_keys_are_rejected() -> Result<()> {
        let allowlist = Allowlist {
            schema_version: "0.4".into(),
            policy: PolicyConfig::default(),
            entries: vec![
                allow_entry("panic-0001", "crates/x/src/lib.rs", "unwrap", "foo.unwrap()", 1),
                allow_entry("panic-0002", "crates/x/src/lib.rs", "unwrap", "foo.unwrap()", 1),
            ],
        };

        let err = match validate_allowlist(&allowlist) {
            Ok(_) => anyhow::bail!("duplicate allowlist key was accepted"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("duplicate no-panic allowlist key"));
        Ok(())
    }

    fn finding(path: &str, family: &'static str, line: usize, snippet: &str) -> Finding {
        Finding { path: path.into(), family, line, snippet: snippet.into() }
    }

    fn allow_entry(id: &str, path: &str, family: &str, snippet: &str, count: usize) -> Entry {
        let (selector_kind, selector_callee) = selector_for_family(family);
        Entry {
            id: id.into(),
            path: path.into(),
            family: family.into(),
            snippet: snippet.into(),
            count,
            classification: "test".into(),
            owner: "tests".into(),
            explanation: "test fixture".into(),
            expires: String::new(),
            selector: Some(Selector {
                kind: selector_kind.into(),
                container: String::new(),
                callee: selector_callee.into(),
                receiver_fingerprint: snippet.into(),
            }),
            last_seen: None,
        }
    }

    fn baseline_entry(path: &str, family: &str, snippet: &str, count: usize) -> BaselineEntry {
        let (selector_kind, selector_callee) = selector_for_family(family);
        BaselineEntry {
            path: path.into(),
            family: family.into(),
            snippet: snippet.into(),
            count,
            selector: Selector {
                kind: selector_kind.into(),
                container: String::new(),
                callee: selector_callee.into(),
                receiver_fingerprint: snippet.into(),
            },
        }
    }
}
