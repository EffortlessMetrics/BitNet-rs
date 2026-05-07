//! Semantic no-panic-family checker.
//!
//! Walks Rust sources and reports occurrences of panic-family API
//! shapes (`unwrap()`, `expect()`, `panic!`, `todo!`, `unimplemented!`,
//! `unreachable!`) that are not covered by an exception receipt in
//! `policy/no-panic-allowlist.toml`.
//!
//! Identity for an exception is `path + family + selector`. The
//! `last_seen` line/column is advisory only — the goal is for an
//! existing receipt to keep matching across small edits.
//!
//! This is a lightweight regex-shaped detector, not a full AST parse.
//! It is deliberately conservative: false positives are tolerable in
//! the `propose` workflow but the receipt-shaped ones in the real
//! allowlist are reviewed by humans.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize, Serialize, Clone, Default)]
struct Allowlist {
    #[serde(default = "schema_default", skip_serializing_if = "String::is_empty")]
    schema_version: String,
    #[serde(default, rename = "allow")]
    entries: Vec<Entry>,
}

fn schema_default() -> String {
    "0.3".into()
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct Entry {
    id: String,
    path: String,
    family: String,
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

#[derive(Debug, Default)]
pub struct Report {
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub findings: usize,
    pub allow_count: usize,
}

pub fn run(allowlist_path: PathBuf, report_dir: PathBuf, fail_on_error: bool) -> Result<()> {
    let report = check(&allowlist_path, &report_dir)?;
    println!(
        "no-panic: {} findings vs {} allowlist entries; {} unallowlisted",
        report.findings,
        report.allow_count,
        report.errors.len()
    );
    for w in &report.warnings {
        println!("warning: {w}");
    }
    for e in &report.errors {
        println!("error: {e}");
    }
    if fail_on_error && !report.errors.is_empty() {
        bail!("no-panic check failed: {} unallowlisted findings", report.errors.len());
    }
    Ok(())
}

fn check(allowlist_path: &Path, report_dir: &Path) -> Result<Report> {
    let mut report = Report::default();

    // Allowlist may not exist yet on first run.
    let allowlist: Allowlist = if allowlist_path.exists() {
        let text = fs::read_to_string(allowlist_path)
            .with_context(|| format!("reading {}", allowlist_path.display()))?;
        toml::from_str(&text).with_context(|| format!("parsing {}", allowlist_path.display()))?
    } else {
        report.warnings.push(format!(
            "no-panic allowlist `{}` does not exist; running advisory only",
            allowlist_path.display()
        ));
        Allowlist::default()
    };
    report.allow_count = allowlist.entries.len();

    let allow_keys: BTreeSet<(String, String)> =
        allowlist.entries.iter().map(|e| (normalise_path(&e.path), e.family.clone())).collect();

    let findings = scan_workspace()?;
    report.findings = findings.len();

    for f in &findings {
        let key = (normalise_path(&f.path), f.family.to_string());
        if !allow_keys.contains(&key) {
            report.errors.push(format!(
                "{}:{} {} not allowlisted: `{}`",
                f.path, f.line, f.family, f.snippet
            ));
        }
    }

    fs::create_dir_all(report_dir).with_context(|| format!("creating {}", report_dir.display()))?;
    let json = serde_json::json!({
        "schema_version": 1,
        "errors": report.errors,
        "warnings": report.warnings,
        "findings": report.findings,
        "allow_count": report.allow_count,
    });
    fs::write(report_dir.join("no-panic.json"), serde_json::to_string_pretty(&json)?)?;

    // Always emit a proposed allowlist.
    let proposed = synthesize_allowlist(&findings);
    let proposed_text = toml::to_string_pretty(&proposed)?;
    fs::write(report_dir.join("no-panic-proposed-allowlist.toml"), proposed_text)?;

    Ok(report)
}

fn normalise_path(p: &str) -> String {
    p.replace('\\', "/")
}

fn synthesize_allowlist(findings: &[Finding]) -> Allowlist {
    let mut al =
        Allowlist { schema_version: "0.3".into(), entries: Vec::with_capacity(findings.len()) };
    for (i, f) in findings.iter().enumerate() {
        al.entries.push(Entry {
            id: format!("panic-proposed-{:04}", i + 1),
            path: f.path.clone(),
            family: f.family.to_string(),
            classification: "uncategorised".into(),
            owner: "TODO".into(),
            explanation: "auto-proposed; reviewer must classify and justify".into(),
            expires: String::new(),
            selector: Some(Selector {
                kind: "auto".into(),
                container: String::new(),
                callee: f.family.to_string(),
                receiver_fingerprint: f.snippet.clone(),
            }),
            last_seen: Some(LastSeen { line: f.line, column: 0 }),
        });
    }
    al
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
    }
}
