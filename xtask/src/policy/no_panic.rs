//! Semantic no-panic allowlist enforcement.
//!
//! BitNet-rs governs panic-family call sites — `unwrap()`, `expect()`,
//! `panic!`, `todo!`, `unimplemented!`, `unreachable!` — via
//! `policy/no-panic-allowlist.toml`. Identity is `path + family + selector`
//! where the selector captures the call's *meaning* (kind + container +
//! callee, with an optional receiver fingerprint). Line/column live under
//! `last_seen` and are *advisory* — they help locate the call site after
//! refactors but never form part of the matching key.
//!
//! The scanner uses `syn` to parse each `.rs` file, walks the AST, and
//! emits findings whose selector fields are stable under benign code
//! motion. Line/column come from `proc-macro2`'s `span-locations` feature.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use quote::ToTokens;
use serde::{Deserialize, Serialize};
use syn::spanned::Spanned;
use syn::visit::Visit;

const ALLOWLIST_PATH: &str = "policy/no-panic-allowlist.toml";
const SCHEMA_VERSION: &str = "0.3";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Family {
    Unwrap,
    Expect,
    PanicMacro,
    Todo,
    Unimplemented,
    Unreachable,
}

impl Family {
    fn name(self) -> &'static str {
        match self {
            Family::Unwrap => "unwrap",
            Family::Expect => "expect",
            Family::PanicMacro => "panic_macro",
            Family::Todo => "todo",
            Family::Unimplemented => "unimplemented",
            Family::Unreachable => "unreachable",
        }
    }

    fn parse(s: &str) -> Option<Family> {
        Some(match s {
            "unwrap" => Family::Unwrap,
            "expect" => Family::Expect,
            "panic_macro" => Family::PanicMacro,
            "todo" => Family::Todo,
            "unimplemented" => Family::Unimplemented,
            "unreachable" => Family::Unreachable,
            _ => return None,
        })
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Allowlist {
    pub schema_version: String,
    #[serde(default, rename = "allow")]
    pub entries: Vec<Entry>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Entry {
    pub path: String,
    pub family: String,
    pub classification: String,
    pub owner: String,
    pub explanation: String,
    pub selector: Selector,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_seen: Option<LastSeen>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires: Option<String>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub retired: bool,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Selector {
    pub kind: String,
    #[serde(default)]
    pub container: String,
    pub callee: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub receiver_fingerprint: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct LastSeen {
    pub line: usize,
    pub column: usize,
}

fn is_false(b: &bool) -> bool {
    !*b
}

#[derive(Debug, Clone)]
pub struct Finding {
    pub path: PathBuf,
    pub family: Family,
    pub selector: Selector,
    pub line: usize,
    pub column: usize,
}

#[derive(Debug, Default)]
pub struct CheckOutcome {
    pub files_scanned: usize,
    pub findings: Vec<Finding>,
    pub matched: usize,
    pub unallowlisted: Vec<Finding>,
    pub unused_entries: Vec<usize>,
    pub expired_entries: Vec<(usize, String)>,
    pub schema_errors: Vec<String>,
    pub drift_hints: Vec<String>,
}

impl CheckOutcome {
    pub fn has_failures(&self, strict: bool) -> bool {
        if !self.schema_errors.is_empty() || !self.expired_entries.is_empty() {
            return true;
        }
        if strict && (!self.unallowlisted.is_empty() || !self.unused_entries.is_empty()) {
            return true;
        }
        false
    }
}

pub fn load_allowlist(repo_root: &Path) -> Result<Allowlist> {
    let path = repo_root.join(ALLOWLIST_PATH);
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let list: Allowlist =
        toml::from_str(&raw).with_context(|| format!("failed to parse {}", path.display()))?;
    if list.schema_version != SCHEMA_VERSION {
        bail!(
            "{}: schema_version {} not supported (expected {})",
            path.display(),
            list.schema_version,
            SCHEMA_VERSION
        );
    }
    Ok(list)
}

/// Scan a single Rust file and emit findings for every panic-family call.
pub fn scan_file(path: &Path, source: &str) -> Result<Vec<Finding>> {
    let file = match syn::parse_file(source) {
        Ok(f) => f,
        // Skip files that don't parse — they may use unstable syntax or
        // are generated. Surface a drift hint via Result rather than fail
        // the whole run.
        Err(e) => bail!("syn parse {}: {}", path.display(), e),
    };
    let mut visitor = PanicFamilyVisitor::new(path.to_path_buf());
    visitor.visit_file(&file);
    Ok(visitor.findings)
}

struct PanicFamilyVisitor {
    path: PathBuf,
    container_stack: Vec<String>,
    findings: Vec<Finding>,
}

impl PanicFamilyVisitor {
    fn new(path: PathBuf) -> Self {
        Self { path, container_stack: Vec::new(), findings: Vec::new() }
    }

    fn current_container(&self) -> String {
        self.container_stack.last().cloned().unwrap_or_else(|| "<module-scope>".to_string())
    }
}

impl<'ast> Visit<'ast> for PanicFamilyVisitor {
    fn visit_item_fn(&mut self, node: &'ast syn::ItemFn) {
        self.container_stack.push(node.sig.ident.to_string());
        syn::visit::visit_item_fn(self, node);
        self.container_stack.pop();
    }

    fn visit_impl_item_fn(&mut self, node: &'ast syn::ImplItemFn) {
        self.container_stack.push(node.sig.ident.to_string());
        syn::visit::visit_impl_item_fn(self, node);
        self.container_stack.pop();
    }

    fn visit_trait_item_fn(&mut self, node: &'ast syn::TraitItemFn) {
        self.container_stack.push(node.sig.ident.to_string());
        syn::visit::visit_trait_item_fn(self, node);
        self.container_stack.pop();
    }

    fn visit_expr_method_call(&mut self, node: &'ast syn::ExprMethodCall) {
        let method = node.method.to_string();
        let family = match method.as_str() {
            "unwrap" => Some(Family::Unwrap),
            "expect" => Some(Family::Expect),
            _ => None,
        };
        if let Some(family) = family {
            let span = node.method.span();
            let start = span.start();
            let receiver = node.receiver.to_token_stream().to_string();
            let selector = Selector {
                kind: "method_call".to_string(),
                container: self.current_container(),
                callee: method,
                receiver_fingerprint: Some(normalize_receiver(&receiver)),
            };
            self.findings.push(Finding {
                path: self.path.clone(),
                family,
                selector,
                line: start.line,
                column: start.column,
            });
        }
        syn::visit::visit_expr_method_call(self, node);
    }

    fn visit_expr_macro(&mut self, node: &'ast syn::ExprMacro) {
        if let Some(family) = macro_family(&node.mac.path) {
            let span = node.mac.path.span();
            let start = span.start();
            let callee =
                node.mac.path.segments.last().map(|s| s.ident.to_string()).unwrap_or_default();
            let selector = Selector {
                kind: "macro_call".to_string(),
                container: self.current_container(),
                callee,
                receiver_fingerprint: None,
            };
            self.findings.push(Finding {
                path: self.path.clone(),
                family,
                selector,
                line: start.line,
                column: start.column,
            });
        }
        syn::visit::visit_expr_macro(self, node);
    }

    // Statements like `panic!(...)` followed by `;` are wrapped in `Stmt::Macro`.
    fn visit_stmt_macro(&mut self, node: &'ast syn::StmtMacro) {
        if let Some(family) = macro_family(&node.mac.path) {
            let span = node.mac.path.span();
            let start = span.start();
            let callee =
                node.mac.path.segments.last().map(|s| s.ident.to_string()).unwrap_or_default();
            let selector = Selector {
                kind: "macro_call".to_string(),
                container: self.current_container(),
                callee,
                receiver_fingerprint: None,
            };
            self.findings.push(Finding {
                path: self.path.clone(),
                family,
                selector,
                line: start.line,
                column: start.column,
            });
        }
        syn::visit::visit_stmt_macro(self, node);
    }
}

fn macro_family(path: &syn::Path) -> Option<Family> {
    // We only care about unqualified or `std::`/`core::` qualified macros.
    let last = path.segments.last()?.ident.to_string();
    match last.as_str() {
        "panic" => Some(Family::PanicMacro),
        "todo" => Some(Family::Todo),
        "unimplemented" => Some(Family::Unimplemented),
        "unreachable" => Some(Family::Unreachable),
        _ => None,
    }
}

/// Reduce a receiver token stream to a stable shape: collapse runs of
/// whitespace, strip trailing whitespace, and cap length so churn-prone
/// arguments don't dominate the fingerprint.
fn normalize_receiver(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut prev_space = false;
    for ch in raw.chars() {
        if ch.is_whitespace() {
            if !prev_space {
                out.push(' ');
                prev_space = true;
            }
        } else {
            out.push(ch);
            prev_space = false;
        }
    }
    let trimmed = out.trim().to_string();
    if trimmed.len() > 160 { format!("{}...", &trimmed[..160]) } else { trimmed }
}

/// Match a finding against an allowlist entry. Identity is path + family +
/// selector (kind, container, callee). Receiver fingerprint, when both
/// finding and entry have it, is also matched. last_seen is *not* part of
/// identity.
fn entry_matches(entry: &Entry, finding: &Finding) -> bool {
    if entry.retired {
        return false;
    }
    if PathBuf::from(&entry.path) != finding.path {
        return false;
    }
    let Some(entry_family) = Family::parse(&entry.family) else {
        return false;
    };
    if entry_family != finding.family {
        return false;
    }
    if entry.selector.kind != finding.selector.kind {
        return false;
    }
    if entry.selector.container != finding.selector.container {
        return false;
    }
    if entry.selector.callee != finding.selector.callee {
        return false;
    }
    match (&entry.selector.receiver_fingerprint, &finding.selector.receiver_fingerprint) {
        (Some(a), Some(b)) if a != b => return false,
        _ => {}
    }
    true
}

pub fn evaluate(allowlist: &Allowlist, findings: &[Finding]) -> Result<CheckOutcome> {
    let mut outcome = CheckOutcome::default();
    outcome.findings = findings.to_vec();

    // Validate entries.
    for (idx, entry) in allowlist.entries.iter().enumerate() {
        if entry.path.is_empty()
            || entry.family.is_empty()
            || entry.owner.is_empty()
            || entry.classification.is_empty()
            || entry.explanation.is_empty()
            || entry.selector.kind.is_empty()
            || entry.selector.callee.is_empty()
        {
            outcome.schema_errors.push(format!(
                "entry {idx}: path/family/owner/classification/explanation/selector.kind/selector.callee must be non-empty"
            ));
        }
        if Family::parse(&entry.family).is_none() {
            outcome.schema_errors.push(format!("entry {idx}: unknown family `{}`", entry.family));
        }
        if let Some(expires) = entry.expires.as_deref() {
            match parse_iso_date(expires) {
                Ok(date) => {
                    if is_expired(&date) {
                        outcome.expired_entries.push((idx, expires.to_string()));
                    }
                }
                Err(e) => outcome
                    .schema_errors
                    .push(format!("entry {idx}: invalid expires `{expires}`: {e}")),
            }
        }
    }

    let mut used: Vec<bool> = vec![false; allowlist.entries.len()];

    for finding in findings {
        let mut matched_idx: Option<usize> = None;
        for (idx, entry) in allowlist.entries.iter().enumerate() {
            if entry_matches(entry, finding) {
                matched_idx = Some(idx);
                break;
            }
        }
        if let Some(idx) = matched_idx {
            used[idx] = true;
            outcome.matched += 1;
            // Drift: line/col differ from last_seen.
            if let Some(ls) = &allowlist.entries[idx].last_seen
                && (ls.line != finding.line || ls.column != finding.column)
            {
                outcome.drift_hints.push(format!(
                    "{}: {}::{} moved from {}:{} to {}:{}",
                    finding.path.display(),
                    finding.selector.container,
                    finding.selector.callee,
                    ls.line,
                    ls.column,
                    finding.line,
                    finding.column
                ));
            }
        } else {
            outcome.unallowlisted.push(finding.clone());
        }
    }

    for (idx, was_used) in used.iter().enumerate() {
        if allowlist.entries[idx].retired {
            continue;
        }
        if !*was_used {
            outcome.unused_entries.push(idx);
        }
    }

    Ok(outcome)
}

fn parse_iso_date(s: &str) -> Result<(i32, u32, u32)> {
    let mut parts = s.split('-');
    let y = parts.next().ok_or_else(|| anyhow!("missing year"))?.parse::<i32>()?;
    let m = parts.next().ok_or_else(|| anyhow!("missing month"))?.parse::<u32>()?;
    let d = parts.next().ok_or_else(|| anyhow!("missing day"))?.parse::<u32>()?;
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        bail!("out-of-range");
    }
    Ok((y, m, d))
}

fn is_expired(date: &(i32, u32, u32)) -> bool {
    let now = chrono::Utc::now().date_naive();
    let (y, m, d) = *date;
    let cmp = (
        now.format("%Y").to_string().parse::<i32>().unwrap_or(0),
        now.format("%m").to_string().parse::<u32>().unwrap_or(0),
        now.format("%d").to_string().parse::<u32>().unwrap_or(0),
    );
    cmp > (y, m, d)
}

/// Scan all tracked Rust files and return aggregated findings.
pub fn scan_workspace(repo_root: &Path) -> Result<(usize, Vec<Finding>, Vec<String>)> {
    let files = super::tracked_files(repo_root)?;
    let mut findings = Vec::new();
    let mut errors = Vec::new();
    let mut scanned = 0usize;
    for rel in files {
        if rel.extension().and_then(|s| s.to_str()) != Some("rs") {
            continue;
        }
        let abs = repo_root.join(&rel);
        let source = match std::fs::read_to_string(&abs) {
            Ok(s) => s,
            Err(e) => {
                errors.push(format!("read {}: {e}", rel.display()));
                continue;
            }
        };
        scanned += 1;
        match scan_file(&rel, &source) {
            Ok(mut fs) => findings.append(&mut fs),
            Err(e) => errors.push(format!("{e}")),
        }
    }
    Ok((scanned, findings, errors))
}

pub fn run_check(repo_root: &Path, strict: bool) -> Result<CheckOutcome> {
    let allowlist = load_allowlist(repo_root)?;
    let (scanned, findings, parse_errors) = scan_workspace(repo_root)?;
    let mut outcome = evaluate(&allowlist, &findings)?;
    outcome.files_scanned = scanned;
    for e in parse_errors {
        outcome.drift_hints.push(format!("parse-error: {e}"));
    }
    write_reports(repo_root, &allowlist, &outcome)?;
    let severity = if outcome.has_failures(strict) {
        "ERROR"
    } else if !outcome.unallowlisted.is_empty()
        || !outcome.unused_entries.is_empty()
        || !outcome.schema_errors.is_empty()
    {
        "WARN"
    } else {
        "OK"
    };
    println!(
        "no-panic-family: {severity} | scanned={}, findings={}, matched={}, unallowlisted={}, unused={}, expired={}, schema_errors={}",
        outcome.files_scanned,
        outcome.findings.len(),
        outcome.matched,
        outcome.unallowlisted.len(),
        outcome.unused_entries.len(),
        outcome.expired_entries.len(),
        outcome.schema_errors.len(),
    );
    Ok(outcome)
}

/// Write a review-only proposal allowlist under `target/bitnet/reports/`.
pub fn run_propose(repo_root: &Path) -> Result<PathBuf> {
    let (_scanned, findings, _errors) = scan_workspace(repo_root)?;
    let entries: Vec<Entry> = findings
        .into_iter()
        .map(|f| Entry {
            path: f.path.to_string_lossy().into_owned(),
            family: f.family.name().to_string(),
            classification: "review".to_string(),
            owner: "TODO".to_string(),
            explanation: "AUTO-PROPOSED — replace with a real reason or migrate the call site."
                .to_string(),
            selector: f.selector,
            last_seen: Some(LastSeen { line: f.line, column: f.column }),
            expires: Some("2026-07-01".to_string()),
            retired: false,
        })
        .collect();
    let proposal = Allowlist { schema_version: SCHEMA_VERSION.to_string(), entries };
    let dir = super::report_dir(repo_root);
    std::fs::create_dir_all(&dir)?;
    let path = dir.join("no-panic-proposed-allowlist.toml");
    std::fs::write(&path, toml::to_string_pretty(&proposal)?)?;
    println!("no-panic propose: wrote {}", path.display());
    Ok(path)
}

fn write_reports(repo_root: &Path, allowlist: &Allowlist, outcome: &CheckOutcome) -> Result<()> {
    let dir = super::report_dir(repo_root);
    std::fs::create_dir_all(&dir)?;

    // Per-family + per-crate counts.
    let mut family_counts: HashMap<&'static str, usize> = HashMap::new();
    let mut crate_counts: HashMap<String, usize> = HashMap::new();
    for f in &outcome.findings {
        *family_counts.entry(f.family.name()).or_default() += 1;
        let crate_name = top_crate(&f.path);
        *crate_counts.entry(crate_name).or_default() += 1;
    }

    let json_path = dir.join("no-panic.json");
    let json = serde_json::json!({
        "report": "no-panic-family",
        "schema_version": 1,
        "files_scanned": outcome.files_scanned,
        "findings": outcome.findings.len(),
        "matched": outcome.matched,
        "unallowlisted": outcome.unallowlisted.len(),
        "unused_entries": outcome.unused_entries.len(),
        "expired_entries": outcome.expired_entries.len(),
        "schema_errors": outcome.schema_errors,
        "drift_hints": outcome.drift_hints,
        "family_counts": family_counts,
        "crate_counts": crate_counts,
    });
    std::fs::write(&json_path, serde_json::to_string_pretty(&json)?)?;

    let mut md = String::new();
    md.push_str("# No-panic family report\n\n");
    md.push_str(&format!(
        "- files scanned: **{}**\n- findings: **{}**\n- matched: **{}**\n- unallowlisted: **{}**\n- unused entries: **{}**\n- expired: **{}**\n- schema errors: **{}**\n- drift hints: **{}**\n",
        outcome.files_scanned,
        outcome.findings.len(),
        outcome.matched,
        outcome.unallowlisted.len(),
        outcome.unused_entries.len(),
        outcome.expired_entries.len(),
        outcome.schema_errors.len(),
        outcome.drift_hints.len(),
    ));

    md.push_str("\n## Findings by family\n\n");
    let mut family_pairs: Vec<_> = family_counts.iter().collect();
    family_pairs.sort_by(|a, b| b.1.cmp(a.1));
    for (k, v) in family_pairs {
        md.push_str(&format!("- `{k}`: {v}\n"));
    }

    md.push_str("\n## Findings by crate (top 30)\n\n");
    let mut crate_pairs: Vec<_> = crate_counts.iter().collect();
    crate_pairs.sort_by(|a, b| b.1.cmp(a.1));
    for (k, v) in crate_pairs.iter().take(30) {
        md.push_str(&format!("- `{k}`: {v}\n"));
    }

    if !outcome.unallowlisted.is_empty() {
        md.push_str("\n## Unallowlisted (sample of 20)\n\n");
        for f in outcome.unallowlisted.iter().take(20) {
            md.push_str(&format!(
                "- `{}` :{}:{} `{}` in `{}::{}`\n",
                f.path.display(),
                f.line,
                f.column,
                f.family.name(),
                f.selector.container,
                f.selector.callee
            ));
        }
    }

    if !outcome.expired_entries.is_empty() {
        md.push_str("\n## Expired entries\n\n");
        for (idx, exp) in &outcome.expired_entries {
            let entry = &allowlist.entries[*idx];
            md.push_str(&format!(
                "- `{}` family={} owner={} (expired {})\n",
                entry.path, entry.family, entry.owner, exp
            ));
        }
    }

    if !outcome.schema_errors.is_empty() {
        md.push_str("\n## Schema errors\n\n");
        for e in &outcome.schema_errors {
            md.push_str(&format!("- {e}\n"));
        }
    }

    if !outcome.drift_hints.is_empty() {
        md.push_str("\n## Drift hints\n\n");
        for d in outcome.drift_hints.iter().take(50) {
            md.push_str(&format!("- {d}\n"));
        }
    }

    std::fs::write(dir.join("no-panic.md"), md)?;
    Ok(())
}

fn top_crate(path: &Path) -> String {
    let s = path.to_string_lossy();
    if let Some(rest) = s.strip_prefix("crates/") {
        if let Some(end) = rest.find('/') {
            return format!("crates/{}", &rest[..end]);
        }
    }
    if let Some(end) = s.find('/') {
        return s[..end].to_string();
    }
    s.into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scan(src: &str) -> Vec<Finding> {
        scan_file(Path::new("test.rs"), src).unwrap()
    }

    #[test]
    fn detects_unwrap_method_call() {
        let src = r#"
            fn outer() {
                let x: Option<i32> = Some(1);
                let _ = x.unwrap();
            }
        "#;
        let f = scan(src);
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].family, Family::Unwrap);
        assert_eq!(f[0].selector.kind, "method_call");
        assert_eq!(f[0].selector.callee, "unwrap");
        assert_eq!(f[0].selector.container, "outer");
    }

    #[test]
    fn detects_panic_macro_in_stmt() {
        let src = r#"
            fn boom() {
                panic!("nope");
            }
        "#;
        let f = scan(src);
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].family, Family::PanicMacro);
        assert_eq!(f[0].selector.kind, "macro_call");
    }

    #[test]
    fn detects_todo_unimplemented_unreachable() {
        let src = r#"
            fn a() { todo!(); }
            fn b() { unimplemented!(); }
            fn c() { unreachable!(); }
        "#;
        let f = scan(src);
        let names: Vec<_> = f.iter().map(|x| x.family.name()).collect();
        assert!(names.contains(&"todo"));
        assert!(names.contains(&"unimplemented"));
        assert!(names.contains(&"unreachable"));
    }

    #[test]
    fn ignores_method_named_unwrap_or() {
        let src = r#"
            fn outer() -> i32 {
                let x: Option<i32> = Some(1);
                x.unwrap_or(0)
            }
        "#;
        let f = scan(src);
        assert!(f.is_empty());
    }

    #[test]
    fn matching_uses_selector_not_line() {
        let entry = Entry {
            path: "test.rs".to_string(),
            family: "unwrap".to_string(),
            classification: "test_helper".to_string(),
            owner: "tests".to_string(),
            explanation: "x".to_string(),
            selector: Selector {
                kind: "method_call".to_string(),
                container: "outer".to_string(),
                callee: "unwrap".to_string(),
                receiver_fingerprint: None,
            },
            last_seen: Some(LastSeen { line: 999, column: 999 }),
            expires: None,
            retired: false,
        };
        let finding = Finding {
            path: PathBuf::from("test.rs"),
            family: Family::Unwrap,
            selector: Selector {
                kind: "method_call".to_string(),
                container: "outer".to_string(),
                callee: "unwrap".to_string(),
                receiver_fingerprint: Some("foo()".to_string()),
            },
            line: 4,
            column: 26,
        };
        assert!(entry_matches(&entry, &finding));
    }

    #[test]
    fn unknown_family_is_schema_error() {
        let allowlist = Allowlist {
            schema_version: SCHEMA_VERSION.to_string(),
            entries: vec![Entry {
                path: "x.rs".to_string(),
                family: "bogus".to_string(),
                classification: "x".to_string(),
                owner: "x".to_string(),
                explanation: "x".to_string(),
                selector: Selector {
                    kind: "method_call".to_string(),
                    container: "x".to_string(),
                    callee: "x".to_string(),
                    receiver_fingerprint: None,
                },
                last_seen: None,
                expires: None,
                retired: false,
            }],
        };
        let outcome = evaluate(&allowlist, &[]).unwrap();
        assert!(outcome.schema_errors.iter().any(|e| e.contains("bogus")));
    }
}
