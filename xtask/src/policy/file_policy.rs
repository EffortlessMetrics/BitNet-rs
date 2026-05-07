//! Non-Rust file allowlist checker (`policy/non-rust-allowlist.toml`).
//!
//! Iterates over tracked files (via `git ls-files` if a repo, otherwise
//! the working tree) and checks each non-Rust file against the
//! allowlist's globs. Files that match no allowlist entry are reported
//! as findings.
//!
//! Rust source under `src/`, `tests/`, `benches/`, `examples/`, `crates/.../src/`,
//! `xtask/.../src/`, etc. is implicitly allowed and skipped — Rust is
//! the default implementation surface.

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Deserialize)]
struct Allowlist {
    #[serde(default, rename = "allow")]
    entries: Vec<Entry>,
}

#[derive(Debug, Deserialize, Clone)]
struct Entry {
    glob: String,
    #[serde(default)]
    kind: String,
    owner: String,
    #[serde(default)]
    surface: String,
    #[serde(default)]
    classification: String,
    #[serde(default)]
    reason: String,
    #[serde(default)]
    covered_by: Vec<String>,
}

#[derive(Debug, Default)]
pub struct Report {
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub allow_count: usize,
    pub file_count: usize,
}

pub fn run(allowlist_path: PathBuf, report_dir: PathBuf, fail_on_error: bool) -> Result<()> {
    let report = check(&allowlist_path, &report_dir)?;
    println!(
        "file-policy: {} files scanned, {} allowlist entries, {} findings",
        report.file_count,
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
        bail!("file-policy check failed: {} errors", report.errors.len());
    }
    Ok(())
}

fn check(allowlist_path: &Path, report_dir: &Path) -> Result<Report> {
    let mut report = Report::default();

    let text = fs::read_to_string(allowlist_path)
        .with_context(|| format!("reading {}", allowlist_path.display()))?;
    let allowlist: Allowlist =
        toml::from_str(&text).with_context(|| format!("parsing {}", allowlist_path.display()))?;
    report.allow_count = allowlist.entries.len();

    let files = list_files()?;
    report.file_count = files.len();

    let entries = &allowlist.entries;
    for f in &files {
        if is_implicit_rust(f) {
            continue;
        }
        if !entries.iter().any(|e| glob_match(&e.glob, f)) {
            report.errors.push(format!("non-Rust file `{f}` not covered by allowlist"));
        }
    }

    fs::create_dir_all(report_dir).with_context(|| format!("creating {}", report_dir.display()))?;
    let json = serde_json::json!({
        "schema_version": 1,
        "errors": report.errors,
        "warnings": report.warnings,
        "allow_count": report.allow_count,
        "file_count": report.file_count,
    });
    fs::write(report_dir.join("file-policy.json"), serde_json::to_string_pretty(&json)?)?;
    Ok(report)
}

fn list_files() -> Result<Vec<String>> {
    // Use `git ls-files` for tracked-file fidelity. If git is unavailable
    // (e.g. running inside an extracted tarball), fall back to walkdir
    // with a small denylist.
    let output = Command::new("git").args(["ls-files"]).output();
    if let Ok(o) = output
        && o.status.success()
    {
        let s = String::from_utf8_lossy(&o.stdout).to_string();
        return Ok(s.lines().map(str::to_string).collect());
    }
    let mut out = Vec::new();
    for entry in walkdir::WalkDir::new(".").into_iter().filter_map(std::result::Result::ok) {
        let p = entry.path();
        if p.is_file() {
            let s = p.strip_prefix(".").unwrap_or(p).to_string_lossy().to_string();
            if !s.contains("target/") && !s.contains(".git/") {
                out.push(s);
            }
        }
    }
    Ok(out)
}

fn is_implicit_rust(path: &str) -> bool {
    path.ends_with(".rs") || path == "Cargo.toml" || path.ends_with("/Cargo.toml")
}

/// Minimal glob support: `*` matches any segment chars except `/`,
/// `**` matches any segment sequence including `/`, and a trailing
/// `/` requires the path to start with the prefix.
fn glob_match(pattern: &str, path: &str) -> bool {
    let p_bytes = pattern.as_bytes();
    let s_bytes = path.as_bytes();
    matches(p_bytes, 0, s_bytes, 0)
}

fn matches(p: &[u8], pi: usize, s: &[u8], si: usize) -> bool {
    let mut pi = pi;
    let mut si = si;
    while pi < p.len() {
        let c = p[pi];
        if c == b'*' {
            // Detect `**`
            let double = pi + 1 < p.len() && p[pi + 1] == b'*';
            if double {
                let next = pi + 2;
                if next >= p.len() {
                    return true;
                }
                // Skip any `/` immediately following `**/`.
                let after = if p.get(next) == Some(&b'/') { next + 1 } else { next };
                let mut j = si;
                loop {
                    if matches(p, after, s, j) {
                        return true;
                    }
                    if j >= s.len() {
                        return false;
                    }
                    j += 1;
                }
            } else {
                let next = pi + 1;
                if next >= p.len() {
                    // single-star matches anything that does not contain `/`
                    return !s[si..].contains(&b'/');
                }
                let mut j = si;
                while j <= s.len() {
                    if !s[si..j].contains(&b'/') && matches(p, next, s, j) {
                        return true;
                    }
                    j += 1;
                }
                return false;
            }
        }
        if c == b'{' {
            // Brace expansion: `{a,b,c}`
            let mut close = pi + 1;
            while close < p.len() && p[close] != b'}' {
                close += 1;
            }
            if close >= p.len() {
                return false;
            }
            let alts = &p[pi + 1..close];
            let mut start = 0;
            for k in 0..=alts.len() {
                if k == alts.len() || alts[k] == b',' {
                    let alt = &alts[start..k];
                    let mut combined = Vec::with_capacity(alt.len() + p.len() - close - 1);
                    combined.extend_from_slice(alt);
                    combined.extend_from_slice(&p[close + 1..]);
                    if matches(&combined, 0, s, si) {
                        return true;
                    }
                    start = k + 1;
                }
            }
            return false;
        }
        if si >= s.len() {
            return false;
        }
        if c != s[si] {
            return false;
        }
        pi += 1;
        si += 1;
    }
    si == s.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glob_basic() {
        assert!(glob_match("docs/**", "docs/foo/bar.md"));
        assert!(glob_match(".github/workflows/*.yml", ".github/workflows/ci.yml"));
        assert!(!glob_match(".github/workflows/*.yml", ".github/workflows/sub/ci.yml"));
        assert!(glob_match("crates/foo/**/*.{h,c}", "crates/foo/bar/baz.h"));
        assert!(glob_match("crates/foo/**/*.{h,c}", "crates/foo/bar/baz.c"));
        assert!(!glob_match("crates/foo/**/*.{h,c}", "crates/foo/bar/baz.rs"));
    }

    #[test]
    fn implicit_rust_skipped() {
        assert!(is_implicit_rust("crates/x/src/main.rs"));
        assert!(is_implicit_rust("Cargo.toml"));
        assert!(is_implicit_rust("crates/x/Cargo.toml"));
        assert!(!is_implicit_rust("README.md"));
    }
}
