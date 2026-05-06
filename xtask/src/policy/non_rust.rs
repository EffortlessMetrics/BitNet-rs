//! Non-Rust file allowlist enforcement.
//!
//! BitNet-rs governs non-Rust *programming and declarative implementation*
//! files via `policy/non-rust-allowlist.toml`. Each entry is a structured
//! receipt: `path` or `glob` identity, plus `kind`, `owner`, `surface`,
//! `classification`, `reason`, optional `covered_by`, optional `expires`,
//! optional `retired`.
//!
//! The default classification of files (Rust files, Markdown, snapshots,
//! generated assets, etc.) is *out of scope* for this checker — it only
//! validates the surfaces that warrant explicit governance. See
//! `is_non_rust_implementation`.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use globset::{Glob, GlobSetBuilder};
use serde::{Deserialize, Serialize};

use super::report::{Report, ReportSeverity};

const ALLOWLIST_PATH: &str = "policy/non-rust-allowlist.toml";
const SCHEMA_VERSION: &str = "1.0";

#[derive(Debug, Deserialize, Serialize)]
pub struct Allowlist {
    pub schema_version: String,
    #[serde(default, rename = "allow")]
    pub entries: Vec<Entry>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Entry {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub glob: Option<String>,
    pub kind: String,
    pub owner: String,
    pub surface: String,
    pub classification: String,
    pub reason: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub covered_by: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires: Option<String>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub retired: bool,
}

fn is_false(b: &bool) -> bool {
    !*b
}

#[derive(Debug, Default)]
pub struct CheckOutcome {
    pub total_tracked: usize,
    pub in_scope: usize,
    pub matched: usize,
    pub uncovered: Vec<PathBuf>,
    pub unused_entries: Vec<usize>,
    pub expired_entries: Vec<(usize, String)>,
    pub schema_errors: Vec<String>,
}

impl CheckOutcome {
    pub fn has_failures(&self, strict: bool) -> bool {
        if !self.schema_errors.is_empty() || !self.expired_entries.is_empty() {
            return true;
        }
        if strict && (!self.uncovered.is_empty() || !self.unused_entries.is_empty()) {
            return true;
        }
        false
    }
}

/// Load the allowlist from `repo_root/policy/non-rust-allowlist.toml`.
pub fn load_allowlist(repo_root: &Path) -> Result<Allowlist> {
    let path = repo_root.join(ALLOWLIST_PATH);
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let list: Allowlist = toml::from_str(&raw)
        .with_context(|| format!("failed to parse {}", path.display()))?;
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

/// Returns true for non-Rust files that are governed by the allowlist.
///
/// Out of scope here:
///   - Rust source (`*.rs`)
///   - Documentation (`*.md`, `*.txt` outside fixtures)
///   - Snapshots / fixtures with Cargo-standard extensions
///   - JSON / SVG / PNG / JPG / MP4 (assets and metadata)
///   - Generated lock files (`Cargo.lock`)
///   - Cargo manifests (`Cargo.toml` is a Rust standard surface)
pub fn is_non_rust_implementation(path: &Path) -> bool {
    let s = path.to_string_lossy();
    if s.starts_with("target/") || s.starts_with(".git/") {
        return false;
    }
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();

    // Cargo standard files are not governed.
    if name == "Cargo.toml" || name == "Cargo.lock" {
        return false;
    }
    // Programming languages we care about.
    const PROGRAMMING_EXTS: &[&str] = &[
        "py", "sh", "ps1", "bat", "rb", "cs", "ts", "js", "nix", "cl", "metal", "cu", "hip",
        "comp", "wgsl", "c", "cpp", "cc", "h", "hpp", "pyi",
    ];
    if PROGRAMMING_EXTS.iter().any(|e| *e == ext) {
        return true;
    }
    // GitHub Actions workflow files.
    if s.starts_with(".github/workflows/") && (ext == "yml" || ext == "yaml") {
        return true;
    }
    // Dockerfiles (named or `Dockerfile.*`).
    if name == "Dockerfile" || name.starts_with("Dockerfile.") {
        return true;
    }
    false
}

/// Match files against the allowlist. Returns the outcome plus per-entry
/// usage counts.
pub fn evaluate(allowlist: &Allowlist, files: &[PathBuf]) -> Result<CheckOutcome> {
    let mut outcome = CheckOutcome::default();
    outcome.total_tracked = files.len();

    let mut entry_paths: Vec<Option<PathBuf>> = Vec::with_capacity(allowlist.entries.len());
    let mut globs = GlobSetBuilder::new();
    let mut glob_owners: Vec<usize> = Vec::new();

    for (idx, entry) in allowlist.entries.iter().enumerate() {
        // Validate entry.
        if entry.path.is_none() && entry.glob.is_none() {
            outcome
                .schema_errors
                .push(format!("entry {idx}: requires `path` or `glob`"));
            entry_paths.push(None);
            continue;
        }
        if entry.path.is_some() && entry.glob.is_some() {
            outcome
                .schema_errors
                .push(format!("entry {idx}: must not set both `path` and `glob`"));
            entry_paths.push(None);
            continue;
        }
        if entry.kind.is_empty()
            || entry.owner.is_empty()
            || entry.surface.is_empty()
            || entry.classification.is_empty()
            || entry.reason.is_empty()
        {
            outcome.schema_errors.push(format!(
                "entry {idx}: kind/owner/surface/classification/reason must be non-empty"
            ));
        }
        // Production/test/tooling surfaces require covered_by.
        let needs_coverage = matches!(
            entry.classification.as_str(),
            "production" | "test" | "tooling"
        );
        if needs_coverage && entry.covered_by.is_empty() && !entry.retired {
            outcome.schema_errors.push(format!(
                "entry {idx}: classification `{}` requires non-empty covered_by",
                entry.classification
            ));
        }
        // Expiry check.
        if let Some(expires) = entry.expires.as_deref() {
            match parse_iso_date(expires) {
                Ok(date) => {
                    if is_expired(&date) {
                        outcome
                            .expired_entries
                            .push((idx, expires.to_string()));
                    }
                }
                Err(e) => outcome
                    .schema_errors
                    .push(format!("entry {idx}: invalid expires `{expires}`: {e}")),
            }
        }

        if let Some(p) = entry.path.as_deref() {
            entry_paths.push(Some(PathBuf::from(p)));
        } else if let Some(g) = entry.glob.as_deref() {
            entry_paths.push(None);
            match Glob::new(g) {
                Ok(glob) => {
                    globs.add(glob);
                    glob_owners.push(idx);
                }
                Err(e) => outcome
                    .schema_errors
                    .push(format!("entry {idx}: invalid glob `{g}`: {e}")),
            }
        }
    }

    let glob_set = globs.build().map_err(|e| anyhow!("glob build: {e}"))?;
    let mut used: Vec<bool> = vec![false; allowlist.entries.len()];

    for path in files {
        if !is_non_rust_implementation(path) {
            continue;
        }
        outcome.in_scope += 1;
        let path_str = path.to_string_lossy();
        let mut matched = false;
        // Path identity matches.
        for (idx, entry_path) in entry_paths.iter().enumerate() {
            if let Some(ep) = entry_path
                && ep == path
            {
                if allowlist.entries[idx].retired {
                    continue;
                }
                used[idx] = true;
                matched = true;
                break;
            }
        }
        if !matched {
            for glob_idx in glob_set.matches(path_str.as_ref()) {
                let entry_idx = glob_owners[glob_idx];
                if allowlist.entries[entry_idx].retired {
                    continue;
                }
                used[entry_idx] = true;
                matched = true;
                break;
            }
        }
        if matched {
            outcome.matched += 1;
        } else {
            outcome.uncovered.push(path.clone());
        }
    }

    // Identify unused active entries.
    for (idx, was_used) in used.iter().enumerate() {
        let entry = &allowlist.entries[idx];
        if entry.retired {
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
    let y = parts
        .next()
        .ok_or_else(|| anyhow!("missing year"))?
        .parse::<i32>()?;
    let m = parts
        .next()
        .ok_or_else(|| anyhow!("missing month"))?
        .parse::<u32>()?;
    let d = parts
        .next()
        .ok_or_else(|| anyhow!("missing day"))?
        .parse::<u32>()?;
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        bail!("out-of-range");
    }
    Ok((y, m, d))
}

fn is_expired(date: &(i32, u32, u32)) -> bool {
    let now = chrono::Utc::now().date_naive();
    let (y, m, d) = *date;
    let cmp = (now.format("%Y").to_string().parse::<i32>().unwrap_or(0),
               now.format("%m").to_string().parse::<u32>().unwrap_or(0),
               now.format("%d").to_string().parse::<u32>().unwrap_or(0));
    cmp > (y, m, d)
}

/// Run the check and write report files.
pub fn run_check(repo_root: &Path, strict: bool) -> Result<CheckOutcome> {
    let allowlist = load_allowlist(repo_root)?;
    let files = super::tracked_files(repo_root)?;
    let outcome = evaluate(&allowlist, &files)?;
    write_reports(repo_root, &allowlist, &outcome)?;
    let severity = if outcome.has_failures(strict) {
        ReportSeverity::Error
    } else if !outcome.uncovered.is_empty()
        || !outcome.unused_entries.is_empty()
        || !outcome.schema_errors.is_empty()
    {
        ReportSeverity::Warn
    } else {
        ReportSeverity::Ok
    };
    println!(
        "non-rust file policy: {severity:?} | tracked={}, in_scope={}, matched={}, uncovered={}, unused={}, expired={}",
        outcome.total_tracked,
        outcome.in_scope,
        outcome.matched,
        outcome.uncovered.len(),
        outcome.unused_entries.len(),
        outcome.expired_entries.len()
    );
    Ok(outcome)
}

fn write_reports(repo_root: &Path, allowlist: &Allowlist, outcome: &CheckOutcome) -> Result<()> {
    let dir = super::report_dir(repo_root);
    std::fs::create_dir_all(&dir).with_context(|| format!("mkdir {}", dir.display()))?;

    // JSON
    let json_path = dir.join("file-policy.json");
    let json_payload = Report {
        report: "non-rust-file-policy",
        schema_version: 1,
        total_tracked: outcome.total_tracked,
        in_scope: outcome.in_scope,
        matched: outcome.matched,
        uncovered: outcome
            .uncovered
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect(),
        unused_entries: outcome
            .unused_entries
            .iter()
            .map(|idx| describe_entry(&allowlist.entries[*idx]))
            .collect(),
        expired_entries: outcome
            .expired_entries
            .iter()
            .map(|(idx, exp)| format!("{}: expired {}", describe_entry(&allowlist.entries[*idx]), exp))
            .collect(),
        schema_errors: outcome.schema_errors.clone(),
    };
    std::fs::write(&json_path, serde_json::to_string_pretty(&json_payload)?)
        .with_context(|| format!("write {}", json_path.display()))?;

    // Markdown
    let md_path = dir.join("file-policy.md");
    let mut md = String::new();
    md.push_str("# Non-Rust file policy report\n\n");
    md.push_str(&format!(
        "- tracked files: **{}**\n- in-scope (non-Rust impl): **{}**\n- matched: **{}**\n",
        outcome.total_tracked, outcome.in_scope, outcome.matched
    ));
    md.push_str(&format!(
        "- uncovered: **{}**\n- unused active entries: **{}**\n- expired entries: **{}**\n- schema errors: **{}**\n",
        outcome.uncovered.len(),
        outcome.unused_entries.len(),
        outcome.expired_entries.len(),
        outcome.schema_errors.len()
    ));
    if !outcome.uncovered.is_empty() {
        md.push_str("\n## Uncovered files\n\n");
        for p in &outcome.uncovered {
            md.push_str(&format!("- `{}`\n", p.display()));
        }
    }
    if !outcome.unused_entries.is_empty() {
        md.push_str("\n## Unused entries\n\n");
        for idx in &outcome.unused_entries {
            md.push_str(&format!("- {}\n", describe_entry(&allowlist.entries[*idx])));
        }
    }
    if !outcome.expired_entries.is_empty() {
        md.push_str("\n## Expired entries\n\n");
        for (idx, exp) in &outcome.expired_entries {
            md.push_str(&format!(
                "- {} (expired {})\n",
                describe_entry(&allowlist.entries[*idx]),
                exp
            ));
        }
    }
    if !outcome.schema_errors.is_empty() {
        md.push_str("\n## Schema errors\n\n");
        for e in &outcome.schema_errors {
            md.push_str(&format!("- {e}\n"));
        }
    }
    std::fs::write(&md_path, md).with_context(|| format!("write {}", md_path.display()))?;

    Ok(())
}

fn describe_entry(entry: &Entry) -> String {
    let id = match (&entry.path, &entry.glob) {
        (Some(p), _) => format!("path={p}"),
        (_, Some(g)) => format!("glob={g}"),
        _ => "<no-identity>".to_string(),
    };
    format!("[{}] owner={} kind={} reason=\"{}\"", id, entry.owner, entry.kind, entry.reason)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_python_as_in_scope() {
        assert!(is_non_rust_implementation(Path::new("crates/bitnet-py/src/lib.py")));
    }

    #[test]
    fn classifies_rust_as_out_of_scope() {
        assert!(!is_non_rust_implementation(Path::new("crates/foo/src/lib.rs")));
    }

    #[test]
    fn classifies_cargo_toml_as_out_of_scope() {
        assert!(!is_non_rust_implementation(Path::new("crates/foo/Cargo.toml")));
    }

    #[test]
    fn classifies_workflow_yml_as_in_scope() {
        assert!(is_non_rust_implementation(Path::new(".github/workflows/ci.yml")));
    }

    #[test]
    fn classifies_root_dockerfile_as_in_scope() {
        assert!(is_non_rust_implementation(Path::new("Dockerfile")));
    }

    #[test]
    fn rejects_entry_without_path_or_glob() {
        let raw = r#"
schema_version = "1.0"

[[allow]]
kind = "x"
owner = "y"
surface = "ci"
classification = "config"
reason = "z"
"#;
        let parsed: Allowlist = toml::from_str(raw).unwrap();
        let outcome = evaluate(&parsed, &[]).unwrap();
        assert!(!outcome.schema_errors.is_empty());
    }

    #[test]
    fn matches_glob_entry() {
        let raw = r#"
schema_version = "1.0"

[[allow]]
glob = "**/*.py"
kind = "python_binding"
owner = "bindings/python"
surface = "language-binding"
classification = "production"
reason = "Python binding."
covered_by = ["cargo check"]
"#;
        let parsed: Allowlist = toml::from_str(raw).unwrap();
        let files = vec![PathBuf::from("crates/bitnet-py/src/lib.py")];
        let outcome = evaluate(&parsed, &files).unwrap();
        assert_eq!(outcome.matched, 1);
        assert!(outcome.uncovered.is_empty());
        assert!(outcome.unused_entries.is_empty());
    }
}
