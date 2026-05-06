//! `xtask ci plan` — Rust-native PR planner / LEM forecaster.
//!
//! Replaces the inline Python planner that previously lived in
//! `.github/workflows/pr-plan.yml`. The Rust version is unit-testable with
//! fixtures and emits a stable `ci-plan.json` schema for downstream tooling
//! (budget warnings, label-aware gating, future learned-budget calibration).
//!
//! Conventions:
//!   - LEM = Linux-equivalent minutes. macOS treated as 10x, GPU Docker as 6x.
//!   - Estimates are rough heuristics. Actual cost is recorded in the GitHub
//!     Actions metrics UI; this module exists to give a per-PR forecast.
//!
//! See `docs/ci/cost-and-verification-policy.md` for rationale.

use anyhow::{Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Touched-area buckets considered by the planner.
///
/// Matched against the list of changed files via the regexes in [`area_patterns`].
const AREAS: &[&str] = &[
    "docs",
    "tracking",
    "workflow",
    "rust_core",
    "rust_production",
    "ripr",
    "gpu",
    "ffi",
    "tokenizer",
    "bdd",
    "fuzz",
    "manifest",
];

fn area_patterns() -> Vec<(&'static str, Vec<&'static str>)> {
    vec![
        (
            "docs",
            vec![
                r"^docs/",
                r"\.md$",
                r"^README",
                r"^CHANGELOG",
                r"^CONTRIBUTING",
                r"^SECURITY",
                r"^COMPATIBILITY",
                r"^THIRD_PARTY",
                r"^CLAUDE\.md$",
            ],
        ),
        ("tracking", vec![r"^\.codex/campaigns/", r"^docs/tracking/"]),
        ("workflow", vec![r"^\.github/workflows/", r"^\.github/actions/"]),
        (
            "rust_core",
            vec![
                r"^crates/",
                r"^tests/",
                r"^xtask/",
                r"^Cargo\.(toml|lock)$",
                r"^rust-toolchain\.toml$",
            ],
        ),
        (
            // ripr triggers on production Rust (crates/*/src, xtask/src,
            // crossval/src) and on its own config files. Test-only diffs do
            // not invoke ripr by default.
            "rust_production",
            vec![r"^crates/[^/]+/src/", r"^crossval/src/", r"^xtask/src/"],
        ),
        ("ripr", vec![r"^ripr\.toml$", r"^policy/ripr-"]),
        (
            "gpu",
            vec![
                r"^crates/bitnet-kernels/",
                r"^crates/bitnet-gpu-hal/",
                r"^crates/bitnet-device-probe/",
                r"^crates/bitnet-device-config-core/",
                r"^crates/bitnet-inference/",
                r"^crates/bitnet-metal/",
                r"^crates/bitnet-opencl/",
                r"^crates/bitnet-vulkan",
                r"^crates/bitnet-wgpu",
                r"^crates/bitnet-webgpu/",
                r"^crates/bitnet-rocm/",
                r"^crates/bitnet-intel-gpu-id/",
                r"^docker/",
            ],
        ),
        (
            "ffi",
            vec![
                r"^crates/bitnet-ffi/",
                r"^crates/bitnet-sys/",
                r"^crates/bitnet-ggml-ffi/",
                r"^crossval/",
            ],
        ),
        ("tokenizer", vec![r"^crates/bitnet-tokenizers/", r"^tests/fixtures/tokenizers/"]),
        (
            "bdd",
            vec![
                r"^crates/bitnet-bdd-",
                r"^crates/bitnet-testing-policy",
                r"^crates/bitnet-testing-scenarios",
                r"^crates/bitnet-runtime-feature-flags",
                r"^crates/bitnet-startup-contract",
                r"^crates/bitnet-feature-contract",
            ],
        ),
        ("fuzz", vec![r"^fuzz/"]),
        ("manifest", vec![r"^Cargo\.(toml|lock)$", r"^rust-toolchain\.toml$"]),
    ]
}

#[derive(Debug, Serialize)]
pub struct Lane {
    pub name: String,
    pub lem: u32,
    pub reason: String,
    pub blocking: bool,
    pub stage: &'static str,
}

#[derive(Debug, Serialize)]
pub struct Plan {
    pub posture: &'static str,
    pub touched: BTreeMap<String, bool>,
    pub labels: Vec<String>,
    pub lanes: Vec<Lane>,
    pub estimated_lem: u32,
    pub band: &'static str,
}

/// LEM banding thresholds. Hand-tunable via `policy/ci-budget.toml`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct BudgetPolicy {
    pub preferred_default_lem: u32,
    pub default_limit_lem: u32,
    pub elevated_limit_lem: u32,
    pub hard_limit_lem: u32,
}

impl Default for BudgetPolicy {
    /// Hard-coded fallback used when `policy/ci-budget.toml` is missing or
    /// malformed. Kept in sync with the values shipped in that file so the
    /// planner produces identical output in either case.
    fn default() -> Self {
        Self {
            preferred_default_lem: 25,
            default_limit_lem: 35,
            elevated_limit_lem: 75,
            hard_limit_lem: 125,
        }
    }
}

#[derive(Debug, Deserialize)]
struct BudgetFile {
    budget: BudgetPolicy,
}

/// Load `policy/ci-budget.toml` if present. Falls back to [`BudgetPolicy::default`]
/// on any failure (missing file, parse error, etc.) so the planner remains
/// usable in fresh checkouts and unit-test fixtures.
pub fn load_budget_policy(policy_dir: Option<&Path>) -> BudgetPolicy {
    let path = policy_dir.unwrap_or_else(|| Path::new("policy")).join("ci-budget.toml");
    let Ok(text) = fs::read_to_string(&path) else {
        return BudgetPolicy::default();
    };
    match toml::from_str::<BudgetFile>(&text) {
        Ok(f) => f.budget,
        Err(e) => {
            eprintln!("warning: {} parse failed ({e}); using built-in defaults", path.display());
            BudgetPolicy::default()
        }
    }
}

fn band_for(total: u32, p: &BudgetPolicy) -> &'static str {
    if total <= p.preferred_default_lem.saturating_sub(13).max(12) {
        // "pennies" band sits below ~12 LEM; we anchor at the smaller of
        // (preferred_default - 13, 12) so docs-only PRs always show pennies.
        "✅ pennies (< 12 LEM)"
    } else if total <= p.default_limit_lem {
        "✅ default budget (< 35 LEM)"
    } else if total <= p.elevated_limit_lem {
        "⚠️  elevated (35–75 LEM)"
    } else if total <= p.hard_limit_lem {
        "⚠️  high (75–125 LEM)"
    } else {
        "🚨 over hard ceiling (> 125 LEM)"
    }
}

/// Pure planner: given the changed files and labels, produce the plan.
///
/// Uses the default budget policy. Tests typically call this; production
/// callers go through [`plan_with_policy`] so `policy/ci-budget.toml` can
/// override band thresholds.
#[allow(dead_code)] // Re-exported for tests + downstream xtask consumers.
pub fn plan_for(changed: &[String], labels: &[String]) -> Plan {
    plan_with_policy(changed, labels, &BudgetPolicy::default())
}

/// Pure planner with explicit budget policy.
pub fn plan_with_policy(changed: &[String], labels: &[String], budget: &BudgetPolicy) -> Plan {
    let mut touched: BTreeMap<String, bool> =
        AREAS.iter().map(|a| ((*a).to_string(), false)).collect();

    let compiled: Vec<(&str, Vec<Regex>)> = area_patterns()
        .into_iter()
        .map(|(area, pats)| {
            let regs = pats.iter().map(|p| Regex::new(p).expect("static regex")).collect();
            (area, regs)
        })
        .collect();

    for path in changed {
        for (area, regs) in &compiled {
            if regs.iter().any(|r| r.is_match(path)) {
                touched.insert((*area).to_string(), true);
            }
        }
    }

    let mut lanes: Vec<Lane> = Vec::new();
    let label_set: std::collections::HashSet<&str> = labels.iter().map(|s| s.as_str()).collect();
    let has = |l: &str| label_set.contains(l);

    let touched_or = |area: &str| -> bool { *touched.get(area).unwrap_or(&false) };

    // CI Core triggers on rust_core (post PR A).
    if touched_or("rust_core") {
        lanes.push(Lane {
            name: "CI (Core) — build/test/clippy/docs".to_string(),
            lem: 22,
            reason: "rust_core changed".to_string(),
            blocking: true,
            stage: "required",
        });
    }
    // BDD grid: gated to bdd/grid/full-ci or main (post PR G).
    if has("bdd") || has("grid") || has("full-ci") {
        lanes.push(Lane {
            name: "BDD Grid Check".to_string(),
            lem: 4,
            reason: "bdd/grid/full-ci label".to_string(),
            blocking: false,
            stage: "label",
        });
    }
    // macOS clippy: gated to macos/full-ci (post PR F). 10x multiplier already baked in.
    if has("macos") || has("full-ci") {
        lanes.push(Lane {
            name: "Clippy (macOS ARM64)".to_string(),
            lem: 15 * 10,
            reason: "macos/full-ci label".to_string(),
            blocking: false,
            stage: "label",
        });
    }
    // Feature matrix: 3-combo PR matrix; full matrix on label.
    if touched_or("rust_core") || touched_or("manifest") {
        if has("feature-matrix") || has("full-ci") {
            lanes.push(Lane {
                name: "Feature Matrix (full ~21 jobs)".to_string(),
                lem: 70,
                reason: "feature-matrix/full-ci label".to_string(),
                blocking: false,
                stage: "label",
            });
        } else {
            lanes.push(Lane {
                name: "Feature Matrix (PR 3-combo)".to_string(),
                lem: 12,
                reason: "rust/manifest changed".to_string(),
                blocking: false,
                stage: "default",
            });
        }
    }
    // GPU CI: triggers on gpu paths (post PR B).
    if touched_or("gpu") {
        lanes.push(Lane {
            name: "GPU CI Matrix (native compile)".to_string(),
            lem: 18,
            reason: "GPU paths changed".to_string(),
            blocking: false,
            stage: "default",
        });
        if has("gpu-ci") || has("docker") || has("full-ci") {
            lanes.push(Lane {
                name: "GPU CI Matrix (Docker, ~6x)".to_string(),
                lem: 90,
                reason: "gpu-ci/docker/full-ci label".to_string(),
                blocking: false,
                stage: "label",
            });
        }
    }
    // Compatibility lanes (post PR E).
    if touched_or("rust_core") {
        lanes.push(Lane {
            name: "Compatibility (MSRV)".to_string(),
            lem: 12,
            reason: "rust_core changed".to_string(),
            blocking: false,
            stage: "default",
        });
    }
    if touched_or("ffi") || has("ffi") || has("abi") || has("full-ci") {
        lanes.push(Lane {
            name: "Compatibility (ABI/FFI)".to_string(),
            lem: 8,
            reason: "ffi area / label".to_string(),
            blocking: false,
            stage: "label",
        });
    }
    if touched_or("tokenizer") || has("tokenizer") || has("full-ci") {
        lanes.push(Lane {
            name: "Compatibility (tokenizer)".to_string(),
            lem: 6,
            reason: "tokenizer area / label".to_string(),
            blocking: false,
            stage: "label",
        });
    }
    // Property smoke (post PR D).
    if has("property-tests") || has("full-ci") {
        lanes.push(Lane {
            name: "Property Tests (smoke)".to_string(),
            lem: 4,
            reason: "property-tests/full-ci label".to_string(),
            blocking: false,
            stage: "label",
        });
    }
    // ripr static exposure (post PR J): production Rust diffs or explicit label.
    // Advisory only — does not gate merges.
    if touched_or("rust_production") || touched_or("ripr") || has("ripr") || has("full-ci") {
        lanes.push(Lane {
            name: "ripr static exposure (advisory)".to_string(),
            lem: 4,
            reason: "production Rust diff / ripr label".to_string(),
            blocking: false,
            stage: "advisory",
        });
    }
    // Always-on cheap guards.
    if !changed.is_empty() {
        lanes.push(Lane {
            name: "Guards / PR Size Guard / Markdownlint / Link Check".to_string(),
            lem: 4,
            reason: "always-on".to_string(),
            blocking: false,
            stage: "default",
        });
    }

    let total: u32 = lanes.iter().map(|l| l.lem).sum();

    let posture: &'static str = if changed.is_empty() {
        "empty"
    } else if touched_or("docs")
        && !(touched_or("rust_core")
            || touched_or("gpu")
            || touched_or("ffi")
            || touched_or("tokenizer"))
    {
        "docs-only"
    } else if touched_or("tracking")
        && !(touched_or("rust_core") || touched_or("gpu") || touched_or("ffi"))
    {
        "tracking-only"
    } else {
        "rust"
    };

    let band: &'static str = band_for(total, budget);

    let mut sorted_labels: Vec<String> = labels.to_vec();
    sorted_labels.sort();

    Plan { posture, touched, labels: sorted_labels, lanes, estimated_lem: total, band }
}

fn render_summary(plan: &Plan) -> String {
    use std::fmt::Write as _;
    let mut s = String::new();
    let _ = writeln!(s, "# PR Plan");
    let _ = writeln!(s);
    let _ = writeln!(s, "- **Posture:** {}", plan.posture);
    let touched_areas: Vec<&str> =
        plan.touched.iter().filter_map(|(a, t)| if *t { Some(a.as_str()) } else { None }).collect();
    let _ = writeln!(
        s,
        "- **Touched areas:** {}",
        if touched_areas.is_empty() { "(none)".to_string() } else { touched_areas.join(", ") }
    );
    let _ = writeln!(
        s,
        "- **Labels:** {}",
        if plan.labels.is_empty() { "(none)".to_string() } else { plan.labels.join(", ") }
    );
    let _ = writeln!(s, "- **Estimated LEM:** {}  ·  {}", plan.estimated_lem, plan.band);
    let _ = writeln!(s);
    let _ = writeln!(s, "| Lane | Estimated LEM | Reason |");
    let _ = writeln!(s, "|---|---:|---|");
    if plan.lanes.is_empty() {
        let _ = writeln!(s, "| (no lanes expected) | 0 | nothing matched |");
    } else {
        for lane in &plan.lanes {
            let _ = writeln!(s, "| {} | {} | {} |", lane.name, lane.lem, lane.reason);
        }
    }
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "> Estimates are rough heuristics derived from path globs + labels. \
         Actual minutes are recorded in the GitHub Actions metrics UI. \
         This job is advisory only and does not gate merges. \
         See [docs/ci/cost-and-verification-policy.md](../blob/main/docs/ci/cost-and-verification-policy.md)."
    );
    s
}

/// Compute changed files via `git diff --name-only base...head`.
///
/// Falls back to `HEAD~1..HEAD` if the base is missing locally (e.g. shallow
/// clones without the base branch fetched).
pub fn git_changed_files(base: Option<&str>, head: Option<&str>) -> Result<Vec<String>> {
    let head_ref = head.unwrap_or("HEAD");

    if let Some(base) = base {
        // Verify base exists.
        let exists = Command::new("git")
            .args(["cat-file", "-e", base])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if exists {
            let out = Command::new("git")
                .args(["diff", "--name-only", &format!("{base}..{head_ref}")])
                .output()
                .context("git diff failed")?;
            return Ok(parse_git_output(&out.stdout));
        }
    }

    let out = Command::new("git")
        .args(["diff", "--name-only", "HEAD~1..HEAD"])
        .output()
        .context("git diff (fallback) failed")?;
    Ok(parse_git_output(&out.stdout))
}

fn parse_git_output(stdout: &[u8]) -> Vec<String> {
    String::from_utf8_lossy(stdout)
        .lines()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

/// Entry point invoked by `xtask ci plan`.
pub fn run(
    base: Option<String>,
    head: Option<String>,
    labels_json: Option<String>,
    json_out: Option<PathBuf>,
    github_summary: Option<PathBuf>,
    dry_run: bool,
) -> Result<()> {
    let labels: Vec<String> = match labels_json.as_deref() {
        Some(s) if !s.trim().is_empty() => {
            serde_json::from_str(s).context("--labels-json must be a JSON array of strings")?
        }
        _ => Vec::new(),
    };

    let changed =
        if dry_run { Vec::new() } else { git_changed_files(base.as_deref(), head.as_deref())? };

    let budget = load_budget_policy(None);
    let plan = plan_with_policy(&changed, &labels, &budget);
    let json = serde_json::to_string_pretty(&plan).context("serialize plan to JSON")?;
    println!("{json}");

    if let Some(path) = json_out {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent).ok();
        }
        fs::write(&path, &json).with_context(|| format!("write {}", path.display()))?;
    }

    if let Some(path) = github_summary {
        let summary = render_summary(&plan);
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .with_context(|| format!("open {}", path.display()))?;
        f.write_all(summary.as_bytes()).context("write step summary")?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lanes_named(plan: &Plan) -> Vec<&str> {
        plan.lanes.iter().map(|l| l.name.as_str()).collect()
    }

    #[test]
    fn empty_pr_has_only_no_lanes_and_pennies_band() {
        let plan = plan_for(&[], &[]);
        assert_eq!(plan.posture, "empty");
        assert_eq!(plan.lanes.len(), 0);
        assert_eq!(plan.estimated_lem, 0);
        assert!(plan.band.contains("pennies"));
    }

    #[test]
    fn docs_only_posture_skips_rust_lanes() {
        let plan =
            plan_for(&["docs/ci/cost-and-verification-policy.md".into(), "README.md".into()], &[]);
        assert_eq!(plan.posture, "docs-only");
        let names = lanes_named(&plan);
        // Only the always-on guards lane should fire.
        assert_eq!(names, vec!["Guards / PR Size Guard / Markdownlint / Link Check"]);
        assert_eq!(plan.estimated_lem, 4);
        assert!(plan.band.contains("pennies"));
    }

    #[test]
    fn rust_crate_change_runs_core_msrv_feature_matrix_ripr_guards() {
        let plan = plan_for(&["crates/bitnet-quantization/src/i2s_qk256.rs".into()], &[]);
        assert_eq!(plan.posture, "rust");
        let names: Vec<&str> = lanes_named(&plan);
        assert!(names.contains(&"CI (Core) — build/test/clippy/docs"));
        assert!(names.contains(&"Feature Matrix (PR 3-combo)"));
        assert!(names.contains(&"Compatibility (MSRV)"));
        assert!(names.contains(&"ripr static exposure (advisory)"));
        assert!(names.contains(&"Guards / PR Size Guard / Markdownlint / Link Check"));
        // No GPU lane on a quantization-only diff.
        assert!(!names.iter().any(|n| n.starts_with("GPU CI")));
        // 22 (Core) + 12 (Feature Matrix PR) + 12 (MSRV) + 4 (ripr) + 4 (guards) = 54.
        // That sits in the elevated band (35–75) — still well below the $1 ceiling
        // but worth surfacing so engineers see the cost of a normal Rust PR.
        assert_eq!(plan.estimated_lem, 54);
        assert!(plan.band.contains("elevated"));
    }

    #[test]
    fn gpu_kernel_change_includes_gpu_native_compile() {
        let plan = plan_for(&["crates/bitnet-kernels/src/cuda_smoke.rs".into()], &[]);
        let names = lanes_named(&plan);
        assert!(names.contains(&"GPU CI Matrix (native compile)"));
        // Without the gpu-ci/docker label, Docker build does NOT fire.
        assert!(!names.contains(&"GPU CI Matrix (Docker, ~6x)"));
    }

    #[test]
    fn full_ci_label_fires_all_label_gated_lanes() {
        let plan =
            plan_for(&["crates/bitnet-kernels/src/cuda_smoke.rs".into()], &["full-ci".into()]);
        let names = lanes_named(&plan);
        assert!(names.contains(&"BDD Grid Check"));
        assert!(names.contains(&"Clippy (macOS ARM64)"));
        assert!(names.contains(&"Feature Matrix (full ~21 jobs)"));
        assert!(names.contains(&"GPU CI Matrix (Docker, ~6x)"));
        assert!(names.contains(&"Compatibility (ABI/FFI)"));
        assert!(names.contains(&"Compatibility (tokenizer)"));
        assert!(names.contains(&"Property Tests (smoke)"));
        assert!(names.contains(&"ripr static exposure (advisory)"));
    }

    #[test]
    fn ripr_eligible_pr_includes_advisory_lane() {
        let plan = plan_for(&["crates/bitnet-inference/src/decoder.rs".into()], &[]);
        assert!(lanes_named(&plan).iter().any(|n| *n == "ripr static exposure (advisory)"));
    }

    #[test]
    fn test_only_rust_diff_skips_ripr_advisory() {
        // tests/ counts as rust_core but NOT rust_production, and ripr does not
        // fire for test-only diffs unless the `ripr` label is applied.
        let plan = plan_for(&["tests/regression/cpu_only.rs".into()], &[]);
        let names = lanes_named(&plan);
        assert!(names.contains(&"CI (Core) — build/test/clippy/docs"));
        assert!(!names.iter().any(|n| n.starts_with("ripr ")));
    }

    #[test]
    fn label_only_ripr_runs_advisory_even_on_docs_pr() {
        let plan = plan_for(&["docs/ci/cost-and-verification-policy.md".into()], &["ripr".into()]);
        assert!(lanes_named(&plan).iter().any(|n| *n == "ripr static exposure (advisory)"));
    }

    #[test]
    fn band_classifies_high_when_full_ci_on_kernel_change() {
        let plan =
            plan_for(&["crates/bitnet-kernels/src/cuda_smoke.rs".into()], &["full-ci".into()]);
        // 22 (Core) + 4 (BDD) + 150 (macOS) + 70 (Feature full)
        // + 18 (GPU native) + 90 (GPU Docker) + 12 (MSRV) + 8 (FFI) + 6 (tok)
        // + 4 (property) + 4 (ripr) + 4 (guards) ≈ 392
        assert!(plan.estimated_lem > 125);
        assert!(plan.band.contains("over hard ceiling"));
    }

    #[test]
    fn parse_git_output_strips_blank_lines() {
        let raw = b"crates/foo.rs\n\ndocs/bar.md\n";
        assert_eq!(
            parse_git_output(raw),
            vec!["crates/foo.rs".to_string(), "docs/bar.md".to_string()]
        );
    }

    #[test]
    fn budget_policy_default_matches_committed_toml() {
        // Committed policy/ci-budget.toml ships values that mirror the
        // BudgetPolicy::default() fallback; they must agree so the planner
        // produces identical output with or without the file.
        let p = BudgetPolicy::default();
        assert_eq!(p.preferred_default_lem, 25);
        assert_eq!(p.default_limit_lem, 35);
        assert_eq!(p.elevated_limit_lem, 75);
        assert_eq!(p.hard_limit_lem, 125);
    }

    #[test]
    fn budget_policy_override_changes_band() {
        let stricter = BudgetPolicy {
            preferred_default_lem: 10,
            default_limit_lem: 20,
            elevated_limit_lem: 40,
            hard_limit_lem: 60,
        };
        let plan = plan_with_policy(
            &["crates/bitnet-quantization/src/i2s_qk256.rs".into()],
            &[],
            &stricter,
        );
        // Same diff that lands at 54 LEM under defaults (elevated band)
        // becomes "high" under the stricter policy — the threshold table
        // moved, the diff didn't.
        assert_eq!(plan.estimated_lem, 54);
        assert!(plan.band.contains("high"));
    }

    #[test]
    fn load_budget_policy_falls_back_when_missing() {
        let tmp = tempfile::tempdir().expect("tempdir");
        // No ci-budget.toml in tmp dir → expect default fallback.
        let p = load_budget_policy(Some(tmp.path()));
        assert_eq!(p, BudgetPolicy::default());
    }

    #[test]
    fn load_budget_policy_reads_toml() {
        let tmp = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            tmp.path().join("ci-budget.toml"),
            r#"
[budget]
preferred_default_lem = 5
default_limit_lem = 10
elevated_limit_lem = 20
hard_limit_lem = 40
"#,
        )
        .unwrap();
        let p = load_budget_policy(Some(tmp.path()));
        assert_eq!(p.preferred_default_lem, 5);
        assert_eq!(p.default_limit_lem, 10);
        assert_eq!(p.elevated_limit_lem, 20);
        assert_eq!(p.hard_limit_lem, 40);
    }
}
