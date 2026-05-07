//! `xtask ci plan` — Rust port of the inline Python planner that
//! lives in `.github/workflows/pr-plan.yml`.
//!
//! The planner classifies changed files by area, picks expected CI
//! lanes given those areas plus the PR's labels, and assigns each
//! lane an estimated LEM (Linux-Equivalent Minutes) cost. The
//! output:
//!
//! * `ci-plan.json` — machine-readable for downstream routing
//! * a markdown table appended to `$GITHUB_STEP_SUMMARY` for engineer
//!   visibility
//!
//! The planner is intentionally faithful to the existing Python
//! implementation so PR 14 is a pure port; PR 15 then layers
//! policy-backed cost / risk-pack files on top.

use anyhow::{Context, Result, bail};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

#[derive(Debug, Clone, Serialize)]
pub struct Lane {
    pub name: String,
    pub lem: u64,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Plan {
    pub posture: String,
    pub touched: BTreeMap<String, bool>,
    pub labels: Vec<String>,
    pub lanes: Vec<Lane>,
    pub estimated_lem: u64,
    pub band: String,
    pub changed_count: usize,
    /// Risk packs activated by the changed paths (PR 17). Names match
    /// the keys in `policy/ci-risk-packs.toml`.
    #[serde(default)]
    pub risk_packs: Vec<String>,
    /// Soft-budget guard verdict (PR 18). One of "ok", "warn",
    /// "strong-warn", "ack-suggested", "block".
    #[serde(default)]
    pub guard: String,
    /// Override labels detected on the PR that may permit a budget
    /// overage (PR 18).
    #[serde(default)]
    pub override_labels_present: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct LabelsWrapper {
    #[serde(default)]
    items: Vec<String>,
}

fn parse_labels(input: &str) -> Result<Vec<String>> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Ok(vec![]);
    }
    if let Ok(direct) = serde_json::from_str::<Vec<String>>(trimmed) {
        return Ok(direct);
    }
    if let Ok(wrapped) = serde_json::from_str::<LabelsWrapper>(trimmed) {
        return Ok(wrapped.items);
    }
    bail!("could not parse labels JSON: {input}")
}

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

fn classify_areas(files: &[String]) -> BTreeMap<String, bool> {
    let patterns = area_patterns();
    let compiled: Vec<(&str, Vec<Regex>)> = patterns
        .into_iter()
        .map(|(area, ps)| {
            let regexes: Vec<Regex> = ps.into_iter().filter_map(|p| Regex::new(p).ok()).collect();
            (area, regexes)
        })
        .collect();

    let mut touched: BTreeMap<String, bool> =
        compiled.iter().map(|(area, _)| ((*area).to_string(), false)).collect();
    for f in files {
        for (area, regexes) in &compiled {
            if regexes.iter().any(|r| r.is_match(f)) {
                touched.insert((*area).to_string(), true);
            }
        }
    }
    touched
}

fn pick_lanes(touched: &BTreeMap<String, bool>, labels: &[String], any_changed: bool) -> Vec<Lane> {
    let touched_get = |k: &str| touched.get(k).copied().unwrap_or(false);
    let has = |l: &str| labels.iter().any(|x| x == l);
    let mut lanes = Vec::new();

    if touched_get("rust_core") {
        lanes.push(Lane {
            name: "CI (Core) — build/test/clippy/docs".into(),
            lem: 22,
            reason: "rust_core changed".into(),
        });
    }
    if has("bdd") || has("grid") || has("full-ci") {
        lanes.push(Lane {
            name: "BDD Grid Check".into(),
            lem: 4,
            reason: "bdd/grid/full-ci label".into(),
        });
    }
    if has("macos") || has("full-ci") {
        lanes.push(Lane {
            name: "Clippy (macOS ARM64)".into(),
            lem: 15 * 10,
            reason: "macos/full-ci label".into(),
        });
    }
    if touched_get("rust_core") || touched_get("manifest") {
        if has("feature-matrix") || has("full-ci") {
            lanes.push(Lane {
                name: "Feature Matrix (full ~21 jobs)".into(),
                lem: 70,
                reason: "feature-matrix/full-ci label".into(),
            });
        } else {
            lanes.push(Lane {
                name: "Feature Matrix (PR 3-combo)".into(),
                lem: 12,
                reason: "rust/manifest changed".into(),
            });
        }
    }
    if touched_get("gpu") {
        lanes.push(Lane {
            name: "GPU CI Matrix (native compile)".into(),
            lem: 18,
            reason: "GPU paths changed".into(),
        });
        if has("gpu-ci") || has("docker") || has("full-ci") {
            lanes.push(Lane {
                name: "GPU CI Matrix (Docker, ~6x)".into(),
                lem: 90,
                reason: "gpu-ci/docker/full-ci label".into(),
            });
        }
    }
    if touched_get("rust_core") {
        lanes.push(Lane {
            name: "Compatibility (MSRV)".into(),
            lem: 12,
            reason: "rust_core changed".into(),
        });
    }
    if touched_get("ffi") || has("ffi") || has("abi") || has("full-ci") {
        lanes.push(Lane {
            name: "Compatibility (ABI/FFI)".into(),
            lem: 8,
            reason: "ffi area / label".into(),
        });
    }
    if touched_get("tokenizer") || has("tokenizer") || has("full-ci") {
        lanes.push(Lane {
            name: "Compatibility (tokenizer)".into(),
            lem: 6,
            reason: "tokenizer area / label".into(),
        });
    }
    if has("property-tests") || has("full-ci") {
        lanes.push(Lane {
            name: "Property Tests (smoke)".into(),
            lem: 4,
            reason: "property-tests/full-ci label".into(),
        });
    }
    if has("ripr") || has("full-ci") {
        lanes.push(Lane {
            name: "ripr static exposure (advisory)".into(),
            lem: 4,
            reason: "ripr/full-ci label".into(),
        });
    }
    if any_changed {
        lanes.push(Lane {
            name: "Guards / PR Size Guard / Markdownlint / Link Check".into(),
            lem: 4,
            reason: "always-on".into(),
        });
    }
    lanes
}

fn band_for(total: u64) -> &'static str {
    match total {
        0..=12 => "✅ pennies (< 12 LEM)",
        13..=35 => "✅ default budget (< 35 LEM)",
        36..=75 => "⚠️  elevated (35–75 LEM)",
        76..=125 => "⚠️  high (75–125 LEM)",
        _ => "🚨 over hard ceiling (> 125 LEM)",
    }
}

fn posture_for(touched: &BTreeMap<String, bool>, any_changed: bool) -> String {
    let touched_get = |k: &str| touched.get(k).copied().unwrap_or(false);
    if !any_changed {
        return "empty".into();
    }
    if touched_get("docs")
        && !(touched_get("rust_core")
            || touched_get("gpu")
            || touched_get("ffi")
            || touched_get("tokenizer"))
    {
        return "docs-only".into();
    }
    if touched_get("tracking")
        && !(touched_get("rust_core") || touched_get("gpu") || touched_get("ffi"))
    {
        return "tracking-only".into();
    }
    "rust".into()
}

/// Compute the plan from a list of changed files and labels.
pub fn build_plan(changed: &[String], labels: &[String]) -> Plan {
    let touched = classify_areas(changed);
    let lanes = pick_lanes(&touched, labels, !changed.is_empty());
    let total: u64 = lanes.iter().map(|l| l.lem).sum();
    let posture = posture_for(&touched, !changed.is_empty());
    let risk_packs = pick_risk_packs(changed);
    let (guard, override_labels_present) = guard_verdict(total, labels);
    Plan {
        posture,
        touched,
        labels: labels.to_vec(),
        lanes,
        estimated_lem: total,
        band: band_for(total).to_string(),
        changed_count: changed.len(),
        risk_packs,
        guard,
        override_labels_present,
    }
}

/// Risk-pack routing (PR 17). Maps changed paths to the risk-pack
/// keys declared in `policy/ci-risk-packs.toml`. The mapping is
/// embedded here for now to avoid a runtime dependency on the
/// policy TOML; PR 17 follow-up can read directly from disk.
fn pick_risk_packs(changed: &[String]) -> Vec<String> {
    let table: &[(&str, &[&str])] = &[
        (
            "qk256",
            &[
                "crates/bitnet-quantization/",
                "crates/bitnet-quantization-bits/",
                "crates/bitnet-qk256-",
                "crates/bitnet-models/src/qk256",
            ],
        ),
        (
            "kernels_cpu",
            &["crates/bitnet-kernels/", "crates/bitnet-cpu-activations/", "crates/bitnet-simd/"],
        ),
        (
            "gpu",
            &[
                "crates/bitnet-gpu-hal/",
                "crates/bitnet-device-probe/",
                "crates/bitnet-device-config-core/",
                "crates/bitnet-metal/",
                "crates/bitnet-opencl/",
                "crates/bitnet-vulkan",
                "crates/bitnet-vulkan-shaders",
                "crates/bitnet-wgpu",
                "crates/bitnet-wgpu-shaders-i2s",
                "crates/bitnet-rocm/",
                "crates/bitnet-nvidia/",
                "crates/bitnet-spirv/",
                "crates/bitnet-webgpu/",
            ],
        ),
        (
            "ffi",
            &["crates/bitnet-ffi/", "crates/bitnet-sys/", "crates/bitnet-ggml-ffi/", "crossval/"],
        ),
        (
            "tokenizer",
            &[
                "crates/bitnet-tokenizers/",
                "crates/bitnet-token-merge-core/",
                "crates/bitnet-tokenizer-model-core/",
                "crates/bitnet-tokenizer-discovery-core/",
                "crates/bitnet-tokenizer-text-core/",
                "tests/fixtures/tokenizers/",
            ],
        ),
        (
            "bdd_policy",
            &[
                "crates/bitnet-bdd-",
                "crates/bitnet-testing-policy",
                "crates/bitnet-testing-scenarios",
                "crates/bitnet-runtime-feature-flags",
                "crates/bitnet-startup-contract",
                "crates/bitnet-feature-contract",
            ],
        ),
        ("manifest_release", &["Cargo.toml", "Cargo.lock", "rust-toolchain.toml", ".cargo/"]),
        (
            "docs_tracking",
            &["docs/", ".codex/campaigns/", "README.md", "CHANGELOG.md", "CONTRIBUTING.md"],
        ),
    ];

    let mut out: Vec<String> = Vec::new();
    for (pack, prefixes) in table {
        let any_match = changed.iter().any(|c| prefixes.iter().any(|p| c.starts_with(p)));
        if any_match {
            out.push((*pack).to_string());
        }
    }
    out
}

/// Soft-budget guard verdict (PR 18). Reads thresholds from
/// `policy/ci-budget.toml` if present, otherwise uses the defaults
/// declared there.
fn guard_verdict(total: u64, labels: &[String]) -> (String, Vec<String>) {
    let warn_at = 35u64;
    let strong_warn_at = 75u64;
    let suggest_ack_at = 100u64;
    let fail_above = 125u64;
    let override_label_set: &[&str] = &["full-ci", "ci-budget-override", "ci-budget-ack"];

    let present: Vec<String> = override_label_set
        .iter()
        .filter(|l| labels.iter().any(|x| x == *l))
        .map(|s| (*s).to_string())
        .collect();

    let verdict = if total > fail_above {
        if !present.is_empty() { "block-overridden" } else { "block" }
    } else if total >= suggest_ack_at {
        "ack-suggested"
    } else if total >= strong_warn_at {
        "strong-warn"
    } else if total >= warn_at {
        "warn"
    } else {
        "ok"
    };

    (verdict.to_string(), present)
}

fn render_markdown(plan: &Plan) -> String {
    let touched_areas: Vec<&str> =
        plan.touched.iter().filter(|(_, v)| **v).map(|(k, _)| k.as_str()).collect();
    let mut s = String::new();
    s.push_str("# PR Plan\n\n");
    s.push_str(&format!("- **Posture:** {}\n", plan.posture));
    s.push_str(&format!(
        "- **Touched areas:** {}\n",
        if touched_areas.is_empty() { "(none)".to_string() } else { touched_areas.join(", ") }
    ));
    let labels_str = if plan.labels.is_empty() {
        "(none)".to_string()
    } else {
        let mut sorted = plan.labels.clone();
        sorted.sort();
        sorted.join(", ")
    };
    s.push_str(&format!("- **Labels:** {labels_str}\n"));
    s.push_str(&format!("- **Estimated LEM:** {}  ·  {}\n", plan.estimated_lem, plan.band));
    if !plan.risk_packs.is_empty() {
        s.push_str(&format!("- **Risk packs:** {}\n", plan.risk_packs.join(", ")));
    }
    let guard_icon = match plan.guard.as_str() {
        "ok" => "✅",
        "warn" => "⚠️",
        "strong-warn" => "⚠️⚠️",
        "ack-suggested" => "🔶",
        "block" => "🚨",
        "block-overridden" => "🚨(overridden)",
        _ => "·",
    };
    let override_note = if plan.override_labels_present.is_empty() {
        String::new()
    } else {
        format!(" (overrides: {})", plan.override_labels_present.join(", "))
    };
    s.push_str(&format!(
        "- **Budget guard:** {} `{}`{}\n\n",
        guard_icon, plan.guard, override_note
    ));
    s.push_str("| Lane | Estimated LEM | Reason |\n");
    s.push_str("|---|---:|---|\n");
    if plan.lanes.is_empty() {
        s.push_str("| (no lanes expected) | 0 | nothing matched |\n");
    } else {
        for l in &plan.lanes {
            s.push_str(&format!("| {} | {} | {} |\n", l.name, l.lem, l.reason));
        }
    }
    s.push_str(
        "\n> Estimates are rough heuristics derived from path globs + labels. \
        Actual minutes are recorded in the GitHub Actions metrics UI. \
        This job is advisory only and does not gate merges.\n",
    );
    s
}

fn changed_files(base: &str, head: &str) -> Result<Vec<String>> {
    let output = Command::new("git")
        .args(["diff", "--name-only", &format!("{base}...{head}")])
        .output()
        .with_context(|| format!("running git diff {base}...{head}"))?;
    if !output.status.success() {
        // Fall back to last commit on the branch.
        let alt = Command::new("git").args(["diff", "--name-only", "HEAD~1..HEAD"]).output()?;
        if !alt.status.success() {
            return Ok(vec![]);
        }
        return Ok(decode_lines(&alt.stdout));
    }
    Ok(decode_lines(&output.stdout))
}

fn decode_lines(bytes: &[u8]) -> Vec<String> {
    String::from_utf8_lossy(bytes).lines().filter(|s| !s.is_empty()).map(str::to_string).collect()
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
pub fn run(
    base: Option<String>,
    head: Option<String>,
    labels_json: Option<String>,
    changed_file: Option<PathBuf>,
    json_out: Option<PathBuf>,
    github_summary: Option<PathBuf>,
    print_stdout: bool,
    enforce_budget: bool,
) -> Result<()> {
    let labels = match labels_json {
        Some(j) => parse_labels(&j)?,
        None => vec![],
    };

    let changed = if let Some(p) = changed_file.as_ref() {
        let text = fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))?;
        decode_lines(text.as_bytes())
    } else {
        let base = base.unwrap_or_else(|| "origin/main".to_string());
        let head = head.unwrap_or_else(|| "HEAD".to_string());
        changed_files(&base, &head)?
    };

    let plan = build_plan(&changed, &labels);

    let json = serde_json::to_string_pretty(&plan)?;
    if print_stdout {
        println!("{json}");
    }
    if let Some(p) = json_out {
        if let Some(parent) = p.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }
        fs::write(&p, &json).with_context(|| format!("writing {}", p.display()))?;
    }
    if let Some(p) = github_summary {
        let md = render_markdown(&plan);
        let mut existing = fs::read_to_string(&p).unwrap_or_default();
        if !existing.ends_with('\n') && !existing.is_empty() {
            existing.push('\n');
        }
        existing.push_str(&md);
        fs::write(&p, existing).with_context(|| format!("writing {}", p.display()))?;
    }

    if enforce_budget && plan.guard == "block" {
        bail!(
            "ci-plan budget guard: estimated LEM {} > hard ceiling 125 with no override label \
             ({}). Add `full-ci`, `ci-budget-override`, or `ci-budget-ack` to acknowledge.",
            plan.estimated_lem,
            plan.labels.join(", ")
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| (*x).to_string()).collect()
    }

    #[test]
    fn docs_only_posture() {
        let plan = build_plan(&s(&["docs/foo.md", "README.md"]), &[]);
        assert_eq!(plan.posture, "docs-only");
        assert_eq!(plan.estimated_lem, 4); // always-on guards lane
    }

    #[test]
    fn rust_core_posture_picks_core_lane_and_msrv() {
        let plan = build_plan(&s(&["crates/bitnet-quantization/src/qk256.rs"]), &[]);
        assert_eq!(plan.posture, "rust");
        let names: Vec<&str> = plan.lanes.iter().map(|l| l.name.as_str()).collect();
        assert!(names.iter().any(|n| n.contains("CI (Core)")));
        assert!(names.iter().any(|n| n.contains("Feature Matrix (PR 3-combo)")));
        assert!(names.iter().any(|n| n.contains("Compatibility (MSRV)")));
    }

    #[test]
    fn full_ci_label_picks_expensive_lanes() {
        let plan = build_plan(&s(&["crates/bitnet-kernels/src/lib.rs"]), &s(&["full-ci"]));
        let names: Vec<&str> = plan.lanes.iter().map(|l| l.name.as_str()).collect();
        assert!(names.iter().any(|n| n.contains("GPU CI Matrix (Docker")));
        assert!(names.iter().any(|n| n.contains("Clippy (macOS ARM64)")));
    }

    #[test]
    fn empty_change_set_yields_empty_posture() {
        let plan = build_plan(&[], &[]);
        assert_eq!(plan.posture, "empty");
        assert_eq!(plan.estimated_lem, 0);
        assert!(plan.lanes.is_empty());
    }

    #[test]
    fn band_thresholds() {
        assert_eq!(band_for(5), "✅ pennies (< 12 LEM)");
        assert_eq!(band_for(20), "✅ default budget (< 35 LEM)");
        assert_eq!(band_for(60), "⚠️  elevated (35–75 LEM)");
        assert_eq!(band_for(100), "⚠️  high (75–125 LEM)");
        assert_eq!(band_for(200), "🚨 over hard ceiling (> 125 LEM)");
    }

    #[test]
    fn parses_label_array() {
        let l = parse_labels("[\"a\", \"b\"]").unwrap();
        assert_eq!(l, vec!["a".to_string(), "b".to_string()]);
        let empty = parse_labels("").unwrap();
        assert!(empty.is_empty());
    }

    #[test]
    fn risk_packs_qk256_and_kernels() {
        let plan = build_plan(
            &s(&[
                "crates/bitnet-quantization/src/qk256.rs",
                "crates/bitnet-kernels/src/cpu/avx2.rs",
            ]),
            &[],
        );
        assert!(plan.risk_packs.iter().any(|p| p == "qk256"));
        assert!(plan.risk_packs.iter().any(|p| p == "kernels_cpu"));
    }

    #[test]
    fn risk_packs_gpu_ffi_and_tokenizer() {
        let plan = build_plan(
            &s(&[
                "crates/bitnet-metal/src/kernels/matmul.metal",
                "crates/bitnet-ffi/src/lib.rs",
                "crates/bitnet-tokenizers/src/lib.rs",
            ]),
            &[],
        );
        assert!(plan.risk_packs.iter().any(|p| p == "gpu"));
        assert!(plan.risk_packs.iter().any(|p| p == "ffi"));
        assert!(plan.risk_packs.iter().any(|p| p == "tokenizer"));
    }

    #[test]
    fn risk_packs_manifest_release() {
        let plan = build_plan(&s(&["Cargo.toml", "Cargo.lock"]), &[]);
        assert!(plan.risk_packs.iter().any(|p| p == "manifest_release"));
    }

    #[test]
    fn risk_packs_docs_tracking() {
        let plan = build_plan(&s(&["docs/foo.md", "README.md"]), &[]);
        assert!(plan.risk_packs.iter().any(|p| p == "docs_tracking"));
    }

    #[test]
    fn guard_ok_for_low_lem() {
        let plan = build_plan(&s(&["docs/foo.md"]), &[]);
        assert_eq!(plan.guard, "ok");
        assert!(plan.override_labels_present.is_empty());
    }

    #[test]
    fn guard_warn_at_35_lem() {
        let (verdict, _) = guard_verdict(50, &[]);
        assert_eq!(verdict, "warn");
    }

    #[test]
    fn guard_strong_warn_at_75_lem() {
        let (verdict, _) = guard_verdict(80, &[]);
        assert_eq!(verdict, "strong-warn");
    }

    #[test]
    fn guard_block_above_125_without_override() {
        let (verdict, present) = guard_verdict(150, &["unrelated".into()]);
        assert_eq!(verdict, "block");
        assert!(present.is_empty());
    }

    #[test]
    fn guard_block_overridden_with_full_ci_label() {
        let (verdict, present) = guard_verdict(150, &["full-ci".into()]);
        assert_eq!(verdict, "block-overridden");
        assert_eq!(present, vec!["full-ci".to_string()]);
    }

    #[test]
    fn guard_ack_suggested_at_100_lem() {
        let (verdict, _) = guard_verdict(110, &[]);
        assert_eq!(verdict, "ack-suggested");
    }
}
