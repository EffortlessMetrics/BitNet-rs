//! `xtask ci plan` — Rust-native PR planner used by
//! `.github/workflows/pr-plan.yml`.
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
//! The planner replaces the legacy inline Python workflow implementation and
//! centralizes path classification, LEM heuristics, risk-pack detection, and
//! soft budget guard output in Rust.

use anyhow::{Context, Result, bail};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;
use std::process::Command;

#[derive(Debug, Clone, Serialize)]
pub struct Lane {
    pub id: String,
    pub name: String,
    pub lem: u64,
    pub reason: String,
    pub blocking: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct SelectedLane {
    pub id: String,
    pub name: String,
    pub estimated_lem: u64,
    pub reason: String,
    pub blocking: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct SkippedLane {
    pub id: String,
    pub name: String,
    pub reason: String,
    pub blocking: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct Budget {
    pub preferred_default_lem: u64,
    pub default_limit_lem: u64,
    pub estimated_lem: u64,
    pub posture: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Classification {
    pub docs_only: bool,
    pub tracker_only: bool,
    pub rust_inputs_changed: bool,
    pub manifest_or_toolchain_changed: bool,
    pub public_api_changed: bool,
    pub gpu_changed: bool,
    pub macos_changed: bool,
    pub model_validation_changed: bool,
    pub coverage_requested: bool,
    pub full_ci_requested: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct Packages {
    pub changed: Vec<String>,
    pub direct_dependents: Vec<String>,
    pub canaries: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Plan {
    pub schema_version: u32,
    pub budget: Budget,
    pub classification: Classification,
    pub selected_lanes: Vec<SelectedLane>,
    pub skipped_lanes: Vec<SkippedLane>,
    pub packages: Packages,
    pub risk_packs: Vec<String>,
    pub labels: Vec<String>,
    #[serde(skip_serializing)]
    pub posture: String,
    #[serde(skip_serializing)]
    pub touched: BTreeMap<String, bool>,
    #[serde(skip_serializing)]
    pub lanes: Vec<Lane>,
    #[serde(skip_serializing)]
    pub estimated_lem: u64,
    #[serde(skip_serializing)]
    pub band: String,
    #[serde(skip_serializing)]
    pub changed_count: usize,
    /// Soft-budget guard verdict (PR 18). One of "ok", "warn",
    /// "strong-warn", "ack-suggested", "block".
    #[serde(skip_serializing)]
    pub guard: String,
    /// Override labels detected on the PR that may permit a budget
    /// overage (PR 18).
    #[serde(skip_serializing)]
    pub override_labels_present: Vec<String>,
}

const CI_PLAN_SCHEMA_VERSION: u32 = 1;
const PREFERRED_DEFAULT_LEM: u64 = 25;
const DEFAULT_LIMIT_LEM: u64 = 35;

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

fn lane(id: &str, name: &str, lem: u64, reason: &str, blocking: bool) -> Lane {
    Lane { id: id.to_string(), name: name.to_string(), lem, reason: reason.to_string(), blocking }
}

fn lane_catalog() -> Vec<SkippedLane> {
    vec![
        skipped_lane("pr-plan", "PR Plan", false),
        skipped_lane("ci-core-build-test", "CI (Core) - build/test/clippy/docs", true),
        skipped_lane("feature-matrix-pr", "Feature Matrix (PR smoke)", true),
        skipped_lane("feature-matrix-full", "Feature Matrix (full)", false),
        skipped_lane("bdd-grid-check", "BDD Grid Check", true),
        skipped_lane("macos-arm64-route", "Route macOS PR lane", false),
        skipped_lane("macos-arm64-clippy", "Clippy (macOS ARM64)", false),
        skipped_lane("performance-tracking-route", "Route Performance Tracking", false),
        skipped_lane("test-telemetry-route", "Route Test Telemetry", false),
        skipped_lane("compatibility-msrv-route", "Route MSRV Compatibility", false),
        skipped_lane("compatibility-msrv", "Compatibility (MSRV)", true),
        skipped_lane("compatibility-ffi-abi", "Compatibility (ABI/FFI)", true),
        skipped_lane("compatibility-tokenizer", "Compatibility (tokenizer)", true),
        skipped_lane("gpu-native", "GPU CI Matrix (native compile)", true),
        skipped_lane("gpu-docker", "GPU CI Matrix (Docker)", false),
        skipped_lane("property-tests", "Property Tests (smoke)", false),
        skipped_lane("ripr-advisory", "ripr static exposure (advisory)", false),
        skipped_lane("always-on-guards", "Guards / PR Size / Markdown / Link", true),
    ]
}

fn skipped_lane(id: &str, name: &str, blocking: bool) -> SkippedLane {
    SkippedLane {
        id: id.to_string(),
        name: name.to_string(),
        reason: "not selected for changed files or labels".to_string(),
        blocking,
    }
}

fn pick_lanes(
    touched: &BTreeMap<String, bool>,
    labels: &[String],
    changed: &[String],
) -> Vec<Lane> {
    let touched_get = |k: &str| touched.get(k).copied().unwrap_or(false);
    let has = |l: &str| labels.iter().any(|x| x == l);
    let any_changed = !changed.is_empty();
    let manifest_or_toolchain = manifest_or_toolchain_changed(changed);
    let public_api = public_api_changed(changed);
    let macos_paths = macos_changed(changed);
    let mut lanes = Vec::new();

    if any_changed {
        lanes.push(lane("pr-plan", "PR Plan", 1, "plan artifact", false));
        lanes.push(lane("macos-arm64-route", "Route macOS PR lane", 1, "cheap route job", false));
        lanes.push(lane(
            "performance-tracking-route",
            "Route Performance Tracking",
            1,
            "cheap route job",
            false,
        ));
        lanes.push(lane(
            "test-telemetry-route",
            "Route Test Telemetry",
            1,
            "cheap route job",
            false,
        ));
        lanes.push(lane(
            "compatibility-msrv-route",
            "Route MSRV Compatibility",
            1,
            "cheap route job",
            false,
        ));
    }
    if touched_get("rust_core") {
        lanes.push(lane(
            "ci-core-build-test",
            "CI (Core) - build/test/clippy/docs",
            22,
            "rust_core changed",
            true,
        ));
    }
    if has("bdd") || has("grid") || has("full-ci") {
        lanes.push(lane("bdd-grid-check", "BDD Grid Check", 4, "bdd/grid/full-ci label", true));
    }
    if macos_paths || has("macos") || has("apple-silicon") || has("metal") || has("full-ci") {
        lanes.push(lane(
            "macos-arm64-clippy",
            "Clippy (macOS ARM64)",
            15 * 10,
            "macOS path or label",
            false,
        ));
    }
    if touched_get("rust_core") || touched_get("manifest") {
        if has("feature-matrix") || has("full-ci") {
            lanes.push(lane(
                "feature-matrix-full",
                "Feature Matrix (full ~21 jobs)",
                70,
                "feature-matrix/full-ci label",
                false,
            ));
        } else {
            lanes.push(lane(
                "feature-matrix-pr",
                "Feature Matrix (PR smoke)",
                12,
                "rust/manifest changed",
                true,
            ));
        }
    }
    if touched_get("gpu") {
        lanes.push(lane(
            "gpu-native",
            "GPU CI Matrix (native compile)",
            18,
            "GPU paths changed",
            true,
        ));
        if has("gpu-ci") || has("docker") || has("full-ci") {
            lanes.push(lane(
                "gpu-docker",
                "GPU CI Matrix (Docker, ~6x)",
                90,
                "gpu-ci/docker/full-ci label",
                false,
            ));
        }
    }
    if manifest_or_toolchain || public_api || has("msrv") || has("compatibility") || has("full-ci")
    {
        lanes.push(lane(
            "compatibility-msrv",
            "Compatibility (MSRV)",
            12,
            "manifest/toolchain/public-api risk or label",
            true,
        ));
    }
    if touched_get("ffi") || has("ffi") || has("abi") || has("full-ci") {
        lanes.push(lane(
            "compatibility-ffi-abi",
            "Compatibility (ABI/FFI)",
            8,
            "ffi area / label",
            true,
        ));
    }
    if touched_get("tokenizer") || has("tokenizer") || has("full-ci") {
        lanes.push(lane(
            "compatibility-tokenizer",
            "Compatibility (tokenizer)",
            6,
            "tokenizer area / label",
            true,
        ));
    }
    if has("property-tests") || has("full-ci") {
        lanes.push(lane(
            "property-tests",
            "Property Tests (smoke)",
            4,
            "property-tests/full-ci label",
            false,
        ));
    }
    if has("ripr") || has("full-ci") {
        lanes.push(lane(
            "ripr-advisory",
            "ripr static exposure (advisory)",
            4,
            "ripr/full-ci label",
            false,
        ));
    }
    if any_changed {
        lanes.push(lane(
            "always-on-guards",
            "Guards / PR Size Guard / Markdownlint / Link Check",
            4,
            "always-on",
            true,
        ));
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

fn budget_posture_for(total: u64) -> &'static str {
    match total {
        0..=12 => "pennies",
        13..=35 => "default",
        36..=75 => "elevated",
        76..=125 => "high",
        _ => "hard",
    }
}

fn is_tracker_path(path: &str) -> bool {
    path.starts_with("docs/tracking/") || path.starts_with(".codex/campaigns/")
}

fn is_docs_path(path: &str) -> bool {
    path.starts_with("docs/")
        || path.ends_with(".md")
        || path.starts_with("README")
        || path.starts_with("CHANGELOG")
        || path.starts_with("CONTRIBUTING")
        || path.starts_with("SECURITY")
        || path.starts_with("COMPATIBILITY")
        || path.starts_with("THIRD_PARTY")
        || path == "CLAUDE.md"
}

fn manifest_or_toolchain_changed(files: &[String]) -> bool {
    files.iter().any(|path| {
        path == "Cargo.toml"
            || path == "Cargo.lock"
            || path == "rust-toolchain.toml"
            || path.starts_with(".cargo/")
            || (path.starts_with("crates/") && path.ends_with("/Cargo.toml"))
    })
}

fn public_api_changed(files: &[String]) -> bool {
    files.iter().any(|path| {
        (path.starts_with("crates/")
            && (path.ends_with("/src/lib.rs") || path.contains("/src/api/")))
            || path.starts_with("crates/bitnet-ffi/")
            || path.starts_with("crates/bitnet-py/")
            || path == "COMPATIBILITY.md"
            || path == "MIGRATION.md"
            || path.starts_with("docs/release/")
    })
}

fn macos_changed(files: &[String]) -> bool {
    files.iter().any(|path| {
        path.starts_with("crates/bitnet-metal/")
            || path == ".github/workflows/macos-arm64.yml"
            || path.starts_with("docs/apple/")
    })
}

fn model_validation_changed(files: &[String]) -> bool {
    files.iter().any(|path| {
        path.starts_with("ci/model-artifacts/")
            || path.starts_with("ci/hardware/")
            || path.starts_with("docs/model-contracts/")
            || path.starts_with("tests/fixtures/models/")
            || path.starts_with("models/")
    })
}

fn build_classification(
    changed: &[String],
    touched: &BTreeMap<String, bool>,
    labels: &[String],
) -> Classification {
    let has_label = |label: &str| labels.iter().any(|item| item == label);
    let tracker_only = !changed.is_empty() && changed.iter().all(|path| is_tracker_path(path));
    let docs_only = !changed.is_empty()
        && changed.iter().all(|path| is_docs_path(path) && !is_tracker_path(path));

    Classification {
        docs_only,
        tracker_only,
        rust_inputs_changed: touched.get("rust_core").copied().unwrap_or(false),
        manifest_or_toolchain_changed: manifest_or_toolchain_changed(changed),
        public_api_changed: public_api_changed(changed),
        gpu_changed: touched.get("gpu").copied().unwrap_or(false),
        macos_changed: macos_changed(changed),
        model_validation_changed: model_validation_changed(changed)
            || has_label("model-validation"),
        coverage_requested: has_label("coverage") || has_label("full-ci"),
        full_ci_requested: has_label("full-ci"),
    }
}

fn posture_for(touched: &BTreeMap<String, bool>, any_changed: bool) -> String {
    let touched_get = |k: &str| touched.get(k).copied().unwrap_or(false);
    if !any_changed {
        return "empty".into();
    }
    if touched_get("tracking")
        && !(touched_get("rust_core") || touched_get("gpu") || touched_get("ffi"))
    {
        return "tracking-only".into();
    }
    if touched_get("docs")
        && !(touched_get("rust_core")
            || touched_get("gpu")
            || touched_get("ffi")
            || touched_get("tokenizer"))
    {
        return "docs-only".into();
    }
    "rust".into()
}

fn selected_lanes(lanes: &[Lane]) -> Vec<SelectedLane> {
    lanes
        .iter()
        .map(|lane| SelectedLane {
            id: lane.id.clone(),
            name: lane.name.clone(),
            estimated_lem: lane.lem,
            reason: lane.reason.clone(),
            blocking: lane.blocking,
        })
        .collect()
}

fn skipped_lanes(lanes: &[Lane]) -> Vec<SkippedLane> {
    let selected: BTreeSet<&str> = lanes.iter().map(|lane| lane.id.as_str()).collect();
    lane_catalog().into_iter().filter(|lane| !selected.contains(lane.id.as_str())).collect()
}

fn build_budget(total: u64) -> Budget {
    Budget {
        preferred_default_lem: PREFERRED_DEFAULT_LEM,
        default_limit_lem: DEFAULT_LIMIT_LEM,
        estimated_lem: total,
        posture: budget_posture_for(total).to_string(),
    }
}

fn package_name_for_path(path: &str) -> Option<String> {
    let mut parts = path.split('/');
    match parts.next()? {
        "crates" => parts.next().map(str::to_string),
        "xtask" => Some("xtask".to_string()),
        "crossval" => Some("bitnet-crossval".to_string()),
        _ => None,
    }
}

fn build_packages(changed: &[String], risk_packs: &[String]) -> Packages {
    let changed_packages: BTreeSet<String> =
        changed.iter().filter_map(|path| package_name_for_path(path)).collect();
    let risk_set: BTreeSet<&str> = risk_packs.iter().map(String::as_str).collect();
    let mut canaries = BTreeSet::new();
    if risk_set.contains("qk256") {
        canaries.insert("bitnet-quantization".to_string());
        canaries.insert("bitnet-models".to_string());
    }
    if risk_set.contains("kernels_cpu") {
        canaries.insert("bitnet-kernels".to_string());
    }
    if risk_set.contains("gpu") {
        canaries.insert("bitnet-gpu-hal".to_string());
    }
    if risk_set.contains("tokenizer") {
        canaries.insert("bitnet-tokenizers".to_string());
    }
    if risk_set.contains("bdd_policy") {
        canaries.insert("bitnet-bdd-grid".to_string());
    }

    Packages {
        changed: changed_packages.into_iter().collect(),
        direct_dependents: Vec::new(),
        canaries: canaries.into_iter().collect(),
    }
}

/// Compute the plan from a list of changed files and labels.
pub fn build_plan(changed: &[String], labels: &[String]) -> Plan {
    let touched = classify_areas(changed);
    let lanes = pick_lanes(&touched, labels, changed);
    let total: u64 = lanes.iter().map(|l| l.lem).sum();
    let posture = posture_for(&touched, !changed.is_empty());
    let risk_packs = pick_risk_packs(changed);
    let (guard, override_labels_present) = guard_verdict(total, labels);
    let classification = build_classification(changed, &touched, labels);
    let packages = build_packages(changed, &risk_packs);
    let selected_lanes = selected_lanes(&lanes);
    let skipped_lanes = skipped_lanes(&lanes);
    Plan {
        schema_version: CI_PLAN_SCHEMA_VERSION,
        budget: build_budget(total),
        classification,
        selected_lanes,
        skipped_lanes,
        packages,
        risk_packs,
        labels: labels.to_vec(),
        posture,
        touched,
        lanes,
        estimated_lem: total,
        band: band_for(total).to_string(),
        changed_count: changed.len(),
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
    if manifest_or_toolchain_changed(changed) && !out.iter().any(|pack| pack == "manifest_release")
    {
        out.push("manifest_release".to_string());
    }
    if public_api_changed(changed) && !out.iter().any(|pack| pack == "public_api") {
        out.push("public_api".to_string());
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
    use serde_json::json;
    use std::path::PathBuf;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| (*x).to_string()).collect()
    }

    fn fixture_lines(name: &str) -> Result<Vec<String>> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("tests")
            .join("fixtures")
            .join("ci-plan")
            .join(name);
        let text = std::fs::read_to_string(&path)?;
        Ok(decode_lines(text.as_bytes()))
    }

    #[test]
    fn docs_only_posture() {
        let plan = build_plan(&s(&["docs/foo.md", "README.md"]), &[]);
        assert_eq!(plan.posture, "docs-only");
        assert_eq!(plan.estimated_lem, 9); // route jobs plus always-on guards
        assert!(plan.classification.docs_only);
        assert_eq!(plan.budget.posture, "pennies");
    }

    #[test]
    fn rust_core_posture_picks_core_lane_without_msrv_for_leaf_edits() {
        let plan = build_plan(&s(&["crates/bitnet-quantization/src/qk256.rs"]), &[]);
        assert_eq!(plan.posture, "rust");
        let names: Vec<&str> = plan.lanes.iter().map(|l| l.name.as_str()).collect();
        assert!(names.iter().any(|n| n.contains("CI (Core)")));
        assert!(names.iter().any(|n| n.contains("Feature Matrix (PR smoke)")));
        assert!(!names.iter().any(|n| n.contains("Compatibility (MSRV)")));
    }

    #[test]
    fn manifest_or_public_api_picks_msrv() {
        let plan = build_plan(&s(&["Cargo.lock", "crates/bitnet-cli/src/lib.rs"]), &[]);
        let names: Vec<&str> = plan.lanes.iter().map(|l| l.name.as_str()).collect();
        assert!(names.iter().any(|n| n.contains("Compatibility (MSRV)")));
        assert!(plan.classification.manifest_or_toolchain_changed);
        assert!(plan.classification.public_api_changed);
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
        assert!(plan.selected_lanes.is_empty());
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
    fn ci_plan_json_schema_has_required_top_level_fields() -> Result<()> {
        let plan = build_plan(&fixture_lines("rust.txt")?, &[]);
        let value = serde_json::to_value(&plan)?;

        assert_eq!(value.get("schema_version"), Some(&json!(1)));
        assert!(value.get("budget").is_some());
        assert!(value.get("classification").is_some());
        assert!(value.get("selected_lanes").is_some());
        assert!(value.get("skipped_lanes").is_some());
        assert!(value.get("packages").is_some());
        assert!(value.get("risk_packs").is_some());
        assert!(value.get("labels").is_some());
        assert!(value.get("lanes").is_none());
        assert!(value.get("touched").is_none());
        Ok(())
    }

    #[test]
    fn ci_plan_fixture_docs_only() -> Result<()> {
        let plan = build_plan(&fixture_lines("docs.txt")?, &[]);
        let value = serde_json::to_value(&plan)?;
        assert_eq!(value.pointer("/classification/docs_only"), Some(&json!(true)));
        assert_eq!(value.pointer("/classification/tracker_only"), Some(&json!(false)));
        assert_eq!(value.pointer("/classification/rust_inputs_changed"), Some(&json!(false)));
        assert_eq!(value.pointer("/budget/preferred_default_lem"), Some(&json!(25)));
        assert_eq!(value.pointer("/budget/default_limit_lem"), Some(&json!(35)));
        Ok(())
    }

    #[test]
    fn ci_plan_fixture_tracker_only() -> Result<()> {
        let plan = build_plan(&fixture_lines("tracker.txt")?, &[]);
        let value = serde_json::to_value(&plan)?;
        assert_eq!(value.pointer("/classification/docs_only"), Some(&json!(false)));
        assert_eq!(value.pointer("/classification/tracker_only"), Some(&json!(true)));
        assert_eq!(plan.posture, "tracking-only");
        Ok(())
    }

    #[test]
    fn ci_plan_fixture_manifest_and_public_api() -> Result<()> {
        let plan = build_plan(&fixture_lines("manifest.txt")?, &[]);
        let value = serde_json::to_value(&plan)?;
        assert_eq!(
            value.pointer("/classification/manifest_or_toolchain_changed"),
            Some(&json!(true))
        );
        assert_eq!(value.pointer("/classification/public_api_changed"), Some(&json!(true)));
        assert!(
            plan.selected_lanes.iter().any(|lane| lane.id == "compatibility-msrv"),
            "manifest/public API fixture should select MSRV"
        );
        Ok(())
    }

    #[test]
    fn ci_plan_fixture_gpu_and_macos() -> Result<()> {
        let plan = build_plan(&fixture_lines("macos.txt")?, &[]);
        let value = serde_json::to_value(&plan)?;
        assert_eq!(value.pointer("/classification/gpu_changed"), Some(&json!(true)));
        assert_eq!(value.pointer("/classification/macos_changed"), Some(&json!(true)));
        assert!(plan.selected_lanes.iter().any(|lane| lane.id == "gpu-native"));
        assert!(plan.selected_lanes.iter().any(|lane| lane.id == "macos-arm64-clippy"));
        Ok(())
    }

    #[test]
    fn ci_plan_fixture_model_validation_and_labels() -> Result<()> {
        let changed = fixture_lines("model-validation.txt")?;
        let plan = build_plan(&changed, &s(&["coverage", "full-ci"]));
        let value = serde_json::to_value(&plan)?;
        assert_eq!(value.pointer("/classification/model_validation_changed"), Some(&json!(true)));
        assert_eq!(value.pointer("/classification/coverage_requested"), Some(&json!(true)));
        assert_eq!(value.pointer("/classification/full_ci_requested"), Some(&json!(true)));
        assert_eq!(value.pointer("/labels"), Some(&json!(["coverage", "full-ci"])));
        Ok(())
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
