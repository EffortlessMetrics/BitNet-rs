//! BDD grid compile-coverage checker.
//!
//! Drives `cargo check --workspace` for each supported cell in the curated BDD
//! grid and reports which cells pass or fail.

use anyhow::{Context, Result, bail};
use bitnet_bdd_grid::curated;
use std::process::Command;

/// Map a BDD feature name to the corresponding Cargo feature flag.
/// Returns `None` for BDD-only concepts with no direct Cargo equivalent.
fn bdd_feature_to_cargo(name: &str) -> Option<&'static str> {
    match name {
        "inference" => Some("cpu"),
        "kernels" => Some("cpu"),
        "tokenizers" => Some("cpu"),
        "gpu" => Some("gpu"),
        "cuda" => Some("cuda"),
        // crossval requires external C++ deps (BITNET_CPP_DIR) – skip in CI
        "crossval" => None,
        "server" => Some("full-cli"),
        "fixtures" => Some("fixtures"),
        // reporting is a test-framework feature that may require extra deps – skip in CI
        "reporting" => None,
        _ => None,
    }
}

/// Run `cargo check --workspace` for every supported cell in the curated BDD grid.
///
/// # Arguments
/// * `cpu_only` – skip cells that require GPU/CUDA features (suitable for PR CI).
/// * `verbose`  – print the full `cargo check` stderr on failure.
/// * `dry_run`  – print the commands that would run without executing them.
pub fn run(cpu_only: bool, verbose: bool, dry_run: bool) -> Result<()> {
    let grid = curated();
    let rows = grid.rows();

    struct CellResult {
        label: String,
        features: Vec<String>,
        status: &'static str,
        success: bool,
    }

    let mut results: Vec<CellResult> = Vec::new();
    let mut failed = 0usize;
    let mut skipped = 0usize;
    // Track which feature sets we've already checked to avoid redundant cargo invocations.
    let mut checked_feature_sets: std::collections::HashSet<String> =
        std::collections::HashSet::new();

    for cell in rows {
        let label = format!("{}/{}", cell.scenario, cell.environment);

        // Map required BDD features → cargo features, sort and deduplicate.
        let mut cargo_features: Vec<String> = cell
            .required_features
            .iter()
            .filter_map(|f| bdd_feature_to_cargo(&f.to_string()))
            .map(str::to_owned)
            .collect();
        cargo_features.sort();
        cargo_features.dedup();

        let has_gpu = cargo_features.iter().any(|f| f == "gpu" || f == "cuda");
        if cpu_only && has_gpu {
            results.push(CellResult {
                label,
                features: cargo_features,
                status: "SKIP (GPU)",
                success: true,
            });
            skipped += 1;
            continue;
        }

        // Fall back to "cpu" when no features mapped.
        if cargo_features.is_empty() {
            cargo_features.push("cpu".to_owned());
        }

        let features_str = cargo_features.join(",");

        if dry_run {
            println!(
                "  [DRY-RUN] cargo check --workspace --locked --no-default-features --features {}",
                features_str
            );
            results.push(CellResult {
                label,
                features: cargo_features,
                status: "DRY-RUN",
                success: true,
            });
            continue;
        }

        // Skip if we already ran cargo check for this exact feature set.
        if checked_feature_sets.contains(&features_str) {
            results.push(CellResult {
                label,
                features: cargo_features,
                status: "PASS (cached)",
                success: true,
            });
            continue;
        }
        checked_feature_sets.insert(features_str.clone());

        let output = Command::new("cargo")
            .args([
                "check",
                "--workspace",
                "--locked",
                "--no-default-features",
                "--features",
                &features_str,
            ])
            .output()
            .with_context(|| "Failed to spawn cargo check")?;

        let success = output.status.success();
        if !success {
            failed += 1;
            if verbose {
                eprintln!(
                    "FAIL [{}] features=[{}]:\n{}",
                    label,
                    features_str,
                    String::from_utf8_lossy(&output.stderr)
                );
            }
        }

        results.push(CellResult {
            label,
            features: cargo_features,
            status: if success { "PASS" } else { "FAIL" },
            success,
        });
    }

    let passed = results
        .iter()
        .filter(|r| r.success && r.status != "SKIP (GPU)" && r.status != "DRY-RUN")
        .count();

    println!("BDD Grid Check Results:");
    println!("{}", "─".repeat(58));
    for r in &results {
        let features_display = format!("[{}]", r.features.join(","));
        println!(
            "{:<24} {:<22} {}",
            r.label,
            features_display,
            if r.status == "PASS" {
                format!("✓ {}", r.status)
            } else if r.status.starts_with("SKIP") {
                format!("- {}", r.status)
            } else if r.status == "DRY-RUN" {
                format!("~ {}", r.status)
            } else {
                format!("✗ {}", r.status)
            }
        );
    }
    println!("{}", "─".repeat(58));
    if dry_run {
        let total = results.len();
        println!("Grid check: {total} cells would be checked (dry-run, none executed)");
    } else {
        println!("Grid check: {passed} passed, {failed} failed, {skipped} skipped");
    }

    if failed > 0 {
        bail!("{failed} grid cell(s) failed cargo check");
    }
    Ok(())
}
