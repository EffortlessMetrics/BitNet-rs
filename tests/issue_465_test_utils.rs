//! Shared test utilities for Issue #465: CPU Path Followup
//!
//! This module provides common test helpers to avoid duplication across test suites.
//! Extracted from individual test files to improve maintainability.

use anyhow::{Context, Result};
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};

/// Receipt schema structure for validation
#[allow(dead_code)]
#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct Receipt {
    #[serde(alias = "version")]
    pub schema_version: String,
    pub compute_path: String,
    pub kernels: Vec<String>,
    #[serde(alias = "throughput_tokens_per_sec", alias = "tokens_per_second")]
    pub tokens_per_sec: f64,
    #[serde(default = "default_success")]
    pub success: bool,
}

#[allow(dead_code)]
fn default_success() -> bool {
    true
}

/// Get the workspace root directory
///
/// When running from the tests/ directory, CARGO_MANIFEST_DIR points to tests/Cargo.toml,
/// so we need to go up one level to get to the workspace root.
pub fn workspace_root() -> PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));

    // Check if we're in tests/ subdirectory (has parent with Cargo.toml)
    if let Some(parent) = manifest_dir.parent() {
        let parent_cargo = parent.join("Cargo.toml");
        if parent_cargo.exists() {
            // Check if parent is workspace root by looking for workspace members
            if let Ok(content) = fs::read_to_string(&parent_cargo)
                && content.contains("[workspace]")
            {
                return parent.to_path_buf();
            }
        }
    }

    // Fallback: current manifest dir
    manifest_dir.to_path_buf()
}

/// Configure deterministic test environment
///
/// Sets required environment variables for reproducible inference:
/// - BITNET_DETERMINISTIC=1
/// - RAYON_NUM_THREADS=1
/// - BITNET_SEED=42
///
/// # Safety
/// Uses unsafe `std::env::set_var` as required by Rust 1.90+ for thread-unsafe operations.
pub fn configure_deterministic_env() {
    unsafe {
        std::env::set_var("BITNET_DETERMINISTIC", "1");
        std::env::set_var("RAYON_NUM_THREADS", "1");
        std::env::set_var("BITNET_SEED", "42");
    }
}

/// Find CPU baseline receipt in baselines directory
///
/// Looks for files matching pattern `YYYYMMDD-cpu.json` in `docs/baselines/`.
///
/// Returns the path to the baseline if found, or an error describing why not found.
#[allow(dead_code)]
pub fn find_cpu_baseline() -> Result<PathBuf> {
    let baselines_dir = workspace_root().join("docs/baselines");

    if !baselines_dir.exists() {
        anyhow::bail!(
            "Baselines directory not found: {}. Run `cargo run -p xtask -- benchmark` to generate baseline.",
            baselines_dir.display()
        );
    }

    let entries = fs::read_dir(&baselines_dir).with_context(|| {
        format!("Failed to read baselines directory: {}", baselines_dir.display())
    })?;

    for entry in entries.flatten() {
        let file_name = entry.file_name();
        let name = file_name.to_string_lossy();

        // Match pattern: YYYYMMDD-cpu.json (minimum 13 chars)
        if name.ends_with("-cpu.json") && name.len() >= 13 {
            return Ok(entry.path());
        }
    }

    anyhow::bail!(
        "No CPU baseline found in {}. Expected file matching pattern YYYYMMDD-cpu.json. \
        Run `cargo run -p xtask -- benchmark --model models/*.gguf --tokens 128` to generate baseline.",
        baselines_dir.display()
    )
}

/// Verify receipt schema compliance
///
/// Validates that a receipt file conforms to v1.0.0 schema requirements:
/// - Required fields: version, compute_path, kernels
/// - Valid compute_path: "real" (not "mocked")
/// - Non-empty kernels array
/// - Kernel ID hygiene: non-empty, ≤128 chars, ≤10K total
/// - Performance metrics present
///
/// Supports both old and new schema field names for backward compatibility.
#[allow(dead_code)]
pub fn verify_receipt_schema(path: &Path) -> Result<()> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("Failed to read receipt file: {}", path.display()))?;

    let receipt: Value = serde_json::from_str(&content)
        .with_context(|| format!("Failed to parse receipt JSON: {}", path.display()))?;

    // Validate version field (support both "version" and "schema_version")
    let version = receipt
        .get("schema_version")
        .or_else(|| receipt.get("version"))
        .and_then(|v| v.as_str())
        .with_context(|| format!("Receipt missing version field: {}", path.display()))?;

    if version != "1.0.0" && version != "1.0" {
        anyhow::bail!(
            "Invalid receipt version '{}' in {}: expected '1.0.0' or '1.0'",
            version,
            path.display()
        );
    }

    // Validate compute_path
    let compute_path = receipt
        .get("compute_path")
        .and_then(|v| v.as_str())
        .with_context(|| format!("Receipt missing compute_path field: {}", path.display()))?;

    if compute_path != "real" {
        anyhow::bail!(
            "Invalid compute_path '{}' in {}: expected 'real' for honest compute",
            compute_path,
            path.display()
        );
    }

    // Validate kernels array
    let kernels = receipt.get("kernels").and_then(|v| v.as_array()).with_context(|| {
        format!("Receipt missing kernels field or not an array: {}", path.display())
    })?;

    if kernels.is_empty() {
        anyhow::bail!(
            "Empty kernels array in {}: honest compute requires kernel IDs",
            path.display()
        );
    }

    // Validate kernel ID hygiene
    for (idx, kernel) in kernels.iter().enumerate() {
        let kernel_id = kernel.as_str().with_context(|| {
            format!("Kernel at index {} is not a string in {}", idx, path.display())
        })?;

        if kernel_id.is_empty() {
            anyhow::bail!("Empty kernel ID at index {} in {}", idx, path.display());
        }

        // Check for whitespace-only kernel IDs
        if kernel_id.trim().is_empty() {
            anyhow::bail!("Whitespace-only kernel ID at index {} in {}", idx, path.display());
        }

        if kernel_id.len() > 128 {
            anyhow::bail!(
                "Kernel ID at index {} exceeds 128 characters in {}: '{}'",
                idx,
                path.display(),
                kernel_id
            );
        }
    }

    if kernels.len() > 10_000 {
        anyhow::bail!("Kernel count {} exceeds 10,000 limit in {}", kernels.len(), path.display());
    }

    // Validate performance metrics (support multiple schema variations)
    let tokens_per_sec = if let Some(performance) = receipt.get("performance") {
        // Old schema: performance.tokens_per_sec
        performance
            .as_object()
            .and_then(|p| p.get("tokens_per_sec"))
            .and_then(|t| t.as_f64())
            .with_context(|| {
                format!("performance.tokens_per_sec is not a number in {}", path.display())
            })?
    } else if let Some(tps) = receipt.get("tokens_per_second") {
        // New schema: tokens_per_second
        tps.as_f64()
            .with_context(|| format!("tokens_per_second is not a number in {}", path.display()))?
    } else if let Some(tps) = receipt.get("throughput_tokens_per_sec") {
        // Alternative schema: throughput_tokens_per_sec
        tps.as_f64().with_context(|| {
            format!("throughput_tokens_per_sec is not a number in {}", path.display())
        })?
    } else {
        anyhow::bail!("Receipt missing performance metrics in {}", path.display());
    };

    if tokens_per_sec < 0.0 {
        anyhow::bail!(
            "Invalid tokens_per_sec {} in {}: must be non-negative",
            tokens_per_sec,
            path.display()
        );
    }

    // Validate success flag (optional, defaults to true)
    if let Some(success_field) = receipt.get("success") {
        let success = success_field
            .as_bool()
            .with_context(|| format!("success field is not a boolean in {}", path.display()))?;

        if !success {
            anyhow::bail!("Receipt has success=false in {}", path.display());
        }
    }

    Ok(())
}

/// Verify CPU kernel IDs are present in receipt
///
/// Checks that at least one kernel ID matches expected CPU kernel prefixes:
/// - i2s_* (I2_S quantization kernels)
/// - tl1_* (TL1 table lookup kernels)
/// - tl2_* (TL2 table lookup kernels)
/// - cpu_* (general CPU kernels)
/// - quantized_matmul (CPU matmul kernels)
pub fn has_cpu_kernel_ids(kernels: &[String]) -> bool {
    const CPU_KERNEL_PREFIXES: &[&str] = &["i2s_", "tl1_", "tl2_", "cpu_", "quantized_matmul"];

    kernels
        .iter()
        .any(|kernel_id| CPU_KERNEL_PREFIXES.iter().any(|prefix| kernel_id.contains(prefix)))
}

/// Create a test receipt for validation testing
///
/// Useful for generating test fixtures with specific compute paths and kernel configurations.
pub fn create_test_receipt(compute_path: &str, kernels: Vec<String>) -> Value {
    serde_json::json!({
        "version": "1.0.0",
        "compute_path": compute_path,
        "kernels": kernels,
        "performance": {
            "tokens_per_sec": 15.3
        },
        "success": true
    })
}

/// Verify git tag exists
///
/// Checks if a git tag exists in the repository.
#[allow(dead_code)]
pub fn git_tag_exists(tag_name: &str) -> Result<bool> {
    let output = std::process::Command::new("git")
        .args(["tag", "-l", tag_name])
        .current_dir(workspace_root())
        .output()
        .context("Failed to list git tags")?;

    if output.status.success() {
        let tags = String::from_utf8_lossy(&output.stdout);
        Ok(tags.contains(tag_name))
    } else {
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workspace_root_exists() {
        let root = workspace_root();
        assert!(root.exists(), "Workspace root should exist");
        assert!(root.join("Cargo.toml").exists(), "Workspace should have Cargo.toml");
    }

    #[test]
    fn test_has_cpu_kernel_ids() {
        let cpu_kernels =
            vec!["i2s_cpu_quantized_matmul".to_string(), "tl1_lut_forward".to_string()];
        assert!(has_cpu_kernel_ids(&cpu_kernels));

        let gpu_kernels = vec!["gemm_gpu_fp16".to_string()];
        assert!(!has_cpu_kernel_ids(&gpu_kernels));

        let empty_kernels: Vec<String> = vec![];
        assert!(!has_cpu_kernel_ids(&empty_kernels));
    }

    #[test]
    fn test_create_test_receipt() {
        let receipt = create_test_receipt("real", vec!["i2s_kernel".to_string()]);
        assert_eq!(receipt["compute_path"], "real");
        assert_eq!(receipt["kernels"][0], "i2s_kernel");
    }
}
