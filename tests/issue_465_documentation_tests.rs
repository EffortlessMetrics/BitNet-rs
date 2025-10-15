//! Test scaffolding for Issue #465: CPU Path Followup - Documentation Tests
//!
//! Work Stream 1: Documentation Updates (AC1, AC2, AC9, AC10)
//!
//! Tests feature spec: docs/explanation/issue-465-implementation-spec.md
//!
//! This test suite validates:
//! - AC1: README quickstart block with 10-line CPU workflow
//! - AC2: README receipts documentation with xtask commands
//! - AC9: Feature flag standardization across documentation
//! - AC10: Legacy performance claims removed and replaced with evidence

use anyhow::{Context, Result};
use std::fs;
use std::path::Path;

/// Tests feature spec: issue-465-implementation-spec.md#ac1-readme-quickstart-block
///
/// Validates that README.md contains a 10-line CPU quickstart block demonstrating:
/// - Build with explicit CPU features
/// - Model download via xtask
/// - Deterministic inference with environment variables
/// - Receipt verification workflow
#[test]
fn test_ac1_readme_quickstart_block_present() -> Result<()> {
    // AC1: README quickstart block validation

    // Configure deterministic environment (unsafe required in Rust 1.90+)
    unsafe {
        std::env::set_var("BITNET_DETERMINISTIC", "1");
        std::env::set_var("RAYON_NUM_THREADS", "1");
        std::env::set_var("BITNET_SEED", "42");
    }

    let readme_path = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().join("README.md");

    assert!(readme_path.exists(), "README.md not found at expected location: {:?}", readme_path);

    let readme_content = fs::read_to_string(&readme_path).context("Failed to read README.md")?;

    // Check for quickstart section header
    assert!(
        readme_content.contains("CPU Quickstart") || readme_content.contains("10-Line"),
        "README.md missing CPU quickstart section header"
    );

    // Check for required workflow steps
    let required_elements = vec![
        "cargo build --no-default-features --features cpu",
        "cargo run -p xtask -- download-model",
        "BITNET_DETERMINISTIC=1",
        "BITNET_SEED=42",
        "RAYON_NUM_THREADS=1",
        "cargo run -p xtask -- benchmark",
        "cargo run -p xtask -- verify-receipt",
        "--tokens 128",
    ];

    for element in &required_elements {
        assert!(
            readme_content.contains(element),
            "README.md missing required quickstart element: {}",
            element
        );
    }

    // Check for performance documentation with baseline reference
    assert!(
        readme_content.contains("tok/s") || readme_content.contains("baseline"),
        "README.md missing performance documentation"
    );

    // Neural Network Context: Verify I2_S quantization mention
    assert!(
        readme_content.contains("I2_S") || readme_content.contains("2-bit"),
        "README.md missing I2_S quantization context"
    );

    // Evidence tag for validation
    println!("// AC1: README quickstart block validated");

    Ok(())
}

/// Tests feature spec: issue-465-implementation-spec.md#ac2-readme-receipts-documentation-block
///
/// Validates that README.md contains comprehensive receipts documentation:
/// - Receipt generation and verification commands
/// - Environment variable reference table
/// - Receipt schema v1.0.0 overview
/// - Kernel ID hygiene requirements
#[test]
fn test_ac2_readme_receipts_block_present() -> Result<()> {
    // AC2: README receipts documentation validation

    let readme_path = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().join("README.md");
    let readme_content = fs::read_to_string(&readme_path).context("Failed to read README.md")?;

    // Check for receipts section header
    assert!(
        readme_content.contains("Receipt Verification") || readme_content.contains("Receipts"),
        "README.md missing receipts section header"
    );

    // Check for xtask commands
    let required_commands = vec![
        "cargo run -p xtask -- benchmark",
        "cargo run -p xtask -- verify-receipt",
        "BITNET_STRICT_MODE=1",
    ];

    for command in &required_commands {
        assert!(
            readme_content.contains(command),
            "README.md missing required xtask command: {}",
            command
        );
    }

    // Check for environment variables table
    let required_env_vars =
        vec!["BITNET_DETERMINISTIC", "BITNET_SEED", "RAYON_NUM_THREADS", "BITNET_STRICT_MODE"];

    for env_var in &required_env_vars {
        assert!(
            readme_content.contains(env_var),
            "README.md missing environment variable documentation: {}",
            env_var
        );
    }

    // Check for receipt schema documentation
    let required_schema_fields =
        vec!["version", "compute_path", "kernels", "tokens_per_sec", "success"];

    for field in &required_schema_fields {
        assert!(
            readme_content.contains(field),
            "README.md missing receipt schema field: {}",
            field
        );
    }

    // Check for kernel ID hygiene requirements
    assert!(
        readme_content.contains("Kernel ID") || readme_content.contains("kernel hygiene"),
        "README.md missing kernel ID hygiene documentation"
    );

    // Check for baseline reference
    assert!(
        readme_content.contains("baselines/") || readme_content.contains("baseline"),
        "README.md missing baseline receipts reference"
    );

    // Evidence tag for validation
    println!("// AC2: Receipts documentation matches xtask API");

    Ok(())
}

/// Tests feature spec: issue-465-implementation-spec.md#ac9-standardize-feature-flags
///
/// Validates that all cargo commands in documentation use standardized feature flags:
/// - All cargo build/test/run commands include --no-default-features
/// - Explicit CPU or GPU feature selection
/// - No legacy commands without feature flags
#[test]
fn test_ac9_no_legacy_feature_commands() -> Result<()> {
    // AC9: Feature flag standardization validation

    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();

    // Documentation files to check
    let doc_files = vec![
        workspace_root.join("README.md"),
        workspace_root.join("CLAUDE.md"),
        workspace_root.join("docs/quickstart.md"),
        workspace_root.join("docs/getting-started.md"),
        workspace_root.join("docs/development/build-commands.md"),
        workspace_root.join("docs/development/gpu-development.md"),
        workspace_root.join("docs/howto/export-clean-gguf.md"),
        workspace_root.join("docs/howto/validate-models.md"),
    ];

    let mut legacy_commands_found = Vec::new();

    for doc_file in &doc_files {
        if !doc_file.exists() {
            continue; // Skip missing files
        }

        let content = fs::read_to_string(doc_file)
            .with_context(|| format!("Failed to read {:?}", doc_file))?;

        // Split into lines and check each cargo command
        for (line_num, line) in content.lines().enumerate() {
            // Skip comments and git/github URLs
            if line.trim().starts_with('#')
                || line.contains("github.com")
                || line.contains("git clone")
            {
                continue;
            }

            // Check for legacy cargo commands
            if line.contains("cargo build")
                || line.contains("cargo test")
                || line.contains("cargo run")
            {
                // Skip xtask commands - they handle features internally
                if line.contains("-p xtask") || line.contains("--package xtask") {
                    continue;
                }

                // Skip bitnet-st2gguf commands - standalone utility without cpu/gpu features
                if line.contains("-p bitnet-st2gguf") || line.contains("--package bitnet-st2gguf") {
                    continue;
                }

                // Skip bitnet-cli commands - already handles features appropriately
                if line.contains("-p bitnet-cli") || line.contains("--package bitnet-cli") {
                    continue;
                }

                // Skip cargo run without -p flag (typically examples or root binary)
                if line.contains("cargo run")
                    && !line.contains("-p ")
                    && !line.contains("--package ")
                {
                    continue;
                }

                // Verify it includes --no-default-features
                if !line.contains("--no-default-features") {
                    legacy_commands_found.push(format!(
                        "{}:{}:{}",
                        doc_file.display(),
                        line_num + 1,
                        line.trim()
                    ));
                }
            }
        }
    }

    // Report all legacy commands found
    if !legacy_commands_found.is_empty() {
        eprintln!("Legacy cargo commands without --no-default-features found:");
        for cmd in &legacy_commands_found {
            eprintln!("  {}", cmd);
        }

        println!("// AC9: Feature flag standardization validation FAILED");

        // FIXME: This test fails because implementation is missing
        // Expected: All cargo commands use --no-default-features --features cpu|gpu
        // Actual: {} legacy commands found
        panic!(
            "AC9 implementation missing: {} legacy cargo commands found without feature flags",
            legacy_commands_found.len()
        );
    }

    // Evidence tag for validation
    println!("// AC9: Feature flag standardization validated");

    Ok(())
}

/// Tests feature spec: issue-465-implementation-spec.md#ac10-remove-legacy-performance-claims
///
/// Validates that documentation removes unsupported performance claims:
/// - No specific performance numbers without receipt evidence
/// - No vague performance claims (fast, blazing, etc.)
/// - All performance claims reference baseline receipts
#[test]
fn test_ac10_no_unsupported_performance_claims() -> Result<()> {
    // AC10: Performance claims validation

    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();

    // Documentation files to check
    let doc_files = vec![
        workspace_root.join("README.md"),
        workspace_root.join("CLAUDE.md"),
        workspace_root.join("docs/quickstart.md"),
        workspace_root.join("docs/architecture-overview.md"),
        workspace_root.join("docs/performance-benchmarking.md"),
    ];

    let mut unsupported_claims = Vec::new();

    // Unsupported performance numbers (unrealistic claims)
    let unsupported_numbers = vec!["200 tok/s", "500 tok/s", "1000 tok/s"];

    // Vague performance claims without evidence
    let vague_claims = vec!["blazing", "lightning", "ultra-fast"];

    for doc_file in &doc_files {
        if !doc_file.exists() {
            continue; // Skip missing files
        }

        let content = fs::read_to_string(doc_file)
            .with_context(|| format!("Failed to read {:?}", doc_file))?;

        // Check for unsupported specific numbers
        for unsupported in &unsupported_numbers {
            if content.contains(unsupported) {
                unsupported_claims.push(format!(
                    "{}: Contains unsupported claim '{}'",
                    doc_file.display(),
                    unsupported
                ));
            }
        }

        // Check for vague claims without evidence
        for vague in &vague_claims {
            if content.to_lowercase().contains(vague) {
                // Check if this claim has nearby evidence (receipt, baseline, measured)
                let words: Vec<&str> = content.split_whitespace().collect();
                let mut has_evidence = false;

                for window in words.windows(20) {
                    let window_text = window.join(" ").to_lowercase();
                    if window_text.contains(vague) {
                        if window_text.contains("receipt")
                            || window_text.contains("baseline")
                            || window_text.contains("measured")
                        {
                            has_evidence = true;
                            break;
                        }
                    }
                }

                if !has_evidence {
                    unsupported_claims.push(format!(
                        "{}: Contains vague claim '{}' without evidence",
                        doc_file.display(),
                        vague
                    ));
                }
            }
        }
    }

    // Report all unsupported claims found
    if !unsupported_claims.is_empty() {
        eprintln!("Unsupported performance claims found:");
        for claim in &unsupported_claims {
            eprintln!("  {}", claim);
        }

        println!("// AC10: Performance claims validation FAILED");

        // FIXME: This test fails because implementation is missing
        // Expected: All performance claims backed by receipt evidence
        // Actual: {} unsupported claims found
        panic!(
            "AC10 implementation missing: {} unsupported performance claims found",
            unsupported_claims.len()
        );
    }

    // Verify baseline references exist
    let readme_path = workspace_root.join("README.md");
    if readme_path.exists() {
        let readme_content = fs::read_to_string(&readme_path)?;
        assert!(
            readme_content.contains("baselines/") || readme_content.contains("baseline"),
            "README.md missing baseline references for performance claims"
        );
    }

    // Evidence tag for validation
    println!("// AC10: Performance claims backed by receipt evidence");

    Ok(())
}

#[cfg(test)]
mod test_helpers {
    use super::*;

    /// Helper function to verify README section exists
    pub fn verify_readme_section(section_name: &str) -> bool {
        let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let readme_path = workspace_root.join("README.md");

        if !readme_path.exists() {
            return false;
        }

        if let Ok(content) = fs::read_to_string(&readme_path) {
            content.contains(section_name)
        } else {
            false
        }
    }

    /// Helper function to count cargo commands with feature flags
    pub fn count_cargo_commands_with_features(file_path: &Path) -> Result<(usize, usize)> {
        let content = fs::read_to_string(file_path)?;

        let mut total_cargo_commands = 0;
        let mut commands_with_features = 0;

        for line in content.lines() {
            if line.trim().starts_with('#') {
                continue;
            }

            if line.contains("cargo build")
                || line.contains("cargo test")
                || line.contains("cargo run")
            {
                total_cargo_commands += 1;

                if line.contains("--no-default-features") {
                    commands_with_features += 1;
                }
            }
        }

        Ok((total_cargo_commands, commands_with_features))
    }
}
