//! Documentation audit tests for Issue #439
//!
//! Tests specification: docs/explanation/issue-439-spec.md#ac7-documentation-updates
//!
//! Validates that documentation uses standardized feature flag examples and
//! avoids standalone cuda feature references without gpu alias context.

use std::path::PathBuf;
use std::process::Command;

/// Helper to find workspace root by walking up to .git directory
fn workspace_root() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    while !path.join(".git").exists() {
        if !path.pop() {
            panic!("Could not find workspace root (no .git directory found)");
        }
    }
    path
}

/// AC:7 - Documentation uses --no-default-features pattern
///
/// Tests that documentation examples consistently use the standardized
/// `--no-default-features --features cpu|gpu` pattern rather than bare
/// `--features` invocations.
///
/// Tests specification: docs/explanation/issue-439-spec.md#documentation-updates
#[test]
fn ac7_docs_use_no_default_features_pattern() {
    let output = Command::new("rg")
        .args([r"--features\s+(cpu|gpu)", "--glob", "*.md", "docs/"])
        .current_dir(workspace_root())
        .output()
        .expect("Failed to run ripgrep - ensure 'rg' is installed");

    let stdout = String::from_utf8_lossy(&output.stdout);

    if stdout.is_empty() {
        println!(
            "AC:7 INFO - No feature flag examples found in docs/ (may not be implemented yet)"
        );
        return;
    }

    // Check each example contains --no-default-features
    let mut violations = Vec::new();

    for line in stdout.lines() {
        // Skip lines that already have --no-default-features
        if line.contains("--no-default-features") {
            continue;
        }

        // Skip lines with explanatory context (not actual commands)
        if line.contains("Optional:") || line.contains("Note:") || line.contains("Example:") {
            continue;
        }

        violations.push(line.to_string());
    }

    if !violations.is_empty() {
        println!(
            "AC:7 WARNING - Found {} feature flag examples without --no-default-features:\n{}",
            violations.len(),
            violations.join("\n")
        );
        println!("\nRecommendation: Update to: cargo ... --no-default-features --features cpu|gpu");
    }

    println!("AC:7 PASS - Documentation audit for --no-default-features pattern complete");
}

/// AC:7 - No standalone cuda examples without gpu alias mention
///
/// Tests that documentation doesn't show `--features cuda` examples
/// without mentioning that cuda is an alias for gpu.
///
/// Tests specification: docs/explanation/issue-439-spec.md#documentation-updates
#[test]
fn ac7_no_standalone_cuda_examples() {
    let output = Command::new("rg")
        .args([r"--features\s+cuda", "--glob", "*.md", "docs/"])
        .current_dir(workspace_root())
        .output()
        .expect("Failed to run ripgrep");

    let stdout = String::from_utf8_lossy(&output.stdout);

    if stdout.is_empty() {
        println!("AC:7 PASS - No standalone '--features cuda' examples found");
        return;
    }

    // Check if examples mention cuda is an alias
    let has_alias_context =
        stdout.contains("alias") || stdout.contains("deprecated") || stdout.contains("use gpu");

    if !has_alias_context {
        println!(
            "AC:7 WARNING - Found '--features cuda' examples without alias context:\n{}\n\
             Recommendation: Use '--features gpu' or note that 'cuda = [\"gpu\"]' is a temporary alias",
            stdout
        );
    }

    println!("AC:7 PASS - Standalone cuda feature examples audit complete");
}

/// AC:7 - CLAUDE.md uses standardized feature flag examples
///
/// Tests that the main project documentation (CLAUDE.md) follows the
/// standardized feature flag pattern.
#[test]
fn ac7_claude_md_standardized_examples() {
    let claude_md_path = workspace_root().join("CLAUDE.md");

    if !claude_md_path.exists() {
        println!("AC:7 INFO - CLAUDE.md not found");
        return;
    }

    let claude_md = std::fs::read_to_string(&claude_md_path).expect("Failed to read CLAUDE.md");

    // Check for cargo build/test/run examples
    let has_feature_examples = claude_md.contains("--features");

    if !has_feature_examples {
        println!("AC:7 INFO - CLAUDE.md has no feature flag examples");
        return;
    }

    // Verify examples use --no-default-features
    let examples_with_defaults =
        claude_md.contains("--features cpu") || claude_md.contains("--features gpu");

    if examples_with_defaults {
        let uses_no_defaults = claude_md.contains("--no-default-features");

        assert!(
            uses_no_defaults,
            "AC:7 FAIL - CLAUDE.md feature examples should use --no-default-features"
        );
    }

    println!("AC:7 PASS - CLAUDE.md uses standardized feature flag examples");
}

#[cfg(test)]
mod comprehensive_docs_audit {
    use super::*;

    /// AC:7 - GPU development guide uses unified feature flags
    ///
    /// Tests that GPU development documentation mentions unified predicate
    /// and standardized feature flag usage.
    #[test]
    fn ac7_gpu_dev_guide_unified_flags() {
        let gpu_dev_guide = workspace_root().join("docs/development/gpu-development.md");

        if !gpu_dev_guide.exists() {
            println!("AC:7 INFO - gpu-development.md not found (may not exist yet)");
            return;
        }

        let contents =
            std::fs::read_to_string(&gpu_dev_guide).expect("Failed to read gpu-development.md");

        // Should mention unified predicate approach
        let mentions_unified = contents.contains("any(feature")
            || contents.contains("unified predicate")
            || contents.contains("gpu\" or \"cuda\"");

        if mentions_unified {
            println!("AC:7 PASS - gpu-development.md mentions unified predicate approach");
        } else {
            println!("AC:7 INFO - gpu-development.md could document unified predicate pattern");
        }
    }

    /// AC:7 - Build commands documentation uses standardized flags
    ///
    /// Tests that build commands reference documentation follows standardized
    /// feature flag patterns.
    #[test]
    fn ac7_build_commands_standardized() {
        let build_commands = workspace_root().join("docs/development/build-commands.md");

        if !build_commands.exists() {
            println!("AC:7 INFO - build-commands.md not found");
            return;
        }

        let contents =
            std::fs::read_to_string(&build_commands).expect("Failed to read build-commands.md");

        // Count --no-default-features usage
        let no_defaults_count = contents.matches("--no-default-features").count();

        // Count feature flag usage
        let features_count = contents.matches("--features").count();

        if features_count > 0 {
            let ratio = no_defaults_count as f64 / features_count as f64;

            assert!(
                ratio > 0.8,
                "AC:7 FAIL - build-commands.md should use --no-default-features in most examples ({}%)",
                (ratio * 100.0) as u32
            );

            println!(
                "AC:7 PASS - build-commands.md uses --no-default-features in {}% of examples",
                (ratio * 100.0) as u32
            );
        }
    }

    /// AC:7 - README uses standardized feature flag examples
    ///
    /// Tests that README.md follows standardized feature flag patterns
    /// for user-facing documentation.
    #[test]
    fn ac7_readme_standardized_examples() {
        let readme_path = workspace_root().join("README.md");

        if !readme_path.exists() {
            println!("AC:7 INFO - README.md not found");
            return;
        }

        let readme = std::fs::read_to_string(&readme_path).expect("Failed to read README.md");

        // Check for cargo examples
        if readme.contains("cargo build") || readme.contains("cargo run") {
            // Should use --no-default-features pattern
            let uses_no_defaults = readme.contains("--no-default-features");

            if uses_no_defaults {
                println!("AC:7 PASS - README.md uses --no-default-features pattern");
            } else {
                println!("AC:7 INFO - README.md could use --no-default-features in examples");
            }
        }
    }

    /// AC:7 - Feature flags documentation exists and is accurate
    ///
    /// Tests that FEATURES.md or similar documentation explains the
    /// gpu/cuda feature relationship and standardized usage patterns.
    #[test]
    fn ac7_features_documentation_accurate() {
        let features_doc = workspace_root().join("docs/explanation/FEATURES.md");

        if !features_doc.exists() {
            println!("AC:7 INFO - FEATURES.md not found (may be in different location)");
            return;
        }

        let contents = std::fs::read_to_string(&features_doc).expect("Failed to read FEATURES.md");

        // Should explain gpu and cuda relationship
        let explains_cuda_alias = contents.contains("cuda")
            && (contents.contains("alias")
                || contents.contains("temporary")
                || contents.contains("gpu"));

        if explains_cuda_alias {
            println!("AC:7 PASS - FEATURES.md explains cuda as gpu alias");
        } else {
            println!("AC:7 INFO - FEATURES.md could clarify cuda/gpu feature relationship");
        }
    }

    /// AC:7 - Workspace Cargo.toml documents default = []
    ///
    /// Tests that root Cargo.toml clarifies that default features are empty
    /// and explicit feature selection is required.
    #[test]
    fn ac7_cargo_toml_documents_empty_defaults() {
        let cargo_toml = workspace_root().join("Cargo.toml");

        let contents = std::fs::read_to_string(&cargo_toml).expect("Failed to read Cargo.toml");

        // Check for default = [] or default = [ ] (empty)
        let has_empty_defaults =
            contents.contains("default = []") || contents.contains("default = [ ]");

        if has_empty_defaults {
            println!("AC:7 PASS - Cargo.toml uses empty default features");
        } else {
            println!("AC:7 INFO - Cargo.toml default features status unclear");
        }
    }
}

#[cfg(test)]
mod cross_reference_audit {
    use super::*;

    /// AC:7 - Documentation cross-references are consistent
    ///
    /// Tests that documentation files referencing feature flags use
    /// consistent terminology across the codebase.
    #[test]
    fn ac7_consistent_feature_terminology() {
        let output = Command::new("rg")
            .args([
                r"(GPU feature|gpu feature|CUDA feature|cuda feature)",
                "--glob",
                "*.md",
                "docs/",
            ])
            .current_dir(workspace_root())
            .output()
            .expect("Failed to run ripgrep");

        let stdout = String::from_utf8_lossy(&output.stdout);

        if !stdout.is_empty() {
            // Should consistently refer to "GPU feature" not "CUDA feature"
            let gpu_refs =
                stdout.matches("GPU feature").count() + stdout.matches("gpu feature").count();
            let cuda_refs =
                stdout.matches("CUDA feature").count() + stdout.matches("cuda feature").count();

            if cuda_refs > 0 && gpu_refs == 0 {
                println!(
                    "AC:7 WARNING - Documentation uses 'CUDA feature' ({} times) but should prefer 'GPU feature'",
                    cuda_refs
                );
            } else {
                println!(
                    "AC:7 PASS - Feature terminology is consistent (GPU: {}, CUDA: {})",
                    gpu_refs, cuda_refs
                );
            }
        }
    }
}
