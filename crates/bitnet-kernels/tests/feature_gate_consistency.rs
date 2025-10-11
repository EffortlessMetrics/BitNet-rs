//! Feature gate consistency tests for Issue #439
//!
//! Tests specification: docs/explanation/issue-439-spec.md#ac1-kernel-gate-unification
//!
//! Validates that GPU code uses unified predicate `#[cfg(any(feature="gpu", feature="cuda"))]`
//! rather than standalone cuda-only gates to prevent compile-time drift.

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

/// AC:1 - Verify no standalone cuda feature gates exist in bitnet-kernels
///
/// This test uses ripgrep to search for standalone `#[cfg(any(feature = "gpu", feature = "cuda"))]`
/// that are not part of the unified `any(feature = "gpu", feature = "cuda")` predicate.
///
/// Tests specification: docs/explanation/issue-439-spec.md#implementation-approach-1
#[test]
fn ac1_no_standalone_cuda_gates_in_kernels() {
    // Search for standalone cuda gates without "any(feature"
    let output = Command::new("rg")
        .args([
            r#"#\[cfg\(feature\s*=\s*"cuda"\)\]"#,
            "--glob",
            "*.rs",
            "--glob",
            "!Cargo.lock",
            "crates/bitnet-kernels/src/",
        ])
        .current_dir(workspace_root())
        .output()
        .expect("Failed to run ripgrep - ensure 'rg' is installed");

    let stdout = String::from_utf8_lossy(&output.stdout);

    // If we find matches, check they all contain "any(feature" pattern
    if !stdout.is_empty() {
        for line in stdout.lines() {
            assert!(
                line.contains("any(feature"),
                "Found standalone cuda feature gate (AC1):\n{}\n\nExpected: #[cfg(any(feature=\"gpu\", feature=\"cuda\"))]",
                line
            );
        }
    }

    // Success: Either no matches, or all matches use unified predicate
}

/// AC:1 - Verify all GPU-specific modules use unified predicate
///
/// Tests that critical GPU modules like gpu/validation.rs use the unified
/// feature gate consistently.
///
/// Tests specification: docs/explanation/issue-439-spec.md#ac1
#[test]
fn ac1_gpu_validation_module_uses_unified_predicate() {
    let validation_path =
        "/home/steven/code/Rust/BitNet-rs/crates/bitnet-kernels/src/gpu/validation.rs";

    // Check if validation.rs exists (may not be implemented yet)
    if !std::path::Path::new(validation_path).exists() {
        // Module not yet implemented - test will pass once created with correct gates
        println!(
            "Note: gpu/validation.rs not yet created - this test validates it uses unified gates"
        );
        return;
    }

    let validation_rs =
        std::fs::read_to_string(validation_path).expect("Failed to read gpu/validation.rs");

    let unified_pattern = r#"#[cfg(any(feature = "gpu", feature = "cuda"))]"#;

    // If file has cfg gates, ensure they use unified predicate
    if validation_rs.contains("#[cfg(feature") {
        assert!(
            validation_rs.contains(unified_pattern)
                || validation_rs.contains(r#"any(feature = "gpu", feature = "cuda")"#),
            "gpu/validation.rs must use unified GPU predicate (AC1)"
        );
    }
}

/// AC:1 - Verify workspace-wide unified gates (comprehensive search)
///
/// Searches all workspace crates for standalone cuda gates to ensure
/// consistent GPU feature detection across all components.
///
/// Tests specification: docs/explanation/issue-439-spec.md#affected-crates
#[test]
fn ac1_workspace_wide_cuda_gate_consistency() {
    let output = Command::new("rg")
        .args([
            r#"#\[cfg\(feature\s*=\s*"cuda"\)\]"#,
            "--glob",
            "*.rs",
            "--glob",
            "!Cargo.lock",
            "crates/",
        ])
        .current_dir(workspace_root())
        .output()
        .expect("Failed to run ripgrep - ensure 'rg' is installed");

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Count violations (standalone cuda without any())
    let mut violations = Vec::new();

    for line in stdout.lines() {
        if !line.contains("any(feature") {
            violations.push(line.to_string());
        }
    }

    assert!(
        violations.is_empty(),
        "Found {} standalone cuda feature gates (AC1):\n{}\n\nAll GPU gates must use: #[cfg(any(feature=\"gpu\", feature=\"cuda\"))]",
        violations.len(),
        violations.join("\n")
    );
}

/// AC:1 - Verify build.rs uses unified GPU detection
///
/// Ensures build scripts check both CARGO_FEATURE_GPU and CARGO_FEATURE_CUDA
/// environment variables for GPU feature detection.
///
/// Tests specification: docs/explanation/issue-439-spec.md#build-script-parity
#[test]
fn ac1_build_scripts_check_both_gpu_features() {
    let build_rs_path = "/home/steven/code/Rust/BitNet-rs/crates/bitnet-kernels/build.rs";

    if !std::path::Path::new(build_rs_path).exists() {
        // build.rs not yet created - test will validate it once implemented
        println!(
            "Note: bitnet-kernels/build.rs not yet created - AC2 will validate unified detection"
        );
        return;
    }

    let build_rs = std::fs::read_to_string(build_rs_path).expect("Failed to read build.rs");

    // Check for unified GPU detection pattern
    let has_unified_detection = build_rs.contains("CARGO_FEATURE_GPU")
        && build_rs.contains("CARGO_FEATURE_CUDA")
        && (build_rs.contains("||") || build_rs.contains("is_some()"));

    assert!(
        has_unified_detection,
        "build.rs must check both CARGO_FEATURE_GPU and CARGO_FEATURE_CUDA (AC1)\n\
         Expected pattern: CARGO_FEATURE_GPU.is_some() || CARGO_FEATURE_CUDA.is_some()"
    );
}

#[cfg(test)]
mod gpu_runtime_checks {
    use super::*;

    /// AC:1 - Verify cfg! macro uses unified predicate in runtime checks
    ///
    /// Tests that runtime GPU capability checks use `cfg!(any(feature="gpu", feature="cuda"))`
    /// consistently across the codebase.
    #[test]
    fn ac1_cfg_macro_uses_unified_predicate() {
        let output = Command::new("rg")
            .args([
                r#"cfg!\(feature\s*=\s*"cuda"\)"#,
                "--glob",
                "*.rs",
                "--glob",
                "!Cargo.lock",
                "crates/",
            ])
            .current_dir(workspace_root())
            .output()
            .expect("Failed to run ripgrep");

        let stdout = String::from_utf8_lossy(&output.stdout);

        // Check all cfg! runtime checks use unified predicate
        for line in stdout.lines() {
            assert!(
                line.contains("any(feature") || line.is_empty(),
                "Found standalone cfg!(feature=\"cuda\") runtime check (AC1):\n{}\n\
                 Expected: cfg!(any(feature=\"gpu\", feature=\"cuda\"))",
                line
            );
        }
    }
}
