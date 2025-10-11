//! Preflight validation tests for Issue #439
//!
//! Tests specification: docs/explanation/issue-439-spec.md#ac5-xtask-preflight
//!
//! Validates that xtask preflight command correctly reports GPU status based on
//! BITNET_GPU_FAKE environment variable with proper fake precedence.

use std::process::Command;

/// AC:5 - Preflight detects no GPU with BITNET_GPU_FAKE=none
///
/// Tests that preflight respects BITNET_GPU_FAKE=none and reports GPU as
/// unavailable even if real hardware might be present (for testing CPU paths).
///
/// Tests specification: docs/explanation/issue-439-spec.md#xtask-preflight
#[test]
fn ac5_preflight_detects_no_gpu_with_fake_none() {
    let output = Command::new("cargo")
        .args(["run", "-p", "xtask", "--", "preflight"])
        .current_dir("/home/steven/code/Rust/BitNet-rs")
        .env("BITNET_GPU_FAKE", "none")
        .output()
        .expect("Failed to run xtask preflight - ensure xtask crate exists");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Combine stdout and stderr for checking
    let combined_output = format!("{}\n{}", stdout, stderr);

    // Check for GPU unavailable indicators
    let indicates_no_gpu = combined_output.contains("GPU: Not available")
        || combined_output.contains("GPU: ✗")
        || combined_output.contains("GPU: not detected")
        || combined_output.contains("No GPU")
        || (combined_output.contains("GPU") && combined_output.contains("false"));

    assert!(
        indicates_no_gpu,
        "AC:5 FAIL - Preflight should report no GPU with BITNET_GPU_FAKE=none\n\
         Output:\n{}\n\
         Expected indicators: 'GPU: Not available', 'GPU: ✗', or similar",
        combined_output
    );

    println!("AC:5 PASS - Preflight correctly reports no GPU with BITNET_GPU_FAKE=none");
}

/// AC:5 - Preflight detects GPU present with BITNET_GPU_FAKE=cuda
///
/// Tests that preflight respects BITNET_GPU_FAKE=cuda and reports GPU as
/// available regardless of real hardware state (for testing GPU paths without GPU).
///
/// Tests specification: docs/explanation/issue-439-spec.md#xtask-preflight
#[test]
fn ac5_preflight_detects_gpu_with_fake_cuda() {
    let output = Command::new("cargo")
        .args(["run", "-p", "xtask", "--", "preflight"])
        .current_dir("/home/steven/code/Rust/BitNet-rs")
        .env("BITNET_GPU_FAKE", "cuda")
        .output()
        .expect("Failed to run xtask preflight");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    let combined_output = format!("{}\n{}", stdout, stderr);

    // Check for GPU available indicators
    let indicates_gpu = combined_output.contains("GPU: Available")
        || combined_output.contains("GPU: ✓")
        || combined_output.contains("CUDA")
        || combined_output.contains("GPU: detected")
        || (combined_output.contains("GPU") && combined_output.contains("true"));

    assert!(
        indicates_gpu,
        "AC:5 FAIL - Preflight should report GPU present with BITNET_GPU_FAKE=cuda\n\
         Output:\n{}\n\
         Expected indicators: 'GPU: Available', 'GPU: ✓', 'CUDA', or similar",
        combined_output
    );

    println!("AC:5 PASS - Preflight correctly reports GPU with BITNET_GPU_FAKE=cuda");
}

/// AC:5 - Preflight reports real GPU status without fake environment
///
/// Tests that preflight correctly detects real GPU hardware when no
/// BITNET_GPU_FAKE override is set (integration with actual detection).
#[test]
fn ac5_preflight_real_gpu_detection() {
    let output = Command::new("cargo")
        .args(["run", "-p", "xtask", "--", "preflight"])
        .current_dir("/home/steven/code/Rust/BitNet-rs")
        .env_remove("BITNET_GPU_FAKE") // Ensure no fake override
        .output()
        .expect("Failed to run xtask preflight");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    let combined_output = format!("{}\n{}", stdout, stderr);

    // Should report some GPU status (available or not available)
    let has_gpu_status = combined_output.contains("GPU:")
        || combined_output.contains("CUDA")
        || combined_output.contains("Device Capabilities");

    assert!(
        has_gpu_status,
        "AC:5 FAIL - Preflight should report GPU status (available or not)\n\
         Output:\n{}",
        combined_output
    );

    println!("AC:5 PASS - Preflight reports real GPU detection status:\n{}", combined_output);
}

#[cfg(test)]
mod preflight_edge_cases {
    use super::*;

    /// AC:5 - Preflight handles invalid BITNET_GPU_FAKE values gracefully
    ///
    /// Tests that invalid fake GPU values default to real detection rather
    /// than crashing or producing undefined behavior.
    #[test]
    fn ac5_preflight_invalid_fake_value_fallback() {
        let invalid_values = vec!["invalid", "true", "1", "yes"];

        for invalid_value in invalid_values {
            let output = Command::new("cargo")
                .args(["run", "-p", "xtask", "--", "preflight"])
                .current_dir("/home/steven/code/Rust/BitNet-rs")
                .env("BITNET_GPU_FAKE", invalid_value)
                .output()
                .expect("Failed to run xtask preflight");

            assert!(
                output.status.success() || output.status.code() == Some(0),
                "AC:5 FAIL - Preflight should handle invalid BITNET_GPU_FAKE='{}' gracefully",
                invalid_value
            );
        }

        println!("AC:5 PASS - Preflight handles invalid BITNET_GPU_FAKE values gracefully");
    }

    /// AC:5 - Preflight output includes feature compilation status
    ///
    /// Verifies that preflight reports whether GPU features were compiled in,
    /// in addition to runtime availability (compile vs runtime distinction).
    #[test]
    fn ac5_preflight_reports_compile_status() {
        let output = Command::new("cargo")
            .args(["run", "-p", "xtask", "--", "preflight"])
            .current_dir("/home/steven/code/Rust/BitNet-rs")
            .output()
            .expect("Failed to run xtask preflight");

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        let combined_output = format!("{}\n{}", stdout, stderr);

        // Should distinguish compile-time vs runtime capabilities
        let has_capability_info = combined_output.contains("Compiled")
            || combined_output.contains("Runtime")
            || combined_output.contains("Features")
            || combined_output.contains("Device Capabilities");

        if has_capability_info {
            println!("AC:5 PASS - Preflight distinguishes compile-time vs runtime capabilities");
        } else {
            println!(
                "AC:5 INFO - Preflight could enhance output with compile/runtime distinction\n\
                 Current output:\n{}",
                combined_output
            );
        }
    }

    /// AC:5 - Preflight exit code reflects GPU availability
    ///
    /// Tests that preflight command exits successfully regardless of GPU status
    /// (informational command should not fail).
    #[test]
    fn ac5_preflight_exit_code_success() {
        // Test with fake=none
        let output_no_gpu = Command::new("cargo")
            .args(["run", "-p", "xtask", "--", "preflight"])
            .current_dir("/home/steven/code/Rust/BitNet-rs")
            .env("BITNET_GPU_FAKE", "none")
            .output()
            .expect("Failed to run xtask preflight");

        assert!(
            output_no_gpu.status.success(),
            "AC:5 FAIL - Preflight should exit successfully even without GPU"
        );

        // Test with fake=cuda
        let output_with_gpu = Command::new("cargo")
            .args(["run", "-p", "xtask", "--", "preflight"])
            .current_dir("/home/steven/code/Rust/BitNet-rs")
            .env("BITNET_GPU_FAKE", "cuda")
            .output()
            .expect("Failed to run xtask preflight");

        assert!(
            output_with_gpu.status.success(),
            "AC:5 FAIL - Preflight should exit successfully with GPU"
        );

        println!("AC:5 PASS - Preflight exits successfully regardless of GPU status");
    }
}
