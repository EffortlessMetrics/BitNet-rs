//! Build script validation tests for Issue #439
//!
//! Tests specification: docs/explanation/issue-439-spec.md#ac2-build-script-parity
//!
//! Validates that build.rs scripts use unified GPU feature detection by checking
//! both CARGO_FEATURE_GPU and CARGO_FEATURE_CUDA environment variables.

use std::fs;
use std::path::Path;

/// AC:2 - Verify build.rs checks both GPU and CUDA features
///
/// Ensures the bitnet-kernels build script combines both feature flags
/// with logical OR for GPU detection: `CARGO_FEATURE_GPU || CARGO_FEATURE_CUDA`
///
/// Tests specification: docs/explanation/issue-439-spec.md#implementation-approach-2
#[test]
fn ac2_build_script_checks_both_features() {
    let build_rs_path = "/home/steven/code/Rust/BitNet-rs/crates/bitnet-kernels/build.rs";

    if !Path::new(build_rs_path).exists() {
        panic!(
            "AC:2 FAIL - build.rs not found at: {}\n\
             Implementation required: Create build.rs with unified GPU detection",
            build_rs_path
        );
    }

    let build_rs = fs::read_to_string(build_rs_path)
        .unwrap_or_else(|_| panic!("AC:2 FAIL - Failed to read build.rs"));

    // Check for both feature environment variables
    let has_gpu_check = build_rs.contains("CARGO_FEATURE_GPU");
    let has_cuda_check = build_rs.contains("CARGO_FEATURE_CUDA");

    assert!(
        has_gpu_check,
        "AC:2 FAIL - build.rs must check CARGO_FEATURE_GPU environment variable"
    );

    assert!(
        has_cuda_check,
        "AC:2 FAIL - build.rs must check CARGO_FEATURE_CUDA environment variable"
    );

    // Verify logical OR combination pattern
    let has_or_operator = build_rs.contains("||");
    let has_is_some = build_rs.contains("is_some()");

    assert!(
        has_or_operator || build_rs.contains("any(["),
        "AC:2 FAIL - build.rs must combine GPU and CUDA features with logical OR or any()"
    );

    assert!(has_is_some, "AC:2 FAIL - build.rs must use .is_some() to check environment variables");

    // Success: build.rs uses unified GPU detection pattern
    println!("AC:2 PASS - build.rs checks both CARGO_FEATURE_GPU and CARGO_FEATURE_CUDA");
}

/// AC:2 - Verify build.rs emits correct rustc-cfg for GPU builds
///
/// Ensures that when GPU features are detected, build.rs emits the
/// `bitnet_build_gpu` cfg flag for conditional compilation.
///
/// Tests specification: docs/explanation/issue-439-spec.md#build-script-parity
#[test]
fn ac2_build_script_emits_gpu_cfg() {
    let build_rs_path = "/home/steven/code/Rust/BitNet-rs/crates/bitnet-kernels/build.rs";

    if !Path::new(build_rs_path).exists() {
        panic!("AC:2 FAIL - build.rs not found, cannot verify cfg emission");
    }

    let build_rs = fs::read_to_string(build_rs_path)
        .unwrap_or_else(|_| panic!("AC:2 FAIL - Failed to read build.rs"));

    // Check for rustc-cfg emission for GPU builds
    let emits_gpu_cfg = build_rs.contains("cargo:rustc-cfg=bitnet_build_gpu")
        || build_rs.contains(r#"println!("cargo:rustc-cfg=bitnet_build_gpu""#);

    assert!(
        emits_gpu_cfg,
        "AC:2 FAIL - build.rs must emit 'cargo:rustc-cfg=bitnet_build_gpu' when GPU features detected"
    );

    println!("AC:2 PASS - build.rs emits bitnet_build_gpu cfg flag");
}

/// AC:2 - Verify workspace crates use consistent build script pattern
///
/// Checks other workspace crates that may have GPU dependencies use
/// the same unified feature detection pattern.
///
/// Tests specification: docs/explanation/issue-439-spec.md#affected-crates
#[test]
fn ac2_workspace_build_scripts_consistency() {
    let workspace_crates_with_gpu = vec![
        "/home/steven/code/Rust/BitNet-rs/crates/bitnet-quantization/build.rs",
        "/home/steven/code/Rust/BitNet-rs/crates/bitnet-inference/build.rs",
    ];

    for build_rs_path in workspace_crates_with_gpu {
        if !Path::new(build_rs_path).exists() {
            // Optional - not all crates may have build scripts
            continue;
        }

        let build_rs = match fs::read_to_string(build_rs_path) {
            Ok(content) => content,
            Err(_) => continue,
        };

        // If build script checks GPU features, verify unified pattern
        if build_rs.contains("CARGO_FEATURE")
            && (build_rs.contains("GPU") || build_rs.contains("CUDA"))
        {
            let has_unified_check = (build_rs.contains("CARGO_FEATURE_GPU")
                && build_rs.contains("CARGO_FEATURE_CUDA"))
                || build_rs.contains("any([");

            assert!(
                has_unified_check,
                "AC:2 FAIL - {} must use unified GPU feature detection\n\
                 Expected: CARGO_FEATURE_GPU || CARGO_FEATURE_CUDA",
                build_rs_path
            );

            println!("AC:2 PASS - {} uses unified GPU detection", build_rs_path);
        }
    }
}

#[cfg(test)]
mod build_script_edge_cases {
    use super::*;

    /// AC:2 - Verify build.rs handles missing features gracefully
    ///
    /// Ensures build script doesn't fail when no GPU features are enabled,
    /// allowing CPU-only builds to succeed.
    #[test]
    fn ac2_build_script_handles_no_features() {
        let build_rs_path = "/home/steven/code/Rust/BitNet-rs/crates/bitnet-kernels/build.rs";

        if !Path::new(build_rs_path).exists() {
            return;
        }

        let build_rs = fs::read_to_string(build_rs_path).unwrap();

        // Verify build script doesn't panic when features are absent
        // Should use .is_some() checks rather than .unwrap()
        assert!(
            !build_rs.contains("unwrap()") || build_rs.contains("is_some()"),
            "AC:2 FAIL - build.rs must handle missing features gracefully (use .is_some())"
        );
    }

    /// AC:2 - Verify build.rs doesn't emit conflicting cfg flags
    ///
    /// Ensures only one unified GPU cfg is emitted rather than multiple
    /// potentially conflicting flags.
    #[test]
    fn ac2_build_script_single_unified_cfg() {
        let build_rs_path = "/home/steven/code/Rust/BitNet-rs/crates/bitnet-kernels/build.rs";

        if !Path::new(build_rs_path).exists() {
            return;
        }

        let build_rs = fs::read_to_string(build_rs_path).unwrap();

        // Count rustc-cfg emissions related to GPU
        let gpu_cfg_count = build_rs.matches("cargo:rustc-cfg").count();

        // Should emit exactly one unified GPU cfg, not separate cuda/gpu cfgs
        assert!(
            gpu_cfg_count <= 2, // Allow for potential FFI or other cfgs
            "AC:2 FAIL - build.rs should emit minimal cfg flags (found: {})",
            gpu_cfg_count
        );
    }
}
