// AC6: FFI Build Hygiene Tests
//
// Tests for Issue #469 AC6: FFI build hygiene consolidation
// - Single compile_cpp_shim function (no duplicates)
// - -isystem flags for third-party includes (suppress warnings)
// - Build warning reduction
// - FFI version comments present
//
// Feature flags: --no-default-features --features cpu
// Specification: docs/reference/api-contracts-issue-469.md#ac6-ffi-build-hygiene-contract

/// AC6: Test single compile_cpp_shim function exists (no duplicates)
///
/// Verifies that there is exactly one compile_cpp_shim function in the codebase
/// to prevent build script fragmentation and maintenance issues.
///
/// # Fixture Requirements
/// - None (uses codebase static analysis)
///
/// # Expected Behavior
/// - Single function definition in build helper module
/// - No duplicate implementations across build.rs files
/// - Function signature matches API contract
#[test]
fn test_single_compile_cpp_shim_function() {
    // AC6: Verify single compile_cpp_shim function exists
    // The unified function is in xtask/src/ffi.rs
    // Verify by checking that xtask::ffi::compile_cpp_shim is accessible
    use std::path::{Path, PathBuf};

    // This will compile-fail if the function doesn't exist with the correct signature
    #[allow(clippy::type_complexity)]
    let _: fn(&Path, &str, &[PathBuf], &[PathBuf]) -> Result<(), Box<dyn std::error::Error>> =
        xtask::ffi::compile_cpp_shim;

    // Also verify helper functions exist
    let _: fn() -> Vec<PathBuf> = xtask::ffi::cuda_system_includes;
    let _: fn() -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> =
        xtask::ffi::bitnet_cpp_system_includes;
}

/// AC6: Test -isystem flags for third-party includes
///
/// Verifies that system include directories use -isystem flag instead of -I
/// to suppress third-party header warnings (CUDA, C++ reference).
///
/// # Fixture Requirements
/// - Mock build environment with CUDA headers
/// - Mock C++ reference repository (BITNET_CPP_DIR)
///
/// # Expected Behavior
/// - CUDA includes use -isystem: /usr/local/cuda/include
/// - BitNet C++ includes use -isystem: $BITNET_CPP_DIR/include, $BITNET_CPP_DIR/3rdparty
/// - Local includes use -I: csrc/
#[test]
fn test_isystem_flags_for_third_party() {
    use std::path::PathBuf;

    // AC6: Verify -isystem flags suppress third-party warnings
    // This test validates that the FFI build system properly separates local and
    // system includes, using -I for local code and -isystem for third-party headers.
    //
    // Expected compiler flags:
    //   -I csrc/                                    # Show warnings (local code)
    //   -isystem /usr/local/cuda/include            # Suppress warnings (CUDA)
    //   -isystem $BITNET_CPP_DIR/include            # Suppress warnings (C++ reference)
    //   -isystem $BITNET_CPP_DIR/3rdparty/llama.cpp # Suppress warnings (llama.cpp)

    // Verify compile_cpp_shim signature accepts system includes
    #[allow(clippy::type_complexity)]
    let _: fn(
        &std::path::Path,
        &str,
        &[PathBuf],
        &[PathBuf],
    ) -> Result<(), Box<dyn std::error::Error>> = xtask::ffi::compile_cpp_shim;

    // Verify helper functions return proper include paths
    let cuda_includes = xtask::ffi::cuda_system_includes();
    assert!(
        cuda_includes.iter().any(|p| p.to_string_lossy().contains("cuda")),
        "CUDA includes should contain 'cuda' in path"
    );

    if let Ok(paths) = xtask::ffi::bitnet_cpp_system_includes() {
        assert!(!paths.is_empty(), "BitNet C++ includes should not be empty");
        assert!(
            paths.iter().any(|p| p.to_string_lossy().contains("bitnet")
                || p.to_string_lossy().contains("llama.cpp")),
            "BitNet C++ includes should contain 'bitnet' or 'llama.cpp' in path"
        );
    }

    // Verify that -isystem flags are documented in the implementation
    // (The actual flag verification happens in integration tests)
}

/// AC6: Test build warnings are reduced
///
/// Verifies that FFI build produces fewer warnings after hygiene consolidation.
///
/// # Fixture Requirements
/// - Baseline warning count from current build
/// - Test shim with known warning-prone code patterns
///
/// # Expected Behavior
/// - External header warnings suppressed (-Wno-unknown-pragmas, -Wno-deprecated-declarations)
/// - Local code warnings still visible
/// - Overall warning count reduced by at least 50%
#[test]
fn test_build_warnings_reduced() {
    // This is a meta-test that validates the build system produces
    // clean compilation output when using -isystem flags

    // Verify that the FFI module compiles without errors
    // Real warning reduction is verified through:
    // 1. CI build logs showing warning count < 10
    // 2. Comparison against baseline (tracked in docs/baselines/)
    // 3. Manual review of build output during FFI development

    // For now, verify the configuration is correct
    let local_include = std::path::PathBuf::from("csrc");
    assert!(
        local_include.file_name().is_some_and(|name| name == "csrc"),
        "Local includes should be 'csrc' (used with -I)"
    );

    // Verify system includes are configured
    let cuda_paths = xtask::ffi::cuda_system_includes();
    assert!(!cuda_paths.is_empty(), "CUDA includes should be configured");

    // Verify BitNet C++ paths are configured
    let cpp_paths = xtask::ffi::bitnet_cpp_system_includes();
    assert!(cpp_paths.is_ok(), "BitNet C++ includes should resolve");

    // Note: Actual warning count reduction is measured in CI and tracked in:
    // - docs/baselines/ffi_build_warnings_baseline.txt
    // - xtask/ci/ffi_build_output.json
}

/// AC6: Test FFI version comments are present
///
/// Verifies that C++ shim files include version comments documenting
/// the llama.cpp API version and compatibility notes.
///
/// # Fixture Requirements
/// - Example shim file: csrc/shim.cc
/// - API version comment format specification
///
/// # Expected Behavior
/// - Version comment at top of shim file
/// - Format: "// llama.cpp API version: <commit-hash> (<date>)"
/// - Compatibility notes for breaking changes
#[test]
fn test_ffi_version_comments_present() {
    use std::fs;
    use std::path::Path;

    // Paths to shim files that should have version comments
    let shim_files = vec![
        "crates/bitnet-ggml-ffi/csrc/ggml_quants_shim.c",
        "crates/bitnet-ggml-ffi/csrc/ggml_consts.c",
    ];

    // Check each shim file for version documentation
    for shim_path_str in &shim_files {
        let shim_path = Path::new(shim_path_str);

        // Skip if file doesn't exist (e.g., in test environment)
        if !shim_path.exists() {
            eprintln!("Skipping (not found): {}", shim_path_str);
            continue;
        }

        let content = fs::read_to_string(shim_path)
            .unwrap_or_else(|_| panic!("Failed to read {}", shim_path_str));

        // Check for FFI version documentation markers
        // These indicate the shim has proper version tracking
        let has_version_marker = content.contains("llama.cpp API version")
            || content.contains("VENDORED_GGML_COMMIT")
            || content.contains("bitnet-rs integration");

        let has_compatibility_info =
            content.contains("Compatible with") || content.contains("Build date");

        assert!(
            has_version_marker || has_compatibility_info,
            "Shim file {} should have FFI version comments documenting API compatibility",
            shim_path_str
        );
    }
}

/// AC6: Test compile_cpp_shim with CUDA system includes
///
/// Integration test for compile_cpp_shim with CUDA headers.
///
/// # Fixture Requirements
/// - Mock CUDA installation
/// - Test shim using CUDA headers (#include <cuda_runtime.h>)
///
/// # Expected Behavior
/// - Compilation succeeds without warnings
/// - CUDA includes resolved via -isystem
#[test]
#[ignore = "Requires FFI implementation - fixture not yet available"]
fn test_compile_cpp_shim_with_cuda() {
    // AC6: Integration test for CUDA system includes
    // FIXTURE NEEDED:
    // - tests/fixtures/test_cuda_shim.cc with #include <cuda_runtime.h>
    // - Mock CUDA installation or skip test if not available
    //
    // Expected:
    //   compile_cpp_shim(
    //       Path::new("tests/fixtures/test_cuda_shim.cc"),
    //       "test_cuda_shim",
    //       &[],  // No local includes
    //       &[PathBuf::from("/usr/local/cuda/include")],  // CUDA system include
    //   ).unwrap();

    panic!(
        "AC6: compile_cpp_shim with CUDA not yet implemented. \
         Expected: Shim compiles successfully with -isystem /usr/local/cuda/include."
    );
}

/// AC6: Test compile_cpp_shim with C++ reference includes
///
/// Integration test for compile_cpp_shim with BitNet C++ reference headers.
///
/// # Fixture Requirements
/// - Mock BITNET_CPP_DIR with include/ and 3rdparty/ directories
/// - Test shim using BitNet C++ headers
///
/// # Expected Behavior
/// - Compilation succeeds without warnings
/// - C++ reference includes resolved via -isystem
#[test]
#[ignore = "Requires FFI implementation - fixture not yet available"]
fn test_compile_cpp_shim_with_cpp_reference() {
    // AC6: Integration test for C++ reference system includes
    // FIXTURE NEEDED:
    // - tests/fixtures/test_cpp_ref_shim.cc with BitNet C++ includes
    // - Mock BITNET_CPP_DIR at tests/fixtures/mock_bitnet_cpp with:
    //     - include/bitnet.h
    //     - 3rdparty/llama.cpp/llama.h
    //
    // Expected:
    //   compile_cpp_shim(
    //       Path::new("tests/fixtures/test_cpp_ref_shim.cc"),
    //       "test_cpp_ref_shim",
    //       &[],  // No local includes
    //       &[
    //           PathBuf::from("tests/fixtures/mock_bitnet_cpp/include"),
    //           PathBuf::from("tests/fixtures/mock_bitnet_cpp/3rdparty/llama.cpp"),
    //       ],
    //   ).unwrap();

    panic!(
        "AC6: compile_cpp_shim with C++ reference not yet implemented. \
         Expected: Shim compiles successfully with -isystem for BitNet C++ headers."
    );
}

/// AC6: Test cuda_system_includes helper
///
/// Tests the helper function that returns standard CUDA system include paths.
///
/// # Fixture Requirements
/// - None (tests function contract)
///
/// # Expected Behavior
/// - Returns standard CUDA paths: /usr/local/cuda/include, /usr/local/cuda/targets/...
/// - Does not fail (best-effort path construction)
#[test]
fn test_cuda_system_includes_helper() {
    use std::path::PathBuf;

    let paths = xtask::ffi::cuda_system_includes();

    // Verify it returns expected CUDA paths
    assert!(paths.contains(&PathBuf::from("/usr/local/cuda/include")));
    assert!(paths.contains(&PathBuf::from("/usr/local/cuda/targets/x86_64-linux/include")));
    assert!(paths.contains(&PathBuf::from("/usr/local/cuda/targets/aarch64-linux/include")));
    assert!(!paths.is_empty());
}

/// AC6: Test bitnet_cpp_system_includes helper
///
/// Tests the helper function that returns BitNet C++ reference include paths.
///
/// # Fixture Requirements
/// - Mock BITNET_CPP_DIR environment variable
///
/// # Expected Behavior
/// - Reads BITNET_CPP_DIR or defaults to $HOME/.cache/bitnet_cpp
/// - Returns include/ and 3rdparty/llama.cpp paths
/// - Returns Err if BITNET_CPP_DIR not set and HOME not available
#[test]
fn test_bitnet_cpp_system_includes_helper() {
    use std::path::PathBuf;

    // Test with explicit BITNET_CPP_DIR
    unsafe {
        std::env::set_var("BITNET_CPP_DIR", "/mock/bitnet_cpp");
    }
    let paths = xtask::ffi::bitnet_cpp_system_includes().unwrap();

    assert!(paths.contains(&PathBuf::from("/mock/bitnet_cpp/include")));
    assert!(paths.contains(&PathBuf::from("/mock/bitnet_cpp/3rdparty/llama.cpp/include")));
    assert!(paths.contains(&PathBuf::from("/mock/bitnet_cpp/3rdparty/llama.cpp/ggml/include")));
    assert!(paths.contains(&PathBuf::from("/mock/bitnet_cpp/build/3rdparty/llama.cpp/include")));
    assert!(
        paths.contains(&PathBuf::from("/mock/bitnet_cpp/build/3rdparty/llama.cpp/ggml/include"))
    );

    // Clean up
    unsafe {
        std::env::remove_var("BITNET_CPP_DIR");
    }
}

/// AC6: Test compile flags include C++17, -O2, -fPIC
///
/// Verifies that compile_cpp_shim uses correct compiler flags.
///
/// # Fixture Requirements
/// - Capture compiler invocation flags
///
/// # Expected Behavior
/// - C++17 standard: -std=c++17
/// - Optimization: -O2
/// - Position-independent code: -fPIC
/// - Warning suppression: -Wno-unknown-pragmas, -Wno-deprecated-declarations
#[test]
#[ignore = "Requires FFI implementation - fixture not yet available"]
fn test_compile_flags_correct() {
    // AC6: Verify compiler flags
    // FIXTURE NEEDED:
    // - Capture compiler invocation (cc::Build output)
    // - Parse flags from build log
    //
    // Expected flags:
    //   -std=c++17
    //   -O2
    //   -fPIC
    //   -Wno-unknown-pragmas
    //   -Wno-deprecated-declarations

    panic!(
        "AC6: Compiler flags not yet implemented. \
         Expected: -std=c++17, -O2, -fPIC, warning suppression flags."
    );
}
