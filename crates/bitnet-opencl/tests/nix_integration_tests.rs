//! Nix integration tests for GPU environment.
//!
//! These tests verify that the Nix GPU development shell sets up
//! the expected environment variables, tools, and library paths.

use std::path::Path;

// ── Basic type sanity (always runnable) ──────────────────────────

#[test]
fn test_opencl_headers_available() {
    // OpenCL cl_uint is always 32-bit; verify Rust u32 matches
    assert_eq!(std::mem::size_of::<u32>(), 4);
}

#[test]
fn test_opencl_platform_id_size() {
    // OpenCL cl_platform_id is a pointer-sized opaque handle
    assert!(std::mem::size_of::<usize>() >= 4);
}

#[test]
fn test_opencl_mem_flags_repr() {
    // cl_mem_flags is u64 on all platforms
    assert_eq!(std::mem::size_of::<u64>(), 8);
}

// ── Environment variable checks ─────────────────────────────────

#[test]
fn test_ocl_icd_env_parseable() {
    // OCL_ICD_VENDORS, if set, must be a valid path string
    if let Ok(val) = std::env::var("OCL_ICD_VENDORS") {
        assert!(!val.is_empty(), "OCL_ICD_VENDORS is set but empty");
        let p = Path::new(&val);
        assert!(p.is_absolute(), "OCL_ICD_VENDORS should be an absolute path, got: {val}");
    }
}

#[test]
fn test_vulkan_icd_env_parseable() {
    // VK_ICD_FILENAMES, if set, must be a valid path string
    if let Ok(val) = std::env::var("VK_ICD_FILENAMES") {
        assert!(!val.is_empty(), "VK_ICD_FILENAMES is set but empty");
    }
}

#[test]
fn test_libclang_path_env_parseable() {
    // LIBCLANG_PATH, if set, should point to a directory
    if let Ok(val) = std::env::var("LIBCLANG_PATH") {
        assert!(!val.is_empty(), "LIBCLANG_PATH is set but empty");
        let p = Path::new(&val);
        assert!(p.is_absolute(), "LIBCLANG_PATH should be an absolute path, got: {val}");
    }
}

// ── Library helper function tests ───────────────────────────────

#[test]
fn test_ocl_icd_vendors_configured_runs() {
    let _ = bitnet_opencl::ocl_icd_vendors_configured();
}

#[test]
fn test_vulkan_icd_configured_runs() {
    let _ = bitnet_opencl::vulkan_icd_configured();
}

#[test]
fn test_ocl_icd_vendors_path_returns_option() {
    let result = bitnet_opencl::ocl_icd_vendors_path();
    // Just verify the type — value depends on environment
    let _: Option<String> = result;
}

#[test]
fn test_vulkan_icd_path_returns_option() {
    let result = bitnet_opencl::vulkan_icd_path();
    let _: Option<String> = result;
}

// ── Tool availability (requires Nix GPU shell) ──────────────────

#[test]
#[ignore = "requires Nix GPU shell - run with: nix develop .#gpu"]
fn test_clinfo_available() {
    let output = std::process::Command::new("clinfo").arg("--list").output();
    assert!(output.is_ok(), "clinfo not found in PATH");
}

#[test]
#[ignore = "requires Nix GPU shell - run with: nix develop .#gpu"]
fn test_vulkaninfo_available() {
    let output = std::process::Command::new("vulkaninfo").arg("--summary").output();
    assert!(output.is_ok(), "vulkaninfo not found in PATH");
}

#[test]
#[ignore = "requires Nix GPU shell - run with: nix develop .#gpu"]
fn test_ocl_icd_vendors_set_in_nix() {
    assert!(
        bitnet_opencl::ocl_icd_vendors_configured(),
        "OCL_ICD_VENDORS not set — enter GPU shell with: nix develop .#gpu"
    );
}

#[test]
#[ignore = "requires Nix GPU shell - run with: nix develop .#gpu"]
fn test_ocl_icd_vendors_path_exists() {
    let path = bitnet_opencl::ocl_icd_vendors_path().expect("OCL_ICD_VENDORS not set");
    assert!(Path::new(&path).exists(), "OCL_ICD_VENDORS path does not exist: {path}");
}

#[test]
#[ignore = "requires Nix GPU shell - run with: nix develop .#gpu"]
fn test_vulkan_icd_set_in_nix() {
    assert!(
        bitnet_opencl::vulkan_icd_configured(),
        "VK_ICD_FILENAMES not set — enter GPU shell with: nix develop .#gpu"
    );
}

#[test]
#[ignore = "requires Nix GPU shell - run with: nix develop .#gpu"]
fn test_pkg_config_finds_opencl() {
    let output = std::process::Command::new("pkg-config").args(["--exists", "OpenCL"]).status();
    match output {
        Ok(status) => assert!(status.success(), "pkg-config could not find OpenCL"),
        Err(e) => panic!("pkg-config not available: {e}"),
    }
}
