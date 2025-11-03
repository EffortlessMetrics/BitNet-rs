//! Quick integration test for AC3-AC4 implementation
//!
//! This test verifies that the platform-specific mock utilities and
//! platform utility functions work correctly.

mod support {
    pub mod backend_helpers;
    pub mod env_guard;
    pub mod mock_fixtures;
    pub mod platform_utils;
}

use support::mock_fixtures::create_mock_backend_libs;
use support::platform_utils::{format_lib_name, get_lib_extension, get_loader_path_var};
use bitnet_crossval::backend::CppBackend;

#[test]
fn test_platform_utils_basic() {
    // Test get_loader_path_var
    let loader_var = get_loader_path_var();

    #[cfg(target_os = "linux")]
    assert_eq!(loader_var, "LD_LIBRARY_PATH");

    #[cfg(target_os = "macos")]
    assert_eq!(loader_var, "DYLD_LIBRARY_PATH");

    #[cfg(target_os = "windows")]
    assert_eq!(loader_var, "PATH");

    // Test get_lib_extension
    let ext = get_lib_extension();

    #[cfg(target_os = "linux")]
    assert_eq!(ext, "so");

    #[cfg(target_os = "macos")]
    assert_eq!(ext, "dylib");

    #[cfg(target_os = "windows")]
    assert_eq!(ext, "dll");

    // Test format_lib_name
    let lib_name = format_lib_name("bitnet");

    #[cfg(target_os = "linux")]
    assert_eq!(lib_name, "libbitnet.so");

    #[cfg(target_os = "macos")]
    assert_eq!(lib_name, "libbitnet.dylib");

    #[cfg(target_os = "windows")]
    assert_eq!(lib_name, "bitnet.dll");
}

#[test]
fn test_create_mock_bitnet_libs() {
    let temp = create_mock_backend_libs(CppBackend::BitNet).unwrap();

    // Verify library exists
    let lib_path = temp.path().join(format_lib_name("bitnet"));
    assert!(lib_path.exists(), "Mock library should exist");

    // Verify file is empty
    let metadata = std::fs::metadata(&lib_path).unwrap();
    assert_eq!(metadata.len(), 0, "Mock library should be empty");

    // Verify permissions on Unix
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = metadata.permissions().mode();
        assert_eq!(mode & 0o777, 0o755, "Should have 0o755 permissions");
    }
}

#[test]
fn test_create_mock_llama_libs() {
    let temp = create_mock_backend_libs(CppBackend::Llama).unwrap();

    // Verify both libraries exist
    let llama_path = temp.path().join(format_lib_name("llama"));
    let ggml_path = temp.path().join(format_lib_name("ggml"));

    assert!(llama_path.exists(), "llama library should exist");
    assert!(ggml_path.exists(), "ggml library should exist");

    // Verify both are empty
    assert_eq!(std::fs::metadata(&llama_path).unwrap().len(), 0);
    assert_eq!(std::fs::metadata(&ggml_path).unwrap().len(), 0);

    // Verify permissions on Unix
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let llama_mode = std::fs::metadata(&llama_path).unwrap().permissions().mode();
        let ggml_mode = std::fs::metadata(&ggml_path).unwrap().permissions().mode();
        assert_eq!(llama_mode & 0o777, 0o755);
        assert_eq!(ggml_mode & 0o777, 0o755);
    }
}
