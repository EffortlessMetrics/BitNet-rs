//! Mock C++ backend library fixtures for testing
//!
//! This module provides utilities to create mock shared library files for testing
//! backend availability detection without requiring actual C++ compilation.
//!
//! # Coverage
//!
//! Tests spec: docs/specs/test-infra-auto-repair-ci.md#ac3
//!
//! # Design
//!
//! Mock libraries are **empty files** with correct naming and permissions:
//! - **Purpose**: Backend availability detection only (no actual loading)
//! - **Contents**: Zero bytes (no symbols, no ABI compatibility issues)
//! - **Loading**: Never loaded via dlopen (filesystem checks only)
//! - **Benefits**: Instant creation, cross-platform, no compilation overhead
//!
//! # Platform Behavior
//!
//! - **Linux**: Creates `lib{name}.so` files with 0o755 permissions
//! - **macOS**: Creates `lib{name}.dylib` files with 0o755 permissions
//! - **Windows**: Creates `{name}.dll` files (no permission handling)
//!
//! # Backend Library Sets
//!
//! - **BitNet**: Single library (`libbitnet.*`)
//! - **Llama**: Two libraries (`libllama.*`, `libggml.*`)
//!
//! # Examples
//!
//! ```rust,no_run
//! use bitnet_crossval::backend::CppBackend;
//! use tests::support::mock_fixtures::create_mock_backend_libs;
//!
//! // Example test function (not executed in doctest)
//! fn test_backend_discovery() {
//!     // Create mock libraries in temp directory
//!     let temp = create_mock_backend_libs(CppBackend::BitNet).unwrap();
//!
//!     // Verify library exists for discovery
//!     let lib_path = temp.path().join("libbitnet.so");  // Linux
//!     assert!(lib_path.exists());
//!
//!     // Verify file is empty (no actual symbols)
//!     let metadata = std::fs::metadata(&lib_path).unwrap();
//!     assert_eq!(metadata.len(), 0);
//!
//!     // Auto-cleanup on drop
//! }
//! ```

use bitnet_crossval::backend::CppBackend;
use std::fs::File;
use tempfile::TempDir;

use super::platform_utils::format_lib_name;

/// Creates temporary directory with platform-specific mock libraries
///
/// # Platform Behavior
/// - **Linux**: Creates `libbitnet.so`, `libllama.so`, `libggml.so` with 0o755 permissions
/// - **macOS**: Creates `libbitnet.dylib`, `libllama.dylib`, `libggml.dylib` with 0o755 permissions
/// - **Windows**: Creates `bitnet.dll`, `llama.dll`, `ggml.dll` (no permission handling)
///
/// # Arguments
/// - `backend`: C++ backend to mock (BitNet or Llama)
///
/// # Returns
/// - `Ok(TempDir)`: Temporary directory with mock libraries (auto-cleans on drop)
/// - `Err(String)`: Error message if creation fails
///
/// # Library Sets by Backend
/// - **BitNet**: Single library file (`libbitnet.*`)
/// - **Llama**: Two library files (`libllama.*`, `libggml.*`)
///
/// # Mock Library Properties
/// - **Size**: 0 bytes (empty files)
/// - **Permissions**: 0o755 on Unix (owner: rwx, group: r-x, other: r-x)
/// - **Symbols**: None (not loadable via dlopen)
/// - **Purpose**: Filesystem discovery only (no actual dynamic loading)
///
/// # Examples
///
/// ```rust,no_run
/// use bitnet_crossval::backend::CppBackend;
/// use tests::support::mock_fixtures::create_mock_backend_libs;
/// use tests::support::platform_utils::get_loader_path_var;
///
/// // Example test function (not executed in doctest)
/// fn test_with_mock_backend() {
///     // Create mock libraries
///     let temp = create_mock_backend_libs(CppBackend::BitNet).unwrap();
///
///     // Set loader path for discovery
///     let loader_var = get_loader_path_var();
///     std::env::set_var(loader_var, temp.path());
///
///     // Backend discovery now succeeds
///     // Tests can proceed without real C++ libraries
///
///     // Automatic cleanup when temp goes out of scope
/// }
/// ```
pub fn create_mock_backend_libs(backend: CppBackend) -> Result<TempDir, String> {
    let temp = TempDir::new().map_err(|e| format!("Failed to create temp dir: {}", e))?;

    let lib_names = match backend {
        CppBackend::BitNet => vec!["bitnet"],
        CppBackend::Llama => vec!["llama", "ggml"],
    };

    for name in lib_names {
        let lib_path = temp.path().join(format_lib_name(name));

        // Create empty file (contents don't matter for discovery)
        File::create(&lib_path)
            .map_err(|e| format!("Failed to create mock library {}: {}", lib_path.display(), e))?;

        // Set executable permissions on Unix platforms
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&lib_path)
                .map_err(|e| format!("Failed to get metadata: {}", e))?
                .permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&lib_path, perms)
                .map_err(|e| format!("Failed to set permissions: {}", e))?;
        }
    }

    Ok(temp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::platform_utils::format_lib_name;

    #[test]
    fn test_create_mock_bitnet_libs() {
        let temp = create_mock_backend_libs(CppBackend::BitNet).unwrap();

        // Verify library file exists
        let lib_path = temp.path().join(format_lib_name("bitnet"));
        assert!(lib_path.exists(), "Mock library should exist");

        // Verify file is empty
        let metadata = std::fs::metadata(&lib_path).unwrap();
        assert_eq!(metadata.len(), 0, "Mock library should be empty");
    }

    #[test]
    fn test_create_mock_llama_libs() {
        let temp = create_mock_backend_libs(CppBackend::Llama).unwrap();

        // Verify both libraries exist
        let llama_path = temp.path().join(format_lib_name("llama"));
        let ggml_path = temp.path().join(format_lib_name("ggml"));

        assert!(llama_path.exists(), "libllama should exist");
        assert!(ggml_path.exists(), "libggml should exist");

        // Verify both are empty
        assert_eq!(std::fs::metadata(&llama_path).unwrap().len(), 0, "libllama should be empty");
        assert_eq!(std::fs::metadata(&ggml_path).unwrap().len(), 0, "libggml should be empty");
    }

    #[test]
    #[cfg(unix)]
    fn test_mock_libs_have_executable_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let temp = create_mock_backend_libs(CppBackend::BitNet).unwrap();
        let lib_path = temp.path().join(format_lib_name("bitnet"));

        let metadata = std::fs::metadata(&lib_path).unwrap();
        let mode = metadata.permissions().mode();

        // Verify 0o755 permissions (owner: rwx, group: r-x, other: r-x)
        assert_eq!(mode & 0o777, 0o755, "Mock library should have 0o755 permissions");
    }

    #[test]
    fn test_mock_libs_platform_specific_naming() {
        let temp = create_mock_backend_libs(CppBackend::BitNet).unwrap();

        // Verify platform-specific naming
        #[cfg(target_os = "linux")]
        {
            let lib_path = temp.path().join("libbitnet.so");
            assert!(lib_path.exists(), "Linux should create libbitnet.so");
        }

        #[cfg(target_os = "macos")]
        {
            let lib_path = temp.path().join("libbitnet.dylib");
            assert!(lib_path.exists(), "macOS should create libbitnet.dylib");
        }

        #[cfg(target_os = "windows")]
        {
            let lib_path = temp.path().join("bitnet.dll");
            assert!(lib_path.exists(), "Windows should create bitnet.dll");
        }
    }

    #[test]
    fn test_temp_dir_cleanup() {
        let temp_path: std::path::PathBuf;

        {
            let temp = create_mock_backend_libs(CppBackend::BitNet).unwrap();
            temp_path = temp.path().to_path_buf();

            // Verify exists within scope
            assert!(temp_path.exists(), "Temp directory should exist while in scope");
        }
        // TempDir dropped, directory should be cleaned up

        // Note: TempDir cleanup is not guaranteed to complete immediately on Windows
        // This test may be flaky on Windows due to file handle timing
        #[cfg(unix)]
        {
            assert!(!temp_path.exists(), "Temp directory should be cleaned up after drop");
        }
    }

    #[test]
    fn test_bitnet_single_library() {
        let temp = create_mock_backend_libs(CppBackend::BitNet).unwrap();

        // Count library files
        let lib_count = std::fs::read_dir(temp.path())
            .unwrap()
            .filter(|entry| if let Ok(entry) = entry { entry.path().is_file() } else { false })
            .count();

        assert_eq!(lib_count, 1, "BitNet should create exactly 1 library file");
    }

    #[test]
    fn test_llama_dual_libraries() {
        let temp = create_mock_backend_libs(CppBackend::Llama).unwrap();

        // Count library files
        let lib_count = std::fs::read_dir(temp.path())
            .unwrap()
            .filter(|entry| if let Ok(entry) = entry { entry.path().is_file() } else { false })
            .count();

        assert_eq!(lib_count, 2, "Llama should create exactly 2 library files");
    }
}
