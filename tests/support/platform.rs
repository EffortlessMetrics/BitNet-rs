//! Platform-specific mock utilities for testing
//!
//! This module provides utilities for creating mock C++ backend libraries
//! for testing backend availability detection across different platforms.
//!
//! # Coverage
//!
//! Tests spec: docs/specs/test-infra-auto-repair-ci.md (AC3-AC5)
//!
//! # Design
//!
//! Provides cross-platform abstractions for:
//! - Library naming conventions (AC4)
//! - Dynamic loader path variables (AC5)
//! - Mock backend library creation (AC3)
//!
//! # Platform Support
//!
//! - **Linux**: `.so` libraries, `LD_LIBRARY_PATH`
//! - **macOS**: `.dylib` libraries, `DYLD_LIBRARY_PATH`
//! - **Windows**: `.dll` libraries, `PATH`

use std::path::Path;

use bitnet_crossval::backend::CppBackend;

/// Formats library name with platform-specific prefix and extension (AC4)
///
/// # Arguments
/// - `base`: Base library name (e.g., "bitnet", "llama", "ggml")
///
/// # Returns
/// - Linux: `lib{base}.so` (e.g., `libbitnet.so`)
/// - macOS: `lib{base}.dylib` (e.g., `libbitnet.dylib`)
/// - Windows: `{base}.dll` (e.g., `bitnet.dll`)
///
/// # Examples
///
/// ```rust
/// use bitnet_tests::support::platform::format_lib_name;
///
/// let name = format_lib_name("bitnet");
///
/// #[cfg(target_os = "linux")]
/// assert_eq!(name, "libbitnet.so");
///
/// #[cfg(target_os = "macos")]
/// assert_eq!(name, "libbitnet.dylib");
///
/// #[cfg(target_os = "windows")]
/// assert_eq!(name, "bitnet.dll");
/// ```
pub fn format_lib_name(base: &str) -> String {
    #[cfg(target_os = "linux")]
    return format!("lib{}.so", base);
    #[cfg(target_os = "macos")]
    return format!("lib{}.dylib", base);
    #[cfg(target_os = "windows")]
    return format!("{}.dll", base);
}

/// Returns platform-specific dynamic loader path variable name (AC5)
///
/// # Returns
/// - `"LD_LIBRARY_PATH"` on Linux
/// - `"DYLD_LIBRARY_PATH"` on macOS
/// - `"PATH"` on Windows
///
/// # Examples
///
/// ```rust
/// use bitnet_tests::support::platform::get_loader_path_var;
///
/// let var = get_loader_path_var();
///
/// #[cfg(target_os = "linux")]
/// assert_eq!(var, "LD_LIBRARY_PATH");
///
/// #[cfg(target_os = "macos")]
/// assert_eq!(var, "DYLD_LIBRARY_PATH");
///
/// #[cfg(target_os = "windows")]
/// assert_eq!(var, "PATH");
/// ```
pub fn get_loader_path_var() -> &'static str {
    #[cfg(target_os = "linux")]
    return "LD_LIBRARY_PATH";
    #[cfg(target_os = "macos")]
    return "DYLD_LIBRARY_PATH";
    #[cfg(target_os = "windows")]
    return "PATH";
}

/// Creates mock backend library files in specified directory (AC3)
///
/// # Platform Behavior
/// - **Linux**: Creates `libbitnet.so`, etc. with 0o755 permissions
/// - **macOS**: Creates `libbitnet.dylib`, etc. with 0o755 permissions
/// - **Windows**: Creates `bitnet.dll`, etc. (no permission handling)
///
/// # Arguments
/// - `dir`: Directory to create library files in
/// - `backend`: C++ backend to mock (BitNet or Llama)
///
/// # Returns
/// - `Ok(())`: Libraries created successfully
/// - `Err(anyhow::Error)`: Creation failed
///
/// # Library Sets by Backend
/// - **BitNet**: `bitnet`, `llama`, `ggml` libraries
/// - **Llama**: `llama`, `ggml` libraries
///
/// # Mock Library Properties
/// - **Size**: 0 bytes (empty files)
/// - **Permissions**: 0o755 on Unix (rwx for owner, r-x for group/other)
/// - **Purpose**: Filesystem discovery only (no actual loading)
///
/// # Examples
///
/// ```rust,no_run
/// use std::path::Path;
/// use bitnet_crossval::backend::CppBackend;
/// use bitnet_tests::support::platform::create_mock_backend_libs;
///
/// // Example test function (not executed in doctest)
/// fn test_with_mock_backend() {
///     let temp_dir = tempfile::tempdir().unwrap();
///     create_mock_backend_libs(temp_dir.path(), CppBackend::BitNet).unwrap();
///
///     // Verify library exists
///     #[cfg(target_os = "linux")]
///     assert!(temp_dir.path().join("libbitnet.so").exists());
/// }
/// ```
pub fn create_mock_backend_libs(dir: &Path, backend: CppBackend) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir)?;

    let libs = match backend {
        CppBackend::BitNet => vec!["bitnet", "llama", "ggml"],
        CppBackend::Llama => vec!["llama", "ggml"],
    };

    for lib in libs {
        let path = dir.join(format_lib_name(lib));
        std::fs::write(&path, b"")?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755))?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_format_lib_name_platform_specific() {
        let name = format_lib_name("bitnet");

        #[cfg(target_os = "linux")]
        assert_eq!(name, "libbitnet.so");

        #[cfg(target_os = "macos")]
        assert_eq!(name, "libbitnet.dylib");

        #[cfg(target_os = "windows")]
        assert_eq!(name, "bitnet.dll");
    }

    #[test]
    fn test_format_lib_name_multiple_stems() {
        for stem in &["bitnet", "llama", "ggml"] {
            let name = format_lib_name(stem);
            assert!(name.contains(stem));

            #[cfg(target_os = "linux")]
            assert!(name.ends_with(".so"));

            #[cfg(target_os = "macos")]
            assert!(name.ends_with(".dylib"));

            #[cfg(target_os = "windows")]
            assert!(name.ends_with(".dll"));
        }
    }

    #[test]
    fn test_get_loader_path_var_platform_specific() {
        let var = get_loader_path_var();

        #[cfg(target_os = "linux")]
        assert_eq!(var, "LD_LIBRARY_PATH");

        #[cfg(target_os = "macos")]
        assert_eq!(var, "DYLD_LIBRARY_PATH");

        #[cfg(target_os = "windows")]
        assert_eq!(var, "PATH");

        assert!(!var.is_empty());
    }

    #[test]
    fn test_create_mock_backend_libs_bitnet() {
        let temp = TempDir::new().unwrap();
        create_mock_backend_libs(temp.path(), CppBackend::BitNet).unwrap();

        // Verify all three libraries exist
        let bitnet_path = temp.path().join(format_lib_name("bitnet"));
        let llama_path = temp.path().join(format_lib_name("llama"));
        let ggml_path = temp.path().join(format_lib_name("ggml"));

        assert!(bitnet_path.exists(), "libbitnet should exist");
        assert!(llama_path.exists(), "libllama should exist");
        assert!(ggml_path.exists(), "libggml should exist");

        // Verify files are empty
        assert_eq!(std::fs::metadata(&bitnet_path).unwrap().len(), 0);
        assert_eq!(std::fs::metadata(&llama_path).unwrap().len(), 0);
        assert_eq!(std::fs::metadata(&ggml_path).unwrap().len(), 0);
    }

    #[test]
    fn test_create_mock_backend_libs_llama() {
        let temp = TempDir::new().unwrap();
        create_mock_backend_libs(temp.path(), CppBackend::Llama).unwrap();

        // Verify both libraries exist
        let llama_path = temp.path().join(format_lib_name("llama"));
        let ggml_path = temp.path().join(format_lib_name("ggml"));

        assert!(llama_path.exists(), "libllama should exist");
        assert!(ggml_path.exists(), "libggml should exist");

        // Verify files are empty
        assert_eq!(std::fs::metadata(&llama_path).unwrap().len(), 0);
        assert_eq!(std::fs::metadata(&ggml_path).unwrap().len(), 0);
    }

    #[test]
    #[cfg(unix)]
    fn test_mock_libs_have_executable_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let temp = TempDir::new().unwrap();
        create_mock_backend_libs(temp.path(), CppBackend::BitNet).unwrap();

        let lib_path = temp.path().join(format_lib_name("bitnet"));
        let metadata = std::fs::metadata(&lib_path).unwrap();
        let mode = metadata.permissions().mode();

        // Verify 0o755 permissions (owner: rwx, group: r-x, other: r-x)
        assert_eq!(mode & 0o777, 0o755);
    }

    #[test]
    fn test_create_mock_backend_libs_creates_dir() {
        let temp = TempDir::new().unwrap();
        let subdir = temp.path().join("nested").join("dir");

        // Should create directory structure
        create_mock_backend_libs(&subdir, CppBackend::Llama).unwrap();

        assert!(subdir.exists());
        assert!(subdir.join(format_lib_name("llama")).exists());
    }

    #[test]
    fn test_platform_specific_library_naming() {
        let temp = TempDir::new().unwrap();
        create_mock_backend_libs(temp.path(), CppBackend::BitNet).unwrap();

        #[cfg(target_os = "linux")]
        {
            assert!(temp.path().join("libbitnet.so").exists());
            assert!(temp.path().join("libllama.so").exists());
            assert!(temp.path().join("libggml.so").exists());
        }

        #[cfg(target_os = "macos")]
        {
            assert!(temp.path().join("libbitnet.dylib").exists());
            assert!(temp.path().join("libllama.dylib").exists());
            assert!(temp.path().join("libggml.dylib").exists());
        }

        #[cfg(target_os = "windows")]
        {
            assert!(temp.path().join("bitnet.dll").exists());
            assert!(temp.path().join("llama.dll").exists());
            assert!(temp.path().join("ggml.dll").exists());
        }
    }
}
